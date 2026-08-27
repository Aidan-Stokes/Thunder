package dataframe

// Parallel partitioned hash grouping (Stage 21, S21.1): the per-row
// encode_row + map-insert loop of dataframe_group_by split across threads via
// src/parallel do_parallel.
//
// Each thread encodes and groups a contiguous row range into its own local
// map (key bytes -> group row indices) plus its own first-seen key order, so
// there is no shared mutable state during the parallel phase. The main thread
// then merges the partitions in chunk order (chunk 0's first-seen order, then
// chunk 1, ...). Because chunks cover ascending row ranges, merging in that
// order reproduces exactly the sequential semantics:
//   - groups appear in first-appearance order,
//   - each group preserves source row order.
//
// Ownership: every allocation (map internals, cloned key strings, row-index
// arrays, encode buffers) comes from the caller's allocator passed through
// Group_By_Context.alloc — never context.allocator inside tasks. On success
// the local entries are adopted by the global Group_By (the local shells are
// dropped without deleting their contents); on error everything is torn down.
//
// The allocator must be safe for concurrent use while the parallel phase runs
// (same constraint as csv_parallel / parallel_gather).

import "core:mem"
import "core:thread"
import "../../libs/parallel"

// PARALLEL_GROUPBY_THRESHOLD is the minimum row count to use the parallel
// grouping path; below it the sequential loop in dataframe_group_by runs.
PARALLEL_GROUPBY_THRESHOLD :: 100_000

// PARALLEL_GROUPBY_THREADS is the thread pool size for parallel grouping.
PARALLEL_GROUPBY_THREADS :: 8

// Group_By_Context carries the shared state for parallel grouping tasks.
// maps/orders/bufs are indexed by task id; errs collects per-task failures.
Group_By_Context :: struct {
	key_ptrs: []^Column,
	alloc:    mem.Allocator,
	maps:     []map[string][dynamic]int,
	orders:   [][dynamic]string,
	bufs:     [][dynamic]byte,
	errs:     []Error,
}

// groupby_chunk_task is the per-thread task proc: encode + hash one
// contiguous row range into the thread's local map.
@(private)
groupby_chunk_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(Group_By_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	start := info.a_start_off[tid]
	end := info.a_end_off[tid]

	local := make(map[string][dynamic]int, 0, ctx.alloc)
	order := make([dynamic]string, 0, ctx.alloc)
	buf := make([dynamic]byte, 0, ctx.alloc)

	for row in start ..< end {
		if enc_err := encode_row(ctx.key_ptrs, row, &buf); enc_err != .None {
			ctx.errs[tid] = enc_err
			return
		}
		key := string(buf[:])
		if _, exists := local[key]; !exists {
			owned, o_err := clone_name(ctx.alloc, key)
			if o_err != .None {
				ctx.errs[tid] = o_err
				return
			}
			local[owned] = make([dynamic]int, 0, ctx.alloc)
			append(&order, owned)
		}
		append(&local[key], row)
	}

	ctx.maps[tid] = local
	ctx.orders[tid] = order
	ctx.bufs[tid] = buf
}

// groupby_locals_destroy fully releases the per-thread locals (keys, rows,
// orders, buffers). Used on error paths where nothing was adopted.
@(private)
groupby_locals_destroy :: proc(ctx: ^Group_By_Context, nthreads: int) {
	for t in 0 ..< nthreads {
		for k in ctx.orders[t] {
			delete_string(k, ctx.alloc)
		}
		for _, v in ctx.maps[t] {
			delete(v)
		}
		delete(ctx.maps[t])
		delete(ctx.orders[t])
		delete(ctx.bufs[t])
	}
	delete(ctx.maps, ctx.alloc)
	delete(ctx.orders, ctx.alloc)
	delete(ctx.bufs, ctx.alloc)
	delete(ctx.errs, ctx.alloc)
}

// group_rows_sequential is the single-threaded grouping loop (unchanged
// semantics; used below PARALLEL_GROUPBY_THRESHOLD).
@(private)
group_rows_sequential :: proc(gb: ^Group_By, rows_total: int) -> Error {
	buf := make([dynamic]byte, gb.alloc)
	defer delete(buf)
	for row in 0 ..< rows_total {
		if enc_err := encode_row(gb.key_ptrs, row, &buf); enc_err != .None {
			return enc_err
		}
		key := string(buf[:])
		if _, exists := gb.groups[key]; !exists {
			owned, o_err := clone_name(gb.alloc, key)
			if o_err != .None {
				return o_err
			}
			gb.groups[owned] = make([dynamic]int, 0, gb.alloc)
			append(&gb.order, owned)
		}
		append(&gb.groups[key], row)
	}
	return .None
}

// group_rows_parallel splits the row range across PARALLEL_GROUPBY_THREADS
// threads, then merges the partitions into gb with identical
// first-appearance/source-order semantics as the sequential loop.
@(private)
group_rows_parallel :: proc(gb: ^Group_By, rows_total: int) -> Error {
	nthreads := min(PARALLEL_GROUPBY_THREADS, rows_total)

	ctx := Group_By_Context{
		key_ptrs = gb.key_ptrs,
		alloc    = gb.alloc,
		maps     = make([]map[string][dynamic]int, nthreads, gb.alloc),
		orders   = make([][dynamic]string, nthreads, gb.alloc),
		bufs     = make([][dynamic]byte, nthreads, gb.alloc),
		errs     = make([]Error, nthreads, gb.alloc),
	}
	defer groupby_locals_destroy(&ctx, nthreads)

	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)

	parallel.do_parallel(&pool, groupby_chunk_task, &ctx, rows_total, nthreads)

	for t in 0 ..< nthreads {
		if ctx.errs[t] != .None {
			return ctx.errs[t]
		}
	}

	// Merge partitions in ascending row-range order. Entries are adopted
	// (ownership moves to the global map); merged row arrays and merged key
	// clones are freed here.
	for t in 0 ..< nthreads {
		for key in ctx.orders[t] {
			rows_local := ctx.maps[t][key]
			if _, exists := gb.groups[key]; !exists {
				// New global group: adopt the key string and its row array.
				gb.groups[key] = rows_local
				append(&gb.order, key)
				continue
			}
			append(&gb.groups[key], ..rows_local[:])
			delete(rows_local)
			// The global map already owns an equal key string; this chunk's
			// clone is redundant.
			delete_string(key, gb.alloc)
		}
		// Free the local map internals (its entries were adopted or freed
		// above), then drop the shells so teardown skips them.
		delete(ctx.maps[t])
		ctx.maps[t] = nil
		clear(&ctx.orders[t])
	}
	return .None
}
