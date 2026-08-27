package dataframe

// Parallel row gathering (S15.8): parallel_gather_rows_core dispatches the
// hot element-copy loop in gather_rows_core across N threads via src/parallel.
// Filter, select, sort, take, and join materialization all funnel through
// gather_rows_core, so this single parallelization benefits all of them.
//
// Only the element-copy loop is parallelized — the validity-bitmap set and
// string-payload repointing are sequential (they are O(ncols) or O(1) relative
// to the element copy).

import "core:mem"
import "core:thread"
import "../../libs/parallel"

// PARALLEL_GATHER_THRESHOLD is the minimum index count to use parallel gather.
PARALLEL_GATHER_THRESHOLD :: 10_000

// PARALLEL_GATHER_THREADS is the thread pool size for parallel gather.
PARALLEL_GATHER_THREADS :: 8

// Gather_Context is the shared state for parallel gather tasks.
Gather_Context :: struct {
	src_data:    rawptr,
	out_data:    rawptr,
	indices:     []int,
	elem_size:   int,
	null_sentinel: bool,
	has_null:    bool,
}

// gather_chunk_task is the per-thread task proc. Each thread copies elements
// for its slice of indices.
@(private)
gather_chunk_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(Gather_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	start := info.a_start_off[tid]
	end := info.a_end_off[tid]
	es := ctx.elem_size

	for i in start ..< end {
		idx := ctx.indices[i]
		if ctx.null_sentinel && idx < 0 {
			continue
		}
		src_ptr := ptr_offset(ctx.src_data, idx * es)
		dst_ptr := ptr_offset(ctx.out_data, i * es)
		mem.copy(dst_ptr, src_ptr, es)
	}
}

// parallel_gather_rows_core runs the element-copy loop of gather_rows_core
// across PARALLEL_GATHER_THREADS threads. The caller must have already
// allocated out and copied categories/inner_valid. Returns true on success;
// on error the caller falls back to the sequential path.
@(private)
parallel_gather_rows_core :: proc(
	src: ^Column, out: ^Column, indices: []int,
	null_sentinel: bool, has_null: bool,
) -> bool {
	n := len(indices)
	nthreads := min(PARALLEL_GATHER_THREADS, n)

	ctx := Gather_Context{
		src_data    = src.data,
		out_data    = out.data,
		indices     = indices,
		elem_size   = src.elem_size,
		null_sentinel = null_sentinel,
		has_null    = has_null,
	}

	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)

	parallel.do_parallel(&pool, gather_chunk_task, &ctx, n, nthreads)
	return true
}
