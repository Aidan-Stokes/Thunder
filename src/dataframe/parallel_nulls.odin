package dataframe

// Parallel null-handling (S15.10): parallel fill_null and coalesce for
// non-string types. String paths are sequential (fill_string_nulls uses
// a payload blob with sequential write_at; coalesce string path needs
// prefix-sum for blob offsets).

import "base:intrinsics"
import "core:mem"
import "core:thread"
import "../../libs/parallel"

PARALLEL_NULL_THRESHOLD :: 65_536
PARALLEL_NULL_THREADS   :: 8

// --- fill_null (non-string) -------------------------------------------------

Fill_Null_Context :: struct {
	out_data: rawptr,
	valid:    []u64,
	fill:     [16]u8,
	elem_size: int,
}

@(private)
fill_null_chunk_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(Fill_Null_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	for i in s ..< e {
		if !bm_get(ctx.valid, i) {
			mem.copy(ptr_offset(ctx.out_data, i * ctx.elem_size), &ctx.fill, ctx.elem_size)
		}
	}
}

// parallel_fill_null replaces NULL rows with a constant value across
// count rows. Only called for non-string dtypes from fill_null_eval.
parallel_fill_null :: proc(out: ^Column, fill_value: rawptr, count: int) {
	if count < PARALLEL_NULL_THRESHOLD {
		size := out.elem_size
		for i in 0 ..< count {
			if !bm_get(out.valid, i) {
				mem.copy(ptr_offset(out.data, i * size), fill_value, size)
			}
		}
		return
	}
	nthreads := min(PARALLEL_NULL_THREADS, count)
	ctx: Fill_Null_Context
	ctx.out_data = out.data
	ctx.valid = out.valid
	mem.copy(&ctx.fill, fill_value, out.elem_size)
	ctx.elem_size = out.elem_size
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, fill_null_chunk_task, &ctx, count, nthreads)
}

// --- coalesce (non-string) --------------------------------------------------

Coalesce_Context :: struct {
	out_data:  rawptr,
	out_valid: []u64,
	col_data:  []rawptr,
	col_valid: [][]u64,
	ncols:     int,
	elem_size: int,
}

@(private)
coalesce_chunk_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(Coalesce_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	for i in s ..< e {
		found := false
		for j in 0 ..< ctx.ncols {
			if ctx.col_valid[j] != nil && !bm_get(ctx.col_valid[j], i) {
				continue
			}
			mem.copy(
				ptr_offset(ctx.out_data, i * ctx.elem_size),
				ptr_offset(ctx.col_data[j], i * ctx.elem_size),
				ctx.elem_size,
			)
			found = true
			break
		}
			if !found && ctx.out_valid != nil {
				bm_set(ctx.out_valid, i, false)
			}
	}
}

// parallel_coalesce_non_string fills out with the first valid value per row
// across ncols columns. Only called for non-string dtypes from coalesce_eval.
parallel_coalesce_non_string :: proc(
	out: ^Column, cols: []Column, count: int,
) {
	ncols := len(cols)
	if count < PARALLEL_NULL_THRESHOLD {
		size := out.elem_size
		for i in 0 ..< count {
			found := false
			for j in 0 ..< ncols {
				if cols[j].valid != nil && !bm_get(cols[j].valid, i) {
					continue
				}
				mem.copy(ptr_offset(out.data, i * size), ptr_offset(cols[j].data, i * size), size)
				found = true
				break
			}
			if !found {
				_ = column_set_null(out, i)
			}
		}
		return
	}
	nthreads := min(PARALLEL_NULL_THREADS, count)
	// Pre-allocate validity bitmap so parallel tasks don't race on alloc.
	if out.valid == nil {
		out.valid = bm_make(count, true, out.alloc)
	}
	col_data := make([]rawptr, ncols, context.allocator)
	col_valid := make([][]u64, ncols, context.allocator)
	defer {
		delete(col_data, context.allocator)
		delete(col_valid, context.allocator)
	}
	for j in 0 ..< ncols {
		col_data[j] = cols[j].data
		col_valid[j] = cols[j].valid
	}
	ctx := Coalesce_Context{
		out_data  = out.data,
		out_valid = out.valid,
		col_data  = col_data,
		col_valid = col_valid,
		ncols     = ncols,
		elem_size = out.elem_size,
	}
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, coalesce_chunk_task, &ctx, count, nthreads)
}

// --- bm_from_bools (parallel) -----------------------------------------------

BM_From_Bools_Context :: struct {
	src:      []bool,
	bits:     []u64,
}

@(private)
bm_from_bools_chunk_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(BM_From_Bools_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	for i in s ..< e {
		mask := u64(1) << uint(i & 63)
		if ctx.src[i] {
			intrinsics.atomic_or(&ctx.bits[i >> 6], mask)
		} else {
			intrinsics.atomic_and(&ctx.bits[i >> 6], ~mask)
		}
	}
}

// parallel_bm_from_bools converts a []bool validity array into a packed
// []u64 bitmap. Falls back to sequential for small arrays.
parallel_bm_from_bools :: proc(src: []bool, allocator: mem.Allocator) -> []u64 {
	if src == nil {
		return nil
	}
	n := len(src)
	bits := bm_make(n, true, allocator)
	if bits == nil && n != 0 {
		return nil
	}
	if n < PARALLEL_NULL_THRESHOLD {
		for i in 0 ..< n {
			bm_set(bits, i, src[i])
		}
		return bits
	}
	nthreads := min(PARALLEL_NULL_THREADS, n)
	ctx := BM_From_Bools_Context{src, bits}
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, bm_from_bools_chunk_task, &ctx, n, nthreads)
	return bits
}

// --- bm_count_false (parallel) ----------------------------------------------

BM_Count_False_Context :: struct {
	valid: []u64,
	parts: []int,
}

@(private)
bm_count_false_chunk_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(BM_Count_False_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	cnt := 0
	for i in s ..< e {
		if !bm_get(ctx.valid, i) {
			cnt += 1
		}
	}
	ctx.parts[tid] = cnt
}

// parallel_bm_count_false counts the number of zero bits in valid[0..n-1].
// Falls back to sequential for small arrays.
parallel_bm_count_false :: proc(valid: []u64, n: int) -> int {
	if n < PARALLEL_NULL_THRESHOLD {
		cnt := 0
		for i in 0 ..< n {
			if !bm_get(valid, i) {
				cnt += 1
			}
		}
		return cnt
	}
	nthreads := min(PARALLEL_NULL_THREADS, n)
	parts := make([]int, nthreads, context.allocator)
	if parts == nil {
		cnt := 0
		for i in 0 ..< n {
			if !bm_get(valid, i) {
				cnt += 1
			}
		}
		return cnt
	}
	defer delete(parts, context.allocator)
	ctx := BM_Count_False_Context{valid, parts}
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, bm_count_false_chunk_task, &ctx, n, nthreads)
	total := 0
	for i in 0 ..< nthreads {
		total += parts[i]
	}
	return total
}
