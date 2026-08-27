package dataframe

// Parallel SIMD wrappers (S15.9): split large arrays across N threads,
// each thread runs the existing SIMD kernel on its slice.  Only the
// all-valid fast path (no NULL checks) is parallelized -- the validity-aware
// path stays sequential because it has per-element branches.

import "core:thread"
import "../../libs/parallel"

PARALLEL_SIMD_THRESHOLD :: 65_536
PARALLEL_SIMD_THREADS   :: 8

// --- f64 contexts + tasks ---------------------------------------------------

SIMD_F64_Binary_Context :: struct {
	lv, rv, ov: []f64,
}

simd_f64_add_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(SIMD_F64_Binary_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	simd_add(ctx.lv[s:e], ctx.rv[s:e], ctx.ov[s:e])
}

simd_f64_sub_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(SIMD_F64_Binary_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	simd_sub(ctx.lv[s:e], ctx.rv[s:e], ctx.ov[s:e])
}

simd_f64_mul_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(SIMD_F64_Binary_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	simd_mul(ctx.lv[s:e], ctx.rv[s:e], ctx.ov[s:e])
}

SIMD_F64_Unary_Context :: struct {
	iv, ov: []f64,
}

simd_f64_neg_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(SIMD_F64_Unary_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	simd_neg(ctx.iv[s:e], ctx.ov[s:e])
}

simd_f64_abs_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(SIMD_F64_Unary_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	simd_abs(ctx.iv[s:e], ctx.ov[s:e])
}

SIMD_F64_Sum_Context :: struct {
	iv:    []f64,
	parts: []f64,
}

simd_f64_sum_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(SIMD_F64_Sum_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	ctx.parts[tid] = simd_sum(ctx.iv[s:e])
}

// --- i64 contexts + tasks ---------------------------------------------------

SIMD_I64_Binary_Context :: struct {
	lv, rv, ov: []i64,
}

simd_i64_add_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(SIMD_I64_Binary_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	simd_add(ctx.lv[s:e], ctx.rv[s:e], ctx.ov[s:e])
}

simd_i64_sub_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(SIMD_I64_Binary_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	simd_sub(ctx.lv[s:e], ctx.rv[s:e], ctx.ov[s:e])
}

simd_i64_mul_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(SIMD_I64_Binary_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	simd_mul(ctx.lv[s:e], ctx.rv[s:e], ctx.ov[s:e])
}

SIMD_I64_Unary_Context :: struct {
	iv, ov: []i64,
}

simd_i64_neg_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(SIMD_I64_Unary_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s, e := info.a_start_off[tid], info.a_end_off[tid]
	simd_neg(ctx.iv[s:e], ctx.ov[s:e])
}

// --- public parallel dispatch -----------------------------------------------

parallel_simd_add_f64 :: proc(lv, rv, ov: []f64) {
	n := len(lv)
	if n < PARALLEL_SIMD_THRESHOLD {
		simd_add(lv, rv, ov)
		return
	}
	nthreads := min(PARALLEL_SIMD_THREADS, n)
	ctx := SIMD_F64_Binary_Context{lv, rv, ov}
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, simd_f64_add_task, &ctx, n, nthreads)
}

parallel_simd_sub_f64 :: proc(lv, rv, ov: []f64) {
	n := len(lv)
	if n < PARALLEL_SIMD_THRESHOLD {
		simd_sub(lv, rv, ov)
		return
	}
	nthreads := min(PARALLEL_SIMD_THREADS, n)
	ctx := SIMD_F64_Binary_Context{lv, rv, ov}
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, simd_f64_sub_task, &ctx, n, nthreads)
}

parallel_simd_mul_f64 :: proc(lv, rv, ov: []f64) {
	n := len(lv)
	if n < PARALLEL_SIMD_THRESHOLD {
		simd_mul(lv, rv, ov)
		return
	}
	nthreads := min(PARALLEL_SIMD_THREADS, n)
	ctx := SIMD_F64_Binary_Context{lv, rv, ov}
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, simd_f64_mul_task, &ctx, n, nthreads)
}

parallel_simd_neg_f64 :: proc(iv, ov: []f64) {
	n := len(iv)
	if n < PARALLEL_SIMD_THRESHOLD {
		simd_neg(iv, ov)
		return
	}
	nthreads := min(PARALLEL_SIMD_THREADS, n)
	ctx := SIMD_F64_Unary_Context{iv, ov}
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, simd_f64_neg_task, &ctx, n, nthreads)
}

parallel_simd_abs_f64 :: proc(iv, ov: []f64) {
	n := len(iv)
	if n < PARALLEL_SIMD_THRESHOLD {
		simd_abs(iv, ov)
		return
	}
	nthreads := min(PARALLEL_SIMD_THREADS, n)
	ctx := SIMD_F64_Unary_Context{iv, ov}
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, simd_f64_abs_task, &ctx, n, nthreads)
}

parallel_simd_sum_f64 :: proc(iv: []f64) -> f64 {
	n := len(iv)
	if n < PARALLEL_SIMD_THRESHOLD {
		return simd_sum(iv)
	}
	nthreads := min(PARALLEL_SIMD_THREADS, n)
	parts := make([]f64, nthreads, context.allocator)
	if parts == nil {
		return simd_sum(iv)
	}
	defer delete(parts, context.allocator)
	ctx := SIMD_F64_Sum_Context{iv, parts}
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, simd_f64_sum_task, &ctx, n, nthreads)
	sum: f64
	for i in 0 ..< nthreads {
		sum += parts[i]
	}
	return sum
}

parallel_simd_add_i64 :: proc(lv, rv, ov: []i64) {
	n := len(lv)
	if n < PARALLEL_SIMD_THRESHOLD {
		simd_add(lv, rv, ov)
		return
	}
	nthreads := min(PARALLEL_SIMD_THREADS, n)
	ctx := SIMD_I64_Binary_Context{lv, rv, ov}
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, simd_i64_add_task, &ctx, n, nthreads)
}

parallel_simd_sub_i64 :: proc(lv, rv, ov: []i64) {
	n := len(lv)
	if n < PARALLEL_SIMD_THRESHOLD {
		simd_sub(lv, rv, ov)
		return
	}
	nthreads := min(PARALLEL_SIMD_THREADS, n)
	ctx := SIMD_I64_Binary_Context{lv, rv, ov}
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, simd_i64_sub_task, &ctx, n, nthreads)
}

parallel_simd_mul_i64 :: proc(lv, rv, ov: []i64) {
	n := len(lv)
	if n < PARALLEL_SIMD_THRESHOLD {
		simd_mul(lv, rv, ov)
		return
	}
	nthreads := min(PARALLEL_SIMD_THREADS, n)
	ctx := SIMD_I64_Binary_Context{lv, rv, ov}
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, simd_i64_mul_task, &ctx, n, nthreads)
}

parallel_simd_neg_i64 :: proc(iv, ov: []i64) {
	n := len(iv)
	if n < PARALLEL_SIMD_THRESHOLD {
		simd_neg(iv, ov)
		return
	}
	nthreads := min(PARALLEL_SIMD_THREADS, n)
	ctx := SIMD_I64_Unary_Context{iv, ov}
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)
	parallel.do_parallel(&pool, simd_i64_neg_task, &ctx, n, nthreads)
}
