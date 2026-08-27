package parallel

// S15.4 tests: do_parallel correctness — sequential fallback,
// even/uneven partition, and full parallel coverage.

import "core:testing"
import "core:thread"

do_par_bool_ctx :: struct {
	arr: ^[]bool,
}

do_par_int_ctx :: struct {
	arr: ^[]int,
}

do_par_one_ctx :: struct {
	hit: bool,
	s0:  int,
	e0:  int,
}

@(test)
do_parallel_zero :: proc(t: ^testing.T) {
	n_called: int
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, 1)
	defer thread.pool_destroy(&pool)

	do_parallel(&pool, proc(task: thread.Task) {
		n := (^int)(task.data)
		n^ += 1
	}, &n_called, 0, 1)
	testing.expect(t, n_called == 0, "n=0 must not call the task")
}

@(test)
do_parallel_one :: proc(t: ^testing.T) {
	ctx: do_par_one_ctx
	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, 1)
	defer thread.pool_destroy(&pool)

	do_parallel(&pool, proc(task: thread.Task) {
		info := (^ParallelInfo(do_par_one_ctx))(task.data)
		info.sl.hit = true
		info.sl.s0 = info.a_start_off[0]
		info.sl.e0 = info.a_end_off[0]
	}, &ctx, 1, 1)
	testing.expect(t, ctx.hit, "n=1 must call the task")
	testing.expect(t, ctx.s0 == 0 && ctx.e0 == 1,
		"n=1 must produce range [0,1)")
}

@(test)
do_parallel_large :: proc(t: ^testing.T) {
	n := 1_000_000
	visited := make([]bool, n, context.allocator)
	defer delete(visited, context.allocator)

	vctx: do_par_bool_ctx
	vctx.arr = &visited

	pool: thread.Pool
	nthreads := 4
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)

	do_parallel(&pool, proc(task: thread.Task) {
		info := (^ParallelInfo(do_par_bool_ctx))(task.data)
		ctx := info.sl
		arr := ctx.arr^
		for i in info.a_start_off[task.user_index] ..< info.a_end_off[task.user_index] {
			arr[i] = true
		}
	}, &vctx, n, nthreads)

	for i in 0 ..< n {
		if !visited[i] {
			testing.expectf(t, false, "index %v not visited", i)
			return
		}
	}
}

@(test)
do_parallel_uneven :: proc(t: ^testing.T) {
	n := 10
	visited := make([]bool, n)
	defer delete(visited)

	vctx: do_par_bool_ctx
	vctx.arr = &visited

	pool: thread.Pool
	nthreads := 3
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)

	do_parallel(&pool, proc(task: thread.Task) {
		info := (^ParallelInfo(do_par_bool_ctx))(task.data)
		ctx := info.sl
		arr := ctx.arr^
		for i in info.a_start_off[task.user_index] ..< info.a_end_off[task.user_index] {
			arr[i] = true
		}
	}, &vctx, n, nthreads)

	for i in 0 ..< n {
		testing.expect(t, visited[i],
			"uneven partition: index not visited")
	}
}

@(test)
do_parallel_tid_range :: proc(t: ^testing.T) {
	n := 100
	tids := make([]int, n, context.allocator)
	defer delete(tids, context.allocator)

	tctx: do_par_int_ctx
	tctx.arr = &tids

	pool: thread.Pool
	nthreads := 4
	thread.pool_init(&pool, context.allocator, nthreads)
	defer thread.pool_destroy(&pool)

	do_parallel(&pool, proc(task: thread.Task) {
		info := (^ParallelInfo(do_par_int_ctx))(task.data)
		ctx := info.sl
		arr := ctx.arr^
		for i in info.a_start_off[task.user_index] ..< info.a_end_off[task.user_index] {
			arr[i] = task.user_index
		}
	}, &tctx, n, nthreads)

	for i in 0 ..< n {
		testing.expect(t, tids[i] >= 0 && tids[i] < nthreads,
			"tid out of range")
	}
}
