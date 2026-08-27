// parallel.odin — S15.4 benchmark: parallel.do_parallel vs sequential.
//
// Run with: odin run benchmarks/parallel.odin -file
//
// Measures wall-clock time for a trivial per-element operation (i64 multiply)
// executed either sequentially or via parallel.do_parallel, across several
// element counts.  Results are microseconds (best of 5).

package main

import "core:fmt"
import "core:time"
import "core:thread"
import "core:mem"

import "../../libs/parallel"

SIZES := []int{1_000, 10_000, 100_000, 1_000_000, 10_000_000}
NTHREADS :: 8

bench_seq :: proc(n: int) -> f64 {
	out := make([]i64, n)
	defer delete(out)
	best := f64(1e18)
	for _ in 0 ..< 5 {
		start := time.now()
		for i in 0 ..< n {
			out[i] = i64(i) * i64(i)
		}
		elapsed := f64(time.duration_microseconds(time.since(start)))
		best = min(best, elapsed)
	}
	return best
}

bench_par :: proc(n: int) -> f64 {
	out := make([]i64, n)
	defer delete(out)
	best := f64(1e18)
	for _ in 0 ..< 5 {
		pool: thread.Pool
		thread.pool_init(&pool, context.allocator, NTHREADS)

		start := time.now()
		parallel.do_parallel(&pool, proc(t: thread.Task) {
			info := (^parallel.ParallelInfo([]i64))(t.data)
			arr := info.sl^
			for i in info.a_start_off[t.user_index] ..< info.a_end_off[t.user_index] {
				arr[i] = i64(i) * i64(i)
			}
		}, &out, n, NTHREADS)
		elapsed := f64(time.duration_microseconds(time.since(start)))

		thread.pool_destroy(&pool)
		best = min(best, elapsed)
	}
	return best
}

main :: proc() {
	fmt.println("S15.4 — parallel.do_parallel benchmark (best of 5)")
	fmt.printf("%-12s | %10s | %10s | %6s\n", "elements", "seq(us)", "par(us)", "speedup")
	fmt.printf("%-12s-+-%10s-+-%10s-+-%6s\n", "------------", "----------", "----------", "------")

	for n in SIZES {
		seq_us := bench_seq(n)
		par_us := bench_par(n)
		speedup := 1.0
		if par_us > 0 {
			speedup = seq_us / par_us
		}
		fmt.printf("%-12d | %10.0f | %10.0f | %5.2fx\n", n, seq_us, par_us, speedup)
	}
}
