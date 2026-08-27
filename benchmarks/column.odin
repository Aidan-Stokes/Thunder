// column.odin — Stage 1.10 benchmark: column create + typed get.
//
// Run with: odin run benchmarks/column.odin -file
//
// Sizes: 1K / 100K / 1M rows. Each size is measured for:
//   - column_from (copying constructor)
//   - column_get (typed) over the whole column
//
// Results are printed as milliseconds; a zero value should be treated with
// suspicion (too fast to measure). No stdlib harness — manual timing.

package main

import "core:fmt"
import "core:time"

import df "../src/dataframe"

SIZES := []int{1_000, 100_000, 1_000_000}

bench_create :: proc(n: int) -> f64 {
	src := make([]i64, n)
	defer delete(src)
	for i in 0 ..< n {
		src[i] = i64(i)
	}

	start := time.now()
	col, err := df.column_from("bench", src)
	df.column_destroy(&col)
	elapsed := time.duration_milliseconds(time.since(start))

	if err != .None {
		fmt.println("  create error:", err)
	}
	return elapsed
}

bench_get :: proc(n: int) -> f64 {
	src := make([]i64, n)
	defer delete(src)
	for i in 0 ..< n {
		src[i] = i64(i)
	}
	col, err := df.column_from("bench", src)
	defer df.column_destroy(&col)
	if err != .None {
		fmt.println("  create error:", err)
		return 0
	}

	sum: i64
	start := time.now()
	for i in 0 ..< n {
		v, valid, g_err := df.column_get(&col, i, i64)
		if g_err != .None {
			fmt.println("  get error:", g_err)
			return 0
		}
		if valid {
			sum += v
		}
	}
	elapsed := time.duration_milliseconds(time.since(start))

	if sum == 0 {
		fmt.println("  (sum check) sum was zero for n =", n)
	}
	return elapsed
}

main :: proc() {
	fmt.printf("%10s | %10s | %10s\n", "rows", "create(ms)", "get(ms)")
	for n in SIZES {
		c := bench_create(n)
		g := bench_get(n)
		fmt.printf("%10d | %10.3f | %10.3f\n", n, c, g)
	}
}
