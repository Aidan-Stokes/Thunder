// join.odin — Stage 15.3 benchmark: hash join tuning + parallel probe.
//
// Run with: odin run benchmarks/join.odin -file
//
// Measures inner join time on integer keys at 100K/1M/10M left rows,
// right side fixed at 100K rows with 10K unique keys (1% key
// selectivity, ~10x fan-out per key).  This stresses the hash-build,
// probe, and materialize phases.
//
// Results are milliseconds (best of 3).

package main

import "core:fmt"
import "core:time"

import df "../src/dataframe"

SIZES := []int{100_000, 1_000_000, 10_000_000}
RIGHT_N :: 100_000
RIGHT_UNIQUE :: 10_000

best_of_3_join :: proc(n: int) -> f64 {
	best := bench_join(n)
	best = min(best, bench_join(n))
	best = min(best, bench_join(n))
	return best
}

bench_join :: proc(n: int) -> f64 {
	left_keys := make([]i32, n)
	right_keys := make([]i32, RIGHT_N)
	right_payload := make([]f64, RIGHT_N)
	defer delete(left_keys)
	defer delete(right_keys)
	defer delete(right_payload)

	for i in 0 ..< n {
		left_keys[i] = i32(i % RIGHT_UNIQUE)
	}
	for i in 0 ..< RIGHT_N {
		right_keys[i] = i32(i % RIGHT_UNIQUE)
		right_payload[i] = f64(i)
	}

	lk, lerr := df.column_from("lk", left_keys)
	if lerr != .None { return 0 }
	defer df.column_destroy(&lk)

	rk, rerr := df.column_from("rk", right_keys)
	if rerr != .None { return 0 }
	defer df.column_destroy(&rk)

	rp, rperr := df.column_from("rp", right_payload)
	if rperr != .None { return 0 }
	defer df.column_destroy(&rp)

	left, lerr2 := df.dataframe_from_columns([]^df.Column{&lk})
	if lerr2 != .None { return 0 }
	defer df.dataframe_destroy(&left)

	right, rerr2 := df.dataframe_from_columns([]^df.Column{&rk, &rp})
	if rerr2 != .None { return 0 }
	defer df.dataframe_destroy(&right)

	start := time.now()
	out, jerr := df.dataframe_inner_join(&left, &right, []string{"lk"}, []string{"rk"})
	elapsed := time.duration_milliseconds(time.since(start))
	if jerr != .None {
		fmt.println("  join err:", jerr)
	}
	defer df.dataframe_destroy(&out)
	return f64(elapsed)
}

main :: proc() {
	fmt.println("Stage 15.3 — hash join baseline")
	fmt.printf("%-12s | %12s\n", "left_rows", "inner(ms)")
	fmt.printf("%-12s-+-%12s\n", "------------", "------------")

	for n in SIZES {
		ms := best_of_3_join(n)
		fmt.printf("%-12d | %12.3f\n", n, ms)
	}
}
