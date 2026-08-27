// arena.odin — S15.6 benchmark: chained expression evaluation with arena.
//
// Run with: odin run benchmarks/arena.odin -file
//
// Measures the impact of OpArena on expression-heavy workloads:
//   - chained:  a + b * 2 - c + 1 (4 ops, 5 intermediate columns)
//   - filter:   filter on complex predicate (value > 0.5) & (value < 0.8)
//   - select:   select with multiple computed columns
//
// Sizes: 10K / 100K / 1M rows.

package main

import "core:fmt"
import "core:time"
import "base:runtime"

import df "../src/dataframe"
import dfx "../src/dataframe/expr"

SIZES := []int{10_000, 100_000, 1_000_000}

make_df :: proc(n: int) -> (d: df.DataFrame, ok: bool) {
	a, a_err := df.column_from("a", make([]f64, n))
	b, b_err := df.column_from("b", make([]f64, n))
	c, c_err := df.column_from("c", make([]f64, n))
	val, v_err := df.column_from("value", make([]f64, n))
	if a_err != .None || b_err != .None || c_err != .None || v_err != .None {
		fmt.println("make_df failed")
		return {}, false
	}
	for i in 0 ..< n {
		df.column_set(&a, i, f64(i))
		df.column_set(&b, i, f64(n - i))
		df.column_set(&c, i, f64(i * 2))
		df.column_set(&val, i, f64(i % 1000) / 1000.0)
	}
	err: df.Error
	d, err = df.dataframe_from_columns([]^df.Column{&a, &b, &c, &val})
	if err != .None {
		fmt.println("from_columns failed:", err)
		return {}, false
	}
	return d, true
}

// bench_chained measures a + b * 2 - c + 1 (4 binary ops, 5 intermediates).
bench_chained :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok { return 0 }
	defer df.dataframe_destroy(&d)

	ctx := dfx.context_create(context.allocator)
	defer dfx.context_destroy(&ctx)

	e := dfx.add(&ctx,
		dfx.sub(&ctx,
			dfx.mul(&ctx, dfx.col(&ctx, "b"), dfx.lit(&ctx, 2.0)),
			dfx.col(&ctx, "c"),
		),
		dfx.lit(&ctx, 1.0),
	)

	start := time.now()
	result, err := df.expr_eval(context.allocator, &d, e)
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None {
		fmt.println("chained eval error:", err)
		return 0
	}
	defer df.column_destroy(&result)

	// checksum to prevent dead-code elimination
	sum: f64
	fv := transmute([]f64)runtime.Raw_Slice{data = result.data, len = df.column_len(&result)}
	for v in fv { sum += v }
	if sum == 0 { fmt.println("  warning: sum == 0") }
	return elapsed
}

// bench_filter measures filter on (value > 0.5) & (value < 0.8).
bench_filter :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok { return 0 }
	defer df.dataframe_destroy(&d)

	ctx := dfx.context_create(context.allocator)
	defer dfx.context_destroy(&ctx)

	pred := dfx.and_(&ctx,
		dfx.gt(&ctx, dfx.col(&ctx, "value"), dfx.lit(&ctx, 0.5)),
		dfx.lt(&ctx, dfx.col(&ctx, "value"), dfx.lit(&ctx, 0.8)),
	)

	start := time.now()
	filtered, err := df.dataframe_filter(&d, pred)
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None {
		fmt.println("filter error:", err)
		return 0
	}
	defer df.dataframe_destroy(&filtered)
	return elapsed
}

// bench_select measures select with 3 computed columns.
bench_select :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok { return 0 }
	defer df.dataframe_destroy(&d)

	ctx := dfx.context_create(context.allocator)
	defer dfx.context_destroy(&ctx)

	sel := []^dfx.Expr{
		dfx.alias(&ctx, dfx.add(&ctx, dfx.col(&ctx, "a"), dfx.col(&ctx, "b")), "ab_sum"),
		dfx.alias(&ctx, dfx.mul(&ctx, dfx.col(&ctx, "a"), dfx.lit(&ctx, 0.5)), "a_half"),
		dfx.alias(&ctx, dfx.sub(&ctx, dfx.col(&ctx, "b"), dfx.col(&ctx, "c")), "b_minus_c"),
	}

	start := time.now()
	out, err := df.dataframe_select(&d, sel)
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None {
		fmt.println("select error:", err)
		return 0
	}
	defer df.dataframe_destroy(&out)
	return elapsed
}

main :: proc() {
	fmt.printf("%10s | %14s | %14s | %14s\n", "rows", "chained(ms)", "filter(ms)", "select(ms)")
	for n in SIZES {
		chained := bench_chained(n)
		filter := bench_filter(n)
		sel := bench_select(n)
		fmt.printf("%10d | %14.3f | %14.3f | %14.3f\n", n, chained, filter, sel)
	}
}
