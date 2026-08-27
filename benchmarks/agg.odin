// agg.odin — Stage 6.6 benchmark: aggregations over 1M / 10M rows.
//
// Run with: odin run benchmarks/agg.odin -file
//
// Workload: a 2-column DataFrame (id i64 = 0..n-1, value f64 =
// (i % 1000) + 0.5), all rows valid. Two tables:
//
//   reference — the cost of a sum over `value` three ways:
//     naive:    hand-rolled single-pass f64 loop over a raw slice (the
//               baseline work the kernel performs)
//     api:      per-column scalar path (stats.odin dataframe_sum: column
//               lookup + kernel + result column alloc/destroy)
//     expr:     full expression path (dataframe_select over an aliased
//               sum_(col("value")) — agg results are unnamed, so alias is
//               required to pass the select name check)
//
//   aggregations — each per-column scalar aggregation over 1M / 10M rows:
//     count, sum, mean, var, min, max, median, quantile, skew, kurtosis,
//     n_unique (over the high-cardinality id column), mode, cov, corr.
//
// Results are printed as milliseconds; a zero value should be treated with
// suspicion (too fast to measure). No stdlib harness — manual timing.

package main

import "core:fmt"
import "core:math"
import "core:time"

import df "../src/dataframe"
import dfx "../src/dataframe/expr"

SIZES := []int{1_000_000, 10_000_000}

// make_df builds a 2-column DataFrame of n rows: id = 0..n-1 (i64),
// value = (i % 1000) + 0.5 (f64, 1000 distinct values, all rows valid).
make_df :: proc(n: int) -> (d: df.DataFrame, ok: bool) {
	id, id_err := df.column_from("id", make([]i64, n))
	if id_err != .None {
		fmt.println("  make_df: id failed:", id_err)
		return {}, false
	}
	value, v_err := df.column_from("value", make([]f64, n))
	if v_err != .None {
		df.column_destroy(&id)
		fmt.println("  make_df: value failed:", v_err)
		return {}, false
	}
	for i in 0 ..< n {
		df.column_set(&id, i, i64(i))
		df.column_set(&value, i, f64(i % 1000) + 0.5)
	}

	err: df.Error
	d, err = df.dataframe_from_columns([]^df.Column{&id, &value})
	if err != .None {
		df.column_destroy(&id)
		df.column_destroy(&value)
		fmt.println("  make_df: from_columns failed:", err)
		return {}, false
	}
	return d, true
}

// bench_naive_sum is the raw work: one pass over an f64 slice.
bench_naive_sum :: proc(n: int) -> f64 {
	values := make([]f64, n)
	defer delete(values)
	for i in 0 ..< n {
		values[i] = f64(i % 1000) + 0.5
	}

	start := time.now()
	sum := 0.0
	for i in 0 ..< n {
		sum += values[i]
	}
	elapsed := time.duration_milliseconds(time.since(start))

	if sum == 0 {
		fmt.println("  (sum check) naive sum was zero for n =", n)
	}
	return elapsed
}

// bench_api_sum times the per-column scalar sum path.
bench_api_sum :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)

	start := time.now()
	v, err := df.dataframe_sum(&d, "value")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None {
		fmt.println("  sum error:", err)
		return 0
	}
	if v == 0 {
		fmt.println("  (sum check) api sum was zero for n =", n)
	}
	return elapsed
}

// bench_expr_sum times the full expression path: eval sum_(col("value")) via
// dataframe_select (the public surface that evaluates expressions).
bench_expr_sum :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)

	ctx := dfx.context_create(context.allocator)
	defer dfx.context_destroy(&ctx)
	e := dfx.alias(&ctx, dfx.sum_(&ctx, dfx.col(&ctx, "value")), "sum")

	start := time.now()
	out, err := df.dataframe_select(&d, []^dfx.Expr{e})
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None {
		fmt.println("  expr sum error:", err)
		return 0
	}
	defer df.dataframe_destroy(&out)
	return elapsed
}

// --- per-column scalar aggregations -----------------------------------------

bench_count :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)
	start := time.now()
	v, err := df.dataframe_count(&d, "id")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None || v != i64(n) {
		fmt.println("  count error:", err, "got", v)
	}
	return elapsed
}

bench_mean :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)
	start := time.now()
	v, err := df.dataframe_mean(&d, "value")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None || math.is_nan(v) {
		fmt.println("  mean error:", err)
	}
	return elapsed
}

bench_var :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)
	start := time.now()
	v, err := df.dataframe_var(&d, "value")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None || math.is_nan(v) {
		fmt.println("  var error:", err)
	}
	return elapsed
}

bench_min :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)
	start := time.now()
	v, err := df.dataframe_min(&d, "value")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None {
		fmt.println("  min error:", err)
		return 0
	}
	if v != f64(0.5) {
		fmt.println("  (min check) min != 0.5 for n =", n)
	}
	return elapsed
}

bench_max :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)
	start := time.now()
	v, err := df.dataframe_max(&d, "value")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None {
		fmt.println("  max error:", err)
		return 0
	}
	if v != f64(999.5) {
		fmt.println("  (max check) max != 999.5 for n =", n)
	}
	return elapsed
}

bench_median :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)
	start := time.now()
	v, err := df.dataframe_median(&d, "value")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None || math.is_nan(v) {
		fmt.println("  median error:", err)
	}
	return elapsed
}

bench_quantile :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)
	start := time.now()
	v, err := df.dataframe_quantile(&d, "value", 0.25)
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None || math.is_nan(v) {
		fmt.println("  quantile error:", err)
	}
	return elapsed
}

bench_skew :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)
	start := time.now()
	v, err := df.dataframe_skew(&d, "value")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None || math.is_nan(v) {
		fmt.println("  skew error:", err)
	}
	return elapsed
}

bench_kurtosis :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)
	start := time.now()
	v, err := df.dataframe_kurtosis(&d, "value")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None || math.is_nan(v) {
		fmt.println("  kurtosis error:", err)
	}
	return elapsed
}

// n_unique runs over the id column: n distinct i64 values, worst case for the
// hash-set kernel.
bench_n_unique :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)
	start := time.now()
	v, err := df.dataframe_n_unique(&d, "id")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None || v != i64(n) {
		fmt.println("  n_unique error:", err, "got", v)
	}
	return elapsed
}

// mode runs over the value column: 1000 distinct values, so the hash map is
// small and per-row lookup dominates.
bench_mode :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)
	start := time.now()
	v, err := df.dataframe_mode(&d, "value")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None {
		fmt.println("  mode error:", err)
		return 0
	}
	if v != f64(0.5) {
		fmt.println("  (mode check) mode != 0.5 for n =", n)
	}
	return elapsed
}

bench_cov :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)
	start := time.now()
	v, err := df.dataframe_cov(&d, "id", "value")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None || math.is_nan(v) {
		fmt.println("  cov error:", err)
	}
	return elapsed
}

bench_corr :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)
	start := time.now()
	v, err := df.dataframe_corr(&d, "id", "value")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None || math.is_nan(v) {
		fmt.println("  corr error:", err)
	}
	return elapsed
}

main :: proc() {
	fmt.printf("%12s | %14s | %14s | %14s\n", "rows", "naive(ms)", "api(ms)", "expr(ms)")
	for n in SIZES {
		naive := bench_naive_sum(n)
		api := bench_api_sum(n)
		expr := bench_expr_sum(n)
		fmt.printf("%12d | %14.3f | %14.3f | %14.3f\n", n, naive, api, expr)
	}

	fmt.println()
	fmt.printf("%-12s | %10s | %10s\n", "agg", "1M(ms)", "10M(ms)")
	row :: proc(name: string, a, b: f64) {
		fmt.printf("%-12s | %10.3f | %10.3f\n", name, a, b)
	}
	row("count", bench_count(1_000_000), bench_count(10_000_000))
	row("sum", bench_api_sum(1_000_000), bench_api_sum(10_000_000))
	row("mean", bench_mean(1_000_000), bench_mean(10_000_000))
	row("var", bench_var(1_000_000), bench_var(10_000_000))
	row("min", bench_min(1_000_000), bench_min(10_000_000))
	row("max", bench_max(1_000_000), bench_max(10_000_000))
	row("median", bench_median(1_000_000), bench_median(10_000_000))
	row("quantile", bench_quantile(1_000_000), bench_quantile(10_000_000))
	row("skew", bench_skew(1_000_000), bench_skew(10_000_000))
	row("kurtosis", bench_kurtosis(1_000_000), bench_kurtosis(10_000_000))
	row("n_unique", bench_n_unique(1_000_000), bench_n_unique(10_000_000))
	row("mode", bench_mode(1_000_000), bench_mode(10_000_000))
	row("cov", bench_cov(1_000_000), bench_cov(10_000_000))
	row("corr", bench_corr(1_000_000), bench_corr(10_000_000))
}
