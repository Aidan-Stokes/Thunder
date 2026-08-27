// ops.odin — Stage 4.7 benchmark: filter + select over 1K/100K/1M/10M rows.
//
// Run with: odin run benchmarks/ops.odin -file
//
// Workload: filter a 3-column DataFrame (id i64, group i32, value f64) on
// `value > 0.5` (~50% of rows kept), then select `id` and `value * 2`. Each
// size measures:
//   - baseline: hand-rolled mask pass + gather into two output slices (the
//     work the API performs, without the eager materialization overhead)
//   - api:      full eager path — build exprs, dataframe_filter (materializes
//     a new 3-column DataFrame), dataframe_select (materializes again)
//
// Results are printed as milliseconds; a zero value should be treated with
// suspicion (too fast to measure). No stdlib harness — manual timing.

package main

import "core:fmt"
import "core:time"

import df "../src/dataframe"
import dfx "../src/dataframe/expr"

SIZES := []int{1_000, 100_000, 1_000_000, 10_000_000}

// make_df builds a 3-column DataFrame of n rows: id = 0..n-1 (i64),
// group = i % 7 (i32), value = (i % 1000) / 1000 (f64, ~50% > 0.5).
make_df :: proc(n: int) -> (d: df.DataFrame, ok: bool) {
	id, id_err := df.column_from("id", make([]i64, n))
	if id_err != .None {
		fmt.println("  make_df: id failed:", id_err)
		return {}, false
	}
	group, g_err := df.column_from("group", make([]i32, n))
	if g_err != .None {
		df.column_destroy(&id)
		fmt.println("  make_df: group failed:", g_err)
		return {}, false
	}
	value, v_err := df.column_from("value", make([]f64, n))
	if v_err != .None {
		df.column_destroy(&id)
		df.column_destroy(&group)
		fmt.println("  make_df: value failed:", v_err)
		return {}, false
	}
	for i in 0 ..< n {
		df.column_set(&id, i, i64(i))
		df.column_set(&group, i, i32(i % 7))
		df.column_set(&value, i, f64(i % 1000) / 1000.0)
	}

	err: df.Error
	d, err = df.dataframe_from_columns([]^df.Column{&id, &group, &value})
	if err != .None {
		df.column_destroy(&id)
		df.column_destroy(&group)
		df.column_destroy(&value)
		fmt.println("  make_df: from_columns failed:", err)
		return {}, false
	}
	return d, true
}

// bench_baseline is the raw work: mask pass + gather into two slices.
bench_baseline :: proc(n: int) -> f64 {
	ids := make([]i64, n)
	defer delete(ids)
	groups := make([]i32, n)
	defer delete(groups)
	values := make([]f64, n)
	defer delete(values)
	for i in 0 ..< n {
		ids[i] = i64(i)
		groups[i] = i32(i % 7)
		values[i] = f64(i % 1000) / 1000.0
	}

	keep := make([]bool, n)
	defer delete(keep)

	start := time.now()
	count := 0
	for i in 0 ..< n {
		keep[i] = values[i] > 0.5
		if keep[i] {
			count += 1
		}
	}
	id_out := make([]i64, count)
	defer delete(id_out)
	val_out := make([]f64, count)
	defer delete(val_out)
	at := 0
	for i in 0 ..< n {
		if keep[i] {
			id_out[at] = ids[i]
			val_out[at] = values[i] * 2
			at += 1
		}
	}
	elapsed := time.duration_milliseconds(time.since(start))

	sum: i64
	for i in 0 ..< len(id_out) {
		sum += id_out[i] + i64(val_out[i])
	}
	if sum == 0 {
		fmt.println("  (sum check) baseline sum was zero for n =", n)
	}
	return elapsed
}

// bench_api is the full eager path: filter then select via the library.
bench_api :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)

	ctx := dfx.context_create(context.allocator)
	defer dfx.context_destroy(&ctx)

	pred := dfx.gt(&ctx, dfx.col(&ctx, "value"), dfx.lit(&ctx, 0.5))
	sel := []^dfx.Expr {
		dfx.col(&ctx, "id"),
		dfx.alias(&ctx, dfx.mul(&ctx, dfx.col(&ctx, "value"), dfx.lit(&ctx, 2.0)), "value_x2"),
	}

	start := time.now()
	filtered, f_err := df.dataframe_filter(&d, pred)
	if f_err == .None {
		defer df.dataframe_destroy(&filtered)
		out, s_err := df.dataframe_select(&filtered, sel)
		if s_err == .None {
			defer df.dataframe_destroy(&out)
		}
	}
	elapsed := time.duration_milliseconds(time.since(start))

	if f_err != .None {
		fmt.println("  filter error:", f_err)
		return 0
	}
	return elapsed
}

main :: proc() {
	fmt.printf("%10s | %14s | %14s | %8s\n", "rows", "baseline(ms)", "api(ms)", "x slower")
	for n in SIZES {
		base := bench_baseline(n)
		api := bench_api(n)
		ratio := 0.0
		if base > 0 {
			ratio = api / base
		}
		fmt.printf("%10d | %14.3f | %14.3f | %8.2f\n", n, base, api, ratio)
	}
}
