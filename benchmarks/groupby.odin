// groupby.odin — Stage 7.6 benchmark: hash-based group_by + agg over 100K /
// 1M / 10M rows.
//
// Run with: odin run benchmarks/groupby.odin -file
//
// Workload: a 2-column DataFrame (key i32 cycling over 1000 distinct groups,
// value f64 = (i % 1000) + 0.5), all rows valid. Two tables:
//
//   reference — the raw single-pass work: a hand-rolled hash-map
//               accumulation of count+sum per key over a raw slice pair (the
//               baseline the groupby does plus its per-row map puts).
//   groupby   — the public API: dataframe_group_by over the key column, then
//               dataframe_group_by_agg over count(value)+sum(value), then
//               destroy. Measures key encode + hash-map grouping + kernel
//               dispatch per group + result materialization.
//
// Since Stage 21 (DESIGN.md §14.7) dataframe_group_by dispatches to parallel
// partitioned hash grouping at >= PARALLEL_GROUPBY_THRESHOLD rows (100K), so
// every size below exercises the parallel kernel; smaller sizes fall back to
// the sequential loop.
//
// Results are printed as milliseconds; a zero value should be treated with
// suspicion (too fast to measure). No stdlib harness — manual timing.

package main

import "core:fmt"
import "core:time"

import df "../src/dataframe"
import dfx "../src/dataframe/expr"

SIZES := []int{100_000, 1_000_000, 10_000_000}
GROUPS: i32 = 1000

// make_df builds a 2-column DataFrame of n rows: key = i % GROUPS (i32),
// value = (i % 1000) + 0.5 (f64), all rows valid.
make_df :: proc(n: int) -> (d: df.DataFrame, ok: bool) {
	key, k_err := df.column_from("key", make([]i32, n))
	if k_err != .None {
		fmt.println("  make_df: key failed:", k_err)
		return {}, false
	}
	value, v_err := df.column_from("value", make([]f64, n))
	if v_err != .None {
		df.column_destroy(&key)
		fmt.println("  make_df: value failed:", v_err)
		return {}, false
	}
	for i in 0 ..< n {
		df.column_set(&key, i, i32(i % int(GROUPS)))
		df.column_set(&value, i, f64(i % 1000) + 0.5)
	}

	err: df.Error
	d, err = df.dataframe_from_columns([]^df.Column{&key, &value})
	if err != .None {
		df.column_destroy(&key)
		df.column_destroy(&value)
		fmt.println("  make_df: from_columns failed:", err)
		return {}, false
	}
	return d, true
}

// bench_naive is the raw work: one pass over two slices accumulating
// count+sum per key into a hash map (the same reduction group_by performs).
bench_naive :: proc(n: int) -> f64 {
	keys := make([]i32, n)
	values := make([]f64, n)
	defer delete(keys)
	defer delete(values)
	for i in 0 ..< n {
		keys[i] = i32(i % int(GROUPS))
		values[i] = f64(i % 1000) + 0.5
	}

	Acc :: struct { count: i64, sum: f64 }
	acc := make(map[i32]Acc, int(GROUPS))
	defer delete(acc)

	start := time.now()
	for i in 0 ..< n {
		a := acc[keys[i]]
		a.count += 1
		a.sum += values[i]
		acc[keys[i]] = a
	}
	elapsed := time.duration_milliseconds(time.since(start))

	if len(acc) != int(GROUPS) {
		fmt.println("  (check) naive group count =", len(acc), "for n =", n)
	}
	return elapsed
}

// bench_groupby times the full public group_by + group_by_agg path.
bench_groupby :: proc(n: int) -> f64 {
	d, ok := make_df(n)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)

	ctx := dfx.context_create(context.allocator)
	defer dfx.context_destroy(&ctx)

	start := time.now()
	gb, gerr := df.dataframe_group_by(&d, []^dfx.Expr{dfx.col(&ctx, "key")})
	if gerr != .None {
		fmt.println("  group_by error:", gerr)
		return 0
	}
	defer df.dataframe_group_by_destroy(&gb)

	out, aerr := df.dataframe_group_by_agg(&gb, []^dfx.Expr{
		dfx.alias(&ctx, dfx.count_(&ctx, dfx.col(&ctx, "value")), "count"),
		dfx.alias(&ctx, dfx.sum_(&ctx, dfx.col(&ctx, "value")), "sum"),
	})
	elapsed := time.duration_milliseconds(time.since(start))
	if aerr != .None {
		fmt.println("  group_by_agg error:", aerr)
		return 0
	}
	defer df.dataframe_destroy(&out)

	if df.dataframe_num_rows(&out) != int(GROUPS) {
		fmt.println("  (check) groupby rows =", df.dataframe_num_rows(&out), "for n =", n)
	}
	return elapsed
}

main :: proc() {
	fmt.printf("%12s | %14s | %14s\n", "rows", "naive(ms)", "groupby(ms)")
	for n in SIZES {
		naive := bench_naive(n)
		groupby := bench_groupby(n)
		fmt.printf("%12d | %14.3f | %14.3f\n", n, naive, groupby)
	}
}
