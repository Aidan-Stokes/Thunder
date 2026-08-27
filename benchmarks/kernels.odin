// kernels.odin — Stage 15.2 benchmark: specialized numeric kernel fast paths.
//
// Run with: odin run benchmarks/kernels.odin -file
//
// Measures the cost of the core numeric operations through the dataframe API
// to establish a baseline BEFORE adding all-valid fast paths to binary_arith,
// numeric_reduce_typed, neg_typed, and func_abs_typed.
//
// Two NULL configurations per operation:
//   all_valid — every row valid (valid == nil), the fast-path target
//   1pct_null — ~1% NULL rows (validity bitmap present)
//
// Results are milliseconds (best of 3).

package main

import "core:fmt"
import "core:mem"
import "core:time"

import df "../src/dataframe"
import dfx "../src/dataframe/expr"

SIZES := []int{1_000_000, 10_000_000}

best_of_3_sum :: proc(n: int, null_pct: int) -> f64 {
	best := bench_sum(n, null_pct)
	best = min(best, bench_sum(n, null_pct))
	best = min(best, bench_sum(n, null_pct))
	return best
}

bench_sum :: proc(n: int, null_pct: int) -> f64 {
	iv := make([]f64, n)
	defer delete(iv)
	valid := null_pct > 0 ? make([]bool, n) : nil
	defer if valid != nil { delete(valid) }

	for i in 0 ..< n {
		iv[i] = f64(i) + 0.5
		if valid != nil {
			valid[i] = i % 100 != 0
		}
	}

	col, cerr := df.column_from_with_valid("v", iv, valid)
	if cerr != .None {
		fmt.println("  sum col err:", cerr)
		return 0
	}
	defer df.column_destroy(&col)

	d, derr := df.dataframe_from_columns([]^df.Column{&col})
	if derr != .None {
		fmt.println("  sum df err:", derr)
		return 0
	}
	defer df.dataframe_destroy(&d)

	start := time.now()
	v, err := df.dataframe_sum(&d, "v")
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None {
		fmt.println("  sum err:", err)
	}
	if v == 0 {
		fmt.println("  (check) sum was zero for n =", n)
	}
	return elapsed
}

best_of_3_add :: proc(n: int, null_pct: int) -> f64 {
	best := bench_add(n, null_pct)
	best = min(best, bench_add(n, null_pct))
	best = min(best, bench_add(n, null_pct))
	return best
}

bench_add :: proc(n: int, null_pct: int) -> f64 {
	iv := make([]f64, n)
	defer delete(iv)
	valid := null_pct > 0 ? make([]bool, n) : nil
	defer if valid != nil { delete(valid) }

	for i in 0 ..< n {
		iv[i] = f64(i) + 0.5
		if valid != nil {
			valid[i] = i % 100 != 0
		}
	}

	a, aerr := df.column_from_with_valid("a", iv, valid)
	if aerr != .None {
		return 0
	}
	defer df.column_destroy(&a)

	b, berr := df.column_from_with_valid("b", iv, valid)
	if berr != .None {
		return 0
	}
	defer df.column_destroy(&b)

	d, derr := df.dataframe_from_columns([]^df.Column{&a, &b})
	if derr != .None {
		return 0
	}
	defer df.dataframe_destroy(&d)

	ctx := dfx.context_create(context.allocator)
	defer dfx.context_destroy(&ctx)
	e := dfx.add(&ctx, dfx.col(&ctx, "a"), dfx.col(&ctx, "b"))

	start := time.now()
	out, err := df.dataframe_select(&d, []^dfx.Expr{e})
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None {
		fmt.println("  add err:", err)
	}
	defer df.dataframe_destroy(&out)
	return elapsed
}

best_of_3_neg :: proc(n: int, null_pct: int) -> f64 {
	best := bench_neg(n, null_pct)
	best = min(best, bench_neg(n, null_pct))
	best = min(best, bench_neg(n, null_pct))
	return best
}

bench_neg :: proc(n: int, null_pct: int) -> f64 {
	iv := make([]f64, n)
	defer delete(iv)
	valid := null_pct > 0 ? make([]bool, n) : nil
	defer if valid != nil { delete(valid) }

	for i in 0 ..< n {
		iv[i] = f64(i) + 0.5
		if valid != nil {
			valid[i] = i % 100 != 0
		}
	}

	col, cerr := df.column_from_with_valid("v", iv, valid)
	if cerr != .None {
		return 0
	}
	defer df.column_destroy(&col)

	d, derr := df.dataframe_from_columns([]^df.Column{&col})
	if derr != .None {
		return 0
	}
	defer df.dataframe_destroy(&d)

	ctx := dfx.context_create(context.allocator)
	defer dfx.context_destroy(&ctx)
	e := dfx.neg(&ctx, dfx.col(&ctx, "v"))

	start := time.now()
	out, err := df.dataframe_select(&d, []^dfx.Expr{e})
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None {
		fmt.println("  neg err:", err)
	}
	defer df.dataframe_destroy(&out)
	return elapsed
}

best_of_3_abs :: proc(n: int, null_pct: int) -> f64 {
	best := bench_abs(n, null_pct)
	best = min(best, bench_abs(n, null_pct))
	best = min(best, bench_abs(n, null_pct))
	return best
}

bench_abs :: proc(n: int, null_pct: int) -> f64 {
	iv := make([]f64, n)
	defer delete(iv)
	valid := null_pct > 0 ? make([]bool, n) : nil
	defer if valid != nil { delete(valid) }

	for i in 0 ..< n {
		iv[i] = f64(i) + 0.5
		if valid != nil {
			valid[i] = i % 100 != 0
		}
	}

	col, cerr := df.column_from_with_valid("v", iv, valid)
	if cerr != .None {
		return 0
	}
	defer df.column_destroy(&col)

	d, derr := df.dataframe_from_columns([]^df.Column{&col})
	if derr != .None {
		return 0
	}
	defer df.dataframe_destroy(&d)

	ctx := dfx.context_create(context.allocator)
	defer dfx.context_destroy(&ctx)
	e := dfx.abs_(&ctx, dfx.col(&ctx, "v"))

	start := time.now()
	out, err := df.dataframe_select(&d, []^dfx.Expr{e})
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None {
		fmt.println("  abs err:", err)
	}
	defer df.dataframe_destroy(&out)
	return elapsed
}

main :: proc() {
	fmt.println("Stage 15.2 — numeric kernel baseline (before fast paths)")
	fmt.printf("%-12s | %5s | %12s | %12s\n", "operation", "null%", "1M(ms)", "10M(ms)")
	fmt.printf("%-12s-+-%5s-+-%12s-+-%12s\n", "------------", "-----", "------------", "------------")

	row :: proc(name: string, pct: int) {
		ms1 := best_of_3_sum(1_000_000, pct)  if name == "sum" else
		       best_of_3_add(1_000_000, pct)  if name == "add" else
		       best_of_3_neg(1_000_000, pct)  if name == "neg" else
		       best_of_3_abs(1_000_000, pct)
		ms10 := best_of_3_sum(10_000_000, pct)  if name == "sum" else
		        best_of_3_add(10_000_000, pct)  if name == "add" else
		        best_of_3_neg(10_000_000, pct)  if name == "neg" else
		        best_of_3_abs(10_000_000, pct)
		fmt.printf("%-12s | %5d | %12.3f | %12.3f\n", name, pct, ms1, ms10)
	}

	row("sum", 0)
	row("sum", 1)
	row("add", 0)
	row("add", 1)
	row("neg", 0)
	row("neg", 1)
	row("abs", 0)
	row("abs", 1)
}
