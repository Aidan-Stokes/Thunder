// optimizer.odin — Stage 12.7 benchmark: projection pushdown scan cost.
//
// Run with: odin run benchmarks/optimizer.odin -file
//
// Workload: a synthetic 20-column CSV (c00..c19, all f64) with 1M rows, where
// a query filters on c05 and projects c00 + c05. Three paths are timed:
//
//   full   eager read of all 20 columns -> filter -> select (no pushdown)
//   pruned eager read of only c00, c05 -> filter (raw scan savings)
//   lazy   collect(select(filter(scan_csv, c00,c05))) — the Stage 12
//          optimizer prunes the Scan_CSV to the two referenced columns, so
//          the lazy path should approach the pruned scan cost.
//
// All three produce the same result (verified row-for-row). Reported in ms
// and MB/s; the full/pruned read ratio shows the scan-cost win of S12.1.

package main

import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"

import df "../src/dataframe"
import dfx "../src/dataframe/expr"
import lz "../src/dataframe/lazy"

SIZE :: 1_000_000
NCOLS :: 20

// write_csv builds the 20-column CSV and returns its path and byte size.
write_csv :: proc() -> (path: string, size: int) {
	path = "/tmp/thunder_bench_opt.csv"

	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)
	strings.write_string(&sb, "c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19\n")
	for i in 0 ..< SIZE {
		score := f64(i % 1000)
		for j in 0 ..< NCOLS {
			if j > 0 {
				strings.write_byte(&sb, ',')
			}
			if j == 0 {
				fmt.sbprintf(&sb, "%.1f", f64(i))
			} else if j == 5 {
				fmt.sbprintf(&sb, "%.1f", score)
			} else {
				fmt.sbprintf(&sb, "%.1f", f64(i+j)*0.5)
			}
		}
		strings.write_byte(&sb, '\n')
	}
	text := strings.to_string(sb)
	size = len(text)
	if os_err := os.write_entire_file_from_string(path, text); os_err != os.ERROR_NONE {
		fmt.println("  write", path, "failed:", os_err)
	}
	return
}

// results_equal compares the projected id/score columns of two results.
results_equal :: proc(a, b: ^df.DataFrame) -> bool {
	if df.dataframe_num_rows(a) != df.dataframe_num_rows(b) {
		return false
	}
	aid, a_err := df.dataframe_column_at(a, 0)
	bid, b_err := df.dataframe_column_at(b, 0)
	asc, c_err := df.dataframe_column_at(a, 1)
	bsc, d_err := df.dataframe_column_at(b, 1)
	if a_err != .None || b_err != .None || c_err != .None || d_err != .None {
		return false
	}
	for i in 0 ..< df.dataframe_num_rows(a) {
		av, _, _ := df.column_get(aid, i, f64)
		bv, _, _ := df.column_get(bid, i, f64)
		as, _, _ := df.column_get(asc, i, f64)
		bs, _, _ := df.column_get(bsc, i, f64)
		if av != bv || as != bs {
			return false
		}
	}
	return true
}

// time_full times run_full three times (best of 3).
time_full :: proc(path: string, pred: ^dfx.Expr, exprs: []^dfx.Expr) -> (ms: f64, res: df.DataFrame) {
	ms = 1e18
	for _ in 0 ..< 3 {
		start := time.now()
		r := run_full(path, pred, exprs)
		if e := time.duration_milliseconds(time.since(start)); e < ms {
			ms = e
		}
		res = r
	}
	return
}

time_pruned :: proc(path: string, pred: ^dfx.Expr, cols: []string) -> (ms: f64, res: df.DataFrame) {
	ms = 1e18
	for _ in 0 ..< 3 {
		start := time.now()
		r := run_pruned(path, pred, cols)
		if e := time.duration_milliseconds(time.since(start)); e < ms {
			ms = e
		}
		res = r
	}
	return
}

time_lazy :: proc(path: string, pred: ^dfx.Expr, exprs: []^dfx.Expr) -> (ms: f64, res: df.DataFrame) {
	ms = 1e18
	for _ in 0 ..< 3 {
		start := time.now()
		r := run_lazy(path, pred, exprs)
		if e := time.duration_milliseconds(time.since(start)); e < ms {
			ms = e
		}
		res = r
	}
	return
}

run_full :: proc(path: string, pred: ^dfx.Expr, exprs: []^dfx.Expr) -> df.DataFrame {
	all, r_err := df.dataframe_read_csv(path)
	if r_err != .None {
		fmt.println("  full read failed:", r_err)
		return {}
	}
	defer df.dataframe_destroy(&all)
	f, f_err := df.dataframe_filter(&all, pred)
	if f_err != .None {
		fmt.println("  full filter failed:", f_err)
		return {}
	}
	defer df.dataframe_destroy(&f)
	s, s_err := df.dataframe_select(&f, exprs)
	if s_err != .None {
		fmt.println("  full select failed:", s_err)
		return {}
	}
	return s
}

run_pruned :: proc(path: string, pred: ^dfx.Expr, cols: []string) -> df.DataFrame {
	part, r_err := df.dataframe_read_csv_with_columns(path, cols)
	if r_err != .None {
		fmt.println("  pruned read failed:", r_err)
		return {}
	}
	defer df.dataframe_destroy(&part)
	f, f_err := df.dataframe_filter(&part, pred)
	if f_err != .None {
		fmt.println("  pruned filter failed:", f_err)
		return {}
	}
	return f
}

run_lazy :: proc(path: string, pred: ^dfx.Expr, exprs: []^dfx.Expr) -> df.DataFrame {
	lf := lz.scan_csv(path)
	defer lz.destroy(&lf)
	lf = lz.select(lz.filter(lf, pred), exprs)
	out, c_err := lz.collect(lf)
	if c_err != .None {
		fmt.println("  lazy collect failed:", c_err)
		return {}
	}
	return out
}

main :: proc() {
	path, size := write_csv()
	size_mb := f64(size) / 1e6
	fmt.printf("== projection pushdown, %d rows x %d cols (%.1f MB) ==", SIZE, NCOLS, size_mb)
	fmt.println()

	ctx := dfx.context_create(context.allocator)
	defer dfx.context_destroy(&ctx)
	pred := dfx.ge(&ctx, dfx.col(&ctx, "c05"), dfx.lit(&ctx, 400.0))
	exprs := []^dfx.Expr{dfx.col(&ctx, "c00"), dfx.col(&ctx, "c05")}
	cols := []string{"c00", "c05"}

	// Warm-up all paths.
	{
		w, w_err := df.dataframe_read_csv_with_columns(path, cols)
		if w_err == .None {
			df.dataframe_destroy(&w)
		}
		lf := lz.scan_csv(path)
		defer lz.destroy(&lf)
		lf = lz.select(lz.filter(lf, pred), exprs)
		wo, c_err := lz.collect(lf)
		if c_err == .None {
			df.dataframe_destroy(&wo)
		}
	}

	// Full eager read of all 20 columns, then filter + select.
	full_ms, full_res := time_full(path, pred, exprs)
	defer df.dataframe_destroy(&full_res)

	// Pruned eager read of only the two projected columns.
	pruned_ms, pruned_res := time_pruned(path, pred, cols)
	defer df.dataframe_destroy(&pruned_res)

	// Lazy: the optimizer prunes the scan to c00, c05 automatically.
	lazy_ms, lazy_res := time_lazy(path, pred, exprs)
	defer df.dataframe_destroy(&lazy_res)

	fmt.println("  rows returned:", df.dataframe_num_rows(&lazy_res))
	fmt.println("  equivalence full==pruned:", results_equal(&full_res, &pruned_res))
	fmt.println("  equivalence full==lazy:  ", results_equal(&full_res, &lazy_res))
	fmt.printf("  full   read20+filter+select: %8.2f ms  (%.0f MB/s)\n", full_ms, size_mb / (full_ms / 1000.0))
	fmt.printf("  pruned read2 +filter:        %8.2f ms  (%.0f MB/s)\n", pruned_ms, size_mb / (pruned_ms / 1000.0))
	fmt.printf("  lazy   collect (pushdown):   %8.2f ms\n", lazy_ms)
	fmt.printf("  scan-cost ratio pruned/full: %.2fx\n", full_ms / pruned_ms)
	fmt.printf("  lazy/full ratio:             %.2fx\n", full_ms / lazy_ms)
}
