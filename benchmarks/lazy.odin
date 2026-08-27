// lazy.odin — Stage 11.6 benchmark: lazy vs eager equivalence + timing on 1M
// rows.
//
// Run with: odin run benchmarks/lazy.odin -file
//
// Workload: a synthetic 4-column CSV (id i64, score f64, ok bool, name
// string) with ~1% NULL score rows, 1M rows. The pipeline is the same in
// both engines — filter(score >= 400) -> sort(score desc) -> limit(10) — so
// the lazy run exercises the full scan->filter->sort->limit collect path.
// This benchmark verifies that the lazy plan produces the identical result
// to the eager pipeline (S11.5) and reports wall time for each: the two must
// be close, since the Stage 11 executor is eager-backed (S11.3) and there is
// no optimizer yet (Stage 12). Reported in ms and rows/s; a zero value
// should be treated with suspicion.

package main

import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"

import df "../src/dataframe"
import dfx "../src/dataframe/expr"
import lz "../src/dataframe/lazy"

SIZE :: 1_000_000

// write_csv builds a 4-column CSV of SIZE rows and returns its path.
write_csv :: proc() -> string {
	path := "/tmp/thunder_bench_lazy.csv"

	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)
	strings.write_string(&sb, "id,score,ok,name\n")
	for i in 0 ..< SIZE {
		score := f64(i % 1000)
		ok := i % 2 == 0
		if i % 97 == 0 {
			fmt.sbprintf(&sb, "%d,,%t,name%07d\n", i, ok, i)
		} else {
			fmt.sbprintf(&sb, "%d,%.2f,%t,name%07d\n", i, score, ok, i)
		}
	}
	text := strings.to_string(sb)
	if os_err := os.write_entire_file_from_string(path, text); os_err != os.ERROR_NONE {
		fmt.println("  write", path, "failed:", os_err)
	}
	return path
}

// results_equal compares the id/score of two pipelined results row for row.
// Both engines must produce the same top-10 by score.
results_equal :: proc(eager, lazy: ^df.DataFrame) -> bool {
	if df.dataframe_num_rows(eager) != df.dataframe_num_rows(lazy) {
		return false
	}
	ec, e_err := df.dataframe_column_at(eager, 0)
	lc, l_err := df.dataframe_column_at(lazy, 0)
	es, es_err := df.dataframe_column_at(eager, 1)
	ls, ls_err := df.dataframe_column_at(lazy, 1)
	if e_err != .None || l_err != .None || es_err != .None || ls_err != .None {
		return false
	}
	for i in 0 ..< df.dataframe_num_rows(eager) {
		eid, _, a_err := df.column_get(ec, i, i64)
		lid, _, b_err := df.column_get(lc, i, i64)
		ev, _, c_err := df.column_get(es, i, f64)
		lv, _, d_err := df.column_get(ls, i, f64)
		if a_err != .None || b_err != .None || c_err != .None || d_err != .None {
			return false
		}
		if eid != lid || ev != lv {
			return false
		}
	}
	return true
}

main :: proc() {
	path := write_csv()
	fmt.println("== lazy vs eager (1M rows): filter(score>=400) -> sort(score desc) -> limit(10) ==")

	// Expression context: must outlive both pipelines (expr nodes are
	// borrowed by the lazy plan until collect).
	ctx := dfx.context_create(context.allocator)
	defer dfx.context_destroy(&ctx)
	pred := dfx.ge(&ctx, dfx.col(&ctx, "score"), dfx.lit(&ctx, 400.0))
	keys := []df.Sort_Key{df.sort_key("score", .Desc)}

	// Warm-up both paths.
	{
		w, w_err := df.dataframe_read_csv(path)
		if w_err == .None {
			df.dataframe_destroy(&w)
		}
		lf := lz.scan_csv(path)
		defer lz.destroy(&lf)
		lf = lz.filter(lf, pred)
		lf = lz.limit(lf, 10)
		wo, c_err := lz.collect(lf)
		if c_err == .None {
			df.dataframe_destroy(&wo)
		}
	}

	eager_ms := f64(1e18)
	eager_res: df.DataFrame
	for _ in 0 ..< 3 {
		start := time.now()
		eager, r_err := df.dataframe_read_csv(path)
		if r_err != .None {
			fmt.println("  eager read failed:", r_err)
			return
		}
		f, f_err := df.dataframe_filter(&eager, pred)
		df.dataframe_destroy(&eager)
		if f_err != .None {
			fmt.println("  eager filter failed:", f_err)
			return
		}
		s, s_err := df.dataframe_sort(&f, keys)
		df.dataframe_destroy(&f)
		if s_err != .None {
			fmt.println("  eager sort failed:", s_err)
			return
		}
		h, h_err := df.dataframe_limit(&s, 10)
		df.dataframe_destroy(&s)
		if h_err != .None {
			fmt.println("  eager limit failed:", h_err)
			return
		}
		elapsed := time.since(start)
		eager_res = h
		if e := time.duration_milliseconds(elapsed); e < eager_ms {
			eager_ms = e
		}
	}
	defer df.dataframe_destroy(&eager_res)

	lazy_ms := f64(1e18)
	lazy_res: df.DataFrame
	for _ in 0 ..< 3 {
		lf := lz.scan_csv(path)
		defer lz.destroy(&lf)
		lf = lz.filter(lf, pred)
		lf = lz.sort(lf, keys)
		lf = lz.limit(lf, 10)

		start := time.now()
		out, c_err := lz.collect(lf)
		if c_err != .None {
			fmt.println("  lazy collect failed:", c_err)
			return
		}
		elapsed := time.since(start)
		if e := time.duration_milliseconds(elapsed); e < lazy_ms {
			lazy_ms = e
		}
		lazy_res = out
	}
	defer df.dataframe_destroy(&lazy_res)

	equal := results_equal(&eager_res, &lazy_res)
	fmt.println("  equivalence (row count + id/score):", equal)
	fmt.println("  rows returned:", df.dataframe_num_rows(&lazy_res))
	fmt.printf("  eager: %.2f ms  (%.0f rows/s)\n", eager_ms, f64(SIZE) / (eager_ms / 1000.0))
	fmt.printf("  lazy:  %.2f ms  (%.0f rows/s)\n", lazy_ms, f64(SIZE) / (lazy_ms / 1000.0))
	fmt.printf("  lazy/eager ratio: %.2f\n", lazy_ms / eager_ms)
}
