package dataframe

// Window function tests (Stage 14.4/14.5, DESIGN.md §18.2).

import "core:testing"
import "core:math"
import "expr"

// window_test_df builds a partitioned fixture:
//
//	grp (string) ["a", "a", "b", "b", "b"]
//	v    (f64)  [1.0, 3.0, 10.0, 20.0, 30.0]
window_test_df :: proc(t: ^testing.T) -> (df: DataFrame, ctx: expr.Context) {
	grp: Column
	v: Column
	err: Error
	grp, err = column_from("grp", []string{"a", "a", "b", "b", "b"})
	testing.expect(t, err == .None, "grp column")
	v, err = column_from("v", []f64{1.0, 3.0, 10.0, 20.0, 30.0})
	testing.expect(t, err == .None, "v column")
	df, err = dataframe_from_columns([]^Column{&grp, &v})
	testing.expect(t, err == .None, "from_columns")
	ctx = expr.context_create(context.allocator)
	return
}

@(test)
window_eval_row_number_all :: proc(t: ^testing.T) {
	df, ctx := window_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.window_row_number(&ctx, nil))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(i64), "row_number dtype")
	ov := column_typed_view(&out, i64)
	testing.expect(t, len(ov) == 5, "row count")
	for i in 0 ..< 5 {
		testing.expect(t, ov[i] == i64(i + 1), "row_number value")
		testing.expect(t, column_is_valid(&out, i), "row_number valid")
	}
}

@(test)
window_eval_row_number_partitioned :: proc(t: ^testing.T) {
	df, ctx := window_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	over := []^expr.Expr{expr.col(&ctx, "grp")}
	out := eval_ok(t, &df, expr.window_row_number(&ctx, over))
	defer column_destroy(&out)

	ov := column_typed_view(&out, i64)
	testing.expect(t, ov[0] == 1 && ov[1] == 2, "partition a")
	testing.expect(t, ov[2] == 1 && ov[3] == 2 && ov[4] == 3, "partition b")
}

@(test)
window_eval_rank_average_partitioned :: proc(t: ^testing.T) {
	df, ctx := window_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	over := []^expr.Expr{expr.col(&ctx, "grp")}
	out := eval_ok(t, &df, expr.window_rank(&ctx, expr.col(&ctx, "v"), .Average, over))
	defer column_destroy(&out)

	ov := column_typed_view(&out, f64)
	testing.expect(t, ov[0] == 1 && ov[1] == 2, "partition a ranks")
	testing.expect(t, ov[2] == 1 && ov[3] == 2 && ov[4] == 3, "partition b ranks")
}

@(test)
window_eval_rank_tie_methods :: proc(t: ^testing.T) {
	grp: Column
	v: Column
	err: Error
	grp, err = column_from("grp", []string{"a", "a", "b", "b", "b"})
	testing.expect(t, err == .None, "grp column")
	v, err = column_from("v", []f64{2.0, 2.0, 1.0, 2.0, 2.0})
	testing.expect(t, err == .None, "v column")
	df, dferr := dataframe_from_columns([]^Column{&grp, &v})
	testing.expect(t, dferr == .None, "from_columns")
	ctx := expr.context_create(context.allocator)
	defer expr_test_destroy(t, &df, &ctx)

	over := []^expr.Expr{expr.col(&ctx, "grp")}
	// partition a: [2,2] ties; partition b: [1,2,2]

	rnk := eval_ok(t, &df, expr.window_rank(&ctx, expr.col(&ctx, "v"), .Min, over))
	defer column_destroy(&rnk)
	ov := column_typed_view(&rnk, f64)
	testing.expect(t, ov[0] == 1 && ov[1] == 1, "min: ties get best")
	testing.expect(t, ov[2] == 1 && ov[3] == 2 && ov[4] == 2, "min: b")

	rnk_max := eval_ok(t, &df, expr.window_rank(&ctx, expr.col(&ctx, "v"), .Max, over))
	defer column_destroy(&rnk_max)
	om := column_typed_view(&rnk_max, f64)
	testing.expect(t, om[0] == 2 && om[1] == 2, "max: ties get worst")
	testing.expect(t, om[2] == 1 && om[3] == 3 && om[4] == 3, "max: b")

	rnk_avg := eval_ok(t, &df, expr.window_rank(&ctx, expr.col(&ctx, "v"), .Average, over))
	defer column_destroy(&rnk_avg)
	oa := column_typed_view(&rnk_avg, f64)
	testing.expect(t, oa[0] == 1.5 && oa[1] == 1.5, "average: ties share mean")
	testing.expect(t, oa[2] == 1 && oa[3] == 2.5 && oa[4] == 2.5, "average: b")

	rnk_dense := eval_ok(t, &df, expr.window_rank(&ctx, expr.col(&ctx, "v"), .Dense, over))
	defer column_destroy(&rnk_dense)
	od := column_typed_view(&rnk_dense, f64)
	testing.expect(t, od[0] == 1 && od[1] == 1, "dense: a")
	testing.expect(t, od[2] == 1 && od[3] == 2 && od[4] == 2, "dense: b")
}

@(test)
window_eval_cum_sum_partitioned :: proc(t: ^testing.T) {
	df, ctx := window_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	over := []^expr.Expr{expr.col(&ctx, "grp")}
	out := eval_ok(t, &df, expr.window_cum_sum(&ctx, expr.col(&ctx, "v"), over))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(f64), "cum_sum dtype = child")
	ov := column_typed_view(&out, f64)
	testing.expect(t, ov[0] == 1.0 && ov[1] == 4.0, "partition a running sum")
	testing.expect(t, ov[2] == 10.0 && ov[3] == 30.0 && ov[4] == 60.0, "partition b running sum")
}

@(test)
window_eval_cum_min_max_partitioned :: proc(t: ^testing.T) {
	df, ctx := window_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	over := []^expr.Expr{expr.col(&ctx, "grp")}
	cmn := eval_ok(t, &df, expr.window_cum_min(&ctx, expr.col(&ctx, "v"), over))
	defer column_destroy(&cmn)
	on := column_typed_view(&cmn, f64)
	testing.expect(t, on[0] == 1 && on[1] == 1 && on[2] == 10 && on[3] == 10 && on[4] == 10, "cum_min")

	cmx := eval_ok(t, &df, expr.window_cum_max(&ctx, expr.col(&ctx, "v"), over))
	defer column_destroy(&cmx)
	ox := column_typed_view(&cmx, f64)
	testing.expect(t, ox[0] == 1 && ox[1] == 3 && ox[2] == 10 && ox[3] == 20 && ox[4] == 30, "cum_max")
}

@(test)
window_eval_cum_sum_null :: proc(t: ^testing.T) {
	// single-column frame with a NULL in the middle: [1.0, NULL, 10.0]
	v: Column
	err: Error
	v, err = column_from("v", []f64{1.0, 2.0, 10.0})
	testing.expect(t, err == .None, "v column")
	testing.expect(t, column_set_valid(&v, 1, false) == .None, "v[1] NULL")
	df, dferr := dataframe_from_columns([]^Column{&v})
	testing.expect(t, dferr == .None, "from_columns")
	ctx := expr.context_create(context.allocator)
	defer expr_test_destroy(t, &df, &ctx)

	out, oerr := expr_eval(context.allocator, &df, expr.window_cum_sum(&ctx, expr.col(&ctx, "v"), nil))
	testing.expect(t, oerr == .None, "cum_sum over whole frame")
	defer column_destroy(&out)

	ov := column_typed_view(&out, f64)
	testing.expect(t, ov[0] == 1.0, "row 0")
	testing.expect(t, !column_is_valid(&out, 1), "NULL in -> NULL out")
	testing.expect(t, ov[2] == 11.0, "running sum continues past NULL")
}

@(test)
window_eval_shift :: proc(t: ^testing.T) {
	df, ctx := window_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	over := []^expr.Expr{expr.col(&ctx, "grp")}
	out := eval_ok(t, &df, expr.window_shift(&ctx, expr.col(&ctx, "v"), 1, over))
	defer column_destroy(&out)

	ov := column_typed_view(&out, f64)
	testing.expect(t, !column_is_valid(&out, 0), "partition head -> NULL")
	testing.expect(t, ov[1] == 1.0, "lag 1 within partition a")
	testing.expect(t, !column_is_valid(&out, 2), "partition b head -> NULL")
	testing.expect(t, ov[3] == 10.0 && ov[4] == 20.0, "lag within partition b")
}

@(test)
window_eval_shift_negative :: proc(t: ^testing.T) {
	df, ctx := window_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	over := []^expr.Expr{expr.col(&ctx, "grp")}
	out := eval_ok(t, &df, expr.window_shift(&ctx, expr.col(&ctx, "v"), -1, over))
	defer column_destroy(&out)

	ov := column_typed_view(&out, f64)
	testing.expect(t, ov[0] == 3.0, "lead within partition a")
	testing.expect(t, !column_is_valid(&out, 1), "partition a tail -> NULL")
	testing.expect(t, ov[2] == 20.0 && ov[3] == 30.0, "lead within partition b")
	testing.expect(t, !column_is_valid(&out, 4), "partition b tail -> NULL")
}

@(test)
window_eval_rolling_mean :: proc(t: ^testing.T) {
	df, ctx := window_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	over := []^expr.Expr{expr.col(&ctx, "grp")}
	out := eval_ok(t, &df, expr.window_rolling(&ctx, expr.col(&ctx, "v"), 2, .Mean, over))
	defer column_destroy(&out)

	ov := column_typed_view(&out, f64)
	testing.expect(t, ov[0] == 1.0, "window clipped at partition head")
	testing.expect(t, ov[1] == 2.0, "mean(1,3)")
	testing.expect(t, ov[2] == 10.0, "partition b head")
	testing.expect(t, ov[3] == 15.0, "mean(10,20)")
	testing.expect(t, ov[4] == 25.0, "mean(20,30)")
}

@(test)
window_eval_cumulative_count :: proc(t: ^testing.T) {
	df, ctx := window_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	over := []^expr.Expr{expr.col(&ctx, "grp")}
	out := eval_ok(t, &df, expr.window_cumulative_eval(&ctx, expr.col(&ctx, "v"), .Count, over))
	defer column_destroy(&out)

	ov := column_typed_view(&out, i64)
	testing.expect(t, ov[0] == 1 && ov[1] == 2, "partition a running count")
	testing.expect(t, ov[2] == 1 && ov[3] == 2 && ov[4] == 3, "partition b running count")
}

@(test)
window_eval_ewma :: proc(t: ^testing.T) {
	df, ctx := window_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	over := []^expr.Expr{expr.col(&ctx, "grp")}
	out := eval_ok(t, &df, expr.window_ewma(&ctx, expr.col(&ctx, "v"), 0.5, over))
	defer column_destroy(&out)

	ov := column_typed_view(&out, f64)
	testing.expect(t, ov[0] == 1.0, "seed = first value")
	testing.expect(t, math.abs(ov[1] - 2.0) < 1e-9, "0.5*3 + 0.5*1")
	testing.expect(t, ov[2] == 10.0, "partition b seed")
	testing.expect(t, math.abs(ov[3] - 15.0) < 1e-9, "0.5*20 + 0.5*10")
	testing.expect(t, math.abs(ov[4] - 22.5) < 1e-9, "0.5*30 + 0.5*15")
}

@(test)
window_eval_rolling_string_min :: proc(t: ^testing.T) {
	df, ctx := window_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	over := []^expr.Expr{expr.col(&ctx, "grp")}
	// rolling min over the string column picks the lexicographically smallest
	out := eval_ok(t, &df, expr.window_rolling(&ctx, expr.col(&ctx, "grp"), 2, .Min, over))
	defer column_destroy(&out)

	ov := column_typed_view(&out, string)
	testing.expect(t, ov[0] == "a", "window [a]")
	testing.expect(t, ov[1] == "a", "window [a,a]")
	testing.expect(t, ov[2] == "b", "window [b]")
	testing.expect(t, ov[3] == "b", "window [b,b]")
	testing.expect(t, ov[4] == "b", "window [b,b]")
}

@(test)
window_eval_empty_frame :: proc(t: ^testing.T) {
	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	df: DataFrame

	out := eval_ok(t, &df, expr.window_row_number(&ctx, nil))
	defer column_destroy(&out)
	testing.expect(t, out.count == 0, "row_number on empty frame")
}

@(test)
window_eval_errors :: proc(t: ^testing.T) {
	df, ctx := window_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	over := []^expr.Expr{expr.col(&ctx, "grp")}

	_, err := expr_eval(context.allocator, &df, expr.window_rolling(&ctx, expr.col(&ctx, "v"), 0, .Mean, over))
	testing.expect(t, err == .Invalid_Argument, "rolling n=0 rejected")

	_, err = expr_eval(context.allocator, &df, expr.window_rolling(&ctx, expr.col(&ctx, "v"), 2, .Quantile, over))
	testing.expect(t, err == .Invalid_Argument, "rolling quantile rejected (no q)")

	_, err = expr_eval(context.allocator, &df, expr.window_cum_sum(&ctx, expr.col(&ctx, "grp"), over))
	testing.expect(t, err == .Unsupported_Operation, "cum_sum over string rejected")

	_, err = expr_eval(context.allocator, &df, expr.window_ewma(&ctx, expr.col(&ctx, "grp"), 0.5, over))
	testing.expect(t, err == .Unsupported_Operation, "ewma over string rejected")

	rnk, rerr := expr_eval(context.allocator, &df, expr.window_rank(&ctx, expr.col(&ctx, "grp"), .Min, over))
	testing.expect(t, rerr == .None, "rank over string allowed (sortable)")
	if rerr == .None {
		column_destroy(&rnk)
	}
}

@(test)
window_eval_typecheck :: proc(t: ^testing.T) {
	df, ctx := window_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	over := []^expr.Expr{expr.col(&ctx, "grp")}

	rn, rerr := expr_typecheck(&df, expr.window_row_number(&ctx, nil))
	testing.expect(t, rerr == .None && rn == typeid_of(i64), "row_number type")

	rs, rserr := expr_typecheck(&df, expr.window_cum_sum(&ctx, expr.col(&ctx, "v"), over))
	testing.expect(t, rserr == .None && rs == typeid_of(f64), "cum_sum type")

	re, reerr := expr_typecheck(&df, expr.window_ewma(&ctx, expr.col(&ctx, "v"), 0.5, over))
	testing.expect(t, reerr == .None && re == typeid_of(f64), "ewma type")

	_, cerr := expr_typecheck(&df, expr.window_cum_sum(&ctx, expr.col(&ctx, "grp"), over))
	testing.expect(t, cerr == .Unsupported_Operation, "cum_sum over string type error")
}
