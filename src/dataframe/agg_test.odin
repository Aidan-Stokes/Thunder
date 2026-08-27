package dataframe

// Stage 6 aggregation tests (DESIGN.md §6.6, ROADMAP S6.5): NULL-only column,
// single row, mixed NULL, type matrix, expr path and per-column scalar path.

import "core:testing"
import "core:math"
import "expr"

// near compares floats with an absolute epsilon.
near :: proc(a, b: f64, eps := f64(1e-9)) -> bool {
	return math.abs(a - b) <= eps
}

// agg_test_df builds the shared Stage 6 fixture:
//
//	x      (i32)   [10, NULL, 30]
//	y      (f64)   [1.5, 2.5, NULL]
//	s      (string) ["b", "a", "b"]
//	flag   (bool)  [true, false, true]
//	nulls  (f64)   [NULL, NULL, NULL]
agg_test_df :: proc(t: ^testing.T) -> (df: DataFrame, ctx: expr.Context) {
	err: Error
	x, y, s, flag, nulls: Column
	x, err = column_from("x", []i32{10, 0, 30})
	testing.expect(t, err == .None, "x column")
	testing.expect(t, column_set_valid(&x, 1, false) == .None, "x[1] NULL")

	y, err = column_from("y", []f64{1.5, 2.5, 0})
	testing.expect(t, err == .None, "y column")
	testing.expect(t, column_set_valid(&y, 2, false) == .None, "y[2] NULL")

	s, err = column_from("s", []string{"b", "a", "b"})
	testing.expect(t, err == .None, "s column")

	flag, err = column_from("flag", []bool{true, false, true})
	testing.expect(t, err == .None, "flag column")

	nulls, err = column_from("nulls", []f64{1, 2, 3})
	testing.expect(t, err == .None, "nulls column")
	for i in 0 ..< 3 {
		testing.expect(t, column_set_valid(&nulls, i, false) == .None, "nulls all NULL")
	}

	df, err = dataframe_from_columns([]^Column{&x, &y, &s, &flag, &nulls})
	testing.expect(t, err == .None, "from_columns")
	ctx = expr.context_create(context.allocator)
	return
}

agg_test_destroy :: proc(t: ^testing.T, df: ^DataFrame, ctx: ^expr.Context) {
	expr.context_destroy(ctx)
	dataframe_destroy(df)
}

// --- per-column scalar API ---------------------------------------------------

@(test)
agg_test_count_mixed_null :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	cnt, err := dataframe_count(&df, "x")
	testing.expect(t, err == .None, "count err")
	testing.expect(t, cnt == 2, "count skips NULL")
}

@(test)
agg_test_count_null_only :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	cnt, err := dataframe_count(&df, "nulls")
	testing.expect(t, err == .None, "count err")
	testing.expect(t, cnt == 0, "count of NULL-only is 0")
}

@(test)
agg_test_count_any_dtype :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	cnt, err := dataframe_count(&df, "s")
	testing.expect(t, err == .None, "count err")
	testing.expect(t, cnt == 3, "count works on string columns")

	Point :: struct { a: i32, b: f64 }
	p, perr := column_from("p", []Point{{1, 2}, {3, 4}})
	testing.expect(t, perr == .None, "struct column")
	df2, d2err := dataframe_from_columns([]^Column{&p})
	testing.expect(t, d2err == .None, "from_columns struct")
	defer dataframe_destroy(&df2)
	cnt, err = dataframe_count(&df2, "p")
	testing.expect(t, err == .None, "count err")
	testing.expect(t, cnt == 2, "count works on struct columns")
}

@(test)
agg_test_sum_mean_skip_null :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	sum, err := dataframe_sum(&df, "x")
	testing.expect(t, err == .None, "sum err")
	testing.expect(t, near(sum, 40), "sum(x) = 40")

	mean: f64
	mean, err = dataframe_mean(&df, "y")
	testing.expect(t, err == .None, "mean err")
	testing.expect(t, near(mean, 2.0), "mean(y) = 2")
}

@(test)
agg_test_var_std :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	v, err := dataframe_var(&df, "x")
	testing.expect(t, err == .None, "var err")
	testing.expect(t, near(v, 200), "sample var(x) = 200")

	std: f64
	std, err = dataframe_std(&df, "x")
	testing.expect(t, err == .None, "std err")
	testing.expect(t, near(std, math.sqrt(f64(200))), "std(x) = sqrt(200)")
}

@(test)
agg_test_median_quantile :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	// odd count: median of [10, 30] is 20
	med, err := dataframe_median(&df, "x")
	testing.expect(t, err == .None, "median err")
	testing.expect(t, near(med, 20), "median(x) = 20")

	// even count: linear interp q=0.5 -> (a+b)/2
	col, cerr := column_from("e", []f64{10, 20, 30, 40})
	testing.expect(t, cerr == .None, "e column")
	df2, d2err := dataframe_from_columns([]^Column{&col})
	testing.expect(t, d2err == .None, "from_columns")
	defer dataframe_destroy(&df2)
	med, err = dataframe_median(&df2, "e")
	testing.expect(t, err == .None, "median err")
	testing.expect(t, near(med, 25), "median([10,20,30,40]) = 25")

	q, qerr := dataframe_quantile(&df2, "e", 0.25)
	testing.expect(t, qerr == .None, "quantile err")
	testing.expect(t, near(q, 17.5), "q0.25 = 17.5")
	q, qerr = dataframe_quantile(&df2, "e", 0.0)
	testing.expect(t, qerr == .None, "quantile err")
	testing.expect(t, near(q, 10), "q0 = 10")
	q, qerr = dataframe_quantile(&df2, "e", 1.0)
	testing.expect(t, qerr == .None, "quantile err")
	testing.expect(t, near(q, 40), "q1 = 40")
}

@(test)
agg_test_quantile_bad_q :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	_, err := dataframe_quantile(&df, "x", 1.5)
	testing.expect(t, err == .Invalid_Argument, "q > 1 rejected")
	_, err = dataframe_quantile(&df, "x", -0.1)
	testing.expect(t, err == .Invalid_Argument, "q < 0 rejected")
}

@(test)
agg_test_min_max :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	mn, mx, mns, mxs: Scalar
	err: Error
	mn, err = dataframe_min(&df, "x")
	testing.expect(t, err == .None, "min err")
	testing.expect(t, mn == i32(10), "min(x) = 10")
	mx, err = dataframe_max(&df, "x")
	testing.expect(t, err == .None, "max err")
	testing.expect(t, mx == i32(30), "max(x) = 30")

	mns, err = dataframe_min(&df, "s")
	testing.expect(t, err == .None, "min string err")
	testing.expect(t, mns == "a", "min(s) = a")
	mxs, err = dataframe_max(&df, "s")
	testing.expect(t, err == .None, "max string err")
	testing.expect(t, mxs == "b", "max(s) = b")
}

@(test)
agg_test_n_unique :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	n, err := dataframe_n_unique(&df, "s")
	testing.expect(t, err == .None, "n_unique err")
	testing.expect(t, n == 2, "n_unique(s) = 2")
	n, err = dataframe_n_unique(&df, "x")
	testing.expect(t, err == .None, "n_unique err")
	testing.expect(t, n == 2, "n_unique(x) = 2")
	n, err = dataframe_n_unique(&df, "nulls")
	testing.expect(t, err == .None, "n_unique err")
	testing.expect(t, n == 0, "n_unique(NULL-only) = 0")
}

@(test)
agg_test_mode :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	// "b" appears twice, "a" once.
	m, err := dataframe_mode(&df, "s")
	testing.expect(t, err == .None, "mode err")
	testing.expect(t, m == "b", "mode(s) = b")

	// [1, 2, 2, 1]: both appear twice; ties resolve to the first value in
	// column order among maximal-frequency values -> 1.
	col, cerr := column_from("m", []i32{1, 2, 2, 1})
	testing.expect(t, cerr == .None, "m column")
	df2, d2err := dataframe_from_columns([]^Column{&col})
	testing.expect(t, d2err == .None, "from_columns")
	defer dataframe_destroy(&df2)
	m, err = dataframe_mode(&df2, "m")
	testing.expect(t, err == .None, "mode err")
	testing.expect(t, m == i32(1), "mode([1,2,2,1]) = 1")

	m, err = dataframe_mode(&df, "flag")
	testing.expect(t, err == .None, "mode bool err")
	testing.expect(t, m == true, "mode(flag) = true")
}

@(test)
agg_test_product :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	col, cerr := column_from("p", []i32{2, 3, 4})
	testing.expect(t, cerr == .None, "p column")
	df2, d2err := dataframe_from_columns([]^Column{&col})
	testing.expect(t, d2err == .None, "from_columns")
	defer dataframe_destroy(&df2)
	prod, err := dataframe_product(&df2, "p")
	testing.expect(t, err == .None, "product err")
	testing.expect(t, near(prod, 24), "product([2,3,4]) = 24")
}

@(test)
agg_test_first_last :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	f, err := dataframe_first(&df, "x")
	testing.expect(t, err == .None, "first err")
	testing.expect(t, f == i32(10), "first(x) = 10")
	l: Scalar
	l, err = dataframe_last(&df, "x")
	testing.expect(t, err == .None, "last err")
	testing.expect(t, l == i32(30), "last(x) = 30")
}

@(test)
agg_test_skew_kurtosis :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	col, cerr := column_from("v", []f64{1, 2, 3, 4, 5})
	testing.expect(t, cerr == .None, "v column")
	df2, d2err := dataframe_from_columns([]^Column{&col})
	testing.expect(t, d2err == .None, "from_columns")
	defer dataframe_destroy(&df2)

	sk, err := dataframe_skew(&df2, "v")
	testing.expect(t, err == .None, "skew err")
	testing.expect(t, near(sk, 0, 1e-12), "skew symmetric = 0")
	ku: f64
	ku, err = dataframe_kurtosis(&df2, "v")
	testing.expect(t, err == .None, "kurtosis err")
	testing.expect(t, near(ku, -1.2, 1e-9), "kurtosis([1..5]) = -1.2")

	// only 2 valid values -> skew undefined -> NaN
	sk, err = dataframe_skew(&df, "x")
	testing.expect(t, err == .None, "skew err")
	testing.expect(t, math.is_nan(sk), "skew of 2 values is NaN")

	// constant data -> m2 = 0 -> NaN
	cc, ccerr := column_from("c", []f64{5, 5, 5})
	testing.expect(t, ccerr == .None, "c column")
	df3, d3err := dataframe_from_columns([]^Column{&cc})
	testing.expect(t, d3err == .None, "from_columns")
	defer dataframe_destroy(&df3)
	ku, err = dataframe_kurtosis(&df3, "c")
	testing.expect(t, err == .None, "kurtosis err")
	testing.expect(t, math.is_nan(ku), "kurtosis of constant data is NaN")
}

@(test)
agg_test_cov_corr :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	x, xerr := column_from("a", []i32{1, 2, 3, 4, 5})
	testing.expect(t, xerr == .None, "a column")
	y, yerr := column_from("b", []i32{2, 4, 6, 8, 10})
	testing.expect(t, yerr == .None, "b column")
	df2, d2err := dataframe_from_columns([]^Column{&x, &y})
	testing.expect(t, d2err == .None, "from_columns")
	defer dataframe_destroy(&df2)

	cv, err := dataframe_cov(&df2, "a", "b")
	testing.expect(t, err == .None, "cov err")
	testing.expect(t, near(cv, 5), "cov(a,b) = 5")
	cr: f64
	cr, err = dataframe_corr(&df2, "a", "b")
	testing.expect(t, err == .None, "corr err")
	testing.expect(t, near(cr, 1), "corr linear = 1")

	// anti-correlated
	z, zerr := column_from("c", []i32{5, 4, 3, 2, 1})
	testing.expect(t, zerr == .None, "c column")
	testing.expect(t, dataframe_add_column(&df2, &z) == .None, "add c")
	cr, err = dataframe_corr(&df2, "a", "c")
	testing.expect(t, err == .None, "corr err")
	testing.expect(t, near(cr, -1), "corr anticorrelated = -1")

	// constant column -> NaN
	k, kerr := column_from("k", []i32{3, 3, 3, 3, 3})
	testing.expect(t, kerr == .None, "k column")
	testing.expect(t, dataframe_add_column(&df2, &k) == .None, "add k")
	cr, err = dataframe_corr(&df2, "a", "k")
	testing.expect(t, err == .None, "corr err")
	testing.expect(t, math.is_nan(cr), "corr constant = NaN")
}

@(test)
agg_test_cov_corr_skip_null :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	// x = [1, NULL, 3, 4, 5], y = [2, 4, 6, 8, 10]; pairwise-valid points
	// (1,2) (3,6) (4,8) (5,10) are perfectly linear -> corr 1, cov 17.5/3.
	x, xerr := column_from("a", []i32{1, 0, 3, 4, 5})
	testing.expect(t, xerr == .None, "a column")
	testing.expect(t, column_set_valid(&x, 1, false) == .None, "a[1] NULL")
	y, yerr := column_from("b", []i32{2, 4, 6, 8, 10})
	testing.expect(t, yerr == .None, "b column")
	df2, d2err := dataframe_from_columns([]^Column{&x, &y})
	testing.expect(t, d2err == .None, "from_columns")
	defer dataframe_destroy(&df2)

	cr, err := dataframe_corr(&df2, "a", "b")
	testing.expect(t, err == .None, "corr err")
	testing.expect(t, near(cr, 1, 1e-12), "corr pairwise-valid = 1")
	cv: f64
	cv, err = dataframe_cov(&df2, "a", "b")
	testing.expect(t, err == .None, "cov err")
	testing.expect(t, near(cv, 17.5 / 3, 1e-9), "cov = 17.5/3")
}

// --- NULL-only and single-row behavior --------------------------------------

@(test)
agg_test_null_only_column :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	_, err := dataframe_sum(&df, "nulls")
	testing.expect(t, err == .Null_Value, "sum(NULL-only) -> Null_Value")
	_, err = dataframe_mean(&df, "nulls")
	testing.expect(t, err == .Null_Value, "mean(NULL-only) -> Null_Value")
	_, err = dataframe_min(&df, "nulls")
	testing.expect(t, err == .Null_Value, "min(NULL-only) -> Null_Value")
	_, err = dataframe_mode(&df, "nulls")
	testing.expect(t, err == .Null_Value, "mode(NULL-only) -> Null_Value")
	_, err = dataframe_first(&df, "nulls")
	testing.expect(t, err == .Null_Value, "first(NULL-only) -> Null_Value")
}

@(test)
agg_test_single_row :: proc(t: ^testing.T) {
	col, cerr := column_from("one", []f64{7})
	testing.expect(t, cerr == .None, "one column")
	df, d2err := dataframe_from_columns([]^Column{&col})
	testing.expect(t, d2err == .None, "from_columns")
	defer dataframe_destroy(&df)

	mean, err := dataframe_mean(&df, "one")
	testing.expect(t, err == .None, "mean err")
	testing.expect(t, near(mean, 7), "mean single = 7")
	cnt: i64
	cnt, err = dataframe_count(&df, "one")
	testing.expect(t, err == .None, "count err")
	testing.expect(t, cnt == 1, "count single = 1")
	// sample variance of one value is undefined -> NaN
	v: f64
	v, err = dataframe_var(&df, "one")
	testing.expect(t, err == .None, "var err")
	testing.expect(t, math.is_nan(v), "var of single row is NaN")
}

// --- expression path ---------------------------------------------------------

@(test)
agg_expr_single_row_result :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.sum_(&ctx, expr.col(&ctx, "x")))
	defer column_destroy(&out)
	testing.expect(t, out.dtype == typeid_of(f64), "sum result dtype")
	testing.expect(t, out.count == 1, "agg result is single row")
	v, valid, err := column_get(&out, 0, f64)
	testing.expect(t, err == .None, "sum get err")
	testing.expect(t, valid, "sum row valid")
	testing.expect(t, near(v, 40), "sum(x) = 40")

	out2 := eval_ok(t, &df, expr.count_(&ctx, expr.col(&ctx, "x")))
	defer column_destroy(&out2)
	c, cvalid, cerr := column_get(&out2, 0, i64)
	testing.expect(t, cerr == .None, "count get err")
	testing.expect(t, cvalid, "count row valid")
	testing.expect(t, c == 2, "count(x) = 2")

	// aggregation over an expression, not just a column reference
	out3 := eval_ok(t, &df, expr.mean_(&ctx, expr.mul(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(2)))))
	defer column_destroy(&out3)
	m, mvalid, merr := column_get(&out3, 0, f64)
	testing.expect(t, merr == .None, "mean get err")
	testing.expect(t, mvalid, "mean(x*2) valid")
	testing.expect(t, near(m, 40), "mean(x*2) = 40")
}

@(test)
agg_expr_null_only_result :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.mean_(&ctx, expr.col(&ctx, "nulls")))
	defer column_destroy(&out)
	testing.expect(t, out.count == 1, "single row")
	testing.expect(t, !column_is_valid(&out, 0), "mean(NULL-only) is NULL row")

	out2 := eval_ok(t, &df, expr.count_(&ctx, expr.col(&ctx, "nulls")))
	defer column_destroy(&out2)
	c, cvalid, cerr := column_get(&out2, 0, i64)
	testing.expect(t, cerr == .None, "count get err")
	testing.expect(t, cvalid, "count row valid")
	testing.expect(t, c == 0, "count(NULL-only) = 0")
}

@(test)
agg_expr_cov_corr :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	x, xerr := column_from("a", []i32{1, 2, 3, 4, 5})
	testing.expect(t, xerr == .None, "a column")
	y, yerr := column_from("b", []i32{2, 4, 6, 8, 10})
	testing.expect(t, yerr == .None, "b column")
	df2, d2err := dataframe_from_columns([]^Column{&x, &y})
	testing.expect(t, d2err == .None, "from_columns")
	defer dataframe_destroy(&df2)

	out := eval_ok(t, &df2, expr.corr_(&ctx, expr.col(&ctx, "a"), expr.col(&ctx, "b")))
	defer column_destroy(&out)
	v, valid, err := column_get(&out, 0, f64)
	testing.expect(t, err == .None, "corr get err")
	testing.expect(t, valid, "corr row valid")
	testing.expect(t, near(v, 1), "corr = 1")
}

// --- type matrix (S6.4) ------------------------------------------------------

@(test)
agg_test_type_matrix_rejects :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	// numeric aggs over non-numeric -> Unsupported_Operation
	_, err := dataframe_sum(&df, "s")
	testing.expect(t, err == .Unsupported_Operation, "sum(string) rejected")
	_, err = dataframe_var(&df, "flag")
	testing.expect(t, err == .Unsupported_Operation, "var(bool) rejected")

	_, err = dataframe_cov(&df, "x", "s")
	testing.expect(t, err == .Unsupported_Operation, "cov(string) rejected")
	_, err = dataframe_corr(&df, "s", "x")
	testing.expect(t, err == .Unsupported_Operation, "corr(string) rejected")

	// min/max are ordering ops: bool has no order in this library.
	_, err = dataframe_min(&df, "flag")
	testing.expect(t, err == .Unsupported_Operation, "min(bool) rejected")

	// typecheck rejects the same cases early.
	_, terr := expr_typecheck(&df, expr.sum_(&ctx, expr.col(&ctx, "s")))
	testing.expect(t, terr == .Unsupported_Operation, "typecheck sum(string)")
	_, terr = expr_typecheck(&df, expr.median_(&ctx, expr.col(&ctx, "flag")))
	testing.expect(t, terr == .Unsupported_Operation, "typecheck median(bool)")
	_, terr = expr_typecheck(&df, expr.quantile_(&ctx, expr.col(&ctx, "x"), 2.0))
	testing.expect(t, terr == .Invalid_Argument, "typecheck quantile(q>1)")
}

@(test)
agg_test_typecheck_result_types :: proc(t: ^testing.T) {
	df, ctx := agg_test_df(t)
	defer agg_test_destroy(t, &df, &ctx)

	tp, terr := expr_typecheck(&df, expr.sum_(&ctx, expr.col(&ctx, "x")))
	testing.expect(t, terr == .None, "sum typecheck")
	testing.expect(t, tp == typeid_of(f64), "sum -> f64")
	tp, terr = expr_typecheck(&df, expr.count_(&ctx, expr.col(&ctx, "x")))
	testing.expect(t, terr == .None, "count typecheck")
	testing.expect(t, tp == typeid_of(i64), "count -> i64")
	tp, terr = expr_typecheck(&df, expr.min_(&ctx, expr.col(&ctx, "x")))
	testing.expect(t, terr == .None, "min typecheck")
	testing.expect(t, tp == typeid_of(i32), "min preserves i32")
	tp, terr = expr_typecheck(&df, expr.mode_(&ctx, expr.col(&ctx, "s")))
	testing.expect(t, terr == .None, "mode typecheck")
	testing.expect(t, tp == typeid_of(string), "mode preserves string")
	tp, terr = expr_typecheck(&df, expr.cov_(&ctx, expr.col(&ctx, "x"), expr.col(&ctx, "y")))
	testing.expect(t, terr == .None, "cov typecheck")
	testing.expect(t, tp == typeid_of(f64), "cov -> f64")
}
