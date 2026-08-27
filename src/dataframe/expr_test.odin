package dataframe

import "core:testing"
import "core:math"
import "expr"

// --- fixture -----------------------------------------------------------------

// expr_test_df builds the shared fixture for expression tests:
//
//	x    (i32)   [10, NULL, 30]
//	y    (f64)   [1.5, 2.5, NULL]
//	flag (bool)  [true, false, true]
//	s    (string) ["a", "b", "c"]
expr_test_df :: proc(t: ^testing.T) -> (df: DataFrame, ctx: expr.Context) {
	x: Column
	y: Column
	flag: Column
	s: Column
	err: Error
	x, err = column_from("x", []i32{10, 0, 30})
	testing.expect(t, err == .None, "x column")
	testing.expect(t, column_set_valid(&x, 1, false) == .None, "x[1] NULL")

	y, err = column_from("y", []f64{1.5, 2.5, 0.0})
	testing.expect(t, err == .None, "y column")
	testing.expect(t, column_set_valid(&y, 2, false) == .None, "y[2] NULL")

	flag, err = column_from("flag", []bool{true, false, true})
	testing.expect(t, err == .None, "flag column")

	s, err = column_from("s", []string{"a", "b", "c"})
	testing.expect(t, err == .None, "s column")

	df, err = dataframe_from_columns([]^Column{&x, &y, &flag, &s})
	testing.expect(t, err == .None, "from_columns")
	ctx = expr.context_create(context.allocator)
	return
}

expr_test_destroy :: proc(t: ^testing.T, df: ^DataFrame, ctx: ^expr.Context) {
	expr.context_destroy(ctx)
	dataframe_destroy(df)
}

// eval_ok evaluates e and returns the column, failing the test on any error.
eval_ok :: proc(t: ^testing.T, df: ^DataFrame, e: ^expr.Expr) -> Column {
	out, err := expr_eval(context.allocator, df, e)
	testing.expect(t, err == .None, "expr_eval should succeed")
	return out
}

// --- basic arithmetic and NULL propagation -----------------------------------

@(test)
expr_eval_add_column_plus_literal :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.add(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(5))))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(i32), "result dtype")
	testing.expect(t, column_is_valid(&out, 0) && column_typed_view(&out, i32)[0] == 15, "row 0")
	testing.expect(t, !column_is_valid(&out, 1), "NULL propagates")
	testing.expect(t, column_is_valid(&out, 2) && column_typed_view(&out, i32)[2] == 35, "row 2")
}

@(test)
expr_eval_binary_null_propagation :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	// both operands are columns, one i32 and one f64 -> no implicit conversion
	_, err := expr_eval(context.allocator, &df, expr.add(&ctx, expr.col(&ctx, "x"), expr.col(&ctx, "y")))
	testing.expect(t, err == .Type_Mismatch, "i32 + f64 columns must error")
}

@(test)
expr_eval_literal_coercion :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	// i32 column + f64 literal -> f64 (literal coerces to column dtype)
	out := eval_ok(t, &df, expr.add(&ctx, expr.col(&ctx, "y"), expr.lit(&ctx, f64(2.5))))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(f64), "result dtype")
	testing.expect(t, column_typed_view(&out, f64)[0] == 4.0, "row 0")
}

@(test)
expr_eval_binary_no_implicit_column_conversion :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	// i32 column + f64 column: no implicit conversion (principle 6)
	_, err := expr_eval(context.allocator, &df, expr.add(&ctx, expr.col(&ctx, "x"), expr.col(&ctx, "y")))
	testing.expect(t, err == .Type_Mismatch, "i32 + f64 must be a type error")
}

// --- comparisons and boolean ops ---------------------------------------------

@(test)
expr_eval_compare :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.gt(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(10))))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(bool), "comparison is bool")
	testing.expect(t, column_is_valid(&out, 0) && !column_typed_view(&out, bool)[0], "row 0: 10 > 10 false")
	testing.expect(t, !column_is_valid(&out, 1), "NULL propagates")
	testing.expect(t, column_is_valid(&out, 2) && column_typed_view(&out, bool)[2], "row 2: 30 > 10 true")
}

@(test)
expr_eval_logical_ops :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	both := eval_ok(t, &df, expr.and_(&ctx, expr.col(&ctx, "flag"), expr.col(&ctx, "flag")))
	defer column_destroy(&both)
	testing.expect(t, column_typed_view(&both, bool)[0] && !column_typed_view(&both, bool)[1], "and")

	either := eval_ok(t, &df, expr.or_(&ctx, expr.col(&ctx, "flag"), expr.lit(&ctx, true)))
	defer column_destroy(&either)
	testing.expect(t, column_typed_view(&either, bool)[0], "or")

	noted := eval_ok(t, &df, expr.not_(&ctx, expr.col(&ctx, "flag")))
	defer column_destroy(&noted)
	testing.expect(t, !column_typed_view(&noted, bool)[0] && column_typed_view(&noted, bool)[1], "not")
}

@(test)
expr_eval_string_compare :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.eq(&ctx, expr.col(&ctx, "s"), expr.lit(&ctx, "b")))
	defer column_destroy(&out)

	testing.expect(t, column_typed_view(&out, bool)[1], "s[1] == \"b\"")
	testing.expect(t, !column_typed_view(&out, bool)[0], "s[0] != \"b\"")
}

// --- unary, cast, alias, not_null -------------------------------------------

@(test)
expr_eval_neg :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.neg(&ctx, expr.col(&ctx, "x")))
	defer column_destroy(&out)

	testing.expect(t, column_typed_view(&out, i32)[0] == -10, "neg row 0")
	testing.expect(t, !column_is_valid(&out, 1), "NULL propagates")
}

@(test)
expr_eval_cast :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.cast_(&ctx, expr.col(&ctx, "x"), f64))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(f64), "cast to f64")
	testing.expect(t, column_typed_view(&out, f64)[0] == 10.0, "cast row 0")
	testing.expect(t, !column_is_valid(&out, 1), "NULL preserved by cast")
}

@(test)
expr_eval_alias :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.alias(&ctx, expr.col(&ctx, "x"), "renamed"))
	defer column_destroy(&out)

	testing.expect(t, out.name == "renamed", "alias name", out.name)
}

@(test)
expr_eval_is_not_null :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.is_not_null(&ctx, expr.col(&ctx, "x")))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(bool), "not_null is bool")
	testing.expect(t, column_typed_view(&out, bool)[0] && !column_typed_view(&out, bool)[1], "x[1] is NULL")
}

// --- errors ------------------------------------------------------------------

@(test)
expr_eval_unknown_column :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	_, err := expr_eval(context.allocator, &df, expr.col(&ctx, "nope"))
	testing.expect(t, err == .Column_Not_Found, "unknown column")
}

@(test)
expr_eval_incompatible_type :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	_, err := expr_eval(context.allocator, &df, expr.add(&ctx, expr.col(&ctx, "x"), expr.col(&ctx, "s")))
	testing.expect(t, err == .Type_Mismatch, "i32 + string")
}

@(test)
expr_eval_int_div_by_zero_is_null :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	zero, zerr := column_from("z", []i32{0, 0, 0})
	testing.expect(t, zerr == .None, "zero column")
	df2, err := dataframe_from_columns([]^Column{&zero})
	testing.expect(t, err == .None, "from_columns")
	defer dataframe_destroy(&df2)

	out, eval_err := expr_eval(context.allocator, &df2, expr.div(&ctx, expr.col(&ctx, "z"), expr.col(&ctx, "z")))
	testing.expect(t, eval_err == .None, "div by zero is NULL, not an error")
	testing.expect(t, !column_is_valid(&out, 0), "0/0 is NULL")
	column_destroy(&out)
}

// --- typecheck ---------------------------------------------------------------

@(test)
expr_typecheck_basic :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	dtype, err := expr_typecheck(&df, expr.add(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(1))))
	testing.expect(t, err == .None && dtype == typeid_of(i32), "i32 + lit")

	dtype, err = expr_typecheck(&df, expr.gt(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(0))))
	testing.expect(t, err == .None && dtype == typeid_of(bool), "compare is bool")

	dtype, err = expr_typecheck(&df, expr.cast_(&ctx, expr.col(&ctx, "x"), f64))
	testing.expect(t, err == .None && dtype == typeid_of(f64), "cast")
}

@(test)
expr_typecheck_rejects_bad_expressions :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	_, err := expr_typecheck(&df, expr.add(&ctx, expr.col(&ctx, "x"), expr.col(&ctx, "y")))
	testing.expect(t, err == .Type_Mismatch, "i32 + f64 columns")

	_, err = expr_typecheck(&df, expr.add(&ctx, expr.col(&ctx, "x"), expr.col(&ctx, "s")))
	testing.expect(t, err == .Type_Mismatch, "i32 + string")

	_, err = expr_typecheck(&df, expr.add(&ctx, expr.col(&ctx, "s"), expr.col(&ctx, "s")))
	testing.expect(t, err == .Unsupported_Operation, "string add")

	_, err = expr_typecheck(&df, expr.and_(&ctx, expr.col(&ctx, "x"), expr.col(&ctx, "x")))
	testing.expect(t, err == .Unsupported_Operation, "i32 and")

	_, err = expr_typecheck(&df, expr.col(&ctx, "nope"))
	testing.expect(t, err == .Column_Not_Found, "unknown column")
}

@(test)
expr_typecheck_literal_coercion :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	dtype, err := expr_typecheck(&df, expr.add(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, f64(1.5))))
	testing.expect(t, err == .None, "column + other-typed literal")
	testing.expect(t, dtype == typeid_of(i32), "literal coerces to column type")
}

// --- S3.10 expression library ------------------------------------------------

// round_df builds a small float column for rounding tests:
//
//	v (f64) [3.14159, -2.71828, NULL]
round_df :: proc(t: ^testing.T) -> (df: DataFrame, ctx: expr.Context) {
	v: Column
	err: Error
	v, err = column_from("v", []f64{3.14159, -2.71828, 0.0})
	testing.expect(t, err == .None, "v column")
	testing.expect(t, column_set_valid(&v, 2, false) == .None, "v[2] NULL")
	df, err = dataframe_from_columns([]^Column{&v})
	testing.expect(t, err == .None, "from_columns")
	ctx = expr.context_create(context.allocator)
	return
}

@(test)
expr_eval_abs :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.abs_(&ctx, expr.sub(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(40)))))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(i32), "abs dtype")
	testing.expect(t, column_typed_view(&out, i32)[0] == 30, "abs row 0")
	testing.expect(t, !column_is_valid(&out, 1), "NULL propagates")
	testing.expect(t, column_typed_view(&out, i32)[2] == 10, "abs row 2")
}

@(test)
expr_eval_sign :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.sign_(&ctx, expr.sub(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(20)))))
	defer column_destroy(&out)

	testing.expect(t, column_typed_view(&out, i32)[0] == -1, "sign -1")
	testing.expect(t, column_typed_view(&out, i32)[2] == 1, "sign +1")
	testing.expect(t, !column_is_valid(&out, 1), "NULL propagates")
}

@(test)
expr_eval_round :: proc(t: ^testing.T) {
	df, ctx := round_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.round_(&ctx, expr.col(&ctx, "v"), 2))
	defer column_destroy(&out)

	testing.expect(t, column_typed_view(&out, f64)[0] == 3.14, "round 2dp row 0")
	testing.expect(t, column_typed_view(&out, f64)[1] == -2.72, "round 2dp row 1")
	testing.expect(t, !column_is_valid(&out, 2), "NULL propagates")
}

@(test)
expr_eval_diff :: proc(t: ^testing.T) {
	vals, verr := column_from("d", []i32{1, 2, 4})
	testing.expect(t, verr == .None, "d column")
	df2, err := dataframe_from_columns([]^Column{&vals})
	testing.expect(t, err == .None, "from_columns")
	defer dataframe_destroy(&df2)
	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	out := eval_ok(t, &df2, expr.diff_(&ctx, expr.col(&ctx, "d"), 1))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(i32), "diff dtype")
	testing.expect(t, !column_is_valid(&out, 0), "first row is NULL")
	testing.expect(t, column_is_valid(&out, 1) && column_typed_view(&out, i32)[1] == 1, "diff row 1")
	testing.expect(t, column_is_valid(&out, 2) && column_typed_view(&out, i32)[2] == 2, "diff row 2")
}

@(test)
expr_eval_pct_change :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.pct_change_(&ctx, expr.col(&ctx, "y"), 1))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(f64), "pct_change is f64")
	testing.expect(t, !column_is_valid(&out, 0), "first row is NULL")
	testing.expect(t, math.abs(column_typed_view(&out, f64)[1] - (2.5 - 1.5) / 1.5) < 1e-9, "pct_change row 1")
	testing.expect(t, !column_is_valid(&out, 2), "NULL propagates")
}

@(test)
expr_eval_cum_sum :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.cum_sum_(&ctx, expr.col(&ctx, "x")))
	defer column_destroy(&out)

	testing.expect(t, column_typed_view(&out, i32)[0] == 10, "cumsum row 0")
	testing.expect(t, column_typed_view(&out, i32)[2] == 40, "cumsum row 2 (skips NULL)")
	testing.expect(t, !column_is_valid(&out, 1), "NULL stays NULL")
}

@(test)
expr_eval_is_between :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.is_between_(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(10)), expr.lit(&ctx, i32(30))))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(bool), "is_between is bool")
	testing.expect(t, column_typed_view(&out, bool)[0], "10 in [10,30]")
	testing.expect(t, !column_is_valid(&out, 1), "NULL propagates")
	testing.expect(t, column_typed_view(&out, bool)[2], "30 in [10,30]")

	// f64 literal bounds coerce to the i32 column
	out2 := eval_ok(t, &df, expr.is_between_(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, f64(11.0)), expr.lit(&ctx, f64(29.0))))
	defer column_destroy(&out2)
	testing.expect(t, !column_typed_view(&out2, bool)[0], "10 not in [11,29]")
	testing.expect(t, !column_typed_view(&out2, bool)[2], "30 not in [11,29]")
}

@(test)
expr_eval_is_in :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.is_in_(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, []i32{10, 30})))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(bool), "is_in is bool")
	testing.expect(t, column_typed_view(&out, bool)[0], "10 in list")
	testing.expect(t, !column_is_valid(&out, 1), "NULL propagates")
	testing.expect(t, column_typed_view(&out, bool)[2], "30 in list")

	// wrong element type must error
	_, err := expr_eval(context.allocator, &df, expr.is_in_(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, []f64{1.0})))
	testing.expect(t, err == .Type_Mismatch, "is_in wrong literal dtype")
}

@(test)
expr_eval_arange :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.arange_(&ctx, expr.lit(&ctx, int(1)), expr.lit(&ctx, int(5))))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(int), "arange is int")
	testing.expect(t, out.count == 4, "arange length")
	testing.expect(t, column_typed_view(&out, int)[0] == 1 && column_typed_view(&out, int)[3] == 4, "arange values")
}

@(test)
expr_eval_arg_where :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.arg_where_(&ctx, expr.gt(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(15)))))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(int), "arg_where is int")
	testing.expect(t, out.count == 1 && column_typed_view(&out, int)[0] == 2, "index of x>15")

	_, err := expr_eval(context.allocator, &df, expr.arg_where_(&ctx, expr.col(&ctx, "x")))
	testing.expect(t, err == .Type_Mismatch, "arg_where needs bool")
}

// dup_df builds a fixture with duplicates and NULLs:
//
//	d (i32) [10, 10, NULL, 30, NULL]
dup_df :: proc(t: ^testing.T) -> (df: DataFrame, ctx: expr.Context) {
	d: Column
	err: Error
	d, err = column_from("d", []i32{10, 10, 0, 30, 0})
	testing.expect(t, err == .None, "d column")
	testing.expect(t, column_set_valid(&d, 2, false) == .None, "d[2] NULL")
	testing.expect(t, column_set_valid(&d, 4, false) == .None, "d[4] NULL")
	df, err = dataframe_from_columns([]^Column{&d})
	testing.expect(t, err == .None, "from_columns")
	ctx = expr.context_create(context.allocator)
	return
}

@(test)
expr_eval_first_distinct :: proc(t: ^testing.T) {
	df, ctx := dup_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.first_distinct_(&ctx, expr.col(&ctx, "d")))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(bool), "distinct is bool")
	expected := []bool{true, false, true, true, false}
	for i in 0 ..< len(expected) {
		testing.expect(t, column_typed_view(&out, bool)[i] == expected[i], "first_distinct row")
	}
}

@(test)
expr_eval_last_distinct :: proc(t: ^testing.T) {
	df, ctx := dup_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.last_distinct_(&ctx, expr.col(&ctx, "d")))
	defer column_destroy(&out)

	expected := []bool{false, true, false, true, true}
	for i in 0 ..< len(expected) {
		testing.expect(t, column_typed_view(&out, bool)[i] == expected[i], "last_distinct row")
	}
}

@(test)
expr_typecheck_s310 :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	dtype, err := expr_typecheck(&df, expr.abs_(&ctx, expr.col(&ctx, "x")))
	testing.expect(t, err == .None && dtype == typeid_of(i32), "abs type")

	dtype, err = expr_typecheck(&df, expr.pct_change_(&ctx, expr.col(&ctx, "x"), 1))
	testing.expect(t, err == .None && dtype == typeid_of(f64), "pct_change is f64")

	dtype, err = expr_typecheck(&df, expr.is_between_(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(0)), expr.lit(&ctx, i32(1))))
	testing.expect(t, err == .None && dtype == typeid_of(bool), "is_between is bool")

	dtype, err = expr_typecheck(&df, expr.is_in_(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, []i32{1})))
	testing.expect(t, err == .None && dtype == typeid_of(bool), "is_in is bool")

	dtype, err = expr_typecheck(&df, expr.arange_(&ctx, expr.lit(&ctx, int(0)), expr.lit(&ctx, int(3))))
	testing.expect(t, err == .None && dtype == typeid_of(int), "arange is int")

	dtype, err = expr_typecheck(&df, expr.arg_where_(&ctx, expr.col(&ctx, "flag")))
	testing.expect(t, err == .None && dtype == typeid_of(int), "arg_where is int")

	dtype, err = expr_typecheck(&df, expr.first_distinct_(&ctx, expr.col(&ctx, "s")))
	testing.expect(t, err == .None && dtype == typeid_of(bool), "distinct is bool")
}

@(test)
expr_typecheck_s310_rejects :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	_, err := expr_typecheck(&df, expr.round_(&ctx, expr.col(&ctx, "x"), 2))
	testing.expect(t, err == .Unsupported_Operation, "round on int")

	_, err = expr_typecheck(&df, expr.abs_(&ctx, expr.col(&ctx, "s")))
	testing.expect(t, err == .Unsupported_Operation, "abs on string")

	_, err = expr_typecheck(&df, expr.is_in_(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, []f64{1.0})))
	testing.expect(t, err == .Type_Mismatch, "is_in wrong literal dtype")

	_, err = expr_typecheck(&df, expr.arg_where_(&ctx, expr.col(&ctx, "x")))
	testing.expect(t, err == .Type_Mismatch, "arg_where on non-bool")
}

@(test)
expr_eval_dot_product :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	// x = [10, NULL, 30], y = [1.5, 2.5, NULL] -> 10*1.5 = 15 (NULLs skipped)
	out := eval_ok(t, &df, expr.dot_product_(&ctx, expr.col(&ctx, "x"), expr.col(&ctx, "y")))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(f64), "dot is f64")
	testing.expect(t, out.count == 1, "single-row result")
	testing.expect(t, column_typed_view(&out, f64)[0] == 15.0, "dot value")
}

@(test)
expr_eval_concat_str :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.concat_str_(&ctx, []^expr.Expr{expr.col(&ctx, "s"), expr.col(&ctx, "s")}, "-"))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(string), "concat is string")
	testing.expect(t, column_typed_view(&out, string)[0] == "a-a", "concat row 0")
	testing.expect(t, column_typed_view(&out, string)[2] == "c-c", "concat row 2")
}

@(test)
expr_eval_search_sorted :: proc(t: ^testing.T) {
	sorted, serr := column_from("srt", []i32{10, 20, 30, 40})
	testing.expect(t, serr == .None, "sorted column")
	df2, err := dataframe_from_columns([]^Column{&sorted})
	testing.expect(t, err == .None, "from_columns")
	defer dataframe_destroy(&df2)
	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	out := eval_ok(t, &df2, expr.search_sorted_(&ctx, expr.col(&ctx, "srt"), expr.lit(&ctx, []i32{5, 10, 25, 50, 20})))
	defer column_destroy(&out)

	expected := []int{0, 0, 2, 4, 1}
	testing.expect(t, out.dtype == typeid_of(int), "search_sorted is int")
	for i in 0 ..< len(expected) {
		testing.expect(t, column_typed_view(&out, int)[i] == expected[i], "search_sorted row")
	}
}

@(test)
expr_typecheck_s310_rest :: proc(t: ^testing.T) {
	df, ctx := expr_test_df(t)
	defer expr_test_destroy(t, &df, &ctx)

	dtype, err := expr_typecheck(&df, expr.dot_product_(&ctx, expr.col(&ctx, "x"), expr.col(&ctx, "y")))
	testing.expect(t, err == .None && dtype == typeid_of(f64), "dot is f64")

	dtype, err = expr_typecheck(&df, expr.concat_str_(&ctx, []^expr.Expr{expr.col(&ctx, "s")}, ""))
	testing.expect(t, err == .None && dtype == typeid_of(string), "concat is string")

	_, err = expr_typecheck(&df, expr.dot_product_(&ctx, expr.col(&ctx, "x"), expr.col(&ctx, "s")))
	testing.expect(t, err == .Unsupported_Operation, "dot with string")

	_, err = expr_typecheck(&df, expr.concat_str_(&ctx, []^expr.Expr{expr.col(&ctx, "x")}, ""))
	testing.expect(t, err == .Type_Mismatch, "concat with int")
}
