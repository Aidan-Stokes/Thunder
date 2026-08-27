package dataframe

import "core:math"
import "core:testing"
import "expr"

// nulls_test_df builds the shared fixture for null-handling tests:
//
//	n (i32)    [10, NULL, 30, 40]
//	m (i32)    [NULL, 20, NULL, NULL]
//	o (i32)    [NULL, 5, NULL, NULL]
//	f (f64)    [1.5, NULL, 3.5, 4.5]
//	g (f64)    [1.5, NULL, NaN, 4.5]
//	s (string) ["a", NULL, "c", "d"]
nulls_test_df :: proc(t: ^testing.T) -> (df: DataFrame, ctx: expr.Context) {
	n: Column
	m: Column
	o: Column
	f: Column
	g: Column
	s: Column
	err: Error

	n, err = column_from("n", []i32{10, 0, 30, 40})
	testing.expect(t, err == .None, "n column")
	testing.expect(t, column_set_valid(&n, 1, false) == .None, "n[1] NULL")

	m, err = column_from("m", []i32{0, 20, 0, 0})
	testing.expect(t, err == .None, "m column")
	testing.expect(t, column_set_valid(&m, 0, false) == .None, "m[0] NULL")
	testing.expect(t, column_set_valid(&m, 2, false) == .None, "m[2] NULL")
	testing.expect(t, column_set_valid(&m, 3, false) == .None, "m[3] NULL")

	o, err = column_from("o", []i32{0, 5, 0, 0})
	testing.expect(t, err == .None, "o column")
	testing.expect(t, column_set_valid(&o, 0, false) == .None, "o[0] NULL")
	testing.expect(t, column_set_valid(&o, 2, false) == .None, "o[2] NULL")
	testing.expect(t, column_set_valid(&o, 3, false) == .None, "o[3] NULL")

	f, err = column_from("f", []f64{1.5, 2.5, 3.5, 4.5})
	testing.expect(t, err == .None, "f column")
	testing.expect(t, column_set_valid(&f, 1, false) == .None, "f[1] NULL")

	g, err = column_from("g", []f64{1.5, 2.5, math.nan_f64(), 4.5})
	testing.expect(t, err == .None, "g column")
	testing.expect(t, column_set_valid(&g, 1, false) == .None, "g[1] NULL")

	s, err = column_from("s", []string{"a", "", "c", "d"})
	testing.expect(t, err == .None, "s column")
	testing.expect(t, column_set_valid(&s, 1, false) == .None, "s[1] NULL")

	df, err = dataframe_from_columns([]^Column{&n, &m, &o, &f, &g, &s})
	testing.expect(t, err == .None, "from_columns")
	ctx = expr.context_create(context.allocator)
	return
}

nulls_test_destroy :: proc(t: ^testing.T, df: ^DataFrame, ctx: ^expr.Context) {
	expr.context_destroy(ctx)
	dataframe_destroy(df)
}

// --- is_nan -------------------------------------------------------------------

@(test)
nulls_is_nan_basic :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.is_nan_(&ctx, expr.col(&ctx, "g")))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(bool), "result dtype")
	testing.expect(t, !column_typed_view(&out, bool)[0], "row 0 false")
	testing.expect(t, !column_is_valid(&out, 1), "NULL stays NULL")
	testing.expect(t, column_typed_view(&out, bool)[2], "row 2 is NaN")
	testing.expect(t, !column_typed_view(&out, bool)[3], "row 3 false")
}

@(test)
nulls_is_nan_rejects_non_float :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	_, err := expr_eval(context.allocator, &df, expr.is_nan_(&ctx, expr.col(&ctx, "n")))
	testing.expect(t, err == .Unsupported_Operation, "int input errors")
}

// --- fill_null ----------------------------------------------------------------

@(test)
nulls_fill_null_i32 :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.fill_null_(&ctx, expr.col(&ctx, "n"), expr.lit(&ctx, i32(5))))
	defer column_destroy(&out)

	nv := column_typed_view(&out, i32)
	for i in 0 ..< out.count {
		testing.expect(t, column_is_valid(&out, i), "all rows valid")
	}
	testing.expect(t, nv[0] == 10 && nv[1] == 5 && nv[2] == 30 && nv[3] == 40, "filled values")
}

@(test)
nulls_fill_null_string :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.fill_null_(&ctx, expr.col(&ctx, "s"), expr.lit(&ctx, "z")))
	defer column_destroy(&out)

	sv := column_typed_view(&out, string)
	testing.expect(t, sv[0] == "a" && sv[1] == "z" && sv[2] == "c" && sv[3] == "d", "filled strings")
	for i in 0 ..< out.count {
		testing.expect(t, column_is_valid(&out, i), "all rows valid")
	}
}

@(test)
nulls_fill_null_type_mismatch :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	_, err := expr_eval(context.allocator, &df, expr.fill_null_(&ctx, expr.col(&ctx, "n"), expr.lit(&ctx, "z")))
	testing.expect(t, err == .Type_Mismatch, "i32 column with string fill errors")
}

@(test)
nulls_fill_null_no_nulls_is_passthrough :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.fill_null_(&ctx, expr.lit(&ctx, i32(5)), expr.lit(&ctx, i32(9))))
	defer column_destroy(&out)

	testing.expect(t, out.count == 4, "literal broadcasts to frame rows")
	testing.expect(t, column_typed_view(&out, i32)[0] == 5, "literal stays")
}

// --- coalesce -----------------------------------------------------------------

@(test)
nulls_coalesce_first_valid :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.coalesce_(&ctx, []^expr.Expr{expr.col(&ctx, "n"), expr.col(&ctx, "m")}))
	defer column_destroy(&out)

	nv := column_typed_view(&out, i32)
	testing.expect(t, nv[0] == 10 && nv[1] == 20 && nv[2] == 30 && nv[3] == 40, "first valid per row")
	for i in 0 ..< out.count {
		testing.expect(t, column_is_valid(&out, i), "all rows valid")
	}
}

@(test)
nulls_coalesce_all_null_row_stays_null :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.coalesce_(&ctx, []^expr.Expr{expr.col(&ctx, "m"), expr.col(&ctx, "o")}))
	defer column_destroy(&out)

	nv := column_typed_view(&out, i32)
	testing.expect(t, !column_is_valid(&out, 0), "row 0 all parts NULL")
	testing.expect(t, nv[1] == 20, "row 1 from first part")
	testing.expect(t, !column_is_valid(&out, 2) && !column_is_valid(&out, 3), "trailing NULLs stay")
}

@(test)
nulls_coalesce_with_literal :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.coalesce_(&ctx, []^expr.Expr{expr.col(&ctx, "m"), expr.lit(&ctx, i32(7))}))
	defer column_destroy(&out)

	nv := column_typed_view(&out, i32)
	testing.expect(t, nv[0] == 7, "row 0 from literal")
	testing.expect(t, nv[1] == 20, "row 1 from column")
	testing.expect(t, nv[2] == 7 && nv[3] == 7, "trailing from literal")
}

@(test)
nulls_coalesce_type_mismatch :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	_, err := expr_eval(context.allocator, &df, expr.coalesce_(&ctx, []^expr.Expr{expr.col(&ctx, "n"), expr.col(&ctx, "s")}))
	testing.expect(t, err == .Type_Mismatch, "differing part dtypes error")
}

@(test)
nulls_coalesce_string :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.coalesce_(&ctx, []^expr.Expr{expr.col(&ctx, "s"), expr.lit(&ctx, "fallback")}))
	defer column_destroy(&out)

	sv := column_typed_view(&out, string)
	testing.expect(t, sv[0] == "a", "row 0 from column")
	testing.expect(t, sv[1] == "fallback", "row 1 from literal")
	testing.expect(t, sv[2] == "c" && sv[3] == "d", "rows 2-3 from column")
	for i in 0 ..< out.count {
		testing.expect(t, column_is_valid(&out, i), "all rows valid")
	}
}

// --- forward / backward fill ---------------------------------------------------

@(test)
nulls_ffill_carries_last_valid :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.ffill_(&ctx, expr.col(&ctx, "n")))
	defer column_destroy(&out)

	nv := column_typed_view(&out, i32)
	testing.expect(t, nv[0] == 10 && nv[1] == 10 && nv[2] == 30 && nv[3] == 40, "filled")
	for i in 0 ..< out.count {
		testing.expect(t, column_is_valid(&out, i), "all rows valid")
	}
}

@(test)
nulls_ffill_leading_null_stays_null :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.ffill_(&ctx, expr.col(&ctx, "m")))
	defer column_destroy(&out)

	nv := column_typed_view(&out, i32)
	testing.expect(t, !column_is_valid(&out, 0), "leading NULL stays NULL")
	testing.expect(t, nv[1] == 20 && nv[2] == 20 && nv[3] == 20, "carried forward")
}

@(test)
nulls_bfill_carries_next_valid :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.bfill_(&ctx, expr.col(&ctx, "m")))
	defer column_destroy(&out)

	nv := column_typed_view(&out, i32)
	testing.expect(t, nv[0] == 20 && nv[1] == 20, "rows 0-1 filled")
	testing.expect(t, !column_is_valid(&out, 2) && !column_is_valid(&out, 3), "trailing NULLs stay NULL")
}

@(test)
nulls_ffill_string :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.ffill_(&ctx, expr.col(&ctx, "s")))
	defer column_destroy(&out)

	sv := column_typed_view(&out, string)
	testing.expect(t, sv[0] == "a" && sv[1] == "a" && sv[2] == "c" && sv[3] == "d", "string ffill")
}

// --- interpolate ----------------------------------------------------------------

@(test)
nulls_interpolate_linear :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	out := eval_ok(t, &df, expr.interpolate_(&ctx, expr.col(&ctx, "f")))
	defer column_destroy(&out)

	testing.expect(t, out.dtype == typeid_of(f64), "result dtype is f64")
	fv := column_typed_view(&out, f64)
	testing.expect(t, math.abs(fv[0] - 1.5) < 1e-12, "row 0")
	testing.expect(t, math.abs(fv[1] - 2.5) < 1e-12, "row 1 interpolated")
	testing.expect(t, math.abs(fv[2] - 3.5) < 1e-12, "row 2")
	testing.expect(t, math.abs(fv[3] - 4.5) < 1e-12, "row 3")
	for i in 0 ..< out.count {
		testing.expect(t, column_is_valid(&out, i), "all rows valid")
	}
}

@(test)
nulls_interpolate_leading_trailing_stay_null :: proc(t: ^testing.T) {
	df, ctx := nulls_test_df(t)
	defer nulls_test_destroy(t, &df, &ctx)

	// fcol = [NULL, NULL, 10, 20] -> interpolate keeps the leading NULLs.
	fcol, _ := column_from("fcol", []f64{0, 0, 10, 20})
	testing.expect(t, column_set_valid(&fcol, 0, false) == .None, "fcol[0] NULL")
	testing.expect(t, column_set_valid(&fcol, 1, false) == .None, "fcol[1] NULL")
	testing.expect(t, dataframe_add_column(&df, &fcol) == .None, "add fcol")

	out := eval_ok(t, &df, expr.interpolate_(&ctx, expr.col(&ctx, "fcol")))
	defer column_destroy(&out)

	fv := column_typed_view(&out, f64)
	testing.expect(t, !column_is_valid(&out, 0) && !column_is_valid(&out, 1), "leading NULLs stay")
	testing.expect(t, math.abs(fv[2] - 10) < 1e-12 && math.abs(fv[3] - 20) < 1e-12, "valid rows pass")
}

// --- drop_nulls rows ------------------------------------------------------------

// drop_fixture builds a frame where only rows 0 and 2 are fully valid:
//
//	a (i32)    [1, 2, 3]
//	b (i32)    [4, NULL, 6]
//	c (string) ["x", "y", "z"]
drop_fixture :: proc(t: ^testing.T) -> DataFrame {
	a, _ := column_from("a", []i32{1, 2, 3})
	b, _ := column_from("b", []i32{4, 0, 6})
	testing.expect(t, column_set_valid(&b, 1, false) == .None, "b[1] NULL")
	c, _ := column_from("c", []string{"x", "y", "z"})
	df, err := dataframe_from_columns([]^Column{&a, &b, &c})
	testing.expect(t, err == .None, "from_columns")
	return df
}

@(test)
nulls_drop_nulls_all_columns :: proc(t: ^testing.T) {
	df := drop_fixture(t)
	defer dataframe_destroy(&df)

	out, err := dataframe_drop_nulls(&df, {}, context.allocator)
	testing.expect(t, err == .None, "drop_nulls succeeds")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 2, "rows kept")
	a_col, a_err := dataframe_get_column(&out, "a")
	testing.expect(t, a_err == .None, "get kept column")
	av := column_typed_view(a_col, i32)
	testing.expect(t, av[0] == 1 && av[1] == 3, "kept values")
}

@(test)
nulls_drop_nulls_subset_columns :: proc(t: ^testing.T) {
	df := drop_fixture(t)
	defer dataframe_destroy(&df)

	out, err := dataframe_drop_nulls(&df, []string{"a"}, context.allocator)
	testing.expect(t, err == .None, "drop_nulls on a")
	defer dataframe_destroy(&out)

	// a is fully valid -> every row is kept.
	testing.expect(t, dataframe_num_rows(&out) == 3, "rows kept")
}

@(test)
nulls_drop_nulls_unknown_column :: proc(t: ^testing.T) {
	df := drop_fixture(t)
	defer dataframe_destroy(&df)

	_, err := dataframe_drop_nulls(&df, []string{"missing"}, context.allocator)
	testing.expect(t, err == .Column_Not_Found, "unknown column errors")
}
