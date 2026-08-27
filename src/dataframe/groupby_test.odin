package dataframe

// Stage 7 groupby tests (ROADMAP S7.5): grouping correctness, empty groups,
// multiple keys, NULL keys, NULL values within groups, the S7.2/S7.3 agg
// matrix, and error cases.

import "core:testing"
import "expr"

// groupby_test_df builds the shared Stage 7 fixture:
//
//	dept  (string) ["eng", "eng", "eng", "sales", "sales", NULL, NULL, "eng"]
//	team  (i32)    [1, 1, 2, 1, 1, 2, 2, 1]
//	value (f64)    [10, 20, 30, 5, NULL, 7, 8, NULL]
//
// Group-by-dept expectations (first-appearance order):
//   eng      rows 0,1,2,7 -> value [10,20,30,NULL]
//   sales    rows 3,4     -> value [5,NULL]
//   NULL     rows 5,6     -> value [7,8]
//
// Group-by-team expectations:
//   team=1 rows 0,1,3,4,7 -> value [10,20,5,NULL,NULL]
//   team=2 rows 2,5,6     -> value [30,7,8]
groupby_test_df :: proc(t: ^testing.T) -> (df: DataFrame, ctx: expr.Context) {
	err: Error
	dept, team, value: Column
	dept, err = column_from("dept", []string{"eng", "eng", "eng", "sales", "sales", "x", "x", "eng"})
	testing.expect(t, err == .None, "dept column")
	testing.expect(t, column_set_valid(&dept, 5, false) == .None, "dept[5] NULL")
	testing.expect(t, column_set_valid(&dept, 6, false) == .None, "dept[6] NULL")

	team, err = column_from("team", []i32{1, 1, 2, 1, 1, 2, 2, 1})
	testing.expect(t, err == .None, "team column")

	value, err = column_from("value", []f64{10, 20, 30, 5, 0, 7, 8, 0})
	testing.expect(t, err == .None, "value column")
	testing.expect(t, column_set_valid(&value, 4, false) == .None, "value[4] NULL")
	testing.expect(t, column_set_valid(&value, 7, false) == .None, "value[7] NULL")

	df, err = dataframe_from_columns([]^Column{&dept, &team, &value})
	testing.expect(t, err == .None, "from_columns")
	ctx = expr.context_create(context.allocator)
	return
}

groupby_test_destroy :: proc(t: ^testing.T, df: ^DataFrame, ctx: ^expr.Context) {
	expr.context_destroy(ctx)
	dataframe_destroy(df)
}

// groupby_agg_ok runs group_by + agg, asserting both succeed and releasing the
// group handle. The caller owns the returned DataFrame.
groupby_agg_ok :: proc(t: ^testing.T, df: ^DataFrame, ctx: ^expr.Context, keys: []^expr.Expr, aggs: []^expr.Expr) -> (out: DataFrame) {
	gb, gerr := dataframe_group_by(df, keys)
	testing.expect(t, gerr == .None, "group_by err")
	if gerr != .None {
		return {}
	}
	defer dataframe_group_by_destroy(&gb)
	err: Error
	out, err = dataframe_group_by_agg(&gb, aggs)
	testing.expect(t, err == .None, "group_by_agg err")
	return out
}

// --- S7.2 count/sum/mean/min/max --------------------------------------------

@(test)
groupby_test_count_sum :: proc(t: ^testing.T) {
	df, ctx := groupby_test_df(t)
	defer groupby_test_destroy(t, &df, &ctx)

	out := groupby_agg_ok(t, &df, &ctx,
		[]^expr.Expr{expr.col(&ctx, "dept")},
		[]^expr.Expr{
			expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "value")), "count"),
			expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "value")), "sum"),
		},
	)
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 3, "3 groups")
	testing.expect(t, dataframe_num_cols(&out) == 3, "dept + count + sum")

	dept := dataframe_get_column(&out, "dept") or_else nil
	cnt := dataframe_get_column(&out, "count") or_else nil
	sum := dataframe_get_column(&out, "sum") or_else nil

	d, _, _ := column_get(dept, 0, string)
	testing.expect(t, d == "eng", "group 0 key = eng")
	c, _, _ := column_get(cnt, 0, i64)
	testing.expect(t, c == 3, "eng count = 3")
	s, _, _ := column_get(sum, 0, f64)
	testing.expect(t, near(s, 60), "eng sum = 60")

	d, _, _ = column_get(dept, 1, string)
	testing.expect(t, d == "sales", "group 1 key = sales")
	c, _, _ = column_get(cnt, 1, i64)
	testing.expect(t, c == 1, "sales count = 1")
	s, _, _ = column_get(sum, 1, f64)
	testing.expect(t, near(s, 5), "sales sum = 5")

	// NULL dept rows form a single group.
	testing.expect(t, !column_is_valid(dept, 2), "group 2 key is NULL")
	c, _, _ = column_get(cnt, 2, i64)
	testing.expect(t, c == 2, "NULL-dept count = 2")
	s, _, _ = column_get(sum, 2, f64)
	testing.expect(t, near(s, 15), "NULL-dept sum = 15")
}

@(test)
groupby_test_min_max_mean :: proc(t: ^testing.T) {
	df, ctx := groupby_test_df(t)
	defer groupby_test_destroy(t, &df, &ctx)

	out := groupby_agg_ok(t, &df, &ctx,
		[]^expr.Expr{expr.col(&ctx, "team")},
		[]^expr.Expr{
			expr.alias(&ctx, expr.min_(&ctx, expr.col(&ctx, "value")), "mn"),
			expr.alias(&ctx, expr.max_(&ctx, expr.col(&ctx, "value")), "mx"),
			expr.alias(&ctx, expr.mean_(&ctx, expr.col(&ctx, "value")), "avg"),
		},
	)
	defer dataframe_destroy(&out)

	mn := dataframe_get_column(&out, "mn") or_else nil
	mx := dataframe_get_column(&out, "mx") or_else nil
	avg := dataframe_get_column(&out, "avg") or_else nil

	// team=1: valid value [10,20,5] -> min 5, max 20, mean 35/3.
	v, _, _ := column_get(mn, 0, f64)
	testing.expect(t, near(v, 5), "team1 min = 5")
	v, _, _ = column_get(mx, 0, f64)
	testing.expect(t, near(v, 20), "team1 max = 20")
	v, _, _ = column_get(avg, 0, f64)
	testing.expect(t, near(v, 35.0 / 3.0), "team1 mean = 35/3")

	// team=2: [30,7,8] -> min 7, max 30, mean 15.
	v, _, _ = column_get(mn, 1, f64)
	testing.expect(t, near(v, 7), "team2 min = 7")
	v, _, _ = column_get(mx, 1, f64)
	testing.expect(t, near(v, 30), "team2 max = 30")
	v, _, _ = column_get(avg, 1, f64)
	testing.expect(t, near(v, 15), "team2 mean = 15")
}

// --- S7.3 var/std/median/quantile/n_unique/first/last ------------------------

@(test)
groupby_test_extended_aggs :: proc(t: ^testing.T) {
	df, ctx := groupby_test_df(t)
	defer groupby_test_destroy(t, &df, &ctx)

	out := groupby_agg_ok(t, &df, &ctx,
		[]^expr.Expr{expr.col(&ctx, "team")},
		[]^expr.Expr{
			expr.alias(&ctx, expr.var_(&ctx, expr.col(&ctx, "value")), "v"),
			expr.alias(&ctx, expr.std_(&ctx, expr.col(&ctx, "value")), "sd"),
			expr.alias(&ctx, expr.median_(&ctx, expr.col(&ctx, "value")), "med"),
			expr.alias(&ctx, expr.quantile_(&ctx, expr.col(&ctx, "value"), 0.25), "q"),
			expr.alias(&ctx, expr.n_unique_(&ctx, expr.col(&ctx, "value")), "nu"),
			expr.alias(&ctx, expr.first_(&ctx, expr.col(&ctx, "value")), "f"),
			expr.alias(&ctx, expr.last_(&ctx, expr.col(&ctx, "value")), "l"),
		},
	)
	defer dataframe_destroy(&out)

	// Row 1 = team=2 group, rows 2,5,6 -> value [30,7,8].
	v := dataframe_get_column(&out, "v") or_else nil
	val, _, _ := column_get(v, 1, f64)
	testing.expect(t, near(val, 169), "team2 sample var = 169")
	sd := dataframe_get_column(&out, "sd") or_else nil
	val, _, _ = column_get(sd, 1, f64)
	testing.expect(t, near(val, 13), "team2 std = 13")
	med := dataframe_get_column(&out, "med") or_else nil
	val, _, _ = column_get(med, 1, f64)
	testing.expect(t, near(val, 8), "team2 median = 8")
	q := dataframe_get_column(&out, "q") or_else nil
	val, _, _ = column_get(q, 1, f64)
	testing.expect(t, near(val, 7.5), "team2 q0.25 = 7.5")
	nu := dataframe_get_column(&out, "nu") or_else nil
	n, _, _ := column_get(nu, 1, i64)
	testing.expect(t, n == 3, "team2 n_unique = 3")
	f := dataframe_get_column(&out, "f") or_else nil
	val, _, _ = column_get(f, 1, f64)
	testing.expect(t, near(val, 30), "team2 first = 30")
	l := dataframe_get_column(&out, "l") or_else nil
	val, _, _ = column_get(l, 1, f64)
	testing.expect(t, near(val, 8), "team2 last = 8")
}

@(test)
groupby_test_string_agg :: proc(t: ^testing.T) {
	df, ctx := groupby_test_df(t)
	defer groupby_test_destroy(t, &df, &ctx)

	out := groupby_agg_ok(t, &df, &ctx,
		[]^expr.Expr{expr.col(&ctx, "team")},
		[]^expr.Expr{
			expr.alias(&ctx, expr.first_(&ctx, expr.col(&ctx, "dept")), "f"),
			expr.alias(&ctx, expr.last_(&ctx, expr.col(&ctx, "dept")), "l"),
		},
	)
	defer dataframe_destroy(&out)

	// team=1 rows 0,1,3,4,7 -> dept [eng,eng,sales,sales,eng]: first/last eng.
	f := dataframe_get_column(&out, "f") or_else nil
	l := dataframe_get_column(&out, "l") or_else nil
	d, _, _ := column_get(f, 0, string)
	testing.expect(t, d == "eng", "team1 first dept = eng")
	d, _, _ = column_get(l, 0, string)
	testing.expect(t, d == "eng", "team1 last dept = eng")

	// team=2 rows 2,5,6 -> dept [eng,NULL,NULL]: first/last valid = eng.
	d, _, _ = column_get(f, 1, string)
	testing.expect(t, d == "eng", "team2 first dept = eng")
	d, _, _ = column_get(l, 1, string)
	testing.expect(t, d == "eng", "team2 last dept = eng (skips NULLs)")
}

// --- multiple keys / NULL keys ----------------------------------------------

@(test)
groupby_test_multiple_keys :: proc(t: ^testing.T) {
	df, ctx := groupby_test_df(t)
	defer groupby_test_destroy(t, &df, &ctx)

	out := groupby_agg_ok(t, &df, &ctx,
		[]^expr.Expr{expr.col(&ctx, "dept"), expr.col(&ctx, "team")},
		[]^expr.Expr{expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "value")), "count")},
	)
	defer dataframe_destroy(&out)

	// First-appearance order:
	//   (eng,1) rows 0,1,7 -> count(value)=2
	//   (eng,2) rows 2     -> 1
	//   (sales,1) rows 3,4 -> 1
	//   (NULL,2) rows 5,6  -> 2
	testing.expect(t, dataframe_num_rows(&out) == 4, "4 groups")
	dept := dataframe_get_column(&out, "dept") or_else nil
	team := dataframe_get_column(&out, "team") or_else nil
	cnt := dataframe_get_column(&out, "count") or_else nil

	d, _, _ := column_get(dept, 0, string)
	tm, _, _ := column_get(team, 0, i32)
	c, _, _ := column_get(cnt, 0, i64)
	testing.expect(t, d == "eng" && tm == 1 && c == 2, "(eng,1) count = 2")

	d, _, _ = column_get(dept, 1, string)
	tm, _, _ = column_get(team, 1, i32)
	c, _, _ = column_get(cnt, 1, i64)
	testing.expect(t, d == "eng" && tm == 2 && c == 1, "(eng,2) count = 1")

	d, _, _ = column_get(dept, 2, string)
	tm, _, _ = column_get(team, 2, i32)
	c, _, _ = column_get(cnt, 2, i64)
	testing.expect(t, d == "sales" && tm == 1 && c == 1, "(sales,1) count = 1")

	// NULL dept + team 2 is one group; the key column carries the NULL flag.
	testing.expect(t, !column_is_valid(dept, 3), "group 3 key dept NULL")
	tm, _, _ = column_get(team, 3, i32)
	c, _, _ = column_get(cnt, 3, i64)
	testing.expect(t, tm == 2 && c == 2, "(NULL,2) count = 2")
}

// --- NULL values inside groups / NULL-only groups ----------------------------

@(test)
groupby_test_null_only_group_value :: proc(t: ^testing.T) {
	k, err := column_from("k", []i32{1, 1, 2})
	testing.expect(t, err == .None, "k column")
	v: Column
	v, err = column_from("v", []f64{1, 2, 3})
	testing.expect(t, err == .None, "v column")
	for i in 0 ..< 3 {
		testing.expect(t, column_set_valid(&v, i, false) == .None, "v NULL")
	}
	df: DataFrame
	df, err = dataframe_from_columns([]^Column{&k, &v})
	testing.expect(t, err == .None, "from_columns")
	defer dataframe_destroy(&df)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	out := groupby_agg_ok(t, &df, &ctx,
		[]^expr.Expr{expr.col(&ctx, "k")},
		[]^expr.Expr{
			expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "v")), "cnt"),
			expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "v")), "sum"),
		},
	)
	defer dataframe_destroy(&out)

	cnt := dataframe_get_column(&out, "cnt") or_else nil
	sum := dataframe_get_column(&out, "sum") or_else nil

	n, _, _ := column_get(cnt, 0, i64)
	testing.expect(t, n == 0, "group 1 count = 0 (no valid rows)")
	testing.expect(t, !column_is_valid(sum, 0), "group 1 sum is NULL")
	n, _, _ = column_get(cnt, 1, i64)
	testing.expect(t, n == 0, "group 2 count = 0 (no valid rows)")
	testing.expect(t, !column_is_valid(sum, 1), "group 2 sum is NULL")
}

// --- agg over a computed expression ------------------------------------------

@(test)
groupby_test_agg_over_expression :: proc(t: ^testing.T) {
	df, ctx := groupby_test_df(t)
	defer groupby_test_destroy(t, &df, &ctx)

	// sum(value * 2) per team; the child is a Binary, so the agg must be
	// aliased. team=1 valid [10,20,5] -> 20+40+10 = 70; team=2 [30,7,8] -> 90.
	e := expr.mul(&ctx, expr.col(&ctx, "value"), expr.lit(&ctx, f64(2)))
	out := groupby_agg_ok(t, &df, &ctx,
		[]^expr.Expr{expr.col(&ctx, "team")},
		[]^expr.Expr{expr.alias(&ctx, expr.sum_(&ctx, e), "sum2")},
	)
	defer dataframe_destroy(&out)

	sum := dataframe_get_column(&out, "sum2") or_else nil
	v, _, _ := column_get(sum, 0, f64)
	testing.expect(t, near(v, 70), "team1 sum(value*2) = 70")
	v, _, _ = column_get(sum, 1, f64)
	testing.expect(t, near(v, 90), "team2 sum(value*2) = 90")
}

// --- consistency and edge cases ----------------------------------------------

@(test)
groupby_test_consistency_with_full_agg :: proc(t: ^testing.T) {
	df, ctx := groupby_test_df(t)
	defer groupby_test_destroy(t, &df, &ctx)

	// A constant key yields one group that must match the whole-column agg.
	out := groupby_agg_ok(t, &df, &ctx,
		[]^expr.Expr{expr.alias(&ctx, expr.lit(&ctx, i32(1)), "one")},
		[]^expr.Expr{expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "value")), "s")},
	)
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 1, "one group")
	s := dataframe_get_column(&out, "s") or_else nil
	v, _, _ := column_get(s, 0, f64)
	whole, werr := dataframe_sum(&df, "value")
	testing.expect(t, werr == .None, "whole sum")
	testing.expect(t, near(v, whole), "group sum == whole-column sum")
}

@(test)
groupby_test_empty_df :: proc(t: ^testing.T) {
	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	k, err := column_from("k", []i32{})
	testing.expect(t, err == .None, "k column")
	v: Column
	v, err = column_from("v", []f64{})
	testing.expect(t, err == .None, "v column")
	df: DataFrame
	df, err = dataframe_from_columns([]^Column{&k, &v})
	testing.expect(t, err == .None, "from_columns")
	defer dataframe_destroy(&df)

	gb, gerr := dataframe_group_by(&df, []^expr.Expr{expr.col(&ctx, "k")})
	testing.expect(t, gerr == .None, "group_by on empty df")
	defer dataframe_group_by_destroy(&gb)

	out, aerr := dataframe_group_by_agg(&gb, []^expr.Expr{
		expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "v")), "s"),
	})
	testing.expect(t, aerr == .None, "group_by_agg on empty df")
	defer dataframe_destroy(&out)

	// The schema is preserved: 0 rows, correct dtypes and names.
	testing.expect(t, dataframe_num_rows(&out) == 0, "0 rows")
	testing.expect(t, dataframe_num_cols(&out) == 2, "k + s")
	kc := dataframe_get_column(&out, "k") or_else nil
	testing.expect(t, kc.dtype == typeid_of(i32), "k dtype preserved")
	sc := dataframe_get_column(&out, "s") or_else nil
	testing.expect(t, sc.dtype == typeid_of(f64), "s dtype preserved")
}

@(test)
groupby_test_result_dtypes :: proc(t: ^testing.T) {
	df, ctx := groupby_test_df(t)
	defer groupby_test_destroy(t, &df, &ctx)

	out := groupby_agg_ok(t, &df, &ctx,
		[]^expr.Expr{expr.col(&ctx, "team")},
		[]^expr.Expr{
			expr.alias(&ctx, expr.min_(&ctx, expr.col(&ctx, "team")), "mteam"),
			expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "dept")), "cnt"),
		},
	)
	defer dataframe_destroy(&out)

	m := dataframe_get_column(&out, "mteam") or_else nil
	testing.expect(t, m.dtype == typeid_of(i32), "min preserves i32")
	c := dataframe_get_column(&out, "cnt") or_else nil
	testing.expect(t, c.dtype == typeid_of(i64), "count -> i64")
}

// --- error cases -------------------------------------------------------------

@(test)
groupby_test_errors :: proc(t: ^testing.T) {
	df, ctx := groupby_test_df(t)
	defer groupby_test_destroy(t, &df, &ctx)

	// Empty key exprs.
	gb, gerr := dataframe_group_by(&df, nil)
	testing.expect(t, gerr == .Invalid_Argument, "empty keys rejected")
	if gerr == .None {
		dataframe_group_by_destroy(&gb)
	}

	// Unnamed key expression.
	e := expr.mul(&ctx, expr.col(&ctx, "team"), expr.lit(&ctx, i32(2)))
	gb, gerr = dataframe_group_by(&df, []^expr.Expr{e})
	testing.expect(t, gerr == .Invalid_Argument, "unnamed key rejected")
	if gerr == .None {
		dataframe_group_by_destroy(&gb)
	}

	// Unknown column as a key.
	gb, gerr = dataframe_group_by(&df, []^expr.Expr{expr.col(&ctx, "nope")})
	testing.expect(t, gerr == .Column_Not_Found, "unknown key column rejected")
	if gerr == .None {
		dataframe_group_by_destroy(&gb)
	}

	gb, gerr = dataframe_group_by(&df, []^expr.Expr{expr.col(&ctx, "dept")})
	testing.expect(t, gerr == .None, "group_by ok")
	defer dataframe_group_by_destroy(&gb)

	// Empty agg list.
	_, aerr := dataframe_group_by_agg(&gb, nil)
	testing.expect(t, aerr == .Invalid_Argument, "empty aggs rejected")

	// Non-aggregation expression (a plain column).
	_, aerr = dataframe_group_by_agg(&gb, []^expr.Expr{expr.col(&ctx, "value")})
	testing.expect(t, aerr == .Unsupported_Operation, "non-agg expression rejected")

	// Unnamed agg result (child is a Binary, no alias).
	e2 := expr.mul(&ctx, expr.col(&ctx, "value"), expr.lit(&ctx, f64(2)))
	_, aerr = dataframe_group_by_agg(&gb, []^expr.Expr{expr.sum_(&ctx, e2)})
	testing.expect(t, aerr == .Invalid_Argument, "unnamed agg result rejected")

	// Unknown column in an agg child.
	_, aerr = dataframe_group_by_agg(&gb, []^expr.Expr{
		expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "nope")), "s"),
	})
	testing.expect(t, aerr == .Column_Not_Found, "unknown agg column rejected")

	// Aggregation kind unsupported for the child dtype.
	_, aerr = dataframe_group_by_agg(&gb, []^expr.Expr{
		expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "dept")), "s"),
	})
	testing.expect(t, aerr == .Unsupported_Operation, "sum(string) rejected")

	// Duplicate result names: two aggs over the same column without aliases.
	_, aerr = dataframe_group_by_agg(&gb, []^expr.Expr{
		expr.count_(&ctx, expr.col(&ctx, "value")),
		expr.sum_(&ctx, expr.col(&ctx, "value")),
	})
	testing.expect(t, aerr == .Duplicate_Column_Name, "duplicate agg names rejected")

	// Key/agg name collision.
	_, aerr = dataframe_group_by_agg(&gb, []^expr.Expr{
		expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "value")), "dept"),
	})
	testing.expect(t, aerr == .Duplicate_Column_Name, "key/agg name collision rejected")
}
