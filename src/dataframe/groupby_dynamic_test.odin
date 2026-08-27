package dataframe

// Stage 14.7 dynamic and rolling groupby tests (ROADMAP S14.7, DESIGN.md
// §18.4): window grouping, NULL handling, closed variants, output ordering,
// pre-epoch exactness, and error cases.

import "core:testing"
import "expr"

// dynamic_test_df builds the shared Stage 14.7 fixture with d0 =
// 2024-01-01T00:00:00:
//
//	ts   (Datetime) [d0, d0+1h, d0+1h30, d0+3h, d0+4h30, d0+5h, NULL, d0+5h30]
//	sym  (string)   [A, A, B, B, A, A, A, A]
//	val  (f64)      [10, 20, 30, 5, NULL, 7, 8, NULL]
//
// Dynamic (every 2h, offset 0, closed .Left) windows:
//
//	start 00:00 rows 0,1,2 | start 02:00 row 3 | start 04:00 rows 4,5,7
//
// (row 6 has a NULL ts and belongs to no window.)
dynamic_test_df :: proc(t: ^testing.T) -> (df: DataFrame, ctx: expr.Context) {
	d0, derr := datetime_create(2024, 1, 1, 0, 0, 0)
	testing.expect(t, derr == .None, "d0")
	d1 := duration_from_hours(1)
	d2 := duration_from_hours(2)
	m30 := duration_from_minutes(30)
	h3 := duration_from_hours(3)
	h4 := duration_from_hours(4)
	h5 := duration_from_hours(5)

	ts, sym, val: Column
	err: Error
	ts, err = column_from("ts", []Datetime{
		d0,
		datetime_add(d0, d1),
		datetime_add(d0, d1 + m30),
		datetime_add(d0, h3),
		datetime_add(d0, h4 + m30),
		datetime_add(d0, h5),
		datetime_add(d0, d2),
		datetime_add(d0, h5 + m30),
	})
	testing.expect(t, err == .None, "ts column")
	testing.expect(t, column_set_valid(&ts, 6, false) == .None, "ts[6] NULL")

	sym, err = column_from("sym", []string{"A", "A", "B", "B", "A", "A", "A", "A"})
	testing.expect(t, err == .None, "sym column")

	val, err = column_from("val", []f64{10, 20, 30, 5, 0, 7, 8, 0})
	testing.expect(t, err == .None, "val column")
	testing.expect(t, column_set_valid(&val, 4, false) == .None, "val[4] NULL")
	testing.expect(t, column_set_valid(&val, 7, false) == .None, "val[7] NULL")

	df, err = dataframe_from_columns([]^Column{&ts, &sym, &val})
	testing.expect(t, err == .None, "from_columns")
	ctx = expr.context_create(context.allocator)
	return
}

dynamic_test_destroy :: proc(t: ^testing.T, df: ^DataFrame, ctx: ^expr.Context) {
	expr.context_destroy(ctx)
	dataframe_destroy(df)
}

// dynamic_groupby_agg_ok builds a dynamic group, runs the agg, and returns the
// owned result DataFrame.
dynamic_groupby_agg_ok :: proc(
	t: ^testing.T,
	df: ^DataFrame,
	time_col: string,
	by: []string,
	every: Duration,
	offset: Duration,
	closed: Closed_Interval,
	aggs: []^expr.Expr,
) -> (out: DataFrame) {
	g, gerr := dataframe_group_by_dynamic(df, time_col, by, every, offset, closed)
	testing.expect(t, gerr == .None, "group_by_dynamic err")
	if gerr != .None {
		return {}
	}
	defer dataframe_dynamic_group_by_destroy(&g)
	err: Error
	out, err = dataframe_dynamic_group_by_agg(&g, aggs)
	testing.expect(t, err == .None, "dynamic agg err")
	return out
}

// rolling_groupby_agg_ok builds a rolling group, runs the agg, and returns the
// owned result DataFrame.
rolling_groupby_agg_ok :: proc(
	t: ^testing.T,
	df: ^DataFrame,
	time_col: string,
	by: []string,
	period: Duration,
	offset: Duration,
	closed: Closed_Interval,
	aggs: []^expr.Expr,
) -> (out: DataFrame) {
	g, gerr := dataframe_group_by_rolling(df, time_col, by, period, offset, closed)
	testing.expect(t, gerr == .None, "group_by_rolling err")
	if gerr != .None {
		return {}
	}
	defer dataframe_rolling_group_by_destroy(&g)
	err: Error
	out, err = dataframe_rolling_group_by_agg(&g, aggs)
	testing.expect(t, err == .None, "rolling agg err")
	return out
}

// --- dynamic groupby ---------------------------------------------------------

@(test)
dynamic_groupby_test_windows_by :: proc(t: ^testing.T) {
	df, ctx := dynamic_test_df(t)
	defer dynamic_test_destroy(t, &df, &ctx)

	out := dynamic_groupby_agg_ok(t, &df, "ts", {"sym"}, duration_from_hours(2), Duration(0), .Left, []^expr.Expr{
		expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "val")), "cnt"),
		expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "val")), "sum"),
	})
	defer dataframe_destroy(&out)

	// Windows in (start ascending, by first-seen) order:
	//   (A, 00:00) rows 0,1 -> count 2, sum 30
	//   (B, 00:00) row 2     -> count 1, sum 30
	//   (B, 02:00) row 3     -> count 1, sum 5
	//   (A, 04:00) rows 4,5,7 -> count 1, sum 7 (rows 4,7 val NULL)
	testing.expect(t, dataframe_num_rows(&out) == 4, "4 windows")
	testing.expect(t, dataframe_num_cols(&out) == 4, "sym + ts + cnt + sum")

	d0, _ := datetime_create(2024, 1, 1, 0, 0, 0)
	h2 := duration_from_hours(2)
	h4 := duration_from_hours(4)

	sym := dataframe_get_column(&out, "sym") or_else nil
	ts := dataframe_get_column(&out, "ts") or_else nil
	cnt := dataframe_get_column(&out, "cnt") or_else nil
	sum := dataframe_get_column(&out, "sum") or_else nil

	s, _, _ := column_get(sym, 0, string)
	tt, _, _ := column_get(ts, 0, Datetime)
	c, _, _ := column_get(cnt, 0, i64)
	v, _, _ := column_get(sum, 0, f64)
	testing.expect(t, s == "A" && tt == d0 && c == 2 && near(v, 30), "window 0 = (A, 00:00)")

	s, _, _ = column_get(sym, 1, string)
	tt, _, _ = column_get(ts, 1, Datetime)
	c, _, _ = column_get(cnt, 1, i64)
	v, _, _ = column_get(sum, 1, f64)
	testing.expect(t, s == "B" && tt == d0 && c == 1 && near(v, 30), "window 1 = (B, 00:00)")

	s, _, _ = column_get(sym, 2, string)
	tt, _, _ = column_get(ts, 2, Datetime)
	c, _, _ = column_get(cnt, 2, i64)
	v, _, _ = column_get(sum, 2, f64)
	testing.expect(t, s == "B" && tt == datetime_add(d0, h2) && c == 1 && near(v, 5), "window 2 = (B, 02:00)")

	s, _, _ = column_get(sym, 3, string)
	tt, _, _ = column_get(ts, 3, Datetime)
	c, _, _ = column_get(cnt, 3, i64)
	v, _, _ = column_get(sum, 3, f64)
	testing.expect(t, s == "A" && tt == datetime_add(d0, h4) && c == 1 && near(v, 7), "window 3 = (A, 04:00)")
}

@(test)
dynamic_groupby_test_no_by :: proc(t: ^testing.T) {
	df, ctx := dynamic_test_df(t)
	defer dynamic_test_destroy(t, &df, &ctx)

	out := dynamic_groupby_agg_ok(t, &df, "ts", {}, duration_from_hours(2), Duration(0), .Left, []^expr.Expr{
		expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "val")), "cnt"),
		expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "val")), "sum"),
	})
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 3, "3 windows (NULL ts dropped)")
	testing.expect(t, dataframe_num_cols(&out) == 3, "ts + cnt + sum")

	d0, _ := datetime_create(2024, 1, 1, 0, 0, 0)
	h2 := duration_from_hours(2)
	h4 := duration_from_hours(4)

	ts := dataframe_get_column(&out, "ts") or_else nil
	cnt := dataframe_get_column(&out, "cnt") or_else nil
	sum := dataframe_get_column(&out, "sum") or_else nil

	tt, _, _ := column_get(ts, 0, Datetime)
	c, _, _ := column_get(cnt, 0, i64)
	v, _, _ := column_get(sum, 0, f64)
	testing.expect(t, tt == d0 && c == 3 && near(v, 60), "00:00 -> 3 rows, sum 60")

	tt, _, _ = column_get(ts, 1, Datetime)
	c, _, _ = column_get(cnt, 1, i64)
	v, _, _ = column_get(sum, 1, f64)
	testing.expect(t, tt == datetime_add(d0, h2) && c == 1 && near(v, 5), "02:00 -> 1 row, sum 5")

	tt, _, _ = column_get(ts, 2, Datetime)
	c, _, _ = column_get(cnt, 2, i64)
	v, _, _ = column_get(sum, 2, f64)
	testing.expect(t, tt == datetime_add(d0, h4) && c == 1 && near(v, 7), "04:00 -> 1 row, sum 7")
}

@(test)
dynamic_groupby_test_closed :: proc(t: ^testing.T) {
	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	d0, _ := datetime_create(2024, 1, 1, 0, 0, 0)
	h2 := duration_from_hours(2)
	h4 := duration_from_hours(4)
	ts, err := column_from("ts", []Datetime{d0, datetime_add(d0, h2), datetime_add(d0, h4)})
	testing.expect(t, err == .None, "ts column")
	df: DataFrame
	df, err = dataframe_from_columns([]^Column{&ts})
	testing.expect(t, err == .None, "from_columns")
	defer dataframe_destroy(&df)

	// The three rows land exactly on boundaries; closed decides which window
	// keeps them.
	variants := []Closed_Interval{.Left, .Right, .Both, .None}
	for closed in variants {
		g, gerr := dataframe_group_by_dynamic(&df, "ts", {}, h2, Duration(0), closed)
		testing.expect(t, gerr == .None, "group_by_dynamic err")
		if gerr != .None {
			continue
		}
		out, aerr := dataframe_dynamic_group_by_agg(&g, []^expr.Expr{
			expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "ts")), "cnt"),
		})
		if aerr == .None {
			tsc := dataframe_get_column(&out, "ts") or_else nil
			cnt := dataframe_get_column(&out, "cnt") or_else nil
			switch closed {
			case .Left, .Both:
				// Boundary rows belong to the window starting at them.
				tt, _, _ := column_get(tsc, 0, Datetime)
				testing.expect(t, tt == d0, "first window starts at d0")
			case .Right, .None:
				// Boundary rows belong to the window ending at them.
				tt, _, _ := column_get(tsc, 0, Datetime)
				testing.expect(t, tt == datetime_add(d0, Duration(-i64(h2))), "first window starts at d0-2h")
			}
			for i in 0 ..< dataframe_num_rows(&out) {
				c, _, _ := column_get(cnt, i, i64)
				testing.expect(t, c == 1, "each window has 1 row")
			}
		}
		dataframe_destroy(&out)
		dataframe_dynamic_group_by_destroy(&g)
	}
}

@(test)
dynamic_groupby_test_pre_epoch :: proc(t: ^testing.T) {
	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	d0, _ := datetime_create(1970, 1, 1, 0, 0, 0)
	h1 := duration_from_hours(1)
	h2 := duration_from_hours(2)
	// -1h and -2h floor to window start -2h; +1h floors to window start 0.
	ts, err := column_from("ts", []Datetime{
		datetime_add(d0, Duration(-i64(h1))),
		datetime_add(d0, Duration(-i64(h2))),
		datetime_add(d0, h1),
	})
	testing.expect(t, err == .None, "ts column")
	df: DataFrame
	df, err = dataframe_from_columns([]^Column{&ts})
	testing.expect(t, err == .None, "from_columns")
	defer dataframe_destroy(&df)

	g, gerr := dataframe_group_by_dynamic(&df, "ts", {}, h2, Duration(0), .Left)
	testing.expect(t, gerr == .None, "group_by_dynamic err")
	defer dataframe_dynamic_group_by_destroy(&g)

	out, aerr := dataframe_dynamic_group_by_agg(&g, []^expr.Expr{
		expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "ts")), "cnt"),
	})
	testing.expect(t, aerr == .None, "agg err")
	defer dataframe_destroy(&out)

	tsc := dataframe_get_column(&out, "ts") or_else nil
	cnt := dataframe_get_column(&out, "cnt") or_else nil
	tt, _, _ := column_get(tsc, 0, Datetime)
	c, _, _ := column_get(cnt, 0, i64)
	testing.expect(t, tt == datetime_add(d0, Duration(-i64(h2))) && c == 2, "pre-epoch window -2h has 2 rows")
	tt, _, _ = column_get(tsc, 1, Datetime)
	c, _, _ = column_get(cnt, 1, i64)
	testing.expect(t, tt == d0 && c == 1, "epoch window 0 has 1 row")
}

@(test)
dynamic_groupby_test_null_by :: proc(t: ^testing.T) {
	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	d0, _ := datetime_create(2024, 1, 1, 0, 0, 0)
	h1 := duration_from_hours(1)
	ts, sym: Column
	err: Error
	ts, err = column_from("ts", []Datetime{d0, datetime_add(d0, h1)})
	testing.expect(t, err == .None, "ts column")
	sym, err = column_from("sym", []string{"A", "A"})
	testing.expect(t, err == .None, "sym column")
	testing.expect(t, column_set_valid(&sym, 1, false) == .None, "sym[1] NULL")
	df: DataFrame
	df, err = dataframe_from_columns([]^Column{&ts, &sym})
	testing.expect(t, err == .None, "from_columns")
	defer dataframe_destroy(&df)

	g, gerr := dataframe_group_by_dynamic(&df, "ts", {"sym"}, h1, Duration(0), .Left)
	testing.expect(t, gerr == .None, "group_by_dynamic err")
	defer dataframe_dynamic_group_by_destroy(&g)

	out, aerr := dataframe_dynamic_group_by_agg(&g, []^expr.Expr{
		expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "ts")), "cnt"),
	})
	testing.expect(t, aerr == .None, "agg err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 2, "2 windows (A and NULL)")
	symc := dataframe_get_column(&out, "sym") or_else nil
	cnt := dataframe_get_column(&out, "cnt") or_else nil
	s, _, _ := column_get(symc, 0, string)
	c, _, _ := column_get(cnt, 0, i64)
	testing.expect(t, s == "A" && c == 1, "A window count 1")
	testing.expect(t, !column_is_valid(symc, 1), "NULL by group keeps NULL key")
	c, _, _ = column_get(cnt, 1, i64)
	testing.expect(t, c == 1, "NULL by window count 1")
}

@(test)
dynamic_groupby_test_empty_df :: proc(t: ^testing.T) {
	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	ts, sym, val: Column
	err: Error
	ts, err = column_from("ts", []Datetime{})
	testing.expect(t, err == .None, "ts column")
	sym, err = column_from("sym", []string{})
	testing.expect(t, err == .None, "sym column")
	val, err = column_from("val", []f64{})
	testing.expect(t, err == .None, "val column")
	df: DataFrame
	df, err = dataframe_from_columns([]^Column{&ts, &sym, &val})
	testing.expect(t, err == .None, "from_columns")
	defer dataframe_destroy(&df)

	g, gerr := dataframe_group_by_dynamic(&df, "ts", {"sym"}, duration_from_hours(1), Duration(0), .Left)
	testing.expect(t, gerr == .None, "group_by_dynamic on empty df")
	defer dataframe_dynamic_group_by_destroy(&g)

	out, aerr := dataframe_dynamic_group_by_agg(&g, []^expr.Expr{
		expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "val")), "s"),
	})
	testing.expect(t, aerr == .None, "dynamic agg on empty df")
	defer dataframe_destroy(&out)

	// Schema preserved: 0 rows, sym + ts + s.
	testing.expect(t, dataframe_num_rows(&out) == 0, "0 rows")
	testing.expect(t, dataframe_num_cols(&out) == 3, "sym + ts + s")
	sc := dataframe_get_column(&out, "s") or_else nil
	testing.expect(t, sc.dtype == typeid_of(f64), "s dtype preserved")
}

// --- rolling groupby ---------------------------------------------------------

@(test)
rolling_groupby_test_basic :: proc(t: ^testing.T) {
	df, ctx := dynamic_test_df(t)
	defer dynamic_test_destroy(t, &df, &ctx)

	out := rolling_groupby_agg_ok(t, &df, "ts", {"sym"}, duration_from_hours(2), Duration(0), .Both, []^expr.Expr{
		expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "val")), "cnt"),
		expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "val")), "sum"),
	})
	defer dataframe_destroy(&out)

	// One row per source row, in source order; windows are per by-group.
	// Group A rows sorted by time: 0,1,4,5,7; group B: 2,3.
	//   row0 (A, 00:00) window {0}        -> count 1, sum 10
	//   row1 (A, 01:00) window {0,1}      -> count 2, sum 30
	//   row2 (B, 01:30) window {2}        -> count 1, sum 30
	//   row3 (B, 03:00) window {2,3}      -> count 2, sum 35
	//   row4 (A, 04:30) window {4}        -> count 0, sum NULL (val NULL)
	//   row5 (A, 05:00) window {4,5}      -> count 1, sum 7
	//   row6 (NULL ts)  window {}         -> count 0, sum NULL
	//   row7 (A, 05:30) window {4,5,7}    -> count 1, sum 7
	testing.expect(t, dataframe_num_rows(&out) == 8, "one row per source row")
	testing.expect(t, dataframe_num_cols(&out) == 4, "sym + ts + cnt + sum")

	cnt := dataframe_get_column(&out, "cnt") or_else nil
	sum := dataframe_get_column(&out, "sum") or_else nil
	tsc := dataframe_get_column(&out, "ts") or_else nil

	d0, _ := datetime_create(2024, 1, 1, 0, 0, 0)
	h1 := duration_from_hours(1)
	m30 := duration_from_minutes(30)

	// Source order and time values are preserved.
	sym := dataframe_get_column(&out, "sym") or_else nil
	s, _, _ := column_get(sym, 0, string)
	tt, _, _ := column_get(tsc, 0, Datetime)
	testing.expect(t, s == "A" && tt == d0, "row0 key/time preserved")
	s, _, _ = column_get(sym, 2, string)
	tt, _, _ = column_get(tsc, 2, Datetime)
	testing.expect(t, s == "B" && tt == datetime_add(d0, h1 + m30), "row2 key/time preserved")
	testing.expect(t, !column_is_valid(tsc, 6), "row6 time stays NULL")

	c, _, _ := column_get(cnt, 0, i64)
	v, _, _ := column_get(sum, 0, f64)
	testing.expect(t, c == 1 && near(v, 10), "row0 window")
	c, _, _ = column_get(cnt, 1, i64)
	v, _, _ = column_get(sum, 1, f64)
	testing.expect(t, c == 2 && near(v, 30), "row1 window")
	c, _, _ = column_get(cnt, 2, i64)
	v, _, _ = column_get(sum, 2, f64)
	testing.expect(t, c == 1 && near(v, 30), "row2 window")
	c, _, _ = column_get(cnt, 3, i64)
	v, _, _ = column_get(sum, 3, f64)
	testing.expect(t, c == 2 && near(v, 35), "row3 window")
	c, _, _ = column_get(cnt, 4, i64)
	testing.expect(t, c == 0 && !column_is_valid(sum, 4), "row4 window empty")
	c, _, _ = column_get(cnt, 5, i64)
	v, _, _ = column_get(sum, 5, f64)
	testing.expect(t, c == 1 && near(v, 7), "row5 window")
	c, _, _ = column_get(cnt, 6, i64)
	testing.expect(t, c == 0 && !column_is_valid(sum, 6), "row6 window empty")
	c, _, _ = column_get(cnt, 7, i64)
	v, _, _ = column_get(sum, 7, f64)
	testing.expect(t, c == 1 && near(v, 7), "row7 window")
}

@(test)
rolling_groupby_test_closed :: proc(t: ^testing.T) {
	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	d0, _ := datetime_create(2024, 1, 1, 0, 0, 0)
	h1 := duration_from_hours(1)
	h2 := duration_from_hours(2)
	ts, val: Column
	err: Error
	ts, err = column_from("ts", []Datetime{d0, datetime_add(d0, h1), datetime_add(d0, h2)})
	testing.expect(t, err == .None, "ts column")
	val, err = column_from("val", []f64{1, 2, 3})
	testing.expect(t, err == .None, "val column")
	df: DataFrame
	df, err = dataframe_from_columns([]^Column{&ts, &val})
	testing.expect(t, err == .None, "from_columns")
	defer dataframe_destroy(&df)

	// Row 2's raw window is [d0, d0+2h]; closed decides which endpoints count.
	expected := [Closed_Interval]int{.Both = 3, .Left = 2, .Right = 2, .None = 1}
	for closed in Closed_Interval {
		g, gerr := dataframe_group_by_rolling(&df, "ts", {}, h2, Duration(0), closed)
		testing.expect(t, gerr == .None, "group_by_rolling err")
		if gerr != .None {
			continue
		}
		out, aerr := dataframe_rolling_group_by_agg(&g, []^expr.Expr{
			expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "val")), "cnt"),
		})
		if aerr == .None {
			cnt := dataframe_get_column(&out, "cnt") or_else nil
			c, _, _ := column_get(cnt, 2, i64)
			testing.expect(t, c == i64(expected[closed]), "row2 window size per closed")
		}
		dataframe_destroy(&out)
		dataframe_rolling_group_by_destroy(&g)
	}
}

@(test)
rolling_groupby_test_null_by :: proc(t: ^testing.T) {
	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	d0, _ := datetime_create(2024, 1, 1, 0, 0, 0)
	h1 := duration_from_hours(1)
	ts, sym, val: Column
	err: Error
	ts, err = column_from("ts", []Datetime{d0, datetime_add(d0, h1)})
	testing.expect(t, err == .None, "ts column")
	sym, err = column_from("sym", []string{"A", "A"})
	testing.expect(t, err == .None, "sym column")
	testing.expect(t, column_set_valid(&sym, 1, false) == .None, "sym[1] NULL")
	val, err = column_from("val", []f64{1, 2})
	testing.expect(t, err == .None, "val column")
	df: DataFrame
	df, err = dataframe_from_columns([]^Column{&ts, &sym, &val})
	testing.expect(t, err == .None, "from_columns")
	defer dataframe_destroy(&df)

	out := rolling_groupby_agg_ok(t, &df, "ts", {"sym"}, h1, Duration(0), .Both, []^expr.Expr{
		expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "val")), "sum"),
	})
	defer dataframe_destroy(&out)

	// Each row's window only covers its own by-group: {0} and {1}.
	sum := dataframe_get_column(&out, "sum") or_else nil
	v, _, _ := column_get(sum, 0, f64)
	testing.expect(t, near(v, 1), "row0 sum 1 (A group only)")
	v, _, _ = column_get(sum, 1, f64)
	testing.expect(t, near(v, 2), "row1 sum 2 (NULL group only)")
}

// --- error cases -------------------------------------------------------------

@(test)
groupby_dynamic_test_errors :: proc(t: ^testing.T) {
	df, ctx := dynamic_test_df(t)
	defer dynamic_test_destroy(t, &df, &ctx)

	// Non-positive window sizes.
	_, err := dataframe_group_by_dynamic(&df, "ts", {}, Duration(0), Duration(0), .Left)
	testing.expect(t, err == .Invalid_Argument, "dynamic every=0 rejected")
	_, err = dataframe_group_by_rolling(&df, "ts", {}, Duration(-i64(duration_from_hours(1))), Duration(0), .Both)
	testing.expect(t, err == .Invalid_Argument, "rolling period<0 rejected")

	// Unknown time column.
	_, err = dataframe_group_by_dynamic(&df, "nope", {}, duration_from_hours(1), Duration(0), .Left)
	testing.expect(t, err == .Column_Not_Found, "dynamic unknown time col")

	// Non-datetime time column.
	_, err = dataframe_group_by_dynamic(&df, "val", {}, duration_from_hours(1), Duration(0), .Left)
	testing.expect(t, err == .Type_Mismatch, "dynamic non-datetime time col")
	_, err = dataframe_group_by_rolling(&df, "val", {}, duration_from_hours(1), Duration(0), .Both)
	testing.expect(t, err == .Type_Mismatch, "rolling non-datetime time col")

	// Empty aggs.
	g, gerr := dataframe_group_by_dynamic(&df, "ts", {}, duration_from_hours(2), Duration(0), .Left)
	testing.expect(t, gerr == .None, "group_by_dynamic ok")
	defer dataframe_dynamic_group_by_destroy(&g)
	_, aerr := dataframe_dynamic_group_by_agg(&g, nil)
	testing.expect(t, aerr == .Invalid_Argument, "dynamic empty aggs rejected")

	r, rerr := dataframe_group_by_rolling(&df, "ts", {}, duration_from_hours(2), Duration(0), .Both)
	testing.expect(t, rerr == .None, "group_by_rolling ok")
	defer dataframe_rolling_group_by_destroy(&r)
	_, aerr = dataframe_rolling_group_by_agg(&r, nil)
	testing.expect(t, aerr == .Invalid_Argument, "rolling empty aggs rejected")

	// Non-agg expression.
	_, aerr = dataframe_dynamic_group_by_agg(&g, []^expr.Expr{expr.col(&ctx, "val")})
	testing.expect(t, aerr == .Unsupported_Operation, "non-agg expression rejected")

	// Unknown agg column.
	_, aerr = dataframe_dynamic_group_by_agg(&g, []^expr.Expr{
		expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "nope")), "s"),
	})
	testing.expect(t, aerr == .Column_Not_Found, "unknown agg col rejected")

	// Duplicate agg names.
	_, aerr = dataframe_dynamic_group_by_agg(&g, []^expr.Expr{
		expr.count_(&ctx, expr.col(&ctx, "val")),
		expr.sum_(&ctx, expr.col(&ctx, "val")),
	})
	testing.expect(t, aerr == .Duplicate_Column_Name, "duplicate agg names rejected")

	// Agg name colliding with the time column name.
	_, aerr = dataframe_dynamic_group_by_agg(&g, []^expr.Expr{
		expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "val")), "ts"),
	})
	testing.expect(t, aerr == .Duplicate_Column_Name, "agg/time name collision rejected")
}
