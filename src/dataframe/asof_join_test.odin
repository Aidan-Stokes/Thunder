package dataframe

// Stage 14.6 as-of join tests (DESIGN.md §18.3): Backward/Forward matching,
// exact `by` grouping, NULL never matches, on-tie resolution, string/Date
// `on` columns, empty sides, the left-major output shape with the `_right`
// collision rule, and error cases.

import "core:testing"

// asof_fixture builds the shared price/quotes fixture.
//
//	left (prices): ts [1, 2, 5, 9, 12], px [10.5, 11.0, 12.5, 14.0, 15.5]
//	right (quotes): ts [2, 4, 8, 10, 12, 12], bid [100, 101, 102, 103, 104, 105]
//
// Backward (greatest right ts <= left ts):
//   ts=1 -> none (NULL), ts=2 -> 100, ts=5 -> 101, ts=9 -> 102, ts=12 -> 105
//   (tie on ts=12 resolves to the last row).
// Forward (smallest right ts >= left ts):
//   ts=1 -> 100, ts=2 -> 100, ts=5 -> 102, ts=9 -> 103, ts=12 -> 104
//   (tie on ts=12 resolves to the first row).
asof_fixture :: proc(t: ^testing.T) -> (left: DataFrame, right: DataFrame) {
	err: Error
	ts, px, rts, bid: Column
	ts, err = column_from("ts", []i64{1, 2, 5, 9, 12})
	testing.expect(t, err == .None, "left ts")
	px, err = column_from("px", []f64{10.5, 11.0, 12.5, 14.0, 15.5})
	testing.expect(t, err == .None, "left px")
	left, err = dataframe_from_columns([]^Column{&ts, &px})
	testing.expect(t, err == .None, "left from_columns")

	rts, err = column_from("ts", []i64{2, 4, 8, 10, 12, 12})
	testing.expect(t, err == .None, "right ts")
	bid, err = column_from("bid", []f64{100, 101, 102, 103, 104, 105})
	testing.expect(t, err == .None, "right bid")
	right, err = dataframe_from_columns([]^Column{&rts, &bid})
	testing.expect(t, err == .None, "right from_columns")
	return
}

// asof_col reads column name from out at row as T and expects it to equal want.
asof_col :: proc(t: ^testing.T, out: ^DataFrame, name: string, row: int, want: $T) {
	c := dataframe_get_column(out, name) or_else nil
	testing.expect(t, c != nil, "column present")
	if c == nil {
		return
	}
	v, valid, _ := column_get(c, row, T)
	testing.expect(t, valid && v == want, "row value")
}

// --- Backward ---------------------------------------------------------------

@(test)
asof_backward_basic :: proc(t: ^testing.T) {
	left, right := asof_fixture(t)
	defer dataframe_destroy(&left)
	defer dataframe_destroy(&right)

	out, err := dataframe_asof_join(&left, &right, "ts", "ts", nil, nil, .Backward)
	testing.expect(t, err == .None, "asof err")
	defer dataframe_destroy(&out)

	// Schema: ts (left), px, bid (right `ts` dropped).
	testing.expect(t, dataframe_num_rows(&out) == 5, "5 rows")
	testing.expect(t, dataframe_num_cols(&out) == 3, "ts + px + bid")
	testing.expect(t, !dataframe_has_column(&out, "ts_right"), "right on dropped")

	asof_col(t, &out, "px", 0, f64(10.5))
	asof_col(t, &out, "px", 4, f64(15.5))

	bid := dataframe_get_column(&out, "bid") or_else nil
	v, valid, _ := column_get(bid, 0, f64)
	testing.expect(t, !valid, "row0 unmatched (ts=1)")
	v, valid, _ = column_get(bid, 1, f64)
	testing.expect(t, valid && near(v, 100), "row1 bid = 100")
	v, _, _ = column_get(bid, 2, f64)
	testing.expect(t, near(v, 101), "row2 bid = 101")
	v, _, _ = column_get(bid, 3, f64)
	testing.expect(t, near(v, 102), "row3 bid = 102")
	v, _, _ = column_get(bid, 4, f64)
	testing.expect(t, near(v, 105), "row4 bid = 105 (last ts=12)")
}

// --- Forward ----------------------------------------------------------------

@(test)
asof_forward_basic :: proc(t: ^testing.T) {
	left, right := asof_fixture(t)
	defer dataframe_destroy(&left)
	defer dataframe_destroy(&right)

	out, err := dataframe_asof_join(&left, &right, "ts", "ts", nil, nil, .Forward)
	testing.expect(t, err == .None, "asof err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 5, "5 rows")
	bid := dataframe_get_column(&out, "bid") or_else nil
	v, _, _ := column_get(bid, 0, f64)
	testing.expect(t, near(v, 100), "row0 bid = 100")
	v, _, _ = column_get(bid, 1, f64)
	testing.expect(t, near(v, 100), "row1 bid = 100 (ts=2 ties)")
	v, _, _ = column_get(bid, 2, f64)
	testing.expect(t, near(v, 102), "row2 bid = 102")
	v, _, _ = column_get(bid, 3, f64)
	testing.expect(t, near(v, 103), "row3 bid = 103")
	v, _, _ = column_get(bid, 4, f64)
	testing.expect(t, near(v, 104), "row4 bid = 104 (first ts=12)")
}

// --- by grouping ------------------------------------------------------------

@(test)
asof_by_group :: proc(t: ^testing.T) {
	err: Error
	lsym, lts, lv, rsym, rts, rv: Column
	lsym, err = column_from("sym", []string{"A", "A", "B", "B"})
	testing.expect(t, err == .None, "left sym")
	lts, err = column_from("ts", []i64{1, 5, 2, 8})
	testing.expect(t, err == .None, "left ts")
	lv, err = column_from("lv", []string{"a1", "a5", "b2", "b8"})
	testing.expect(t, err == .None, "left lv")
	left, l_err := dataframe_from_columns([]^Column{&lsym, &lts, &lv})
	testing.expect(t, l_err == .None, "left df")

	rsym, err = column_from("sym", []string{"A", "A", "A", "B", "B"})
	testing.expect(t, err == .None, "right sym")
	rts, err = column_from("ts", []i64{0, 4, 10, 1, 9})
	testing.expect(t, err == .None, "right ts")
	rv, err = column_from("rv", []i64{10, 40, 100, 1, 90})
	testing.expect(t, err == .None, "right rv")
	right, r_err := dataframe_from_columns([]^Column{&rsym, &rts, &rv})
	testing.expect(t, r_err == .None, "right df")

	defer dataframe_destroy(&left)
	defer dataframe_destroy(&right)

	out, j_err := dataframe_asof_join(&left, &right, "ts", "ts", []string{"sym"}, []string{"sym"}, .Backward)
	testing.expect(t, j_err == .None, "asof err")
	defer dataframe_destroy(&out)

	// (A,1)->A:0 (10), (A,5)->A:4 (40), (B,2)->B:1 (1), (B,8)->B:1 (1).
	testing.expect(t, dataframe_num_rows(&out) == 4, "4 rows")
	// The right `sym` and `ts` are dropped; only `rv` is carried.
	testing.expect(t, dataframe_num_cols(&out) == 4, "sym + ts + lv + rv")
	testing.expect(t, !dataframe_has_column(&out, "sym_right"), "right by dropped")
	rv_col := dataframe_get_column(&out, "rv") or_else nil
	v, _, _ := column_get(rv_col, 0, i64)
	testing.expect(t, v == 10, "row0 rv = 10")
	v, _, _ = column_get(rv_col, 1, i64)
	testing.expect(t, v == 40, "row1 rv = 40")
	v, _, _ = column_get(rv_col, 2, i64)
	testing.expect(t, v == 1, "row2 rv = 1")
	v, _, _ = column_get(rv_col, 3, i64)
	testing.expect(t, v == 1, "row3 rv = 1")
}

// --- multi-column by --------------------------------------------------------

@(test)
asof_multi_by :: proc(t: ^testing.T) {
	err: Error
	lsym, lexch, lts, rsym, rexch, rts, rv: Column
	lsym, err = column_from("sym", []string{"A", "A", "B"})
	testing.expect(t, err == .None, "left sym")
	lexch, err = column_from("exch", []string{"X", "Y", "X"})
	testing.expect(t, err == .None, "left exch")
	lts, err = column_from("ts", []i64{3, 3, 3})
	testing.expect(t, err == .None, "left ts")
	left, l_err := dataframe_from_columns([]^Column{&lsym, &lexch, &lts})
	testing.expect(t, l_err == .None, "left df")

	rsym, err = column_from("sym", []string{"A", "A", "A", "B", "B"})
	testing.expect(t, err == .None, "right sym")
	rexch, err = column_from("exch", []string{"X", "X", "Y", "X", "Y"})
	testing.expect(t, err == .None, "right exch")
	rts, err = column_from("ts", []i64{1, 2, 2, 2, 4})
	testing.expect(t, err == .None, "right ts")
	rv, err = column_from("rv", []i64{10, 20, 30, 40, 50})
	testing.expect(t, err == .None, "right rv")
	right, r_err := dataframe_from_columns([]^Column{&rsym, &rexch, &rts, &rv})
	testing.expect(t, r_err == .None, "right df")

	defer dataframe_destroy(&left)
	defer dataframe_destroy(&right)

	out, j_err := dataframe_asof_join(&left, &right, "ts", "ts", []string{"sym", "exch"}, []string{"sym", "exch"}, .Backward)
	testing.expect(t, j_err == .None, "asof err")
	defer dataframe_destroy(&out)

	// (A,X,3)->(A,X) greatest ts<=3 = 2 (20); (A,Y,3)->30; (B,X,3)->40.
	rv_col := dataframe_get_column(&out, "rv") or_else nil
	v, _, _ := column_get(rv_col, 0, i64)
	testing.expect(t, v == 20, "row0 rv = 20")
	v, _, _ = column_get(rv_col, 1, i64)
	testing.expect(t, v == 30, "row1 rv = 30")
	v, _, _ = column_get(rv_col, 2, i64)
	testing.expect(t, v == 40, "row2 rv = 40")
}

// --- NULL semantics ---------------------------------------------------------

@(test)
asof_null_on_unmatched :: proc(t: ^testing.T) {
	err: Error
	ts, val: Column
	ts, err = column_from("ts", []i64{3, 5, 7})
	testing.expect(t, err == .None, "left ts")
	testing.expect(t, column_set_valid(&ts, 1, false) == .None, "ts[1] NULL")
	val, err = column_from("val", []i64{10, 20, 30})
	testing.expect(t, err == .None, "left val")
	left, l_err := dataframe_from_columns([]^Column{&ts, &val})
	testing.expect(t, l_err == .None, "left df")

	rts, rv: Column
	rts, err = column_from("ts", []i64{2, 4, 6})
	testing.expect(t, err == .None, "right ts")
	rv, err = column_from("rv", []i64{20, 40, 60})
	testing.expect(t, err == .None, "right rv")
	right, r_err := dataframe_from_columns([]^Column{&rts, &rv})
	testing.expect(t, r_err == .None, "right df")

	defer dataframe_destroy(&left)
	defer dataframe_destroy(&right)

	out, j_err := dataframe_asof_join(&left, &right, "ts", "ts", nil, nil, .Backward)
	testing.expect(t, j_err == .None, "asof err")
	defer dataframe_destroy(&out)

	rv_col := dataframe_get_column(&out, "rv") or_else nil
	v, valid, _ := column_get(rv_col, 0, i64)
	testing.expect(t, valid && v == 20, "row0 rv = 20")
	v, valid, _ = column_get(rv_col, 1, i64)
	testing.expect(t, !valid, "row1 NULL on -> unmatched")
	v, valid, _ = column_get(rv_col, 2, i64)
	testing.expect(t, valid && v == 60, "row2 rv = 60")
}

@(test)
asof_null_by_unmatched :: proc(t: ^testing.T) {
	err: Error
	gsym, gts, gval, rsym, rts, rv: Column
	gsym, err = column_from("sym", []string{"A", "B"})
	testing.expect(t, err == .None, "left sym")
	testing.expect(t, column_set_valid(&gsym, 1, false) == .None, "sym[1] NULL")
	gts, err = column_from("ts", []i64{5, 5})
	testing.expect(t, err == .None, "left ts")
	gval, err = column_from("val", []i64{1, 2})
	testing.expect(t, err == .None, "left val")
	left, l_err := dataframe_from_columns([]^Column{&gsym, &gts, &gval})
	testing.expect(t, l_err == .None, "left df")

	rsym, err = column_from("sym", []string{"A", "A"})
	testing.expect(t, err == .None, "right sym")
	rts, err = column_from("ts", []i64{1, 9})
	testing.expect(t, err == .None, "right ts")
	rv, err = column_from("rv", []i64{10, 90})
	testing.expect(t, err == .None, "right rv")
	right, r_err := dataframe_from_columns([]^Column{&rsym, &rts, &rv})
	testing.expect(t, r_err == .None, "right df")

	defer dataframe_destroy(&left)
	defer dataframe_destroy(&right)

	out, j_err := dataframe_asof_join(&left, &right, "ts", "ts", []string{"sym"}, []string{"sym"}, .Backward)
	testing.expect(t, j_err == .None, "asof err")
	defer dataframe_destroy(&out)

	rv_col := dataframe_get_column(&out, "rv") or_else nil
	v, valid, _ := column_get(rv_col, 0, i64)
	testing.expect(t, valid && v == 10, "row0 rv = 10")
	v, valid, _ = column_get(rv_col, 1, i64)
	testing.expect(t, !valid, "row1 NULL by -> unmatched")
}

@(test)
asof_right_null_on_excluded :: proc(t: ^testing.T) {
	err: Error
	lts: Column
	lts, err = column_from("ts", []i64{4, 7})
	testing.expect(t, err == .None, "left ts")
	left, l_err := dataframe_from_columns([]^Column{&lts})
	testing.expect(t, l_err == .None, "left df")

	rts, rv: Column
	rts, err = column_from("ts", []i64{4, 6, 8})
	testing.expect(t, err == .None, "right ts")
	testing.expect(t, column_set_valid(&rts, 0, false) == .None, "right ts[0] NULL")
	rv, err = column_from("rv", []i64{40, 60, 80})
	testing.expect(t, err == .None, "right rv")
	right, r_err := dataframe_from_columns([]^Column{&rts, &rv})
	testing.expect(t, r_err == .None, "right df")

	defer dataframe_destroy(&left)
	defer dataframe_destroy(&right)

	out, j_err := dataframe_asof_join(&left, &right, "ts", "ts", nil, nil, .Backward)
	testing.expect(t, j_err == .None, "asof err")
	defer dataframe_destroy(&out)

	// ts=4 cannot match the NULL ts=4 right row (excluded); ts=7 -> the
	// greatest valid right ts <= 7 is 6 -> 60.
	rv_col := dataframe_get_column(&out, "rv") or_else nil
	v, valid, _ := column_get(rv_col, 0, i64)
	testing.expect(t, !valid, "row0 NULL right on excluded")
	v, valid, _ = column_get(rv_col, 1, i64)
	testing.expect(t, valid && v == 60, "row1 rv = 60")
}

// --- string and Date `on` ---------------------------------------------------

@(test)
asof_string_on :: proc(t: ^testing.T) {
	err: Error
	lk, lv, rk, rv: Column
	lk, err = column_from("k", []string{"b", "d", "a", "x"})
	testing.expect(t, err == .None, "left k")
	lv, err = column_from("lv", []i64{1, 2, 3, 4})
	testing.expect(t, err == .None, "left lv")
	left, l_err := dataframe_from_columns([]^Column{&lk, &lv})
	testing.expect(t, l_err == .None, "left df")

	rk, err = column_from("k", []string{"a", "c", "e"})
	testing.expect(t, err == .None, "right k")
	rv, err = column_from("rv", []i64{10, 30, 50})
	testing.expect(t, err == .None, "right rv")
	right, r_err := dataframe_from_columns([]^Column{&rk, &rv})
	testing.expect(t, r_err == .None, "right df")

	defer dataframe_destroy(&left)
	defer dataframe_destroy(&right)

	out, j_err := dataframe_asof_join(&left, &right, "k", "k", nil, nil, .Backward)
	testing.expect(t, j_err == .None, "asof backward err")
	defer dataframe_destroy(&out)

	// "b"->a (10), "d"->c (30), "a"->a (10), "x"->e (50).
	rv_col := dataframe_get_column(&out, "rv") or_else nil
	wants := []i64{10, 30, 10, 50}
	for want, i in wants {
		v, _, _ := column_get(rv_col, i, i64)
		testing.expect(t, v == want, "backward row")
	}
	dataframe_destroy(&out)

	out, j_err = dataframe_asof_join(&left, &right, "k", "k", nil, nil, .Forward)
	testing.expect(t, j_err == .None, "asof forward err")
	defer dataframe_destroy(&out)

	// "b"->c (30), "d"->e (50), "a"->a (10), "x"->none.
	rv_col = dataframe_get_column(&out, "rv") or_else nil
	wants = []i64{30, 50, 10, 0}
	for want, i in wants {
		v, valid, _ := column_get(rv_col, i, i64)
		testing.expect(t, want == 0 && !valid || valid && v == want, "forward row")
	}
}

@(test)
asof_date_on :: proc(t: ^testing.T) {
	err: Error
	ld, lv, rd, rv: Column
	ld, err = column_from("d", []Date{Date(1), Date(4), Date(9)})
	testing.expect(t, err == .None, "left d")
	lv, err = column_from("lv", []i64{1, 2, 3})
	testing.expect(t, err == .None, "left lv")
	left, l_err := dataframe_from_columns([]^Column{&ld, &lv})
	testing.expect(t, l_err == .None, "left df")

	rd, err = column_from("d", []Date{Date(2), Date(5), Date(7)})
	testing.expect(t, err == .None, "right d")
	rv, err = column_from("rv", []i64{20, 50, 70})
	testing.expect(t, err == .None, "right rv")
	right, r_err := dataframe_from_columns([]^Column{&rd, &rv})
	testing.expect(t, r_err == .None, "right df")

	defer dataframe_destroy(&left)
	defer dataframe_destroy(&right)

	out, j_err := dataframe_asof_join(&left, &right, "d", "d", nil, nil, .Backward)
	testing.expect(t, j_err == .None, "asof err")
	defer dataframe_destroy(&out)

	// Date(1)->none, Date(4)->2 (20), Date(9)->7 (70).
	rv_col := dataframe_get_column(&out, "rv") or_else nil
	v, valid, _ := column_get(rv_col, 0, i64)
	testing.expect(t, !valid, "row0 unmatched")
	v, valid, _ = column_get(rv_col, 1, i64)
	testing.expect(t, valid && v == 20, "row1 rv = 20")
	v, valid, _ = column_get(rv_col, 2, i64)
	testing.expect(t, valid && v == 70, "row2 rv = 70")
}

// --- output shape -----------------------------------------------------------

@(test)
asof_name_collision_suffix :: proc(t: ^testing.T) {
	err: Error
	lts, lpx, rts, rpx: Column
	lts, err = column_from("ts", []i64{1, 2})
	testing.expect(t, err == .None, "left ts")
	lpx, err = column_from("px", []f64{1.0, 2.0})
	testing.expect(t, err == .None, "left px")
	left, l_err := dataframe_from_columns([]^Column{&lts, &lpx})
	testing.expect(t, l_err == .None, "left df")

	rts, err = column_from("ts", []i64{2, 3})
	testing.expect(t, err == .None, "right ts")
	rpx, err = column_from("px", []f64{20.0, 30.0})
	testing.expect(t, err == .None, "right px")
	right, r_err := dataframe_from_columns([]^Column{&rts, &rpx})
	testing.expect(t, r_err == .None, "right df")

	defer dataframe_destroy(&left)
	defer dataframe_destroy(&right)

	out, j_err := dataframe_asof_join(&left, &right, "ts", "ts", nil, nil, .Backward)
	testing.expect(t, j_err == .None, "asof err")
	defer dataframe_destroy(&out)

	// Columns: ts, px, px_right. Right `ts` dropped; right `px` suffixed.
	testing.expect(t, dataframe_has_column(&out, "px_right"), "px_right exists")
	asof_col(t, &out, "px_right", 1, f64(20.0))
	c := dataframe_get_column(&out, "px_right") or_else nil
	_, valid, _ := column_get(c, 0, f64)
	testing.expect(t, !valid, "row0 px_right NULL")
}

@(test)
asof_empty_left :: proc(t: ^testing.T) {
	err: Error
	lts: Column
	lts, err = column_from("ts", []i64{})
	testing.expect(t, err == .None, "left ts")
	left, l_err := dataframe_from_columns([]^Column{&lts})
	testing.expect(t, l_err == .None, "left df")

	rts, rv: Column
	rts, err = column_from("ts", []i64{1, 2})
	testing.expect(t, err == .None, "right ts")
	rv, err = column_from("rv", []i64{10, 20})
	testing.expect(t, err == .None, "right rv")
	right, r_err := dataframe_from_columns([]^Column{&rts, &rv})
	testing.expect(t, r_err == .None, "right df")

	defer dataframe_destroy(&left)
	defer dataframe_destroy(&right)

	out, j_err := dataframe_asof_join(&left, &right, "ts", "ts", nil, nil, .Backward)
	testing.expect(t, j_err == .None, "asof err")
	defer dataframe_destroy(&out)

	// Schema with 0 rows: ts (left) + rv (right).
	testing.expect(t, dataframe_num_rows(&out) == 0, "0 rows")
	testing.expect(t, dataframe_num_cols(&out) == 2, "ts + rv")
}

@(test)
asof_no_by_single_partition :: proc(t: ^testing.T) {
	left, right := asof_fixture(t)
	defer dataframe_destroy(&left)
	defer dataframe_destroy(&right)

	// Explicit empty `by` = one partition over all rows (same as nil).
	out, err := dataframe_asof_join(&left, &right, "ts", "ts", []string{}, []string{}, .Backward)
	testing.expect(t, err == .None, "asof err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 5, "5 rows")
	bid := dataframe_get_column(&out, "bid") or_else nil
	v, valid, _ := column_get(bid, 1, f64)
	testing.expect(t, valid && near(v, 100), "row1 bid = 100")
}

// --- validation -------------------------------------------------------------

@(test)
asof_validates :: proc(t: ^testing.T) {
	left, right := asof_fixture(t)
	defer dataframe_destroy(&left)
	defer dataframe_destroy(&right)

	// Missing `on` column.
	_, err := dataframe_asof_join(&left, &right, "nope", "ts", nil, nil, .Backward)
	testing.expect(t, err == .Column_Not_Found, "missing left on")
	_, err = dataframe_asof_join(&left, &right, "ts", "nope", nil, nil, .Backward)
	testing.expect(t, err == .Column_Not_Found, "missing right on")

	// `on` dtype mismatch.
	rs, r_err := column_from("ts", []string{"a", "b"})
	testing.expect(t, r_err == .None, "right ts string")
	rdf, d_err := dataframe_from_columns([]^Column{&rs})
	testing.expect(t, d_err == .None, "right df string")
	defer dataframe_destroy(&rdf)
	_, err = dataframe_asof_join(&left, &rdf, "ts", "ts", nil, nil, .Backward)
	testing.expect(t, err == .Type_Mismatch, "on dtype mismatch")

	// `by` length mismatch.
	_, err = dataframe_asof_join(&left, &right, "ts", "ts", []string{"px"}, nil, .Backward)
	testing.expect(t, err == .Invalid_Argument, "by length mismatch")

	// Missing `by` column.
	_, err = dataframe_asof_join(&left, &right, "ts", "ts", []string{"nope"}, []string{"nope"}, .Backward)
	testing.expect(t, err == .Column_Not_Found, "missing by")

	// Duplicate `by` name.
	_, err = dataframe_asof_join(&left, &right, "ts", "ts", []string{"px", "px"}, []string{"px", "px"}, .Backward)
	testing.expect(t, err == .Invalid_Argument, "duplicate by")

	// Non-sortable `on` (List columns on both sides, so dtypes match).
	lx, lx_err := list_from_slices("xs", [][]i64{{1, 2}})
	testing.expect(t, lx_err == .None, "left list")
	ldf, ldf_err := dataframe_from_columns([]^Column{&lx})
	testing.expect(t, ldf_err == .None, "left df list")
	defer dataframe_destroy(&ldf)
	rx, rx_err := list_from_slices("xs", [][]i64{{1}})
	testing.expect(t, rx_err == .None, "right list")
	rlist, rlist_err := dataframe_from_columns([]^Column{&rx})
	testing.expect(t, rlist_err == .None, "right df list")
	defer dataframe_destroy(&rlist)
	_, err = dataframe_asof_join(&ldf, &rlist, "xs", "xs", nil, nil, .Backward)
	testing.expect(t, err == .Unsupported_Operation, "non-sortable on")
}
