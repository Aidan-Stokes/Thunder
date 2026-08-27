package dataframe

// Stage 14 reshaping tests (ROADMAP S14.1): dataframe_melt and
// dataframe_pivot correctness, NULL and ownership behavior, and error cases.

import "core:testing"

// melt_fixture builds a small wide frame:
//
//	id (i32)  [1, 2]
//	a  (i32)  [10, 20]
//	b  (i32)  [30, 40]
melt_fixture :: proc(t: ^testing.T) -> (df: DataFrame) {
	err: Error
	id, _ := column_from("id", []i32{1, 2})
	a, _ := column_from("a", []i32{10, 20})
	b, _ := column_from("b", []i32{30, 40})
	df, err = dataframe_from_columns([]^Column{&id, &a, &b})
	testing.expect(t, err == .None, "fixture build")
	return
}

// pivot_fixture builds:
//
//	item (string) [a, a, b]
//	day  (string) [mo, tu, mo]
//	val  (i64)    [1, 2, 3]
pivot_fixture :: proc(t: ^testing.T) -> (df: DataFrame) {
	err: Error
	item, _ := column_from("item", []string{"a", "a", "b"})
	day, _ := column_from("day", []string{"mo", "tu", "mo"})
	val, _ := column_from("val", []i64{1, 2, 3})
	df, err = dataframe_from_columns([]^Column{&item, &day, &val})
	testing.expect(t, err == .None, "fixture build")
	return
}

@test
reshape_melt_basic :: proc(t: ^testing.T) {
	df := melt_fixture(t)
	defer dataframe_destroy(&df)

	out, err := dataframe_melt(&df, []string{"id"}, []string{"a", "b"})
	testing.expect(t, err == .None, "melt err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_cols(&out) == 3, "3 output columns")
	testing.expect(t, dataframe_num_rows(&out) == 4, "4 output rows")

	id_col, _ := dataframe_get_column(&out, "id")
	var_col, _ := dataframe_get_column(&out, "variable")
	val_col, _ := dataframe_get_column(&out, "value")
	want_slice1 := []i32{1, 2, 1, 2}
	for want, i in want_slice1 {
		v, valid, _ := column_get(id_col, i, i32)
		testing.expect(t, valid && v == want, "id row")
	}
	want_slice2 := []string{"a", "a", "b", "b"}
	for want, i in want_slice2 {
		v, valid, _ := column_get(var_col, i, string)
		testing.expect(t, valid && v == want, "variable row")
	}
	want_slice3 := []i32{10, 20, 30, 40}
	for want, i in want_slice3 {
		v, valid, _ := column_get(val_col, i, i32)
		testing.expect(t, valid && v == want, "value row")
	}
}

@test
reshape_melt_default_value_vars :: proc(t: ^testing.T) {
	df := melt_fixture(t)
	defer dataframe_destroy(&df)

	out, err := dataframe_melt(&df, []string{"id"}, nil)
	testing.expect(t, err == .None, "melt err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_cols(&out) == 3, "3 output columns")
	val_col, _ := dataframe_get_column(&out, "value")
	want_slice4 := []i32{10, 20, 30, 40}
	for want, i in want_slice4 {
		v, valid, _ := column_get(val_col, i, i32)
		testing.expect(t, valid && v == want, "default value_vars")
	}
}

@test
reshape_melt_strings_and_nulls :: proc(t: ^testing.T) {
	id, _ := column_from("id", []i32{1, 2})
	s, _ := column_from("s", []string{"x", "y"})
	testing.expect(t, column_set_valid(&s, 1, false) == .None, "s[1] NULL")
	df, ferr := dataframe_from_columns([]^Column{&id, &s})
	testing.expect(t, ferr == .None, "frame build")
	defer dataframe_destroy(&df)

	out, err := dataframe_melt(&df, []string{"id"}, []string{"s"})
	testing.expect(t, err == .None, "melt err")
	defer dataframe_destroy(&out)

	val_col, _ := dataframe_get_column(&out, "value")
	v0, valid0, _ := column_get(val_col, 0, string)
	testing.expect(t, valid0 && v0 == "x", "row 0 value")
	_, valid1, _ := column_get(val_col, 1, string)
	testing.expect(t, !valid1, "row 1 NULL preserved")
}

@test
reshape_melt_owned_after_source_destroy :: proc(t: ^testing.T) {
	df := melt_fixture(t)
	out, err := dataframe_melt(&df, []string{"id"}, []string{"a", "b"})
	testing.expect(t, err == .None, "melt err")
	dataframe_destroy(&df)

	defer dataframe_destroy(&out)
	val_col, _ := dataframe_get_column(&out, "value")
	var_col, _ := dataframe_get_column(&out, "variable")
	v, valid, _ := column_get(val_col, 0, i32)
	testing.expect(t, valid && v == 10, "value survives source destroy")
	s, svalid, _ := column_get(var_col, 2, string)
	testing.expect(t, svalid && s == "b", "variable survives source destroy")
}

@test
reshape_melt_custom_names :: proc(t: ^testing.T) {
	df := melt_fixture(t)
	defer dataframe_destroy(&df)

	out, err := dataframe_melt(&df, []string{"id"}, []string{"a", "b"}, "var", "val")
	testing.expect(t, err == .None, "melt err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_has_column(&out, "var"), "custom variable name")
	testing.expect(t, dataframe_has_column(&out, "val"), "custom value name")
	testing.expect(t, !dataframe_has_column(&out, "value"), "no default value column")
}

@test
reshape_melt_error_cases :: proc(t: ^testing.T) {
	df := melt_fixture(t)
	defer dataframe_destroy(&df)

	// Mixed value dtypes.
	mixed, _ := dataframe_copy(&df)
	defer dataframe_destroy(&mixed)
	s, _ := column_from("s", []string{"x", "y"})
	testing.expect(t, dataframe_add_column(&mixed, &s) == .None, "add string col")
	_, err := dataframe_melt(&mixed, []string{"id"}, []string{"a", "s"})
	testing.expect(t, err == .Type_Mismatch, "mixed dtypes")

	// Unknown id and unknown value column.
	_, err = dataframe_melt(&df, []string{"nope"}, []string{"a"})
	testing.expect(t, err == .Column_Not_Found, "unknown id")
	_, err = dataframe_melt(&df, []string{"id"}, []string{"nope"})
	testing.expect(t, err == .Column_Not_Found, "unknown value var")

	// A column listed as both id and value.
	_, err = dataframe_melt(&df, []string{"id"}, []string{"id", "a"})
	testing.expect(t, err == .Invalid_Argument, "id/value overlap")

	// Empty value_vars with nothing else in the frame.
	id_only, _ := dataframe_copy(&df)
	defer dataframe_destroy(&id_only)
	testing.expect(t, dataframe_remove_column(&id_only, "a") == .None, "drop a")
	testing.expect(t, dataframe_remove_column(&id_only, "b") == .None, "drop b")
	_, err = dataframe_melt(&id_only, []string{"id"}, nil)
	testing.expect(t, err == .Invalid_Argument, "no value columns")

	// Name collisions.
	_, err = dataframe_melt(&df, []string{"id"}, []string{"a"}, "id", "value")
	testing.expect(t, err == .Invalid_Argument, "variable_name clashes with id")
}


@test
reshape_pivot_basic_sum :: proc(t: ^testing.T) {
	df := pivot_fixture(t)
	defer dataframe_destroy(&df)

	out, err := dataframe_pivot(&df, []string{"item"}, "day", "val", .Sum)
	testing.expect(t, err == .None, "pivot err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 2, "2 index groups")
	testing.expect(t, dataframe_num_cols(&out) == 3, "item + 2 pivot columns")

	item_col, _ := dataframe_get_column(&out, "item")
	want_slice5 := []string{"a", "b"}
	for want, i in want_slice5 {
		v, valid, _ := column_get(item_col, i, string)
		testing.expect(t, valid && v == want, "index key order")
	}
	mo, _ := dataframe_get_column(&out, "mo")
	tu, _ := dataframe_get_column(&out, "tu")
	am, av, _ := column_get(mo, 0, f64)
	testing.expect(t, av && am == 1.0, "a x mo = 1")
	at, atv, _ := column_get(tu, 0, f64)
	testing.expect(t, atv && at == 2.0, "a x tu = 2")
	bm, bv, _ := column_get(mo, 1, f64)
	testing.expect(t, bv && bm == 3.0, "b x mo = 3")
	_, bt, _ := column_get(tu, 1, f64)
	testing.expect(t, !bt, "b x tu has no rows -> NULL")
}

@test
reshape_pivot_empty_index :: proc(t: ^testing.T) {
	df := pivot_fixture(t)
	defer dataframe_destroy(&df)

	out, err := dataframe_pivot(&df, nil, "day", "val", .Sum)
	testing.expect(t, err == .None, "pivot err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 1, "single group")
	testing.expect(t, dataframe_num_cols(&out) == 2, "mo + tu only")
	mo, _ := dataframe_get_column(&out, "mo")
	tu, _ := dataframe_get_column(&out, "tu")
	m, mv, _ := column_get(mo, 0, f64)
	testing.expect(t, mv && m == 4.0, "mo sums rows 0+2 = 4")
	u, uv, _ := column_get(tu, 0, f64)
	testing.expect(t, uv && u == 2.0, "tu sums row 1 = 2")
}

@test
reshape_pivot_count_and_null_cell :: proc(t: ^testing.T) {
	df := pivot_fixture(t)
	defer dataframe_destroy(&df)

	out, err := dataframe_pivot(&df, []string{"item"}, "day", "val", .Count)
	testing.expect(t, err == .None, "pivot err")
	defer dataframe_destroy(&out)

	tu, _ := dataframe_get_column(&out, "tu")
	c0, v0, _ := column_get(tu, 0, i64)
	testing.expect(t, v0 && c0 == 1, "a x tu count 1")
	c1, v1, _ := column_get(tu, 1, i64)
	testing.expect(t, v1 && c1 == 0, "empty cell counts as 0")
}

@test
reshape_pivot_mean :: proc(t: ^testing.T) {
	item, _ := column_from("item", []string{"a", "a", "a"})
	day, _ := column_from("day", []string{"mo", "mo", "tu"})
	val, _ := column_from("val", []i64{10, 30, 5})
	df, ferr := dataframe_from_columns([]^Column{&item, &day, &val})
	testing.expect(t, ferr == .None, "frame build")
	defer dataframe_destroy(&df)

	out, err := dataframe_pivot(&df, []string{"item"}, "day", "val", .Mean)
	testing.expect(t, err == .None, "pivot err")
	defer dataframe_destroy(&out)

	mo, _ := dataframe_get_column(&out, "mo")
	m, mv, _ := column_get(mo, 0, f64)
	testing.expect(t, mv && m == 20.0, "mean of 10,30")
}

@test
reshape_pivot_multi_index :: proc(t: ^testing.T) {
	grp, _ := column_from("grp", []string{"g1", "g1", "g2"})
	item, _ := column_from("item", []string{"a", "b", "a"})
	day, _ := column_from("day", []string{"mo", "mo", "tu"})
	val, _ := column_from("val", []i64{1, 2, 3})
	df, ferr := dataframe_from_columns([]^Column{&grp, &item, &day, &val})
	testing.expect(t, ferr == .None, "frame build")
	defer dataframe_destroy(&df)

	out, err := dataframe_pivot(&df, []string{"grp", "item"}, "day", "val", .Sum)
	testing.expect(t, err == .None, "pivot err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 3, "3 index groups")
	testing.expect(t, dataframe_has_column(&out, "grp") && dataframe_has_column(&out, "item"), "index columns present")
	mo, _ := dataframe_get_column(&out, "mo")
	m, mv, _ := column_get(mo, 0, f64)
	testing.expect(t, mv && m == 1.0, "g1,a x mo = 1")
	m2, mv2, _ := column_get(mo, 1, f64)
	testing.expect(t, mv2 && m2 == 2.0, "g1,b x mo = 2")
	tu, _ := dataframe_get_column(&out, "tu")
	u, uv, _ := column_get(tu, 2, f64)
	testing.expect(t, uv && u == 3.0, "g2,a x tu = 3")
}

@test
reshape_pivot_null_columns_value :: proc(t: ^testing.T) {
	item, _ := column_from("item", []string{"a", "a", "a"})
	day, _ := column_from("day", []string{"mo", "tu", "tu"})
	testing.expect(t, column_set_valid(&day, 2, false) == .None, "day[2] NULL")
	val, _ := column_from("val", []i64{1, 2, 100})
	df, ferr := dataframe_from_columns([]^Column{&item, &day, &val})
	testing.expect(t, ferr == .None, "frame build")
	defer dataframe_destroy(&df)

	out, err := dataframe_pivot(&df, []string{"item"}, "day", "val", .Sum)
	testing.expect(t, err == .None, "pivot err")
	defer dataframe_destroy(&out)

	tu, _ := dataframe_get_column(&out, "tu")
	v, valid, _ := column_get(tu, 0, f64)
	testing.expect(t, valid && v == 2.0, "NULL columns row contributes nothing")
}

@test
reshape_pivot_column_name_collision :: proc(t: ^testing.T) {
	day, _ := column_from("day", []string{"x", "x"})
	cat, _ := column_from("cat", []string{"day", "mo"})
	val, _ := column_from("val", []i32{1, 2})
	df, ferr := dataframe_from_columns([]^Column{&day, &cat, &val})
	testing.expect(t, ferr == .None, "frame build")
	defer dataframe_destroy(&df)

	// The columns value "day" would collide with the index column name "day".
	_, err := dataframe_pivot(&df, []string{"day"}, "cat", "val", .Sum)
	testing.expect(t, err == .Invalid_Argument, "pivot name collides with index")
}

@test
reshape_pivot_error_cases :: proc(t: ^testing.T) {
	df := pivot_fixture(t)
	defer dataframe_destroy(&df)

	_, err := dataframe_pivot(&df, []string{"item"}, "nope", "val", .Sum)
	testing.expect(t, err == .Column_Not_Found, "unknown columns col")
	_, err = dataframe_pivot(&df, []string{"item"}, "day", "nope", .Sum)
	testing.expect(t, err == .Column_Not_Found, "unknown values col")
	_, err = dataframe_pivot(&df, []string{"item"}, "day", "day", .Sum)
	testing.expect(t, err == .Invalid_Argument, "columns == values")
	_, err = dataframe_pivot(&df, []string{"day"}, "day", "val", .Sum)
	testing.expect(t, err == .Invalid_Argument, "columns in index")

	// Sum on a string value column is not supported.
	s, _ := column_from("s", []string{"x", "y", "z"})
	err2 := dataframe_add_column(&df, &s)
	testing.expect(t, err2 == .None, "add string col")
	_, err = dataframe_pivot(&df, []string{"item"}, "day", "s", .Sum)
	testing.expect(t, err == .Unsupported_Operation, "Sum on strings rejected")
}
