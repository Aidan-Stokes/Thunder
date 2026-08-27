package dataframe

import "core:testing"

// --- creation and basic access ----------------------------------------------

@(test)
dataframe_empty_after_create :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	testing.expect(t, dataframe_num_rows(&df) == 0, "empty row count")
	testing.expect(t, dataframe_num_cols(&df) == 0, "empty column count")
	testing.expect(t, !dataframe_has_column(&df, "x"), "no columns present")

	_, g_err := dataframe_get_column(&df, "x")
	testing.expect(t, g_err == .Column_Not_Found, "get on empty df")
	_, a_err := dataframe_column_at(&df, 0)
	testing.expect(t, a_err == .Out_Of_Bounds, "column_at on empty df")

	schema, s_err := dataframe_schema(&df)
	defer schema_destroy(&schema)
	testing.expect(t, s_err == .None && schema_len(&schema) == 0, "empty schema")
}

@(test)
dataframe_add_multiple_types :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	age, a_err := column_from("age", []i32{25, 30, 35})
	testing.expect(t, a_err == .None, "age column")
	testing.expect(t, dataframe_add_column(&df, &age) == .None, "add age")

	name, n_err := column_from("name", []string{"ada", "grace", "katherine"})
	testing.expect(t, n_err == .None, "name column")
	testing.expect(t, dataframe_add_column(&df, &name) == .None, "add name")

	score, s_err := column_from_with_valid("score", []f64{9.5, 8.0, 7.25}, []bool{true, false, true})
	testing.expect(t, s_err == .None, "score column")
	testing.expect(t, dataframe_add_column(&df, &score) == .None, "add score")

	testing.expect(t, dataframe_num_cols(&df) == 3, "3 columns")
	testing.expect(t, dataframe_num_rows(&df) == 3, "3 rows")
	testing.expect(t, dataframe_has_column(&df, "age"), "has age")
	testing.expect(t, dataframe_has_column(&df, "name"), "has name")
	testing.expect(t, dataframe_has_column(&df, "score"), "has score")

	// get by name, in insertion order via column_at
	col, g_err := dataframe_get_column(&df, "score")
	testing.expect(t, g_err == .None && column_name(col) == "score", "get score")
	testing.expect(t, column_dtype(col) == typeid_of(f64), "score dtype")

	c0, at_err := dataframe_column_at(&df, 0)
	testing.expect(t, at_err == .None && column_name(c0) == "age", "column 0 is age")
	c2, at_err2 := dataframe_column_at(&df, 2)
	testing.expect(t, at_err2 == .None && column_name(c2) == "score", "column 2 is score")

	// typed values, including NULL preserved
	v, ok, _ := column_get(c0, 1, i32)
	testing.expect(t, ok && v == 30, "age value")
	vf, okf, _ := column_get(c2, 0, f64)
	testing.expect(t, okf && vf == 9.5, "score value")
	_, okf, _ = column_get(c2, 1, f64)
	testing.expect(t, !okf, "NULL row preserved")
}

@(test)
dataframe_add_column_transfers_ownership :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("x", []i32{1, 2, 3})
	testing.expect(t, err == .None, "constructor")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add column")

	// source is zeroed; destroying it is a safe no-op
	testing.expect(t, col.name == "" && col.data == nil && col.valid == nil, "source zeroed")
	column_destroy(&col)

	// the df owns the data
	got, g_err := dataframe_get_column(&df, "x")
	testing.expect(t, g_err == .None, "get after transfer")
	v, ok, _ := column_get(got, 2, i32)
	testing.expect(t, ok && v == 3, "value intact after transfer")
}

@(test)
dataframe_add_rejects_zeroed_or_reused_column :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("x", []i32{1})
	testing.expect(t, err == .None, "constructor")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "first add")

	// transferring the same (now zeroed) column again must fail loudly
	testing.expect(t, dataframe_add_column(&df, &col) == .Column_Name_Empty, "reused column rejected")
}

@(test)
dataframe_add_validation_errors_leave_source_intact :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	// length mismatch
	short, err := column_from("short", []i32{1, 2})
	testing.expect(t, err == .None, "short column")
	testing.expect(t, dataframe_add_column(&df, &short) == .None, "add short (first)")

	long, l_err := column_from("long", []i64{1, 2, 3})
	testing.expect(t, l_err == .None, "long column")
	testing.expect(t, dataframe_add_column(&df, &long) == .Length_Mismatch, "length mismatch")
	testing.expect(t, long.name == "long" && column_len(&long) == 3, "source intact on length mismatch")
	column_destroy(&long)

	// duplicate name
	dup, d_err := column_from("short", []i32{7})
	testing.expect(t, d_err == .None, "dup column")
	testing.expect(t, dataframe_add_column(&df, &dup) == .Duplicate_Column_Name, "duplicate name")
	testing.expect(t, dup.name == "short" && column_len(&dup) == 1, "source intact on duplicate")
	column_destroy(&dup)

	// empty name
	no_name, n_err := column_from("", []i32{7})
	testing.expect(t, n_err == .None, "no-name column")
	testing.expect(t, dataframe_add_column(&df, &no_name) == .Column_Name_Empty, "empty name")
	testing.expect(t, column_len(&no_name) == 1, "source intact on empty name")
	column_destroy(&no_name)
}

// --- remove and rename ------------------------------------------------------

@(test)
dataframe_remove_column_basic :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	a, _ := column_from("a", []i32{1, 2})
	b, _ := column_from("b", []i32{3, 4})
	c, _ := column_from("c", []i32{5, 6})
	testing.expect(t, dataframe_add_column(&df, &a) == .None, "add a")
	testing.expect(t, dataframe_add_column(&df, &b) == .None, "add b")
	testing.expect(t, dataframe_add_column(&df, &c) == .None, "add c")

	testing.expect(t, dataframe_remove_column(&df, "b") == .None, "remove b")
	testing.expect(t, dataframe_num_cols(&df) == 2, "2 columns left")
	testing.expect(t, !dataframe_has_column(&df, "b"), "b gone")

	// order preserved
	c0, _ := dataframe_column_at(&df, 0)
	c1, _ := dataframe_column_at(&df, 1)
	testing.expect(t, column_name(c0) == "a" && column_name(c1) == "c", "order preserved")
	testing.expect(t, dataframe_num_rows(&df) == 2, "rows intact")

	// removing a missing column errors and changes nothing
	testing.expect(t, dataframe_remove_column(&df, "nope") == .Column_Not_Found, "remove missing")
	testing.expect(t, dataframe_num_cols(&df) == 2, "unchanged after failed remove")

	// remove all
	testing.expect(t, dataframe_remove_column(&df, "a") == .None, "remove a")
	testing.expect(t, dataframe_remove_column(&df, "c") == .None, "remove c")
	testing.expect(t, dataframe_num_cols(&df) == 0, "empty again")
	testing.expect(t, dataframe_num_rows(&df) == 0, "0 rows")
}

@(test)
dataframe_rename_column_basic :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("old", []i32{1, 2, 3})
	testing.expect(t, err == .None, "constructor")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add column")

	testing.expect(t, dataframe_rename_column(&df, "old", "new") == .None, "rename ok")
	testing.expect(t, !dataframe_has_column(&df, "old"), "old name gone")
	testing.expect(t, dataframe_has_column(&df, "new"), "new name present")
	got, g_err := dataframe_get_column(&df, "new")
	testing.expect(t, g_err == .None, "get renamed")
	v, ok, _ := column_get(got, 0, i32)
	testing.expect(t, ok && v == 1, "data intact after rename")

	// rename to its own name is a no-op
	testing.expect(t, dataframe_rename_column(&df, "new", "new") == .None, "self rename")

	// errors
	testing.expect(t, dataframe_rename_column(&df, "nope", "x") == .Column_Not_Found, "rename missing")
	testing.expect(t, dataframe_rename_column(&df, "new", "") == .Column_Name_Empty, "rename to empty")

	other, o_err := column_from("other", []i32{9, 9, 9})
	testing.expect(t, o_err == .None, "other column")
	testing.expect(t, dataframe_add_column(&df, &other) == .None, "add other")
	testing.expect(t, dataframe_rename_column(&df, "new", "other") == .Duplicate_Column_Name, "rename to existing")
}

// --- schema -----------------------------------------------------------------

@(test)
dataframe_schema_matches_columns :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	a, _ := column_from("age", []i32{1})
	s, _ := column_from("score", []f64{2.0})
	testing.expect(t, dataframe_add_column(&df, &a) == .None, "add age")
	testing.expect(t, dataframe_add_column(&df, &s) == .None, "add score")

	schema, s_err := dataframe_schema(&df)
	defer schema_destroy(&schema)
	testing.expect(t, s_err == .None && schema_len(&schema) == 2, "schema length")

	f0, f0_err := schema_field_at(&schema, 0)
	testing.expect(t, f0_err == .None && f0.name == "age" && f0.dtype == typeid_of(i32), "field 0")
	f1, f1_err := schema_field_at(&schema, 1)
	testing.expect(t, f1_err == .None && f1.name == "score" && f1.dtype == typeid_of(f64), "field 1")
}

// --- copy -------------------------------------------------------------------

@(test)
dataframe_copy_is_deep :: proc(t: ^testing.T) {
	src := dataframe_create(context.allocator)
	defer dataframe_destroy(&src)

	a, a_err := column_from_with_valid("a", []i32{1, 2, 3}, []bool{true, false, true})
	testing.expect(t, a_err == .None, "a column")
	testing.expect(t, dataframe_add_column(&src, &a) == .None, "add a")

	dst, err := dataframe_copy(&src, context.allocator)
	defer dataframe_destroy(&dst)
	testing.expect(t, err == .None, "copy returned error")
	testing.expect(t, dataframe_num_cols(&dst) == 1 && dataframe_num_rows(&dst) == 3, "copy dims")

	d_col, _ := dataframe_get_column(&dst, "a")
	s_col, _ := dataframe_get_column(&src, "a")

	// NULL preserved in copy
	_, ok, _ := column_get(d_col, 1, i32)
	testing.expect(t, !ok, "NULL preserved in copy")

	// independent buffers
	testing.expect(t, column_set(d_col, 0, i32(99)) == .None, "mutate copy")
	v, okv, _ := column_get(s_col, 0, i32)
	testing.expect(t, okv && v == 1, "original untouched")
	testing.expect(t, column_set_null(d_col, 2) == .None, "NULL in copy only")
	_, okv, _ = column_get(s_col, 2, i32)
	testing.expect(t, okv, "original row 2 still valid")
}

@(test)
dataframe_copy_renaming_copy_does_not_affect_source :: proc(t: ^testing.T) {
	src := dataframe_create(context.allocator)
	defer dataframe_destroy(&src)

	a, _ := column_from("a", []i32{1})
	testing.expect(t, dataframe_add_column(&src, &a) == .None, "add a")

	dst, err := dataframe_copy(&src, context.allocator)
	defer dataframe_destroy(&dst)
	testing.expect(t, err == .None, "copy")

	testing.expect(t, dataframe_rename_column(&dst, "a", "b") == .None, "rename copy")
	testing.expect(t, dataframe_has_column(&src, "a"), "source name intact")
	testing.expect(t, !dataframe_has_column(&dst, "a"), "copy renamed")
}

// --- from_columns -----------------------------------------------------------

@(test)
dataframe_from_columns_transfers :: proc(t: ^testing.T) {
	a, a_err := column_from("a", []i32{1, 2})
	testing.expect(t, a_err == .None, "a")
	b, b_err := column_from_with_valid("b", []string{"x", ""}, []bool{true, false})
	testing.expect(t, b_err == .None, "b")

	df, err := dataframe_from_columns([]^Column{&a, &b})
	defer dataframe_destroy(&df)
	testing.expect(t, err == .None, "from_columns")

	// sources zeroed
	testing.expect(t, a.name == "" && a.data == nil, "a zeroed")
	testing.expect(t, b.name == "" && b.data == nil, "b zeroed")

	// df has the data
	testing.expect(t, dataframe_num_cols(&df) == 2 && dataframe_num_rows(&df) == 2, "dims")
	col_b, g_err := dataframe_get_column(&df, "b")
	testing.expect(t, g_err == .None, "get b")
	_, ok, _ := column_get(col_b, 1, string)
	testing.expect(t, !ok, "NULL preserved")
	testing.expect(t, column_dtype(col_b) == typeid_of(string), "b dtype")
}

@(test)
dataframe_from_columns_empty :: proc(t: ^testing.T) {
	df, err := dataframe_from_columns(nil)
	defer dataframe_destroy(&df)
	testing.expect(t, err == .None, "empty from_columns")
	testing.expect(t, dataframe_num_cols(&df) == 0 && dataframe_num_rows(&df) == 0, "empty dims")
}

@(test)
dataframe_from_columns_validation_consumes_nothing :: proc(t: ^testing.T) {
	// duplicate names
	a, a_err := column_from("dup", []i32{1})
	testing.expect(t, a_err == .None, "a")
	b, b_err := column_from("dup", []i32{2})
	testing.expect(t, b_err == .None, "b")
	_, err := dataframe_from_columns([]^Column{&a, &b})
	testing.expect(t, err == .Duplicate_Column_Name, "duplicate names rejected")
	testing.expect(t, a.name == "dup" && b.name == "dup", "sources intact on duplicate")
	column_destroy(&a)
	column_destroy(&b)

	// length mismatch
	c, c_err := column_from("c", []i32{1})
	testing.expect(t, c_err == .None, "c")
	d, d_err := column_from("d", []i32{1, 2})
	testing.expect(t, d_err == .None, "d")
	_, err = dataframe_from_columns([]^Column{&c, &d})
	testing.expect(t, err == .Length_Mismatch, "length mismatch rejected")
	testing.expect(t, column_len(&c) == 1 && column_len(&d) == 2, "sources intact on mismatch")
	column_destroy(&c)
	column_destroy(&d)

	// empty name
	e, e_err := column_from("", []i32{1})
	testing.expect(t, e_err == .None, "e")
	_, err = dataframe_from_columns([]^Column{&e})
	testing.expect(t, err == .Column_Name_Empty, "empty name rejected")
	testing.expect(t, column_len(&e) == 1, "source intact on empty name")
	column_destroy(&e)
}

// --- NULL / empty-string / single-row boundary ------------------------------

@(test)
dataframe_null_and_empty_string_distinct :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from_with_valid("s", []string{"", "hello"}, []bool{false, true})
	testing.expect(t, err == .None, "column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add")

	got, g_err := dataframe_get_column(&df, "s")
	testing.expect(t, g_err == .None, "get")
	s0, ok0, _ := column_get(got, 0, string)
	testing.expect(t, !ok0, "NULL is not a value")
	_ = s0
	s1, ok1, _ := column_get(got, 1, string)
	testing.expect(t, ok1 && s1 == "hello", "valid string intact")
}

@(test)
dataframe_single_row :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("one", []bool{true})
	testing.expect(t, err == .None, "column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add")
	testing.expect(t, dataframe_num_rows(&df) == 1, "one row")
	testing.expect(t, dataframe_num_cols(&df) == 1, "one col")
}

@(test)
dataframe_add_column_sets_row_count_from_first :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	first, err := column_from("first", []f64{1.0, 2.0, 3.0, 4.0})
	testing.expect(t, err == .None, "first column")
	testing.expect(t, dataframe_add_column(&df, &first) == .None, "add first")
	testing.expect(t, dataframe_num_rows(&df) == 4, "row count from first column")

	second, s_err := column_from("second", []i32{1, 2, 3})
	testing.expect(t, s_err == .None, "second column")
	testing.expect(t, dataframe_add_column(&df, &second) == .Length_Mismatch, "second must match 4 rows")
	column_destroy(&second)
}
