package dataframe

import "core:testing"

// --- creation and basic CRUD -----------------------------------------------

@(test)
column_set_create_empty :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	testing.expect(t, cs.count == 0, "empty column set count")
	testing.expect(t, cs.names == nil || len(cs.names) == 0, "empty names")
}

@(test)
column_set_add_one_column :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	col, c_err := column_from("x", []i32{1, 2, 3})
	testing.expect(t, c_err == .None, "column_from")
	testing.expect(t, column_set_add(&cs, &col) == .None, "add")
	testing.expect(t, col.name == "", "source zeroed after add")
	testing.expect(t, cs.count == 1, "count after add")
	testing.expect(t, cs.names[0] == "x", "name stored")
	testing.expect(t, cs.dtypes[0] == typeid_of(i32), "dtype stored")
	testing.expect(t, cs.rows[0] == 3, "row count stored")
}

@(test)
column_set_add_multiple :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	c1, _ := column_from("a", []int{10, 20})
	column_set_add(&cs, &c1)
	c2, _ := column_from("b", []int{30, 40})
	column_set_add(&cs, &c2)
	c3, _ := column_from("c", []int{50, 60})
	column_set_add(&cs, &c3)
	testing.expect(t, cs.count == 3, "three columns")
}

@(test)
column_set_get_by_name :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	col, err1 := column_from("alpha", []i64{1, 2})
	testing.expect(t, err1 == .None, "alpha from")
	testing.expect(t, column_set_add(&cs, &col) == .None, "add alpha")

	col2, err2 := column_from("beta", []i64{3, 4})
	testing.expect(t, err2 == .None, "beta from")
	testing.expect(t, column_set_add(&cs, &col2) == .None, "add beta")

	idx, ok := column_set_get(&cs, "beta")
	testing.expect(t, ok, "get beta exists")
	testing.expect(t, cs.dtypes[idx] == typeid_of(i64), "beta is i64")

	idx2, ok2 := column_set_get(&cs, "alpha")
	testing.expect(t, ok2, "get alpha exists")
	testing.expect(t, cs.dtypes[idx2] == typeid_of(i64), "alpha is i64")

	_, ok3 := column_set_get(&cs, "missing")
	testing.expect(t, !ok3, "missing name")
}

@(test)
column_set_remove_entry :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	c1, _ := column_from("a", []i32{1, 2, 3})
	column_set_add(&cs, &c1)
	c2, _ := column_from("b", []i32{4, 5, 6})
	column_set_add(&cs, &c2)
	c3, _ := column_from("c", []i32{7, 8, 9})
	column_set_add(&cs, &c3)

	testing.expect(t, column_set_remove(&cs, 1) == .None, "remove middle")
	testing.expect(t, cs.count == 2, "count after remove")

	// Verify order preserved: a, c
	_, ok_a := column_set_get(&cs, "a")
	_, ok_b := column_set_get(&cs, "b")
	_, ok_c := column_set_get(&cs, "c")
	testing.expect(t, ok_a, "a still exists")
	testing.expect(t, !ok_b, "b removed")
	testing.expect(t, ok_c, "c still exists")

	// Verify data survived for a and c
	v0 := cs_typed_view(&cs, 0, i32)
	v1 := cs_typed_view(&cs, 1, i32)
	testing.expect(t, v0[0] == 1 && v1[0] == 7, "data intact")
}

@(test)
column_set_remove_out_of_bounds :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	testing.expect(t, column_set_remove(&cs, 0) == .Out_Of_Bounds, "oob remove")
}

@(test)
column_set_rename_entry :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	col, _ := column_from("old_name", []i32{1})
	column_set_add(&cs, &col)

	testing.expect(t, column_set_rename(&cs, 0, "new_name") == .None, "rename")
	_, ok := column_set_get(&cs, "new_name")
	testing.expect(t, ok, "find by new name")
	_, old := column_set_get(&cs, "old_name")
	testing.expect(t, !old, "old name gone")
	testing.expect(t, cs.names[0] == "new_name", "internal name updated")
}

@(test)
column_set_rename_empty_name :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	col, _ := column_from("x", []i32{1})
	column_set_add(&cs, &col)

	testing.expect(t, column_set_rename(&cs, 0, "") == .Column_Name_Empty, "empty name")
	testing.expect(t, column_set_rename(&cs, 0, "x") == .None, "same name ok")
}

@(test)
column_set_rename_dup_name :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	c1, _ := column_from("a", []i32{1})
	column_set_add(&cs, &c1)
	c2, _ := column_from("b", []i32{2})
	column_set_add(&cs, &c2)

	testing.expect(t, column_set_rename(&cs, 0, "b") == .Duplicate_Column_Name, "dup rename")
}

// --- typed view and data access --------------------------------------------

@(test)
column_set_typed_view :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	col, _ := column_from("vals", []f64{1.1, 2.2, 3.3})
	column_set_add(&cs, &col)

	view := cs_typed_view(&cs, 0, f64)
	testing.expect(t, len(view) == 3, "view len")
	testing.expect(t, view[0] == 1.1 && view[2] == 3.3, "view values")
}

@(test)
column_set_data_accessors :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	col, _ := column_from("x", []i32{42})
	column_set_add(&cs, &col)

	testing.expect(t, cs_name(&cs, 0) == "x", "cs_name")
	testing.expect(t, cs_dtype(&cs, 0) == typeid_of(i32), "cs_dtype")
	testing.expect(t, cs_data(&cs, 0) != nil, "cs_data non-nil")
}

// --- to_column round-trip --------------------------------------------------

@(test)
column_set_to_column_round_trip :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	col, _ := column_from_with_valid("test", []i32{10, 20, 30}, []bool{true, false, true})
	column_set_add(&cs, &col)

	view := column_set_to_column(&cs, 0)
	testing.expect(t, view.name == "test", "name round-trip")
	testing.expect(t, view.dtype == typeid_of(i32), "dtype round-trip")
	testing.expect(t, view.count == 3, "count round-trip")
	testing.expect(t, view.data != nil, "data round-trip")

	iv := column_typed_view(&view, i32)
	testing.expect(t, iv[0] == 10 && iv[1] == 20 && iv[2] == 30, "values round-trip")
	testing.expect(t, !row_valid(view.valid, 1), "null preserved")
}

// --- deep copy -------------------------------------------------------------

@(test)
column_set_copy_deep :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	c1, _ := column_from("a", []i32{1, 2, 3})
	column_set_add(&cs, &c1)
	c2, _ := column_from_with_valid("b", []f64{4.0, 5.0, 6.0}, []bool{true, false, true})
	column_set_add(&cs, &c2)

	cp, err := column_set_copy(&cs, context.allocator)
	testing.expect(t, err == .None, "copy ok")
	defer column_set_destroy(&cp)

	testing.expect(t, cp.count == 2, "copy count")
	testing.expect(t, cp.names[0] == "a" && cp.names[1] == "b", "copy names")

	// Data must be independent (modifying copy shouldn't affect original)
	v0 := cs_typed_view(&cp, 0, i32)
	testing.expect(t, v0[0] == 1 && v0[2] == 3, "copy data correct")

	v1 := cs_typed_view(&cp, 1, f64)
	testing.expect(t, v1[0] == 4.0, "copy f64 data")
	testing.expect(t, !row_valid(cp.valids[1], 1), "copy validity")

	// Modify original — copy must not change
	src_v0 := cs_typed_view(&cs, 0, i32)
	src_v0[0] = 99
	cp_v0 := cs_typed_view(&cp, 0, i32)
	testing.expect(t, cp_v0[0] == 1, "copy independent of source")
}

// --- add validation errors -------------------------------------------------

@(test)
column_set_add_empty_name :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	col, _ := column_from("", []i32{1})
	testing.expect(t, column_set_add(&cs, &col) == .Column_Name_Empty, "empty name rejected")
}

@(test)
column_set_add_duplicate_name :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	c1, _ := column_from("dup", []i32{1, 2})
	column_set_add(&cs, &c1)
	c2, _ := column_from("dup", []i32{3, 4})
	testing.expect(t, column_set_add(&cs, &c2) == .Duplicate_Column_Name, "dup rejected")
}

@(test)
column_set_add_length_mismatch :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	c1, _ := column_from("a", []i32{1, 2, 3})
	column_set_add(&cs, &c1)
	c2, _ := column_from("b", []i32{1, 2})
	testing.expect(t, column_set_add(&cs, &c2) == .Length_Mismatch, "length mismatch")
}

// --- empty column set edge cases -------------------------------------------

@(test)
column_set_empty_copy :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	cp, err := column_set_copy(&cs, context.allocator)
	testing.expect(t, err == .None && cp.count == 0, "empty copy")
	defer column_set_destroy(&cp)
}

@(test)
column_set_remove_all :: proc(t: ^testing.T) {
	cs := column_set_create(context.allocator)
	defer column_set_destroy(&cs)

	c1, _ := column_from("x", []i32{1})
	column_set_add(&cs, &c1)
	c2, _ := column_from("y", []i32{2})
	column_set_add(&cs, &c2)

	column_set_remove(&cs, 0)
	column_set_remove(&cs, 0)
	testing.expect(t, cs.count == 0, "all removed")
}
