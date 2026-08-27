package dataframe

import "core:testing"

// --- creation and accessors ------------------------------------------------

@(test)
column_create_each_dtype :: proc(t: ^testing.T) {
	i8s   := []i8{   -1, 0, 127 }
	i16s  := []i16{  -1, 0, 32767 }
	i32s  := []i32{  -1, 0, 2147483647 }
	i64s  := []i64{  -1, 0, 9223372036854775807 }
	u8s   := []u8{   0, 128, 255 }
	u16s  := []u16{  0, 128, 65535 }
	u32s  := []u32{  0, 128, 4294967295 }
	u64s  := []u64{  0, 128, 18446744073709551615 }
	f32s  := []f32{  -1.5, 0.0, 3.25 }
	f64s  := []f64{  -1.5, 0.0, 3.25 }
	bools := []bool{ true, false, true }
	strs  := []string{ "a", "", "ccc" }

	cols := make([dynamic]Column, context.allocator)
	defer {
		for &c in cols {
			column_destroy(&c)
		}
		delete(cols)
	}

	col: Column
	err: Error

	col, err = column_from("i8", i8s)
	testing.expect(t, err == .None, "i8 constructor")
	append(&cols, col)
	col, err = column_from("i16", i16s)
	testing.expect(t, err == .None, "i16 constructor")
	append(&cols, col)
	col, err = column_from("i32", i32s)
	testing.expect(t, err == .None, "i32 constructor")
	append(&cols, col)
	col, err = column_from("i64", i64s)
	testing.expect(t, err == .None, "i64 constructor")
	append(&cols, col)
	col, err = column_from("u8", u8s)
	testing.expect(t, err == .None, "u8 constructor")
	append(&cols, col)
	col, err = column_from("u16", u16s)
	testing.expect(t, err == .None, "u16 constructor")
	append(&cols, col)
	col, err = column_from("u32", u32s)
	testing.expect(t, err == .None, "u32 constructor")
	append(&cols, col)
	col, err = column_from("u64", u64s)
	testing.expect(t, err == .None, "u64 constructor")
	append(&cols, col)
	col, err = column_from("f32", f32s)
	testing.expect(t, err == .None, "f32 constructor")
	append(&cols, col)
	col, err = column_from("f64", f64s)
	testing.expect(t, err == .None, "f64 constructor")
	append(&cols, col)
	col, err = column_from("bool", bools)
	testing.expect(t, err == .None, "bool constructor")
	append(&cols, col)
	col, err = column_from("string", strs)
	testing.expect(t, err == .None, "string constructor")
	append(&cols, col)

	testing.expect(t, len(cols) == 12, "expected 12 columns")

	// accessors per column
	for &c in cols {
		testing.expect(t, column_len(&c) == 3, "expected length 3")
		testing.expect(t, column_elem_size(&c) > 0, "elem size set")
		testing.expect(t, column_is_all_valid(&c), "expected all valid")
		testing.expect(t, column_is_valid(&c, 0), "row 0 should be valid")
		testing.expect(t, column_data(&c) != nil, "data buffer present")
	}

	// typed values
	v_i8, ok1, g_err1 := column_get(&cols[0], 0, i8)
	testing.expect(t, g_err1 == .None && ok1 && v_i8 == -1, "i8 value")
	v_i8b, ok1b, _ := column_get(&cols[0], 2, i8)
	testing.expect(t, ok1b && v_i8b == 127, "i8 max")

	v_i64, ok2, g_err2 := column_get(&cols[3], 2, i64)
	testing.expect(t, g_err2 == .None && ok2 && v_i64 == 9223372036854775807, "i64 max")

	v_u64, ok3, _ := column_get(&cols[7], 2, u64)
	testing.expect(t, ok3 && v_u64 == 18446744073709551615, "u64 max")

	v_f32, ok4, _ := column_get(&cols[8], 2, f32)
	testing.expect(t, ok4 && v_f32 == 3.25, "f32 value")

	v_f64, ok5, _ := column_get(&cols[9], 0, f64)
	testing.expect(t, ok5 && v_f64 == -1.5, "f64 value")

	v_bool, ok6, _ := column_get(&cols[10], 2, bool)
	testing.expect(t, ok6 && v_bool, "bool value")

	v_str, ok7, _ := column_get(&cols[11], 1, string)
	testing.expect(t, ok7 && v_str == "", "empty string is valid, distinct from NULL")

	// dtype: typeid
	testing.expect(t, column_dtype(&cols[0]) == typeid_of(i8), "dtype i8")
	testing.expect(t, column_dtype(&cols[11]) == typeid_of(string), "dtype string")
}

@(test)
column_any_type_struct :: proc(t: ^testing.T) {
	Point :: struct {
		x: i32,
		y: i32,
	}
	pts := []Point{{x = 1, y = 2}, {x = 3, y = 4}, {x = 5, y = 6}}

	col, err := column_from("points", pts)
	defer column_destroy(&col)
	testing.expect(t, err == .None, "struct constructor")
	testing.expect(t, column_dtype(&col) == typeid_of(Point), "struct dtype")
	testing.expect(t, column_len(&col) == 3, "struct length")

	p, ok, _ := column_get(&col, 1, Point)
	testing.expect(t, ok && p.x == 3 && p.y == 4, "struct value")

	// setters work on structs too
	testing.expect(t, column_set(&col, 0, Point{x = 9, y = 9}) == .None, "set struct")
	p, ok, _ = column_get(&col, 0, Point)
	testing.expect(t, ok && p.x == 9, "struct value read back")
}

@(test)
column_constructor_copies_input :: proc(t: ^testing.T) {
	src := make([]i32, 4, context.allocator)
	defer delete(src)
	for _, i in src {
		src[i] = i32(i) * 10
	}

	col, err := column_from("copied", src)
	defer column_destroy(&col)
	testing.expect(t, err == .None, "constructor returned error")

	// mutating the source after construction must not affect the column
	src[0] = 999
	v, ok, _ := column_get(&col, 0, i32)
	testing.expect(t, ok && v == 0, "column must own an independent copy")
	v, ok, _ = column_get(&col, 3, i32)
	testing.expect(t, ok && v == 30, "last value intact")
}

@(test)
column_name_is_owned_copy :: proc(t: ^testing.T) {
	name_bytes := []u8{'a', 'g', 'e'}
	name := string(name_bytes)
	col, err := column_from(name, []i32{1, 2})
	defer column_destroy(&col)
	testing.expect(t, err == .None, "constructor returned error")
	testing.expect(t, column_name(&col) == "age", "name copied")
}

// --- NULL semantics --------------------------------------------------------

@(test)
column_null_vs_zero_distinct :: proc(t: ^testing.T) {
	values := []i32{0, 42, 7}
	valid  := []bool{true, false, true}
	col, err := column_from_with_valid("x", values, valid)
	defer column_destroy(&col)
	testing.expect(t, err == .None, "constructor returned error")
	testing.expect(t, !column_is_all_valid(&col), "should have NULLs")

	// row 0: value is literal 0 but the row IS valid -> not NULL
	v, ok, _ := column_get(&col, 0, i32)
	testing.expect(t, ok && v == 0, "zero is a value, not NULL")

	// row 1: NULL. value slot is 42 but must be reported invalid.
	v, ok, _ = column_get(&col, 1, i32)
	testing.expect(t, !ok, "NULL row must report invalid")
	testing.expect(t, v == 0, "NULL row value is zeroed")

	// row 2
	v, ok, _ = column_get(&col, 2, i32)
	testing.expect(t, ok && v == 7, "valid row value")

	// validity array access
	val_arr := column_valid(&col)
	testing.expect(t, val_arr != nil && !bm_get(val_arr, 1), "validity array visible")
}

@(test)
column_null_string_distinct_from_empty :: proc(t: ^testing.T) {
	values := []string{"", "hello"}
	valid  := []bool{false, true}
	col, err := column_from_with_valid("s", values, valid)
	defer column_destroy(&col)
	testing.expect(t, err == .None, "constructor returned error")

	s, ok, _ := column_get(&col, 0, string)
	testing.expect(t, !ok, "NULL string must be invalid")
	testing.expect(t, s == "", "NULL string getter returns empty")

	s, ok, _ = column_get(&col, 1, string)
	testing.expect(t, ok && s == "hello", "valid string")
}

// --- empty and boundary cases ----------------------------------------------

@(test)
column_from_empty_values :: proc(t: ^testing.T) {
	col, err := column_from("empty", []f64{})
	defer column_destroy(&col)
	testing.expect(t, err == .None, "constructor returned error")
	testing.expect(t, column_len(&col) == 0, "empty length")
	testing.expect(t, column_data(&col) == nil, "no buffer for empty column")
	testing.expect(t, column_is_all_valid(&col), "empty column all valid")

	_, _, g_err := column_get(&col, 0, f64)
	testing.expect(t, g_err == .Out_Of_Bounds, "get on empty must be out of bounds")
}

@(test)
column_single_row :: proc(t: ^testing.T) {
	col, err := column_from("b", []bool{true})
	defer column_destroy(&col)
	testing.expect(t, err == .None && column_len(&col) == 1, "single row")
	b, ok, _ := column_get(&col, 0, bool)
	testing.expect(t, ok && b, "single value")
	_, _, g_err := column_get(&col, 1, bool)
	testing.expect(t, g_err == .Out_Of_Bounds, "past end")
	_, _, g_err = column_get(&col, -1, bool)
	testing.expect(t, g_err == .Out_Of_Bounds, "negative index")
}

// --- error cases -----------------------------------------------------------

@(test)
column_get_type_mismatch :: proc(t: ^testing.T) {
	col, err := column_from("x", []i32{1, 2, 3})
	defer column_destroy(&col)
	testing.expect(t, err == .None, "constructor returned error")

	_, _, g_err := column_get(&col, 0, i64)
	testing.expect(t, g_err == .Type_Mismatch, "i64 get on i32 column")
	_, _, g_err = column_get(&col, 0, f64)
	testing.expect(t, g_err == .Type_Mismatch, "f64 get on i32 column")
	_, _, g_err = column_get(&col, 0, string)
	testing.expect(t, g_err == .Type_Mismatch, "string get on i32 column")
	_, _, g_err = column_get(&col, 0, bool)
	testing.expect(t, g_err == .Type_Mismatch, "bool get on i32 column")
	_, _, g_err = column_get(&col, 0, u32)
	testing.expect(t, g_err == .Type_Mismatch, "u32 get on i32 column (signedness matters)")
}

@(test)
column_with_valid_length_mismatch :: proc(t: ^testing.T) {
	values := []i32{1, 2, 3}
	valid  := []bool{true, false}
	_, err := column_from_with_valid("x", values, valid)
	testing.expect(t, err == .Length_Mismatch, "valid length mismatch")
}

// --- setters ---------------------------------------------------------------

@(test)
column_setters_and_null :: proc(t: ^testing.T) {
	col, err := column_from("x", []i32{0, 0, 0})
	defer column_destroy(&col)
	testing.expect(t, err == .None, "constructor returned error")

	testing.expect(t, column_set(&col, 0, i32(11)) == .None, "set i32")
	v, ok, _ := column_get(&col, 0, i32)
	testing.expect(t, ok && v == 11, "value read back")

	// setter marks the row valid even if it was NULL
	testing.expect(t, column_set_null(&col, 1) == .None, "set null")
	_, ok, _ = column_get(&col, 1, i32)
	testing.expect(t, !ok, "row 1 now NULL")
	testing.expect(t, column_set(&col, 1, i32(5)) == .None, "set over null")
	v, ok, _ = column_get(&col, 1, i32)
	testing.expect(t, ok && v == 5, "row 1 valid again")

	// out of range
	testing.expect(t, column_set(&col, 3, i32(1)) == .Out_Of_Bounds, "set out of bounds")
	testing.expect(t, column_set_null(&col, -1) == .Out_Of_Bounds, "set null negative")

	// type mismatch on set
	testing.expect(t, column_set(&col, 0, 1.0) == .Type_Mismatch, "set wrong type")
}

@(test)
column_set_valid_lazy_alloc :: proc(t: ^testing.T) {
	col, err := column_from("x", []f64{1.0, 2.0, 3.0})
	defer column_destroy(&col)
	testing.expect(t, err == .None && column_valid(&col) == nil, "starts all-valid, no alloc")

	// marking one row NULL allocates the array; others stay valid
	testing.expect(t, column_set_null(&col, 1) == .None, "mark null")
	testing.expect(t, column_valid(&col) != nil, "validity allocated lazily")
	_, ok, _ := column_get(&col, 0, f64)
	testing.expect(t, ok, "row 0 still valid")
	_, ok, _ = column_get(&col, 2, f64)
	testing.expect(t, ok, "row 2 still valid")

	// set_all_valid frees the array
	testing.expect(t, column_set_all_valid(&col) == .None, "set all valid")
	testing.expect(t, column_valid(&col) == nil, "validity freed")
	testing.expect(t, column_is_valid(&col, 1), "row 1 valid again")
}

// --- copies ----------------------------------------------------------------

@(test)
column_copy_is_deep :: proc(t: ^testing.T) {
	src, src_err := column_from_with_valid("src", []i32{1, 2, 3}, []bool{true, false, true})
	defer column_destroy(&src)
	testing.expect(t, src_err == .None, "constructor returned error")

	dst, err := column_copy(&src, context.allocator)
	defer column_destroy(&dst)
	testing.expect(t, err == .None, "copy returned error")
	testing.expect(t, column_name(&dst) == "src", "name copied")
	testing.expect(t, column_dtype(&dst) == typeid_of(i32), "dtype copied")
	testing.expect(t, column_len(&dst) == 3, "length copied")

	// NULL row preserved
	_, ok, _ := column_get(&dst, 1, i32)
	testing.expect(t, !ok, "NULL row preserved in copy")

	// independent buffers
	testing.expect(t, column_set(&dst, 0, i32(99)) == .None, "mutate copy")
	src_v, src_ok, _ := column_get(&src, 0, i32)
	testing.expect(t, src_ok && src_v == 1, "original untouched")
	testing.expect(t, column_set_null(&dst, 2) == .None, "NULL copy only")
	src_v2, src_ok2, _ := column_get(&src, 2, i32)
	testing.expect(t, src_ok2 && src_v2 == 3, "original row 2 still valid")
}

@(test)
column_clone_valid_strips_nulls :: proc(t: ^testing.T) {
	src, src_err := column_from_with_valid("src", []i32{1, 2, 3}, []bool{true, false, true})
	defer column_destroy(&src)
	testing.expect(t, src_err == .None, "constructor returned error")

	dst, err := column_clone_valid(&src, context.allocator)
	defer column_destroy(&dst)
	testing.expect(t, err == .None, "clone valid returned error")
	testing.expect(t, column_is_all_valid(&dst), "all rows valid after clone")
	v, ok, _ := column_get(&dst, 1, i32)
	testing.expect(t, ok && v == 2, "value slot preserved, now valid")
}

@(test)
column_destroy_zeroes_struct :: proc(t: ^testing.T) {
	col, err := column_from("x", []string{"a"})
	testing.expect(t, err == .None, "constructor returned error")
	column_destroy(&col)
	testing.expect(t, col.name == "" && col.data == nil && col.valid == nil, "destroyed column zeroed")
}

// --- schema ----------------------------------------------------------------

@(test)
schema_basics :: proc(t: ^testing.T) {
	fields := []Field{
		{name = "age", dtype = typeid_of(i32)},
		{name = "name", dtype = typeid_of(string)},
		{name = "score", dtype = typeid_of(f64)},
	}
	schema, schema_err := schema_create(fields)
	defer schema_destroy(&schema)
	testing.expect(t, schema_err == .None, "schema_create returned error")

	testing.expect(t, schema_len(&schema) == 3, "schema length")
	testing.expect(t, schema_has_column(&schema, "age"), "has age")
	testing.expect(t, !schema_has_column(&schema, "nope"), "missing column")

	i, idx_err := schema_index_of(&schema, "score")
	testing.expect(t, idx_err == .None && i == 2, "index of score")
	_, idx_err = schema_index_of(&schema, "nope")
	testing.expect(t, idx_err == .Column_Not_Found, "missing index error")

	f, f_err := schema_field_at(&schema, 1)
	testing.expect(t, f_err == .None && f.name == "name" && f.dtype == typeid_of(string), "field at 1")
	_, f_err = schema_field_at(&schema, 3)
	testing.expect(t, f_err == .Out_Of_Bounds, "field at past end")
}
