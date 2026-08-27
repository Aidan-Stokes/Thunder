package dataframe

// Stage 14.3 explode/unnest tests: row widening with row repetition, NULL
// list/element/struct semantics, repeated List columns with recomputed
// offsets, owned string contents surviving the source's destruction, and
// struct reflection field extraction.

import "core:testing"

Point :: struct {
	x: i64,
	y: f64,
}

Person :: struct {
	name: string,
	age:  i32,
}

// explode widens the row count; other columns repeat the outer row.
@test
explode_basic :: proc(t: ^testing.T) {
	xs, err := list_from_slices("xs", [][]i64{{1, 2}, {}, {3, 4, 5}})
	testing.expect(t, err == .None, "list err")
	defer column_destroy(&xs)
	id, ierr := column_from("id", []i64{10, 20, 30})
	testing.expect(t, ierr == .None, "id err")
	defer column_destroy(&id)

	df, derr := dataframe_from_columns([]^Column{&id, &xs})
	testing.expect(t, derr == .None, "df err")
	defer dataframe_destroy(&df)

	out, xerr := dataframe_explode(&df, "xs")
	testing.expect(t, xerr == .None, "explode err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 5, "5 output rows")
	oc := dataframe_column_at(&out, 0) or_else nil
	want := []i64{10, 10, 30, 30, 30}
	for want_v, r in want {
		v, valid, _ := column_get(oc, r, i64)
		testing.expect(t, valid && v == want_v, "id repeated")
	}
	xc := dataframe_column_at(&out, 1) or_else nil
	testing.expect(t, xc.dtype == typeid_of(i64), "exploded dtype is inner")
	want_x := []i64{1, 2, 3, 4, 5}
	for want_v, r in want_x {
		v, valid, _ := column_get(xc, r, i64)
		testing.expect(t, valid && v == want_v, "element value")
	}
}

// a NULL list row explodes to one all-NULL row.
@test
explode_null_row_is_all_null :: proc(t: ^testing.T) {
	valid := []bool{true, false, true}
	xs, err := list_from_slices_with_valid("xs", [][]i64{{1}, {2, 3}, {4}}, valid)
	testing.expect(t, err == .None, "list err")
	defer column_destroy(&xs)
	id, ierr := column_from("id", []i64{10, 20, 30})
	testing.expect(t, ierr == .None, "id err")
	defer column_destroy(&id)

	df, derr := dataframe_from_columns([]^Column{&id, &xs})
	testing.expect(t, derr == .None, "df err")
	defer dataframe_destroy(&df)

	out, xerr := dataframe_explode(&df, "xs")
	testing.expect(t, xerr == .None, "explode err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 3, "3 output rows")
	idc := dataframe_column_at(&out, 0) or_else nil
	xc := dataframe_column_at(&out, 1) or_else nil
	want_valid := []bool{true, false, true}
	want_id := []i64{10, 0, 30}
	for r in 0 ..< 3 {
		v, valid_r, _ := column_get(idc, r, i64)
		testing.expect(t, valid_r == want_valid[r], "id validity")
		if valid_r {
			testing.expect(t, v == want_id[r], "id value")
		}
	}
	_, xv, _ := column_get(xc, 1, i64)
	testing.expect(t, !xv, "exploded NULL row is NULL")
}

// a NULL element within a valid list yields a NULL row in the exploded column
// while other columns still repeat their value.
@test
explode_null_element :: proc(t: ^testing.T) {
	inner_valid := []bool{true, false, true}
	elems, err := column_from_with_valid("elems", []i64{1, 2, 3}, inner_valid)
	testing.expect(t, err == .None, "elems err")
	defer column_destroy(&elems)
	xs, lerr := list_from_column("xs", &elems, []int{0, 3})
	testing.expect(t, lerr == .None, "list err")
	defer column_destroy(&xs)

	label, lerr2 := column_from("label", []string{"r0"})
	testing.expect(t, lerr2 == .None, "label err")
	defer column_destroy(&label)

	df, derr := dataframe_from_columns([]^Column{&label, &xs})
	testing.expect(t, derr == .None, "df err")
	defer dataframe_destroy(&df)

	out, xerr := dataframe_explode(&df, "xs")
	testing.expect(t, xerr == .None, "explode err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 3, "3 output rows")
	xc := dataframe_column_at(&out, 1) or_else nil
	_, valid_1, _ := column_get(xc, 1, i64)
	testing.expect(t, !valid_1, "NULL element stays NULL")
	lc := dataframe_column_at(&out, 0) or_else nil
	s0, valid_0, _ := column_get(lc, 0, string)
	testing.expect(t, valid_0 && s0 == "r0", "label repeated at 0")
	s1, valid_1b, _ := column_get(lc, 1, string)
	testing.expect(t, valid_1b && s1 == "r0", "label repeated at 1")
}

// owned string contents are deep-copied into the exploded column: strings
// survive the source DataFrame's destruction.
@test
explode_owned_strings_survive :: proc(t: ^testing.T) {
	names, err := owned_string_column(context.allocator, "elems", []string{"aa", "bb", "cc"})
	testing.expect(t, err == .None, "owned err")
	xs, lerr := list_from_column("xs", &names, []int{0, 1, 3})
	testing.expect(t, lerr == .None, "list err")
	defer column_destroy(&xs)

	df, derr := dataframe_from_columns([]^Column{&xs})
	testing.expect(t, derr == .None, "df err")
	out, xerr := dataframe_explode(&df, "xs")
	testing.expect(t, xerr == .None, "explode err")
	// destroy the source now: the exploded strings must be independent.
	dataframe_destroy(&df)

	oc := dataframe_column_at(&out, 0) or_else nil
	testing.expect(t, oc != nil && oc.dtype == typeid_of(string), "exploded string column")
	s0, v0, _ := column_get(oc, 0, string)
	testing.expect(t, v0 && s0 == "aa", "string 0 survives")
	s2, v2, _ := column_get(oc, 2, string)
	testing.expect(t, v2 && s2 == "cc", "string 2 survives")
	testing.expect(t, oc.payload != nil, "exploded column owns contents")
	dataframe_destroy(&out)
}

// exploding one of two list columns repeats the other with recomputed offsets
// and a preserved payload.
@test
explode_two_lists :: proc(t: ^testing.T) {
	a, err := list_from_slices("a", [][]i64{{1, 2}, {3}})
	testing.expect(t, err == .None, "list a err")
	defer column_destroy(&a)
	b, berr := list_from_slices("b", [][]string{{"x", "y"}, {"z"}})
	testing.expect(t, berr == .None, "list b err")
	defer column_destroy(&b)

	df, derr := dataframe_from_columns([]^Column{&a, &b})
	testing.expect(t, derr == .None, "df err")
	defer dataframe_destroy(&df)

	out, xerr := dataframe_explode(&df, "a")
	testing.expect(t, xerr == .None, "explode err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 3, "3 output rows")
	bc := dataframe_column_at(&out, 1) or_else nil
	refs := column_typed_view(bc, List_Ref)
	testing.expect(t, refs[0] == List_Ref{0, 2}, "b row 0 ref")
	testing.expect(t, refs[1] == List_Ref{2, 2}, "b row 1 ref")
	testing.expect(t, refs[2] == List_Ref{4, 1}, "b row 2 ref")
	gb, gerr := list_get(bc, 0)
	testing.expect(t, gerr == .None, "b list_get err")
	defer column_destroy(&gb)
	s, valid, _ := column_get(&gb, 1, string)
	testing.expect(t, valid && s == "x", "repeated list kept")
	s, valid, _ = column_get(&gb, 2, string)
	testing.expect(t, valid && s == "z", "second list kept")
}

// explode rejects non-list columns and missing names.
@test
explode_validates :: proc(t: ^testing.T) {
	id, ierr := column_from("id", []i64{1, 2})
	testing.expect(t, ierr == .None, "id err")
	defer column_destroy(&id)
	df, derr := dataframe_from_columns([]^Column{&id})
	testing.expect(t, derr == .None, "df err")
	defer dataframe_destroy(&df)

	_, xerr := dataframe_explode(&df, "id")
	testing.expect(t, xerr == .Invalid_Argument, "non-list rejected")
	_, xerr = dataframe_explode(&df, "nope")
	testing.expect(t, xerr == .Column_Not_Found, "missing rejected")
}

// unnest splits a struct column into one column per field.
@test
unnest_struct_fields :: proc(t: ^testing.T) {
	pts, err := column_from("pt", []Point{{x = 1, y = 2.5}, {x = 3, y = 4.5}})
	testing.expect(t, err == .None, "column_from err")
	defer column_destroy(&pts)

	df, derr := dataframe_from_columns([]^Column{&pts})
	testing.expect(t, derr == .None, "df err")
	defer dataframe_destroy(&df)

	out, uerr := dataframe_unnest(&df, "pt")
	testing.expect(t, uerr == .None, "unnest err")
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_cols(&out) == 2, "2 field columns")
	testing.expect(t, dataframe_num_rows(&out) == 2, "rows preserved")
	xc := dataframe_get_column(&out, "pt_x") or_else nil
	testing.expect(t, xc != nil && xc.dtype == typeid_of(i64), "pt_x dtype")
	yc := dataframe_get_column(&out, "pt_y") or_else nil
	testing.expect(t, yc != nil && yc.dtype == typeid_of(f64), "pt_y dtype")

	x0, v0, _ := column_get(xc, 0, i64)
	testing.expect(t, v0 && x0 == 1, "x0")
	y1, v1, _ := column_get(yc, 1, f64)
	testing.expect(t, v1 && y1 == 4.5, "y1")
}

// a NULL struct row yields NULL in every field column.
@test
unnest_null_struct_row :: proc(t: ^testing.T) {
	valid := []bool{true, false, true}
	pts, err := column_from_with_valid("pt", []Point{{x = 1, y = 1.0}, {x = 2, y = 2.0}, {x = 3, y = 3.0}}, valid)
	testing.expect(t, err == .None, "column_from err")
	defer column_destroy(&pts)

	df, derr := dataframe_from_columns([]^Column{&pts})
	testing.expect(t, derr == .None, "df err")
	defer dataframe_destroy(&df)

	out, uerr := dataframe_unnest(&df, "pt")
	testing.expect(t, uerr == .None, "unnest err")
	defer dataframe_destroy(&out)

	xc := dataframe_get_column(&out, "pt_x") or_else nil
	_, v1, _ := column_get(xc, 1, i64)
	testing.expect(t, !v1, "NULL struct -> NULL field")
	yc := dataframe_get_column(&out, "pt_y") or_else nil
	_, v1b, _ := column_get(yc, 1, f64)
	testing.expect(t, !v1b, "NULL struct -> NULL other field")
}

// string fields are extracted; the field columns carry the borrowed contents.
@test
unnest_string_field :: proc(t: ^testing.T) {
	people, err := column_from("p", []Person{{name = "ada", age = 36}, {name = "grace", age = 45}})
	testing.expect(t, err == .None, "column_from err")
	defer column_destroy(&people)

	df, derr := dataframe_from_columns([]^Column{&people})
	testing.expect(t, derr == .None, "df err")
	defer dataframe_destroy(&df)

	out, uerr := dataframe_unnest(&df, "p")
	testing.expect(t, uerr == .None, "unnest err")
	defer dataframe_destroy(&out)

	nc := dataframe_get_column(&out, "p_name") or_else nil
	testing.expect(t, nc != nil && nc.dtype == typeid_of(string), "p_name dtype")
	s, v, _ := column_get(nc, 1, string)
	testing.expect(t, v && s == "grace", "name extracted")
	ac := dataframe_get_column(&out, "p_age") or_else nil
	a, va, _ := column_get(ac, 0, i32)
	testing.expect(t, va && a == 36, "age extracted")
}

// unnest rejects non-struct columns.
@test
unnest_validates :: proc(t: ^testing.T) {
	id, ierr := column_from("id", []i64{1, 2})
	testing.expect(t, ierr == .None, "id err")
	defer column_destroy(&id)
	df, derr := dataframe_from_columns([]^Column{&id})
	testing.expect(t, derr == .None, "df err")
	defer dataframe_destroy(&df)

	_, uerr := dataframe_unnest(&df, "id")
	testing.expect(t, uerr == .Invalid_Argument, "non-struct rejected")
	_, uerr = dataframe_unnest(&df, "nope")
	testing.expect(t, uerr == .Column_Not_Found, "missing rejected")
}

// gather carries an owned list-of-strings column: inner strings survive a
// dataframe_head (re-pointed into the gathered payload).
@test
gather_list_of_strings_repoints :: proc(t: ^testing.T) {
	names, err := owned_string_column(context.allocator, "elems", []string{"aa", "bb", "cc", "dd"})
	testing.expect(t, err == .None, "owned err")
	xs, lerr := list_from_column("xs", &names, []int{0, 2, 4})
	testing.expect(t, lerr == .None, "list err")
	defer column_destroy(&xs)

	df, derr := dataframe_from_columns([]^Column{&xs})
	testing.expect(t, derr == .None, "df err")
	defer dataframe_destroy(&df)

	hd, herr := dataframe_head(&df, 1)
	testing.expect(t, herr == .None, "head err")
	// destroy the source: the gathered strings must stay valid.
	dataframe_destroy(&df)

	lc := dataframe_column_at(&hd, 0) or_else nil
	g, gerr := list_get(lc, 0)
	testing.expect(t, gerr == .None, "list_get err 0")
	defer column_destroy(&g)
	s0, v0, _ := column_get(&g, 0, string)
	testing.expect(t, v0 && s0 == "aa", "gathered string 0")
	g1, gerr1 := list_get(lc, 1)
	testing.expect(t, gerr1 == .None, "list_get err 1")
	defer column_destroy(&g1)
	s1, v1, _ := column_get(&g1, 0, string)
	testing.expect(t, v1 && s1 == "bb", "gathered string 1")
	dataframe_destroy(&hd)
}
