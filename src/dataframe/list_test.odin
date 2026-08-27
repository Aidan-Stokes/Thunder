package dataframe

// Stage 14.2 list-dtype tests: builders (from_column transfer, from_slices),
// inner NULL semantics, list_count/get/gather/unique, list_to_struct, deep
// copy and gather carry of the payload, and rendering.

import "core:strings"
import "core:testing"

// list_from_slices builds one List_Ref per row indexing into the payload.
@test
list_from_slices_basic :: proc(t: ^testing.T) {
	c, err := list_from_slices("xs", [][]int{{1, 2}, {}, {3}})
	testing.expect(t, err == .None, "list_from_slices err")
	defer column_destroy(&c)

	testing.expect(t, c.dtype == typeid_of(List_Ref), "dtype is List_Ref")
	testing.expect(t, c.inner_dtype == typeid_of(int), "inner dtype")
	testing.expect(t, c.count == 3, "3 rows")
	refs := column_typed_view(&c, List_Ref)
	testing.expect(t, refs[0] == List_Ref{0, 2}, "row 0 ref")
	testing.expect(t, refs[1] == List_Ref{2, 0}, "row 1 empty ref")
	testing.expect(t, refs[2] == List_Ref{2, 1}, "row 2 ref")

	elems := list_inner_view(&c, int)
	testing.expect(t, len(elems) == 3, "3 inner elements")
	testing.expect(t, elems[0] == 1 && elems[1] == 2 && elems[2] == 3, "element values")
	testing.expect(t, c.inner_valid == nil, "no inner NULLs")
}

// list_from_slices_with_valid marks outer NULL rows; their refs stay zeroed
// and list_count mirrors the NULLs.
@test
list_from_slices_with_valid_null_row :: proc(t: ^testing.T) {
	c, err := list_from_slices_with_valid("xs", [][]i64{{1, 2}, {3, 4}, {5}}, []bool{true, false, true})
	testing.expect(t, err == .None, "with_valid err")
	defer column_destroy(&c)

	testing.expect(t, !column_is_valid(&c, 1), "row 1 NULL")
	refs := column_typed_view(&c, List_Ref)
	testing.expect(t, refs[1] == List_Ref{}, "NULL row ref zeroed")

	cnt, cerr := list_count(&c)
	testing.expect(t, cerr == .None, "list_count err")
	defer column_destroy(&cnt)
	v, valid, _ := column_get(&cnt, 1, i64)
	testing.expect(t, !valid, "count NULL at row 1")
	_ = v
	v0, valid0, _ := column_get(&cnt, 0, i64)
	testing.expect(t, valid0 && v0 == 2, "count 2 at row 0")
}

// list_from_column transfers the element column: elems is zeroed, offsets
// partition the payload, and an owned string element column is merged into one
// blob with re-pointed headers.
@test
list_from_column_transfers_elem_column :: proc(t: ^testing.T) {
	elem_names := []string{"aa", "bb", "cc", "dd"}
	elems, err := owned_string_column(context.allocator, "elems", elem_names)
	testing.expect(t, err == .None, "owned_string_column err")
	offsets := []int{0, 2, 2, 4}
	c, lerr := list_from_column("xs", &elems, offsets)
	testing.expect(t, lerr == .None, "list_from_column err")
	defer column_destroy(&c)

	testing.expect(t, elems.name == "" && elems.data == nil, "element column transferred/zeroed")
	testing.expect(t, c.inner_dtype == typeid_of(string), "inner dtype string")
	testing.expect(t, c.payload != nil, "payload allocated")

	g, gerr := list_get(&c, 1)
	testing.expect(t, gerr == .None, "list_get err")
	defer column_destroy(&g)
	s, valid, _ := column_get(&g, 2, string)
	testing.expect(t, valid && s == "dd", "inner string from merged blob")

	g0, gerr0 := list_get(&c, 0)
	testing.expect(t, gerr0 == .None, "list_get[0] err")
	defer column_destroy(&g0)
	s0, valid0, _ := column_get(&g0, 0, string)
	testing.expect(t, valid0 && s0 == "aa", "first inner string")
}

// list_count reports element counts and mirrors outer NULLs.
@test
list_count_counts_elements :: proc(t: ^testing.T) {
	c, err := list_from_slices("xs", [][]f64{{1.5, 2.5}, {}, {3.5, 4.5, 5.5}})
	testing.expect(t, err == .None, "err")
	defer column_destroy(&c)

	cnt, cerr := list_count(&c)
	testing.expect(t, cerr == .None, "list_count err")
	defer column_destroy(&cnt)
	want := []i64{2, 0, 3}
	for want_v, r in want {
		v, valid, gerr := column_get(&cnt, r, i64)
		testing.expect(t, gerr == .None, "count get")
		testing.expect(t, valid, "count valid")
		testing.expect(t, v == want_v, "count value")
	}
}

// list_get reads one element position; NULL rows, out-of-range indices, and
// NULL elements all yield NULL.
@test
list_get_position_semantics :: proc(t: ^testing.T) {
	valid := []bool{true, true, true, true, false}
	c, err := list_from_slices_with_valid("xs", [][]i64{{10, 11}, {12, 13}, {14}, {}, {20, 21}}, valid)
	testing.expect(t, err == .None, "err")
	defer column_destroy(&c)

	g, gerr := list_get(&c, 1)
	testing.expect(t, gerr == .None, "list_get err")
	defer column_destroy(&g)

	want := []i64{11, 13, 0, 0, 0}
	want_valid := []bool{true, true, false, false, false}
	for r in 0 ..< 5 {
		v, is_valid, _ := column_get(&g, r, i64)
		testing.expect(t, is_valid == want_valid[r], "validity at row")
		if want_valid[r] {
			testing.expect(t, v == want[r], "value at row")
		}
	}

	g0, gerr0 := list_get(&c, 0)
	testing.expect(t, gerr0 == .None, "list_get[0] err")
	defer column_destroy(&g0)
	v, is_valid, _ := column_get(&g0, 3, i64)
	testing.expect(t, !is_valid, "row 3 empty -> NULL")
	_ = v
}

// list_gather uses a per-row index.
@test
list_gather_per_row_indices :: proc(t: ^testing.T) {
	c, err := list_from_slices("xs", [][]i64{{10, 11, 12}, {20}, {30, 31}})
	testing.expect(t, err == .None, "err")
	defer column_destroy(&c)

	g, gerr := list_gather(&c, []int{2, 0, 1})
	testing.expect(t, gerr == .None, "list_gather err")
	defer column_destroy(&g)
	v, valid, _ := column_get(&g, 0, i64)
	testing.expect(t, valid && v == 12, "row 0 picks 12")
	v1, valid1, _ := column_get(&g, 1, i64)
	testing.expect(t, valid1 && v1 == 20, "row 1 picks 20")
	v2, valid2, _ := column_get(&g, 2, i64)
	testing.expect(t, valid2 && v2 == 31, "row 2 index 1 picks 31")
}

// list_get/list_gather reject non-list columns.
@test
list_ops_reject_non_list :: proc(t: ^testing.T) {
	n, err := column_from("n", []i64{1, 2})
	testing.expect(t, err == .None, "column_from err")
	defer column_destroy(&n)
	_, cerr := list_count(&n)
	testing.expect(t, cerr == .Invalid_Argument, "count rejects")
	_, gerr := list_get(&n, 0)
	testing.expect(t, gerr == .Invalid_Argument, "get rejects")
	_, uerr := list_unique(&n)
	testing.expect(t, uerr == .Invalid_Argument, "unique rejects")
}

// list_unique deduplicates per row in first-seen order, skipping NULL elements.
@test
list_unique_dedups_first_seen :: proc(t: ^testing.T) {
	c, err := list_from_slices("xs", [][]i64{{1, 2, 1, 3, 2}, {4}, {}, {5, 5}})
	testing.expect(t, err == .None, "err")
	defer column_destroy(&c)

	u, uerr := list_unique(&c)
	testing.expect(t, uerr == .None, "list_unique err")
	defer column_destroy(&u)

	refs := column_typed_view(&u, List_Ref)
	want_lens := []int{3, 1, 0, 1}
	for want_len, r in want_lens {
		testing.expect(t, refs[r].len == want_len, "unique len")
	}
	elems := list_inner_view(&u, i64)
	want := []i64{1, 2, 3, 4, 5}
	for want_v, e in want {
		testing.expect(t, elems[e] == want_v, "unique element")
	}
	testing.expect(t, u.inner_valid == nil, "NULL elements skipped, no inner valid")
}

// list_to_struct produces one column per element position; short rows and
// NULL rows yield NULL in the missing positions.
@test
list_to_struct_fields :: proc(t: ^testing.T) {
	c, err := list_from_slices("xs", [][]string{{"a", "b"}, {"c"}, {"d", "e", "f"}})
	testing.expect(t, err == .None, "err")
	defer column_destroy(&c)

	df, serr := list_to_struct(&c)
	testing.expect(t, serr == .None, "list_to_struct err")
	defer dataframe_destroy(&df)

	testing.expect(t, dataframe_num_cols(&df) == 3, "3 field columns")
	for f in 0 ..< 3 {
		col := dataframe_column_at(&df, f) or_else nil
		testing.expect(t, col != nil, "field col exists")
		testing.expect(t, col.dtype == typeid_of(string), "field dtype")
	}

	f0 := dataframe_column_at(&df, 0) or_else nil
	sa, valid_a, _ := column_get(f0, 0, string)
	testing.expect(t, valid_a && sa == "a", "field_0 row 0")
	_, valid_1, _ := column_get(f0, 1, string)
	testing.expect(t, valid_1, "field_0 row 1")

	f2 := dataframe_column_at(&df, 2) or_else nil
	sf, valid_f, _ := column_get(f2, 2, string)
	testing.expect(t, valid_f && sf == "f", "field_2 row 2")
	_, valid_short, _ := column_get(f2, 0, string)
	testing.expect(t, !valid_short, "row 0 has no third element")
}

// column_copy deep-copies a List column: the copy owns an independent payload
// and the string inner headers re-point into the copy's blob.
@test
list_column_copy_deep :: proc(t: ^testing.T) {
	elem_names := []string{"aa", "bb", "cc", "dd"}
	elems, err := owned_string_column(context.allocator, "elems", elem_names)
	testing.expect(t, err == .None, "owned_string_column err")
	c, lerr := list_from_column("xs", &elems, []int{0, 2, 2, 4})
	testing.expect(t, lerr == .None, "list_from_column err")
	defer column_destroy(&c)

	cp, cerr := column_copy(&c, context.allocator)
	testing.expect(t, cerr == .None, "column_copy err")
	defer column_destroy(&cp)
	testing.expect(t, cp.payload != nil && cp.payload != c.payload, "independent payload")
	testing.expect(t, cp.inner_valid == nil, "inner_valid copied")

	g, gerr := list_get(&cp, 1)
	testing.expect(t, gerr == .None, "list_get err")
	defer column_destroy(&g)
	s, valid, _ := column_get(&g, 2, string)
	testing.expect(t, valid && s == "dd", "inner string survives copy")
}

// dataframe_head carries a List column through gather_rows_core wholesale.
@test
list_column_gathers :: proc(t: ^testing.T) {
	c, err := list_from_slices("xs", [][]i64{{1, 2}, {3}, {4, 5, 6}})
	testing.expect(t, err == .None, "err")
	defer column_destroy(&c)
	id, ierr := column_from("id", []i64{10, 20, 30})
	testing.expect(t, ierr == .None, "id err")
	defer column_destroy(&id)
	df, derr := dataframe_from_columns([]^Column{&id, &c})
	testing.expect(t, derr == .None, "dataframe err")
	defer dataframe_destroy(&df)

	hd, herr := dataframe_head(&df, 2)
	testing.expect(t, herr == .None, "dataframe_head err")
	defer dataframe_destroy(&hd)

	lc := dataframe_column_at(&hd, 1) or_else nil
	testing.expect(t, lc != nil, "list col present")
	refs := column_typed_view(lc, List_Ref)
	testing.expect(t, refs[0] == List_Ref{0, 2}, "head row 0 ref")
	testing.expect(t, refs[1] == List_Ref{2, 1}, "head row 1 ref")
	g, gerr := list_get(lc, 0)
	testing.expect(t, gerr == .None, "list_get err")
	defer column_destroy(&g)
	v, valid, _ := column_get(&g, 1, i64)
	testing.expect(t, valid && v == 3, "gathered inner element")
}

// dataframe_to_string renders list rows as [e0, e1, ...] and NULL rows as null.
@test
list_rendering :: proc(t: ^testing.T) {
	valid := []bool{true, false}
	c, err := list_from_slices_with_valid("xs", [][]i64{{1, 2}, {3, 4}}, valid)
	testing.expect(t, err == .None, "err")
	defer column_destroy(&c)
	df, derr := dataframe_from_columns([]^Column{&c})
	testing.expect(t, derr == .None, "dataframe err")
	defer dataframe_destroy(&df)

	s, serr := dataframe_to_string(&df)
	testing.expect(t, serr == .None, "to_string err")
	defer delete(s)
	contains := strings.contains(s, "[1, 2]")
	testing.expect(t, contains, "renders [1, 2]")
	contains_null := strings.contains(s, "null")
	testing.expect(t, contains_null, "renders null row")
}

// list_from_column rejects malformed offsets and nested lists.
@test
list_from_column_validates :: proc(t: ^testing.T) {
	elems, err := column_from("elems", []i64{1, 2, 3})
	testing.expect(t, err == .None, "column_from err")
	defer column_destroy(&elems)

	_, cerr := list_from_column("xs", &elems, []int{0, 1})
	testing.expect(t, cerr == .Invalid_Argument, "offsets must reach count")
	_, cerr = list_from_column("xs", &elems, []int{0, 2, 1})
	testing.expect(t, cerr == .Invalid_Argument, "offsets must be monotone")
	_, cerr = list_from_column("", &elems, []int{0, 3})
	testing.expect(t, cerr == .Column_Name_Empty, "name required")

	inner, ierr := list_from_slices("nested", [][]i64{{1}})
	testing.expect(t, ierr == .None, "inner list err")
	defer column_destroy(&inner)
	_, cerr = list_from_column("bad", &inner, []int{0, 1})
	testing.expect(t, cerr == .Unsupported_Operation, "no nested lists")
}
