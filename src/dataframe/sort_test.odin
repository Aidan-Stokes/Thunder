package dataframe

// Stage 5 tests: sort, argsort, sort_by.

import "core:testing"
import "core:fmt"
import "core:math"
import "expr"

// sort_fixture builds a small df:
//
//	id    i32    [1, 2, 3, 4, 5, 6]
//	g     i32    [2, 1, 2, 1, 2, 1]
//	v     f64    [5.0, 4.0, 3.0, 2.0, 1.0, 0.5]
//	s     string ["c", "b", "a", "c", "b", "a"]
//	flag  bool   [true, false, true, false, true, false]
sort_fixture :: proc(t: ^testing.T) -> (df: DataFrame, err: Error) {
	id, e1 := column_from("id", []i32{1, 2, 3, 4, 5, 6})
	g, e2 := column_from("g", []i32{2, 1, 2, 1, 2, 1})
	v, e3 := column_from("v", []f64{5.0, 4.0, 3.0, 2.0, 1.0, 0.5})
	s, e4 := column_from("s", []string{"c", "b", "a", "c", "b", "a"})
	flag, e5 := column_from("flag", []bool{true, false, true, false, true, false})
	testing.expect(t, e1 == .None && e2 == .None && e3 == .None && e4 == .None && e5 == .None, "sort fixture columns")
	d, df_err := dataframe_from_columns([]^Column{&id, &g, &v, &s, &flag})
	testing.expect(t, df_err == .None, "sort fixture from_columns")
	return d, df_err
}

// expect_sorted_checks the id column of a sorted result in order.
expect_sorted_id :: proc(t: ^testing.T, what: string, out: ^DataFrame, want: []i32) {
	testing.expect(t, dataframe_num_rows(out) == len(want), fmt.tprintf("%s row count", what))
	id, err := dataframe_get_column(out, "id")
	testing.expect(t, err == .None, fmt.tprintf("%s has id", what))
	expect_i32_col(t, "id", id, want)
}

@(test)
sort_single_column_asc :: proc(t: ^testing.T) {
	df, err := sort_fixture(t)
	defer dataframe_destroy(&df)
	testing.expect(t, err == .None, "fixture")

	out, s_err := dataframe_sort(&df, []Sort_Key{sort_key("v")})
	defer dataframe_destroy(&out)
	testing.expect(t, s_err == .None, "sort asc")
	expect_sorted_id(t, "asc", &out, []i32{6, 5, 4, 3, 2, 1})
}

@(test)
sort_single_column_desc :: proc(t: ^testing.T) {
	df, err := sort_fixture(t)
	defer dataframe_destroy(&df)
	testing.expect(t, err == .None, "fixture")

	out, s_err := dataframe_sort(&df, []Sort_Key{sort_key("v", .Desc)})
	defer dataframe_destroy(&out)
	testing.expect(t, s_err == .None, "sort desc")
	expect_sorted_id(t, "desc", &out, []i32{1, 2, 3, 4, 5, 6})
}

@(test)
sort_string_column :: proc(t: ^testing.T) {
	df, err := sort_fixture(t)
	defer dataframe_destroy(&df)
	testing.expect(t, err == .None, "fixture")

	out, s_err := dataframe_sort(&df, []Sort_Key{sort_key("s")})
	defer dataframe_destroy(&out)
	testing.expect(t, s_err == .None, "string sort")
	// ties on "a" keep source order: rows 3 then 6; "c": rows 1 then 4
	expect_sorted_id(t, "string asc", &out, []i32{3, 6, 2, 5, 1, 4})
}

@(test)
sort_bool_column :: proc(t: ^testing.T) {
	df, err := sort_fixture(t)
	defer dataframe_destroy(&df)
	testing.expect(t, err == .None, "fixture")

	out, s_err := dataframe_sort(&df, []Sort_Key{sort_key("flag")})
	defer dataframe_destroy(&out)
	testing.expect(t, s_err == .None, "bool sort")
	expect_sorted_id(t, "bool asc (false first)", &out, []i32{2, 4, 6, 1, 3, 5})
}

@(test)
sort_float_total_order :: proc(t: ^testing.T) {
	f, f_err := column_from("f", []f64{1.0, math.nan_f64(), math.inf_f64(1), -0.0, math.inf_f64(-1), 0.0})
	testing.expect(t, f_err == .None, "f column")
	// -inf < 0.0 == -0.0 < 1.0 < +inf < NaN
	fdf := df_from_cols(t, &f)
	defer dataframe_destroy(&fdf)
	out, s_err := dataframe_sort(&fdf, []Sort_Key{sort_key("f")})
	defer dataframe_destroy(&out)
	testing.expect(t, s_err == .None, "float sort")

	fv := column_typed_view(&out.col_views[0], f64)
	testing.expect(t, out.col_views[0].count == 6, "row count")
	testing.expect(t, fv[0] == math.inf_f64(-1), "first is -inf")
	testing.expect(t, fv[1] == 0.0 && fv[2] == 0.0, "-0.0 and +0.0 together")
	testing.expect(t, fv[3] == 1.0, "1.0")
	testing.expect(t, fv[4] == math.inf_f64(1), "second-last is +inf")
	testing.expect(t, fv[5] != fv[5], "last is NaN")
}

@(test)
sort_multi_key_mixed_directions :: proc(t: ^testing.T) {
	df, err := sort_fixture(t)
	defer dataframe_destroy(&df)
	testing.expect(t, err == .None, "fixture")

	// g ascending, then v descending within each g.
	out, s_err := dataframe_sort(&df, []Sort_Key{sort_key("g"), sort_key("v", .Desc)})
	defer dataframe_destroy(&out)
	testing.expect(t, s_err == .None, "multi-key sort")
	// g=1 rows (2,4,6) with v desc: 4.0, 2.0, 0.5 -> ids 2,4,6
	// g=2 rows (1,3,5) with v desc: 5.0, 3.0, 1.0 -> ids 1,3,5
	expect_sorted_id(t, "multi-key", &out, []i32{2, 4, 6, 1, 3, 5})
}

@(test)
sort_null_ordering_default_last :: proc(t: ^testing.T) {
	id, e1 := column_from("id", []i32{1, 2, 3, 4})
	x, e2 := column_from("x", []i32{0, 1, 0, 3})
	testing.expect(t, e1 == .None && e2 == .None, "columns")
	testing.expect(t, column_set_valid(&x, 0, false) == .None, "x[0] NULL") // id 1 is NULL
	testing.expect(t, column_set_valid(&x, 2, false) == .None, "x[2] NULL") // id 3 is NULL
	df, df_err := dataframe_from_columns([]^Column{&id, &x})
	defer dataframe_destroy(&df)
	testing.expect(t, df_err == .None, "from_columns")

	out, s_err := dataframe_sort(&df, []Sort_Key{sort_key("x")})
	defer dataframe_destroy(&out)
	testing.expect(t, s_err == .None, "nulls last")
	expect_sorted_id(t, "nulls last", &out, []i32{2, 4, 1, 3})
}

@(test)
sort_null_ordering_first :: proc(t: ^testing.T) {
	id, e1 := column_from("id", []i32{1, 2, 3, 4})
	x, e2 := column_from("x", []i32{0, 1, 0, 3})
	testing.expect(t, e1 == .None && e2 == .None, "columns")
	testing.expect(t, column_set_valid(&x, 0, false) == .None, "x[0] NULL")
	testing.expect(t, column_set_valid(&x, 2, false) == .None, "x[2] NULL")
	df, df_err := dataframe_from_columns([]^Column{&id, &x})
	defer dataframe_destroy(&df)
	testing.expect(t, df_err == .None, "from_columns")

	out, s_err := dataframe_sort(&df, []Sort_Key{sort_key("x", .Asc, true)})
	defer dataframe_destroy(&out)
	testing.expect(t, s_err == .None, "nulls first")
	expect_sorted_id(t, "nulls first", &out, []i32{1, 3, 2, 4})
}

@(test)
sort_null_ties_fall_through :: proc(t: ^testing.T) {
	// Two rows NULL in the first key: the second key decides.
	id, e1 := column_from("id", []i32{1, 2, 3})
	k, e2 := column_from("k", []i32{0, 5, 0})
	order, e3 := column_from("order", []i32{2, 0, 1})
	testing.expect(t, e1 == .None && e2 == .None && e3 == .None, "columns")
	testing.expect(t, column_set_valid(&k, 0, false) == .None, "k[0] NULL")
	testing.expect(t, column_set_valid(&k, 2, false) == .None, "k[2] NULL")
	df, df_err := dataframe_from_columns([]^Column{&id, &k, &order})
	defer dataframe_destroy(&df)
	testing.expect(t, df_err == .None, "from_columns")

	// NULLs first by k, ties broken by order ascending: id1 (order 2), id3 (order 1), id2 (k=5).
	out, s_err := dataframe_sort(&df, []Sort_Key{sort_key("k", .Asc, true), sort_key("order")})
	defer dataframe_destroy(&out)
	testing.expect(t, s_err == .None, "null tie fall-through")
	expect_sorted_id(t, "null tie", &out, []i32{3, 1, 2})
}

@(test)
sort_stable_keeps_source_order :: proc(t: ^testing.T) {
	id, e1 := column_from("id", []i32{1, 2, 3, 4})
	k, e2 := column_from("k", []i32{7, 7, 7, 7})
	testing.expect(t, e1 == .None && e2 == .None, "columns")
	df, df_err := dataframe_from_columns([]^Column{&id, &k})
	defer dataframe_destroy(&df)
	testing.expect(t, df_err == .None, "from_columns")

	out, s_err := dataframe_sort(&df, []Sort_Key{sort_key("k")})
	defer dataframe_destroy(&out)
	testing.expect(t, s_err == .None, "stable sort")
	expect_sorted_id(t, "stable", &out, []i32{1, 2, 3, 4})
}

@(test)
sort_idempotent :: proc(t: ^testing.T) {
	df, err := sort_fixture(t)
	defer dataframe_destroy(&df)
	testing.expect(t, err == .None, "fixture")

	out1, e1 := dataframe_sort(&df, []Sort_Key{sort_key("g"), sort_key("s", .Desc)})
	defer dataframe_destroy(&out1)
	testing.expect(t, e1 == .None, "first sort")
	out2, e2 := dataframe_sort(&out1, []Sort_Key{sort_key("g"), sort_key("s", .Desc)})
	defer dataframe_destroy(&out2)
	testing.expect(t, e2 == .None, "second sort")

	expect_sorted_id(t, "idempotent first", &out1, []i32{4, 2, 6, 1, 5, 3})
	expect_sorted_id(t, "idempotent second", &out2, []i32{4, 2, 6, 1, 5, 3})
	// second pass must not reorder anything
	for i in 0 ..< dataframe_num_rows(&out2) {
		a, _ := dataframe_get_column(&out1, "id")
		b, _ := dataframe_get_column(&out2, "id")
		testing.expect(t, column_typed_view(a, i32)[i] == column_typed_view(b, i32)[i], fmt.tprintf("row %d unchanged", i))
	}
}

@(test)
argsort_is_permutation :: proc(t: ^testing.T) {
	df, err := sort_fixture(t)
	defer dataframe_destroy(&df)
	testing.expect(t, err == .None, "fixture")

	perm, p_err := dataframe_argsort(&df, []Sort_Key{sort_key("v")})
	defer delete(perm, context.allocator)
	testing.expect(t, p_err == .None, "argsort")
	testing.expect(t, len(perm) == 6, "permutation length")

	// perm must be a permutation of 0..5.
	seen := make([]bool, 6, context.allocator)
	defer delete(seen, context.allocator)
	for idx in perm {
		testing.expect(t, idx >= 0 && idx < 6, "index in range")
		seen[idx] = true
	}
	for i in 0 ..< 6 {
		testing.expect(t, seen[i], fmt.tprintf("index %d present", i))
	}

	// sorting rows by perm reproduces the sorted DataFrame's id order.
	sorted, s_err := dataframe_sort(&df, []Sort_Key{sort_key("v")})
	defer dataframe_destroy(&sorted)
	testing.expect(t, s_err == .None, "sort for comparison")
	id_col, _ := dataframe_get_column(&df, "id")
	idv := column_typed_view(id_col, i32)
	want := []i32{6, 5, 4, 3, 2, 1}
	for i in 0 ..< 6 {
		testing.expect(t, idv[perm[i]] == want[i], fmt.tprintf("perm row %d", i))
	}
}

@(test)
argsort_descending :: proc(t: ^testing.T) {
	df, err := sort_fixture(t)
	defer dataframe_destroy(&df)
	testing.expect(t, err == .None, "fixture")

	perm, p_err := dataframe_argsort(&df, []Sort_Key{sort_key("v", .Desc)})
	defer delete(perm, context.allocator)
	testing.expect(t, p_err == .None, "argsort desc")
	id_col, _ := dataframe_get_column(&df, "id")
	idv := column_typed_view(id_col, i32)
	want := []i32{1, 2, 3, 4, 5, 6}
	for i in 0 ..< 6 {
		testing.expect(t, idv[perm[i]] == want[i], fmt.tprintf("desc perm row %d", i))
	}
}

@(test)
sort_by_expression_key :: proc(t: ^testing.T) {
	x, e1 := column_from("x", []i32{3, -2, 1, -4, 5})
	id, e2 := column_from("id", []i32{1, 2, 3, 4, 5})
	testing.expect(t, e1 == .None && e2 == .None, "columns")
	df, df_err := dataframe_from_columns([]^Column{&x, &id})
	defer dataframe_destroy(&df)
	testing.expect(t, df_err == .None, "from_columns")

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	by := []^expr.Expr{expr.abs_(&ctx, expr.col(&ctx, "x"))}
	out, s_err := dataframe_sort_by(&df, by, nil)
	defer dataframe_destroy(&out)
	testing.expect(t, s_err == .None, "sort_by abs")
	// |x| asc: |1|, |-2|, |3|, |-4|, |5| -> ids 3, 2, 1, 4, 5
	expect_sorted_id(t, "sort_by abs", &out, []i32{3, 2, 1, 4, 5})
}

@(test)
sort_empty_dataframe :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	out, err := dataframe_sort(&df, []Sort_Key{})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .Invalid_Argument, "empty keys rejected")
}

@(test)
sort_single_column_df :: proc(t: ^testing.T) {
	x, e1 := column_from("x", []i32{3, 1, 2})
	testing.expect(t, e1 == .None, "column")
	df, df_err := dataframe_from_columns([]^Column{&x})
	defer dataframe_destroy(&df)
	testing.expect(t, df_err == .None, "from_columns")

	out, s_err := dataframe_sort(&df, []Sort_Key{sort_key("x")})
	defer dataframe_destroy(&out)
	testing.expect(t, s_err == .None, "single column sort")
	testing.expect(t, dataframe_num_cols(&out) == 1, "one column kept")
	xv := column_typed_view(&out.col_views[0], i32)
	testing.expect(t, xv[0] == 1 && xv[1] == 2 && xv[2] == 3, "values sorted")
}

@(test)
sort_errors :: proc(t: ^testing.T) {
	df, err := sort_fixture(t)
	defer dataframe_destroy(&df)
	testing.expect(t, err == .None, "fixture")

	_, err = dataframe_sort(&df, []Sort_Key{})
	testing.expect(t, err == .Invalid_Argument, "empty keys rejected")

	_, err = dataframe_sort(&df, []Sort_Key{sort_key("nope")})
	testing.expect(t, err == .Column_Not_Found, "unknown column rejected")

	_, err = dataframe_argsort(&df, []Sort_Key{sort_key("nope")})
	testing.expect(t, err == .Column_Not_Found, "argsort unknown column rejected")

	_, err = dataframe_sort_by(&df, nil, nil)
	testing.expect(t, err == .Invalid_Argument, "sort_by empty rejected")
}

@(test)
sort_unsortable_dtype :: proc(t: ^testing.T) {
	Point :: struct {
		x, y: i32,
	}
	p, e1 := column_from("p", []Point{{1, 2}, {3, 4}})
	testing.expect(t, e1 == .None, "column")
	df, df_err := dataframe_from_columns([]^Column{&p})
	defer dataframe_destroy(&df)
	testing.expect(t, df_err == .None, "from_columns")

	_, err := dataframe_sort(&df, []Sort_Key{sort_key("p")})
	testing.expect(t, err == .Unsupported_Operation, "struct dtype rejected")
}

// --- helpers ----------------------------------------------------------------

// df_from_cols builds a DataFrame from raw columns (for single-column tests).
df_from_cols :: proc(t: ^testing.T, cols: ..^Column) -> (out: DataFrame) {
	arr := make([]^Column, len(cols))
	defer delete(arr)
	for c, i in cols {
		arr[i] = c
	}
	d, err := dataframe_from_columns(arr)
	testing.expect(t, err == .None, "from_columns")
	return d
}
