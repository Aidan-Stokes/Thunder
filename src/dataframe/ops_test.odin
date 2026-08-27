package dataframe

// Stage 4 tests: filter, take, head, tail, slice, limit, select,
// select_by_name, with_columns, drop, unique, partition_by.

import "core:testing"
import "core:fmt"
import "expr"

// --- fixture -----------------------------------------------------------------

// ops_test_df builds the shared Stage 4 fixture:
//
//	id    i32    [1, 2, 3, 4, 5, 6]
//	g     i32    [1, 1, 2, 2, 3, 3]
//	x     i32    [10, 20, NULL, 40, 50, 60]      (row 2 NULL)
//	y     f64    [1.5, 2.5, 3.5, NULL, 5.5, 6.5] (row 3 NULL)
//	s     string ["a", "b", "c", "a", "b", "c"]
//	flag  bool   [true, false, true, false, true, false]
ops_test_df :: proc(t: ^testing.T) -> (df: DataFrame, ctx: expr.Context) {
	id, id_err := column_from("id", []i32{1, 2, 3, 4, 5, 6})
	testing.expect(t, id_err == .None, "id column")
	g, g_err := column_from("g", []i32{1, 1, 2, 2, 3, 3})
	testing.expect(t, g_err == .None, "g column")
	x, x_err := column_from("x", []i32{10, 20, 0, 40, 50, 60})
	testing.expect(t, x_err == .None, "x column")
	testing.expect(t, column_set_valid(&x, 2, false) == .None, "x[2] NULL")
	y, y_err := column_from("y", []f64{1.5, 2.5, 3.5, 0.0, 5.5, 6.5})
	testing.expect(t, y_err == .None, "y column")
	testing.expect(t, column_set_valid(&y, 3, false) == .None, "y[3] NULL")
	s, s_err := column_from("s", []string{"a", "b", "c", "a", "b", "c"})
	testing.expect(t, s_err == .None, "s column")
	flag, f_err := column_from("flag", []bool{true, false, true, false, true, false})
	testing.expect(t, f_err == .None, "flag column")

	d, df_err := dataframe_from_columns([]^Column{&id, &g, &x, &y, &s, &flag})
	testing.expect(t, df_err == .None, "from_columns")
	df = d
	ctx = expr.context_create(context.allocator)
	return
}
ops_test_destroy :: proc(t: ^testing.T, df: ^DataFrame, ctx: ^expr.Context) {
	expr.context_destroy(ctx)
	dataframe_destroy(df)
}

// expect_i32_col checks the values and validity of an i32 column.
expect_i32_col :: proc(t: ^testing.T, what: string, col: ^Column, want: []i32, want_valid: []bool = nil) {
	testing.expect(t, col.dtype == typeid_of(i32), fmt.tprintf("%s dtype", what))
	testing.expect(t, col.count == len(want), fmt.tprintf("%s row count", what))
	if col.count != len(want) {
		return
	}
	v := column_typed_view(col, i32)
	for i in 0 ..< len(want) {
		testing.expect(t, v[i] == want[i], fmt.tprintf("%s row value", what))
	}
	if want_valid != nil {
		for i in 0 ..< len(want) {
			testing.expect(t, column_is_valid(col, i) == want_valid[i], fmt.tprintf("%s row validity", what))
		}
	}
}

// build_small builds a two-column df: k (i32, with NULLs at rows 1,3) and
// id (i32) — used by NULL-grouping tests.
build_small :: proc(t: ^testing.T, valid_rows: []bool) -> DataFrame {
	k, k_err := column_from_with_valid("k", []i32{5, 0, 5, 0}, valid_rows)
	testing.expect(t, k_err == .None, "k column")
	id, id_err := column_from("id", []i32{1, 2, 3, 4})
	testing.expect(t, id_err == .None, "id column")
	df, df_err := dataframe_from_columns([]^Column{&k, &id})
	testing.expect(t, df_err == .None, "from_columns")
	return df
}

// --- filter ------------------------------------------------------------------

@(test)
filter_gt_literal :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_filter(&df, expr.gt(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(30))))
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "filter succeeds")
	testing.expect(t, dataframe_num_rows(&out) == 3, "kept row count")
	testing.expect(t, dataframe_num_cols(&out) == 6, "column count preserved")
	id_col, _ := dataframe_get_column(&out, "id")
	expect_i32_col(t, "id", id_col, []i32{4, 5, 6})
	x_col, _ := dataframe_get_column(&out, "x")
	expect_i32_col(t, "x", x_col, []i32{40, 50, 60}, []bool{true, true, true})
	y_col, _ := dataframe_get_column(&out, "y")
	testing.expect(t, y_col.dtype == typeid_of(f64) && column_is_valid(y_col, 0) == false, "NULL carried over")
}

@(test)
filter_null_predicate_excluded :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	// x[2] is NULL -> excluded even though the comparison would be true
	out, err := dataframe_filter(&df, expr.gt(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(0))))
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "filter succeeds")
	testing.expect(t, dataframe_num_rows(&out) == 5, "NULL row excluded (10, 20, 40, 50, 60 kept)")
	id_col, _ := dataframe_get_column(&out, "id")
	expect_i32_col(t, "id", id_col, []i32{1, 2, 4, 5, 6})
}

@(test)
filter_bool_column :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_filter(&df, expr.col(&ctx, "flag"))
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "filter on bool column")
	testing.expect(t, dataframe_num_rows(&out) == 3, "true rows only")
	id_col, _ := dataframe_get_column(&out, "id")
	expect_i32_col(t, "id", id_col, []i32{1, 3, 5})
}

@(test)
filter_non_bool_predicate :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	_, err := dataframe_filter(&df, expr.add(&ctx, expr.col(&ctx, "id"), expr.lit(&ctx, i32(1))))
	testing.expect(t, err == .Type_Mismatch, "numeric predicate is a type error")
}

@(test)
filter_zero_row_df :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	df0, err := dataframe_head(&df, 0)
	defer dataframe_destroy(&df0)
	testing.expect(t, err == .None, "head 0")

	out, f_err := dataframe_filter(&df0, expr.gt(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(0))))
	defer dataframe_destroy(&out)
	testing.expect(t, f_err == .None, "filter on 0-row df")
	testing.expect(t, dataframe_num_rows(&out) == 0, "0 rows out")
	testing.expect(t, dataframe_num_cols(&out) == 6, "columns preserved")
}

@(test)
filter_all_false_predicate :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_filter(&df, expr.eq(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(999))))
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "filter succeeds")
	testing.expect(t, dataframe_num_rows(&out) == 0, "no rows kept")
	testing.expect(t, dataframe_num_cols(&out) == 6, "schema kept on empty result")
}

@(test)
filter_unknown_column :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	_, err := dataframe_filter(&df, expr.gt(&ctx, expr.col(&ctx, "nope"), expr.lit(&ctx, i32(1))))
	testing.expect(t, err == .Column_Not_Found, "unknown column in predicate")
}

// --- take --------------------------------------------------------------------

@(test)
take_indices_duplicates :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_take(&df, []int{0, 2, 2, 5})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "take succeeds")
	testing.expect(t, dataframe_num_rows(&out) == 4, "row count")
	id_col, _ := dataframe_get_column(&out, "id")
	expect_i32_col(t, "id", id_col, []i32{1, 3, 3, 6})
}

@(test)
take_null_preserved :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_take(&df, []int{2, 3})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "take succeeds")
	x_col, _ := dataframe_get_column(&out, "x")
	expect_i32_col(t, "x", x_col, []i32{0, 40}, []bool{false, true})
	y_col, _ := dataframe_get_column(&out, "y")
	testing.expect(t, column_is_valid(y_col, 0) && !column_is_valid(y_col, 1), "NULL flags copied")
}

@(test)
take_out_of_bounds :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	_, err := dataframe_take(&df, []int{0, 6})
	testing.expect(t, err == .Out_Of_Bounds, "high index rejected")
	_, err = dataframe_take(&df, []int{-1})
	testing.expect(t, err == .Out_Of_Bounds, "negative index rejected")
}

@(test)
take_empty :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_take(&df, []int{})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "empty take")
	testing.expect(t, dataframe_num_rows(&out) == 0, "0 rows")
	testing.expect(t, dataframe_num_cols(&out) == 6, "columns kept")
}

// --- head / tail / slice / limit ---------------------------------------------

@(test)
head_tail_slice_limit :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	{
		head, err := dataframe_head(&df, 2)
		testing.expect(t, err == .None, "head 2")
		id_col, _ := dataframe_get_column(&head, "id")
		expect_i32_col(t, "head id", id_col, []i32{1, 2})
		dataframe_destroy(&head)
	}
	{
		head0, err := dataframe_head(&df, 0)
		testing.expect(t, err == .None && dataframe_num_rows(&head0) == 0, "head 0 empty")
		dataframe_destroy(&head0)
	}
	{
		headbig, err := dataframe_head(&df, 10)
		testing.expect(t, err == .None && dataframe_num_rows(&headbig) == 6, "head clamps to rows")
		dataframe_destroy(&headbig)
	}
	{
		_, err := dataframe_head(&df, -1)
		testing.expect(t, err == .Invalid_Argument, "head negative rejected")
	}
	{
		tail, err := dataframe_tail(&df, 2)
		testing.expect(t, err == .None, "tail 2")
		id_col, _ := dataframe_get_column(&tail, "id")
		expect_i32_col(t, "tail id", id_col, []i32{5, 6})
		dataframe_destroy(&tail)
	}
	{
		tail0, err := dataframe_tail(&df, 0)
		testing.expect(t, err == .None && dataframe_num_rows(&tail0) == 0, "tail 0 empty")
		dataframe_destroy(&tail0)
	}
	{
		slice1, err := dataframe_slice(&df, 2, 3)
		testing.expect(t, err == .None, "slice 2,3")
		id_col, _ := dataframe_get_column(&slice1, "id")
		expect_i32_col(t, "slice id", id_col, []i32{3, 4, 5})
		dataframe_destroy(&slice1)
	}
	{
		slice2, err := dataframe_slice(&df, 4, 10)
		testing.expect(t, err == .None, "slice clamps length")
		id_col, _ := dataframe_get_column(&slice2, "id")
		expect_i32_col(t, "clamped slice id", id_col, []i32{5, 6})
		dataframe_destroy(&slice2)
	}
	{
		_, err := dataframe_slice(&df, -1, 2)
		testing.expect(t, err == .Invalid_Argument, "negative offset rejected")
	}
	{
		_, err := dataframe_slice(&df, 2, -1)
		testing.expect(t, err == .Invalid_Argument, "negative length rejected")
	}
	{
		limit, err := dataframe_limit(&df, 2)
		testing.expect(t, err == .None, "limit")
		id_col, _ := dataframe_get_column(&limit, "id")
		expect_i32_col(t, "limit id", id_col, []i32{1, 2})
		dataframe_destroy(&limit)
	}
}

// --- select ------------------------------------------------------------------

@(test)
select_columns_order :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_select(&df, []^expr.Expr{expr.col(&ctx, "id"), expr.col(&ctx, "s")})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "select succeeds")
	testing.expect(t, dataframe_num_cols(&out) == 2, "two columns")
	testing.expect(t, dataframe_num_rows(&out) == 6, "row count")
	id_col, _ := dataframe_get_column(&out, "id")
	expect_i32_col(t, "id", id_col, []i32{1, 2, 3, 4, 5, 6})
	s_col, _ := dataframe_get_column(&out, "s")
	testing.expect(t, s_col.dtype == typeid_of(string), "s dtype")
	sv := column_typed_view(s_col, string)
	want_s := []string{"a", "b", "c", "a", "b", "c"}
	for i in 0 ..< 6 {
		testing.expect(t, sv[i] == want_s[i], "s value")
	}
}

@(test)
select_single_column_null_preserved :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_select(&df, []^expr.Expr{expr.col(&ctx, "x")})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "select succeeds")
	x_col, _ := dataframe_get_column(&out, "x")
	expect_i32_col(t, "x", x_col, []i32{10, 20, 0, 40, 50, 60}, []bool{true, true, false, true, true, true})
}

@(test)
select_unnamed_expr :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	_, err := dataframe_select(&df, []^expr.Expr{expr.add(&ctx, expr.col(&ctx, "id"), expr.lit(&ctx, i32(1)))})
	testing.expect(t, err == .Invalid_Argument, "unnamed computed expr rejected")
}

@(test)
select_alias :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_select(&df, []^expr.Expr{expr.alias(&ctx, expr.add(&ctx, expr.col(&ctx, "id"), expr.lit(&ctx, i32(1))), "id+1")})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "select aliased expr")
	testing.expect(t, dataframe_has_column(&out, "id+1"), "aliased name present")
	plus, _ := dataframe_get_column(&out, "id+1")
	expect_i32_col(t, "id+1", plus, []i32{2, 3, 4, 5, 6, 7})
}

@(test)
select_duplicate_names :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	_, err := dataframe_select(&df, []^expr.Expr{expr.col(&ctx, "x"), expr.col(&ctx, "x")})
	testing.expect(t, err == .Duplicate_Column_Name, "duplicate result name rejected")
}

@(test)
select_unknown_column :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	_, err := dataframe_select(&df, []^expr.Expr{expr.col(&ctx, "nope")})
	testing.expect(t, err == .Column_Not_Found, "unknown column rejected")
}

@(test)
select_empty :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_select(&df, []^expr.Expr{})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "empty select")
	testing.expect(t, dataframe_num_cols(&out) == 0 && dataframe_num_rows(&out) == 0, "empty result")
}

@(test)
select_by_name :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_select_by_name(&df, []string{"s", "id"})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "select_by_name succeeds")
	testing.expect(t, dataframe_num_cols(&out) == 2, "two columns")
	testing.expect(t, dataframe_num_rows(&out) == 6, "row count")
	s_col, _ := dataframe_get_column(&out, "s")
	testing.expect(t, s_col != nil, "first column is s")
	id_col, _ := dataframe_get_column(&out, "id")
	expect_i32_col(t, "id", id_col, []i32{1, 2, 3, 4, 5, 6})
}

@(test)
select_by_name_errors :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	_, err := dataframe_select_by_name(&df, []string{"nope"})
	testing.expect(t, err == .Column_Not_Found, "unknown column rejected")
	_, err = dataframe_select_by_name(&df, []string{"x", "x"})
	testing.expect(t, err == .Duplicate_Column_Name, "duplicate names rejected")
}

// --- with_columns / drop ------------------------------------------------------

@(test)
with_columns_append :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_with_columns(&df, []^expr.Expr{expr.alias(&ctx, expr.add(&ctx, expr.col(&ctx, "id"), expr.lit(&ctx, i32(1))), "id2")})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "with_columns succeeds")
	testing.expect(t, dataframe_num_cols(&out) == 7, "appended column")
	testing.expect(t, dataframe_has_column(&out, "id2"), "id2 present")
	testing.expect(t, !dataframe_has_column(&df, "id2"), "source unchanged")

	id2, _ := dataframe_get_column(&out, "id2")
	expect_i32_col(t, "id2", id2, []i32{2, 3, 4, 5, 6, 7})

	// verify source df untouched
	testing.expect(t, dataframe_num_cols(&df) == 6 && dataframe_num_rows(&df) == 6, "source intact")
}

@(test)
with_columns_replace :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_with_columns(&df, []^expr.Expr{expr.alias(&ctx, expr.add(&ctx, expr.col(&ctx, "id"), expr.lit(&ctx, i32(10))), "id")})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "with_columns replace succeeds")
	testing.expect(t, dataframe_num_cols(&out) == 6, "column count unchanged")
	id_col, _ := dataframe_get_column(&out, "id")
	expect_i32_col(t, "id replaced", id_col, []i32{11, 12, 13, 14, 15, 16})
	s_col, _ := dataframe_get_column(&out, "s")
	testing.expect(t, s_col != nil, "other columns preserved")
}

@(test)
with_columns_unnamed_expr :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	_, err := dataframe_with_columns(&df, []^expr.Expr{expr.add(&ctx, expr.col(&ctx, "id"), expr.lit(&ctx, i32(1)))})
	testing.expect(t, err == .Invalid_Argument, "unnamed expr rejected")
}

@(test)
with_columns_empty_is_copy :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_with_columns(&df, []^expr.Expr{})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "empty with_columns")
	testing.expect(t, dataframe_num_cols(&out) == 6 && dataframe_num_rows(&out) == 6, "deep copy")
	x_col, _ := dataframe_get_column(&out, "x")
	expect_i32_col(t, "x", x_col, []i32{10, 20, 0, 40, 50, 60}, []bool{true, true, false, true, true, true})
}

@(test)
drop_columns :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_drop(&df, []string{"g"})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "drop succeeds")
	testing.expect(t, dataframe_num_cols(&out) == 5, "one column dropped")
	testing.expect(t, !dataframe_has_column(&out, "g"), "g gone")
	testing.expect(t, dataframe_has_column(&out, "id") && dataframe_has_column(&out, "s"), "others kept")
}

@(test)
drop_empty_and_all :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	copy_df, copy_err := dataframe_drop(&df, []string{})
	defer dataframe_destroy(&copy_df)
	testing.expect(t, copy_err == .None && dataframe_num_cols(&copy_df) == 6, "empty drop is a copy")

	empty, err := dataframe_drop(&df, []string{"id", "g", "x", "y", "s", "flag"})
	defer dataframe_destroy(&empty)
	testing.expect(t, err == .None && dataframe_num_cols(&empty) == 0, "drop all")
}

@(test)
drop_unknown_column :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	_, err := dataframe_drop(&df, []string{"nope"})
	testing.expect(t, err == .Column_Not_Found, "unknown column rejected")
}

// --- unique ------------------------------------------------------------------

@(test)
unique_all_columns :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_unique(&df, []string{})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "unique over all columns")
	testing.expect(t, dataframe_num_rows(&out) == 6, "all rows distinct by id")
}

@(test)
unique_subset :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_unique(&df, []string{"s"})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "unique over s")
	testing.expect(t, dataframe_num_rows(&out) == 3, "first occurrence of a, b, c")
	id_col, _ := dataframe_get_column(&out, "id")
	expect_i32_col(t, "id", id_col, []i32{1, 2, 3})
}

@(test)
unique_multikey :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_unique(&df, []string{"g", "s"})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "unique over g,s")
	testing.expect(t, dataframe_num_rows(&out) == 6, "every (g,s) pair distinct")
}

@(test)
unique_key_column :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out, err := dataframe_unique(&df, []string{"g"})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "unique over g")
	testing.expect(t, dataframe_num_rows(&out) == 3, "three groups")
	id_col, _ := dataframe_get_column(&out, "id")
	expect_i32_col(t, "id", id_col, []i32{1, 3, 5})
}

@(test)
unique_null_rows_collapse :: proc(t: ^testing.T) {
	small := build_small(t, []bool{true, false, true, false})
	defer dataframe_destroy(&small)

	// k = [5, NULL, 5, NULL] -> distinct values: 5 (row 0), NULL (row 1)
	out, err := dataframe_unique(&small, []string{"k"})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "unique with NULL keys")
	testing.expect(t, dataframe_num_rows(&out) == 2, "one value group + one NULL group")
	id_col, _ := dataframe_get_column(&out, "id")
	expect_i32_col(t, "id", id_col, []i32{1, 2})
}

@(test)
unique_errors :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	_, err := dataframe_unique(&df, []string{"nope"})
	testing.expect(t, err == .Column_Not_Found, "unknown column rejected")
	_, err = dataframe_unique(&df, []string{"s", "s"})
	testing.expect(t, err == .Invalid_Argument, "duplicate names rejected")
}

@(test)
unique_empty_df :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)
	out, err := dataframe_unique(&df, []string{})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "unique on empty df")
	testing.expect(t, dataframe_num_cols(&out) == 0 && dataframe_num_rows(&out) == 0, "empty result")
}

@(test)
unique_float_canonicalization :: proc(t: ^testing.T) {
	// -0.0 and +0.0 share a key; 1.5 and 1.5 share a key
	vals := []f64{-0.0, 0.0, 1.5, 1.5, 2.0}
	f, f_err := column_from("f", vals)
	testing.expect(t, f_err == .None, "f column")
	id, id_err := column_from("id", []i32{1, 2, 3, 4, 5})
	testing.expect(t, id_err == .None, "id column")
	df, df_err := dataframe_from_columns([]^Column{&f, &id})
	defer dataframe_destroy(&df)
	testing.expect(t, df_err == .None, "from_columns")

	out, err := dataframe_unique(&df, []string{"f"})
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "unique floats")
	testing.expect(t, dataframe_num_rows(&out) == 3, "groups: {0.0}, {1.5}, {2.0}")
	id_col, _ := dataframe_get_column(&out, "id")
	expect_i32_col(t, "id", id_col, []i32{1, 3, 5})
}

// --- partition_by --------------------------------------------------------------

@(test)
partition_by_key_column :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	parts, err := dataframe_partition_by(&df, []^expr.Expr{expr.col(&ctx, "g")})
	defer dataframe_partitions_destroy(parts, context.allocator)
	testing.expect(t, err == .None, "partition_by succeeds")
	testing.expect(t, len(parts) == 3, "three partitions")

	// first-appearance order: g=1, then g=2, then g=3
	expect_i32_col(t, "p0 id", &parts[0].col_views[0], []i32{1, 2})
	expect_i32_col(t, "p1 id", &parts[1].col_views[0], []i32{3, 4})
	expect_i32_col(t, "p2 id", &parts[2].col_views[0], []i32{5, 6})
}

@(test)
partition_by_string_column :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	parts, err := dataframe_partition_by(&df, []^expr.Expr{expr.col(&ctx, "s")})
	defer dataframe_partitions_destroy(parts, context.allocator)
	testing.expect(t, err == .None, "partition_by succeeds")
	testing.expect(t, len(parts) == 3, "three partitions of two rows each")
	for &p in parts {
		testing.expect(t, dataframe_num_rows(&p) == 2, "two rows per partition")
	}
}

@(test)
partition_by_null_keys :: proc(t: ^testing.T) {
	small := build_small(t, []bool{true, false, true, false})
	defer dataframe_destroy(&small)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	parts, err := dataframe_partition_by(&small, []^expr.Expr{expr.col(&ctx, "k")})
	defer dataframe_partitions_destroy(parts, context.allocator)
	testing.expect(t, err == .None, "partition_by with NULL keys")
	testing.expect(t, len(parts) == 2, "one value partition + one NULL partition")
	// first partition is the value 5 (rows 0,2), second is NULL (rows 1,3)
	id0, _ := dataframe_get_column(&parts[0], "id")
	expect_i32_col(t, "p0 id", id0, []i32{1, 3})
	id1, _ := dataframe_get_column(&parts[1], "id")
	expect_i32_col(t, "p1 id", id1, []i32{2, 4})
}

@(test)
partition_by_errors :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	parts, err := dataframe_partition_by(&df, []^expr.Expr{})
	testing.expect(t, err == .Invalid_Argument, "no keys rejected")
	if parts != nil {
		dataframe_partitions_destroy(parts, context.allocator)
	}
	_, err = dataframe_partition_by(&df, []^expr.Expr{expr.col(&ctx, "nope")})
	testing.expect(t, err == .Column_Not_Found, "unknown key rejected")
}

@(test)
partition_by_single_column_df :: proc(t: ^testing.T) {
	id, id_err := column_from("id", []i32{7, 8, 9})
	testing.expect(t, id_err == .None, "id column")
	df, df_err := dataframe_from_columns([]^Column{&id})
	defer dataframe_destroy(&df)
	testing.expect(t, df_err == .None, "from_columns")

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	parts, err := dataframe_partition_by(&df, []^expr.Expr{expr.col(&ctx, "id")})
	defer dataframe_partitions_destroy(parts, context.allocator)
	testing.expect(t, err == .None, "partition_by on single column")
	testing.expect(t, len(parts) == 3, "three one-row partitions")
	for &p, i in parts {
		testing.expect(t, dataframe_num_rows(&p) == 1, "one row")
		expect_i32_col(t, "id", &p.col_views[0], []i32{i32(i + 7)})
	}
}

// --- misc ---------------------------------------------------------------------

@(test)
ops_source_unchanged :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	out1, _ := dataframe_filter(&df, expr.gt(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(30))))
	defer dataframe_destroy(&out1)
	out2, _ := dataframe_select(&df, []^expr.Expr{expr.col(&ctx, "x")})
	defer dataframe_destroy(&out2)
	out3, _ := dataframe_take(&df, []int{2, 3})
	defer dataframe_destroy(&out3)
	out4, _ := dataframe_with_columns(&df, []^expr.Expr{expr.alias(&ctx, expr.col(&ctx, "id"), "id2")})
	defer dataframe_destroy(&out4)

	// source df fully untouched
	testing.expect(t, dataframe_num_cols(&df) == 6 && dataframe_num_rows(&df) == 6, "source intact")
	x_col, _ := dataframe_get_column(&df, "x")
	expect_i32_col(t, "x", x_col, []i32{10, 20, 0, 40, 50, 60}, []bool{true, true, false, true, true, true})
	y_col, _ := dataframe_get_column(&df, "y")
	testing.expect(t, y_col != nil && !column_is_valid(y_col, 3), "source NULL flags intact")
}

@(test)
ops_null_filled_column_filter :: proc(t: ^testing.T) {
	// a single-column df filtered on a NULL-only column yields 0 rows
	b, b_err := column_from_with_valid("b", []i32{0, 0, 0}, []bool{false, false, false})
	testing.expect(t, b_err == .None, "b column")
	df, df_err := dataframe_from_columns([]^Column{&b})
	defer dataframe_destroy(&df)
	testing.expect(t, df_err == .None, "from_columns")

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	out, err := dataframe_filter(&df, expr.eq(&ctx, expr.col(&ctx, "b"), expr.lit(&ctx, i32(0))))
	defer dataframe_destroy(&out)
	testing.expect(t, err == .None, "filter succeeds")
	testing.expect(t, dataframe_num_rows(&out) == 0, "NULL rows excluded")
	testing.expect(t, dataframe_num_cols(&out) == 1, "schema kept")
}
