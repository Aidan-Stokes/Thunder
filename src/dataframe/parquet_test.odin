package dataframe

import "core:testing"
import "core:fmt"

// ── Parquet roundtrip tests ────────────────────────────────────────────────

@(test)
parquet_roundtrip_i64 :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("ids", []i64{10, 20, 30, 40, 50})
	testing.expect(t, err == .None, "create i64 column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add ids")

	data, w_err := dataframe_write_parquet(&df)
	testing.expect(t, w_err == .None, "write parquet")
	defer delete(data)

	df2, ok := dataframe_read_parquet(data)
	testing.expect(t, ok, "read parquet")
	defer dataframe_destroy(&df2)

	testing.expect(t, dataframe_num_rows(&df2) == 5, "row count")
	testing.expect(t, dataframe_num_cols(&df2) == 1, "col count")

	got, g_err := dataframe_get_column(&df2, "ids")
	testing.expect(t, g_err == .None, "get ids back")
	testing.expect(t, got.dtype == typeid_of(i64), "dtype i64")

	view := column_typed_view(got, i64)
	want := []i64{10, 20, 30, 40, 50}
	for i in 0 ..< len(want) {
		testing.expect(t, view[i] == want[i], fmt.tprintf("ids[%d] = %d want %d", i, view[i], want[i]))
	}
}

@(test)
parquet_roundtrip_i32 :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("x", []i32{1, -2, 3, 0, 100})
	testing.expect(t, err == .None, "create i32 column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add x")

	data, w_err := dataframe_write_parquet(&df)
	testing.expect(t, w_err == .None, "write parquet")
	defer delete(data)

	df2, ok := dataframe_read_parquet(data)
	testing.expect(t, ok, "read parquet")
	defer dataframe_destroy(&df2)

	got, g_err := dataframe_get_column(&df2, "x")
	testing.expect(t, g_err == .None, "get x back")
	testing.expect(t, got.dtype == typeid_of(i32), "dtype i32")

	view := column_typed_view(got, i32)
	want := []i32{1, -2, 3, 0, 100}
	for i in 0 ..< len(want) {
		testing.expect(t, view[i] == want[i], fmt.tprintf("x[%d] = %d want %d", i, view[i], want[i]))
	}
}

@(test)
parquet_roundtrip_f64 :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("vals", []f64{1.5, -2.7, 0.0, 100.123})
	testing.expect(t, err == .None, "create f64 column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add vals")

	data, w_err := dataframe_write_parquet(&df)
	testing.expect(t, w_err == .None, "write parquet")
	defer delete(data)

	df2, ok := dataframe_read_parquet(data)
	testing.expect(t, ok, "read parquet")
	defer dataframe_destroy(&df2)

	got, g_err := dataframe_get_column(&df2, "vals")
	testing.expect(t, g_err == .None, "get vals back")
	testing.expect(t, got.dtype == typeid_of(f64), "dtype f64")

	view := column_typed_view(got, f64)
	want := []f64{1.5, -2.7, 0.0, 100.123}
	for i in 0 ..< len(want) {
		testing.expect(t, view[i] == want[i], fmt.tprintf("vals[%d] = %v want %v", i, view[i], want[i]))
	}
}

@(test)
parquet_roundtrip_f32 :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("f", []f32{1.0, 2.5, -3.75})
	testing.expect(t, err == .None, "create f32 column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add f")

	data, w_err := dataframe_write_parquet(&df)
	testing.expect(t, w_err == .None, "write parquet")
	defer delete(data)

	df2, ok := dataframe_read_parquet(data)
	testing.expect(t, ok, "read parquet")
	defer dataframe_destroy(&df2)

	got, g_err := dataframe_get_column(&df2, "f")
	testing.expect(t, g_err == .None, "get f back")
	testing.expect(t, got.dtype == typeid_of(f32), "dtype f32")

	view := column_typed_view(got, f32)
	want := []f32{1.0, 2.5, -3.75}
	for i in 0 ..< len(want) {
		testing.expect(t, view[i] == want[i], fmt.tprintf("f[%d] = %v want %v", i, view[i], want[i]))
	}
}

@(test)
parquet_roundtrip_bool :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("flags", []bool{true, false, true, true, false})
	testing.expect(t, err == .None, "create bool column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add flags")

	data, w_err := dataframe_write_parquet(&df)
	testing.expect(t, w_err == .None, "write parquet")
	defer delete(data)

	df2, ok := dataframe_read_parquet(data)
	testing.expect(t, ok, "read parquet")
	defer dataframe_destroy(&df2)

	got, g_err := dataframe_get_column(&df2, "flags")
	testing.expect(t, g_err == .None, "get flags back")
	testing.expect(t, got.dtype == typeid_of(bool), "dtype bool")

	view := column_typed_view(got, bool)
	want := []bool{true, false, true, true, false}
	for i in 0 ..< len(want) {
		testing.expect(t, view[i] == want[i], fmt.tprintf("flags[%d] = %v want %v", i, view[i], want[i]))
	}
}

@(test)
parquet_roundtrip_string :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("names", []string{"hello", "world", "odin", ""})
	testing.expect(t, err == .None, "create string column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add names")

	data, w_err := dataframe_write_parquet(&df)
	testing.expect(t, w_err == .None, "write parquet")
	defer delete(data)

	df2, ok := dataframe_read_parquet(data)
	testing.expect(t, ok, "read parquet")
	defer dataframe_destroy(&df2)

	got, g_err := dataframe_get_column(&df2, "names")
	testing.expect(t, g_err == .None, "get names back")

	view := column_typed_view(got, string)
	want := []string{"hello", "world", "odin", ""}
	for i in 0 ..< len(want) {
		testing.expect(t, view[i] == want[i], fmt.tprintf("names[%d] = '%s' want '%s'", i, view[i], want[i]))
	}
}

@(test)
parquet_roundtrip_multi_column :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	c1, err1 := column_from("id", []i64{1, 2, 3})
	testing.expect(t, err1 == .None, "create id")
	testing.expect(t, dataframe_add_column(&df, &c1) == .None, "add id")

	c2, err2 := column_from("score", []f64{10.5, 20.0, 30.75})
	testing.expect(t, err2 == .None, "create score")
	testing.expect(t, dataframe_add_column(&df, &c2) == .None, "add score")

	c3, err3 := column_from("label", []string{"a", "b", "c"})
	testing.expect(t, err3 == .None, "create label")
	testing.expect(t, dataframe_add_column(&df, &c3) == .None, "add label")

	data, w_err := dataframe_write_parquet(&df)
	testing.expect(t, w_err == .None, "write parquet")
	defer delete(data)

	df2, ok := dataframe_read_parquet(data)
	testing.expect(t, ok, "read parquet")
	defer dataframe_destroy(&df2)

	testing.expect(t, dataframe_num_rows(&df2) == 3, "row count")
	testing.expect(t, dataframe_num_cols(&df2) == 3, "col count")

	// Verify id column
	got_id, gid_err := dataframe_get_column(&df2, "id")
	testing.expect(t, gid_err == .None, "get id")
	view_id := column_typed_view(got_id, i64)
	testing.expect(t, view_id[0] == 1, "id[0]")
	testing.expect(t, view_id[1] == 2, "id[1]")
	testing.expect(t, view_id[2] == 3, "id[2]")

	// Verify score column
	got_score, gs_err := dataframe_get_column(&df2, "score")
	testing.expect(t, gs_err == .None, "get score")
	view_score := column_typed_view(got_score, f64)
	testing.expect(t, view_score[0] == 10.5, "score[0]")
	testing.expect(t, view_score[1] == 20.0, "score[1]")
	testing.expect(t, view_score[2] == 30.75, "score[2]")

	// Verify label column
	got_label, gl_err := dataframe_get_column(&df2, "label")
	testing.expect(t, gl_err == .None, "get label")
	view_label := column_typed_view(got_label, string)
	testing.expect(t, view_label[0] == "a", "label[0]")
	testing.expect(t, view_label[1] == "b", "label[1]")
	testing.expect(t, view_label[2] == "c", "label[2]")
}

@(test)
parquet_roundtrip_large :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	n := 1000
	vals := make([]i64, n, context.allocator)
	defer delete(vals)
	for i in 0 ..< n {
		vals[i] = i64(i * 7)
	}

	col, err := column_from("big", vals)
	testing.expect(t, err == .None, "create large column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add big")

	data, w_err := dataframe_write_parquet(&df)
	testing.expect(t, w_err == .None, "write parquet")
	defer delete(data)

	df2, ok := dataframe_read_parquet(data)
	testing.expect(t, ok, "read parquet")
	defer dataframe_destroy(&df2)

	testing.expect(t, dataframe_num_rows(&df2) == n, "row count")

	got, g_err := dataframe_get_column(&df2, "big")
	testing.expect(t, g_err == .None, "get big back")
	view := column_typed_view(got, i64)

	for i in 0 ..< n {
		testing.expect(t, view[i] == vals[i], fmt.tprintf("big[%d] = %d want %d", i, view[i], vals[i]))
	}
}

@(test)
parquet_roundtrip_empty :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	empty: []i64
	col, err := column_from("empty_col", empty)
	testing.expect(t, err == .None, "create empty column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add empty")

	data, w_err := dataframe_write_parquet(&df)
	testing.expect(t, w_err == .None, "write parquet")
	defer delete(data)

	df2, ok := dataframe_read_parquet(data)
	testing.expect(t, ok, "read parquet")
	defer dataframe_destroy(&df2)

	testing.expect(t, dataframe_num_rows(&df2) == 0, "row count")
	testing.expect(t, dataframe_num_cols(&df2) == 1, "col count")
}

@(test)
parquet_write_empty_df_fails :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	_, err := dataframe_write_parquet(&df)
	testing.expect(t, err == .Invalid_Argument, "write empty df should fail")
}
