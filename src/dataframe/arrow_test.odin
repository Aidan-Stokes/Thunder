package dataframe

import "core:testing"
import "core:os"
import "core:fmt"

@(test)
arrow_roundtrip_i64 :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_i64.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("ids", []i64{10, 20, 30, 40, 50})
	testing.expect(t, err == .None, "create ids column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add ids")

	w_err := dataframe_write_arrow(&df, path)
	testing.expect(t, w_err == .None, "write arrow")

	df2, r_err := dataframe_read_arrow(path)
	testing.expect(t, r_err == .None, "read arrow")
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
arrow_roundtrip_f64 :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_f64.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("values", []f64{1.5, 2.5, 3.5})
	testing.expect(t, err == .None, "create f64 column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add values")

	testing.expect(t, dataframe_write_arrow(&df, path) == .None, "write")

	df2, r_err := dataframe_read_arrow(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df2)

	got, _ := dataframe_get_column(&df2, "values")
	view := column_typed_view(got, f64)
	testing.expect(t, view[0] == 1.5, "values[0]")
	testing.expect(t, view[1] == 2.5, "values[1]")
	testing.expect(t, view[2] == 3.5, "values[2]")
}

@(test)
arrow_roundtrip_bool :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_bool.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("flags", []bool{true, false, true, true})
	testing.expect(t, err == .None, "create bool column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add flags")

	testing.expect(t, dataframe_write_arrow(&df, path) == .None, "write")

	df2, r_err := dataframe_read_arrow(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df2)

	got, _ := dataframe_get_column(&df2, "flags")
	view := column_typed_view(got, bool)
	want := []bool{true, false, true, true}
	for i in 0 ..< len(want) {
		testing.expect(t, view[i] == want[i], fmt.tprintf("flags[%d] = %v want %v", i, view[i], want[i]))
	}
}

@(test)
arrow_roundtrip_string :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_str.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("names", []string{"alice", "bob", "charlie"})
	testing.expect(t, err == .None, "create string column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add names")

	testing.expect(t, dataframe_write_arrow(&df, path) == .None, "write")

	df2, r_err := dataframe_read_arrow(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df2)

	got, _ := dataframe_get_column(&df2, "names")
	view := column_typed_view(got, string)
	testing.expect(t, view[0] == "alice", "names[0]")
	testing.expect(t, view[1] == "bob", "names[1]")
	testing.expect(t, view[2] == "charlie", "names[2]")
}

@(test)
arrow_roundtrip_string_with_nulls :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_strnull.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from_with_valid("labels", []string{"a", "b", "c", "d"}, []bool{true, false, true, false})
	testing.expect(t, err == .None, "create string column with nulls")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add labels")

	testing.expect(t, dataframe_write_arrow(&df, path) == .None, "write")

	df2, r_err := dataframe_read_arrow(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df2)

	got, _ := dataframe_get_column(&df2, "labels")
	testing.expect(t, column_is_valid(got, 0), "labels[0] valid")
	testing.expect(t, !column_is_valid(got, 1), "labels[1] null")
	testing.expect(t, column_is_valid(got, 2), "labels[2] valid")
	testing.expect(t, !column_is_valid(got, 3), "labels[3] null")

	view := column_typed_view(got, string)
	testing.expect(t, view[0] == "a", "labels[0]")
	testing.expect(t, view[2] == "c", "labels[2]")
}

@(test)
arrow_roundtrip_i32_with_nulls :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_i32null.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from_with_valid("scores", []i32{100, 0, 300}, []bool{true, false, true})
	testing.expect(t, err == .None, "create i32 column with nulls")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add scores")

	testing.expect(t, dataframe_write_arrow(&df, path) == .None, "write")

	df2, r_err := dataframe_read_arrow(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df2)

	got, _ := dataframe_get_column(&df2, "scores")
	testing.expect(t, column_is_valid(got, 0), "scores[0] valid")
	testing.expect(t, !column_is_valid(got, 1), "scores[1] null")
	testing.expect(t, column_is_valid(got, 2), "scores[2] valid")

	view := column_typed_view(got, i32)
	testing.expect(t, view[0] == 100, "scores[0]")
	testing.expect(t, view[2] == 300, "scores[2]")
}

@(test)
arrow_roundtrip_multi_column :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_multi.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	c1, err1 := column_from("id", []i64{1, 2, 3})
	c2, err2 := column_from("name", []string{"x", "y", "z"})
	c3, err3 := column_from("val", []f64{1.0, 2.0, 3.0})
	testing.expect(t, err1 == .None && err2 == .None && err3 == .None, "create columns")
	testing.expect(t, dataframe_add_column(&df, &c1) == .None, "add id")
	testing.expect(t, dataframe_add_column(&df, &c2) == .None, "add name")
	testing.expect(t, dataframe_add_column(&df, &c3) == .None, "add val")

	testing.expect(t, dataframe_write_arrow(&df, path) == .None, "write")

	df2, r_err := dataframe_read_arrow(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df2)

	testing.expect(t, dataframe_num_rows(&df2) == 3, "row count")
	testing.expect(t, dataframe_num_cols(&df2) == 3, "col count")

	id_col, _ := dataframe_get_column(&df2, "id")
	id_view := column_typed_view(id_col, i64)
	testing.expect(t, id_view[0] == 1 && id_view[1] == 2 && id_view[2] == 3, "id values")

	name_col, _ := dataframe_get_column(&df2, "name")
	name_view := column_typed_view(name_col, string)
	testing.expect(t, name_view[0] == "x" && name_view[1] == "y" && name_view[2] == "z", "name values")

	val_col, _ := dataframe_get_column(&df2, "val")
	val_view := column_typed_view(val_col, f64)
	testing.expect(t, val_view[0] == 1.0 && val_view[1] == 2.0 && val_view[2] == 3.0, "val values")
}

@(test)
arrow_roundtrip_empty :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_empty.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	w_err := dataframe_write_arrow(&df, path)
	testing.expect(t, w_err == .Invalid_Argument, "write empty df fails")

	_, r_err := dataframe_read_arrow("/tmp/arrow_nonexistent_file.arrow")
	testing.expect(t, r_err == .Invalid_Schema, "read nonexistent fails")
}

@(test)
arrow_roundtrip_u8_u16 :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_u8u16.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	c1, err1 := column_from("bytes", []u8{0, 1, 255})
	c2, err2 := column_from("shorts", []u16{1000, 2000, 30000})
	testing.expect(t, err1 == .None && err2 == .None, "create columns")
	testing.expect(t, dataframe_add_column(&df, &c1) == .None, "add bytes")
	testing.expect(t, dataframe_add_column(&df, &c2) == .None, "add shorts")

	testing.expect(t, dataframe_write_arrow(&df, path) == .None, "write")

	df2, r_err := dataframe_read_arrow(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df2)

	b_col, _ := dataframe_get_column(&df2, "bytes")
	b_view := column_typed_view(b_col, u8)
	testing.expect(t, b_view[0] == 0 && b_view[1] == 1 && b_view[2] == 255, "u8 roundtrip")

	s_col, _ := dataframe_get_column(&df2, "shorts")
	s_view := column_typed_view(s_col, u16)
	testing.expect(t, s_view[0] == 1000 && s_view[1] == 2000 && s_view[2] == 30000, "u16 roundtrip")
}

@(test)
arrow_stream_roundtrip_i64 :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_stream_i64.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("ids", []i64{10, 20, 30})
	testing.expect(t, err == .None, "create ids column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add ids")

	w_err := dataframe_write_arrow_stream(&df, path)
	testing.expect(t, w_err == .None, "write stream")

	df2, r_err := dataframe_read_arrow_stream(path)
	testing.expect(t, r_err == .None, "read stream")
	defer dataframe_destroy(&df2)

	testing.expect(t, dataframe_num_rows(&df2) == 3, "row count")
	got, g_err := dataframe_get_column(&df2, "ids")
	testing.expect(t, g_err == .None, "get ids back")
	view := column_typed_view(got, i64)
	testing.expect(t, view[0] == 10 && view[1] == 20 && view[2] == 30, "ids values")
}

@(test)
arrow_stream_roundtrip_string_with_nulls :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_stream_strnull.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from_with_valid("labels", []string{"a", "b", "c"}, []bool{true, false, true})
	testing.expect(t, err == .None, "create string column with nulls")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add labels")

	testing.expect(t, dataframe_write_arrow_stream(&df, path) == .None, "write stream")

	df2, r_err := dataframe_read_arrow_stream(path)
	testing.expect(t, r_err == .None, "read stream")
	defer dataframe_destroy(&df2)

	got, _ := dataframe_get_column(&df2, "labels")
	testing.expect(t, column_is_valid(got, 0), "labels[0] valid")
	testing.expect(t, !column_is_valid(got, 1), "labels[1] null")
	testing.expect(t, column_is_valid(got, 2), "labels[2] valid")

	view := column_typed_view(got, string)
	testing.expect(t, view[0] == "a", "labels[0]")
	testing.expect(t, view[2] == "c", "labels[2]")
}

@(test)
arrow_stream_roundtrip_multi_column :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_stream_multi.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	c1, _ := column_from("id", []i64{1, 2, 3})
	c2, _ := column_from("name", []string{"x", "y", "z"})
	c3, _ := column_from("val", []f64{1.0, 2.0, 3.0})
	testing.expect(t, dataframe_add_column(&df, &c1) == .None, "add id")
	testing.expect(t, dataframe_add_column(&df, &c2) == .None, "add name")
	testing.expect(t, dataframe_add_column(&df, &c3) == .None, "add val")

	testing.expect(t, dataframe_write_arrow_stream(&df, path) == .None, "write stream")

	df2, r_err := dataframe_read_arrow_stream(path)
	testing.expect(t, r_err == .None, "read stream")
	defer dataframe_destroy(&df2)

	testing.expect(t, dataframe_num_rows(&df2) == 3, "row count")
	testing.expect(t, dataframe_num_cols(&df2) == 3, "col count")

	id_col, _ := dataframe_get_column(&df2, "id")
	id_view := column_typed_view(id_col, i64)
	testing.expect(t, id_view[0] == 1 && id_view[1] == 2 && id_view[2] == 3, "id values")

	name_col, _ := dataframe_get_column(&df2, "name")
	name_view := column_typed_view(name_col, string)
	testing.expect(t, name_view[0] == "x" && name_view[1] == "y" && name_view[2] == "z", "name values")
}

@(test)
arrow_roundtrip_date :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_date.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	d1, d1_ok := date_create(2024, 1, 15)
	d2, d2_ok := date_create(2024, 6, 30)
	d3, d3_ok := date_create(2025, 12, 31)
	testing.expect(t, d1_ok == .None && d2_ok == .None && d3_ok == .None, "create dates")
	col, err := column_from("dates", []Date{d1, d2, d3})
	testing.expect(t, err == .None, "create Date column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add dates")

	testing.expect(t, dataframe_write_arrow(&df, path) == .None, "write arrow")

	df2, r_err := dataframe_read_arrow(path)
	testing.expect(t, r_err == .None, "read arrow")
	defer dataframe_destroy(&df2)

	got, g_err := dataframe_get_column(&df2, "dates")
	testing.expect(t, g_err == .None, "get dates back")
	testing.expect(t, got.dtype == typeid_of(Date), "dtype Date")

	view := column_typed_view(got, Date)
	testing.expect(t, view[0] == d1, "dates[0]")
	testing.expect(t, view[1] == d2, "dates[1]")
	testing.expect(t, view[2] == d3, "dates[2]")
}

@(test)
arrow_roundtrip_datetime :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_datetime.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	dt1, dt1_ok := datetime_create(2024, 1, 15, 10, 30, 0, 0)
	dt2, dt2_ok := datetime_create(2024, 6, 30, 23, 59, 59, 999999)
	testing.expect(t, dt1_ok == .None && dt2_ok == .None, "create datetimes")
	col, err := column_from("timestamps", []Datetime{dt1, dt2})
	testing.expect(t, err == .None, "create Datetime column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add timestamps")

	testing.expect(t, dataframe_write_arrow(&df, path) == .None, "write arrow")

	df2, r_err := dataframe_read_arrow(path)
	testing.expect(t, r_err == .None, "read arrow")
	defer dataframe_destroy(&df2)

	got, g_err := dataframe_get_column(&df2, "timestamps")
	testing.expect(t, g_err == .None, "get timestamps back")
	testing.expect(t, got.dtype == typeid_of(Datetime), "dtype Datetime")

	view := column_typed_view(got, Datetime)
	testing.expect(t, view[0] == dt1, "timestamps[0]")
	testing.expect(t, view[1] == dt2, "timestamps[1]")
}

@(test)
arrow_roundtrip_duration :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_duration.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	dur1 := duration_from_seconds(100)
	dur2 := duration_from_minutes(5)
	col, err := column_from("durations", []Duration{dur1, dur2})
	testing.expect(t, err == .None, "create Duration column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add durations")

	testing.expect(t, dataframe_write_arrow(&df, path) == .None, "write arrow")

	df2, r_err := dataframe_read_arrow(path)
	testing.expect(t, r_err == .None, "read arrow")
	defer dataframe_destroy(&df2)

	got, g_err := dataframe_get_column(&df2, "durations")
	testing.expect(t, g_err == .None, "get durations back")
	testing.expect(t, got.dtype == typeid_of(Duration), "dtype Duration")

	view := column_typed_view(got, Duration)
	testing.expect(t, view[0] == dur1, "durations[0]")
	testing.expect(t, view[1] == dur2, "durations[1]")
}

@(test)
arrow_roundtrip_temporal_with_nulls :: proc(t: ^testing.T) {
	path := "/tmp/df_arrow_temporal_nulls.arrow"
	defer os.remove(path)

	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	d1, d1_ok := date_create(2024, 1, 1)
	d2, d2_ok := date_create(2024, 6, 15)
	testing.expect(t, d1_ok == .None && d2_ok == .None, "create dates")
	col, err := column_from_with_valid("dates", []Date{d1, d2}, []bool{true, false})
	testing.expect(t, err == .None, "create Date column with nulls")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add dates")

	testing.expect(t, dataframe_write_arrow(&df, path) == .None, "write arrow")

	df2, r_err := dataframe_read_arrow(path)
	testing.expect(t, r_err == .None, "read arrow")
	defer dataframe_destroy(&df2)

	got, _ := dataframe_get_column(&df2, "dates")
	testing.expect(t, column_is_valid(got, 0), "dates[0] valid")
	testing.expect(t, !column_is_valid(got, 1), "dates[1] null")

	view := column_typed_view(got, Date)
	testing.expect(t, view[0] == d1, "dates[0]")
}
