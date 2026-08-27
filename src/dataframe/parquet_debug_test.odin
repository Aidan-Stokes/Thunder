package dataframe

import "core:testing"
import "core:fmt"
import "core:mem"

@(test)
parquet_debug_roundtrip :: proc(t: ^testing.T) {
	df := dataframe_create(context.allocator)
	defer dataframe_destroy(&df)

	col, err := column_from("ids", []i64{10, 20, 30})
	testing.expect(t, err == .None, "create i64 column")
	testing.expect(t, dataframe_add_column(&df, &col) == .None, "add ids")

	data, w_err := dataframe_write_parquet(&df)
	testing.expect(t, w_err == .None, "write parquet")
	defer delete(data)
	testing.expect(t, len(data) > 16, "file has bytes")

	testing.expect(t, string(data[:4]) == parquet_MAGIC, "starts with PAR1")
	testing.expect(t, string(data[len(data)-4:]) == parquet_MAGIC, "ends with PAR1")

	footer_len_pos := len(data) - 8
	footer_len: u32
	mem.copy(rawptr(&footer_len), rawptr(&data[footer_len_pos]), 4)

	footer_start := len(data) - 8 - int(footer_len)
	footer_bytes := data[footer_start : footer_start + int(footer_len)]

	fmt.printf("footer bytes (%d): ", len(footer_bytes))
	for i in 0 ..< min(80, len(footer_bytes)) {
		fmt.printf("%02x ", footer_bytes[i])
	}
	fmt.printf("\n")

	fr: Thrift_Reader
	thrift_reader_init(&fr, footer_bytes)

	// Manual field-by-field read to see what fids appear
	for {
		fid, ftype, hok := thrift_read_field_header(&fr)
		if !hok { break }
		fmt.printf("field fid=%d ftype=%d pos=%d\n", fid, ftype, fr.pos)
		if ftype == 0 { break }
		// Skip this field's value
		skip_field(&fr, ftype)
	}

	fmt.printf("--- now re-parse ---\n")
	thrift_reader_init(&fr, footer_bytes)
	meta, mok := deserialize_file_meta(&fr, context.allocator)
	defer _destroy_file_meta(&meta, context.allocator)

	testing.expect(t, mok, fmt.tprintf("footer parse ok=%v", mok))
	testing.expect(t, len(meta.schema) >= 2, fmt.tprintf("schema count=%d", len(meta.schema)))
	testing.expect(t, len(meta.row_groups) >= 1, fmt.tprintf("rg count=%d", len(meta.row_groups)))

	if len(meta.row_groups) >= 1 {
		rg := meta.row_groups[0]
		testing.expect(t, rg.num_rows == 3, fmt.tprintf("rg.num_rows=%d", rg.num_rows))
		testing.expect(t, len(rg.columns) == 1, fmt.tprintf("rg.cols=%d", len(rg.columns)))
	}

	df2, ok := dataframe_read_parquet(data)
	testing.expect(t, ok, "read_parquet")
	if ok {
		defer dataframe_destroy(&df2)
		testing.expect(t, dataframe_num_rows(&df2) == 3, "rows")
		testing.expect(t, dataframe_num_cols(&df2) == 1, "cols")

		got, g_err := dataframe_get_column(&df2, "ids")
		testing.expect(t, g_err == .None, "get ids")
		view := column_typed_view(got, i64)
		testing.expect(t, view[0] == 10, "ids[0]")
		testing.expect(t, view[1] == 20, "ids[1]")
		testing.expect(t, view[2] == 30, "ids[2]")
	}
}
