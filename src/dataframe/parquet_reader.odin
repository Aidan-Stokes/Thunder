package dataframe

// Parquet reader: reads a Parquet file into a DataFrame.
// Supports plain-encoded columns with optional LZ4 compression.

import "core:mem"
import "core:c"
import lz4 "../../libs/odin-lz4/lz4"

Parquet_Read_Options :: struct {
	columns: []string, // empty = all columns
}

parquet_read_options_default :: proc() -> Parquet_Read_Options {
	return {}
}

dataframe_read_parquet :: proc(
	data: []u8,
	opts := Parquet_Read_Options{},
	allocator := context.allocator,
) -> (result: DataFrame, ok: bool) {

	result = dataframe_create(allocator)

	// ── 1. Validate magic ──────────────────────────────────────────────
	if len(data) < 12 { return }
	if string(data[:4]) != parquet_MAGIC || string(data[len(data)-4:]) != parquet_MAGIC {
		return
	}

	// ── 2. Read footer length and footer ──────────────────────────────
	footer_len_pos := len(data) - 8
	footer_len: u32
	mem.copy(rawptr(&footer_len), rawptr(&data[footer_len_pos]), 4)
	footer_start := len(data) - 8 - int(footer_len)
	if footer_start < 4 || footer_start + int(footer_len) > len(data) - 4 {
		return
	}
	footer_bytes := data[footer_start : footer_start + int(footer_len)]

	// ── 3. Parse footer (FileMetaData) ────────────────────────────────
	footer_reader: Thrift_Reader
	thrift_reader_init(&footer_reader, footer_bytes)

	meta, mok := deserialize_file_meta(&footer_reader, allocator)
	if !mok { return }
	defer _destroy_file_meta(&meta, allocator)

	if len(meta.schema) < 2 { return }
	if len(meta.row_groups) < 1 { return }

	rg := meta.row_groups[0]

	// ── 4. Filter columns if requested ────────────────────────────────
	num_schema_cols := len(meta.schema) - 1
	num_rg_cols := len(rg.columns)

	col_indices := make([dynamic]int, allocator)
	defer delete(col_indices)

	if len(opts.columns) > 0 {
		for req_name in opts.columns {
			for i in 1 ..< len(meta.schema) {
				if meta.schema[i].name == req_name {
					append(&col_indices, i - 1)
					break
				}
			}
		}
	} else {
		for i in 0 ..< num_rg_cols {
			append(&col_indices, i)
		}
	}

	num_cols := len(col_indices)
	if num_cols == 0 { return }

	num_rows := int(rg.num_rows)

	// ── 5. Read and decode each column ─────────────────────────────────
	result_columns := make([dynamic]Column, allocator)
	defer delete(result_columns)

	result_names := make([dynamic]string, allocator)
	defer {
		for n in result_names { delete(n, allocator) }
		delete(result_names)
	}

	for ci in col_indices {
		if ci >= num_rg_cols { continue }

		chunk := rg.columns[ci]
		col_meta := chunk.metadata

		page_offset := int(col_meta.data_page_offset)
		if page_offset < 4 || page_offset >= len(data) - 4 {
			continue
		}

		col_data_end := page_offset
		if ci + 1 < num_rg_cols {
			next_chunk := rg.columns[ci + 1]
			col_data_end = int(next_chunk.metadata.data_page_offset)
		} else {
			col_data_end = footer_start
		}

		if col_data_end <= page_offset { continue }

		// Skip page header via thrift
		page_reader: Thrift_Reader
		thrift_reader_init(&page_reader, data[page_offset:])
		_, _, _ = deserialize_page_header(&page_reader)
		page_data_start := page_offset + page_reader.pos

		if page_data_start >= col_data_end && col_meta.num_values > 0 { continue }
		raw_col_data := data[page_data_start:col_data_end]

		// Decompress if needed
		col_data: []u8
		lz4_scratch: []u8
		if col_meta.codec == .LZ4 && len(raw_col_data) >= 4 {
			uncomp_size: u32
			mem.copy(rawptr(&uncomp_size), rawptr(&raw_col_data[0]), 4)
			lz4_payload := raw_col_data[4:]
			dst := make([]u8, int(uncomp_size), allocator)
			if dst != nil && len(lz4_payload) > 0 {
				written := lz4.decompress_safe(
					src = &lz4_payload[0],
					dst = &dst[0],
					compressedSize = c.int(len(lz4_payload)),
					dstCapacity = c.int(int(uncomp_size)),
				)
				if written >= 0 {
					col_data = dst[:int(written)]
					lz4_scratch = dst
				}
			}
			// If decompression failed or produced no data, try raw
			if col_data == nil {
				col_data = raw_col_data
			}
		} else {
			col_data = raw_col_data
		}

		column_name := ""
		if ci + 1 < len(meta.schema) {
			column_name = meta.schema[ci + 1].name
		}

		col, col_ok := _decode_parquet_column(
			col_data, col_meta.type_, col_meta.num_values,
			column_name, allocator,
		)
		if lz4_scratch != nil { delete(lz4_scratch, allocator) }
		if !col_ok { continue }

		append(&result_columns, col)
		name_copy := make([]u8, len(column_name), allocator)
		if name_copy != nil { copy(name_copy, column_name) }
		append(&result_names, string(name_copy))
	}

	// ── 6. Build DataFrame ─────────────────────────────────────────────
	final_cols := len(result_columns)
	if final_cols == 0 { return }

	for i in 0 ..< final_cols {
		col := result_columns[i]
		err := dataframe_add_column(&result, &col)
		if err != .None {
			column_destroy(&col)
		}
	}

	return result, true
}

@(private)
_decode_parquet_column :: proc(
	data: []u8, phys_type: Parquet_Physical_Type, num_values: i64,
	name: string, allocator: mem.Allocator,
) -> (col: Column, ok: bool) {

	count := int(num_values)
	if count == 0 {
		empty: []i32
		col, err := column_from(name, empty, allocator)
		return col, err == .None
	}

	#partial switch phys_type {
	case .BOOLEAN:
		return _decode_parquet_bool(data, count, name, allocator)
	case .INT32:
		return _decode_parquet_i32(data, count, name, allocator)
	case .INT64:
		return _decode_parquet_i64(data, count, name, allocator)
	case .FLOAT:
		return _decode_parquet_f32(data, count, name, allocator)
	case .DOUBLE:
		return _decode_parquet_f64(data, count, name, allocator)
	case .BYTE_ARRAY:
		return _decode_parquet_string(data, count, name, allocator)
	}
	return {}, false
}

@(private)
_decode_parquet_bool :: proc(data: []u8, count: int, name: string, allocator: mem.Allocator) -> (Column, bool) {
	bools := make([]bool, count, allocator)
	if bools == nil && count > 0 { return {}, false }
	for i in 0 ..< count {
		byte_idx := i / 8
		bit_idx := i % 8
		if byte_idx >= len(data) { delete(bools, allocator); return {}, false }
		bools[i] = (data[byte_idx] >> u8(bit_idx)) & 1 == 1
	}
	col, err := column_from(name, bools, allocator)
	delete(bools, allocator)
	return col, err == .None
}

@(private)
_decode_parquet_i32 :: proc(data: []u8, count: int, name: string, allocator: mem.Allocator) -> (Column, bool) {
	if len(data) < count * 4 { return {}, false }
	ints := make([]i32, count, allocator)
	if ints == nil && count > 0 { return {}, false }
	for i in 0 ..< count {
		mem.copy(rawptr(&ints[i]), rawptr(&data[i * 4]), 4)
	}
	col, err := column_from(name, ints, allocator)
	delete(ints, allocator)
	return col, err == .None
}

@(private)
_decode_parquet_i64 :: proc(data: []u8, count: int, name: string, allocator: mem.Allocator) -> (Column, bool) {
	if len(data) < count * 8 { return {}, false }
	longs := make([]i64, count, allocator)
	if longs == nil && count > 0 { return {}, false }
	for i in 0 ..< count {
		mem.copy(rawptr(&longs[i]), rawptr(&data[i * 8]), 8)
	}
	col, err := column_from(name, longs, allocator)
	delete(longs, allocator)
	return col, err == .None
}

@(private)
_decode_parquet_f32 :: proc(data: []u8, count: int, name: string, allocator: mem.Allocator) -> (Column, bool) {
	if len(data) < count * 4 { return {}, false }
	floats := make([]f32, count, allocator)
	if floats == nil && count > 0 { return {}, false }
	for i in 0 ..< count {
		mem.copy(rawptr(&floats[i]), rawptr(&data[i * 4]), 4)
	}
	col, err := column_from(name, floats, allocator)
	delete(floats, allocator)
	return col, err == .None
}

@(private)
_decode_parquet_f64 :: proc(data: []u8, count: int, name: string, allocator: mem.Allocator) -> (Column, bool) {
	if len(data) < count * 8 { return {}, false }
	doubles := make([]f64, count, allocator)
	if doubles == nil && count > 0 { return {}, false }
	for i in 0 ..< count {
		mem.copy(rawptr(&doubles[i]), rawptr(&data[i * 8]), 8)
	}
	col, err := column_from(name, doubles, allocator)
	delete(doubles, allocator)
	return col, err == .None
}

@(private)
_decode_parquet_string :: proc(data: []u8, count: int, name: string, allocator: mem.Allocator) -> (col: Column, ok: bool) {
	// Validate and size the shared blob in one pass.
	pos := 0
	total := 0
	for i in 0 ..< count {
		if pos + 4 > len(data) { return {}, false }
		s_len: u32
		mem.copy(rawptr(&s_len), rawptr(&data[pos]), 4)
		pos += 4
		if pos + int(s_len) > len(data) { return {}, false }
		total += int(s_len)
		pos += int(s_len)
	}

	blob := make([]u8, total, allocator)
	headers := make([]string, count, allocator)
	if (blob == nil && total > 0) || (headers == nil && count > 0) {
		if blob != nil { delete(blob, allocator) }
		if headers != nil { delete(headers, allocator) }
		return {}, false
	}

	pos = 0
	cursor := 0
	for i in 0 ..< count {
		s_len: u32
		mem.copy(rawptr(&s_len), rawptr(&data[pos]), 4)
		pos += 4
		if s_len > 0 {
			copy(blob[cursor : cursor + int(s_len)], data[pos : pos + int(s_len)])
			headers[i] = string(blob[cursor : cursor + int(s_len)])
			cursor += int(s_len)
		} else {
			headers[i] = ""
		}
		pos += int(s_len)
	}

	out, err := column_from(name, headers, allocator)
	delete(headers, allocator) // column_from copied the headers
	if err != .None {
		delete(blob, allocator)
		return {}, false
	}
	col = out
	// Transfer blob ownership so column_destroy frees the string contents.
	col.payload = raw_data(blob)
	col.payload_size = total
	return col, true
}

@(private)
_destroy_file_meta :: proc(meta: ^FileMetaData, allocator: mem.Allocator) {
	for i in 0 ..< len(meta.schema) {
		delete_string(meta.schema[i].name, allocator)
	}
	delete(meta.schema)
	for rg in meta.row_groups {
		for cc in rg.columns {
			delete(cc.metadata.encodings)
			for p in cc.metadata.path_in_schema {
				delete_string(p, allocator)
			}
			delete(cc.metadata.path_in_schema)
		}
		delete(rg.columns)
	}
	delete(meta.row_groups)
	delete(meta.key_value_metadata)
	delete_string(meta.created_by, allocator)
}

// ── Footer deserialization ─────────────────────────────────────────────────

deserialize_file_meta :: proc(r: ^Thrift_Reader, allocator := context.allocator) -> (meta: FileMetaData, ok: bool) {
	thrift_read_struct_begin(r)
	defer thrift_read_struct_end(r)
	if r.has_error || r.pos >= len(r.data) { return }

	meta.version = 2
	meta.num_rows = 0

	for {
		fid, ftype, hok := thrift_read_field_header(r)
		if !hok { break }
		if ftype == 0 { break }

		switch fid {
		case 1:
			v, vok := thrift_read_i32(r)
			if vok { meta.version = v }
		case 2:
			meta.schema, ok = deserialize_schema_list(r, allocator)
			if !ok { return }
		case 3:
			v, vok := thrift_read_i64(r)
			if vok { meta.num_rows = v }
		case 4:
			meta.row_groups, ok = deserialize_row_groups(r, allocator)
			if !ok { return }
		case 5:
			v, vok := thrift_read_string(r, allocator)
			if vok { meta.created_by = v }
		case:
			skip_field(r, ftype)
		}
	}

	ok = true
	return
}

deserialize_schema_list :: proc(r: ^Thrift_Reader, allocator := context.allocator) -> (result: [dynamic]SchemaElement, ok: bool) {
	_, count, lok := thrift_read_list_begin(r)
	if !lok { return }
	result = make([dynamic]SchemaElement, allocator)
	for i in 0 ..< count {
		elem, eok := deserialize_schema_element(r, allocator)
		if !eok { return }
		append(&result, elem)
	}
	ok = true
	return
}

deserialize_schema_element :: proc(r: ^Thrift_Reader, allocator := context.allocator) -> (elem: SchemaElement, ok: bool) {
	thrift_read_struct_begin(r)
	defer thrift_read_struct_end(r)

	for {
		fid, ftype, hok := thrift_read_field_header(r)
		if !hok { break }
		if ftype == 0 { break }

		switch fid {
		case 1:
			v, vok := thrift_read_i32(r)
			if vok { elem.type_ = Parquet_Physical_Type(v) }
		case 2:
			v, vok := thrift_read_string(r, allocator)
			if vok { elem.name = v }
		case 6:
			v, vok := thrift_read_i32(r)
			if vok {
				elem.converted_type = Parquet_Converted_Type(v)
				elem.converted_type_is_set = true
			}
		case 7:
			v, vok := thrift_read_i32(r)
			if vok { elem.scale = v }
		case 8:
			v, vok := thrift_read_i32(r)
			if vok { elem.precision = v }
		case 9:
			v, vok := thrift_read_i32(r)
			if vok {
				elem.field_id = v
				elem.field_id_is_set = true
			}
		case 10:
			deserialize_logical_type(r, &elem)
		case:
			skip_field(r, ftype)
		}
	}

	ok = elem.name != ""
	return
}

deserialize_logical_type :: proc(r: ^Thrift_Reader, elem: ^SchemaElement) {
	thrift_read_struct_begin(r)
	defer thrift_read_struct_end(r)
	for {
		fid, ftype, hok := thrift_read_field_header(r)
		if !hok { break }
		if ftype == 0 { break }
		switch fid {
		case 1:
			elem.logical_type_string = true
			elem.logical_type_is_set = true
			skip_struct(r)
		case 6:
			elem.logical_type_date = true
			elem.logical_type_is_set = true
			skip_struct(r)
		case 7:
			elem.logical_type_time = true
			elem.logical_type_is_set = true
			skip_struct(r)
		case 8:
			elem.logical_type_timestamp = true
			elem.logical_type_is_set = true
			skip_struct(r)
		case:
			skip_field(r, ftype)
		}
	}
}

deserialize_row_groups :: proc(r: ^Thrift_Reader, allocator := context.allocator) -> (result: [dynamic]RowGroup, ok: bool) {
	_, count, lok := thrift_read_list_begin(r)
	if !lok { return }
	result = make([dynamic]RowGroup, allocator)
	for i in 0 ..< count {
		rg, rgok := deserialize_row_group(r, allocator)
		if !rgok { return }
		append(&result, rg)
	}
	ok = true
	return
}

deserialize_row_group :: proc(r: ^Thrift_Reader, allocator := context.allocator) -> (rg: RowGroup, ok: bool) {
	thrift_read_struct_begin(r)
	defer thrift_read_struct_end(r)
	for {
		fid, ftype, hok := thrift_read_field_header(r)
		if !hok { break }
		if ftype == 0 { break }
		switch fid {
		case 1:
			rg.columns, ok = deserialize_column_chunks(r, allocator)
			if !ok { return }
		case 2:
			v, vok := thrift_read_i64(r)
			if vok { rg.total_byte_size = v }
		case 3:
			v, vok := thrift_read_i64(r)
			if vok { rg.num_rows = v }
		case 5:
			v, vok := thrift_read_i64(r)
			if vok { rg.file_offset = v }
		case 6:
			v, vok := thrift_read_i64(r)
			if vok { rg.total_compressed_size = v }
		case:
			skip_field(r, ftype)
		}
	}
	ok = true
	return
}

deserialize_column_chunks :: proc(r: ^Thrift_Reader, allocator := context.allocator) -> (chunks: [dynamic]ColumnChunk, ok: bool) {
	_, count, lok := thrift_read_list_begin(r)
	if !lok { return }
	chunks = make([dynamic]ColumnChunk, allocator)
	for i in 0 ..< count {
		cc, cok := deserialize_column_chunk(r, allocator)
		if !cok { return }
		append(&chunks, cc)
	}
	ok = true
	return
}

deserialize_column_chunk :: proc(r: ^Thrift_Reader, allocator := context.allocator) -> (cc: ColumnChunk, ok: bool) {
	thrift_read_struct_begin(r)
	defer thrift_read_struct_end(r)
	for {
		fid, ftype, hok := thrift_read_field_header(r)
		if !hok { break }
		if ftype == 0 { break }
		switch fid {
		case 1:
			v, vok := thrift_read_i64(r)
			if vok { cc.file_offset = v }
		case 2:
			cc.metadata, ok = deserialize_column_meta_data(r, allocator)
			if !ok { return }
		case:
			skip_field(r, ftype)
		}
	}
	ok = true
	return
}

deserialize_column_meta_data :: proc(r: ^Thrift_Reader, allocator := context.allocator) -> (meta: ColumnMetaData, ok: bool) {
	thrift_read_struct_begin(r)
	defer thrift_read_struct_end(r)
	for {
		fid, ftype, hok := thrift_read_field_header(r)
		if !hok { break }
		if ftype == 0 { break }
		switch fid {
		case 1:
			v, vok := thrift_read_i32(r)
			if vok { meta.type_ = Parquet_Physical_Type(v) }
		case 2:
			meta.encodings, ok = deserialize_encoding_list(r, allocator)
			if !ok { return }
		case 3:
			meta.path_in_schema, ok = deserialize_string_list(r, allocator)
			if !ok { return }
		case 4:
			v, vok := thrift_read_i32(r)
			if vok { meta.codec = Parquet_Compression(v) }
		case 5:
			v, vok := thrift_read_i64(r)
			if vok { meta.num_values = v }
		case 6:
			v, vok := thrift_read_i64(r)
			if vok { meta.total_uncompressed_size = v }
		case 7:
			v, vok := thrift_read_i64(r)
			if vok { meta.total_compressed_size = v }
		case 8:
			skip_field(r, ftype)
		case 9:
			v, vok := thrift_read_i64(r)
			if vok { meta.data_page_offset = v }
		case 10:
			v, vok := thrift_read_i64(r)
			if vok { meta.index_page_offset = v }
		case 11:
			v, vok := thrift_read_i64(r)
			if vok { meta.dictionary_page_offset = v }
		case 12:
			v, vok := thrift_read_i64(r)
			if vok { meta.bloom_filter_offset = v }
		case:
			skip_field(r, ftype)
		}
	}
	ok = true
	return
}

deserialize_encoding_list :: proc(r: ^Thrift_Reader, allocator := context.allocator) -> (result: [dynamic]Parquet_Encoding, ok: bool) {
	_, count, lok := thrift_read_list_begin(r)
	if !lok { return }
	result = make([dynamic]Parquet_Encoding, allocator)
	for i in 0 ..< count {
		v, vok := thrift_read_i32(r)
		if !vok { return }
		append(&result, Parquet_Encoding(v))
	}
	ok = true
	return
}

deserialize_string_list :: proc(r: ^Thrift_Reader, allocator := context.allocator) -> (result: [dynamic]string, ok: bool) {
	_, count, lok := thrift_read_list_begin(r)
	if !lok { return }
	result = make([dynamic]string, allocator)
	for i in 0 ..< count {
		v, vok := thrift_read_string(r, allocator)
		if !vok { return }
		append(&result, v)
	}
	ok = true
	return
}

// deserialize_page_header reads a Parquet PageHeader.
deserialize_page_header :: proc(r: ^Thrift_Reader) -> (type_: Parquet_Page_Type, uncompressed_size: i32, compressed_size: i32) {
	thrift_read_struct_begin(r)
	defer thrift_read_struct_end(r)
	for {
		fid, ftype, hok := thrift_read_field_header(r)
		if !hok { break }
		if ftype == 0 { break }
		switch fid {
		case 1:
			v, vok := thrift_read_i32(r)
			if vok { type_ = Parquet_Page_Type(v) }
		case 2:
			v, vok := thrift_read_i32(r)
			if vok { uncompressed_size = v }
		case 3:
			v, vok := thrift_read_i32(r)
			if vok { compressed_size = v }
		case 5, 7, 8:
			skip_struct(r)
		case:
			skip_field(r, ftype)
		}
	}
	return
}

// skip_field skips over a Thrift field value based on its type.
skip_field :: proc(r: ^Thrift_Reader, ttype: u8) {
	switch ttype {
	case 0:
	case 1, 2:
	case 3:
		thrift_read_byte(r)
	case 4:
		if r.pos + 8 <= len(r.data) { r.pos += 8 }
	case 5:
		thrift_read_varint_i32(r)
	case 6:
		thrift_read_varint_i32(r)
	case 7:
		thrift_read_varint_i64(r)
	case 8:
		length, lok := thrift_read_varint_i32(r)
		if lok && int(length) >= 0 {
			r.pos += int(length)
		}
	case 9, 10:
		skip_list(r)
	case 11:
		skip_map(r)
	case 12:
		skip_struct(r)
	case:
		r.has_error = true
	}
}

skip_struct :: proc(r: ^Thrift_Reader) {
	thrift_read_struct_begin(r)
	defer thrift_read_struct_end(r)
	for {
		_, ftype, hok := thrift_read_field_header(r)
		if !hok || ftype == 0 { break }
		skip_field(r, ftype)
	}
}

skip_list :: proc(r: ^Thrift_Reader) {
	elem_type, count, lok := thrift_read_list_begin(r)
	if !lok { return }
	for i in 0 ..< count {
		skip_field(r, elem_type)
	}
}

skip_map :: proc(r: ^Thrift_Reader) {
	_, count, lok := thrift_read_list_begin(r)
	if !lok { return }
	for i in 0 ..< count {
		skip_struct(r)
	}
}
