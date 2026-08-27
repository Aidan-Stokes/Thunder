package dataframe

// Parquet writer: writes a DataFrame to a byte slice in Parquet format.
// Supports plain encoding and optional LZ4 block compression.
//
// Column encoding is done by converting each column to an OdinArrow Array
// via the existing _arrow_array_from_column bridge, then encoding the
// Arrow buffer data as PLAIN Parquet pages.

import "core:mem"
import "core:c"
import "core:os"
import oa "../../libs/odinarrow"
import lz4 "../../libs/odin-lz4/lz4"

Parquet_Write_Options :: struct {
	compression: Parquet_Compression,
	row_group_size: int, // max rows per row group (default 65536)
}

parquet_write_options_default :: proc() -> Parquet_Write_Options {
	return Parquet_Write_Options{
		compression = .LZ4,
		row_group_size = 65536,
	}
}

// dataframe_write_parquet writes a DataFrame to a byte slice in Parquet format.
dataframe_write_parquet :: proc(
	df: ^DataFrame,
	opts := Parquet_Write_Options{},
	allocator := context.allocator,
) -> ([]u8, Error) {

	n := df.columns.count
	if n == 0 {
		return nil, .Invalid_Argument
	}

	num_rows := df.columns.rows[0]

	// ── 1. Build schema ─────────────────────────────────────────────────
	schema := make([dynamic]SchemaElement, allocator)
	defer delete(schema)

	// Root element (group)
	append(&schema, SchemaElement{
		type_ = .BOOLEAN,
		name  = "schema",
	})

	for col_idx in 0 ..< n {
		col := column_set_to_column(&df.columns, col_idx)
		arrow_dt, dt_ok := arrow_type_for_dtype(col.dtype)
		if !dt_ok {
			return nil, .Unsupported_Operation
		}
		elem: SchemaElement
		elem.name = col.name
		_populate_schema_element_from_arrow(&elem, arrow_dt)
		append(&schema, elem)
	}

	// ── 2. Convert columns to Arrow Arrays and encode ──────────────────

	encoded_cols := make([dynamic]Parquet_Col_Encoded, allocator)
	defer {
		for c in encoded_cols { delete(c.encoded, allocator) }
		delete(encoded_cols)
	}

	for col_idx in 0 ..< n {
		col := column_set_to_column(&df.columns, col_idx)

		arr, a_err := _arrow_array_from_column(&col, allocator)
		if a_err != .None {
			return nil, .Allocator_Failure
		}
		defer oa.array_free(&arr)

		phys_type, p_ok := _arrow_type_to_parquet_phys(arr.type)
		if !p_ok {
			return nil, .Unsupported_Operation
		}

		// Encode PLAIN from the Arrow buffers
		_tw: Thrift_Writer
		thrift_writer_init(&_tw, arr.length * 8, allocator)
		defer thrift_writer_destroy(&_tw)

		if !_encode_arrow_plain(&arr, &_tw) {
			return nil, .Unsupported_Operation
		}

		encoded := thrift_writer_bytes(&_tw)

		// Apply LZ4 compression
		final_data: []u8
		actual_codec := Parquet_Compression(.UNCOMPRESSED)
		if opts.compression == .LZ4 && len(encoded) > 0 {
			bound := lz4.compressBound(c.int(len(encoded)))
			compressed := make([]u8, int(bound) + 4, allocator)
			if compressed != nil {
				uncomp_size := u32(len(encoded))
				mem.copy(rawptr(&compressed[0]), rawptr(&uncomp_size), 4)
				written := lz4.compress_default(
					src = &encoded[0],
					dst = &compressed[4],
					srcSize = c.int(len(encoded)),
					dstCapacity = c.int(int(bound)),
				)
				if written > 0 {
					final_data = compressed[:4 + int(written)]
					actual_codec = .LZ4
				}
			}
			if final_data == nil {
				final_data = make([]u8, len(encoded), allocator)
				copy(final_data, encoded)
			}
		} else {
			final_data = make([]u8, len(encoded), allocator)
			copy(final_data, encoded)
		}

		nc: i64 = i64(arr.null_count)

		// Build a minimal DATA_PAGE header
		page_hdr_buf: Thrift_Writer
		thrift_writer_init(&page_hdr_buf, 64, allocator)
		defer thrift_writer_destroy(&page_hdr_buf)
		_serialize_page_header(&page_hdr_buf, i32(len(encoded)), i32(len(final_data)))
		page_hdr := thrift_writer_bytes(&page_hdr_buf)

		// Prepend page header to column data
		combined := make([]u8, len(page_hdr) + len(final_data), allocator)
		copy(combined, page_hdr)
		copy(combined[len(page_hdr):], final_data)
		if final_data != nil { delete(final_data, allocator) }

		append(&encoded_cols, Parquet_Col_Encoded{
			encoded    = combined,
			num_values = arr.length,
			null_count = nc,
			phys_type  = phys_type,
			encoding   = .PLAIN,
			codec      = actual_codec,
			path       = col.name,
		})
	}

	// ── 3. Build Parquet structures ─────────────────────────────────────
	total_byte_size: i64
	for c in encoded_cols { total_byte_size += i64(len(c.encoded)) }

	col_chunks := make([dynamic]ColumnChunk, allocator)
	defer {
		for cc in col_chunks {
			delete(cc.metadata.encodings)
			delete(cc.metadata.path_in_schema)
			delete(cc.metadata.key_value_metadata)
		}
		delete(col_chunks)
	}

	file_offset := i64(len(parquet_MAGIC))

	for i in 0 ..< n {
		enc := &encoded_cols[i]
		chunk: ColumnChunk
		chunk.file_offset = file_offset

		meta: ColumnMetaData
		meta.type_ = enc.phys_type
		meta.encodings = make([dynamic]Parquet_Encoding, allocator)
		append(&meta.encodings, enc.encoding)
		meta.path_in_schema = make([dynamic]string, allocator)
		append(&meta.path_in_schema, enc.path)
		meta.codec = enc.codec
		meta.num_values = i64(enc.num_values)
		meta.total_uncompressed_size = i64(enc.num_values) * i64(parquet_type_size(enc.phys_type))
		meta.total_compressed_size = i64(len(enc.encoded))
		meta.data_page_offset = file_offset

		chunk.metadata = meta
		append(&col_chunks, chunk)

		file_offset += i64(len(enc.encoded))
	}

	rg: RowGroup
	rg.columns = col_chunks
	rg.total_byte_size = total_byte_size
	rg.num_rows = i64(num_rows)
	rg.file_offset = i64(len(parquet_MAGIC))
	rg.total_compressed_size = total_byte_size

	file_meta: FileMetaData
	file_meta.version = 2
	file_meta.schema = schema
	file_meta.num_rows = i64(num_rows)
	file_meta.row_groups = make([dynamic]RowGroup, allocator)
	append(&file_meta.row_groups, rg)
	file_meta.created_by = "Thunder DataFrame"
	defer delete(file_meta.row_groups) // chunk arrays freed by col_chunks' defer

	// ── 4. Serialize footer ─────────────────────────────────────────────
	footer_buf: Thrift_Writer
	thrift_writer_init(&footer_buf, 4096, allocator)
	defer thrift_writer_destroy(&footer_buf)

	serialize_file_meta(&footer_buf, &file_meta)
	footer_bytes := thrift_writer_bytes(&footer_buf)

	// ── 5. Assemble file ────────────────────────────────────────────────
	file_size := len(parquet_MAGIC)*2 + int(total_byte_size) + len(footer_bytes) + 4
	file := make([]u8, file_size, allocator)
	pos := 0

	copy(file[pos:], parquet_MAGIC)
	pos += len(parquet_MAGIC)

	for c in encoded_cols {
		copy(file[pos:], c.encoded)
		pos += len(c.encoded)
	}

	copy(file[pos:], footer_bytes)
	pos += len(footer_bytes)

	footer_len := u32(len(footer_bytes))
	mem.copy(rawptr(&file[pos]), rawptr(&footer_len), 4)
	pos += 4

	copy(file[pos:], parquet_MAGIC)

	return file, .None
}

// dataframe_write_parquet_file is a convenience wrapper that writes to a file.
dataframe_write_parquet_file :: proc(
	df: ^DataFrame,
	path: string,
	opts := Parquet_Write_Options{},
	allocator := context.allocator,
) -> Error {
	data, err := dataframe_write_parquet(df, opts, allocator)
	if err != .None { return err }
	defer delete(data, allocator)

	flags := os.O_CREATE | os.O_WRONLY | os.O_TRUNC
	f, ferr := os.open(path, flags, os.Permissions_Default_File)
	if ferr != nil { return .Parquet_Error }
	defer os.close(f)

	os.write(f, data)
	return .None
}

// ── Helpers ────────────────────────────────────────────────────────────────

@(private)
_populate_schema_element_from_arrow :: proc(elem: ^SchemaElement, dt: oa.DataType) {
	arrow_type_name := oa.type_name(dt)
	switch arrow_type_name {
	case "bool":
		elem.type_ = .BOOLEAN
	case "int8", "int16", "int32", "uint8", "uint16", "uint32":
		elem.type_ = .INT32
	case "int64", "uint64":
		elem.type_ = .INT64
	case "float32":
		elem.type_ = .FLOAT
	case "float64":
		elem.type_ = .DOUBLE
	case "string", "large_string":
		elem.type_ = .BYTE_ARRAY
		elem.converted_type = .UTF8
		elem.converted_type_is_set = true
		elem.logical_type_string = true
		elem.logical_type_is_set = true
	case "date64":
		elem.type_ = .INT32
		elem.converted_type = .DATE
		elem.converted_type_is_set = true
		elem.logical_type_date = true
		elem.logical_type_is_set = true
	case "time64":
		elem.type_ = .INT64
		elem.converted_type = .TIME_MILLIS
		elem.converted_type_is_set = true
		elem.logical_type_time = true
		elem.logical_type_is_set = true
	case "timestamp":
		elem.type_ = .INT64
		elem.converted_type = .TIMESTAMP_MILLIS
		elem.converted_type_is_set = true
		elem.logical_type_timestamp = true
		elem.logical_type_is_set = true
	case "duration":
		elem.type_ = .INT64
	case:
		elem.type_ = .BYTE_ARRAY
	}
}

@(private)
_arrow_type_to_parquet_phys :: proc(dt: oa.DataType) -> (Parquet_Physical_Type, bool) {
	arrow_type_name := oa.type_name(dt)
	pt: Parquet_Physical_Type
	ok := true
	switch arrow_type_name {
	case "bool":              pt = .BOOLEAN
	case "int8", "int16", "int32", "uint8", "uint16", "uint32": pt = .INT32
	case "int64", "uint64", "time64", "timestamp", "duration": pt = .INT64
	case "float32":           pt = .FLOAT
	case "float64":           pt = .DOUBLE
	case "string", "large_string", "binary", "large_binary": pt = .BYTE_ARRAY
	case "date64":            pt = .INT32
	case:                  pt = .BYTE_ARRAY; ok = false
	}
	return pt, ok
}

@(private)
_encode_arrow_plain :: proc(arr: ^oa.Array, w: ^Thrift_Writer) -> bool {
	if arr.length == 0 { return true }

	arrow_type_name := oa.type_name(arr.type)
	result := _do_encode_arrow_plain(arr, w, arrow_type_name)
	return result
}

@(private)
_do_encode_arrow_plain :: proc(arr: ^oa.Array, w: ^Thrift_Writer, type_name: string) -> bool {
	result := true
	switch type_name {
		case "bool":
		data := arr.buffers[1].data
		if data == nil { result = false } else {
			bm := cast([^]u8)data
			off := arr.offset
			cur_byte := u8(0)
			bit_pos := 0
			for i in 0 ..< arr.length {
				byte_idx := (off + i) / 8
				bit_idx := (off + i) % 8
				val := (bm[byte_idx] >> u8(bit_idx)) & 1 == 1
				if val { cur_byte |= 1 << u8(bit_pos) }
				bit_pos += 1
				if bit_pos >= 8 {
					thrift_ensure(w, 1)
					w.buf[w.pos] = cur_byte
					w.pos += 1
					cur_byte = 0
					bit_pos = 0
				}
			}
			if bit_pos > 0 {
				thrift_ensure(w, 1)
				w.buf[w.pos] = cur_byte
				w.pos += 1
			}
		}
	case "int32", "uint32":
		data := arr.buffers[1].data
		if data == nil { result = false } else {
			ints := cast([^]i32)data
			off := arr.offset
			thrift_ensure(w, arr.length * 4)
			for i in 0 ..< arr.length {
				v := ints[off + i]
				mem.copy(rawptr(&w.buf[w.pos]), rawptr(&v), 4)
				w.pos += 4
			}
		}
	case "int64", "uint64", "time64", "timestamp", "duration":
		data := arr.buffers[1].data
		if data == nil { result = false } else {
			longs := cast([^]i64)data
			off := arr.offset
			thrift_ensure(w, arr.length * 8)
			for i in 0 ..< arr.length {
				v := longs[off + i]
				mem.copy(rawptr(&w.buf[w.pos]), rawptr(&v), 8)
				w.pos += 8
			}
		}
	case "float32":
		data := arr.buffers[1].data
		if data == nil { result = false } else {
			floats := cast([^]f32)data
			off := arr.offset
			thrift_ensure(w, arr.length * 4)
			for i in 0 ..< arr.length {
				v := floats[off + i]
				mem.copy(rawptr(&w.buf[w.pos]), rawptr(&v), 4)
				w.pos += 4
			}
		}
	case "float64":
		data := arr.buffers[1].data
		if data == nil { result = false } else {
			floats := cast([^]f64)data
			off := arr.offset
			thrift_ensure(w, arr.length * 8)
			for i in 0 ..< arr.length {
				v := floats[off + i]
				mem.copy(rawptr(&w.buf[w.pos]), rawptr(&v), 8)
				w.pos += 8
			}
		}
	case "string", "large_string":
		offsets_raw := arr.buffers[1].data
		data := arr.buffers[2].data
		if data == nil || offsets_raw == nil { result = false } else {
			off := arr.offset
			offsets := cast([^]i32)offsets_raw
			for i in 0 ..< arr.length {
				s_start := int(offsets[off + i])
				s_end := int(offsets[off + i + 1])
				s_len := s_end - s_start
				thrift_ensure(w, 4 + s_len)
				v := u32(s_len)
				mem.copy(rawptr(&w.buf[w.pos]), rawptr(&v), 4)
				w.pos += 4
				if s_len > 0 {
					mem.copy(rawptr(&w.buf[w.pos]), rawptr(&data[s_start]), s_len)
					w.pos += s_len
				}
			}
		}
	case "date64":
		data := arr.buffers[1].data
		if data == nil { result = false } else {
			millis := cast([^]i64)data
			off := arr.offset
			thrift_ensure(w, arr.length * 4)
			for i in 0 ..< arr.length {
				days := i32((millis[off + i]) / 86400000)
				mem.copy(rawptr(&w.buf[w.pos]), rawptr(&days), 4)
				w.pos += 4
			}
		}
	case:
		result = false
	}
	return result
}

// ── Page header ──────────────────────────────────────────────────────────

_serialize_page_header :: proc(w: ^Thrift_Writer, uncompressed_size: i32, compressed_size: i32) {
	thrift_write_struct_begin(w)
	thrift_write_i32_field(w, 0, 1) // type = DATA_PAGE
	thrift_write_i32_field(w, uncompressed_size, 2)
	thrift_write_i32_field(w, compressed_size, 3)
	thrift_write_struct_end(w)
}

// ── Footer serialization ──────────────────────────────────────────────────

serialize_file_meta :: proc(w: ^Thrift_Writer, meta: ^FileMetaData) {
	thrift_write_struct_begin(w)
	// field 1: version (i32)
	thrift_write_i32_field(w, meta.version, 1)
	// field 2: schema (list<SchemaElement>)
	_serialize_schema_list(w, &meta.schema)
	// field 3: num_rows (i64)
	thrift_write_i64_field(w, meta.num_rows, 3)
	// field 4: row_groups (list<RowGroup>)
	_serialize_row_groups(w, &meta.row_groups)
	// field 5: created_by (string)
	thrift_write_string_field(w, meta.created_by, 5)
	thrift_write_struct_end(w)
}

_serialize_schema_list :: proc(w: ^Thrift_Writer, schema: ^[dynamic]SchemaElement) {
	thrift_write_field_header(w, 9, 2) // type 9 = list
	thrift_write_list_begin(w, 12, len(schema^)) // elem type 12 = struct
	for elem in schema^ {
		_serialize_schema_element(w, elem)
	}
}

_serialize_schema_element :: proc(w: ^Thrift_Writer, elem: SchemaElement) {
	thrift_write_struct_begin(w)
	thrift_write_i32_field(w, i32(elem.type_), 1)
	thrift_write_string_field(w, elem.name, 2)
	if elem.converted_type_is_set {
		thrift_write_i32_field(w, i32(elem.converted_type), 6)
	}
	if elem.scale != 0 {
		thrift_write_i32_field(w, elem.scale, 7)
	}
	if elem.precision != 0 {
		thrift_write_i32_field(w, elem.precision, 8)
	}
	if elem.field_id_is_set {
		thrift_write_i32_field(w, elem.field_id, 9)
	}
	if elem.logical_type_is_set {
		_serialize_logical_type(w, elem)
	}
	thrift_write_struct_end(w)
}

_serialize_logical_type :: proc(w: ^Thrift_Writer, elem: SchemaElement) {
	thrift_write_field_header(w, 12, 10) // type 12 = struct
	thrift_write_struct_begin(w)
	if elem.logical_type_string {
		thrift_write_field_header(w, 12, 1) // String logical type struct
		thrift_write_struct_begin(w)
		thrift_write_struct_end(w)
	}
	if elem.logical_type_date {
		thrift_write_field_header(w, 12, 6) // Date logical type struct
		thrift_write_struct_begin(w)
		thrift_write_struct_end(w)
	}
	if elem.logical_type_time {
		thrift_write_field_header(w, 12, 7) // Time logical type struct
		thrift_write_struct_begin(w)
		thrift_write_struct_end(w)
	}
	if elem.logical_type_timestamp {
		thrift_write_field_header(w, 12, 8) // Timestamp logical type struct
		thrift_write_struct_begin(w)
		thrift_write_struct_end(w)
	}
	thrift_write_struct_end(w)
}

_serialize_row_groups :: proc(w: ^Thrift_Writer, rgs: ^[dynamic]RowGroup) {
	thrift_write_field_header(w, 9, 4) // type 9 = list
	thrift_write_list_begin(w, 12, len(rgs^)) // elem type 12 = struct
	for i in 0 ..< len(rgs^) {
		_serialize_row_group(w, &rgs[i])
	}
}

_serialize_row_group :: proc(w: ^Thrift_Writer, rg: ^RowGroup) {
	thrift_write_struct_begin(w)
	// field 1: columns
	_serialize_column_chunks(w, &rg.columns)
	// field 2: total_byte_size
	thrift_write_i64_field(w, rg.total_byte_size, 2)
	// field 3: num_rows
	thrift_write_i64_field(w, rg.num_rows, 3)
	// field 5: file_offset
	thrift_write_i64_field(w, rg.file_offset, 5)
	// field 6: total_compressed_size
	thrift_write_i64_field(w, rg.total_compressed_size, 6)
	thrift_write_struct_end(w)
}

_serialize_column_chunks :: proc(w: ^Thrift_Writer, chunks: ^[dynamic]ColumnChunk) {
	thrift_write_field_header(w, 9, 1) // type 9 = list
	thrift_write_list_begin(w, 12, len(chunks^))
	for i in 0 ..< len(chunks^) {
		_serialize_column_chunk(w, &chunks[i])
	}
}

_serialize_column_chunk :: proc(w: ^Thrift_Writer, cc: ^ColumnChunk) {
	thrift_write_struct_begin(w)
	thrift_write_i64_field(w, cc.file_offset, 1)
	_serialize_column_meta_data(w, &cc.metadata)
	thrift_write_struct_end(w)
}

_serialize_column_meta_data :: proc(w: ^Thrift_Writer, meta: ^ColumnMetaData) {
	thrift_write_field_header(w, 12, 2) // ColumnChunk.meta_data is field 2
	thrift_write_struct_begin(w)
	thrift_write_i32_field(w, i32(meta.type_), 1)
	// field 2: encodings (list<i32>)
	thrift_write_field_header(w, 9, 2)
	thrift_write_list_begin(w, 6, len(meta.encodings)) // elem type 6 = i32
	for enc in meta.encodings {
		thrift_write_varint_i32(w, i32(enc))
	}
	// field 3: path_in_schema (list<string>)
	thrift_write_field_header(w, 9, 3)
	thrift_write_list_begin(w, 8, len(meta.path_in_schema)) // elem type 8 = string
	for s in meta.path_in_schema {
		// Write string as length-prefixed bytes (no field header inside a list)
		thrift_write_varint_i32(w, i32(len(s)))
		if len(s) > 0 {
			thrift_ensure(w, len(s))
			copy(w.buf[w.pos:], s)
			w.pos += len(s)
		}
	}
	// field 4: codec
	thrift_write_i32_field(w, i32(meta.codec), 4)
	// field 5: num_values
	thrift_write_i64_field(w, meta.num_values, 5)
	// field 6: total_uncompressed_size
	thrift_write_i64_field(w, meta.total_uncompressed_size, 6)
	// field 7: total_compressed_size
	thrift_write_i64_field(w, meta.total_compressed_size, 7)
	// field 9: data_page_offset
	thrift_write_i64_field(w, meta.data_page_offset, 9)
	// field 10: index_page_offset
	thrift_write_i64_field(w, meta.index_page_offset, 10)
	// field 11: dictionary_page_offset
	thrift_write_i64_field(w, meta.dictionary_page_offset, 11)
	// field 12: bloom_filter_offset
	thrift_write_i64_field(w, meta.bloom_filter_offset, 12)
	thrift_write_struct_end(w)
}
