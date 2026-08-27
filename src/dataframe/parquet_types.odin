package dataframe

// Parquet type definitions and file structure constants.

parquet_MAGIC :: "PAR1"

// Parquet_Col_Encoded holds encoded column data during Parquet write.
Parquet_Col_Encoded :: struct {
	encoded:    []u8,
	num_values: int,
	null_count: i64,
	phys_type:  Parquet_Physical_Type,
	encoding:   Parquet_Encoding,
	codec:      Parquet_Compression,
	path:       string,
}

// Physical types (stored in the file).
Parquet_Physical_Type :: enum u32 {
	BOOLEAN              = 0,
	INT32                = 1,
	INT64                = 2,
	INT96                = 3,
	FLOAT                = 4,
	DOUBLE               = 5,
	BYTE_ARRAY           = 6,
	FIXED_LEN_BYTE_ARRAY = 7,
}

// Encodings.
Parquet_Encoding :: enum u32 {
	PLAIN           = 0,
	PLAIN_DICTIONARY = 2,
	RLE             = 3,
	RLE_DICTIONARY  = 4,
	BYTE_STREAM_SPLIT = 5,
}

// Compression codecs (values per parquet.thrift CompressionCodec).
Parquet_Compression :: enum u32 {
	UNCOMPRESSED = 0,
	SNAPPY       = 1,
	GZIP         = 2,
	LZO          = 3,
	BROTLI       = 4,
	LZ4          = 5,
	ZSTD         = 6,
	LZ4_RAW      = 7,
}

// Page types.
Parquet_Page_Type :: enum u32 {
	DATA_PAGE       = 0,
	INDEX_PAGE      = 1,
	DICTIONARY_PAGE = 2,
	DATA_PAGE_V2    = 3,
}

// Converted types (logical annotations).
Parquet_Converted_Type :: enum u32 {
	UTF8             = 0,
	MAP              = 1,
	MENU             = 2,
	ENUM             = 3,
	DECIMAL          = 4,
	DATE             = 5,
	TIME_MILLIS      = 6,
	TIMESTAMP_MILLIS = 7,
	UNKNOWN          = 12,
}

// Thrift encoding for Parquet types.
THRIFT_ENCODING :: enum u32 {
	PLA  = 0,
	PLE  = 1,
	RLE  = 2,
	BIT_PACKED = 3,
}

// ── Parquet structures ─────────────────────────────────────────────────────

SchemaElement :: struct {
	type_:             Parquet_Physical_Type,
	type_length:       i32,
	name:              string,
	converted_type:    Parquet_Converted_Type,
	converted_type_is_set: bool,
	scale:             i32,
	precision:         i32,
	field_id:          i32,
	field_id_is_set:   bool,
	logical_type_is_set: bool,
	// Logical type info (simplified — no nested union needed for basics)
	logical_type_date:     bool,
	logical_type_time:     bool,
	logical_type_timestamp: bool,
	logical_type_string:   bool,
}

ColumnChunk :: struct {
	file_offset:          i64,
	metadata:             ColumnMetaData,
}

ColumnMetaData :: struct {
	type_:                Parquet_Physical_Type,
	encodings:            [dynamic]Parquet_Encoding,
	path_in_schema:       [dynamic]string,
	codec:                Parquet_Compression,
	num_values:           i64,
	total_uncompressed_size: i64,
	total_compressed_size:   i64,
	key_value_metadata:   [dynamic]KeyValue,
	data_page_offset:     i64,
	index_page_offset:    i64,
	dictionary_page_offset: i64,
	bloom_filter_offset:  i64,
}

KeyValue :: struct {
	key:   string,
	value: string,
}

RowGroup :: struct {
	columns:   [dynamic]ColumnChunk,
	total_byte_size: i64,
	num_rows:  i64,
	sorting_columns: [dynamic]SortingColumn,
	file_offset: i64,
	total_compressed_size: i64,
	ordinal:    i16,
}

SortingColumn :: struct {
	column_idx: i32,
	descending: bool,
	nulls_first: bool,
}

FileMetaData :: struct {
	version:          i32,
	schema:           [dynamic]SchemaElement,
	num_rows:         i64,
	row_groups:       [dynamic]RowGroup,
	key_value_metadata: [dynamic]KeyValue,
	created_by:       string,
}

// ── RLE bit-packed encoding for booleans (definition/ repetition levels) ──

@(private)
parquet_encode_rle_bitpacked :: proc(writer: ^Thrift_Writer, values: []bool) {
	n := len(values)
	pos := 0
	for pos < n {
		// Find run of same value
		val := values[pos]
		run_len := 1
		for pos + run_len < n && values[pos + run_len] == val && run_len < 128 {
			run_len += 1
		}
		if val {
			// RLE run: header = (run_len << 1) | 1
			thrift_ensure(writer, 1)
			writer.buf[writer.pos] = u8(run_len << 1) | 1
			writer.pos += 1
		} else {
			// Bit-packed run: header = (run_len << 1) | 0, then ceil(run_len/8) bytes
			header := u8(run_len << 1)
			thrift_ensure(writer, 1)
			writer.buf[writer.pos] = header
			writer.pos += 1
			bytes_needed := (run_len + 7) / 8
			thrift_ensure(writer, bytes_needed)
			for b in 0 ..< bytes_needed {
				writer.buf[writer.pos + b] = 0
			}
			writer.pos += bytes_needed
		}
		pos += run_len
	}
}

@(private)
parquet_encode_levels :: proc(writer: ^Thrift_Writer, values: []bool) {
	// For plain encoding, levels are RLE-bit-packed.
	// Simplified: write as RLE bit-packed
	parquet_encode_rle_bitpacked(writer, values)
}

@(private)
parquet_encode_levels_uniform :: proc(writer: ^Thrift_Writer, n: int, val: bool) {
	// All same value
	if val {
		thrift_ensure(writer, 1)
		writer.buf[writer.pos] = u8(n << 1) | 1
		writer.pos += 1
	} else {
		header := u8(n << 1)
		thrift_ensure(writer, 1)
		writer.buf[writer.pos] = header
		writer.pos += 1
		bytes_needed := (n + 7) / 8
		thrift_ensure(writer, bytes_needed)
		for b in 0 ..< bytes_needed {
			writer.buf[writer.pos + b] = 0
		}
		writer.pos += bytes_needed
	}
}

@(private)
parquet_max_definition_level :: proc(cs: ^ColumnSet, col_idx: int) -> int {
	if col_idx < 0 || col_idx >= cs.count { return 0 }
	if cs.valids[col_idx] != nil { return 1 }
	return 0
}

parquet_type_size :: proc(phys: Parquet_Physical_Type) -> int {
	#partial switch phys {
	case .BOOLEAN:  return 1 // bit-packed
	case .INT32:    return 4
	case .INT64:    return 8
	case .INT96:    return 12
	case .FLOAT:    return 4
	case .DOUBLE:   return 8
	}
	return 0
}
