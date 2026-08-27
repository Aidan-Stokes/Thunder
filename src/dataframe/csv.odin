package dataframe

// CSV I/O (Stage 9, DESIGN.md §11): dataframe_read_csv,
// dataframe_read_csv_with_schema, dataframe_write_csv.
//
// Built on the RFC 4180 reader/writer in core:encoding/csv and on
// core:strconv for typed field parsing (AGENTS.md principle 2): the record
// split, quoting, and escape handling are all stdlib; this file only glues
// records to typed columns.
//
// Type inference (S9.2) samples the first sample_rows records per column,
// then validates every record: a field that cannot be parsed as the inferred
// type is an error (.Type_Mismatch) — never a silent type change (principle 7).
// Inference picks the narrowest type that parses every sampled non-NULL
// value: bool > i64 > f64 > string. Columns with no sampled non-NULL value
// fall back to string.
//
// NULL (S9.4): a field equal to null_token is read as NULL and a NULL is
// written as null_token. The default token is the empty string, so unquoted
// and quoted empty fields round-trip as NULL. NULLs are skipped by type
// inference and never constrain the inferred dtype.
//
// Supported column types for CSV are bool, i64, f64, and string; anything
// else is .Unsupported_Operation (also enforced for read_csv_with_schema).

import "core:encoding/csv"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strconv"
import "core:strings"

// CSV_Options configures the CSV reader and writer. The zero value selects
// the defaults below.
CSV_Options :: struct {
	// delimiter is the field separator; 0 selects ','.
	delimiter: rune,
	// comment, when non-zero, is the line-comment character; lines starting
	// with it (no leading whitespace) are ignored.
	comment: rune,
	// null_token marks a field as NULL when reading and is written for NULL
	// rows. The default "" means empty fields are NULL.
	null_token: string,
	// sample_rows caps how many records are used for type inference
	// (default 1000) before the rest of the file is validated.
	sample_rows: int,
}

// dataframe_read_csv reads path and returns a DataFrame whose column types
// are inferred from the data (S9.1–S9.2). The first record must be the
// header. The result owns its columns (string contents are deep-copied).
dataframe_read_csv :: proc(path: string, options: CSV_Options = {}, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	data, o_err := os.read_entire_file(path, allocator)
	if o_err != os.ERROR_NONE {
		return {}, .CSV_Error
	}
	defer delete(data)

	kinds, names, k_err := csv_infer_types(string(data), allocator, options)
	if k_err != .None {
		return {}, k_err
	}
	defer delete(kinds)
	defer csv_delete_strings(names, allocator)

	if len(data) >= CSV_PARALLEL_THRESHOLD && options.comment == 0 {
		out, err = csv_build_columns_parallel(string(data), allocator, options, names, kinds, CSV_PARALLEL_DEFAULT_THREADS)
	} else {
		out, err = csv_build_columns(string(data), allocator, options, names, kinds)
	}
	if err != .None {
		return {}, err
	}
	return out, .None
}

// dataframe_read_csv_with_schema reads path using an explicit schema: the
// header must match the schema's field names in order (else .Invalid_Schema)
// and every field is parsed as the schema's dtype (S9.3). Supported schema
// dtypes are bool, i64, f64, and string.
dataframe_read_csv_with_schema :: proc(path: string, schema: Schema, options: CSV_Options = {}, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	if len(schema.fields) == 0 {
		return {}, .Invalid_Schema
	}
	data, o_err := os.read_entire_file(path, allocator)
	if o_err != os.ERROR_NONE {
		return {}, .CSV_Error
	}
	defer delete(data)

	r: csv.Reader
	csv_new_reader(&r, string(data), allocator, options)
	defer csv.reader_destroy(&r)

	header, r_err := csv.read(&r)
	if r_err != nil {
		return {}, .CSV_Error
	}
	if len(header) != len(schema.fields) {
		return {}, .Length_Mismatch
	}
	for i in 0 ..< len(schema.fields) {
		if schema.fields[i].name != header[i] {
			return {}, .Invalid_Schema
		}
	}

	kinds := make([]CSV_Kind, len(schema.fields), allocator)
	if kinds == nil && len(schema.fields) != 0 {
		return {}, .Allocator_Failure
	}
	defer delete(kinds)
	for i in 0 ..< len(schema.fields) {
		k, k_err := csv_kind_from_typeid(schema.fields[i].dtype)
		if k_err != .None {
			return {}, k_err
		}
		kinds[i] = k
	}

	names, n_err := csv_own_strings(allocator, header)
	if n_err != .None {
		return {}, n_err
	}
	defer csv_delete_strings(names, allocator)

	if len(data) >= CSV_PARALLEL_THRESHOLD && options.comment == 0 {
		out, err = csv_build_columns_parallel(string(data), allocator, options, names, kinds, CSV_PARALLEL_DEFAULT_THREADS)
	} else {
		out, err = csv_build_columns(string(data), allocator, options, names, kinds)
	}
	if err != .None {
		return {}, err
	}
	return out, .None
}

// dataframe_read_csv_with_columns reads path but materializes only the named
// columns, in the order given (S12.1 projection pushdown). Unknown names are
// an error (.Column_Not_Found); duplicates in `columns` are an error
// (.Duplicate_Column_Name). Only the requested fields are sampled, type
// checked, parsed, and buffered; the other fields are skipped entirely (their
// dtype never constrains anything). The result owns its columns.
dataframe_read_csv_with_columns :: proc(path: string, columns: []string, options: CSV_Options = {}, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	if len(columns) == 0 {
		return {}, .Invalid_Argument
	}
	data, o_err := os.read_entire_file(path, allocator)
	if o_err != os.ERROR_NONE {
		return {}, .CSV_Error
	}
	defer delete(data)

	field_idx := make([]int, len(columns), allocator)
	if field_idx == nil && len(columns) != 0 {
		return {}, .Allocator_Failure
	}
	defer delete(field_idx)

	r: csv.Reader
	csv_new_reader(&r, string(data), allocator, options)
	defer csv.reader_destroy(&r)
	header, h_err := csv.read(&r)
	if h_err != nil {
		return {}, .CSV_Error
	}
	if verr := csv_validate_header(header); verr != .None {
		return {}, verr
	}
	for i in 0 ..< len(columns) {
		c := columns[i]
		for k in 0 ..< i {
			if columns[k] == c {
				return {}, .Duplicate_Column_Name
			}
		}
		found := false
		for j in 0 ..< len(header) {
			if header[j] == c {
				field_idx[i] = j
				found = true
				break
			}
		}
		if !found {
			return {}, .Column_Not_Found
		}
	}

	kinds, names, k_err := csv_infer_types_keep(string(data), allocator, options, field_idx)
	if k_err != .None {
		return {}, k_err
	}
	defer delete(kinds)
	defer csv_delete_strings(names, allocator)

	if len(data) >= CSV_PARALLEL_THRESHOLD && options.comment == 0 {
		out, err = csv_build_columns_keep_parallel(string(data), allocator, options, names, kinds, field_idx, len(header), CSV_PARALLEL_DEFAULT_THREADS)
	} else {
		out, err = csv_build_columns_keep(string(data), allocator, options, names, kinds, field_idx, len(header))
	}
	if err != .None {
		return {}, err
	}
	return out, .None
}

// dataframe_write_csv writes df as a CSV file: a header row of column names
// followed by one record per row (S9.5). NULL rows are written as
// null_token; f64 uses shortest round-trip formatting. Supported column
// types are bool, i64, f64, and string.
dataframe_write_csv :: proc(df: ^DataFrame, path: string, options: CSV_Options = {}, allocator := context.allocator) -> Error {
	if df.columns.count == 0 {
		return .Invalid_Argument
	}

	sb := strings.builder_make(allocator)
	defer strings.builder_destroy(&sb)
	w: csv.Writer
	w.comma = csv_delimiter(options)
	csv.writer_init(&w, strings.to_writer(&sb))

	header := make([]string, df.columns.count, allocator)
	if header == nil && df.columns.count != 0 {
		return .Allocator_Failure
	}
	defer delete(header)
	for i in 0 ..< df.columns.count {
		header[i] = cs_name(&df.columns, i)
	}
	if io_err := csv.write(&w, header); io_err != .None {
		return .CSV_Error
	}

	row := make([]string, df.columns.count, allocator)
	if row == nil && df.columns.count != 0 {
		return .Allocator_Failure
	}
	defer delete(row)
	for r in 0 ..< dataframe_num_rows(df) {
		for i in 0 ..< df.columns.count {
			col := column_set_to_column(&df.columns, i)
			field, f_err := csv_format_field(&col, r, options.null_token)
			if f_err != .None {
				return f_err
			}
			row[i] = field
		}
		if io_err := csv.write(&w, row); io_err != .None {
			return .CSV_Error
		}
	}

	if o_err := os.write_entire_file_from_string(path, strings.to_string(sb)); o_err != os.ERROR_NONE {
		return .CSV_Error
	}
	return .None
}

// --- internal --------------------------------------------------------------

// CSV_Kind is the set of column types CSV can carry.
CSV_Kind :: enum {
	Bool,
	I64,
	F64,
	String,
}

@(private)
csv_delimiter :: proc(options: CSV_Options) -> rune {
	if options.delimiter == 0 {
		return ','
	}
	return options.delimiter
}

@(private)
csv_sample_rows :: proc(options: CSV_Options) -> int {
	if options.sample_rows == 0 {
		return 1000
	}
	return options.sample_rows
}

// csv_new_reader initializes r to read over the in-memory CSV text. Record
// fields alias an internal buffer (no per-field allocations); callers must
// copy anything they keep before the next read. r must be initialized in
// place (never returned by value): the stdlib reader wires bufio to an
// internal strings.Reader by pointer.
@(private)
csv_new_reader :: proc(r: ^csv.Reader, data: string, allocator: mem.Allocator, options: CSV_Options) {
	csv.reader_init_with_string(r, data, allocator)
	r.comma = csv_delimiter(options)
	r.comment = options.comment
	r.reuse_record = true
	r.reuse_record_buffer = true
}

// csv_read_header reads and validates the header record. The returned field
// strings alias the reader's internal buffer and are valid until the next
// read.
@(private)
csv_read_header :: proc(data: string, allocator: mem.Allocator, options: CSV_Options) -> (header: []string, err: Error) {
	r: csv.Reader
	csv_new_reader(&r, data, allocator, options)
	defer csv.reader_destroy(&r)
	rec, r_err := csv.read(&r)
	if r_err != nil {
		return nil, .CSV_Error
	}
	if err = csv_validate_header(rec); err != .None {
		return nil, err
	}
	return rec, .None
}

@(private)
csv_validate_header :: proc(header: []string) -> Error {
	for name, i in header {
		if name == "" {
			return .Column_Name_Empty
		}
		for j in 0 ..< i {
			if header[j] == name {
				return .Duplicate_Column_Name
			}
		}
	}
	return .None
}

// csv_infer_types reads the header, samples up to sample_rows records for
// type inference, and validates every record against the inferred kinds
// (S9.2). The returned names are owned copies of the header strings.
@(private)
csv_infer_types :: proc(data: string, allocator: mem.Allocator, options: CSV_Options) -> (kinds: []CSV_Kind, names: []string, err: Error) {
	r: csv.Reader
	csv_new_reader(&r, data, allocator, options)
	defer csv.reader_destroy(&r)

	header, r_err := csv.read(&r)
	if r_err != nil {
		return nil, nil, .CSV_Error
	}
	if err = csv_validate_header(header); err != .None {
		return nil, nil, err
	}
	names, err = csv_own_strings(allocator, header)
	if err != .None {
		return nil, nil, err
	}

	n := len(names)
	kinds = make([]CSV_Kind, n, allocator)
	if kinds == nil && n != 0 {
		csv_delete_strings(names, allocator)
		return nil, nil, .Allocator_Failure
	}
	still_bool := make([]bool, n, allocator)
	still_int := make([]bool, n, allocator)
	still_float := make([]bool, n, allocator)
	saw_value := make([]bool, n, allocator)
	if still_bool == nil || still_int == nil || still_float == nil || saw_value == nil {
		delete(kinds, allocator)
		delete(still_bool, allocator)
		delete(still_int, allocator)
		delete(still_float, allocator)
		delete(saw_value, allocator)
		csv_delete_strings(names, allocator)
		return nil, nil, .Allocator_Failure
	}
	defer delete(still_bool)
	defer delete(still_int)
	defer delete(still_float)
	defer delete(saw_value)
	for i in 0 ..< n {
		still_bool[i] = true
		still_int[i] = true
		still_float[i] = true
	}

	sample := csv_sample_rows(options)
	rows := 0
	for rows < sample {
		rec, rec_err := csv.read(&r)
		if csv.is_io_error(rec_err, .EOF) {
			break
		}
		if rec_err != nil {
			csv_delete_strings(names, allocator)
			delete(kinds, allocator)
			return nil, nil, .CSV_Error
		}
		for i in 0 ..< n {
			if rec[i] == options.null_token {
				continue
			}
			saw_value[i] = true
			switch csv_classify(rec[i]) {
			case .Bool:
				still_int[i] = false
				still_float[i] = false
			case .I64:
				still_bool[i] = false
			case .F64:
				still_bool[i] = false
				still_int[i] = false
			case .String:
				still_bool[i] = false
				still_int[i] = false
				still_float[i] = false
			}
		}
		rows += 1
	}

	for i in 0 ..< n {
		switch {
		case !saw_value[i]:
			kinds[i] = .String
		case still_bool[i]:
			kinds[i] = .Bool
		case still_int[i]:
			kinds[i] = .I64
		case still_float[i]:
			kinds[i] = .F64
		case:
			kinds[i] = .String
		}
	}

	// Validate every remaining record against the inferred kinds.
	for {
		rec, rec_err := csv.read(&r)
		if csv.is_io_error(rec_err, .EOF) {
			break
		}
		if rec_err != nil {
			csv_delete_strings(names, allocator)
			delete(kinds, allocator)
			return nil, nil, .CSV_Error
		}
		for i in 0 ..< n {
			if rec[i] == options.null_token {
				continue
			}
			if !csv_field_parses(kinds[i], rec[i]) {
				csv_delete_strings(names, allocator)
				delete(kinds, allocator)
				return nil, nil, .Type_Mismatch
			}
		}
	}
	return kinds, names, .None
}

// csv_infer_types_keep is csv_infer_types for a column subset: kinds and
// names are inferred only for the header fields listed in field_idx. Records
// are still width-validated, but fields that are not kept never constrain
// inference or get parsed — the S12.1 win of skipping dropped columns.
@(private)
csv_infer_types_keep :: proc(data: string, allocator: mem.Allocator, options: CSV_Options, field_idx: []int) -> (kinds: []CSV_Kind, names: []string, err: Error) {
	r: csv.Reader
	csv_new_reader(&r, data, allocator, options)
	defer csv.reader_destroy(&r)

	header, r_err := csv.read(&r)
	if r_err != nil {
		return nil, nil, .CSV_Error
	}
	if err = csv_validate_header(header); err != .None {
		return nil, nil, err
	}

	m := len(field_idx)
	names = make([]string, m, allocator)
	if names == nil && m != 0 {
		return nil, nil, .Allocator_Failure
	}
	kinds = make([]CSV_Kind, m, allocator)
	if kinds == nil && m != 0 {
		csv_delete_strings(names, allocator)
		return nil, nil, .Allocator_Failure
	}
	for i in 0 ..< m {
		owned, o_err := clone_name(allocator, header[field_idx[i]])
		if o_err != .None {
			csv_delete_strings(names, allocator)
			delete(kinds, allocator)
			return nil, nil, o_err
		}
		names[i] = owned
	}

	still_bool := make([]bool, m, allocator)
	still_int := make([]bool, m, allocator)
	still_float := make([]bool, m, allocator)
	saw_value := make([]bool, m, allocator)
	if still_bool == nil || still_int == nil || still_float == nil || saw_value == nil {
		delete(kinds, allocator)
		delete(still_bool, allocator)
		delete(still_int, allocator)
		delete(still_float, allocator)
		delete(saw_value, allocator)
		csv_delete_strings(names, allocator)
		return nil, nil, .Allocator_Failure
	}
	defer delete(still_bool)
	defer delete(still_int)
	defer delete(still_float)
	defer delete(saw_value)
	for i in 0 ..< m {
		still_bool[i] = true
		still_int[i] = true
		still_float[i] = true
	}

	sample := csv_sample_rows(options)
	rows := 0
	for rows < sample {
		rec, rec_err := csv.read(&r)
		if csv.is_io_error(rec_err, .EOF) {
			break
		}
		if rec_err != nil {
			csv_delete_strings(names, allocator)
			delete(kinds, allocator)
			return nil, nil, .CSV_Error
		}
		if len(rec) != len(header) {
			csv_delete_strings(names, allocator)
			delete(kinds, allocator)
			return nil, nil, .CSV_Error
		}
		for i in 0 ..< m {
			field := rec[field_idx[i]]
			if field == options.null_token {
				continue
			}
			saw_value[i] = true
			switch csv_classify(field) {
			case .Bool:
				still_int[i] = false
				still_float[i] = false
			case .I64:
				still_bool[i] = false
			case .F64:
				still_bool[i] = false
				still_int[i] = false
			case .String:
				still_bool[i] = false
				still_int[i] = false
				still_float[i] = false
			}
		}
		rows += 1
	}

	for i in 0 ..< m {
		switch {
		case !saw_value[i]:
			kinds[i] = .String
		case still_bool[i]:
			kinds[i] = .Bool
		case still_int[i]:
			kinds[i] = .I64
		case still_float[i]:
			kinds[i] = .F64
		case:
			kinds[i] = .String
		}
	}

	// Validate every remaining record against the inferred kinds of the kept
	// columns only.
	for {
		rec, rec_err := csv.read(&r)
		if csv.is_io_error(rec_err, .EOF) {
			break
		}
		if rec_err != nil {
			csv_delete_strings(names, allocator)
			delete(kinds, allocator)
			return nil, nil, .CSV_Error
		}
		for i in 0 ..< m {
			field := rec[field_idx[i]]
			if field == options.null_token {
				continue
			}
			if !csv_field_parses(kinds[i], field) {
				csv_delete_strings(names, allocator)
				delete(kinds, allocator)
				return nil, nil, .Type_Mismatch
			}
		}
	}
	return kinds, names, .None
}
@(private)
csv_field_parses :: proc(kind: CSV_Kind, s: string) -> bool {
	switch kind {
	case .Bool:
		return csv_ci_eq(s, "true") || csv_ci_eq(s, "false")
	case .I64:
		return csv_is_i64(s)
	case .F64:
		return csv_is_f64(s)
	case .String:
		return true
	}
	return false
}

// csv_build_columns reads the header (discarded), parses every record into
// per-column buffers, and materializes the DataFrame. names are owned and
// used as column names; kinds must match their count.
@(private)
csv_build_columns :: proc(data: string, allocator: mem.Allocator, options: CSV_Options, names: []string, kinds: []CSV_Kind) -> (out: DataFrame, err: Error) {
	r: csv.Reader
	csv_new_reader(&r, data, allocator, options)
	defer csv.reader_destroy(&r)

	header, h_err := csv.read(&r)
	if h_err != nil {
		return {}, .CSV_Error
	}
	if len(header) != len(kinds) {
		return {}, .Length_Mismatch
	}

	bufs := make([]CSV_Column_Buffer, len(kinds), allocator)
	if bufs == nil && len(kinds) != 0 {
		return {}, .Allocator_Failure
	}
	defer csv_column_buffers_destroy(bufs, allocator)
	for i in 0 ..< len(kinds) {
		bufs[i] = csv_column_buffer_new(kinds[i], allocator)
	}

	for {
		rec, rec_err := csv.read(&r)
		if csv.is_io_error(rec_err, .EOF) {
			break
		}
		if rec_err != nil {
			return {}, .CSV_Error
		}
		if len(rec) != len(kinds) {
			return {}, .CSV_Error
		}
		if err = csv_append_record(bufs, allocator, options, rec); err != .None {
			return {}, err
		}
	}

	out = dataframe_create(allocator)
	for i in 0 ..< len(kinds) {
		col, c_err := csv_column_from_buffer(allocator, names[i], &bufs[i])
		if c_err != .None {
			dataframe_destroy(&out)
			return {}, c_err
		}
		if a_err := dataframe_add_column(&out, &col); a_err != .None {
			column_destroy(&col)
			dataframe_destroy(&out)
			return {}, a_err
		}
	}
	return out, .None
}

// csv_build_columns_keep is csv_build_columns for a column subset: only the
// columns listed in names/kinds are materialized. field_idx[i] is the record
// field index of the i-th kept column; width is the total field count per
// record (records are still width-validated). S12.1 projection pushdown uses
// this to read only the columns a plan needs.
@(private)
csv_build_columns_keep :: proc(data: string, allocator: mem.Allocator, options: CSV_Options, names: []string, kinds: []CSV_Kind, field_idx: []int, width: int) -> (out: DataFrame, err: Error) {
	r: csv.Reader
	csv_new_reader(&r, data, allocator, options)
	defer csv.reader_destroy(&r)

	header, h_err := csv.read(&r)
	if h_err != nil {
		return {}, .CSV_Error
	}
	if len(header) != width {
		return {}, .Length_Mismatch
	}

	bufs := make([]CSV_Column_Buffer, len(kinds), allocator)
	if bufs == nil && len(kinds) != 0 {
		return {}, .Allocator_Failure
	}
	defer csv_column_buffers_destroy(bufs, allocator)
	for i in 0 ..< len(kinds) {
		bufs[i] = csv_column_buffer_new(kinds[i], allocator)
	}

	for {
		rec, rec_err := csv.read(&r)
		if csv.is_io_error(rec_err, .EOF) {
			break
		}
		if rec_err != nil {
			return {}, .CSV_Error
		}
		if len(rec) != width {
			return {}, .CSV_Error
		}
		if err = csv_append_record_keep(bufs, allocator, options, rec, field_idx); err != .None {
			return {}, err
		}
	}

	out = dataframe_create(allocator)
	for i in 0 ..< len(kinds) {
		col, c_err := csv_column_from_buffer(allocator, names[i], &bufs[i])
		if c_err != .None {
			dataframe_destroy(&out)
			return {}, c_err
		}
		if a_err := dataframe_add_column(&out, &col); a_err != .None {
			column_destroy(&col)
			dataframe_destroy(&out)
			return {}, a_err
		}
	}
	return out, .None
}
// to the blob start, so they stay valid while the blob grows; the string
// headers are materialized after the last append.
CSV_String_Seg :: struct {
	start: int,
	len:   int,
}

CSV_Column_Buffer :: struct {
	kind:     CSV_Kind,
	bools:    [dynamic]bool,
	i64s:     [dynamic]i64,
	f64s:     [dynamic]f64,
	segs:     [dynamic]CSV_String_Seg, // string column: bytes per row
	blob:     [dynamic]byte,           // string column: owned string bytes
	valid:    [dynamic]bool,
	has_null: bool,
}

@(private)
csv_column_buffer_new :: proc(kind: CSV_Kind, allocator: mem.Allocator) -> (b: CSV_Column_Buffer) {
	b = CSV_Column_Buffer{kind = kind}
	b.valid = make([dynamic]bool, allocator)
	switch kind {
	case .Bool:
		b.bools = make([dynamic]bool, allocator)
	case .I64:
		b.i64s = make([dynamic]i64, allocator)
	case .F64:
		b.f64s = make([dynamic]f64, allocator)
	case .String:
		b.segs = make([dynamic]CSV_String_Seg, allocator)
		b.blob = make([dynamic]byte, allocator)
	}
	return b
}

@(private)
csv_column_buffers_destroy :: proc(bufs: []CSV_Column_Buffer, allocator: mem.Allocator) {
	for i in 0 ..< len(bufs) {
		b := &bufs[i]
		if b.bools != nil {
			delete(b.bools)
		}
		if b.i64s != nil {
			delete(b.i64s)
		}
		if b.f64s != nil {
			delete(b.f64s)
		}
		if b.segs != nil {
			delete(b.segs)
		}
		// blob ownership transfers to the column payload on success; after
		// the transfer the blob is zeroed and this delete is a no-op.
		if b.blob != nil {
			delete(b.blob)
		}
		if b.valid != nil {
			delete(b.valid)
		}
	}
	delete(bufs, allocator)
}

// csv_append_record parses one data record into the column buffers. The
// record was already validated, so parse failures are programming errors
// unless a NULL token sneaks a field through.
@(private)
csv_append_record :: proc(bufs: []CSV_Column_Buffer, allocator: mem.Allocator, options: CSV_Options, rec: []string) -> Error {
	for i in 0 ..< len(bufs) {
		b := &bufs[i]
		if rec[i] == options.null_token {
			b.has_null = true
			append(&b.valid, false)
			switch b.kind {
			case .Bool:
				append(&b.bools, false)
			case .I64:
				append(&b.i64s, 0)
			case .F64:
				append(&b.f64s, 0)
			case .String:
				append(&b.segs, CSV_String_Seg{})
			}
			continue
		}
		append(&b.valid, true)
		switch b.kind {
		case .Bool:
			v, ok := csv_parse_bool(rec[i])
			if !ok {
				return .Type_Mismatch
			}
			append(&b.bools, v)
		case .I64:
			v, ok := csv_parse_i64(rec[i])
			if !ok {
				return .Type_Mismatch
			}
			append(&b.i64s, v)
		case .F64:
			v, ok := csv_parse_f64(rec[i])
			if !ok {
				return .Type_Mismatch
			}
			append(&b.f64s, v)
		case .String:
			start := len(b.blob)
			append(&b.blob, ..transmute([]byte)(rec[i]))
			append(&b.segs, CSV_String_Seg{start = start, len = len(rec[i])})
		}
	}
	return .None
}

// csv_append_record_keep is csv_append_record for a column subset: buffer i
// parses the record field field_idx[i]. Field validation (row width, ragged
// rows) is handled by the caller.
@(private)
csv_append_record_keep :: proc(bufs: []CSV_Column_Buffer, allocator: mem.Allocator, options: CSV_Options, rec: []string, field_idx: []int) -> Error {
	for i in 0 ..< len(bufs) {
		b := &bufs[i]
		field := rec[field_idx[i]]
		if field == options.null_token {
			b.has_null = true
			append(&b.valid, false)
			switch b.kind {
			case .Bool:
				append(&b.bools, false)
			case .I64:
				append(&b.i64s, 0)
			case .F64:
				append(&b.f64s, 0)
			case .String:
				append(&b.segs, CSV_String_Seg{})
			}
			continue
		}
		append(&b.valid, true)
		switch b.kind {
		case .Bool:
			v, ok := csv_parse_bool(field)
			if !ok {
				return .Type_Mismatch
			}
			append(&b.bools, v)
		case .I64:
			v, ok := csv_parse_i64(field)
			if !ok {
				return .Type_Mismatch
			}
			append(&b.i64s, v)
		case .F64:
			v, ok := csv_parse_f64(field)
			if !ok {
				return .Type_Mismatch
			}
			append(&b.f64s, v)
		case .String:
			start := len(b.blob)
			append(&b.blob, ..transmute([]byte)(field))
			append(&b.segs, CSV_String_Seg{start = start, len = len(field)})
		}
	}
	return .None
}

// csv_column_from_buffer turns a filled buffer into an owned Column. For
// string columns the byte blob is transferred into the column's payload, so
// the column owns its string contents.
@(private)
csv_column_from_buffer :: proc(allocator: mem.Allocator, name: string, buf: ^CSV_Column_Buffer) -> (col: Column, err: Error) {
	valid_arg: []bool
	if buf.has_null {
		valid_arg = buf.valid[:]
	}
	switch buf.kind {
	case .Bool:
		return column_from_with_valid(name, buf.bools[:], valid_arg, allocator)
	case .I64:
		return column_from_with_valid(name, buf.i64s[:], valid_arg, allocator)
	case .F64:
		return column_from_with_valid(name, buf.f64s[:], valid_arg, allocator)
	case .String:
		// Materialize headers into the now-stable blob (no more appends).
		strs := make([]string, len(buf.segs), allocator)
		if strs == nil && len(buf.segs) != 0 {
			return {}, .Allocator_Failure
		}
		blob_bytes := buf.blob[:]
		for s, i in buf.segs {
			strs[i] = string(blob_bytes[s.start:s.start + s.len])
		}
		col, err = column_from_with_valid(name, strs, valid_arg, allocator)
		delete(strs, allocator)
		if err != .None {
			return {}, err
		}
		if len(buf.blob) != 0 {
			col.payload = raw_data(buf.blob)
			col.payload_size = len(buf.blob)
			buf.blob = {} // column owns the bytes now
		}
		return col, .None
	}
	return {}, .Unsupported_Operation
}

// --- typed field helpers ---------------------------------------------------

@(private)
csv_classify :: proc(s: string) -> CSV_Kind {
	if csv_ci_eq(s, "true") || csv_ci_eq(s, "false") {
		return .Bool
	}
	if csv_is_i64(s) {
		return .I64
	}
	if csv_is_f64(s) {
		return .F64
	}
	return .String
}

@(private)
csv_is_i64 :: proc(s: string) -> bool {
	n := 0
	_, ok := strconv.parse_i64(s, &n)
	return ok
}

@(private)
csv_is_f64 :: proc(s: string) -> bool {
	n := 0
	_, ok := strconv.parse_f64(s, &n)
	return ok
}

@(private)
csv_parse_bool :: proc(s: string) -> (bool, bool) {
	if csv_ci_eq(s, "true") {
		return true, true
	}
	if csv_ci_eq(s, "false") {
		return false, true
	}
	return false, false
}

@(private)
csv_parse_i64 :: proc(s: string) -> (i64, bool) {
	n := 0
	return strconv.parse_i64(s, &n)
}

@(private)
csv_parse_f64 :: proc(s: string) -> (f64, bool) {
	n := 0
	return strconv.parse_f64(s, &n)
}

// csv_ci_eq compares two ASCII strings case-insensitively without allocating.
@(private)
csv_ci_eq :: proc(a, b: string) -> bool {
	if len(a) != len(b) {
		return false
	}
	for i in 0 ..< len(a) {
		ca, cb := a[i], b[i]
		if 'A' <= ca && ca <= 'Z' {
			ca += 32
		}
		if 'A' <= cb && cb <= 'Z' {
			cb += 32
		}
		if ca != cb {
			return false
		}
	}
	return true
}

@(private)
csv_own_strings :: proc(allocator: mem.Allocator, src: []string) -> ([]string, Error) {
	out := make([]string, len(src), allocator)
	if out == nil && len(src) != 0 {
		return nil, .Allocator_Failure
	}
	for s, i in src {
		owned, o_err := clone_name(allocator, s)
		if o_err != .None {
			csv_delete_strings(out, allocator)
			return nil, o_err
		}
		out[i] = owned
	}
	return out, .None
}

@(private)
csv_delete_strings :: proc(ss: []string, allocator: mem.Allocator) {
	for s in ss {
		if s != "" {
			delete_string(s, allocator)
		}
	}
	delete(ss, allocator)
}

@(private)
csv_kind_from_typeid :: proc(id: typeid) -> (CSV_Kind, Error) {
	switch id {
	case typeid_of(bool):
		return .Bool, .None
	case typeid_of(i64):
		return .I64, .None
	case typeid_of(f64):
		return .F64, .None
	case typeid_of(string):
		return .String, .None
	case:
		return {}, .Unsupported_Operation
	}
}

@(private)
csv_format_field :: proc(col: ^Column, row: int, null_token: string) -> (string, Error) {
	if !column_is_valid(col, row) {
		return null_token, .None
	}
	switch col.dtype {
	case typeid_of(bool):
		v, _, _ := column_get(col, row, bool)
		if v {
			return "true", .None
		}
		return "false", .None
	case typeid_of(i64):
		v, _, _ := column_get(col, row, i64)
		return fmt.tprintf("%d", v), .None
	case typeid_of(f64):
		v, _, _ := column_get(col, row, f64)
		return fmt.tprintf("%v", v), .None
	case typeid_of(string):
		v, _, _ := column_get(col, row, string)
		return v, .None
	case:
		return "", .Unsupported_Operation
	}
}
