package dataframe

// JSON and NDJSON I/O (Stage 13, DESIGN.md §11): dataframe_read_json,
// dataframe_read_json_with_schema, dataframe_read_ndjson, dataframe_write_json,
// dataframe_write_ndjson.
//
// Built on core:encoding/json for parsing (AGENTS.md principle 2): the token
// scan, string unescaping, and number grammar are all stdlib; this file only
// glues typed JSON values to columns and formats columns back to JSON.
//
// JSON is self-describing, so there is no inference pass: a column's dtype is
// fixed by its first non-NULL value and every later value must be exactly that
// type (.Type_Mismatch, never a silent widening — principle 7, same rule as
// CSV). Nested Array/Object values are unsupported until List/Struct dtypes
// land (S14.2) and raise .Unsupported_Operation. JSON null is always NULL and
// never constrains the dtype.
//
// core:encoding/json parses objects into an unordered map, so read column
// order is key-alphabetical (deterministic; the stdlib unparse sorts maps the
// same way). Writes keep the DataFrame's column order; reading that file back
// re-sorts alphabetically.
//
// Reads are strict JSON (json.Specification.JSON — no JSON5 comments, trailing
// commas, or unquoted keys) and parse every integral token as i64. f64 output
// always carries a decimal point or exponent so an integral float reads back
// as f64, never i64. NaN and ±Inf are written verbatim (like CSV) and those
// files fail the strict reader, so NaN/±Inf values are write-only.
//
// Known upstream quirk: core:encoding/json leaks a few bytes when it aborts
// mid-object (an already-cloned key that was never inserted into the object
// map). It only happens on invalid input, so every affected file fails with
// .JSON_Error anyway; see ROADMAP.md Stage 13.

import "core:encoding/json"
import "core:fmt"
import "core:io"
import "core:math"
import "core:mem"
import "core:os"
import "core:sort"
import "core:strings"

// JSON_Options configures the JSON writer. The zero value selects compact
// output. Reads take no options: JSON is self-describing (see DESIGN.md §11).
JSON_Options :: struct {
	// pretty indents the array and puts one object per line when writing.
	pretty: bool,
}

// dataframe_read_json reads path — a JSON array of objects, one per row — and
// returns a DataFrame with key-alphabetical columns. Every element must be an
// object and values must be homogeneous per column (see file comment). A key
// that appears only as null falls back to string (CSV's no-sample fallback);
// a key absent from every record is not a column.
dataframe_read_json :: proc(path: string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	records, r_err := json_read_records(path, allocator)
	if r_err != .None {
		return {}, r_err
	}
	defer json_destroy_records(records, allocator)
	return json_build_from_records(records, allocator, nil)
}

// dataframe_read_json_with_schema is dataframe_read_json restricted to the
// columns of schema, in schema order. A key outside the schema is
// .Invalid_Schema; a missing key reads as NULL; a schema column present in no
// record becomes an all-NULL column.
dataframe_read_json_with_schema :: proc(path: string, schema: Schema, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	records, r_err := json_read_records(path, allocator)
	if r_err != .None {
		return {}, r_err
	}
	defer json_destroy_records(records, allocator)
	s := schema
	return json_build_from_records(records, allocator, &s)
}

// dataframe_read_ndjson reads a newline-delimited stream of objects (one
// object per line, blank lines skipped) with the same column and dtype rules
// as dataframe_read_json.
dataframe_read_ndjson :: proc(path: string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	records, r_err := json_read_lines(path, allocator)
	if r_err != .None {
		return {}, r_err
	}
	defer json_destroy_records(records, allocator)
	return json_build_from_records(records, allocator, nil)
}

// dataframe_write_json writes df as a JSON array of objects, one per row, in
// column order. NULL is null; supported dtypes are bool, i64, f64, string.
dataframe_write_json :: proc(df: ^DataFrame, path: string, options: JSON_Options = {}, allocator := context.allocator) -> Error {
	if df.columns.count == 0 {
		return .Invalid_Argument
	}

	sb := strings.builder_make(allocator)
	defer strings.builder_destroy(&sb)
	w := strings.to_writer(&sb)

	rows := dataframe_num_rows(df)
	if options.pretty {
		json_write(w, "[\n") or_return
	} else {
		json_write(w, "[") or_return
	}
	for r in 0 ..< rows {
		if options.pretty {
			json_write(w, "  ") or_return
		}
		if err := json_write_object(w, df, r); err != .None {
			return err
		}
		if r + 1 < rows {
			json_write(w, ",") or_return
		}
		if options.pretty {
			json_write(w, "\n") or_return
		}
	}
	json_write(w, "]") or_return

	if o_err := os.write_entire_file_from_string(path, strings.to_string(sb)); o_err != os.ERROR_NONE {
		return .JSON_Error
	}
	return .None
}

// dataframe_write_ndjson writes one compact object per line.
dataframe_write_ndjson :: proc(df: ^DataFrame, path: string, allocator := context.allocator) -> Error {
	if df.columns.count == 0 {
		return .Invalid_Argument
	}

	sb := strings.builder_make(allocator)
	defer strings.builder_destroy(&sb)
	w := strings.to_writer(&sb)

	for r in 0 ..< dataframe_num_rows(df) {
		if err := json_write_object(w, df, r); err != .None {
			return err
		}
		json_write(w, "\n") or_return
	}

	if o_err := os.write_entire_file_from_string(path, strings.to_string(sb)); o_err != os.ERROR_NONE {
		return .JSON_Error
	}
	return .None
}

// --- internal: read ---------------------------------------------------------

// JSON_Kind is the set of column types JSON can carry. .Null is a transient
// marker for a parsed null value and is never a column dtype.
JSON_Kind :: enum {
	Null,
	Bool,
	I64,
	F64,
	String,
}

// json_read_records reads path as a JSON array of objects and returns one
// json.Value (an Object) per element. The returned slice and every object are
// owned by the caller (json_destroy_records).
@(private)
json_read_records :: proc(path: string, allocator: mem.Allocator) -> (records: []json.Value, err: Error) {
	data, o_err := os.read_entire_file(path, allocator)
	if o_err != os.ERROR_NONE {
		return nil, .JSON_Error
	}
	defer delete(data)

	root, p_err := json.parse_string(string(data), json.Specification.JSON, true, allocator)
	if p_err != .None {
		return nil, .JSON_Error
	}
	arr, is_arr := root.(json.Array)
	if !is_arr {
		json.destroy_value(root, allocator)
		return nil, .Type_Mismatch
	}

	// Validate before copying: root owns every element, so the destroy paths
	// below free everything exactly once.
	for v in arr {
		if _, ok := v.(json.Object); !ok {
			json.destroy_value(root, allocator)
			return nil, .Type_Mismatch
		}
	}

	records = make([]json.Value, len(arr), allocator)
	if records == nil && len(arr) != 0 {
		json.destroy_value(root, allocator)
		return nil, .Allocator_Failure
	}
	for v, i in arr {
		records[i] = v
	}
	delete(arr) // container only; elements transfer to records
	return records, .None
}

// json_read_lines reads an NDJSON stream: each non-blank line parses as one
// object. The returned slice and every object are owned by the caller
// (json_destroy_records).
@(private)
json_read_lines :: proc(path: string, allocator: mem.Allocator) -> (records: []json.Value, err: Error) {
	data, o_err := os.read_entire_file(path, allocator)
	if o_err != os.ERROR_NONE {
		return nil, .JSON_Error
	}
	defer delete(data)

	recs := make([dynamic]json.Value, allocator)
	lines, l_err := strings.split_lines(string(data), allocator)
	if l_err != nil {
		delete(recs)
		return nil, .Allocator_Failure
	}
	defer delete(lines)

	for line in lines {
		trimmed := strings.trim_space(line)
		if len(trimmed) == 0 {
			continue
		}
		v, p_err := json.parse_string(trimmed, json.Specification.JSON, true, allocator)
		if p_err != .None {
			json_destroy_value_slice(recs[:], allocator)
			delete(recs)
			return nil, .JSON_Error
		}
		if _, ok := v.(json.Object); !ok {
			json.destroy_value(v, allocator)
			json_destroy_value_slice(recs[:], allocator)
			delete(recs)
			return nil, .Type_Mismatch
		}
		if _, a_err := append(&recs, v); a_err != nil {
			json.destroy_value(v, allocator)
			json_destroy_value_slice(recs[:], allocator)
			delete(recs)
			return nil, .Allocator_Failure
		}
	}
	return recs[:], .None
}

@(private)
json_destroy_records :: proc(records: []json.Value, allocator: mem.Allocator) {
	json_destroy_value_slice(records, allocator)
	delete(records, allocator)
}

@(private)
json_destroy_value_slice :: proc(records: []json.Value, allocator: mem.Allocator) {
	for v in records {
		json.destroy_value(v, allocator)
	}
}

// json_build_from_records materializes a DataFrame from parsed objects.
// schema == nil infers the columns and dtypes from the data; otherwise the
// schema selects the columns and fixes the dtypes.
@(private)
json_build_from_records :: proc(records: []json.Value, allocator: mem.Allocator, schema: ^Schema) -> (out: DataFrame, err: Error) {
	names: []string
	kinds: []JSON_Kind
	defer if names != nil {
		csv_delete_strings(names, allocator)
	}
	defer delete(kinds)

	if schema != nil {
		for rec in records {
			obj := rec.(json.Object)
			for key in obj {
				if !schema_has_column(schema, key) {
					return {}, .Invalid_Schema
				}
			}
		}
		names = make([]string, len(schema.fields), allocator)
		kinds = make([]JSON_Kind, len(schema.fields), allocator)
		if names == nil && len(schema.fields) != 0 {
			return {}, .Allocator_Failure
		}
		if kinds == nil && len(schema.fields) != 0 {
			return {}, .Allocator_Failure
		}
		for i in 0 ..< len(schema.fields) {
			k, k_err := csv_kind_from_typeid(schema.fields[i].dtype)
			if k_err != .None {
				return {}, k_err
			}
			kinds[i] = json_kind_from_csv(k)
			owned, o_err := clone_name(allocator, schema.fields[i].name)
			if o_err != .None {
				return {}, o_err
			}
			names[i] = owned
		}
	} else {
		names, kinds, err = json_analyze_records(records, allocator)
		if err != .None {
			return {}, err
		}
	}

	bufs := make([]CSV_Column_Buffer, len(names), allocator)
	if bufs == nil && len(names) != 0 {
		return {}, .Allocator_Failure
	}
	defer csv_column_buffers_destroy(bufs, allocator)
	for i in 0 ..< len(names) {
		bufs[i] = csv_column_buffer_new(json_kind_to_csv(kinds[i]), allocator)
	}

	for rec in records {
		obj := rec.(json.Object)
		for i in 0 ..< len(names) {
			b := &bufs[i]
			if v, ok := obj[names[i]]; ok {
				if err := json_append_value(b, v); err != .None {
					return {}, err
				}
			} else {
				if err := json_append_null(b); err != .None {
					return {}, err
				}
			}
		}
	}

	out = dataframe_create(allocator)
	for i in 0 ..< len(names) {
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

// json_analyze_records walks every record once, fixing each column's dtype by
// its first non-NULL value and rejecting any later value of a different type.
// A key that appears only as null (or never as a non-NULL value) falls back to
// string — the same fallback CSV applies to columns with no sampled non-NULL
// value. The returned names are owned clones; kinds is a parallel owned slice.
@(private)
json_analyze_records :: proc(records: []json.Value, allocator: mem.Allocator) -> (names: []string, kinds: []JSON_Kind, err: Error) {
	seen := make(map[string]bool, allocator)
	defer delete(seen)
	key_kind := make(map[string]JSON_Kind, allocator)
	defer delete(key_kind)

	for rec in records {
		obj := rec.(json.Object)
		for key, val in obj {
			seen[key] = true
			k, k_err := json_value_kind(val)
			if k_err != .None {
				return nil, nil, k_err
			}
			if k == .Null {
				continue
			}
			if prev, found := key_kind[key]; found {
				if prev != k {
					return nil, nil, .Type_Mismatch
				}
			} else {
				key_kind[key] = k
			}
		}
	}
	for key in seen {
		if _, found := key_kind[key]; !found {
			key_kind[key] = .String
		}
	}

	keys := make([]string, len(key_kind), allocator)
	if keys == nil && len(key_kind) != 0 {
		return nil, nil, .Allocator_Failure
	}
	i := 0
	for k in key_kind {
		keys[i] = k
		i += 1
	}
	sort.quick_sort(keys)

	names = make([]string, len(keys), allocator)
	if names == nil && len(keys) != 0 {
		delete(keys, allocator)
		return nil, nil, .Allocator_Failure
	}
	kinds = make([]JSON_Kind, len(keys), allocator)
	if kinds == nil && len(keys) != 0 {
		delete(keys, allocator)
		csv_delete_strings(names, allocator)
		return nil, nil, .Allocator_Failure
	}
	for k, i in keys {
		owned, o_err := clone_name(allocator, k)
		if o_err != .None {
			delete(keys, allocator)
			csv_delete_strings(names, allocator)
			delete(kinds, allocator)
			return nil, nil, o_err
		}
		names[i] = owned
		kinds[i] = key_kind[k]
	}
	delete(keys, allocator)
	return names, kinds, .None
}

@(private)
json_value_kind :: proc(v: json.Value) -> (JSON_Kind, Error) {
	switch x in v {
	case nil, json.Null:
		return .Null, .None
	case json.Boolean:
		return .Bool, .None
	case json.Integer:
		return .I64, .None
	case json.Float:
		return .F64, .None
	case json.String:
		return .String, .None
	case json.Array, json.Object:
		return {}, .Unsupported_Operation
	}
	return {}, .Unsupported_Operation
}

// json_kind_from_csv maps a CSV_Kind to the matching JSON_Kind. .Null is
// never produced here: schema dtypes are physical types.
@(private)
json_kind_from_csv :: proc(k: CSV_Kind) -> JSON_Kind {
	switch k {
	case .Bool:
		return .Bool
	case .I64:
		return .I64
	case .F64:
		return .F64
	case .String:
		return .String
	}
	return .String
}

// json_kind_to_csv maps a JSON_Kind to the CSV_Column_Buffer kind it fills.
// .Null never reaches a buffer (column kinds are fixed during analysis).
@(private)
json_kind_to_csv :: proc(k: JSON_Kind) -> CSV_Kind {
	switch k {
	case .Null, .Bool:
		return .Bool
	case .I64:
		return .I64
	case .F64:
		return .F64
	case .String:
		return .String
	}
	return .String
}

@(private)
json_append_value :: proc(b: ^CSV_Column_Buffer, v: json.Value) -> Error {
	switch x in v {
	case nil, json.Null:
		return json_append_null(b)
	case json.Boolean:
		if b.kind != .Bool {
			return .Type_Mismatch
		}
		append(&b.valid, true)
		if _, err := append(&b.bools, bool(x)); err != nil {
			return .Allocator_Failure
		}
	case json.Integer:
		if b.kind != .I64 {
			return .Type_Mismatch
		}
		append(&b.valid, true)
		if _, err := append(&b.i64s, i64(x)); err != nil {
			return .Allocator_Failure
		}
	case json.Float:
		if b.kind != .F64 {
			return .Type_Mismatch
		}
		append(&b.valid, true)
		if _, err := append(&b.f64s, f64(x)); err != nil {
			return .Allocator_Failure
		}
	case json.String:
		if b.kind != .String {
			return .Type_Mismatch
		}
		append(&b.valid, true)
		start := len(b.blob)
		if _, err := append(&b.blob, ..transmute([]byte)(string(x))); err != nil {
			return .Allocator_Failure
		}
		if _, err := append(&b.segs, CSV_String_Seg{start = start, len = len(x)}); err != nil {
			return .Allocator_Failure
		}
	case json.Array, json.Object:
		return .Unsupported_Operation
	case:
		return .Type_Mismatch
	}
	return .None
}

@(private)
json_append_null :: proc(b: ^CSV_Column_Buffer) -> Error {
	b.has_null = true
	if _, err := append(&b.valid, false); err != nil {
		return .Allocator_Failure
	}
	switch b.kind {
	case .Bool:
		if _, err := append(&b.bools, false); err != nil {
			return .Allocator_Failure
		}
	case .I64:
		if _, err := append(&b.i64s, 0); err != nil {
			return .Allocator_Failure
		}
	case .F64:
		if _, err := append(&b.f64s, 0); err != nil {
			return .Allocator_Failure
		}
	case .String:
		if _, err := append(&b.segs, CSV_String_Seg{}); err != nil {
			return .Allocator_Failure
		}
	case:
		return .Type_Mismatch
	}
	return .None
}

// --- internal: write --------------------------------------------------------

@(private)
json_write :: proc(w: io.Writer, s: string) -> Error {
	if _, err := io.write_string(w, s); err != nil {
		return .JSON_Error
	}
	return .None
}

@(private)
json_write_string :: proc(w: io.Writer, s: string) -> Error {
	if _, err := io.write_quoted_string(w, s, '"', nil, true); err != nil {
		return .JSON_Error
	}
	return .None
}

// json_write_f64 writes v in shortest form (like CSV) with a decimal point or
// exponent forced so the value reads back as a float, never an integer. NaN
// and ±Inf are written verbatim (documented, write-only).
@(private)
json_write_f64 :: proc(w: io.Writer, v: f64) -> Error {
	s := fmt.tprintf("%v", v)
	need_decimal := !math.is_nan(v) && !math.is_inf(v)
	for c in s {
		if c == '.' || c == 'e' || c == 'E' {
			need_decimal = false
			break
		}
	}
	json_write(w, s) or_return
	if need_decimal {
		json_write(w, ".0") or_return
	}
	return .None
}

// json_write_object writes one row as {"name": value, ...} in column order.
@(private)
json_write_object :: proc(w: io.Writer, df: ^DataFrame, row: int) -> Error {
	json_write(w, "{") or_return
	for i in 0 ..< df.columns.count {
		col := column_set_to_column(&df.columns, i)
		if i > 0 {
			json_write(w, ", ") or_return
		}
		json_write_string(w, col.name) or_return
		json_write(w, ": ") or_return
		if !column_is_valid(&col, row) {
			json_write(w, "null") or_return
			continue
		}
		switch col.dtype {
		case typeid_of(bool):
			v, _, _ := column_get(&col, row, bool)
			if v {
				json_write(w, "true") or_return
			} else {
				json_write(w, "false") or_return
			}
		case typeid_of(i64):
			v, _, _ := column_get(&col, row, i64)
			if _, err := io.write_i64(w, v); err != nil {
				return .JSON_Error
			}
		case typeid_of(f64):
			v, _, _ := column_get(&col, row, f64)
			json_write_f64(w, v) or_return
		case typeid_of(string):
			v, _, _ := column_get(&col, row, string)
			json_write_string(w, v) or_return
		case:
			return .Unsupported_Operation
		}
	}
	json_write(w, "}") or_return
	return .None
}
