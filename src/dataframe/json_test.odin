package dataframe

// Stage 13 JSON/NDJSON I/O tests (ROADMAP S13.4): round-trips, strict typing,
// NULL/missing keys, all-null string fallback, schemas, alphabetical column
// order, string escaping, NDJSON, and error cases. Tests write scratch files
// under the OS temp dir.

import "core:fmt"
import "core:math"
import "core:os"
import "core:strings"
import "core:testing"

// json_path returns a unique scratch path for a test.
json_path :: proc(name: string) -> string {
	return strings.concatenate([]string{"/tmp/thunder_json_", name, ".json"})
}

// json_write_text writes raw JSON text to a scratch file.
json_write_text :: proc(t: ^testing.T, name: string, text: string) -> string {
	path := json_path(name)
	if err := os.write_entire_file_from_string(path, text); err != os.ERROR_NONE {
		testing.expect(t, false, fmt.tprintf("write %s: %v", name, err))
	}
	return path
}

// json_cleanup removes a scratch file and frees the path string returned by
// json_path.
json_cleanup :: proc(t: ^testing.T, path: string) {
	if err := os.remove(path); err != os.ERROR_NONE {
		testing.expect(t, false, fmt.tprintf("remove %s: %v", path, err))
	}
	delete_string(path, context.allocator)
}

// json_df builds the canonical 3-row frame used by the round-trip tests.
// Columns are in key-alphabetical order so a write-then-read keeps the order.
json_df :: proc(t: ^testing.T) -> DataFrame {
	b, _ := column_from("b", []bool{true, false, true})
	testing.expect(t, column_set_valid(&b, 1, false) == .None, "b[1] NULL")
	f, _ := column_from("f", []f64{0.1, 1.0 / 3.0, 1e-300})
	i, _ := column_from("i", []i64{1, -2, 3})
	testing.expect(t, column_set_valid(&i, 2, false) == .None, "i[2] NULL")
	s, _ := column_from("s", []string{`say "hi"`, "world,wide", ""})
	df, err := dataframe_from_columns([]^Column{&b, &f, &i, &s})
	testing.expect(t, err == .None, "from_columns")
	return df
}

// --- round-trip -------------------------------------------------------------

@(test)
json_test_roundtrip :: proc(t: ^testing.T) {
	df := json_df(t)
	defer dataframe_destroy(&df)

	path := json_path("roundtrip")
	defer json_cleanup(t, path)
	testing.expect(t, dataframe_write_json(&df, path) == .None, "write")

	back, r_err := dataframe_read_json(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&back)

	testing.expect(t, df_equal(&df, &back), "round-trip frame")
	csv_expect_dtypes(t, &back, []typeid{typeid_of(bool), typeid_of(f64), typeid_of(i64), typeid_of(string)})
}

@(test)
json_test_roundtrip_pretty :: proc(t: ^testing.T) {
	df := json_df(t)
	defer dataframe_destroy(&df)

	path := json_path("pretty")
	defer json_cleanup(t, path)
	testing.expect(t, dataframe_write_json(&df, path, JSON_Options{pretty = true}) == .None, "write pretty")

	back, r_err := dataframe_read_json(path)
	testing.expect(t, r_err == .None, "read pretty")
	defer dataframe_destroy(&back)
	testing.expect(t, df_equal(&df, &back), "pretty round-trip")

	text, _ := os.read_entire_file(path, context.allocator)
	defer delete(text)
	testing.expect(t, strings.contains(string(text), "\n"), "pretty output has newlines")
}

@(test)
json_test_roundtrip_f64_specials :: proc(t: ^testing.T) {
	f, _ := column_from("x", []f64{0.0, 3.0, 1e16, 0.5})
	df, _ := dataframe_from_columns([]^Column{&f})
	defer dataframe_destroy(&df)

	path := json_path("f64")
	defer json_cleanup(t, path)
	testing.expect(t, dataframe_write_json(&df, path) == .None, "write")

	// Integral floats must read back as f64, never i64.
	back, r_err := dataframe_read_json(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&back)
	xx := dataframe_get_column(&back, "x") or_else nil
	testing.expect(t, xx.dtype == typeid_of(f64), "integral floats stay f64")
	v0, _, _ := column_get(xx, 0, f64)
	v1, _, _ := column_get(xx, 1, f64)
	v2, _, _ := column_get(xx, 2, f64)
	v3, _, _ := column_get(xx, 3, f64)
	testing.expect(t, v0 == 0.0 && v1 == 3.0 && v2 == 1e16 && near(v3, 0.5), "f64 values")
}

@(test)
json_test_write_f64_nan_inf :: proc(t: ^testing.T) {
	f, _ := column_from("x", []f64{math.nan_f64(), math.inf_f64(1), math.inf_f64(-1)})
	df, _ := dataframe_from_columns([]^Column{&f})
	defer dataframe_destroy(&df)

	path := json_path("f64_specials")
	defer json_cleanup(t, path)
	testing.expect(t, dataframe_write_json(&df, path) == .None, "write")

	text, _ := os.read_entire_file(path, context.allocator)
	defer delete(text)
	testing.expect(t, strings.contains(string(text), "NaN"), "NaN written verbatim")

	// NaN/Inf are write-only: the strict reader rejects the literal.
	_, r_err := dataframe_read_json(path)
	testing.expect(t, r_err == .JSON_Error, "NaN literal rejected on read")
}

@(test)
json_test_read_alphabetical_columns :: proc(t: ^testing.T) {
	path := json_write_text(t, "alphabet",
		`[{"z": 1, "a": "x", "m": 2.5}, {"z": 2, "a": "y", "m": 3.5}]`)
	defer json_cleanup(t, path)

	df, r_err := dataframe_read_json(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df)

	testing.expect(t, dataframe_num_cols(&df) == 3, "3 cols")
	a := dataframe_column_at(&df, 0) or_else nil
	m := dataframe_column_at(&df, 1) or_else nil
	z := dataframe_column_at(&df, 2) or_else nil
	testing.expect(t, a.name == "a" && m.name == "m" && z.name == "z", "key-alphabetical order")
	testing.expect(t, a.dtype == typeid_of(string) && m.dtype == typeid_of(f64) && z.dtype == typeid_of(i64), "dtypes")
	va, _, _ := column_get(a, 1, string)
	testing.expect(t, va == "y", "a[1]")
}

@(test)
json_test_string_escaping :: proc(t: ^testing.T) {
	s, _ := column_from("s", []string{`tab"quo\slash`, "line\nbreak", "ünïcødé"})
	df, _ := dataframe_from_columns([]^Column{&s})
	defer dataframe_destroy(&df)

	path := json_path("escaping")
	defer json_cleanup(t, path)
	testing.expect(t, dataframe_write_json(&df, path) == .None, "write")

	back, r_err := dataframe_read_json(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&back)
	testing.expect(t, df_equal(&df, &back), "escaping round-trip")
}

@(test)
json_test_read_escaped_input :: proc(t: ^testing.T) {
	path := json_write_text(t, "escaped", `[{"s": "a\u0041\u00e9"}]`)
	defer json_cleanup(t, path)

	df, r_err := dataframe_read_json(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df)
	s := dataframe_get_column(&df, "s") or_else nil
	v, _, _ := column_get(s, 0, string)
	testing.expect(t, v == "aAé", `\u escapes decoded`)
}

// --- typing and NULL --------------------------------------------------------

@(test)
json_test_type_strictness :: proc(t: ^testing.T) {
	// Integer and Float in one column: .Type_Mismatch, never widening.
	path := json_write_text(t, "mixed_num", `[{"x": 1}, {"x": 2.5}]`)
	defer json_cleanup(t, path)
	_, r_err := dataframe_read_json(path)
	testing.expect(t, r_err == .Type_Mismatch, "int+float mismatch")

	// Two different scalar types across rows.
	path2 := json_write_text(t, "mixed_type", `[{"x": 1}, {"x": "one"}]`)
	defer json_cleanup(t, path2)
	_, r2_err := dataframe_read_json(path2)
	testing.expect(t, r2_err == .Type_Mismatch, "int+string mismatch")
}

@(test)
json_test_nested_unsupported :: proc(t: ^testing.T) {
	path := json_write_text(t, "nested_obj", `[{"a": 1, "o": {"k": 1}}, {"a": 2, "o": {"k": 2}}]`)
	defer json_cleanup(t, path)
	_, r_err := dataframe_read_json(path)
	testing.expect(t, r_err == .Unsupported_Operation, "nested object unsupported")

	path2 := json_write_text(t, "nested_arr", `[{"a": 1, "o": [1, 2]}]`)
	defer json_cleanup(t, path2)
	_, r2_err := dataframe_read_json(path2)
	testing.expect(t, r2_err == .Unsupported_Operation, "nested array unsupported")
}

@(test)
json_test_null_and_missing :: proc(t: ^testing.T) {
	path := json_write_text(t, "null_missing",
		`[{"a": 1, "b": null}, {"a": null, "b": "x"}, {"a": 2, "b": "y"}]`)
	defer json_cleanup(t, path)

	df, r_err := dataframe_read_json(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df)

	a := dataframe_get_column(&df, "a") or_else nil
	b := dataframe_get_column(&df, "b") or_else nil
	testing.expect(t, a.dtype == typeid_of(i64) && b.dtype == typeid_of(string), "dtypes")
	testing.expect(t, column_is_valid(a, 0) && !column_is_valid(a, 1) && column_is_valid(a, 2), "a validity")
	testing.expect(t, !column_is_valid(b, 0) && column_is_valid(b, 1) && column_is_valid(b, 2), "b validity")
	v, _, _ := column_get(b, 1, string)
	testing.expect(t, v == "x", "b[1]")
}

@(test)
json_test_all_null_falls_back_to_string :: proc(t: ^testing.T) {
	path := json_write_text(t, "all_null", `[{"a": 1, "b": null}, {"a": 2, "b": null}]`)
	defer json_cleanup(t, path)

	df, r_err := dataframe_read_json(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df)

	b := dataframe_get_column(&df, "b") or_else nil
	testing.expect(t, b != nil, "null-only key kept")
	testing.expect(t, b.dtype == typeid_of(string), "null-only key falls back to string")
	testing.expect(t, !column_is_valid(b, 0) && !column_is_valid(b, 1), "all NULL")
}

// --- explicit schema --------------------------------------------------------

@(test)
json_test_read_with_schema :: proc(t: ^testing.T) {
	path := json_write_text(t, "schema", `[{"a": 1, "b": "x"}, {"a": 2}]`)
	defer json_cleanup(t, path)

	schema, s_err := schema_create([]Field{
		{name = "b", dtype = typeid_of(string)},
		{name = "a", dtype = typeid_of(i64)},
	})
	testing.expect(t, s_err == .None, "schema_create")
	defer schema_destroy(&schema)

	df, r_err := dataframe_read_json_with_schema(path, schema)
	testing.expect(t, r_err == .None, "read with schema")
	defer dataframe_destroy(&df)

	testing.expect(t, dataframe_num_cols(&df) == 2, "2 cols")
	b := dataframe_column_at(&df, 0) or_else nil
	a := dataframe_column_at(&df, 1) or_else nil
	testing.expect(t, b.name == "b" && a.name == "a", "schema order")
	testing.expect(t, b.dtype == typeid_of(string) && a.dtype == typeid_of(i64), "schema dtypes")
	v, _, _ := column_get(b, 0, string)
	testing.expect(t, v == "x", "b[0]")
	testing.expect(t, !column_is_valid(b, 1), "missing key is NULL")
}

@(test)
json_test_read_with_schema_all_null_column :: proc(t: ^testing.T) {
	path := json_write_text(t, "schema_all_null", `[{"a": 1}, {"a": 2}]`)
	defer json_cleanup(t, path)

	schema, _ := schema_create([]Field{
		{name = "a", dtype = typeid_of(i64)},
		{name = "z", dtype = typeid_of(string)},
	})
	defer schema_destroy(&schema)

	df, r_err := dataframe_read_json_with_schema(path, schema)
	testing.expect(t, r_err == .None, "read with schema")
	defer dataframe_destroy(&df)
	z := dataframe_get_column(&df, "z") or_else nil
	testing.expect(t, z != nil, "absent schema column kept")
	testing.expect(t, z.dtype == typeid_of(string), "all-null schema col dtype")
	testing.expect(t, !column_is_valid(z, 0) && !column_is_valid(z, 1), "all-null schema col")
}

@(test)
json_test_read_with_schema_extra_key :: proc(t: ^testing.T) {
	path := json_write_text(t, "schema_extra", `[{"a": 1, "c": 2}]`)
	defer json_cleanup(t, path)

	schema, _ := schema_create([]Field{{name = "a", dtype = typeid_of(i64)}})
	defer schema_destroy(&schema)

	_, r_err := dataframe_read_json_with_schema(path, schema)
	testing.expect(t, r_err == .Invalid_Schema, "extra key rejected")
}

@(test)
json_test_read_with_schema_unsupported_type :: proc(t: ^testing.T) {
	path := json_write_text(t, "schema_unsupported", `[{"a": 1}]`)
	defer json_cleanup(t, path)

	schema, _ := schema_create([]Field{{name = "a", dtype = typeid_of(i32)}})
	defer schema_destroy(&schema)

	_, r_err := dataframe_read_json_with_schema(path, schema)
	testing.expect(t, r_err == .Unsupported_Operation, "unsupported dtype")
}

@(test)
json_test_read_with_schema_value_mismatch :: proc(t: ^testing.T) {
	path := json_write_text(t, "schema_mismatch", `[{"a": 1, "b": 2}]`)
	defer json_cleanup(t, path)

	schema, _ := schema_create([]Field{
		{name = "a", dtype = typeid_of(i64)},
		{name = "b", dtype = typeid_of(string)},
	})
	defer schema_destroy(&schema)

	_, r_err := dataframe_read_json_with_schema(path, schema)
	testing.expect(t, r_err == .Type_Mismatch, "value/dtype mismatch")
}

// --- NDJSON ----------------------------------------------------------------

@(test)
json_test_ndjson_roundtrip :: proc(t: ^testing.T) {
	df := json_df(t)
	defer dataframe_destroy(&df)

	path := json_path("ndjson")
	defer json_cleanup(t, path)
	testing.expect(t, dataframe_write_ndjson(&df, path) == .None, "write ndjson")

	back, r_err := dataframe_read_ndjson(path)
	testing.expect(t, r_err == .None, "read ndjson")
	defer dataframe_destroy(&back)
	testing.expect(t, df_equal(&df, &back), "ndjson round-trip")
}

@(test)
json_test_ndjson_blank_lines :: proc(t: ^testing.T) {
	path := json_write_text(t, "ndjson_blank", `{"a": 1, "b": "x"}` + "\n\n" + `{"a": 2}` + "\n")
	defer json_cleanup(t, path)

	df, r_err := dataframe_read_ndjson(path)
	testing.expect(t, r_err == .None, "read ndjson")
	defer dataframe_destroy(&df)
	testing.expect(t, dataframe_num_rows(&df) == 2, "2 rows")
	a := dataframe_get_column(&df, "a") or_else nil
	b := dataframe_get_column(&df, "b") or_else nil
	testing.expect(t, a.dtype == typeid_of(i64) && b.dtype == typeid_of(string), "dtypes")
	testing.expect(t, !column_is_valid(b, 1), "missing key NULL")
}

@(test)
json_test_ndjson_errors :: proc(t: ^testing.T) {
	path := json_write_text(t, "ndjson_bad", `{"a": 1}` + "\n" + `[1, 2]` + "\n")
	defer json_cleanup(t, path)
	_, r_err := dataframe_read_ndjson(path)
	testing.expect(t, r_err == .Type_Mismatch, "non-object line")

	path2 := json_write_text(t, "ndjson_malformed", `{"a": 1}` + "\n" + `{oops}` + "\n")
	defer json_cleanup(t, path2)
	_, r2_err := dataframe_read_ndjson(path2)
	testing.expect(t, r2_err == .JSON_Error, "malformed line")
}

// --- error cases ------------------------------------------------------------

@(test)
json_test_read_errors :: proc(t: ^testing.T) {
	path := json_path("missing")
	defer delete_string(path, context.allocator)
	_, r_err := dataframe_read_json(path)
	testing.expect(t, r_err == .JSON_Error, "missing file")

	path2 := json_write_text(t, "malformed", `[{"a": 1}oops]`)
	defer json_cleanup(t, path2)
	_, r2_err := dataframe_read_json(path2)
	testing.expect(t, r2_err == .JSON_Error, "malformed JSON")

	path3 := json_write_text(t, "not_array", `{"a": 1}`)
	defer json_cleanup(t, path3)
	_, r3_err := dataframe_read_json(path3)
	testing.expect(t, r3_err == .Type_Mismatch, "top level object")

	path4 := json_write_text(t, "not_object", `[1, 2]`)
	defer json_cleanup(t, path4)
	_, r4_err := dataframe_read_json(path4)
	testing.expect(t, r4_err == .Type_Mismatch, "element not object")

	path5 := json_write_text(t, "empty_array", `[]`)
	defer json_cleanup(t, path5)
	df, r5_err := dataframe_read_json(path5)
	testing.expect(t, r5_err == .None, "empty array reads")
	defer dataframe_destroy(&df)
	testing.expect(t, dataframe_num_cols(&df) == 0 && dataframe_num_rows(&df) == 0, "0x0 frame")
}

@(test)
json_test_write_errors :: proc(t: ^testing.T) {
	c, _ := column_from("n", []i32{1, 2, 3})
	df, _ := dataframe_from_columns([]^Column{&c})
	defer dataframe_destroy(&df)

	path := json_path("unsupported")
	testing.expect(t, dataframe_write_json(&df, path) == .Unsupported_Operation, "i32 unsupported")
	testing.expect(t, dataframe_write_ndjson(&df, path) == .Unsupported_Operation, "i32 unsupported ndjson")
	delete_string(path, context.allocator)

	empty, e_err := dataframe_from_columns([]^Column{})
	testing.expect(t, e_err == .None, "empty frame")
	defer dataframe_destroy(&empty)
	path2 := json_path("empty_write")
	testing.expect(t, dataframe_write_json(&empty, path2) == .Invalid_Argument, "empty frame write")
	delete_string(path2, context.allocator)
}
