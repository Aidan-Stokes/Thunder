package dataframe

// Stage 9 CSV I/O tests (ROADMAP S9.6): round-trip, type inference vs
// explicit schema, delimiters, NULL tokens, quoting, malformed rows, and
// error cases. Tests write scratch files under the OS temp dir.

import "core:fmt"
import "core:math"
import "core:os"
import "core:strings"
import "core:testing"

// csv_path returns a unique scratch path for a test.
csv_path :: proc(name: string) -> string {
	return strings.concatenate([]string{"/tmp/thunder_csv_", name, ".csv"})
}

// csv_write_text writes raw CSV text to a scratch file.
csv_write_text :: proc(t: ^testing.T, name: string, text: string) -> string {
	path := csv_path(name)
	if err := os.write_entire_file_from_string(path, text); err != os.ERROR_NONE {
		testing.expect(t, false, fmt.tprintf("write %s: %v", name, err))
	}
	return path
}

// csv_cleanup removes a scratch file and frees the path string returned by
// csv_path.
csv_cleanup :: proc(t: ^testing.T, path: string) {
	if err := os.remove(path); err != os.ERROR_NONE {
		testing.expect(t, false, fmt.tprintf("remove %s: %v", path, err))
	}
	delete_string(path, context.allocator)
}

// csv_expect_dtypes asserts the column dtypes of df.
csv_expect_dtypes :: proc(t: ^testing.T, df: ^DataFrame, want: []typeid) {
	testing.expect(t, dataframe_num_cols(df) == len(want), "column count")
	for w, i in want {
		col := dataframe_column_at(df, i) or_else nil
		testing.expect(t, col.dtype == w, fmt.tprintf("col %d dtype", i))
	}
}

// --- round-trip -------------------------------------------------------------

@(test)
csv_test_roundtrip :: proc(t: ^testing.T) {
	b, i, f, s: Column
	b, _ = column_from("flag", []bool{true, false, true})
	testing.expect(t, column_set_valid(&b, 1, false) == .None, "flag[1] NULL")
	i, _ = column_from("n", []i64{1, -2, 3})
	testing.expect(t, column_set_valid(&i, 2, false) == .None, "n[2] NULL")
	f, _ = column_from("x", []f64{0.1, 1.0 / 3.0, 1e-300})
	s, _ = column_from("s", []string{"hello", "world,wide", "say \"hi\""})
	df, df_err := dataframe_from_columns([]^Column{&b, &i, &f, &s})
	testing.expect(t, df_err == .None, "from_columns")
	defer dataframe_destroy(&df)

	path := csv_path("roundtrip")
	defer csv_cleanup(t, path)
	testing.expect(t, dataframe_write_csv(&df, path) == .None, "write")

	back, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&back)

	csv_expect_dtypes(t, &back, []typeid{typeid_of(bool), typeid_of(i64), typeid_of(f64), typeid_of(string)})
	testing.expect(t, dataframe_num_rows(&back) == 3, "3 rows")

	flag := dataframe_get_column(&back, "flag") or_else nil
	testing.expect(t, column_is_valid(flag, 0) && !column_is_valid(flag, 1) && column_is_valid(flag, 2), "flag validity")
	bf, _, _ := column_get(flag, 0, bool)
	testing.expect(t, bf, "flag[0] true")

	nn := dataframe_get_column(&back, "n") or_else nil
	vn, _, _ := column_get(nn, 1, i64)
	testing.expect(t, vn == -2, "n[1] = -2")
	testing.expect(t, !column_is_valid(nn, 2), "n[2] NULL")

	xx := dataframe_get_column(&back, "x") or_else nil
	vx, _, _ := column_get(xx, 1, f64)
	testing.expect(t, near(vx, 1.0 / 3.0), "x[1] = 1/3")

	ss := dataframe_get_column(&back, "s") or_else nil
	v0, _, _ := column_get(ss, 0, string)
	testing.expect(t, v0 == "hello", "s[0]")
	v1, _, _ := column_get(ss, 1, string)
	testing.expect(t, v1 == "world,wide", "s[1] embedded comma")
	v2, _, _ := column_get(ss, 2, string)
	testing.expect(t, v2 == `say "hi"`, "s[2] escaped quote")
}

@(test)
csv_test_roundtrip_f64_specials :: proc(t: ^testing.T) {
	f, _ := column_from("x", []f64{math.nan_f64(), math.inf_f64(1), math.inf_f64(-1), 0.0})
	df, _ := dataframe_from_columns([]^Column{&f})
	defer dataframe_destroy(&df)

	path := csv_path("roundtrip_f64")
	defer csv_cleanup(t, path)
	testing.expect(t, dataframe_write_csv(&df, path) == .None, "write")

	back, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&back)

	xx := dataframe_get_column(&back, "x") or_else nil
	testing.expect(t, xx.dtype == typeid_of(f64), "inferred f64")
	v0, ok0, _ := column_get(xx, 0, f64)
	testing.expect(t, ok0 && math.is_nan(v0), "NaN round-trip")
	v1, _, _ := column_get(xx, 1, f64)
	testing.expect(t, math.is_inf(v1, 1), "+Inf round-trip")
	v2, _, _ := column_get(xx, 2, f64)
	testing.expect(t, math.is_inf(v2, -1), "-Inf round-trip")
	v3, _, _ := column_get(xx, 3, f64)
	testing.expect(t, v3 == 0.0, "0 round-trip")
}

@(test)
csv_test_empty_strings :: proc(t: ^testing.T) {
	// Empty fields are NULL by default. A single-column frame cannot carry
	// empty values (blank lines are skipped by the stdlib reader), so use
	// two columns: each row is still a non-blank line.
	s, _ := column_from("s", []string{"", "a", ""})
	t2, _ := column_from("t", []string{"x", "y", "z"})
	df, _ := dataframe_from_columns([]^Column{&s, &t2})
	defer dataframe_destroy(&df)

	path := csv_path("empty_strings")
	defer csv_cleanup(t, path)
	testing.expect(t, dataframe_write_csv(&df, path) == .None, "write")

	back, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&back)

	ss := dataframe_get_column(&back, "s") or_else nil
	testing.expect(t, ss.dtype == typeid_of(string), "inferred string")
	testing.expect(t, !column_is_valid(ss, 0) && column_is_valid(ss, 1) && !column_is_valid(ss, 2), "validity")
	vs, _, _ := column_get(ss, 1, string)
	testing.expect(t, vs == "a", "middle value")
}

// --- type inference ---------------------------------------------------------

@(test)
csv_test_inference :: proc(t: ^testing.T) {
	path := csv_write_text(t, "inference",
		"id,age,score,ok,name\n" +
		"1,25,3.5,true,ada\n" +
		"2,,,false,grace\n" +
		"3,30,4.25,true,\n")
	defer csv_cleanup(t, path)

	df, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df)

	csv_expect_dtypes(t, &df, []typeid{typeid_of(i64), typeid_of(i64), typeid_of(f64), typeid_of(bool), typeid_of(string)})
	testing.expect(t, dataframe_num_rows(&df) == 3, "3 rows")

	id := dataframe_get_column(&df, "id") or_else nil
	v, _, _ := column_get(id, 2, i64)
	testing.expect(t, v == 3, "id[2] = 3")

	// NULLs from empty fields; NULL does not constrain inference.
	age := dataframe_get_column(&df, "age") or_else nil
	testing.expect(t, !column_is_valid(age, 1), "age[1] NULL")
	ok := dataframe_get_column(&df, "ok") or_else nil
	testing.expect(t, column_is_valid(ok, 1) && column_is_valid(ok, 2), "ok validity")
	name := dataframe_get_column(&df, "name") or_else nil
	testing.expect(t, column_is_valid(name, 0) && !column_is_valid(name, 2), "name validity")
}

@(test)
csv_test_inference_string_fallback :: proc(t: ^testing.T) {
	// A value that cannot be parsed as the sampled type after the sample
	// window is a .Type_Mismatch (never a silent widening).
	path := csv_write_text(t, "inference_bad",
		"a,b\n" +
		"1,2\n" +
		"3,4\n" +
		"5,nope\n")
	defer csv_cleanup(t, path)

	_, r_err := dataframe_read_csv(path, CSV_Options{sample_rows = 2})
	testing.expect(t, r_err == .Type_Mismatch, "type mismatch")
}

@(test)
csv_test_inference_bool_vs_int :: proc(t: ^testing.T) {
	// Strict bool: only true/false. Numeric 0/1 stays i64, not bool.
	path := csv_write_text(t, "inference_bool",
		"b,flag\n" +
		"0,true\n" +
		"1,false\n")
	defer csv_cleanup(t, path)

	df, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df)
	csv_expect_dtypes(t, &df, []typeid{typeid_of(i64), typeid_of(bool)})
}

// --- explicit schema --------------------------------------------------------

@(test)
csv_test_read_with_schema :: proc(t: ^testing.T) {
	path := csv_write_text(t, "with_schema",
		"k,v\n" +
		"1,2.5\n" +
		"2,3.5\n")
	defer csv_cleanup(t, path)

	schema, s_err := schema_create([]Field{
		{name = "k", dtype = typeid_of(i64)},
		{name = "v", dtype = typeid_of(f64)},
	})
	testing.expect(t, s_err == .None, "schema_create")
	defer schema_destroy(&schema)

	df, r_err := dataframe_read_csv_with_schema(path, schema)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df)
	csv_expect_dtypes(t, &df, []typeid{typeid_of(i64), typeid_of(f64)})
}

@(test)
csv_test_read_with_schema_name_mismatch :: proc(t: ^testing.T) {
	path := csv_write_text(t, "with_schema_mismatch", "k,v\n1,2\n")
	defer csv_cleanup(t, path)

	schema, _ := schema_create([]Field{
		{name = "x", dtype = typeid_of(i64)},
		{name = "v", dtype = typeid_of(i64)},
	})
	defer schema_destroy(&schema)

	_, r_err := dataframe_read_csv_with_schema(path, schema)
	testing.expect(t, r_err == .Invalid_Schema, "name mismatch")
}

@(test)
csv_test_read_with_schema_count_mismatch :: proc(t: ^testing.T) {
	path := csv_write_text(t, "with_schema_count", "k,v\n1,2\n")
	defer csv_cleanup(t, path)

	schema, _ := schema_create([]Field{
		{name = "k", dtype = typeid_of(i64)},
	})
	defer schema_destroy(&schema)

	_, r_err := dataframe_read_csv_with_schema(path, schema)
	testing.expect(t, r_err == .Length_Mismatch, "count mismatch")
}

@(test)
csv_test_read_with_schema_unsupported_type :: proc(t: ^testing.T) {
	path := csv_write_text(t, "with_schema_unsupported", "k,v\n1,2\n")
	defer csv_cleanup(t, path)

	schema, _ := schema_create([]Field{
		{name = "k", dtype = typeid_of(i32)},
		{name = "v", dtype = typeid_of(i64)},
	})
	defer schema_destroy(&schema)

	_, r_err := dataframe_read_csv_with_schema(path, schema)
	testing.expect(t, r_err == .Unsupported_Operation, "unsupported dtype")
}

@(test)
csv_test_read_with_schema_value_mismatch :: proc(t: ^testing.T) {
	path := csv_write_text(t, "with_schema_value", "k,v\n1,abc\n")
	defer csv_cleanup(t, path)

	schema, _ := schema_create([]Field{
		{name = "k", dtype = typeid_of(i64)},
		{name = "v", dtype = typeid_of(i64)},
	})
	defer schema_destroy(&schema)

	_, r_err := dataframe_read_csv_with_schema(path, schema)
	testing.expect(t, r_err == .Type_Mismatch, "value mismatch")
}

// --- options ----------------------------------------------------------------

@(test)
csv_test_delimiter :: proc(t: ^testing.T) {
	path := csv_write_text(t, "delimiter",
		"a;b\n" +
		"1;2\n" +
		"3;4\n")
	defer csv_cleanup(t, path)

	df, r_err := dataframe_read_csv(path, CSV_Options{delimiter = ';'})
	testing.expect(t, r_err == .None, "read with ; delimiter")
	defer dataframe_destroy(&df)
	testing.expect(t, dataframe_num_cols(&df) == 2, "2 cols")

	// Round-trip with the delimiter via write_csv.
	path2 := csv_path("delimiter_out")
	defer csv_cleanup(t, path2)
	testing.expect(t, dataframe_write_csv(&df, path2, CSV_Options{delimiter = ';'}) == .None, "write ;")
	back, b_err := dataframe_read_csv(path2, CSV_Options{delimiter = ';'})
	testing.expect(t, b_err == .None, "re-read ;")
	defer dataframe_destroy(&back)
	testing.expect(t, dataframe_num_rows(&back) == 2, "2 rows")
}

@(test)
csv_test_custom_null_token :: proc(t: ^testing.T) {
	path := csv_write_text(t, "null_token",
		"a,b\n" +
		"1,NA\n" +
		"NA,x\n")
	defer csv_cleanup(t, path)

	df, r_err := dataframe_read_csv(path, CSV_Options{null_token = "NA"})
	testing.expect(t, r_err == .None, "read with NA token")
	defer dataframe_destroy(&df)

	a := dataframe_get_column(&df, "a") or_else nil
	testing.expect(t, a.dtype == typeid_of(i64), "a i64")
	testing.expect(t, !column_is_valid(a, 1), "a[1] NA -> NULL")
	b := dataframe_get_column(&df, "b") or_else nil
	testing.expect(t, column_is_valid(b, 1), "b[1] x valid")
	testing.expect(t, !column_is_valid(b, 0), "b[0] NA -> NULL")

	// With the NA token the empty string is a real value, not NULL.
	path2 := csv_write_text(t, "null_token2",
		"a\n" +
		"x\n" +
		"\n")
	defer csv_cleanup(t, path2)
	df2, r2_err := dataframe_read_csv(path2, CSV_Options{null_token = "NA"})
	testing.expect(t, r2_err == .None, "read with NA token 2")
	defer dataframe_destroy(&df2)
	a2 := dataframe_get_column(&df2, "a") or_else nil
	testing.expect(t, column_is_valid(a2, 2), "empty is a value with custom token")
	sa, _, _ := column_get(a2, 2, string)
	testing.expect(t, sa == "", "empty string value")
}

@(test)
csv_test_comment :: proc(t: ^testing.T) {
	path := csv_write_text(t, "comment",
		"# header note\n" +
		"a,b\n" +
		"1,2\n" +
		"# data note\n" +
		"3,4\n")
	defer csv_cleanup(t, path)

	df, r_err := dataframe_read_csv(path, CSV_Options{comment = '#'})
	testing.expect(t, r_err == .None, "read with comment")
	defer dataframe_destroy(&df)
	testing.expect(t, dataframe_num_rows(&df) == 2, "comment lines skipped")
}

// --- quoting and malformed input -------------------------------------------

@(test)
csv_test_quoted_multiline :: proc(t: ^testing.T) {
	path := csv_write_text(t, "quoted_multiline",
		"a,b\n" +
		"\"line1\nline2\",\"x\"\n" +
		"\"comma,here\",\"quote\"\"inside\"\"\"\n")
	defer csv_cleanup(t, path)

	df, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df)
	testing.expect(t, dataframe_num_rows(&df) == 2, "2 rows")

	a := dataframe_get_column(&df, "a") or_else nil
	va, _, _ := column_get(a, 0, string)
	testing.expect(t, va == "line1\nline2", "multiline field")
	vb, _, _ := column_get(a, 1, string)
	testing.expect(t, vb == "comma,here", "comma in quotes")
	b := dataframe_get_column(&df, "b") or_else nil
	vc, _, _ := column_get(b, 1, string)
	testing.expect(t, vc == `quote"inside"`, "escaped quotes")
}

@(test)
csv_test_malformed_ragged :: proc(t: ^testing.T) {
	path := csv_write_text(t, "ragged", "a,b\n1\n")
	defer csv_cleanup(t, path)
	_, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .CSV_Error, "ragged row")
}

@(test)
csv_test_malformed_quotes :: proc(t: ^testing.T) {
	path := csv_write_text(t, "bad_quotes", "a,b\n\"unclosed,2\n")
	defer csv_cleanup(t, path)
	_, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .CSV_Error, "unclosed quote")
}

@(test)
csv_test_missing_file :: proc(t: ^testing.T) {
	path := csv_path("does_not_exist")
	defer delete_string(path, context.allocator)
	_, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .CSV_Error, "missing file")
}

@(test)
csv_test_empty_file :: proc(t: ^testing.T) {
	path := csv_write_text(t, "empty", "")
	defer csv_cleanup(t, path)
	_, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .CSV_Error, "empty file has no header")
}

@(test)
csv_test_header_only :: proc(t: ^testing.T) {
	path := csv_write_text(t, "header_only", "a,b,c\n")
	defer csv_cleanup(t, path)
	df, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&df)
	testing.expect(t, dataframe_num_rows(&df) == 0, "0 rows")
	testing.expect(t, dataframe_num_cols(&df) == 3, "schema preserved")
}

@(test)
csv_test_duplicate_header :: proc(t: ^testing.T) {
	path := csv_write_text(t, "dup_header", "a,a\n1,2\n")
	defer csv_cleanup(t, path)
	_, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .Duplicate_Column_Name, "duplicate header")
}

@(test)
csv_test_write_unsupported_dtype :: proc(t: ^testing.T) {
	c, _ := column_from("n", []i32{1, 2, 3})
	df, _ := dataframe_from_columns([]^Column{&c})
	defer dataframe_destroy(&df)

	path := csv_path("unsupported_write")
	testing.expect(t, dataframe_write_csv(&df, path) == .Unsupported_Operation, "i32 unsupported")
	delete_string(path, context.allocator)
}

@(test)
csv_test_header_only_write :: proc(t: ^testing.T) {
	// A 0-row df writes just the header. No sampled values means every
	// column infers as string on read-back (documented fallback).
	a, _ := column_from("a", []i64{})
	b, _ := column_from("b", []string{})
	df, _ := dataframe_from_columns([]^Column{&a, &b})
	defer dataframe_destroy(&df)

	path := csv_path("header_only_write")
	defer csv_cleanup(t, path)
	testing.expect(t, dataframe_write_csv(&df, path) == .None, "write")
	back, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&back)
	testing.expect(t, dataframe_num_rows(&back) == 0, "0 rows")
	csv_expect_dtypes(t, &back, []typeid{typeid_of(string), typeid_of(string)})
}

@(test)
csv_test_write_string_roundtrip_via_text :: proc(t: ^testing.T) {
	// write_csv must survive string columns containing the delimiter and
	// quotes when re-read (round-trip path exercised via csv_test_roundtrip).
	s, _ := column_from("s", []string{"a,b", "c\"d", "e\nf"})
	df, _ := dataframe_from_columns([]^Column{&s})
	defer dataframe_destroy(&df)

	path := csv_path("write_quoting")
	defer csv_cleanup(t, path)
	testing.expect(t, dataframe_write_csv(&df, path) == .None, "write")
	back, r_err := dataframe_read_csv(path)
	testing.expect(t, r_err == .None, "read")
	defer dataframe_destroy(&back)
	ss := dataframe_get_column(&back, "s") or_else nil
	v0, _, _ := column_get(ss, 0, string)
	v1, _, _ := column_get(ss, 1, string)
	v2, _, _ := column_get(ss, 2, string)
	testing.expect(t, v0 == "a,b" && v1 == "c\"d" && v2 == "e\nf", "quoted round-trip")
}

// --- column subset (projection pushdown, S12.1) -----------------------------

csv_col_subset_csv :: proc(t: ^testing.T, name: string) -> string {
	text := strings.join([]string{
		"a,b,c",
		"1,x,3.5",
		"2,y,4.5",
		"3,z,5.5",
	}, "\n")
	path := csv_write_text(t, name, text)
	delete(text)
	return path
}

csv_col_subset_expect :: proc(t: ^testing.T, df: ^DataFrame, cols: []string) {
	testing.expect(t, dataframe_num_cols(df) == len(cols), "subset col count")
	testing.expect(t, dataframe_num_rows(df) == 3, "subset row count")
	for c, i in cols {
		col := dataframe_column_at(df, i) or_else nil
		testing.expect(t, col.name == c, "subset col order")
	}
}

@(test)
csv_test_read_with_columns_subset :: proc(t: ^testing.T) {
	path := csv_col_subset_csv(t, "cols_subset")
	defer csv_cleanup(t, path)

	df, err := dataframe_read_csv_with_columns(path, []string{"a"})
	testing.expect(t, err == .None, "read [a]")
	defer dataframe_destroy(&df)
	csv_col_subset_expect(t, &df, []string{"a"})
	testing.expect(t, df.col_views[0].dtype == typeid_of(i64), "a dtype")
	v, _, _ := column_get(&df.col_views[0], 2, i64)
	testing.expect(t, v == 3, "a[2] = 3")
}

@(test)
csv_test_read_with_columns_order :: proc(t: ^testing.T) {
	path := csv_col_subset_csv(t, "cols_order")
	defer csv_cleanup(t, path)

	df, err := dataframe_read_csv_with_columns(path, []string{"c", "a"})
	testing.expect(t, err == .None, "read [c, a]")
	defer dataframe_destroy(&df)
	csv_col_subset_expect(t, &df, []string{"c", "a"})
	vc, _, _ := column_get(&df.col_views[0], 1, f64)
	testing.expect(t, vc == 4.5, "c[1] = 4.5")
	va, _, _ := column_get(&df.col_views[1], 1, i64)
	testing.expect(t, va == 2, "a[1] = 2")
}

@(test)
csv_test_read_with_columns_unknown :: proc(t: ^testing.T) {
	path := csv_col_subset_csv(t, "cols_unknown")
	defer csv_cleanup(t, path)

	_, err := dataframe_read_csv_with_columns(path, []string{"zzz"})
	testing.expect(t, err == .Column_Not_Found, "unknown column error")
}

@(test)
csv_test_read_with_columns_duplicate :: proc(t: ^testing.T) {
	path := csv_col_subset_csv(t, "cols_duplicate")
	defer csv_cleanup(t, path)

	_, err := dataframe_read_csv_with_columns(path, []string{"a", "a"})
	testing.expect(t, err == .Duplicate_Column_Name, "duplicate column error")
}

@(test)
csv_test_read_with_columns_empty :: proc(t: ^testing.T) {
	_, err := dataframe_read_csv_with_columns("/nonexistent", []string{})
	testing.expect(t, err == .Invalid_Argument, "empty columns error")
}
