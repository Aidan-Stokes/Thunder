package dataframe

import "core:testing"

// --- dataframe_to_string -----------------------------------------------------

@(test)
to_string_mixed_columns :: proc(t: ^testing.T) {
	age, a_err := column_from("age", []i32{25, 30, 35})
	testing.expect(t, a_err == .None, "age column")
	name, n_err := column_from("name", []string{"ada", "grace", "katherine"})
	testing.expect(t, n_err == .None, "name column")
	salary, s_err := column_from_with_valid("salary", []f64{150000, 0, 95000}, []bool{true, false, true})
	testing.expect(t, s_err == .None, "salary column")
	df, d_err := dataframe_from_columns([]^Column{&age, &name, &salary})
	testing.expect(t, d_err == .None, "dataframe")
	defer dataframe_destroy(&df)

	want := "shape: (3, 3)\n" +
		"age  name       salary\n" +
		"25   ada        150000\n" +
		"30   grace      null\n" +
		"35   katherine  95000\n"
	got, g_err := dataframe_to_string(&df)
	testing.expect(t, g_err == .None, "to_string err")
	defer delete(got)
	if got != want {
		testing.expectf(t, got == want, "to_string mismatch\n--- got ---\n%q\n--- want ---\n%q", got, want)
	}
}

@(test)
to_string_null_distinct_from_zero :: proc(t: ^testing.T) {
	x, x_err := column_from_with_valid("x", []f64{0, 1, 0}, []bool{true, true, false})
	testing.expect(t, x_err == .None, "x column")
	s, s_err := column_from_with_valid("s", []string{"", "a", ""}, []bool{true, true, false})
	testing.expect(t, s_err == .None, "s column")
	df, d_err := dataframe_from_columns([]^Column{&x, &s})
	testing.expect(t, d_err == .None, "dataframe")
	defer dataframe_destroy(&df)

	want := "shape: (3, 2)\n" +
		"x     s\n" +
		"0     \n" +
		"1     a\n" +
		"null  null\n"
	got, g_err := dataframe_to_string(&df)
	testing.expect(t, g_err == .None, "to_string err")
	defer delete(got)
	if got != want {
		testing.expectf(t, got == want, "to_string mismatch\n--- got ---\n%q\n--- want ---\n%q", got, want)
	}
}

@(test)
to_string_empty_rows :: proc(t: ^testing.T) {
	id, i_err := column_from("id", []i32{})
	testing.expect(t, i_err == .None, "id column")
	df, d_err := dataframe_from_columns([]^Column{&id})
	testing.expect(t, d_err == .None, "dataframe")
	defer dataframe_destroy(&df)

	want := "shape: (0, 1)\nid\n"
	got, g_err := dataframe_to_string(&df)
	testing.expect(t, g_err == .None, "to_string err")
	defer delete(got)
	testing.expect(t, got == want, "0-row table")
}

@(test)
to_string_empty_dataframe :: proc(t: ^testing.T) {
	df := dataframe_create()
	defer dataframe_destroy(&df)

	want := "shape: (0, 0)\n"
	got, g_err := dataframe_to_string(&df)
	testing.expect(t, g_err == .None, "to_string err")
	defer delete(got)
	testing.expectf(t, got == want, "empty table: got %q want %q", got, want)
}

@(test)
to_string_bool_and_float :: proc(t: ^testing.T) {
	flag, f_err := column_from("flag", []bool{true, false})
	testing.expect(t, f_err == .None, "flag column")
	pi, p_err := column_from("pi", []f64{3.14159, -2.5})
	testing.expect(t, p_err == .None, "pi column")
	df, d_err := dataframe_from_columns([]^Column{&flag, &pi})
	testing.expect(t, d_err == .None, "dataframe")
	defer dataframe_destroy(&df)

	want := "shape: (2, 2)\n" +
		"flag   pi\n" +
		"true   3.14159\n" +
		"false  -2.5\n"
	got, g_err := dataframe_to_string(&df)
	testing.expect(t, g_err == .None, "to_string err")
	defer delete(got)
	if got != want {
		testing.expectf(t, got == want, "to_string mismatch\n--- got ---\n%q\n--- want ---\n%q", got, want)
	}
}

// --- dataframe_print ---------------------------------------------------------

@(test)
print_smoke :: proc(t: ^testing.T) {
	a, a_err := column_from("a", []i64{1, 2})
	testing.expect(t, a_err == .None, "a column")
	df, d_err := dataframe_from_columns([]^Column{&a})
	testing.expect(t, d_err == .None, "dataframe")
	defer dataframe_destroy(&df)
	dataframe_print(&df) // must not crash
}
