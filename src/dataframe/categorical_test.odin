package dataframe

// Stage 18 categorical tests: first-seen code assignment, NULL handling,
// enum validation, category-table ownership through copy/gather/sort/unique,
// and string-based (not code-based) semantics across the query engine.

import "core:testing"

// The category table is ordered by first appearance; codes follow that order.
@test
categorical_first_seen_codes :: proc(t: ^testing.T) {
	c, err := categorical_from_strings("cat", []string{"b", "a", "b", "c", "a"})
	testing.expect(t, err == .None, "categorical_from_strings err")
	defer column_destroy(&c)

	testing.expect(t, c.categorical_kind == .Categorical, "kind == .Categorical")
	levels := categorical_categories(&c)
	testing.expect(t, len(levels) == 3, "3 categories")
	testing.expect(t, levels[0] == "b" && levels[1] == "a" && levels[2] == "c", "first-seen order")
	codes := []i64{0, 1, 0, 2, 1}
	for want, row in codes {
		v, valid, gerr := column_get(&c, row, i64)
		testing.expect(t, gerr == .None, "get code")
		testing.expect(t, valid, "row valid")
		testing.expect(t, v == want, "code value")
	}
}
// Categoricals support NULL: the row is marked invalid and categorical_value
// reports ok=false even though a code byte is stored underneath.
@test
categorical_with_valid_and_null :: proc(t: ^testing.T) {
	c, err := categorical_from_strings_with_valid("cat", []string{"a", "b", "c"}, []bool{true, false, true})
	testing.expect(t, err == .None, "with_valid err")
	defer column_destroy(&c)

	_, valid, _ := column_get(&c, 1, i64)
	testing.expect(t, !valid, "row 1 NULL")
	testing.expect(t, column_is_valid(&c, 1) == false, "row 1 invalid")
	_, ok := categorical_value(&c, 1)
	testing.expect(t, !ok, "NULL has no category value")
	_, valid0, _ := column_get(&c, 0, i64)
	testing.expect(t, valid0, "row 0 valid")
}

// categorical_value returns the category string for a valid row.
@test
categorical_value_roundtrip :: proc(t: ^testing.T) {
	c, err := categorical_from_strings("cat", []string{"delta", "alpha", "delta"})
	testing.expect(t, err == .None, "from_strings err")
	defer column_destroy(&c)

	s, ok := categorical_value(&c, 2)
	testing.expect(t, ok, "value ok")
	testing.expect(t, s == "delta", "roundtrip string")
}

// categories outside the table return a defensive zero result.
@test
categorical_out_of_range_defensive :: proc(t: ^testing.T) {
	c, err := categorical_from_strings("cat", []string{"only"})
	testing.expect(t, err == .None, "from_strings err")
	defer column_destroy(&c)

	i64s := make([]i64, 1, context.allocator)
	defer delete(i64s)
	i64s[0] = 7
	testing.expect(t, column_set(&c, 0, i64s[0]) == .None, "set out-of-range code")
	s, ok := categorical_value(&c, 0)
	testing.expect(t, !ok, "out-of-range code has no value")
	testing.expect(t, s == "", "out-of-range code gives empty string")
}

// enum_from_strings validates values against levels and errors on duplicates.
@test
enum_valid_and_levels :: proc(t: ^testing.T) {
	c, err := enum_from_strings("rank", []string{"low", "mid", "high"}, []string{"mid", "high", "low", "mid"})
	testing.expect(t, err == .None, "enum err")
	defer column_destroy(&c)

	testing.expect(t, c.categorical_kind == .Enum, "kind == .Enum")
	levels := enum_levels(&c)
	testing.expect(t, len(levels) == 3, "3 levels")
	testing.expect(t, levels[0] == "low" && levels[1] == "mid" && levels[2] == "high", "level order")
	codes := []i64{1, 2, 0, 1}
	for want, row in codes {
		v, valid, gerr := column_get(&c, row, i64)
		testing.expect(t, gerr == .None, "get enum code")
		testing.expect(t, valid, "valid")
		testing.expect(t, v == want, "enum code")
	}
}

@test
enum_invalid_value :: proc(t: ^testing.T) {
	c, err := enum_from_strings("rank", []string{"low", "high"}, []string{"low", "bogus"})
	testing.expect(t, err == .Invalid_Argument, "value outside levels rejected")
	column_destroy(&c)
}

@test
enum_duplicate_levels :: proc(t: ^testing.T) {
	c, err := enum_from_strings("rank", []string{"low", "low"}, []string{"low"})
	testing.expect(t, err == .Invalid_Argument, "duplicate levels rejected")
	column_destroy(&c)
}

// Non-categorical columns report no category table and no levels.
@test
categorical_categories_non_categorical :: proc(t: ^testing.T) {
	c, err := column_from("s", []string{"plain", "strings"})
	testing.expect(t, err == .None, "string column")
	defer column_destroy(&c)

	testing.expect(t, categorical_categories(&c) == nil, "no categories")
	testing.expect(t, enum_levels(&c) == nil, "no levels")
}

// column_copy carries the category table; mutating the copy's codes must not
// corrupt the original (fresh table per column).
@test
categorical_copy_carries_table :: proc(t: ^testing.T) {
	c, err := categorical_from_strings("cat", []string{"x", "y", "x"})
	testing.expect(t, err == .None, "from_strings err")
	defer column_destroy(&c)

	cp, cerr := column_copy(&c, context.allocator)
	testing.expect(t, cerr == .None, "copy err")
	defer column_destroy(&cp)

	testing.expect(t, cp.categorical_kind == .Categorical, "copy keeps kind")
	s, ok := categorical_value(&cp, 1)
	testing.expect(t, ok && s == "y", "copy reads categories")
	testing.expect(t, &cp.categories[0] != &c.categories[0], "fresh table per column")
	// Altering the copy must not disturb the source.
	i64s := make([]i64, 1, context.allocator)
	defer delete(i64s)
	i64s[0] = 0
	testing.expect(t, column_set(&cp, 1, i64s[0]) == .None, "set copy code")
	s2, _ := categorical_value(&c, 1)
	testing.expect(t, s2 == "y", "source unchanged")
}

// dataframe_take keeps the category table so categorical_value still works.
@test
categorical_take_carries_table :: proc(t: ^testing.T) {
	c, err := categorical_from_strings("cat", []string{"a", "b", "c", "b"})
	testing.expect(t, err == .None, "from_strings err")
	df, ferr := dataframe_from_columns([]^Column{&c})
	testing.expect(t, ferr == .None, "frame err")
	column_destroy(&c)
	defer dataframe_destroy(&df)

	tk, terr := dataframe_take(&df, []int{3, 1}, context.allocator)
	testing.expect(t, terr == .None, "take err")
	defer dataframe_destroy(&tk)

	out, _ := dataframe_get_column(&tk, "cat")
	testing.expect(t, out != nil, "taken column")
	s, ok := categorical_value(out, 0)
	testing.expect(t, ok && s == "b", "take keeps categories")
	testing.expect(t, out.categorical_kind == .Categorical, "take keeps kind")
}

// Sorting a categorical orders by the category string, not the code.
// First-seen codes: b->0, a->1. String order is a < b, so ascending sort must
// put all a rows before b rows despite code 0 meaning b.
@test
categorical_sort_by_string :: proc(t: ^testing.T) {
	c, err := categorical_from_strings("cat", []string{"b", "a", "b", "a"})
	testing.expect(t, err == .None, "from_strings err")
	df, ferr := dataframe_from_columns([]^Column{&c})
	testing.expect(t, ferr == .None, "frame err")
	column_destroy(&c)
	defer dataframe_destroy(&df)

	sorted, serr := dataframe_sort(&df, []Sort_Key{sort_key("cat")}, context.allocator)
	testing.expect(t, serr == .None, "sort err")
	defer dataframe_destroy(&sorted)

	out, _ := dataframe_get_column(&sorted, "cat")
	values := make([]string, 4, context.allocator)
	defer delete(values)
	for i in 0 ..< 4 {
		s, ok := categorical_value(out, i)
		testing.expect(t, ok, "value ok")
		values[i] = s
	}
	testing.expect(t, values[0] == "a" && values[1] == "a" && values[2] == "b" && values[3] == "b", "string-ordered")
}

// Unique is computed over category strings, so identical strings collapse
// even when their code assignments differ.
@test
categorical_unique_by_string :: proc(t: ^testing.T) {
	c, err := categorical_from_strings("cat", []string{"x", "y", "x", "z"})
	testing.expect(t, err == .None, "from_strings err")
	df, ferr := dataframe_from_columns([]^Column{&c})
	testing.expect(t, ferr == .None, "frame err")
	column_destroy(&c)
	defer dataframe_destroy(&df)

	uniq, uerr := dataframe_unique(&df, []string{"cat"}, context.allocator)
	testing.expect(t, uerr == .None, "unique err")
	defer dataframe_destroy(&uniq)

	testing.expect(t, dataframe_num_rows(&uniq) == 3, "3 distinct strings")
	out, _ := dataframe_get_column(&uniq, "cat")
	seen := make(map[string]bool, 8, context.allocator)
	defer delete(seen)
	for i in 0 ..< 3 {
		s, ok := categorical_value(out, i)
		testing.expect(t, ok, "value ok")
		seen[s] = true
	}
	testing.expect(t, len(seen) == 3, "distinct values")
	testing.expect(t, seen["x"] && seen["y"] && seen["z"], "all categories present")
}

// Joins must match on the category string even when the two sides assigned
// different codes to the same strings (left: x->0,y->1; right: y->0,x->1).
@test
categorical_join_by_string :: proc(t: ^testing.T) {
	l_c, lerr := categorical_from_strings("k", []string{"x", "y"})
	testing.expect(t, lerr == .None, "left cat err")
	l_v, verr := column_from("lv", []i64{1, 2})
	testing.expect(t, verr == .None, "left value err")
	l, ferr := dataframe_from_columns([]^Column{&l_c, &l_v})
	testing.expect(t, ferr == .None, "left frame err")
	column_destroy(&l_c)
	column_destroy(&l_v)
	defer dataframe_destroy(&l)

	r_c, rerr := categorical_from_strings("k", []string{"y", "x"})
	testing.expect(t, rerr == .None, "right cat err")
	r_v, verr2 := column_from("rv", []i64{10, 20})
	testing.expect(t, verr2 == .None, "right value err")
	r, ferr2 := dataframe_from_columns([]^Column{&r_c, &r_v})
	testing.expect(t, ferr2 == .None, "right frame err")
	column_destroy(&r_c)
	column_destroy(&r_v)
	defer dataframe_destroy(&r)

	joined, jerr := dataframe_inner_join(&l, &r, []string{"k"}, []string{"k"}, context.allocator)
	testing.expect(t, jerr == .None, "join err")
	defer dataframe_destroy(&joined)

	// x (row 0) pairs with rv 20; y (row 1) pairs with rv 10.
	testing.expect(t, dataframe_num_rows(&joined) == 2, "2 matches")
	lv, _ := dataframe_get_column(&joined, "lv")
	rv, _ := dataframe_get_column(&joined, "rv")
	for i in 0 ..< 2 {
		lval, _, e1 := column_get(lv, i, i64)
		rval, _, e2 := column_get(rv, i, i64)
		testing.expect(t, e1 == .None && e2 == .None, "read joined values")
		if lval == 1 {
			testing.expect(t, rval == 20, "x joins with x")
		} else {
			testing.expect(t, lval == 2 && rval == 10, "y joins with y")
		}
	}
}
