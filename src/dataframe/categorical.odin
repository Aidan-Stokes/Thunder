package dataframe

// S14.10 categorical / enum dtypes (DESIGN.md §18.7).
//
// A categorical column is dictionary-encoded: the element buffer holds i64
// codes into an owned category table (col.categories), tagged with
// Categorical_Kind. `categorical_from_strings` assigns codes in first-seen
// order; `enum_from_strings` validates every value against a caller-provided
// `levels` list and stores the levels verbatim as the category table.
//
// The table is owned (each string and the slice), so grouping, sorting, and
// copying never depend on the lifetime of the constructor's input strings.

import "core:mem"

// Categorical_Kind distinguishes a plain categorical from a validated enum.
Categorical_Kind :: enum byte {
	None,
	Categorical,
	Enum,
}

// is_categorical reports whether col carries a category table.
@(private)
is_categorical :: proc(col: ^Column) -> bool {
	return col.categories != nil
}

// categorical_from_strings builds a categorical column: codes are assigned in
// first-seen order of the values, so the category table holds the distinct
// values in their first-appearance order. The category strings are copied.
categorical_from_strings :: proc(name: string, values: []string, allocator := context.allocator) -> (Column, Error) {
	return categorical_from_strings_with_valid(name, values, nil, allocator)
}

// categorical_from_strings_with_valid is categorical_from_strings with an
// explicit validity array; NULL rows get code 0 (unused).
categorical_from_strings_with_valid :: proc(name: string, values: []string, valid: []bool, allocator := context.allocator) -> (out: Column, err: Error) {
	if valid != nil && len(valid) != len(values) {
		return {}, .Length_Mismatch
	}

	codes := make([]i64, len(values), allocator)
	if codes == nil && len(values) != 0 {
		return {}, .Allocator_Failure
	}
	index := make(map[string]i64, len(values), allocator)
	defer delete(index)
	owned := make([dynamic]string, 0, len(values), allocator)
	for i in 0 ..< len(values) {
		if valid != nil && !valid[i] {
			continue
		}
		if code, seen := index[values[i]]; seen {
			codes[i] = code
			continue
		}
		code := i64(len(owned))
		own, o_err := clone_name(allocator, values[i])
		if o_err != .None {
			for s in owned {
				delete_string(s, allocator)
			}
			delete(owned)
			delete(codes, allocator)
			return {}, o_err
		}
		append(&owned, own)
		index[values[i]] = code
		codes[i] = code
	}

	out, err = column_from_with_valid(name, codes, valid, allocator)
	delete(codes, allocator)
	if err != .None {
		for s in owned {
			delete_string(s, allocator)
		}
		delete(owned)
		return {}, err
	}
	out.categories = owned[:]
	out.categorical_kind = .Categorical
	return out, .None
}

// enum_from_strings builds a validated categorical column whose codes index
// the caller's `levels` table (copied). A value not present in levels is
// .Invalid_Argument, as is a duplicated level.
enum_from_strings :: proc(name: string, levels: []string, values: []string, allocator := context.allocator) -> (Column, Error) {
	return enum_from_strings_with_valid(name, levels, values, nil, allocator)
}

// enum_from_strings_with_valid is enum_from_strings with an explicit validity
// array; NULL rows get code 0 (unused).
enum_from_strings_with_valid :: proc(name: string, levels: []string, values: []string, valid: []bool, allocator := context.allocator) -> (out: Column, err: Error) {
	if valid != nil && len(valid) != len(values) {
		return {}, .Length_Mismatch
	}

	index := make(map[string]i64, len(levels), allocator)
	defer delete(index)
	for i in 0 ..< len(levels) {
		if _, dup := index[levels[i]]; dup {
			return {}, .Invalid_Argument
		}
		index[levels[i]] = i64(i)
	}

	codes := make([]i64, len(values), allocator)
	if codes == nil && len(values) != 0 {
		return {}, .Allocator_Failure
	}
	for i in 0 ..< len(values) {
		if valid != nil && !valid[i] {
			continue
		}
		code, ok := index[values[i]]
		if !ok {
			delete(codes, allocator)
			return {}, .Invalid_Argument
		}
		codes[i] = code
	}

	out, err = column_from_with_valid(name, codes, valid, allocator)
	delete(codes, allocator)
	if err != .None {
		return {}, err
	}
	table, t_err := clone_strings_owned(allocator, levels)
	if t_err != .None {
		column_destroy(&out)
		return {}, t_err
	}
	out.categories = table
	out.categorical_kind = .Enum
	return out, .None
}

// categorical_categories returns the borrowed category table (code order).
// For a non-categorical column it is nil.
categorical_categories :: proc(col: ^Column) -> []string {
	if !is_categorical(col) {
		return nil
	}
	return col.categories
}

// enum_levels returns the borrowed levels table. For any column that is not
// an enum it is nil.
enum_levels :: proc(col: ^Column) -> []string {
	if col.categorical_kind != .Enum {
		return nil
	}
	return col.categories
}

// categorical_value returns the category string of row i. ok=false for a NULL
// row, a non-categorical column, or an out-of-range code.
categorical_value :: proc(col: ^Column, i: int) -> (value: string, ok: bool) {
	if !is_categorical(col) || !column_is_valid(col, i) {
		return "", false
	}
	code := column_typed_view(col, i64)[i]
	if code < 0 || code >= i64(len(col.categories)) {
		return "", false
	}
	return col.categories[code], true
}
