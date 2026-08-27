package dataframe

// Row distinctness (Stage 4): unique.

import "core:mem"

// dataframe_unique returns a new DataFrame holding the first occurrence of
// each distinct combination of the named columns (keep="first", stable row
// order). An empty cols keeps the whole-row key (all columns). NULL is a
// per-column key value: rows that are NULL in the same key columns share a
// key, and the first NULL-keyed row is kept (polars behavior). The result
// owns its columns.
dataframe_unique :: proc(df: ^DataFrame, cols: []string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	key_cols, k_err := resolve_key_columns(df, allocator, cols)
	if k_err != .None {
		return {}, k_err
	}
	defer delete(key_cols, allocator)

	rows := dataframe_num_rows(df)
	keep := make([]bool, rows, allocator)
	if keep == nil && rows != 0 {
		return {}, .Allocator_Failure
	}
	defer delete(keep, allocator)

	seen := make(map[string]bool, 0, allocator)
	defer {
		for k in seen {
			delete_string(k, allocator)
		}
		delete(seen)
	}

	buf := make([dynamic]byte, allocator)
	defer delete(buf)

	for row in 0 ..< rows {
		if enc_err := encode_row(key_cols, row, &buf); enc_err != .None {
			return {}, enc_err
		}
		key := string(buf[:])
		if key in seen {
			continue
		}
		owned, o_err := clone_name(allocator, key)
		if o_err != .None {
			return {}, o_err
		}
		seen[owned] = true // the map stores the header only; we own the bytes
		keep[row] = true
	}

	n := 0
	for row in 0 ..< rows {
		if keep[row] {
			n += 1
		}
	}
	indices := make([]int, n, allocator)
	if indices == nil && n != 0 {
		return {}, .Allocator_Failure
	}
	defer delete(indices, allocator)
	at := 0
	for row in 0 ..< rows {
		if keep[row] {
			indices[at] = row
			at += 1
		}
	}
	return take_columns(df, allocator, indices)
}

// resolve_key_columns returns borrowed pointers to the columns named in cols;
// an empty cols means all columns (in order). Duplicate names are rejected.
@(private)
resolve_key_columns :: proc(df: ^DataFrame, allocator: mem.Allocator, cols: []string) -> ([]^Column, Error) {
	n := len(cols)
	if n == 0 {
		n = df.columns.count
	}
	out := make([]^Column, n, allocator)
	if out == nil && n != 0 {
		return nil, .Allocator_Failure
	}

	names := cols
	if len(cols) == 0 {
		names = make([]string, n, allocator)
		for i in 0 ..< df.columns.count {
			names[i] = cs_name(&df.columns, i)
		}
	}
	for name, i in names {
		for j in 0 ..< i {
			if names[j] == name {
				delete(out, allocator)
				if len(cols) == 0 {
					delete(names, allocator)
				}
				return nil, .Invalid_Argument
			}
		}
		idx, found := column_set_get(&df.columns, name)
		if !found {
			delete(out, allocator)
			if len(cols) == 0 {
				delete(names, allocator)
			}
			return nil, .Column_Not_Found
		}
		out[i] = &df.col_views[idx]
	}
	if len(cols) == 0 {
		delete(names, allocator)
	}
	return out, .None
}
