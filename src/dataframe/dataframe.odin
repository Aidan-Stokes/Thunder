package dataframe

import "core:mem"

// DataFrame is an ordered, owned collection of typed columns (DESIGN.md §4).
// It is the eager root object of the public API.
//
// Storage is two-level:
//   - `columns` (ColumnSet): SoA layout for hot-path iteration (filter,
//     join, group_by, print). Only the fields needed in tight loops are
//     stored per-column in separate arrays.
//   - `col_views` ([dynamic]Column): backward-compatible view for existing
//     code that expects ^Column pointers. Views BORROW data from ColumnSet;
//     they own nothing and must never be destroyed by callers.
//
// Ownership (DESIGN.md §4.1–4.3):
//   - A DataFrame owns its columns (via ColumnSet). Every column's element
//     buffer, validity bitmap, and cold metadata are owned by ColumnSet.
//   - `alloc` is captured at dataframe_create and used for the ColumnSet.
//   - Destroy with dataframe_destroy, which destroys the ColumnSet.
//   - `dataframe_add_column` and `dataframe_from_columns` TRANSFER ownership:
//     on success the source ^Column is zeroed (`col^ = {}`) and must not be
//     destroyed or reused. On validation error nothing is consumed.
//   - `dataframe_get_column` / `dataframe_column_at` return borrowed ^Column
//     pointers into the view array; never destroy them and do not use them
//     past the DataFrame's lifetime.
//
// Derived state: schema and row count are derived from the columns, never
// stored (there is one source of truth — the ColumnSet entry).
DataFrame :: struct {
	columns:  ColumnSet,
	col_views: [dynamic]Column, // backward-compatible borrowed views
	alloc:    mem.Allocator,
}

// dataframe_create returns an empty DataFrame that owns nothing yet.
dataframe_create :: proc(allocator := context.allocator) -> DataFrame {
	return DataFrame {
		columns  = column_set_create(allocator),
		col_views = make([dynamic]Column, allocator),
		alloc    = allocator,
	}
}

// dataframe_destroy destroys every column and releases the ColumnSet and views.
// After destroy the struct is zeroed and must not be used.
dataframe_destroy :: proc(df: ^DataFrame) {
	column_set_destroy(&df.columns)
	// col_views borrow data from ColumnSet — do NOT column_destroy them.
	delete(df.col_views)
	df^ = {}
}

// dataframe_add_column validates col (non-empty, unique name; length matches
// the DataFrame's row count unless it has no columns yet) and then transfers
// ownership: col is zeroed on success (DESIGN.md §4.1). On any error col is
// left untouched.
dataframe_add_column :: proc(df: ^DataFrame, col: ^Column) -> Error {
	if col.name == "" {
		return .Column_Name_Empty
	}
	if dataframe_has_column(df, col.name) {
		return .Duplicate_Column_Name
	}
	if df.columns.count != 0 && df.columns.rows[0] != col.count {
		return .Length_Mismatch
	}
	// Transfer ownership into ColumnSet.
	idx := df.columns.count
	if err := column_set_add(&df.columns, col); err != .None {
		return err
	}
	// Append a borrowed view for backward compatibility.
	view := column_set_to_column(&df.columns, idx)
	append(&df.col_views, view)
	return .None
}

// dataframe_create_with_schema builds a zero-row DataFrame shaped by schema:
// one empty column per field, with the field's name and dtype.
dataframe_create_with_schema :: proc(schema: Schema, allocator := context.allocator) -> (df: DataFrame, err: Error) {
	df = dataframe_create(allocator)
	for field in schema.fields {
		c, cerr := column_empty(field.name, field.dtype, allocator)
		if cerr != .None {
			dataframe_destroy(&df)
			return {}, cerr
		}
		if add_err := dataframe_add_column(&df, &c); add_err != .None {
			column_destroy(&c)
			dataframe_destroy(&df)
			return {}, add_err
		}
	}
	return df, .None
}

// dataframe_from_columns creates a DataFrame from columns. Every column is
// validated first (names non-empty and unique, lengths equal); only then is
// ownership transferred by zeroing each source. On error nothing is consumed.
dataframe_from_columns :: proc(columns: []^Column, allocator := context.allocator) -> (df: DataFrame, err: Error) {
	if err = validate_columns(columns); err != .None {
		return {}, err
	}
	df = dataframe_create(allocator)
	for c in columns {
		if add_err := dataframe_add_column(&df, c); add_err != .None {
			dataframe_destroy(&df)
			return {}, add_err
		}
	}
	for c in columns {
		c^ = {}
	}
	return df, .None
}

// validate_columns checks that every column has a non-empty, unique name and
// that all lengths match. Does not consume anything.
@(private)
validate_columns :: proc(columns: []^Column) -> Error {
	count := -1
	for c, i in columns {
		if c.name == "" {
			return .Column_Name_Empty
		}
		if count == -1 {
			count = c.count
		} else if c.count != count {
			return .Length_Mismatch
		}
		for j in 0 ..< i {
			if columns[j].name == c.name {
				return .Duplicate_Column_Name
			}
		}
	}
	return .None
}

// dataframe_remove_column destroys the named column and removes it from the
// DataFrame, preserving the order of the remaining columns.
dataframe_remove_column :: proc(df: ^DataFrame, name: string) -> Error {
	i, ok := column_set_get(&df.columns, name)
	if !ok {
		return .Column_Not_Found
	}
	column_set_remove(&df.columns, i)
	ordered_remove(&df.col_views, i)
	return .None
}

// dataframe_rename_column renames old_name to new_name.
dataframe_rename_column :: proc(df: ^DataFrame, old_name: string, new_name: string) -> Error {
	if new_name == "" {
		return .Column_Name_Empty
	}
	if old_name == new_name {
		return .None
	}
	i, ok := column_set_get(&df.columns, old_name)
	if !ok {
		return .Column_Not_Found
	}
	if _, dup := column_set_get(&df.columns, new_name); dup {
		return .Duplicate_Column_Name
	}
	meta := &df.columns.metas[i]
	new_copy := clone_name(df.alloc, new_name) or_return
	if df.columns.names[i] != "" {
		delete_string(df.columns.names[i], meta.alloc)
	}
	df.columns.names[i] = new_copy
	// Update the view.
	df.col_views[i].name = new_copy
	return .None
}

// dataframe_get_column returns a borrowed pointer to the named column
// (DESIGN.md §4.2). The pointer is invalidated by dataframe_destroy or
// dataframe_remove_column and must never be freed by the caller.
dataframe_get_column :: proc(df: ^DataFrame, name: string) -> (^Column, Error) {
	i, ok := column_set_get(&df.columns, name)
	if !ok {
		return nil, .Column_Not_Found
	}
	return &df.col_views[i], .None
}

// dataframe_column_at returns a borrowed pointer to the i-th column, in
// insertion order.
dataframe_column_at :: proc(df: ^DataFrame, i: int) -> (^Column, Error) {
	if i < 0 || i >= df.columns.count {
		return nil, .Out_Of_Bounds
	}
	return &df.col_views[i], .None
}

// dataframe_has_column reports whether a column named name exists.
dataframe_has_column :: proc(df: ^DataFrame, name: string) -> bool {
	_, ok := column_set_get(&df.columns, name)
	return ok
}

// dataframe_num_rows returns the row count. It is derived from the columns;
// an empty DataFrame has 0 rows.
dataframe_num_rows :: proc(df: ^DataFrame) -> int {
	if df.columns.count == 0 {
		return 0
	}
	return df.columns.rows[0]
}

// dataframe_num_cols returns the number of columns.
dataframe_num_cols :: proc(df: ^DataFrame) -> int {
	return df.columns.count
}

// dataframe_schema materializes an owned Schema from the columns.
dataframe_schema :: proc(df: ^DataFrame) -> (Schema, Error) {
	n := df.columns.count
	fields := make([]Field, n, df.alloc)
	if fields == nil && n != 0 {
		return {}, .Allocator_Failure
	}
	for i in 0 ..< n {
		fields[i] = Field{name = df.columns.names[i], dtype = df.columns.dtypes[i]}
	}
	return Schema{fields = fields, alloc = df.alloc}, .None
}

// dataframe_copy returns a deep, independent copy of df.
dataframe_copy :: proc(df: ^DataFrame, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	out_cs, cs_err := column_set_copy(&df.columns, allocator)
	if cs_err != .None {
		return {}, cs_err
	}
	out = DataFrame{
		columns  = out_cs,
		col_views = make([dynamic]Column, allocator),
		alloc    = allocator,
	}
	// Rebuild view array from the copied ColumnSet.
	for i in 0 ..< out_cs.count {
		view := column_set_to_column(&out_cs, i)
		append(&out.col_views, view)
	}
	return out, .None
}
