package dataframe

// Reshaping (Stage 14, DESIGN.md §18.8): dataframe_melt and dataframe_pivot.
//
// melt stacks value columns into long form (variable, value) pairs; pivot
// turns the distinct values of one column into new columns, aggregating a
// value column over each (index, column) cell with the Stage 6 kernels.

import "core:mem"
import "core:strings"
import "base:runtime"
import "expr"

// scalar_to_string renders the value of column col at row as its canonical
// string form: numbers via fmt, bools, strings verbatim, temporal types ISO,
// and categoricals by their category string. Callers must ensure row is valid
// (render_cell renders invalid rows as "null"). The returned string is owned
// by the caller.
scalar_to_string :: proc(col: ^Column, row: int, allocator := context.allocator) -> (string, Error) {
	sb := strings.builder_make(allocator)
	defer strings.builder_destroy(&sb)
	render_cell(col, row, &sb)
	s, cerr := strings.clone(strings.to_string(sb), allocator)
	if cerr != .None {
		return {}, .Allocator_Failure
	}
	return s, .None
}

// dataframe_melt stacks the value columns into long form: one output row per
// (id row x value column). The output carries the id columns, a string column
// named variable_name holding the source column name, and a value_name column
// of stacked values. All value columns must share one dtype (.Type_Mismatch
// otherwise); empty value_vars = every non-id column. The value column is
// value-column-major: all rows of the first value column, then the second,
// and so on (pandas/polars ordering). NULLs are preserved row-wise.
dataframe_melt :: proc(
	df: ^DataFrame,
	id_vars: []string,
	value_vars: []string,
	variable_name := "variable",
	value_name := "value",
	allocator := context.allocator,
) -> (out: DataFrame, err: Error) {
	if variable_name == "" || value_name == "" {
		return {}, .Column_Name_Empty
	}
	if variable_name == value_name {
		return {}, .Invalid_Argument
	}

	id_set := make(map[string]bool, len(id_vars), allocator)
	defer delete(id_set)
	id_cols := make([]^Column, len(id_vars), allocator)
	if id_cols == nil && len(id_vars) != 0 {
		return {}, .Allocator_Failure
	}
	defer delete(id_cols, allocator)
	for name, i in id_vars {
		if name == variable_name || name == value_name || id_set[name] {
			return {}, .Invalid_Argument
		}
		id_set[name] = true
		c, g_err := dataframe_get_column(df, name)
		if g_err != .None {
			return {}, g_err
		}
		id_cols[i] = c
	}

	var_idx := make([dynamic]int, allocator)
	defer delete(var_idx)
	if len(value_vars) == 0 {
		for i in 0 ..< df.columns.count {
			if !id_set[cs_name(&df.columns, i)] {
				append(&var_idx, i)
			}
		}
	} else {
		for name in value_vars {
			if id_set[name] {
				return {}, .Invalid_Argument
			}
			i, found := column_set_get(&df.columns, name)
			if !found {
				return {}, .Column_Not_Found
			}
			append(&var_idx, i)
		}
	}
	n_vars := len(var_idx)
	if n_vars == 0 {
		return {}, .Invalid_Argument
	}
	dtype := cs_dtype(&df.columns, var_idx[0])
	for i in 1 ..< n_vars {
		if cs_dtype(&df.columns, var_idx[i]) != dtype {
			return {}, .Type_Mismatch
		}
	}

	n_rows := dataframe_num_rows(df)
	total := n_rows * n_vars

	out = dataframe_create(allocator)

	// Id columns, repeated once per value column (value-column-major order).
	idx := make([]int, total, allocator)
	if idx == nil && total != 0 {
		dataframe_destroy(&out)
		return {}, .Allocator_Failure
	}
	defer delete(idx, allocator)
	for k in 0 ..< total {
		idx[k] = k % n_rows
	}
	for i in 0 ..< len(id_cols) {
		g, g_err := gather_rows(id_cols[i], allocator, idx)
		if g_err != .None {
			dataframe_destroy(&out)
			return {}, g_err
		}
		if a_err := dataframe_add_column(&out, &g); a_err != .None {
			column_destroy(&g)
			dataframe_destroy(&out)
			return {}, a_err
		}
	}

	// Variable column: the source column name per output row (owned strings).
	names := make([]string, total, allocator)
	if names == nil && total != 0 {
		dataframe_destroy(&out)
		return {}, .Allocator_Failure
	}
	defer delete(names, allocator)
	for k in 0 ..< total {
		names[k] = cs_name(&df.columns, var_idx[k / n_rows])
	}
	vc, v_err := owned_string_column(allocator, variable_name, names)
	if v_err != .None {
		dataframe_destroy(&out)
		return {}, v_err
	}
	if a_err := dataframe_add_column(&out, &vc); a_err != .None {
		column_destroy(&vc)
		dataframe_destroy(&out)
		return {}, a_err
	}

	// Value column: one gathered block per value column, concatenated. Each
	// block owns its strings (gather_rows re-points into a fresh payload), and
	// concat_columns merges the payloads so the result is self-contained.
	all_rows := make([]int, n_rows, allocator)
	if all_rows == nil && n_rows != 0 {
		dataframe_destroy(&out)
		return {}, .Allocator_Failure
	}
	defer delete(all_rows, allocator)
	for r in 0 ..< n_rows {
		all_rows[r] = r
	}
	blocks := make([dynamic]Column, 0, n_vars, allocator)
	defer {
		for &b in blocks {
			column_destroy(&b)
		}
		delete(blocks)
	}
	for i in 0 ..< n_vars {
		src_col := column_set_to_column(&df.columns, var_idx[i])
		blk, b_err := gather_rows(&src_col, allocator, all_rows)
		if b_err != .None {
			dataframe_destroy(&out)
			return {}, b_err
		}
		append(&blocks, blk)
	}
	val, c_err := concat_columns(allocator, value_name, blocks[:])
	if c_err != .None {
		dataframe_destroy(&out)
		return {}, c_err
	}
	if a_err := dataframe_add_column(&out, &val); a_err != .None {
		column_destroy(&val)
		dataframe_destroy(&out)
		return {}, a_err
	}
	return out, .None
}

// dataframe_pivot builds one row per distinct index key (empty index = a
// single group of all rows) and one column per distinct value of the `columns`
// column, named by that value's canonical string form (scalar_to_string).
// Each cell aggregates the `values` column over the matching rows with the
// Stage 6 kernel `agg` (validate_agg governs dtype compatibility); cells with
// no rows are NULL (Count yields 0). Rows where the `columns` value is NULL
// take part in no cell. Two distinct values that stringify identically, or a
// pivot column name colliding with an index column name, are an
// .Invalid_Argument.
dataframe_pivot :: proc(
	df: ^DataFrame,
	index: []string,
	columns: string,
	values: string,
	agg: expr.Agg_Kind,
	q: f64 = 0.5,
	allocator := context.allocator,
) -> (out: DataFrame, err: Error) {
	cols_col, g_err := dataframe_get_column(df, columns)
	if g_err != .None {
		return {}, g_err
	}
	val_col, v_err := dataframe_get_column(df, values)
	if v_err != .None {
		return {}, v_err
	}
	if columns == values {
		return {}, .Invalid_Argument
	}
	for name in index {
		if name == columns || name == values {
			return {}, .Invalid_Argument
		}
	}
	if a_err := validate_agg(agg, val_col.dtype); a_err != .None {
		return {}, a_err
	}
	res_dtype, ok := agg_result_dtype(agg, val_col.dtype)
	if !ok {
		return {}, .Invalid_Argument
	}

	n_rows := dataframe_num_rows(df)

	// Distinct values of the `columns` column in first-appearance order: an
	// encoded-key map to the rows with that value, the owned display names,
	// and a per-row value index (-1 when the row's columns value is NULL).
	val_map := make(map[string][dynamic]int, 0, allocator)
	defer {
		// keys are owned strings also referenced by pivot_order below; free
		// each allocation exactly once (pivot_order only deletes its array).
		for k, rows in val_map {
			delete(rows)
			delete_string(k, allocator)
		}
		delete(val_map)
	}
	pivot_order := make([dynamic]string, allocator)
	defer delete(pivot_order)
	pivot_names := make([dynamic]string, allocator)
	defer {
		for n in pivot_names {
			delete_string(n, allocator)
		}
		delete(pivot_names)
	}
	row_pivot := make([]int, n_rows, allocator)
	if row_pivot == nil && n_rows != 0 {
		return {}, .Allocator_Failure
	}
	defer delete(row_pivot, allocator)
	for i in 0 ..< n_rows {
		row_pivot[i] = -1
	}

	buf := make([dynamic]byte, allocator)
	defer delete(buf)
	one_key := []^Column{cols_col}
	for row in 0 ..< n_rows {
		if !row_valid(cols_col.valid, row) {
			continue
		}
		clear(&buf)
		if e_err := encode_row(one_key, row, &buf); e_err != .None {
			return {}, e_err
		}
		key := string(buf[:])
		if _, exists := val_map[key]; !exists {
			owned, o_err := clone_name(allocator, key)
			if o_err != .None {
				return {}, o_err
			}
			name, n_err := scalar_to_string(cols_col, row, allocator)
			if n_err != .None {
				delete_string(owned, allocator)
				return {}, n_err
			}
			if name == "" {
				delete_string(name, allocator)
				delete_string(owned, allocator)
				return {}, .Column_Name_Empty
			}
			for existing in pivot_names {
				if existing == name {
					delete_string(name, allocator)
					delete_string(owned, allocator)
					return {}, .Invalid_Argument
				}
			}
			for idx_name in index {
				if idx_name == name {
					delete_string(name, allocator)
					delete_string(owned, allocator)
					return {}, .Invalid_Argument
				}
			}
			val_map[owned] = make([dynamic]int, allocator)
			append(&pivot_order, owned)
			append(&pivot_names, name)
		}
		append(&val_map[key], row)
	}
	for v in 0 ..< len(pivot_order) {
		for r in val_map[pivot_order[v]] {
			row_pivot[r] = v
		}
	}

	// Index keys: rows hashed on the encoded key columns (empty index = every
	// row shares the empty key), groups in first-appearance order.
	index_ptrs := make([]^Column, len(index), allocator)
	if index_ptrs == nil && len(index) != 0 {
		return {}, .Allocator_Failure
	}
	defer delete(index_ptrs, allocator)
	for name, i in index {
		c, c_err := dataframe_get_column(df, name)
		if c_err != .None {
			return {}, c_err
		}
		index_ptrs[i] = c
	}
	group_map := make(map[string][dynamic]int, 0, allocator)
	defer {
		// keys are owned strings also referenced by group_order; free each
		// allocation exactly once (group_order only deletes its array).
		for k, rows in group_map {
			delete(rows)
			delete_string(k, allocator)
		}
		delete(group_map)
	}
	group_order := make([dynamic]string, allocator)
	defer delete(group_order)
	for row in 0 ..< n_rows {
		clear(&buf)
		if e_err := encode_row(index_ptrs, row, &buf); e_err != .None {
			return {}, e_err
		}
		key := string(buf[:])
		if _, exists := group_map[key]; !exists {
			owned, o_err := clone_name(allocator, key)
			if o_err != .None {
				return {}, o_err
			}
			group_map[owned] = make([dynamic]int, allocator)
			append(&group_order, owned)
		}
		append(&group_map[key], row)
	}
	n_groups := len(group_order)

	out = dataframe_create(allocator)

	// Index key columns: the first row of each group carries the key values.
	first_rows := make([]int, n_groups, allocator)
	if first_rows == nil && n_groups != 0 {
		dataframe_destroy(&out)
		return {}, .Allocator_Failure
	}
	defer delete(first_rows, allocator)
	for gi in 0 ..< n_groups {
		first_rows[gi] = group_map[group_order[gi]][0]
	}
	for i in 0 ..< len(index) {
		key_col, k_err := gather_rows(index_ptrs[i], allocator, first_rows)
		if k_err != .None {
			dataframe_destroy(&out)
			return {}, k_err
		}
		if a_err := dataframe_add_column(&out, &key_col); a_err != .None {
			column_destroy(&key_col)
			dataframe_destroy(&out)
			return {}, a_err
		}
	}

	// Pivot value columns, one per distinct columns value.
	n_pivots := len(pivot_order)
	pivot_cols := make([]Column, n_pivots, allocator)
	if pivot_cols == nil && n_pivots != 0 {
		dataframe_destroy(&out)
		return {}, .Allocator_Failure
	}
	defer {
		for &pc in pivot_cols {
			column_destroy(&pc)
		}
		delete(pivot_cols, allocator)
	}
	for v in 0 ..< n_pivots {
		pc, a_err := column_alloc(allocator, pivot_names[v], res_dtype, size_of_ty(res_dtype), align_of_ty(res_dtype), n_groups)
		if a_err != .None {
			dataframe_destroy(&out)
			return {}, a_err
		}
		pivot_cols[v] = pc
	}

	// Each cell (group, pivot value) aggregates the rows of the group whose
	// columns value matches. cell_rows[v] is a scratch list reused per group.
	cell_rows := make([dynamic][dynamic]int, 0, n_pivots, allocator)
	defer {
		for &cr in cell_rows {
			delete(cr)
		}
		delete(cell_rows)
	}
	for v in 0 ..< n_pivots {
		append(&cell_rows, make([dynamic]int, allocator))
	}
	for gi in 0 ..< n_groups {
		rows := group_map[group_order[gi]]
		for &cr in cell_rows {
			clear(&cr)
		}
		for row in rows {
			if v := row_pivot[row]; v >= 0 {
				append(&cell_rows[v], row)
			}
		}
		for v in 0 ..< n_pivots {
			if r_err := run_group_agg(allocator, val_col, agg, q, &pivot_cols[v], cell_rows[v][:], gi); r_err != .None {
				dataframe_destroy(&out)
				return {}, r_err
			}
		}
	}
	for v in 0 ..< n_pivots {
		if a_err := dataframe_add_column(&out, &pivot_cols[v]); a_err != .None {
			dataframe_destroy(&out)
			return {}, a_err
		}
	}
	return out, .None
}

// owned_string_column builds an owned string column: each string's bytes are
// copied into the column payload and the headers re-pointed into it.
@(private)
owned_string_column :: proc(allocator: mem.Allocator, name: string, values: []string) -> (out: Column, err: Error) {
	out, err = column_alloc(allocator, name, string, size_of(string), align_of(string), len(values))
	if err != .None {
		return {}, err
	}
	total := 0
	for v in values {
		total += len(v)
	}
	if total > 0 {
		blob, a_err := mem.alloc(total, 1, allocator)
		if a_err != .None || blob == nil {
			column_destroy(&out)
			return {}, .Allocator_Failure
		}
		out.payload = blob
		out.payload_size = total
		cursor := uintptr(blob)
		ov := column_typed_view(&out, string)
		for v, i in values {
			if len(v) != 0 {
				mem.copy(rawptr(cursor), raw_data(v), len(v))
			}
			ov[i] = transmute(string)runtime.Raw_String {
				data = (^u8)(cursor),
				len  = len(v),
			}
			cursor += uintptr(len(v))
		}
	}
	return out, .None
}

// concat_columns stacks blocks vertically into one column of dtype
// blocks[0].dtype, deep-copying each block's string payload and re-pointing
// the merged headers into the copy. Validity arrays are concatenated (a block
// without one contributes all-valid rows). All blocks must share one dtype and
// element size.
@(private)
concat_columns :: proc(allocator: mem.Allocator, name: string, blocks: []Column) -> (out: Column, err: Error) {
	if len(blocks) == 0 {
		return {}, .Invalid_Argument
	}
	dtype := blocks[0].dtype
	total := 0
	for &b in blocks {
		total += b.count
	}
	out, err = column_alloc(allocator, name, dtype, blocks[0].elem_size, blocks[0].align, total)
	if err != .None {
		return {}, err
	}
	if blocks[0].categories != nil {
		cat_copy, c_err := clone_strings_owned(allocator, blocks[0].categories)
		if c_err != .None {
			column_destroy(&out)
			return {}, c_err
		}
		out.categories = cat_copy
		out.categorical_kind = blocks[0].categorical_kind
	}

	dst := out.data
	for &b in blocks {
		if b.data != nil {
			mem.copy(dst, b.data, b.elem_size * b.count)
		}
		dst = ptr_offset(dst, b.elem_size * b.count)
	}

	need_valid := false
	for &b in blocks {
		if b.valid != nil {
			need_valid = true
			break
		}
	}
	if need_valid {
		v := bm_make(total, true, allocator)
		if v == nil && total != 0 {
			column_destroy(&out)
			return {}, .Allocator_Failure
		}
		dv := 0
		for &b in blocks {
			for r in 0 ..< b.count {
				bm_set(v, dv, row_valid(b.valid, r))
				dv += 1
			}
		}
		out.valid = v
	}

	if dtype == typeid_of(string) {
		total_sz := 0
		for &b in blocks {
			total_sz += b.payload_size
		}
		if total_sz > 0 {
			blob, a_err := mem.alloc(total_sz, 1, allocator)
			if a_err != .None || blob == nil {
				column_destroy(&out)
				return {}, .Allocator_Failure
			}
			out.payload = blob
			out.payload_size = total_sz
			cursor := uintptr(blob)
			ov := column_typed_view(&out, string)
			dv := 0
			for &b in blocks {
				old_base := uintptr(b.payload)
				new_base := cursor
				if b.payload_size > 0 {
					mem.copy(rawptr(cursor), b.payload, b.payload_size)
					cursor += uintptr(b.payload_size)
				}
				for r in 0 ..< b.count {
					raw := transmute(runtime.Raw_String)column_typed_view(&b, string)[r]
					if uintptr(raw.data) >= old_base && uintptr(raw.data) < old_base + uintptr(b.payload_size) {
						ov[dv] = transmute(string)runtime.Raw_String {
							data = (^u8)(new_base + (uintptr(raw.data) - old_base)),
							len  = raw.len,
						}
					} else {
						ov[dv] = column_typed_view(&b, string)[r]
					}
					dv += 1
				}
			}
		}
	}
	return out, .None
}
