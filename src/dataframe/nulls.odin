package dataframe

import "base:runtime"
import "core:math"
import "core:mem"
import "expr"

// --- S3.9 null-handling expression evaluation ---------------------------------

// is_nan_eval implements the Is_Nan node: a bool column that is true where the
// float child row is NaN. NULL rows stay NULL.
@(private)
is_nan_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Is_Nan) -> (Column, Error) {
	child, cerr := eval_child(allocator, df, n.expr)
	if cerr != .None {
		return {}, cerr
	}
	defer column_destroy(&child)

	out, err := column_alloc(allocator, child.name, typeid_of(bool), size_of(bool), align_of(bool), child.count)
	if err != .None {
		return {}, err
	}
	ov := column_typed_view(&out, bool)
	switch child.dtype {
	case typeid_of(f32):
		fv := column_typed_view(&child, f32)
		for i in 0 ..< child.count {
			if row_valid(child.valid, i) {
				ov[i] = math.is_nan(fv[i])
			} else {
				_ = column_set_null(&out, i)
			}
		}
	case typeid_of(f64):
		fv := column_typed_view(&child, f64)
		for i in 0 ..< child.count {
			if row_valid(child.valid, i) {
				ov[i] = math.is_nan(fv[i])
			} else {
				_ = column_set_null(&out, i)
			}
		}
	case:
		column_destroy(&out)
		return {}, .Unsupported_Operation
	}
	return out, .None
}

// fill_string_nulls rewrites the payload of a string column so NULL rows carry
// value, compacting the shared string blob and repointing every header. The
// column becomes fully valid.
@(private)
fill_string_nulls :: proc(out: ^Column, value: string) -> Error {
	n_null := 0
	if out.valid != nil {
		n_null = bm_count_false(out.valid, out.count)
	}
	if n_null == 0 {
		return .None
	}

	fill_bytes := n_null * len(value)
	old := out.payload
	old_size := out.payload_size
	new_size := old_size + fill_bytes
	blob: rawptr
	if new_size != 0 {
		b, a_err := mem.alloc(new_size, 1, out.alloc)
		if a_err != .None || b == nil {
			return .Allocator_Failure
		}
		blob = b
		if old != nil {
			mem.copy(blob, old, old_size)
		}
	}
	if old != nil {
		mem.free_with_size(old, old_size, out.alloc)
	}
	out.payload = blob
	out.payload_size = new_size

	old_base := uintptr(old)
	new_base := uintptr(blob)
	sv := column_typed_view(out, string)
	write_at := 0
	for i in 0 ..< out.count {
		if out.valid != nil && !bm_get(out.valid, i) {
			if len(value) != 0 {
				mem.copy(ptr_offset(blob, old_size + write_at * len(value)), raw_data(value), len(value))
			}
			sv[i] = transmute(string)(runtime.Raw_String {
				data = (^u8)(new_base + uintptr(old_size) + uintptr(write_at * len(value))),
				len  = len(value),
			})
			write_at += 1
		}
	}
	if old_base != 0 {
		for &s in sv {
			raw := transmute(runtime.Raw_String)s
			if uintptr(raw.data) >= old_base && uintptr(raw.data) < old_base + uintptr(old_size) {
				s = transmute(string)(runtime.Raw_String {
					data = (^u8)(new_base + (uintptr(raw.data) - old_base)),
					len  = raw.len,
				})
			}
		}
	}
	if out.valid != nil {
		delete(out.valid, out.alloc)
		out.valid = nil
	}
	return .None
}

// fill_null_eval implements the Fill_Null node: the child's NULL rows are
// replaced by a constant value of the same dtype. The result is fully valid.
@(private)
fill_null_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Fill_Null) -> (Column, Error) {
	lit, ok := n.value^.(expr.Lit)
	if !ok {
		return {}, .Invalid_Argument
	}
	child, cerr := eval_child(allocator, df, n.expr)
	if cerr != .None {
		return {}, cerr
	}
	defer column_destroy(&child)
	if lit.dtype != child.dtype {
		return {}, .Type_Mismatch
	}

	out, err := column_copy(&child, allocator)
	if err != .None {
		return {}, err
	}
	if out.valid == nil {
		return out, .None
	}

	if child.dtype == typeid_of(string) {
		value, lit_ok := expr.lit_as(&lit, string)
		if !lit_ok {
			column_destroy(&out)
			return {}, .Type_Mismatch
		}
		if ferr := fill_string_nulls(&out, value); ferr != .None {
			column_destroy(&out)
			return {}, ferr
		}
		return out, .None
	}

	parallel_fill_null(&out, &lit.data, out.count)
	delete(out.valid, out.alloc)
	out.valid = nil
	return out, .None
}

// drop_null_flags frees the validity mask once every row is known-valid.
@(private)
drop_null_flags :: proc(col: ^Column) {
	if col.valid == nil {
		return
	}
	for i in 0 ..< col.count {
		if !bm_get(col.valid, i) {
			return
		}
	}
	delete(col.valid, col.alloc)
	col.valid = nil
}

// fill_forward_eval implements the Forward_Fill node: each NULL row takes the
// value of the nearest preceding non-NULL row. Leading NULLs stay NULL.
@(private)
fill_forward_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Forward_Fill) -> (Column, Error) {
	child, cerr := eval_child(allocator, df, n.expr)
	if cerr != .None {
		return {}, cerr
	}
	defer column_destroy(&child)

	out, err := column_copy(&child, allocator)
	if err != .None {
		return {}, err
	}
	if out.valid == nil {
		return out, .None
	}
	size := out.elem_size
	last := -1
	for i in 0 ..< out.count {
		if bm_get(out.valid, i) {
			last = i
		} else if last >= 0 {
			mem.copy(ptr_offset(out.data, i * size), ptr_offset(out.data, last * size), size)
			bm_set(out.valid, i, true)
		}
	}
	drop_null_flags(&out)
	return out, .None
}

// fill_backward_eval implements the Backward_Fill node: each NULL row takes the
// value of the nearest following non-NULL row. Trailing NULLs stay NULL.
@(private)
fill_backward_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Backward_Fill) -> (Column, Error) {
	child, cerr := eval_child(allocator, df, n.expr)
	if cerr != .None {
		return {}, cerr
	}
	defer column_destroy(&child)

	out, err := column_copy(&child, allocator)
	if err != .None {
		return {}, err
	}
	if out.valid == nil {
		return out, .None
	}
	size := out.elem_size
	next := -1
	for i := out.count - 1; i >= 0; i -= 1 {
		if bm_get(out.valid, i) {
			next = i
		} else if next >= 0 {
			mem.copy(ptr_offset(out.data, i * size), ptr_offset(out.data, next * size), size)
			bm_set(out.valid, i, true)
		}
	}
	drop_null_flags(&out)
	return out, .None
}

// scalar_as_f64 reads row i of a numeric column as f64.
@(private)
scalar_as_f64 :: proc(col: ^Column, i: int) -> f64 {
	switch col.dtype {
	case typeid_of(i8):
		return f64(column_typed_view(col, i8)[i])
	case typeid_of(i16):
		return f64(column_typed_view(col, i16)[i])
	case typeid_of(i32):
		return f64(column_typed_view(col, i32)[i])
	case typeid_of(i64):
		return f64(column_typed_view(col, i64)[i])
	case typeid_of(u8):
		return f64(column_typed_view(col, u8)[i])
	case typeid_of(u16):
		return f64(column_typed_view(col, u16)[i])
	case typeid_of(u32):
		return f64(column_typed_view(col, u32)[i])
	case typeid_of(u64):
		return f64(column_typed_view(col, u64)[i])
	case typeid_of(int):
		return f64(column_typed_view(col, int)[i])
	case typeid_of(uint):
		return f64(column_typed_view(col, uint)[i])
	case typeid_of(f32):
		return f64(column_typed_view(col, f32)[i])
	case typeid_of(f64):
		return column_typed_view(col, f64)[i]
	case:
		return 0
	}
}

// interpolate_eval implements the Interpolate node: NULL rows between two
// non-NULL values are filled by linear interpolation. The result is f64.
// Leading and trailing NULLs stay NULL.
@(private)
interpolate_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Interpolate) -> (Column, Error) {
	child, cerr := eval_child(allocator, df, n.expr)
	if cerr != .None {
		return {}, cerr
	}
	defer column_destroy(&child)
	if !is_numeric_type(child.dtype) {
		return {}, .Unsupported_Operation
	}

	out, err := column_alloc(allocator, child.name, typeid_of(f64), size_of(f64), align_of(f64), child.count)
	if err != .None {
		return {}, err
	}
	if child.valid != nil {
		vc, verr := clone_slice(allocator, child.valid)
		if verr != .None {
			column_destroy(&out)
			return {}, verr
		}
		out.valid = vc
	}

	ov := column_typed_view(&out, f64)
	last_i := -1
	last_v := f64(0)
	pending := 0
	for i in 0 ..< child.count {
		if row_valid(child.valid, i) {
			v := scalar_as_f64(&child, i)
			if pending > 0 && last_i >= 0 {
				step := (v - last_v) / f64(pending + 1)
				for k := last_i + 1; k < i; k += 1 {
					ov[k] = last_v + step * f64(k - last_i)
					if out.valid != nil {
						bm_set(out.valid, k, true)
					}
				}
			}
			ov[i] = v
			last_i = i
			last_v = v
			pending = 0
		} else {
			pending += 1
		}
	}
	return out, .None
}

// coalesce_eval implements the Coalesce node: the first non-NULL value per row
// across the parts, in order. A row with all-NULL parts stays NULL.
@(private)
coalesce_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Coalesce) -> (Column, Error) {
	if len(n.exprs) == 0 {
		return {}, .Invalid_Argument
	}
	cols := make([]Column, len(n.exprs), allocator)
	if cols == nil {
		return {}, .Allocator_Failure
	}
	for i in 0 ..< len(n.exprs) {
		col, cerr := eval_child(allocator, df, n.exprs[i])
		if cerr != .None {
			for j in 0 ..< i {
				column_destroy(&cols[j])
			}
			delete(cols, allocator)
			return {}, cerr
		}
		cols[i] = col
	}
	defer {
		for i in 0 ..< len(cols) {
			column_destroy(&cols[i])
		}
		delete(cols, allocator)
	}

	dtype := cols[0].dtype
	size := cols[0].elem_size
	count := cols[0].count
	for i in 1 ..< len(cols) {
		if cols[i].dtype != dtype {
			return {}, .Type_Mismatch
		}
		if cols[i].count != count {
			return {}, .Length_Mismatch
		}
	}

	out, err := column_alloc(allocator, cols[0].name, dtype, size, cols[0].align, count)
	if err != .None {
		return {}, err
	}

	need_strings := dtype == typeid_of(string)
	blob: rawptr
	if need_strings {
		total_size := 0
		for i in 0 ..< count {
			for &c in cols {
				if row_valid(c.valid, i) {
					total_size += len(column_typed_view(&c, string)[i])
					break
				}
			}
		}
		if total_size != 0 {
			b, a_err := mem.alloc(total_size, 1, allocator)
			if a_err != .None || b == nil {
				column_destroy(&out)
				return {}, .Allocator_Failure
			}
			blob = b
			out.payload = blob
			out.payload_size = total_size
		}
	}

	write_at := 0
	if need_strings {
		for i in 0 ..< count {
			found := false
			for &c in cols {
				if !row_valid(c.valid, i) {
					continue
				}
				s := column_typed_view(&c, string)[i]
				if len(s) != 0 {
					mem.copy(ptr_offset(blob, write_at), raw_data(s), len(s))
				}
				column_typed_view(&out, string)[i] = transmute(string)(runtime.Raw_String {
					data = (^u8)(uintptr(blob) + uintptr(write_at)),
					len  = len(s),
				})
				write_at += len(s)
				found = true
				break
			}
			if !found {
				_ = column_set_null(&out, i)
			}
		}
	} else {
		parallel_coalesce_non_string(&out, cols, count)
	}
	return out, .None
}

// --- S3.11 drop_null rows ------------------------------------------------------

// df_column_names collects the column names of df into an owned slice.
@(private)
df_column_names :: proc(df: ^DataFrame, allocator: mem.Allocator) -> []string {
	names := make([]string, dataframe_num_cols(df), allocator)
	for i in 0 ..< dataframe_num_cols(df) {
		names[i] = cs_name(&df.columns, i)
	}
	return names
}

// drop_nulls_core keeps the rows that have no NULL value in any of names.
@(private)
drop_nulls_core :: proc(df: ^DataFrame, names: []string, allocator: mem.Allocator) -> (out: DataFrame, err: Error) {
	rows := dataframe_num_rows(df)
	keep := make([dynamic]int, 0, rows, allocator)
	defer delete(keep)
	for i in 0 ..< rows {
		row_ok := true
		for j in 0 ..< len(names) {
			col, gerr := dataframe_get_column(df, names[j])
			if gerr != .None {
				return {}, gerr
			}
			if !column_is_valid(col, i) {
				row_ok = false
				break
			}
		}
		if row_ok {
			append(&keep, i)
		}
	}
	return take_columns(df, allocator, keep[:])
}

// dataframe_drop_nulls returns a new DataFrame with every row removed that has
// a NULL value in any of cols. With no cols, any column counts. The result
// copies the data; df is left untouched.
dataframe_drop_nulls :: proc(df: ^DataFrame, cols: []string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	if len(cols) == 0 {
		names := df_column_names(df, allocator)
		defer delete(names, allocator)
		return drop_nulls_core(df, names, allocator)
	}
	return drop_nulls_core(df, cols, allocator)
}
