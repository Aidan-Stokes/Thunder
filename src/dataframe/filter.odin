package dataframe

// Row-subset operations (Stage 4): filter, take, head, tail, slice, limit.
//
// All of these materialize new column buffers from row indices (DESIGN.md
// §4.3): the result owns its columns and is fully independent of the source,
// which is only borrowed and never modified.

import "core:mem"
import "base:runtime"
import "expr"

// dataframe_filter evaluates predicate (a bool expression) against df and
// returns a new DataFrame holding the rows for which the predicate is true
// and non-NULL, in source order. NULL predicate rows are excluded. The result
// owns its columns.
dataframe_filter :: proc(df: ^DataFrame, predicate: ^expr.Expr, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	oa: OpArena
	op_arena_init(&oa, allocator)
	defer op_arena_destroy(&oa)
	mask, eval_err := expr_eval(allocator, df, predicate, &oa)
	if eval_err != .None {
		return {}, eval_err
	}
	defer column_destroy(&mask)
	if mask.dtype != typeid_of(bool) {
		return {}, .Type_Mismatch
	}

	indices, i_err := mask_true_indices(&mask, allocator)
	if i_err != .None {
		return {}, i_err
	}
	defer delete(indices, allocator)
	return take_columns(df, allocator, indices)
}

// dataframe_take returns a new DataFrame holding the rows at indices, in the
// order given (indices may repeat). Every index must be in [0, num_rows);
// otherwise nothing is consumed and .Out_Of_Bounds is returned. NULL rows are
// carried over as NULL.
dataframe_take :: proc(df: ^DataFrame, indices: []int, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	rows := dataframe_num_rows(df)
	for idx in indices {
		if idx < 0 || idx >= rows {
			return {}, .Out_Of_Bounds
		}
	}
	return take_columns(df, allocator, indices)
}

// dataframe_head returns a new DataFrame holding the first n rows of df.
// n is clamped to the row count; n < 0 is an error.
dataframe_head :: proc(df: ^DataFrame, n: int, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	if n < 0 {
		return {}, .Invalid_Argument
	}
	rows := dataframe_num_rows(df)
	nn := min(n, rows)
	indices := make([]int, nn, allocator)
	if indices == nil && nn != 0 {
		return {}, .Allocator_Failure
	}
	defer delete(indices, allocator)
	for i in 0 ..< nn {
		indices[i] = i
	}
	return take_columns(df, allocator, indices)
}

// dataframe_limit is an alias of dataframe_head (lazy-collect naming).
dataframe_limit :: proc(df: ^DataFrame, n: int, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	return dataframe_head(df, n, allocator)
}

// dataframe_tail returns a new DataFrame holding the last n rows of df.
// n is clamped to the row count; n < 0 is an error.
dataframe_tail :: proc(df: ^DataFrame, n: int, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	if n < 0 {
		return {}, .Invalid_Argument
	}
	rows := dataframe_num_rows(df)
	start := max(rows - n, 0)
	count := rows - start
	indices := make([]int, count, allocator)
	if indices == nil && count != 0 {
		return {}, .Allocator_Failure
	}
	defer delete(indices, allocator)
	for i in 0 ..< count {
		indices[i] = start + i
	}
	return take_columns(df, allocator, indices)
}

// dataframe_slice returns a new DataFrame holding the rows
// [offset, offset+length), clamped to the row count. Negative offset or
// length is an error.
dataframe_slice :: proc(df: ^DataFrame, offset: int, length: int, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	if offset < 0 || length < 0 {
		return {}, .Invalid_Argument
	}
	rows := dataframe_num_rows(df)
	off := min(offset, rows)
	ln := min(length, rows - off)
	indices := make([]int, ln, allocator)
	if indices == nil && ln != 0 {
		return {}, .Allocator_Failure
	}
	defer delete(indices, allocator)
	for i in 0 ..< ln {
		indices[i] = off + i
	}
	return take_columns(df, allocator, indices)
}

// --- private row-gathering machinery -----------------------------------------

// mask_true_indices returns the row indices where mask is valid and true.
@(private)
mask_true_indices :: proc(mask: ^Column, allocator: mem.Allocator) -> ([]int, Error) {
	mv := column_typed_view(mask, bool)
	count := 0
	for i in 0 ..< mask.count {
		if row_valid(mask.valid, i) && mv[i] {
			count += 1
		}
	}
	out := make([]int, count, allocator)
	if out == nil && count != 0 {
		return nil, .Allocator_Failure
	}
	at := 0
	for i in 0 ..< mask.count {
		if row_valid(mask.valid, i) && mv[i] {
			out[at] = i
			at += 1
		}
	}
	return out, .None
}

// take_columns materializes a new DataFrame holding the rows at indices
// (validated by the callers). Each column is gathered independently.
@(private)
take_columns :: proc(df: ^DataFrame, allocator: mem.Allocator, indices: []int) -> (out: DataFrame, err: Error) {
	out = dataframe_create(allocator)
	cs := &df.columns
	for i in 0 ..< cs.count {
		// Build a lightweight Column directly from ColumnSet fields,
		// avoiding the full column_set_to_column function call overhead.
		meta := &cs.metas[i]
		col := Column{
			name            = cs.names[i],
			dtype           = cs.dtypes[i],
			elem_size       = cs.sizes[i],
			align           = meta.align,
			count           = cs.rows[i],
			data            = cs.data[i],
			valid           = cs.valids[i],
			alloc           = meta.alloc,
			payload         = meta.payload,
			payload_size    = meta.payload_size,
			categories      = meta.categories,
			categorical_kind = meta.categorical_kind,
			inner_dtype     = meta.inner_dtype,
			inner_valid     = meta.inner_valid,
		}
		gathered, g_err := gather_rows(&col, allocator, indices)
		if g_err != .None {
			dataframe_destroy(&out)
			return {}, g_err
		}
		if a_err := dataframe_add_column(&out, &gathered); a_err != .None {
			column_destroy(&gathered)
			dataframe_destroy(&out)
			return {}, a_err
		}
	}
	return out, .None
}

// gather_rows returns a new column holding src's rows at indices. Element
// buffers are copied bytewise (works for any element type, no boxing); NULL
// flags are copied row-wise; an owned string payload is deep-copied and the
// gathered string headers re-pointed into the copy (mirrors column_copy).
// Callers must validate indices against src.count.
@(private)
gather_rows :: proc(src: ^Column, allocator: mem.Allocator, indices: []int) -> (out: Column, err: Error) {
	return gather_rows_core(src, allocator, src.name, indices, false)
}

// gather_rows_core is gather_rows plus two join needs: the output column may
// take a different name (suffix handling), and a -1 index means "NULL row"
// (no copy; the row is marked invalid) rather than an out-of-bounds access.
// Callers must validate all non-negative indices against src.count.
@(private)
gather_rows_core :: proc(src: ^Column, allocator: mem.Allocator, name: string, indices: []int, null_sentinel: bool) -> (out: Column, err: Error) {
	out, err = column_alloc(allocator, name, src.dtype, src.elem_size, src.align, len(indices))
	if err != .None {
		return {}, err
	}
	if src.categories != nil {
		cat_copy, c_err := clone_strings_owned(allocator, src.categories)
		if c_err != .None {
			column_destroy(&out)
			return {}, c_err
		}
		out.categories = cat_copy
		out.categorical_kind = src.categorical_kind
	}
	if src.inner_dtype != nil {
		iv, iv_err := clone_slice(allocator, src.inner_valid)
		if iv_err != .None {
			column_destroy(&out)
			return {}, iv_err
		}
		out.inner_valid = iv
		out.inner_dtype = src.inner_dtype
	}
	if len(indices) == 0 {
		return out, .None
	}

	has_null := null_sentinel && slice_has_negative(indices)
	if len(indices) >= PARALLEL_GATHER_THRESHOLD && src.elem_size > 0 && src.data != nil {
		parallel_gather_rows_core(src, &out, indices, null_sentinel, has_null)
	} else {
		for idx, i in indices {
			if null_sentinel && idx < 0 {
				continue
			}
			src_ptr := ptr_offset(src.data, idx * src.elem_size)
			dst_ptr := ptr_offset(out.data, i * src.elem_size)
			mem.copy(dst_ptr, src_ptr, src.elem_size)
		}
	}

	if src.valid != nil || has_null {
		v := bm_make(len(indices), true, allocator)
		if v == nil {
			column_destroy(&out)
			return {}, .Allocator_Failure
		}
		for idx, i in indices {
			if null_sentinel && idx < 0 {
				bm_set(v, i, false)
			} else {
				bm_set(v, i, row_valid(src.valid, idx))
			}
		}
		out.valid = v
	}

	if src.payload != nil {
		blob, a_err := mem.alloc(src.payload_size, 1, allocator)
		if a_err != .None || blob == nil {
			column_destroy(&out)
			return {}, .Allocator_Failure
		}
		mem.copy(blob, src.payload, src.payload_size)
		out.payload = blob
		out.payload_size = src.payload_size
		if src.dtype == typeid_of(string) {
			old_base := uintptr(src.payload)
			new_base := uintptr(blob)
			ov := column_typed_view(&out, string)
			for &s in ov {
				raw := transmute(runtime.Raw_String)s
				if uintptr(raw.data) >= old_base && uintptr(raw.data) < old_base + uintptr(src.payload_size) {
					repoint := runtime.Raw_String {
						data = (^u8)(new_base + (uintptr(raw.data) - old_base)),
						len  = raw.len,
					}
					s = transmute(string)repoint
				}
			}
		}
		// List columns keep the whole element buffer (offsets stay valid);
		// string inner headers are re-pointed the same way.
		if src.dtype == typeid_of(List_Ref) && src.inner_dtype == typeid_of(string) {
			old_base := uintptr(src.payload)
			new_base := uintptr(blob)
			ov := list_inner_view(&out, string)
			for &s in ov {
				raw := transmute(runtime.Raw_String)s
				if uintptr(raw.data) >= old_base && uintptr(raw.data) < old_base + uintptr(src.payload_size) {
					repoint := runtime.Raw_String {
						data = (^u8)(new_base + (uintptr(raw.data) - old_base)),
						len  = raw.len,
					}
					s = transmute(string)repoint
				}
			}
		}
	}
	return out, .None
}

// slice_has_negative reports whether any element of indices is negative
// (used by gather_rows_core to decide whether a validity array is needed).
@(private)
slice_has_negative :: proc(indices: []int) -> bool {
	for idx in indices {
		if idx < 0 {
			return true
		}
	}
	return false
}
