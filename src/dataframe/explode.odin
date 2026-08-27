package dataframe

// explode/unnest (Stage 14.3, DESIGN.md §18.1).
//
// dataframe_explode widens the row count: one output row per element of the
// exploded List column, with the other columns repeating the outer row. A NULL
// list row explodes to one all-NULL row. dataframe_unnest splits a struct
// column into one column per field, named `<col>_<field>`, reading the field
// layout with typeid reflection; a NULL struct row yields NULL in every field
// column.

import "core:fmt"
import "core:mem"
import "base:runtime"

// dataframe_explode returns a new DataFrame with one output row per element of
// the List column col_name. Other columns repeat their row value for every
// element; a NULL list row emits one row that is NULL across all columns
// (including elements that are NULL within a valid list). The output owns the
// exploded payload: repeated List columns keep their payload wholesale and get
// recomputed offsets, and in-blob string headers are re-pointed.
dataframe_explode :: proc(df: ^DataFrame, col_name: string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	lc: Column
	lc_idx := -1
	for i in 0 ..< df.columns.count {
		if cs_name(&df.columns, i) == col_name {
			lc = column_set_to_column(&df.columns, i)
			lc_idx = i
			break
		}
	}
	if lc_idx < 0 {
		return {}, .Column_Not_Found
	}
	if lc.dtype != typeid_of(List_Ref) {
		return {}, .Invalid_Argument
	}
	refs := column_typed_view(&lc, List_Ref)
	n_rows := dataframe_num_rows(df)

	total := 0
	for r in 0 ..< n_rows {
		if column_is_valid(&lc, r) {
			total += refs[r].len
		} else {
			total += 1
		}
	}

	out = dataframe_create(allocator)
	for ci in 0 ..< df.columns.count {
		src := column_set_to_column(&df.columns, ci)
		c: Column
		if ci == lc_idx {
			c, err = explode_elements(&lc, refs, total, allocator)
		} else if src.dtype == typeid_of(List_Ref) {
			c, err = explode_repeat_list(&src, &lc, refs, total, allocator)
		} else {
			c, err = explode_repeat(&src, &lc, refs, total, allocator)
		}
		if err != .None {
			dataframe_destroy(&out)
			return {}, err
		}
		if a_err := dataframe_add_column(&out, &c); a_err != .None {
			column_destroy(&c)
			dataframe_destroy(&out)
			return {}, .Allocator_Failure
		}
	}
	return out, .None
}

// explode_elements builds the exploded column: one inner element per output
// row (NULL elements and NULL list rows stay NULL). Owned string contents are
// deep-copied into the output column (mirrors column_copy/concat_columns).
@(private)
explode_elements :: proc(lc: ^Column, refs: []List_Ref, total: int, allocator: mem.Allocator) -> (out: Column, err: Error) {
	inner_size, inner_align, ok := type_layout(lc.inner_dtype)
	if !ok {
		return {}, .Unsupported_Operation
	}
	out, err = column_alloc(allocator, lc.name, lc.inner_dtype, inner_size, inner_align, total)
	if err != .None {
		return {}, err
	}
	out.valid = bm_make(total, false, allocator)
	if out.valid == nil && total != 0 {
		column_destroy(&out)
		return {}, .Allocator_Failure
	}
	// zero the buffer so invalid rows never expose garbage string headers
	// to finalize_string_contents.
	if total != 0 {
		mem.zero(out.data, total * inner_size)
	}
	row := 0
	for r in 0 ..< lc.count {
		if !column_is_valid(lc, r) {
			row += 1
			continue
		}
		for e in refs[r].off ..< refs[r].off + refs[r].len {
			if inner_is_valid(lc, e) {
				mem.copy(ptr_offset(out.data, row * inner_size), ptr_offset(lc.payload, e * inner_size), inner_size)
				bm_set(out.valid, row, true)
			}
			row += 1
		}
	}
	if lc.inner_dtype == typeid_of(string) {
		if f_err := finalize_string_contents(&out, uintptr(lc.payload), lc.payload_size, allocator); f_err != .None {
			column_destroy(&out)
			return {}, f_err
		}
	}
	return out, .None
}

// finalize_string_contents copies the bytes of every string in col whose data
// points inside [old_base, old_base + old_size) into a new owned payload and
// re-points the headers (used to take ownership of contents that live in a
// source column's blob). Borrowed strings (data outside the range) are left
// as-is.
@(private)
finalize_string_contents :: proc(col: ^Column, old_base: uintptr, old_size: int, allocator: mem.Allocator) -> Error {
	ov := column_typed_view(col, string)
	total := 0
	for s in ov {
		raw := transmute(runtime.Raw_String)s
		if raw.data != nil && uintptr(raw.data) >= old_base && uintptr(raw.data) < old_base + uintptr(old_size) {
			total += raw.len
		}
	}
	if total == 0 {
		return .None
	}
	blob, a_err := mem.alloc(total, 1, allocator)
	if a_err != .None || blob == nil {
		return .Allocator_Failure
	}
	cursor := uintptr(blob)
	for &s in ov {
		raw := transmute(runtime.Raw_String)s
		if raw.data != nil && uintptr(raw.data) >= old_base && uintptr(raw.data) < old_base + uintptr(old_size) {
			if raw.len != 0 {
				mem.copy(rawptr(cursor), raw.data, raw.len)
			}
			s = transmute(string)runtime.Raw_String {data = (^u8)(cursor), len = raw.len}
			cursor += uintptr(raw.len)
		}
	}
	col.payload = blob
	col.payload_size = total
	return .None
}

// explode_repeat builds a repeated column: every element of the exploded row
// repeats the source row's value; NULL list rows and NULL source rows yield
// NULL. Categorical category tables and owned string payloads are carried.
@(private)
explode_repeat :: proc(src: ^Column, lc: ^Column, refs: []List_Ref, total: int, allocator: mem.Allocator) -> (out: Column, err: Error) {
	out, err = column_alloc(allocator, src.name, src.dtype, src.elem_size, src.align, total)
	if err != .None {
		return {}, err
	}
	out.valid = bm_make(total, false, allocator)
	if out.valid == nil && total != 0 {
		column_destroy(&out)
		return {}, .Allocator_Failure
	}
	row := 0
	for r in 0 ..< lc.count {
		if !column_is_valid(lc, r) {
			row += 1
			continue
		}
		for _ in 0 ..< refs[r].len {
			if row_valid(src.valid, r) {
				mem.copy(ptr_offset(out.data, row * src.elem_size), ptr_offset(src.data, r * src.elem_size), src.elem_size)
				bm_set(out.valid, row, true)
			}
			row += 1
		}
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
					s = transmute(string)runtime.Raw_String {
						data = (^u8)(new_base + (uintptr(raw.data) - old_base)),
						len  = raw.len,
					}
				}
			}
		}
	}
	return out, .None
}

// explode_repeat_list repeats a non-exploded List column: each output row gets
// its own copy of the source row's element region (an exploded row can repeat
// the same list several times), the List_Ref offsets are recomputed per output
// row, and string inner contents are appended to the new blob and re-pointed.
@(private)
explode_repeat_list :: proc(src: ^Column, lc: ^Column, refs: []List_Ref, total: int, allocator: mem.Allocator) -> (out: Column, err: Error) {
	out, err = column_alloc(allocator, src.name, List_Ref, size_of(List_Ref), align_of(List_Ref), total)
	if err != .None {
		return {}, err
	}
	out.inner_dtype = src.inner_dtype
	if src.inner_valid != nil {
		iv, iv_err := clone_slice(allocator, src.inner_valid)
		if iv_err != .None {
			column_destroy(&out)
			return {}, iv_err
		}
		out.inner_valid = iv
	}
	out.valid = bm_make(total, false, allocator)
	if out.valid == nil && total != 0 {
		column_destroy(&out)
		return {}, .Allocator_Failure
	}
	src_refs := column_typed_view(src, List_Ref)
	out_refs := column_typed_view(&out, List_Ref)

	elem_size := src.elem_size
	row := 0
	total_elements := 0
	for r in 0 ..< lc.count {
		n := 1
		if column_is_valid(lc, r) {
			n = refs[r].len
		}
		total_elements += n * src_refs[r].len
	}
	elem_bytes := total_elements * elem_size
	contents_total := 0
	if src.inner_dtype == typeid_of(string) && src.payload != nil && list_total_elems(src) != 0 {
		old_base := uintptr(src.payload)
		old_end := old_base + uintptr(src.payload_size)
		src_strs := list_inner_view(src, string)
		for s in src_strs {
			raw := transmute(runtime.Raw_String)s
			if raw.data != nil && uintptr(raw.data) >= old_base && uintptr(raw.data) < old_end {
				contents_total += raw.len
			}
		}
	}
	if elem_bytes + contents_total > 0 {
		blob, a_err := mem.alloc(elem_bytes + contents_total, 1, allocator)
		if a_err != .None || blob == nil {
			column_destroy(&out)
			return {}, .Allocator_Failure
		}
		out.payload = blob
		out.payload_size = elem_bytes + contents_total
		elem_cursor := 0
		for r in 0 ..< lc.count {
			n := 1
			if column_is_valid(lc, r) {
				n = refs[r].len
			}
			sr := src_refs[r]
			for _ in 0 ..< n {
				out_refs[row] = List_Ref {off = elem_cursor, len = sr.len}
				if sr.len != 0 {
					mem.copy(ptr_offset(blob, elem_cursor * elem_size), ptr_offset(src.payload, sr.off * elem_size), sr.len * elem_size)
				}
				elem_cursor += sr.len
				row += 1
			}
		}
		if src.inner_dtype == typeid_of(string) {
			old_base := uintptr(src.payload)
			old_end := old_base + uintptr(src.payload_size)
			cursor := uintptr(blob) + uintptr(elem_bytes)
			dst_strs := list_inner_view(&out, string)
			for _, e in dst_strs {
				raw := transmute(runtime.Raw_String)dst_strs[e]
				if raw.data != nil && uintptr(raw.data) >= old_base && uintptr(raw.data) < old_end {
					if raw.len != 0 {
						mem.copy(rawptr(cursor), raw.data, raw.len)
					}
					dst_strs[e] = transmute(string)runtime.Raw_String {
						data = (^u8)(cursor),
						len  = raw.len,
					}
					cursor += uintptr(raw.len)
				}
			}
		}
	}
	row = 0
	for r in 0 ..< lc.count {
		n := 1
		if column_is_valid(lc, r) {
			n = refs[r].len
		}
		for _ in 0 ..< n {
			if column_is_valid(lc, r) && row_valid(src.valid, r) {
				bm_set(out.valid, row, true)
			}
			row += 1
		}
	}
	return out, .None
}

// dataframe_unnest returns a new DataFrame with the struct-valued column
// col_name split into one column per field, named `<col>_<field>`. Field
// layout is read with typeid reflection; a NULL struct row yields NULL in
// every field column. String fields borrow their contents (as in column_from);
// List-typed fields are unsupported.
dataframe_unnest :: proc(df: ^DataFrame, col_name: string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	src := dataframe_get_column(df, col_name) or_return
	si, is_struct := unnest_struct_info(src.dtype)
	if !is_struct {
		return {}, .Invalid_Argument
	}
	n_fields := int(si.field_count)

	out = dataframe_create(allocator)
	// allocate every field column first so failures are clean.
	for f in 0 ..< n_fields {
		ft := si.types[f]
		if ft.id == typeid_of(List_Ref) {
			dataframe_destroy(&out)
			return {}, .Unsupported_Operation
		}
		c, c_err := column_alloc(allocator, fmt.tprintf("%s_%s", col_name, si.names[f]), ft.id, int(ft.size), int(ft.align), src.count)
		if c_err != .None {
			dataframe_destroy(&out)
			return {}, c_err
		}
		if a_err := dataframe_add_column(&out, &c); a_err != .None {
			column_destroy(&c)
			dataframe_destroy(&out)
			return {}, .Allocator_Failure
		}
	}
	if src.count == 0 {
		return out, .None
	}
	for f in 0 ..< n_fields {
		ft := si.types[f]
		field_off := int(si.offsets[f])
		field_size := int(ft.size)
		dst := ptr_offset(cs_data(&out.columns, f), 0)
		base := ptr_offset(src.data, 0)
		out.columns.valids[f] = bm_make(src.count, false, allocator)
		out.col_views[f].valid = out.columns.valids[f]
		if out.columns.valids[f] == nil {
			dataframe_destroy(&out)
			return {}, .Allocator_Failure
		}
		for r in 0 ..< src.count {
			if row_valid(src.valid, r) {
				mem.copy(ptr_offset(dst, r * field_size), ptr_offset(base, r * src.elem_size + field_off), field_size)
				bm_set(out.columns.valids[f], r, true)
			}
		}
	}
	return out, .None
}

// unnest_struct_info returns the Type_Info_Struct of dtype, unwrapping named
// types, or ok=false when dtype is not a struct.
@(private)
unnest_struct_info :: proc(dtype: typeid) -> (si: runtime.Type_Info_Struct, ok: bool) {
	ti := type_info_of(dtype)
	#partial switch v in ti.variant {
	case runtime.Type_Info_Named:
		return unnest_struct_info(v.base.id)
	case runtime.Type_Info_Struct:
		return v, true
	case:
		return {}, false
	}
}
