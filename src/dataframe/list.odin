package dataframe

// List dtype (Stage 14.2, DESIGN.md §18.1): a List column stores one List_Ref
// per row indexing into the column's inner element buffer.
//
//   - dtype       = typeid_of(List_Ref)
//   - data        = the List_Ref array (one per row)
//   - payload     = the inner element buffer (all rows' elements concatenated).
//     For string elements the string contents are appended behind the element
//     bytes in the same blob and the headers point into that region.
//   - inner_dtype = the element type; inner_valid = per-element validity
//     (nil == all elements valid). A NULL *element* is distinct from a NULL
//     *row* (the outer `valid` flag says the row's list is NULL).
//
// A NULL outer row leaves its List_Ref zeroed (off/len 0) and is marked
// invalid. Ownership: the payload and inner_valid are owned by the Column,
// freed by column_destroy, deep-copied by column_copy and carried wholesale by
// gather_rows_core.

import "core:fmt"
import "core:mem"
import "base:runtime"

// List_Ref is the per-row header of a List column: off is the element index of
// the row's first element in the payload element region and len is the element
// count. The zero value denotes an empty list (or a NULL row, marked invalid).
List_Ref :: struct {
	off: int,
	len: int,
}

// list_from_column builds a List column by transferring the element column
// elems: its data becomes the payload (for string elements the element column's
// own payload — the string contents — is merged into one blob and the headers
// re-pointed) and its valid array becomes inner_valid. `offsets` has
// len(elems)+1 entries: offsets[0] == 0, offsets[len(offsets)-1] == elems.count,
// monotone non-decreasing. Row r holds elements [offsets[r], offsets[r+1]).
// On success elems is zeroed (ownership transferred); on error it is untouched.
list_from_column :: proc(name: string, elems: ^Column, offsets: []int, allocator := context.allocator) -> (out: Column, err: Error) {
	if name == "" {
		return {}, .Column_Name_Empty
	}
	if len(offsets) == 0 || offsets[0] != 0 || offsets[len(offsets) - 1] != elems.count {
		return {}, .Invalid_Argument
	}
	if elems.dtype == typeid_of(List_Ref) {
		// no nested lists
		return {}, .Unsupported_Operation
	}
	if elems.categories != nil || elems.inner_dtype != nil {
		// categorical/enum and nested inner columns are not list-able yet
		return {}, .Unsupported_Operation
	}
	for i in 1 ..< len(offsets) {
		if offsets[i] < offsets[i - 1] {
			return {}, .Invalid_Argument
		}
	}

	n_rows := len(offsets) - 1
	out, err = column_alloc(allocator, name, List_Ref, size_of(List_Ref), align_of(List_Ref), n_rows)
	if err != .None {
		return {}, err
	}

	// merged payload: element headers, then (for string elements) their bytes.
	header_bytes := elems.count * elems.elem_size
	total := header_bytes + elems.payload_size
	if total > 0 {
		blob, a_err := mem.alloc(total, elems.align, allocator)
		if a_err != .None || blob == nil {
			column_destroy(&out)
			return {}, .Allocator_Failure
		}
		if header_bytes > 0 {
			mem.copy(blob, elems.data, header_bytes)
		}
		if elems.payload_size > 0 {
			mem.copy(ptr_offset(blob, header_bytes), elems.payload, elems.payload_size)
		}
		if elems.dtype == typeid_of(string) && elems.payload != nil && elems.count > 0 {
			// re-point inner string headers that pointed into the old payload.
			old_base := uintptr(elems.payload)
			old_end := old_base + uintptr(elems.payload_size)
			new_base := uintptr(blob) + uintptr(header_bytes)
			headers := transmute([]string)runtime.Raw_Slice{data = blob, len = elems.count}
			for _, i in headers {
				raw := transmute(runtime.Raw_String)headers[i]
				if uintptr(raw.data) >= old_base && uintptr(raw.data) < old_end {
					headers[i] = transmute(string)runtime.Raw_String {
						data = (^u8)(new_base + (uintptr(raw.data) - old_base)),
						len  = raw.len,
					}
				}
			}
		}
		out.payload = blob
		out.payload_size = total
	}

	out.inner_dtype = elems.dtype
	out.inner_valid = elems.valid

	refs := column_typed_view(&out, List_Ref)
	for r in 0 ..< n_rows {
		refs[r] = List_Ref {off = offsets[r], len = offsets[r + 1] - offsets[r]}
	}

	// the element column's data/payload were merged into our blob above and
	// its valid array was transferred; free the leftover allocations.
	if elems.name != "" {
		delete_string(elems.name, elems.alloc)
	}
	if elems.data != nil {
		mem.free_with_size(elems.data, header_bytes, elems.alloc)
	}
	if elems.payload != nil {
		mem.free_with_size(elems.payload, elems.payload_size, elems.alloc)
	}
	elems^ = {}
	return out, .None
}

// list_from_slices builds a List column where row i holds values[i]. String
// contents are borrowed (documented limitation, as in column_from).
list_from_slices :: proc(name: string, values: [][]$T, allocator := context.allocator) -> (out: Column, err: Error) {
	return list_from_slices_with_valid(name, values, nil, allocator)
}

// list_from_slices_with_valid is list_from_slices with an explicit outer
// validity array (`nil` == every row valid).
list_from_slices_with_valid :: proc(name: string, values: [][]$T, row_valid: []bool, allocator := context.allocator) -> (out: Column, err: Error) {
	n_rows := len(values)
	if row_valid != nil && len(row_valid) != n_rows {
		return {}, .Length_Mismatch
	}
	total := 0
	for v in values {
		total += len(v)
	}
	elems := make([]T, total, allocator)
	if elems == nil && total != 0 {
		return {}, .Allocator_Failure
	}
	offsets := make([]int, n_rows + 1, allocator)
	if offsets == nil && n_rows + 1 != 0 {
		delete(elems, allocator)
		return {}, .Allocator_Failure
	}
	k := 0
	for i in 0 ..< n_rows {
		offsets[i] = k
		for v in values[i] {
			elems[k] = v
			k += 1
		}
	}
	offsets[n_rows] = k

	elem_col, cerr := column_from("__elems", elems, allocator)
	if cerr != .None {
		delete(elems, allocator)
		delete(offsets, allocator)
		return {}, cerr
	}
	delete(elems, allocator)

	out, err = list_from_column(name, &elem_col, offsets, allocator)
	if err != .None {
		column_destroy(&elem_col)
		delete(offsets, allocator)
		return {}, err
	}
	delete(offsets, allocator)

	if row_valid != nil {
		out.valid = bm_from_bools(row_valid, allocator)
		if out.valid == nil {
			column_destroy(&out)
			return {}, .Allocator_Failure
		}
		// a NULL outer row leaves its List_Ref zeroed (DESIGN.md §18.1).
		refs := column_typed_view(&out, List_Ref)
		for r in 0 ..< n_rows {
			if !row_valid[r] {
				refs[r] = List_Ref {}
			}
		}
	}
	return out, .None
}

// list_total_elems returns the number of inner elements across all rows. NULL
// outer rows carry a zeroed List_Ref, so the count is the maximum off+len seen
// (offsets are cumulative within the payload).
@(private)
list_total_elems :: proc(col: ^Column) -> int {
	refs := column_typed_view(col, List_Ref)
	total := 0
	for r in 0 ..< col.count {
		if off_plus_len := refs[r].off + refs[r].len; off_plus_len > total {
			total = off_plus_len
		}
	}
	return total
}

// list_inner_view is a typed view of the payload element region.
@(private)
list_inner_view :: proc(col: ^Column, $T: typeid) -> []T {
	return transmute([]T)runtime.Raw_Slice {data = col.payload, len = list_total_elems(col)}
}

// inner_is_valid reports whether inner element e is non-NULL.
@(private)
inner_is_valid :: proc(col: ^Column, e: int) -> bool {
	return col.inner_valid == nil || bm_get(col.inner_valid, e)
}

// list_count returns an i64 column with the element count per row; NULL where
// the row itself is NULL.
list_count :: proc(col: ^Column, allocator := context.allocator) -> (out: Column, err: Error) {
	if col.dtype != typeid_of(List_Ref) {
		return {}, .Invalid_Argument
	}
	out, err = column_alloc(allocator, col.name, i64, size_of(i64), align_of(i64), col.count)
	if err != .None {
		return {}, err
	}
	out.valid, err = clone_slice(allocator, col.valid)
	if err != .None {
		column_destroy(&out)
		return {}, err
	}
	refs := column_typed_view(col, List_Ref)
	dst := column_typed_view(&out, i64)
	for r in 0 ..< col.count {
		dst[r] = i64(refs[r].len)
	}
	return out, .None
}

// list_get returns a column of inner_dtype holding the element at `index` of
// every row; NULL where the row is NULL, the index is out of range, or the
// element is NULL.
list_get :: proc(col: ^Column, index: int, allocator := context.allocator) -> (out: Column, err: Error) {
	if col.dtype != typeid_of(List_Ref) {
		return {}, .Invalid_Argument
	}
	return list_get_impl(col, index, nil, allocator)
}

// list_gather is list_get with a per-row index.
list_gather :: proc(col: ^Column, indices: []int, allocator := context.allocator) -> (out: Column, err: Error) {
	if col.dtype != typeid_of(List_Ref) {
		return {}, .Invalid_Argument
	}
	if len(indices) != col.count {
		return {}, .Length_Mismatch
	}
	return list_get_impl(col, 0, indices, allocator)
}

// list_get_impl implements list_get (indices == nil) and list_gather
// (indices != nil). Indices are row-relative when per-row.
@(private)
list_get_impl :: proc(col: ^Column, scalar_index: int, indices: []int, allocator := context.allocator) -> (out: Column, err: Error) {
	inner := col.inner_dtype
	size, align, ok := type_layout(inner)
	if !ok {
		return {}, .Unsupported_Operation
	}
	out, err = column_alloc(allocator, col.name, inner, size, align, col.count)
	if err != .None {
		return {}, err
	}
	valid := bm_make(col.count, false, allocator)
	if valid == nil && col.count != 0 {
		column_destroy(&out)
		return {}, .Allocator_Failure
	}
	refs := column_typed_view(col, List_Ref)
	idx := scalar_index
	switch inner {
	case typeid_of(bool):   list_get_typed(col, refs, idx, indices, bool, &out, valid)
	case typeid_of(i8):     list_get_typed(col, refs, idx, indices, i8, &out, valid)
	case typeid_of(i16):    list_get_typed(col, refs, idx, indices, i16, &out, valid)
	case typeid_of(i32):    list_get_typed(col, refs, idx, indices, i32, &out, valid)
	case typeid_of(i64):    list_get_typed(col, refs, idx, indices, i64, &out, valid)
	case typeid_of(u8):     list_get_typed(col, refs, idx, indices, u8, &out, valid)
	case typeid_of(u16):    list_get_typed(col, refs, idx, indices, u16, &out, valid)
	case typeid_of(u32):    list_get_typed(col, refs, idx, indices, u32, &out, valid)
	case typeid_of(u64):    list_get_typed(col, refs, idx, indices, u64, &out, valid)
	case typeid_of(int):    list_get_typed(col, refs, idx, indices, int, &out, valid)
	case typeid_of(uint):   list_get_typed(col, refs, idx, indices, uint, &out, valid)
	case typeid_of(f32):    list_get_typed(col, refs, idx, indices, f32, &out, valid)
	case typeid_of(f64):    list_get_typed(col, refs, idx, indices, f64, &out, valid)
	case typeid_of(string): list_get_typed(col, refs, idx, indices, string, &out, valid)
	case typeid_of(Date):     list_get_typed(col, refs, idx, indices, Date, &out, valid)
	case typeid_of(Datetime): list_get_typed(col, refs, idx, indices, Datetime, &out, valid)
	case typeid_of(Time):     list_get_typed(col, refs, idx, indices, Time, &out, valid)
	case typeid_of(Duration): list_get_typed(col, refs, idx, indices, Duration, &out, valid)
	case:
		column_destroy(&out)
		delete(valid, allocator)
		return {}, .Unsupported_Operation
	}
	out.valid = valid
	return out, .None
}

// list_get_typed is the element copy kernel for list_get/list_gather.
@(private)
list_get_typed :: proc(col: ^Column, refs: []List_Ref, scalar_index: int, indices: []int, $T: typeid, out: ^Column, valid: []u64) {
	src := list_inner_view(col, T)
	dst := column_typed_view(out, T)
	for r in 0 ..< col.count {
		if !column_is_valid(col, r) {
			continue
		}
		i := scalar_index
		if indices != nil {
			i = indices[r]
		}
		if i < 0 || i >= refs[r].len {
			continue
		}
		e := refs[r].off + i
		if inner_is_valid(col, e) {
			dst[r] = src[e]
			bm_set(valid, r, true)
		}
	}
}

// list_unique returns a List column holding the per-row deduplicated elements
// (first-seen order, NULL elements skipped); NULL rows stay NULL.
list_unique :: proc(col: ^Column, allocator := context.allocator) -> (out: Column, err: Error) {
	if col.dtype != typeid_of(List_Ref) {
		return {}, .Invalid_Argument
	}
	out, err = column_alloc(allocator, col.name, List_Ref, size_of(List_Ref), align_of(List_Ref), col.count)
	if err != .None {
		return {}, err
	}
	out.valid, err = clone_slice(allocator, col.valid)
	if err != .None {
		column_destroy(&out)
		return {}, err
	}
	out.inner_dtype = col.inner_dtype
	refs := column_typed_view(col, List_Ref)
	refs_out := column_typed_view(&out, List_Ref)
	payload: rawptr
	psize := 0
	p_err := Error.None
	switch col.inner_dtype {
	case typeid_of(bool):   payload, psize, p_err = list_unique_typed(col, refs, refs_out, bool, allocator)
	case typeid_of(i8):     payload, psize, p_err = list_unique_typed(col, refs, refs_out, i8, allocator)
	case typeid_of(i16):    payload, psize, p_err = list_unique_typed(col, refs, refs_out, i16, allocator)
	case typeid_of(i32):    payload, psize, p_err = list_unique_typed(col, refs, refs_out, i32, allocator)
	case typeid_of(i64):    payload, psize, p_err = list_unique_typed(col, refs, refs_out, i64, allocator)
	case typeid_of(u8):     payload, psize, p_err = list_unique_typed(col, refs, refs_out, u8, allocator)
	case typeid_of(u16):    payload, psize, p_err = list_unique_typed(col, refs, refs_out, u16, allocator)
	case typeid_of(u32):    payload, psize, p_err = list_unique_typed(col, refs, refs_out, u32, allocator)
	case typeid_of(u64):    payload, psize, p_err = list_unique_typed(col, refs, refs_out, u64, allocator)
	case typeid_of(int):    payload, psize, p_err = list_unique_typed(col, refs, refs_out, int, allocator)
	case typeid_of(uint):   payload, psize, p_err = list_unique_typed(col, refs, refs_out, uint, allocator)
	case typeid_of(f32):    payload, psize, p_err = list_unique_typed(col, refs, refs_out, f32, allocator)
	case typeid_of(f64):    payload, psize, p_err = list_unique_typed(col, refs, refs_out, f64, allocator)
	case typeid_of(string): payload, psize, p_err = list_unique_typed(col, refs, refs_out, string, allocator)
	case typeid_of(Date):     payload, psize, p_err = list_unique_typed(col, refs, refs_out, Date, allocator)
	case typeid_of(Datetime): payload, psize, p_err = list_unique_typed(col, refs, refs_out, Datetime, allocator)
	case typeid_of(Time):     payload, psize, p_err = list_unique_typed(col, refs, refs_out, Time, allocator)
	case typeid_of(Duration): payload, psize, p_err = list_unique_typed(col, refs, refs_out, Duration, allocator)
	case:
		column_destroy(&out)
		return {}, .Unsupported_Operation
	}
	if p_err != .None {
		column_destroy(&out)
		return {}, p_err
	}
	out.payload = payload
	out.payload_size = psize
	return out, .None
}

// list_unique_typed builds the deduplicated payload; valid elements are
// written in first-seen order so the resulting payload needs no inner_valid.
@(private)
list_unique_typed :: proc(col: ^Column, refs: []List_Ref, refs_out: []List_Ref, $T: typeid, allocator: mem.Allocator) -> (payload: rawptr, payload_size: int, err: Error) {
	buf: [dynamic]T
	defer delete(buf)
	src := list_inner_view(col, T)
	for r in 0 ..< col.count {
		if !column_is_valid(col, r) {
			continue // refs_out[r] stays zeroed
		}
		refs_out[r].off = len(buf)
		for e in refs[r].off ..< refs[r].off + refs[r].len {
			if !inner_is_valid(col, e) {
				continue
			}
			v := src[e]
			dup := false
			for j in refs_out[r].off ..< len(buf) {
				if buf[j] == v {
					dup = true
					break
				}
			}
			if !dup {
				append(&buf, v)
			}
		}
		refs_out[r].len = len(buf) - refs_out[r].off
	}
	if len(buf) == 0 {
		return nil, 0, .None
	}
	blob, a_err := mem.alloc(len(buf) * size_of(T), align_of(T), allocator)
	if a_err != .None || blob == nil {
		return nil, 0, .Allocator_Failure
	}
	mem.copy(blob, raw_data(buf), len(buf) * size_of(T))
	return blob, len(buf) * size_of(T), .None
}

// list_to_struct returns a DataFrame with one column per element position,
// named field_0..field_k-1 where k is the longest valid list. Rows with fewer
// elements (and NULL rows) get NULL in the missing positions.
list_to_struct :: proc(col: ^Column, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	if col.dtype != typeid_of(List_Ref) {
		return {}, .Invalid_Argument
	}
	refs := column_typed_view(col, List_Ref)
	k := 0
	for r in 0 ..< col.count {
		if column_is_valid(col, r) && refs[r].len > k {
			k = refs[r].len
		}
	}
	out = dataframe_create(allocator)
	inner_size, inner_align, lok := type_layout(col.inner_dtype)
	if !lok {
		dataframe_destroy(&out)
		return {}, .Unsupported_Operation
	}
	for f in 0 ..< k {
		c, cerr := column_alloc(allocator, fmt.tprintf("field_%d", f, allocator), col.inner_dtype, inner_size, inner_align, col.count)
		if cerr != .None {
			dataframe_destroy(&out)
			return {}, cerr
		}
		c.valid = bm_make(col.count, false, allocator)
		if c.valid == nil && col.count != 0 {
			column_destroy(&c)
			dataframe_destroy(&out)
			return {}, .Allocator_Failure
		}
		if a_err := dataframe_add_column(&out, &c); a_err != .None {
			column_destroy(&c)
			dataframe_destroy(&out)
			return {}, .Allocator_Failure
		}
	}
	switch col.inner_dtype {
	case typeid_of(bool):   list_to_struct_fill(col, refs, bool, &out)
	case typeid_of(i8):     list_to_struct_fill(col, refs, i8, &out)
	case typeid_of(i16):    list_to_struct_fill(col, refs, i16, &out)
	case typeid_of(i32):    list_to_struct_fill(col, refs, i32, &out)
	case typeid_of(i64):    list_to_struct_fill(col, refs, i64, &out)
	case typeid_of(u8):     list_to_struct_fill(col, refs, u8, &out)
	case typeid_of(u16):    list_to_struct_fill(col, refs, u16, &out)
	case typeid_of(u32):    list_to_struct_fill(col, refs, u32, &out)
	case typeid_of(u64):    list_to_struct_fill(col, refs, u64, &out)
	case typeid_of(int):    list_to_struct_fill(col, refs, int, &out)
	case typeid_of(uint):   list_to_struct_fill(col, refs, uint, &out)
	case typeid_of(f32):    list_to_struct_fill(col, refs, f32, &out)
	case typeid_of(f64):    list_to_struct_fill(col, refs, f64, &out)
	case typeid_of(string): list_to_struct_fill(col, refs, string, &out)
	case typeid_of(Date):     list_to_struct_fill(col, refs, Date, &out)
	case typeid_of(Datetime): list_to_struct_fill(col, refs, Datetime, &out)
	case typeid_of(Time):     list_to_struct_fill(col, refs, Time, &out)
	case typeid_of(Duration): list_to_struct_fill(col, refs, Duration, &out)
	case:
		dataframe_destroy(&out)
		return {}, .Unsupported_Operation
	}
	return out, .None
}

// list_to_struct_fill copies element `field` of every row into output column f.
@(private)
list_to_struct_fill :: proc(col: ^Column, refs: []List_Ref, $T: typeid, out: ^DataFrame) {
	src := list_inner_view(col, T)
	for f in 0 ..< out.columns.count {
		dst := cs_typed_view(&out.columns, f, T)
		for r in 0 ..< col.count {
			if !column_is_valid(col, r) || f >= refs[r].len {
				continue
			}
			e := refs[r].off + f
			if inner_is_valid(col, e) {
				dst[r] = src[e]
				bm_set(out.columns.valids[f], r, true)
			}
		}
	}
}
