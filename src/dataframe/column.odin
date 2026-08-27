package dataframe

import "core:mem"
import "base:runtime"

// Column is an owned, typed column buffer plus a packed validity bitmap
// (DESIGN.md §2.2). It stores a type-tagged raw element buffer and supports
// *any* Odin type, including arbitrary user structs.
//
// Ownership:
//   - A Column owns `data` (its element buffer), `valid` (optional packed
//     validity bitmap), and a copy of `name`. All were allocated with
//     `col.alloc`.
//   - The value *contents* of a `[]string` column are borrowed, not copied:
//     only the string headers are stored. String data must outlive the column
//     (documented limitation; deep string ownership is a later milestone).
//   - Destroy with column_destroy, which releases data, valid, and name.
//   - Copy with column_copy (deep, byte-for-byte).
//
// NULL semantics:
//   - `valid == nil` means every row is valid (the common, allocation-free
//     case).
//   - bit i of valid[i>>6] == 0 means row i is NULL. NULL is distinct from
//     zero, NaN, and the empty string (principle 7).
//   - Accessors return `valid=false` for NULL rows without an error.
//
// Type safety:
//   - The strict runtime check `col.dtype == typeid_of(T)` in every typed
//     accessor means there is no silent conversion (principle 6).
Column :: struct {
	name:      string,
	dtype:     typeid, // logical + physical type of the elements
	elem_size: int,    // size_of(dtype)
	align:     int,    // align_of(dtype)
	count:     int,    // number of rows
	data:      rawptr, // element buffer; nil when count == 0
	valid:     []u64,  // packed bitmap; nil == all rows valid
	alloc:     mem.Allocator,
	// payload is an optional owned allocation backing string contents (used by
	// string-producing expressions such as concat_str). Freed by
	// column_destroy and deep-copied by column_copy; nil when unused.
	payload:      rawptr,
	payload_size: int,
	// categories is an owned category table for categorical/enum columns
	// (dtype i64 codes). nil for non-categorical columns. The strings and the
	// slice are owned: freed by column_destroy, deep-copied by column_copy and
	// gather_rows_core. categorical_kind tags the table's semantics.
	categories:       []string,
	categorical_kind: Categorical_Kind,
	// List columns (dtype List_Ref, DESIGN.md §18.1) carry the inner element
	// type and the per-element validity array here; inner_dtype is nil for
	// every other column. The inner element data lives in `payload` (for
	// string elements their bytes follow the headers in the same blob, and
	// the headers point into it), so the wholesale payload copies in
	// column_copy and gather_rows_core keep them valid.
	inner_dtype: typeid,
	inner_valid: []u64,
}

// clone_name allocates an owned copy of name.
@(private)
clone_name :: proc(allocator: mem.Allocator, name: string) -> (string, Error) {
	out := make([]byte, len(name), allocator)
	if out == nil && len(name) != 0 {
		return "", .Allocator_Failure
	}
	copy(out, transmute([]byte)name)
	return string(out), .None
}

// column_from builds a Column of any element type T from values. The element
// buffer is copied; for string columns the string contents are borrowed.
column_from :: proc(name: string, values: []$T, allocator := context.allocator) -> (Column, Error) {
	return column_from_with_valid(name, values, nil, allocator)
}

// column_from_with_valid is column_from with an explicit validity array.
// `valid` may be nil (every row valid); otherwise len(valid) == len(values).
// The []bool is converted to a packed []u64 bitmap for storage.
column_from_with_valid :: proc(name: string, values: []$T, valid: []bool, allocator := context.allocator) -> (out: Column, err: Error) {
	if valid != nil && len(valid) != len(values) {
		return {}, .Length_Mismatch
	}

	name_copy := clone_name(allocator, name) or_return

	out = Column {
		name      = name_copy,
		dtype     = typeid_of(T),
		elem_size = size_of(T),
		align     = align_of(T),
		count     = len(values),
		alloc     = allocator,
	}

	if len(values) != 0 {
		size := len(values) * size_of(T)
		data, a_err := mem.alloc(size, align_of(T), allocator)
		if a_err != .None || data == nil {
			delete_string(out.name, out.alloc)
			return {}, .Allocator_Failure
		}
		mem.copy(data, raw_data(values), size)
		out.data = data
	}

	out.valid = bm_from_bools(valid, allocator)
	if valid != nil && out.valid == nil {
		if out.data != nil {
			mem.free_with_size(out.data, out.elem_size * out.count, out.alloc)
		}
		delete_string(out.name, out.alloc)
		return {}, .Allocator_Failure
	}

	return out, .None
}

// clone_slice allocates an independent copy of src (generic).
@(private)
clone_slice :: proc(allocator: mem.Allocator, src: []$T) -> ([]T, Error) {
	out := make([]T, len(src), allocator)
	if out == nil && len(src) != 0 {
		return nil, .Allocator_Failure
	}
	copy(out, src)
	return out, .None
}

// clone_strings_owned deep-copies a slice of strings: each string's bytes and
// the slice itself are owned by the result.
@(private)
clone_strings_owned :: proc(allocator: mem.Allocator, src: []string) -> ([]string, Error) {
	out := make([]string, len(src), allocator)
	if out == nil && len(src) != 0 {
		return nil, .Allocator_Failure
	}
	for i in 0 ..< len(src) {
		owned, o_err := clone_name(allocator, src[i])
		if o_err != .None {
			for j in 0 ..< i {
				delete_string(out[j], allocator)
			}
			delete(out, allocator)
			return nil, o_err
		}
		out[i] = owned
	}
	return out, .None
}

// --- packed validity bitmap helpers -------------------------------------------
// valid: []u64 stores one bit per row, bit i of word i>>6 (little-endian).
// nil means every row is valid — the zero-allocation fast path.

bm_words :: proc(n: int) -> int {
	return (n + 63) / 64
}

bm_get :: proc(valid: []u64, i: int) -> bool {
	return (valid[i >> 6] & (u64(1) << uint(i & 63))) != 0
}

bm_set :: proc(valid: []u64, i: int, v: bool) {
	if v {
		valid[i >> 6] |= u64(1) << uint(i & 63)
	} else {
		valid[i >> 6] &= ~(u64(1) << uint(i & 63))
	}
}

bm_init_all :: proc(valid: []u64) {
	for &w in valid {
		w = ~u64(0)
	}
}

// bm_make allocates a bitmap of n bits with all bits set to `init`.
@(private)
bm_make :: proc(n: int, init: bool, allocator: mem.Allocator) -> []u64 {
	w := bm_words(n)
	if w == 0 {
		return nil
	}
	bits := make([]u64, w, allocator)
	if bits == nil {
		return nil
	}
	if init {
		bm_init_all(bits)
	}
	return bits
}

// bm_count_false returns the number of zero bits in valid[0..n-1].
@(private)
bm_count_false :: proc(valid: []u64, n: int) -> int {
	return parallel_bm_count_false(valid, n)
}

// bm_from_bools converts a []bool validity array into a packed []u64 bitmap.
// Returns nil when src is nil (all valid).
@(private)
bm_from_bools :: proc(src: []bool, allocator: mem.Allocator) -> []u64 {
	return parallel_bm_from_bools(src, allocator)
}

// column_destroy releases the element buffer, validity array, owned name, and
// payload. For categorical columns the owned category table is released too.
// After destroy the struct is zeroed and must not be used.
column_destroy :: proc(col: ^Column) {
	if col.data != nil {
		mem.free_with_size(col.data, col.elem_size * col.count, col.alloc)
	}
	if col.valid != nil {
		delete(col.valid, col.alloc)
	}
	if col.payload != nil {
		mem.free_with_size(col.payload, col.payload_size, col.alloc)
	}
	if col.categories != nil {
		for s in col.categories {
			delete_string(s, col.alloc)
		}
		delete(col.categories, col.alloc)
	}
	if col.inner_valid != nil {
		delete(col.inner_valid, col.alloc)
	}
	if col.name != "" {
		delete_string(col.name, col.alloc)
	}
	col^ = {}
}

// column_name returns the column name.
column_name :: proc(col: ^Column) -> string {
	return col.name
}

// column_rename replaces the column name with an owned copy of name. The
// empty name is .Column_Name_Empty; on allocation failure the column is
// unchanged.
column_rename :: proc(col: ^Column, name: string) -> Error {
	if name == "" {
		return .Column_Name_Empty
	}
	copy := clone_name(col.alloc, name) or_return
	if col.name != "" {
		delete_string(col.name, col.alloc)
	}
	col.name = copy
	return .None
}

// column_dtype returns the typeid of the column's elements.
column_dtype :: proc(col: ^Column) -> typeid {
	return col.dtype
}

// column_elem_size returns the size in bytes of one element.
column_elem_size :: proc(col: ^Column) -> int {
	return col.elem_size
}

// column_len returns the number of rows in the column.
column_len :: proc(col: ^Column) -> int {
	return col.count
}

// column_data returns the raw element buffer, or nil for an empty column.
// The returned pointer is owned by the column and must not be freed by the
// caller.
column_data :: proc(col: ^Column) -> rawptr {
	return col.data
}

// column_valid returns the packed validity bitmap, or nil when every row is
// valid. The returned slice is owned by the column and must not be freed by the
// caller.
column_valid :: proc(col: ^Column) -> []u64 {
	return col.valid
}

// column_is_all_valid reports whether the column has no NULL rows.
column_is_all_valid :: proc(col: ^Column) -> bool {
	return col.valid == nil
}

// column_is_valid reports whether row i is non-NULL. The caller is
// responsible for bounds checking against column_len.
column_is_valid :: proc(col: ^Column, i: int) -> bool {
	if col.valid == nil {
		return true
	}
	return bm_get(col.valid, i)
}

// column_typed_view returns the element buffer as a typed slice. Only valid
// when col.dtype == typeid_of(T) and col.data != nil.
@(private)
column_typed_view :: proc(col: ^Column, $T: typeid) -> []T {
	return transmute([]T)(runtime.Raw_Slice{data = col.data, len = col.count})
}

// column_set_valid marks row i valid/invalid, allocating the validity bitmap
// on first use. Returns an error for out-of-range i or allocation failure.
column_set_valid :: proc(col: ^Column, i: int, valid: bool) -> Error {
	if i < 0 || i >= col.count {
		return .Out_Of_Bounds
	}
	if col.valid == nil {
		if valid {
			return .None
		}
		bits := bm_make(col.count, true, col.alloc)
		if bits == nil && col.count != 0 {
			return .Allocator_Failure
		}
		col.valid = bits
	}
	bm_set(col.valid, i, valid)
	return .None
}

// column_set_null marks row i NULL.
column_set_null :: proc(col: ^Column, i: int) -> Error {
	return column_set_valid(col, i, false)
}

// column_set_all_valid clears the validity array so every row is valid.
column_set_all_valid :: proc(col: ^Column) -> Error {
	if col.valid != nil {
		delete(col.valid, col.alloc)
		col.valid = nil
	}
	return .None
}

// column_copy_validity marks out rows invalid wherever src rows are invalid.
// Precondition: out and src have the same row count (enforced by callers).
@(private)
column_copy_validity :: proc(out, src: ^Column) {
	if src.valid == nil {
		return
	}
	for i in 0 ..< src.count {
		if !bm_get(src.valid, i) {
			_ = column_set_valid(out, i, false)
		}
	}
}

// column_clear_validity sets every row of col to all_valid (clearing the
// validity array when all_valid is true).
@(private)
column_clear_validity :: proc(col: ^Column, all_valid: bool) {
	if all_valid {
		_ = column_set_all_valid(col)
	}
}

// --- typed value access ----------------------------------------------------
//
// Getter convention: (value, valid, err). A NULL row returns valid=false with
// no error; a real problem (out of range, wrong type) returns err. T is the
// requested element type and must be passed explicitly, e.g.
//     v, valid, err := column_get(&col, 0, i64)

column_get :: proc(col: ^Column, i: int, $T: typeid) -> (value: T, valid: bool, err: Error) {
	if col.dtype != typeid_of(T) {
		return {}, false, .Type_Mismatch
	}
	if i < 0 || i >= col.count {
		return {}, false, .Out_Of_Bounds
	}
	if col.valid != nil && !bm_get(col.valid, i) {
		return {}, false, .None
	}
	return column_typed_view(col, T)[i], true, .None
}

// --- typed value set -------------------------------------------------------
//
// Setters write the value and mark the row valid. T is inferred from value.
// Type checking is strict; there is no silent conversion (principle 6).

column_set :: proc(col: ^Column, i: int, value: $T) -> Error {
	if col.dtype != typeid_of(T) {
		return .Type_Mismatch
	}
	if i < 0 || i >= col.count {
		return .Out_Of_Bounds
	}
	column_typed_view(col, T)[i] = value
	return column_set_valid(col, i, true)
}

// column_copy returns a deep, independent copy of col allocated with
// allocator. The copy owns its name, element buffer, validity array, and
// payload (string contents are cloned and element headers re-pointed into the
// cloned payload).
column_copy :: proc(col: ^Column, allocator := context.allocator) -> (out: Column, err: Error) {
	out = column_alloc(allocator, col.name, col.dtype, col.elem_size, col.align, col.count) or_return
	if col.data != nil {
		mem.copy(out.data, col.data, col.elem_size * col.count)
	}
	if col.payload != nil {
		blob, a_err := mem.alloc(col.payload_size, 1, allocator)
		if a_err != .None || blob == nil {
			column_destroy(&out)
			return {}, .Allocator_Failure
		}
		mem.copy(blob, col.payload, col.payload_size)
		out.payload = blob
		out.payload_size = col.payload_size
		// re-point string headers from the old payload into the new one.
		if col.dtype == typeid_of(string) {
			old_base := uintptr(col.payload)
			new_base := uintptr(blob)
			ov := column_typed_view(&out, string)
			for &s in ov {
				raw := transmute(runtime.Raw_String)s
				if uintptr(raw.data) >= old_base && uintptr(raw.data) < old_base + uintptr(col.payload_size) {
					new_raw := runtime.Raw_String {
						data = (^u8)(new_base + (uintptr(raw.data) - old_base)),
						len  = raw.len,
					}
					s = transmute(string)new_raw
				}
			}
		}
		// List columns keep the whole element buffer (offsets stay valid);
		// string inner headers are re-pointed the same way.
		if col.dtype == typeid_of(List_Ref) && col.inner_dtype == typeid_of(string) {
			old_base := uintptr(col.payload)
			new_base := uintptr(blob)
			ov := list_inner_view(&out, string)
			for &s in ov {
				raw := transmute(runtime.Raw_String)s
				if uintptr(raw.data) >= old_base && uintptr(raw.data) < old_base + uintptr(col.payload_size) {
					new_raw := runtime.Raw_String {
						data = (^u8)(new_base + (uintptr(raw.data) - old_base)),
						len  = raw.len,
					}
					s = transmute(string)new_raw
				}
			}
		}
	}
	if col.valid != nil {
		valid_copy, v_err := clone_slice(allocator, col.valid)
		if v_err != .None {
			column_destroy(&out)
			return {}, v_err
		}
		out.valid = valid_copy
	}
	if col.categories != nil {
		cat_copy, c_err := clone_strings_owned(allocator, col.categories)
		if c_err != .None {
			column_destroy(&out)
			return {}, c_err
		}
		out.categories = cat_copy
		out.categorical_kind = col.categorical_kind
	}
	if col.inner_dtype != nil {
		iv, iv_err := clone_slice(allocator, col.inner_valid)
		if iv_err != .None {
			column_destroy(&out)
			return {}, iv_err
		}
		out.inner_valid = iv
		out.inner_dtype = col.inner_dtype
	}
	return out, .None
}

// column_alloc reserves a Column of the given element type and row count
// without copying any values. The name is copied; data and valid start nil.
@(private)
column_alloc :: proc(allocator: mem.Allocator, name: string, dtype: typeid, elem_size: int, align: int, count: int) -> (col: Column, err: Error) {
	col = Column {
		name      = clone_name(allocator, name) or_return,
		dtype     = dtype,
		elem_size = elem_size,
		align     = align,
		count     = count,
		alloc     = allocator,
	}
	if count != 0 {
		data, a_err := mem.alloc(count * elem_size, align, allocator)
		if a_err != .None || data == nil {
			delete_string(col.name, col.alloc)
			return {}, .Allocator_Failure
		}
		col.data = data
	}
	return col, .None
}

// column_empty builds a zero-row Column of the given element type with the
// given name. It is the building block for schema-shaped empty DataFrames
// (dataframe_create_with_schema). Unsupported dtypes return
// .Unsupported_Operation.
column_empty :: proc(name: string, dtype: typeid, allocator := context.allocator) -> (Column, Error) {
	if name == "" {
		return {}, .Column_Name_Empty
	}
	size, align, ok := type_layout(dtype)
	if !ok {
		return {}, .Unsupported_Operation
	}
	return column_alloc(allocator, name, dtype, size, align, 0)
}

// column_clone_valid returns a deep copy of col with every row marked valid
// (NULL rows expose their underlying value slot).
column_clone_valid :: proc(col: ^Column, allocator: mem.Allocator) -> (out: Column, err: Error) {
	out = column_copy(col, allocator) or_return
	if out.valid != nil {
		delete(out.valid, out.alloc)
		out.valid = nil
	}
	return out, .None
}
