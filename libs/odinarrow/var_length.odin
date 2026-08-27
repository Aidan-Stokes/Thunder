package odinarrow

import "core:mem"

// ── String_Builder ────────────────────────────────────────────────────────────
//
// Produces Arrow UTF-8 (i32-offset) arrays.
// Buffer layout: [validity_bitmap, i32_offsets, utf8_bytes]
//
// offsets always contains length+1 values; it is pre-seeded with [0].

// Internal storage uses raw buffers (like Primitive_Builder) rather than
// [dynamic] so each append is a direct indexed write + memcpy instead of going
// through the dynamic-array runtime. finish() then hands those buffers straight
// to the Array (zero-copy). `offsets` always holds length+1 entries with
// offsets[0] == 0.
String_Builder :: struct {
	data:       [^]u8,   // utf8 bytes
	data_len:   int,
	data_cap:   int,
	offsets:    [^]i32,  // length+1 valid entries; offsets[0] == 0
	off_cap:    int,     // capacity in i32 elements
	bitmap:     [^]u8,   // validity (all-zero = all valid); written lazily
	bm_cap:     int,
	null_count: int,
	length:     int,
	allocator:  mem.Allocator,
}

string_builder_make :: proc(initial_cap := 64, allocator := context.allocator) -> String_Builder {
	cap      := max(initial_cap, 1)
	data_cap := cap * 8
	off_cap  := cap + 1
	bm       := bitmap_byte_count(cap)
	// data and offsets are fully written before the array is read → leave
	// uninitialised; the validity bitmap must start zeroed.
	data_bytes, _ := mem.alloc_bytes_non_zeroed(data_cap,               ARROW_ALIGNMENT, allocator)
	off_bytes,  _ := mem.alloc_bytes_non_zeroed(off_cap * size_of(i32), ARROW_ALIGNMENT, allocator)
	bm_raw,     _ := mem.alloc(bm,                                      ARROW_ALIGNMENT, allocator)
	b := String_Builder{
		data      = raw_data(data_bytes),             data_cap = data_cap,
		offsets   = cast([^]i32)raw_data(off_bytes),  off_cap  = off_cap,
		bitmap    = cast([^]u8)bm_raw,    bm_cap   = bm,
		allocator = allocator,
	}
	b.offsets[0] = 0
	return b
}

string_builder_append :: #force_inline proc(b: ^String_Builder, s: string) {
	n := len(s)
	if b.data_len + n > b.data_cap { _sb_grow_data(b, b.data_len + n) }
	if b.length + 1 >= b.off_cap   { _sb_grow_offsets(b) }
	// Builder buffer and the input string never alias → memcpy, not memmove.
	if n > 0 { mem.copy_non_overlapping(rawptr(&b.data[b.data_len]), rawptr(raw_data(s)), n) }
	b.data_len += n
	b.offsets[b.length + 1] = i32(b.data_len)
	if b.null_count > 0 {
		byte_i := b.length >> 3
		if byte_i >= b.bm_cap { _sb_grow_bitmap(b, byte_i + 1) }
		b.bitmap[byte_i] |= 1 << u8(b.length & 7)
	}
	b.length += 1
}

string_builder_append_null :: #force_inline proc(b: ^String_Builder) {
	if b.length + 1 >= b.off_cap { _sb_grow_offsets(b) }
	byte_i := b.length >> 3
	if byte_i >= b.bm_cap { _sb_grow_bitmap(b, byte_i + 1) }
	if b.null_count == 0 { bitmap_set_all(b.bitmap, b.length) }
	// Null bit stays 0 (pre-zeroed); offset repeats the previous value.
	b.offsets[b.length + 1] = i32(b.data_len)
	b.null_count += 1
	b.length += 1
}

// Zero-copy: hands the data/offsets/validity buffers directly to the Array and
// leaves the builder empty (re-grows from nil on the next append).
string_builder_finish :: proc(b: ^String_Builder, allocator := context.allocator) -> (arr: Array, err: mem.Allocator_Error) {
	bitmap_buf: Buffer
	if b.null_count > 0 {
		bitmap_buf = Buffer{ data = b.bitmap, size = bitmap_byte_count(b.length), capacity = b.bm_cap, allocator = b.allocator }
		b.bitmap = nil; b.bm_cap = 0
	}
	off_buf := Buffer{
		data      = cast([^]u8)b.offsets,
		size      = (b.length + 1) * size_of(i32),
		capacity  = b.off_cap * size_of(i32),
		allocator = b.allocator,
	}
	b.offsets = nil; b.off_cap = 0
	data_buf := Buffer{ data = b.data, size = b.data_len, capacity = b.data_cap, allocator = b.allocator }
	b.data = nil; b.data_cap = 0

	arr = Array{
		type       = String_Type{},
		length     = b.length,
		null_count = b.null_count,
		offset     = 0,
		buffers    = {bitmap_buf, off_buf, data_buf},
	}
	_ = allocator
	return
}

string_builder_reset :: proc(b: ^String_Builder) {
	b.data_len   = 0
	b.null_count = 0
	b.length     = 0
	if b.bitmap  != nil { mem.zero(rawptr(b.bitmap), b.bm_cap) }
	if b.offsets != nil { b.offsets[0] = 0 }
}

string_builder_destroy :: proc(b: ^String_Builder) {
	if b.data    != nil { mem.free(rawptr(b.data),    b.allocator) }
	if b.offsets != nil { mem.free(rawptr(b.offsets), b.allocator) }
	if b.bitmap  != nil { mem.free(rawptr(b.bitmap),  b.allocator) }
	b^ = {}
}

// ── String_Builder growth helpers ──────────────────────────────────────────────

_sb_grow_data :: proc(b: ^String_Builder, needed: int) {
	new_cap := max(b.data_cap * 2, needed, 16)
	new_raw, _ := mem.resize(rawptr(b.data), b.data_cap, new_cap, ARROW_ALIGNMENT, b.allocator)
	b.data     = cast([^]u8)new_raw
	b.data_cap = new_cap
}

_sb_grow_offsets :: proc(b: ^String_Builder) {
	new_cap := max(b.off_cap * 2, 16)  // elements
	new_raw, _ := mem.resize(rawptr(b.offsets), b.off_cap * size_of(i32), new_cap * size_of(i32), ARROW_ALIGNMENT, b.allocator)
	if b.off_cap == 0 { (cast([^]i32)new_raw)[0] = 0 }  // re-establish offsets[0] after a consuming finish
	b.offsets = cast([^]i32)new_raw
	b.off_cap = new_cap
}

_sb_grow_bitmap :: proc(b: ^String_Builder, needed: int) {
	new_cap := max(b.bm_cap * 2, needed, 8)
	new_raw, _ := mem.resize(rawptr(b.bitmap), b.bm_cap, new_cap, ARROW_ALIGNMENT, b.allocator)
	if new_cap > b.bm_cap {
		mem.zero(rawptr((cast([^]u8)new_raw)[b.bm_cap:]), new_cap - b.bm_cap)
	}
	b.bitmap = cast([^]u8)new_raw
	b.bm_cap = new_cap
}

// ── Binary_Builder ────────────────────────────────────────────────────────────
// Identical to String_Builder except produces Binary_Type arrays (no UTF-8 assertion).

Binary_Builder :: struct {
	offsets:    [dynamic]i32,
	data:       [dynamic]u8,
	bitmap:     [dynamic]u8,
	null_count: int,
	length:     int,
}

binary_builder_make :: proc(initial_cap := 64, allocator := context.allocator) -> Binary_Builder {
	b := Binary_Builder{
		offsets = make([dynamic]i32, 0, initial_cap + 1, allocator),
		data    = make([dynamic]u8,  0, initial_cap * 8, allocator),
		bitmap  = make([dynamic]u8,  0, (initial_cap + 7) / 8, allocator),
	}
	append(&b.offsets, i32(0))
	return b
}

binary_builder_append :: proc(b: ^Binary_Builder, bytes: []u8) {
	i      := b.length
	byte_i := i >> 3
	if byte_i >= len(b.bitmap) {
		append(&b.bitmap, u8(0))
	}
	b.bitmap[byte_i] |= 1 << u8(i & 7)
	append(&b.data, ..bytes)
	append(&b.offsets, i32(len(b.data)))
	b.length += 1
}

binary_builder_append_null :: proc(b: ^Binary_Builder) {
	i      := b.length
	byte_i := i >> 3
	if byte_i >= len(b.bitmap) {
		append(&b.bitmap, u8(0))
	}
	append(&b.offsets, b.offsets[len(b.offsets) - 1])
	b.null_count += 1
	b.length += 1
}

binary_builder_finish :: proc(b: ^Binary_Builder, allocator := context.allocator) -> (arr: Array, err: mem.Allocator_Error) {
	return _var_finish(b.offsets[:], b.data[:], b.bitmap[:], b.null_count, b.length, Binary_Type{}, allocator)
}

binary_builder_destroy :: proc(b: ^Binary_Builder) {
	delete(b.offsets)
	delete(b.data)
	delete(b.bitmap)
	b^ = {}
}

// ── Large_String_Builder ──────────────────────────────────────────────────────
//
// Produces Arrow LargeUtf8 (i64-offset) arrays — identical to String_Builder
// except the offsets buffer holds i64 values, lifting the 2 GiB total-bytes
// limit that i32 offsets impose.
// Buffer layout: [validity_bitmap, i64_offsets, utf8_bytes]

Large_String_Builder :: struct {
	offsets:    [dynamic]i64,
	data:       [dynamic]u8,
	bitmap:     [dynamic]u8,
	null_count: int,
	length:     int,
}

large_string_builder_make :: proc(initial_cap := 64, allocator := context.allocator) -> Large_String_Builder {
	bm_bytes := bitmap_byte_count(max(initial_cap, 1))
	b := Large_String_Builder{
		offsets = make([dynamic]i64, 0, initial_cap + 1, allocator),
		data    = make([dynamic]u8,  0, initial_cap * 8, allocator),
		bitmap  = make([dynamic]u8,  bm_bytes, bm_bytes, allocator),
	}
	append(&b.offsets, i64(0))
	return b
}

large_string_builder_append :: proc(b: ^Large_String_Builder, s: string) {
	i := b.length
	if b.null_count > 0 {
		byte_i := i >> 3
		if byte_i >= len(b.bitmap) { resize(&b.bitmap, byte_i + 1) }
		b.bitmap[byte_i] |= 1 << u8(i & 7)
	}
	append(&b.data, ..transmute([]u8)s)
	append(&b.offsets, i64(len(b.data)))
	b.length += 1
}

large_string_builder_append_null :: proc(b: ^Large_String_Builder) {
	i      := b.length
	byte_i := i >> 3
	if byte_i >= len(b.bitmap) { resize(&b.bitmap, byte_i + 1) }
	if b.null_count == 0 { bitmap_set_all(raw_data(b.bitmap[:]), i) }
	// Null bit stays 0 (pre-zeroed).
	append(&b.offsets, b.offsets[len(b.offsets) - 1])
	b.null_count += 1
	b.length += 1
}

large_string_builder_finish :: proc(b: ^Large_String_Builder, allocator := context.allocator) -> (arr: Array, err: mem.Allocator_Error) {
	return _var_finish_i64(b.offsets[:], b.data[:], b.bitmap[:], b.null_count, b.length, Large_String_Type{}, allocator)
}

large_string_builder_reset :: proc(b: ^Large_String_Builder) {
	clear(&b.offsets)
	clear(&b.data)
	clear(&b.bitmap)
	append(&b.offsets, i64(0))
	b.null_count = 0
	b.length     = 0
}

large_string_builder_destroy :: proc(b: ^Large_String_Builder) {
	delete(b.offsets)
	delete(b.data)
	delete(b.bitmap)
	b^ = {}
}

// ── Large_Binary_Builder ──────────────────────────────────────────────────────
// Identical to Large_String_Builder but produces Large_Binary_Type arrays (no
// UTF-8 assumption).

Large_Binary_Builder :: struct {
	offsets:    [dynamic]i64,
	data:       [dynamic]u8,
	bitmap:     [dynamic]u8,
	null_count: int,
	length:     int,
}

large_binary_builder_make :: proc(initial_cap := 64, allocator := context.allocator) -> Large_Binary_Builder {
	bm_bytes := bitmap_byte_count(max(initial_cap, 1))
	b := Large_Binary_Builder{
		offsets = make([dynamic]i64, 0, initial_cap + 1, allocator),
		data    = make([dynamic]u8,  0, initial_cap * 8, allocator),
		bitmap  = make([dynamic]u8,  bm_bytes, bm_bytes, allocator),
	}
	append(&b.offsets, i64(0))
	return b
}

large_binary_builder_append :: proc(b: ^Large_Binary_Builder, bytes: []u8) {
	i := b.length
	if b.null_count > 0 {
		byte_i := i >> 3
		if byte_i >= len(b.bitmap) { resize(&b.bitmap, byte_i + 1) }
		b.bitmap[byte_i] |= 1 << u8(i & 7)
	}
	append(&b.data, ..bytes)
	append(&b.offsets, i64(len(b.data)))
	b.length += 1
}

large_binary_builder_append_null :: proc(b: ^Large_Binary_Builder) {
	i      := b.length
	byte_i := i >> 3
	if byte_i >= len(b.bitmap) { resize(&b.bitmap, byte_i + 1) }
	if b.null_count == 0 { bitmap_set_all(raw_data(b.bitmap[:]), i) }
	append(&b.offsets, b.offsets[len(b.offsets) - 1])
	b.null_count += 1
	b.length += 1
}

large_binary_builder_finish :: proc(b: ^Large_Binary_Builder, allocator := context.allocator) -> (arr: Array, err: mem.Allocator_Error) {
	return _var_finish_i64(b.offsets[:], b.data[:], b.bitmap[:], b.null_count, b.length, Large_Binary_Type{}, allocator)
}

large_binary_builder_destroy :: proc(b: ^Large_Binary_Builder) {
	delete(b.offsets)
	delete(b.data)
	delete(b.bitmap)
	b^ = {}
}

// ── Array accessors ───────────────────────────────────────────────────────────

// Zero-copy string view. Valid only while the Array is alive. Does NOT copy.
array_get_string :: proc(arr: ^Array, i: int) -> string {
	assert(i >= 0 && i < arr.length, "array_get_string: out of bounds")
	offsets := cast([^]i32)arr.buffers[1].data
	idx     := arr.offset + i
	start   := int(offsets[idx])
	end     := int(offsets[idx + 1])
	if start == end do return ""
	return transmute(string)arr.buffers[2].data[start:end]
}

// Zero-copy binary slice. Valid only while the Array is alive. Does NOT copy.
array_get_binary :: proc(arr: ^Array, i: int) -> []u8 {
	assert(i >= 0 && i < arr.length, "array_get_binary: out of bounds")
	offsets := cast([^]i32)arr.buffers[1].data
	idx     := arr.offset + i
	start   := int(offsets[idx])
	end     := int(offsets[idx + 1])
	return arr.buffers[2].data[start:end]
}

// Zero-copy string view into a LargeUtf8 (i64-offset) array. Does NOT copy.
array_get_large_string :: proc(arr: ^Array, i: int) -> string {
	assert(i >= 0 && i < arr.length, "array_get_large_string: out of bounds")
	offsets := cast([^]i64)arr.buffers[1].data
	idx     := arr.offset + i
	start   := int(offsets[idx])
	end     := int(offsets[idx + 1])
	if start == end do return ""
	return transmute(string)arr.buffers[2].data[start:end]
}

// Zero-copy binary slice from a LargeBinary (i64-offset) array. Does NOT copy.
array_get_large_binary :: proc(arr: ^Array, i: int) -> []u8 {
	assert(i >= 0 && i < arr.length, "array_get_large_binary: out of bounds")
	offsets := cast([^]i64)arr.buffers[1].data
	idx     := arr.offset + i
	start   := int(offsets[idx])
	end     := int(offsets[idx + 1])
	return arr.buffers[2].data[start:end]
}

// ── shared finish impl ────────────────────────────────────────────────────────

_var_finish :: proc(
	offsets:    []i32,
	data_bytes: []u8,
	bitmap:     []u8,
	null_count: int,
	length:     int,
	data_type:  DataType,
	allocator:  mem.Allocator,
) -> (arr: Array, err: mem.Allocator_Error) {
	// Validity bitmap
	bitmap_buf: Buffer
	if null_count > 0 {
		n_bm := bitmap_byte_count(length)
		bitmap_buf = buffer_make(n_bm, allocator) or_return
		src_n := (length + 7) / 8
		if src_n > 0 && len(bitmap) > 0 {
			copy_n := min(src_n, len(bitmap))
			mem.copy(rawptr(bitmap_buf.data), rawptr(raw_data(bitmap)), copy_n)
		}
	}

	// Offsets buffer: (length + 1) i32 values
	off_bytes := (length + 1) * size_of(i32)
	off_buf, err2 := buffer_make(off_bytes, allocator)
	if err2 != nil {
		buffer_free(&bitmap_buf)
		err = err2
		return
	}
	if len(offsets) > 0 {
		mem.copy(rawptr(off_buf.data), rawptr(raw_data(offsets)), len(offsets) * size_of(i32))
	}

	// Data buffer
	data_buf, err3 := buffer_make(len(data_bytes), allocator)
	if err3 != nil {
		buffer_free(&bitmap_buf)
		buffer_free(&off_buf)
		err = err3
		return
	}
	if len(data_bytes) > 0 {
		mem.copy(rawptr(data_buf.data), rawptr(raw_data(data_bytes)), len(data_bytes))
	}

	arr = Array{
		type       = data_type,
		length     = length,
		null_count = null_count,
		offset     = 0,
		buffers    = {bitmap_buf, off_buf, data_buf},
	}
	return
}

// Shared finish for the i64-offset (Large*) variants. Mirrors _var_finish but
// the offsets buffer holds (length + 1) i64 values.
_var_finish_i64 :: proc(
	offsets:    []i64,
	data_bytes: []u8,
	bitmap:     []u8,
	null_count: int,
	length:     int,
	data_type:  DataType,
	allocator:  mem.Allocator,
) -> (arr: Array, err: mem.Allocator_Error) {
	// Validity bitmap
	bitmap_buf: Buffer
	if null_count > 0 {
		n_bm := bitmap_byte_count(length)
		bitmap_buf = buffer_make(n_bm, allocator) or_return
		src_n := (length + 7) / 8
		if src_n > 0 && len(bitmap) > 0 {
			copy_n := min(src_n, len(bitmap))
			mem.copy(rawptr(bitmap_buf.data), rawptr(raw_data(bitmap)), copy_n)
		}
	}

	// Offsets buffer: (length + 1) i64 values
	off_bytes := (length + 1) * size_of(i64)
	off_buf, err2 := buffer_make(off_bytes, allocator)
	if err2 != nil {
		buffer_free(&bitmap_buf)
		err = err2
		return
	}
	if len(offsets) > 0 {
		mem.copy(rawptr(off_buf.data), rawptr(raw_data(offsets)), len(offsets) * size_of(i64))
	}

	// Data buffer
	data_buf, err3 := buffer_make(len(data_bytes), allocator)
	if err3 != nil {
		buffer_free(&bitmap_buf)
		buffer_free(&off_buf)
		err = err3
		return
	}
	if len(data_bytes) > 0 {
		mem.copy(rawptr(data_buf.data), rawptr(raw_data(data_bytes)), len(data_bytes))
	}

	arr = Array{
		type       = data_type,
		length     = length,
		null_count = null_count,
		offset     = 0,
		buffers    = {bitmap_buf, off_buf, data_buf},
	}
	return
}
