package odinarrow

import "core:mem"

// Primitive_Builder accumulates values of type T and produces an owned Array
// on finish(). T must be one of: bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64.
//
// Internal storage uses raw buffers ([^]T) rather than [dynamic]T to
// eliminate the dynamic-array descriptor overhead on every append — direct
// indexed writes replace per-call bounds-check + length-load + length-store.
Primitive_Builder :: struct($T: typeid) {
	data:       [^]T,   // raw value buffer
	cap:        int,
	bitmap:     [^]u8,  // validity bitmap (all-zero = all valid); sized lazily
	bm_cap:     int,
	null_count: int,
	length:     int,
	allocator:  mem.Allocator,
}

// Allocate a new builder with preallocated capacity.
builder_make :: proc($T: typeid, initial_cap := 64, allocator := context.allocator) -> Primitive_Builder(T) {
	cap := max(initial_cap, 1)
	bm  := bitmap_byte_count(cap)
	// Values are fully overwritten before the array is read, so the data buffer
	// is left uninitialised; the bitmap must start zeroed (lazy valid bits).
	data_bytes, _ := mem.alloc_bytes_non_zeroed(cap * size_of(T), ARROW_ALIGNMENT, allocator)
	bm_raw,     _ := mem.alloc(bm, ARROW_ALIGNMENT, allocator)
	return Primitive_Builder(T){
		data      = cast([^]T)raw_data(data_bytes),
		cap       = cap,
		bitmap    = cast([^]u8)bm_raw,
		bm_cap    = bm,
		allocator = allocator,
	}
}

// Append a non-null value.
builder_append :: #force_inline proc(b: ^Primitive_Builder($T), val: T) {
	if b.length >= b.cap { _builder_grow_data(b) }
	b.data[b.length] = val
	if b.null_count > 0 {
		byte_i := b.length >> 3
		if byte_i >= b.bm_cap { _builder_grow_bitmap(b, byte_i + 1) }
		b.bitmap[byte_i] |= 1 << u8(b.length & 7)
	}
	b.length += 1
}

// Append a null value.
builder_append_null :: #force_inline proc(b: ^Primitive_Builder($T)) {
	if b.length >= b.cap { _builder_grow_data(b) }
	byte_i := b.length >> 3
	if byte_i >= b.bm_cap { _builder_grow_bitmap(b, byte_i + 1) }
	if b.null_count == 0 {
		// First null: retroactively mark all previous elements valid.
		bitmap_set_all(b.bitmap, b.length)
	}
	// Null bit stays 0 (bitmap zeroed on alloc/grow).
	zero: T
	b.data[b.length] = zero
	b.null_count += 1
	b.length += 1
}

// Produce an immutable Array from the builder's accumulated values.
//
// For every non-bool type this is zero-copy: the builder's data and validity
// buffers are handed directly to the Array (no alloc + copy of the values),
// and the builder is left empty so it can be safely destroyed or reused (the
// next append re-grows from nil).  Bool still packs its 1-byte-per-value buffer
// into a bitmap, which requires a fresh allocation.
builder_finish :: proc(
	b:         ^Primitive_Builder($T),
	allocator := context.allocator,
) -> (arr: Array, err: mem.Allocator_Error) {
	// Validity bitmap — only materialised when there are nulls; transferred.
	bitmap_buf: Buffer
	if b.null_count > 0 {
		bitmap_buf = Buffer{
			data      = b.bitmap,
			size      = bitmap_byte_count(b.length),
			capacity  = b.bm_cap,
			allocator = b.allocator,
		}
		b.bitmap = nil
		b.bm_cap = 0
	}

	// Data buffer.
	data_buf: Buffer
	when T == bool {
		// Bool arrays are bit-packed, so the [^]bool scratch can't be reused.
		n_data_bytes := bitmap_byte_count(b.length)
		buf, buf_err := buffer_make(n_data_bytes, allocator)
		if buf_err != nil { buffer_free(&bitmap_buf); err = buf_err; return }
		for i in 0..<b.length {
			if b.data[i] { buf.data[i >> 3] |= 1 << u8(i & 7) }
		}
		data_buf = buf
	} else {
		// Transfer ownership of the value buffer directly — no copy.
		data_buf = Buffer{
			data      = cast([^]u8)b.data,
			size      = b.length * size_of(T),
			capacity  = b.cap * size_of(T),
			allocator = b.allocator,
		}
		b.data = nil
		b.cap  = 0
	}

	arr = Array{
		type       = _data_type_for(T),
		length     = b.length,
		null_count = b.null_count,
		offset     = 0,
		buffers    = {bitmap_buf, data_buf, {}},
	}
	return
}

builder_reset :: proc(b: ^Primitive_Builder($T)) {
	mem.zero(rawptr(b.bitmap), b.bm_cap)
	b.null_count = 0
	b.length     = 0
}

builder_destroy :: proc(b: ^Primitive_Builder($T)) {
	if b.data   != nil { mem.free(rawptr(b.data),   b.allocator) }
	if b.bitmap != nil { mem.free(rawptr(b.bitmap), b.allocator) }
	b^ = {}
}

// ── growth helpers ────────────────────────────────────────────────────────────

_builder_grow_data :: proc(b: ^Primitive_Builder($T)) {
	new_cap := max(b.cap * 2, 16)
	new_raw, _ := mem.resize(rawptr(b.data), b.cap * size_of(T), new_cap * size_of(T), ARROW_ALIGNMENT, b.allocator)
	b.data = cast([^]T)new_raw
	b.cap  = new_cap
}

_builder_grow_bitmap :: proc(b: ^Primitive_Builder($T), needed: int) {
	new_cap := max(b.bm_cap * 2, needed, 8)
	new_raw, _ := mem.resize(rawptr(b.bitmap), b.bm_cap, new_cap, ARROW_ALIGNMENT, b.allocator)
	if new_cap > b.bm_cap {
		mem.zero(rawptr((cast([^]u8)new_raw)[b.bm_cap:]), new_cap - b.bm_cap)
	}
	b.bitmap = cast([^]u8)new_raw
	b.bm_cap = new_cap
}

// ── type helpers ──────────────────────────────────────────────────────────────

_data_type_for :: proc($T: typeid) -> DataType {
	when T == bool { return Bool_Type{} }
	else when T == i8  { return Int8_Type{} }
	else when T == i16 { return Int16_Type{} }
	else when T == i32 { return Int32_Type{} }
	else when T == i64 { return Int64_Type{} }
	else when T == u8  { return UInt8_Type{} }
	else when T == u16 { return UInt16_Type{} }
	else when T == u32 { return UInt32_Type{} }
	else when T == u64 { return UInt64_Type{} }
	else when T == f32 { return Float32_Type{} }
	else when T == f64 { return Float64_Type{} }
	else {
		#assert(false, "Primitive_Builder: unsupported type T")
		return {}
	}
}
