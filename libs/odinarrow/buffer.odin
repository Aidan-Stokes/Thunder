package odinarrow

import "core:mem"

ARROW_ALIGNMENT :: 64 // AVX-512 width; Arrow spec minimum

Buffer :: struct {
	data:      [^]u8,
	size:      int,
	capacity:  int,
	allocator: mem.Allocator,
}

// Round n up to the next multiple of alignment (alignment must be power of 2)
align_up :: #force_inline proc "contextless" (n, alignment: int) -> int {
	return (n + alignment - 1) & ~(alignment - 1)
}

// Allocate a zero-initialized, ARROW_ALIGNMENT-aligned buffer of `size` bytes.
// Capacity is always a multiple of ARROW_ALIGNMENT so bitmaps can be cast to u64[].
buffer_make :: proc(size: int, allocator := context.allocator) -> (buf: Buffer, err: mem.Allocator_Error) {
	capacity := max(align_up(size, ARROW_ALIGNMENT), ARROW_ALIGNMENT)
	raw := mem.alloc(capacity, ARROW_ALIGNMENT, allocator) or_return
	mem.zero(raw, capacity)
	buf = Buffer{
		data      = cast([^]u8)raw,
		size      = size,
		capacity  = capacity,
		allocator = allocator,
	}
	return
}

// Like buffer_make but leaves the bytes uninitialized. Only safe when the
// caller fully overwrites every logical element (e.g. element-wise kernels that
// write dst[i] for all i). Avoids a redundant zeroing pass over large buffers.
buffer_make_uninit :: proc(size: int, allocator := context.allocator) -> (buf: Buffer, err: mem.Allocator_Error) {
	capacity := max(align_up(size, ARROW_ALIGNMENT), ARROW_ALIGNMENT)
	raw := mem.alloc_bytes_non_zeroed(capacity, ARROW_ALIGNMENT, allocator) or_return
	buf = Buffer{
		data      = raw_data(raw),
		size      = size,
		capacity  = capacity,
		allocator = allocator,
	}
	return
}

// Release memory. Safe to call on zero-value Buffer and on slice-view Buffers
// (allocator.procedure == nil means the Buffer does not own its memory).
buffer_free :: proc(buf: ^Buffer) {
	if buf.data != nil && buf.allocator.procedure != nil {
		mem.free(rawptr(buf.data), buf.allocator)
	}
	buf^ = {}
}

// Zero-copy view into an existing buffer.
// The returned Buffer does NOT own its memory (allocator is zeroed); never free it.
buffer_slice :: proc(buf: ^Buffer, from, to: int) -> Buffer {
	assert(from >= 0 && to <= buf.size && from <= to, "buffer_slice: bounds out of range")
	return Buffer{
		data     = buf.data[from:],
		size     = to - from,
		capacity = buf.capacity - from,
	}
}

// Deep copy — caller owns the new buffer.
buffer_copy :: proc(src: ^Buffer, allocator := context.allocator) -> (buf: Buffer, err: mem.Allocator_Error) {
	buf = buffer_make(src.size, allocator) or_return
	mem.copy(rawptr(buf.data), rawptr(src.data), src.size)
	return
}

// Grow or shrink the buffer.
// Growing zeroes new bytes; shrinking never reallocates.
buffer_resize :: proc(buf: ^Buffer, new_size: int) -> mem.Allocator_Error {
	new_cap := align_up(new_size, ARROW_ALIGNMENT)
	if new_cap <= buf.capacity {
		if new_size > buf.size {
			mem.zero(rawptr(buf.data[buf.size:]), new_size - buf.size)
		}
		buf.size = new_size
		return nil
	}
	old_cap := buf.capacity
	new_raw := mem.resize(rawptr(buf.data), buf.capacity, new_cap, ARROW_ALIGNMENT, buf.allocator) or_return
	new_data := cast([^]u8)new_raw
	mem.zero(rawptr(new_data[old_cap:]), new_cap - old_cap)
	buf.data     = new_data
	buf.size     = new_size
	buf.capacity = new_cap
	return nil
}
