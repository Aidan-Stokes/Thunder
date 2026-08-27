package odinarrow

import "core:mem"

// Array is the central type: an immutable, nullable, typed column of values.
//
// Memory layout (matches Apache Arrow columnar format):
//   buffers[0] — validity bitmap (nil if null_count == 0; no nulls → omitted)
//   buffers[1] — data values (contiguous typed array, or bit-packed for Bool)
//   buffers[2] — reserved for Phase 3 (offsets for variable-length types)
//
// Zero-copy slicing: offset and length change; buffers are shared.
// Ownership: buffers with allocator.procedure == nil are NOT freed by array_free
// (this is set by array_slice so sliced arrays are safe to free independently).
Array :: struct {
	type:       DataType,
	length:     int,
	null_count: int,    // -1 = unknown; call array_null_count to recompute
	offset:     int,    // element offset into buffers (non-zero after slice)
	buffers:    [3]Buffer,
	children:   []Array, // Phase 4+: nested types
}

// True if element i is null.
array_is_null :: #force_inline proc "contextless" (arr: ^Array, i: int) -> bool {
	bm := &arr.buffers[0]
	if bm.data == nil do return false
	return !bitmap_get(bm.data, arr.offset + i)
}

// True if element i is valid (not null).
array_is_valid :: #force_inline proc "contextless" (arr: ^Array, i: int) -> bool {
	bm := &arr.buffers[0]
	if bm.data == nil do return true
	return bitmap_get(bm.data, arr.offset + i)
}

// Return the null count, recounting from the bitmap if the cached value is -1.
array_null_count :: proc(arr: ^Array) -> int {
	if arr.null_count >= 0 do return arr.null_count
	if arr.buffers[0].data == nil {
		arr.null_count = 0
		return 0
	}
	// Count set (valid) bits in the range [offset, offset+length)
	total  := bitmap_popcount(arr.buffers[0].data, arr.offset + arr.length)
	prefix := arr.offset > 0 ? bitmap_popcount(arr.buffers[0].data, arr.offset) : 0
	arr.null_count = arr.length - (total - prefix)
	return arr.null_count
}

// O(1) zero-copy slice into [from, to). The returned Array shares all buffers
// with the parent — do NOT call array_free on it while the parent is live.
// null_count is set to -1 (unknown); call array_null_count to recompute.
array_slice :: proc(arr: Array, from, to: int) -> Array {
	assert(from >= 0 && to <= arr.length && from <= to, "array_slice: out of bounds")
	result           := arr
	result.offset     = arr.offset + from
	result.length     = to - from
	result.null_count = -1
	// Clear allocators so array_free won't attempt to double-free shared buffers
	for &buf in result.buffers {
		buf.allocator = {}
	}
	return result
}

// Get element i as type T.
// For Bool arrays: T = bool (values are bit-packed; use bitmap_get).
// For all other primitive arrays: T must match the array's DataType.
// Out-of-bounds panics in debug builds; undefined behaviour in release.
array_get :: proc(arr: ^Array, i: int, $T: typeid) -> T {
	assert(i >= 0 && i < arr.length, "array_get: index out of bounds")
	idx := arr.offset + i
	when T == bool {
		return bitmap_get(arr.buffers[1].data, idx)
	} else {
		return (cast([^]T)arr.buffers[1].data)[idx]
	}
}

// Like array_get but also returns validity. Second return is false when null.
array_try_get :: proc(arr: ^Array, i: int, $T: typeid) -> (val: T, valid: bool) {
	valid = array_is_valid(arr, i)
	if valid {
		val = array_get(arr, i, T)
	}
	return
}

// Free all owned buffers. Children freed recursively.
// Safe to call on zero-value Array and on sliced arrays (shared buffers
// are skipped because their allocator.procedure == nil).
array_free :: proc(arr: ^Array, allocator := context.allocator) {
	for &buf in arr.buffers {
		buffer_free(&buf)
	}
	for &child in arr.children {
		array_free(&child, allocator)
	}
	if arr.children != nil {
		delete(arr.children, allocator)
	}
	arr^ = {}
}

// Recount nulls by scanning the bitmap. Forces arr.null_count >= 0.
// Equivalent to array_null_count but always scans (ignores the cache).
array_recount_nulls :: proc(arr: ^Array) -> int {
	arr.null_count = -1
	return array_null_count(arr)
}

// Helper: copy an Array and its owned buffers into a new allocation.
// Children are NOT deep-copied (Phase 4+ concern).
array_copy :: proc(src: ^Array, allocator := context.allocator) -> (dst: Array, err: mem.Allocator_Error) {
	dst = src^
	for i in 0..<3 {
		if src.buffers[i].data != nil && src.buffers[i].allocator.procedure != nil {
			dst.buffers[i] = buffer_copy(&src.buffers[i], allocator) or_return
		}
	}
	return
}
