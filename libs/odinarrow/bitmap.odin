package odinarrow

import "base:intrinsics"

// Bytes required to store n_bits bits, padded to ARROW_ALIGNMENT.
// The padding keeps cast-to-[^]u64 safe and matches the Arrow spec.
bitmap_byte_count :: #force_inline proc "contextless" (n_bits: int) -> int {
	return align_up((n_bits + 7) / 8, ARROW_ALIGNMENT)
}

// True if element i is valid (not null). Bit i maps to byte i/8, bit i%8.
bitmap_get :: #force_inline proc "contextless" (data: [^]u8, i: int) -> bool {
	return (data[i >> 3] >> u8(i & 7)) & 1 == 1
}

// Mark element i as valid.
bitmap_set :: #force_inline proc "contextless" (data: [^]u8, i: int) {
	data[i >> 3] |= 1 << u8(i & 7)
}

// Mark element i as null.
bitmap_clear :: #force_inline proc "contextless" (data: [^]u8, i: int) {
	data[i >> 3] &= ~(1 << u8(i & 7))
}

// Count valid (set) bits in the first n_bits bits.
// Correctly handles n_bits that are not a multiple of 8: the final partial byte
// is masked before counting so that bits belonging to neighbouring elements
// (e.g. past the end of a slice) are never included in the total.
bitmap_popcount :: proc "contextless" (data: [^]u8, n_bits: int) -> int {
	if n_bits == 0 do return 0
	n_bytes := (n_bits + 7) / 8
	count   := 0
	n_words := n_bytes / 8

	// 8-byte words — safe because buffer capacity is 64-byte aligned,
	// so the underlying allocation is always u64-addressable from byte 0.
	data64 := cast([^]u64)data
	for w in 0..<n_words {
		count += int(intrinsics.count_ones(data64[w]))
	}

	// Remaining full bytes (all except the last partial byte)
	tail_start := n_words * 8
	for i := tail_start; i < n_bytes - 1; i += 1 {
		count += int(intrinsics.count_ones(data[i]))
	}

	// Last byte: mask away bits beyond n_bits
	if n_bytes > tail_start {
		last_byte := data[n_bytes - 1]
		remainder := uint(n_bits) % 8
		if remainder != 0 {
			last_byte &= u8((1 << remainder) - 1)
		}
		count += int(intrinsics.count_ones(last_byte))
	}
	return count
}

// Set all n_bits bits to 1 (mark every element valid).
// Bits beyond n_bits within the last byte are cleared to preserve the zero-tail invariant.
bitmap_set_all :: proc "contextless" (data: [^]u8, n_bits: int) {
	if n_bits == 0 do return
	n_bytes := (n_bits + 7) / 8
	for i in 0..<n_bytes {
		data[i] = 0xFF
	}
	remainder := uint(n_bits) % 8
	if remainder != 0 {
		data[n_bytes - 1] = u8((1 << remainder) - 1)
	}
}

// Clear all bytes in the bitmap buffer (mark every element null).
bitmap_clear_all :: proc "contextless" (data: [^]u8, n_bytes: int) {
	for i in 0..<n_bytes {
		data[i] = 0
	}
}
