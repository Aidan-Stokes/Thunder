package odinarrow

import "base:intrinsics"
import "core:mem"

// ── sum ───────────────────────────────────────────────────────────────────────

// Sum of all non-null numeric elements, returned as f64.
// valid_count is the number of non-null elements that contributed.
compute_sum :: proc(arr: ^Array) -> (sum: f64, valid_count: int) {
	switch _ in arr.type {
	case Int8_Type:    return _sum_typed(arr, i8)
	case Int16_Type:   return _sum_typed(arr, i16)
	case Int32_Type:   return _sum_typed(arr, i32)
	case Int64_Type:   return _sum_typed(arr, i64)
	case UInt8_Type:   return _sum_typed(arr, u8)
	case UInt16_Type:  return _sum_typed(arr, u16)
	case UInt32_Type:  return _sum_typed(arr, u32)
	case UInt64_Type:  return _sum_typed(arr, u64)
	case Float32_Type: return _sum_typed(arr, f32)
	case Float64_Type: return _sum_typed(arr, f64)
	case Null_Type, Bool_Type,
	     String_Type, Large_String_Type,
	     Binary_Type, Large_Binary_Type,
	     Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		panic("compute_sum: type does not support numeric summation")
	}
	return
}

// ── null-aware run reducers ─────────────────────────────────────────────────────
//
// Reduce a contiguous, all-valid run of `n` elements. These let the null-aware
// aggregation paths process bitmap bytes that are 0xFF (8 valid in a row) with
// the SIMD kernels instead of testing one bit at a time.

_sum_run :: #force_inline proc(data: [^]$T, n: int) -> f64 {
	when T == f64      { return _sum_f64_simd(data, n) }
	else when T == i32 { return f64(_sum_i32_simd(data, n)) }
	else {
		s: f64
		for i in 0..<n { s += f64(data[i]) }
		return s
	}
}

_min_run :: #force_inline proc(data: [^]$T, n: int) -> T {
	when T == i32 { return _min_i32_simd(data, n) }
	else {
		best := data[0]
		for i in 1..<n { best = min(best, data[i]) }
		return best
	}
}

_max_run :: #force_inline proc(data: [^]$T, n: int) -> T {
	when T == i32 { return _max_i32_simd(data, n) }
	else {
		best := data[0]
		for i in 1..<n { best = max(best, data[i]) }
		return best
	}
}

_min_max_run :: #force_inline proc(data: [^]$T, n: int) -> (lo: T, hi: T) {
	when T == i32 { return _min_max_i32_simd(data, n) }
	else {
		lo = data[0]; hi = data[0]
		for i in 1..<n {
			v := data[i]
			if v < lo { lo = v }
			if v > hi { hi = v }
		}
		return
	}
}

_sum_typed :: #force_inline proc(arr: ^Array, $T: typeid) -> (sum: f64, valid_count: int) {
	data := cast([^]T)arr.buffers[1].data
	off  := arr.offset
	n    := arr.length
	if arr.buffers[0].data == nil {
		when T == f64 {
			sum = off == 0 ? _sum_f64_simd(data, n) : _sum_f64_simd(data[off:], n)
		} else when T == i32 {
			sum = f64(off == 0 ? _sum_i32_simd(data, n) : _sum_i32_simd(data[off:], n))
		} else {
			for i in 0..<n { sum += f64(data[off + i]) }
		}
		valid_count = n
	} else {
		vbits := arr.buffers[0].data
		if off & 7 != 0 {
			// Unaligned slice offset: bitmap bytes don't line up with element
			// groups, so fall back to per-element.
			for i in 0..<n {
				if array_is_valid(arr, i) { sum += f64(data[off + i]); valid_count += 1 }
			}
		} else {
			// Coalesce all-valid (0xFF) bytes into runs reduced by the SIMD
			// kernel; skip all-null (0) bytes; bit-test only mixed bytes.
			bstart     := off >> 3
			full_bytes := n >> 3
			run_lo     := -1
			for byi in 0..<full_bytes {
				byte := vbits[bstart + byi]
				if byte == 0xFF {
					if run_lo < 0 { run_lo = byi << 3 }
					valid_count += 8
					continue
				}
				if run_lo >= 0 {
					sum += _sum_run(data[off + run_lo:], (byi << 3) - run_lo)
					run_lo = -1
				}
				if byte != 0 {
					eb := byi << 3
					for k in 0..<8 {
						if (byte >> uint(k)) & 1 == 1 { sum += f64(data[off + eb + k]); valid_count += 1 }
					}
				}
			}
			if run_lo >= 0 {
				sum += _sum_run(data[off + run_lo:], (full_bytes << 3) - run_lo)
			}
			for i := full_bytes << 3; i < n; i += 1 {
				if bitmap_get(vbits, off + i) { sum += f64(data[off + i]); valid_count += 1 }
			}
		}
	}
	return
}

// ── min / max ─────────────────────────────────────────────────────────────────

// Minimum of all non-null elements. Returns (0, 0) when all elements are null.
compute_min :: proc(arr: ^Array) -> (min_val: f64, valid_count: int) {
	switch _ in arr.type {
	case Int8_Type:    return _min_typed(arr, i8)
	case Int16_Type:   return _min_typed(arr, i16)
	case Int32_Type:   return _min_typed(arr, i32)
	case Int64_Type:   return _min_typed(arr, i64)
	case UInt8_Type:   return _min_typed(arr, u8)
	case UInt16_Type:  return _min_typed(arr, u16)
	case UInt32_Type:  return _min_typed(arr, u32)
	case UInt64_Type:  return _min_typed(arr, u64)
	case Float32_Type: return _min_typed(arr, f32)
	case Float64_Type: return _min_typed(arr, f64)
	case Null_Type, Bool_Type,
	     String_Type, Large_String_Type,
	     Binary_Type, Large_Binary_Type,
	     Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		panic("compute_min: type does not support ordering")
	}
	return
}

// Maximum of all non-null elements.
compute_max :: proc(arr: ^Array) -> (max_val: f64, valid_count: int) {
	switch _ in arr.type {
	case Int8_Type:    return _max_typed(arr, i8)
	case Int16_Type:   return _max_typed(arr, i16)
	case Int32_Type:   return _max_typed(arr, i32)
	case Int64_Type:   return _max_typed(arr, i64)
	case UInt8_Type:   return _max_typed(arr, u8)
	case UInt16_Type:  return _max_typed(arr, u16)
	case UInt32_Type:  return _max_typed(arr, u32)
	case UInt64_Type:  return _max_typed(arr, u64)
	case Float32_Type: return _max_typed(arr, f32)
	case Float64_Type: return _max_typed(arr, f64)
	case Null_Type, Bool_Type,
	     String_Type, Large_String_Type,
	     Binary_Type, Large_Binary_Type,
	     Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		panic("compute_max: type does not support ordering")
	}
	return
}

// Comparisons run in the native element type T (not f64): converting per
// element blocks integer SIMD and costs ~7x on large arrays. The single
// f64 conversion happens once, on the final result.
_min_typed :: #force_inline proc(arr: ^Array, $T: typeid) -> (min_val: f64, valid_count: int) {
	data := cast([^]T)arr.buffers[1].data
	off  := arr.offset
	n    := arr.length
	if n == 0 do return
	best: T
	if arr.buffers[0].data == nil {
		when T == i32 {
			best = off == 0 ? _min_i32_simd(data, n) : _min_i32_simd(data[off:], n)
		} else {
			best = data[off]
			for i in 1..<n { best = min(best, data[off + i]) }
		}
		valid_count = n
	} else {
		vbits := arr.buffers[0].data
		if off & 7 != 0 {
			for i in 0..<n {
				if array_is_valid(arr, i) {
					v := data[off + i]
					if valid_count == 0 || v < best { best = v }
					valid_count += 1
				}
			}
		} else {
			bstart     := off >> 3
			full_bytes := n >> 3
			seen       := false
			run_lo     := -1
			for byi in 0..<full_bytes {
				byte := vbits[bstart + byi]
				if byte == 0xFF {
					if run_lo < 0 { run_lo = byi << 3 }
					valid_count += 8
					continue
				}
				if run_lo >= 0 {
					r := _min_run(data[off + run_lo:], (byi << 3) - run_lo)
					best = seen ? min(best, r) : r; seen = true
					run_lo = -1
				}
				if byte != 0 {
					eb := byi << 3
					for k in 0..<8 {
						if (byte >> uint(k)) & 1 == 1 {
							v := data[off + eb + k]
							best = seen ? min(best, v) : v; seen = true
							valid_count += 1
						}
					}
				}
			}
			if run_lo >= 0 {
				r := _min_run(data[off + run_lo:], (full_bytes << 3) - run_lo)
				best = seen ? min(best, r) : r; seen = true
			}
			for i := full_bytes << 3; i < n; i += 1 {
				if bitmap_get(vbits, off + i) {
					v := data[off + i]
					best = seen ? min(best, v) : v; seen = true
					valid_count += 1
				}
			}
		}
	}
	if valid_count > 0 { min_val = f64(best) }
	return
}

_max_typed :: #force_inline proc(arr: ^Array, $T: typeid) -> (max_val: f64, valid_count: int) {
	data := cast([^]T)arr.buffers[1].data
	off  := arr.offset
	n    := arr.length
	if n == 0 do return
	best: T
	if arr.buffers[0].data == nil {
		when T == i32 {
			best = off == 0 ? _max_i32_simd(data, n) : _max_i32_simd(data[off:], n)
		} else {
			best = data[off]
			for i in 1..<n { best = max(best, data[off + i]) }
		}
		valid_count = n
	} else {
		vbits := arr.buffers[0].data
		if off & 7 != 0 {
			for i in 0..<n {
				if array_is_valid(arr, i) {
					v := data[off + i]
					if valid_count == 0 || v > best { best = v }
					valid_count += 1
				}
			}
		} else {
			bstart     := off >> 3
			full_bytes := n >> 3
			seen       := false
			run_lo     := -1
			for byi in 0..<full_bytes {
				byte := vbits[bstart + byi]
				if byte == 0xFF {
					if run_lo < 0 { run_lo = byi << 3 }
					valid_count += 8
					continue
				}
				if run_lo >= 0 {
					r := _max_run(data[off + run_lo:], (byi << 3) - run_lo)
					best = seen ? max(best, r) : r; seen = true
					run_lo = -1
				}
				if byte != 0 {
					eb := byi << 3
					for k in 0..<8 {
						if (byte >> uint(k)) & 1 == 1 {
							v := data[off + eb + k]
							best = seen ? max(best, v) : v; seen = true
							valid_count += 1
						}
					}
				}
			}
			if run_lo >= 0 {
				r := _max_run(data[off + run_lo:], (full_bytes << 3) - run_lo)
				best = seen ? max(best, r) : r; seen = true
			}
			for i := full_bytes << 3; i < n; i += 1 {
				if bitmap_get(vbits, off + i) {
					v := data[off + i]
					best = seen ? max(best, v) : v; seen = true
					valid_count += 1
				}
			}
		}
	}
	if valid_count > 0 { max_val = f64(best) }
	return
}

// ── min_max (single pass) ─────────────────────────────────────────────────────

// Min and max in one pass — ~2× faster than calling compute_min + compute_max.
compute_min_max :: proc(arr: ^Array) -> (min_val: f64, max_val: f64, valid_count: int) {
	switch _ in arr.type {
	case Int8_Type:    return _min_max_typed(arr, i8)
	case Int16_Type:   return _min_max_typed(arr, i16)
	case Int32_Type:   return _min_max_typed(arr, i32)
	case Int64_Type:   return _min_max_typed(arr, i64)
	case UInt8_Type:   return _min_max_typed(arr, u8)
	case UInt16_Type:  return _min_max_typed(arr, u16)
	case UInt32_Type:  return _min_max_typed(arr, u32)
	case UInt64_Type:  return _min_max_typed(arr, u64)
	case Float32_Type: return _min_max_typed(arr, f32)
	case Float64_Type: return _min_max_typed(arr, f64)
	case Null_Type, Bool_Type,
	     String_Type, Large_String_Type,
	     Binary_Type, Large_Binary_Type,
	     Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		panic("compute_min_max: type does not support ordering")
	}
	return
}

_min_max_typed :: #force_inline proc(arr: ^Array, $T: typeid) -> (min_val: f64, max_val: f64, valid_count: int) {
	data := cast([^]T)arr.buffers[1].data
	off  := arr.offset
	n    := arr.length
	if n == 0 { return }
	lo, hi: T
	if arr.buffers[0].data == nil {
		when T == i32 {
			if off == 0 {
				lo, hi = _min_max_i32_simd(data, n)
			} else {
				lo, hi = _min_max_i32_simd(data[off:], n)
			}
		} else {
			lo = data[off]; hi = data[off]
			for i in 1..<n {
				v := data[off + i]
				if v < lo { lo = v }
				if v > hi { hi = v }
			}
		}
		valid_count = n
	} else {
		vbits := arr.buffers[0].data
		if off & 7 != 0 {
			for i in 0..<n {
				if array_is_valid(arr, i) {
					v := data[off + i]
					if valid_count == 0 || v < lo { lo = v }
					if valid_count == 0 || v > hi { hi = v }
					valid_count += 1
				}
			}
		} else {
			bstart     := off >> 3
			full_bytes := n >> 3
			seen       := false
			run_lo     := -1
			for byi in 0..<full_bytes {
				byte := vbits[bstart + byi]
				if byte == 0xFF {
					if run_lo < 0 { run_lo = byi << 3 }
					valid_count += 8
					continue
				}
				if run_lo >= 0 {
					rl, rh := _min_max_run(data[off + run_lo:], (byi << 3) - run_lo)
					if seen { lo = min(lo, rl); hi = max(hi, rh) } else { lo = rl; hi = rh; seen = true }
					run_lo = -1
				}
				if byte != 0 {
					eb := byi << 3
					for k in 0..<8 {
						if (byte >> uint(k)) & 1 == 1 {
							v := data[off + eb + k]
							if seen { if v < lo { lo = v }; if v > hi { hi = v } } else { lo = v; hi = v; seen = true }
							valid_count += 1
						}
					}
				}
			}
			if run_lo >= 0 {
				rl, rh := _min_max_run(data[off + run_lo:], (full_bytes << 3) - run_lo)
				if seen { lo = min(lo, rl); hi = max(hi, rh) } else { lo = rl; hi = rh; seen = true }
			}
			for i := full_bytes << 3; i < n; i += 1 {
				if bitmap_get(vbits, off + i) {
					v := data[off + i]
					if seen { if v < lo { lo = v }; if v > hi { hi = v } } else { lo = v; hi = v; seen = true }
					valid_count += 1
				}
			}
		}
	}
	if valid_count > 0 { min_val = f64(lo); max_val = f64(hi) }
	return
}

// ── mean ──────────────────────────────────────────────────────────────────────

// Mean of all non-null numeric elements. Returns (0, 0) when all null.
compute_mean :: proc(arr: ^Array) -> (mean: f64, valid_count: int) {
	sum: f64
	sum, valid_count = compute_sum(arr)
	if valid_count > 0 {
		mean = sum / f64(valid_count)
	}
	return
}

// ── count ─────────────────────────────────────────────────────────────────────

// Count total elements and non-null (valid) elements.
compute_count :: proc(arr: ^Array) -> (total: int, valid: int) {
	total = arr.length
	nc    := array_null_count(arr)
	valid  = total - nc
	return
}

// ── filter ────────────────────────────────────────────────────────────────────

// Apply a Bool mask to arr, returning a new Array with only the passing rows.
// mask must be Bool_Type and have the same length as arr.
// Null mask entries are treated as false (element excluded).
compute_filter :: proc(arr, mask: ^Array, allocator := context.allocator) -> (result: Array, err: mem.Allocator_Error) {
	assert(arr.length == mask.length, "compute_filter: length mismatch")
	_, is_bool := mask.type.(Bool_Type)
	assert(is_bool, "compute_filter: mask must be Bool type")

	switch _ in arr.type {
	case Int8_Type:    return _filter_typed(arr, mask, i8,  allocator)
	case Int16_Type:   return _filter_typed(arr, mask, i16, allocator)
	case Int32_Type:   return _filter_typed(arr, mask, i32, allocator)
	case Int64_Type:   return _filter_typed(arr, mask, i64, allocator)
	case UInt8_Type:   return _filter_typed(arr, mask, u8,  allocator)
	case UInt16_Type:  return _filter_typed(arr, mask, u16, allocator)
	case UInt32_Type:  return _filter_typed(arr, mask, u32, allocator)
	case UInt64_Type:  return _filter_typed(arr, mask, u64, allocator)
	case Float32_Type: return _filter_typed(arr, mask, f32, allocator)
	case Float64_Type: return _filter_typed(arr, mask, f64, allocator)
	case Bool_Type:    return _filter_bool(arr, mask, allocator)
	case String_Type:  return _filter_string(arr, mask, allocator)
	case Binary_Type:  return _filter_binary(arr, mask, allocator)
	case Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		return _filter_typed(arr, mask, i64, allocator)
	case Null_Type, Large_String_Type, Large_Binary_Type:
		panic("compute_filter: unsupported source type")
	}
	return
}

_mask_passes :: #force_inline proc "contextless" (mask: ^Array, i: int) -> bool {
	return array_is_valid(mask, i) && bitmap_get(mask.buffers[1].data, mask.offset + i)
}

_filter_typed :: proc(arr, mask: ^Array, $T: typeid, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	n         := arr.length
	mask_bits := mask.buffers[1].data
	arr_off   := arr.offset
	mask_off  := mask.offset

	// No-null fast path: skip the builder entirely — allocate exact output size and
	// write values directly. Avoids per-element bitmap maintenance and dynamic-array overhead.
	if arr.buffers[0].data == nil && mask.buffers[0].data == nil {
		out_count := _filter_count_bits(mask_bits, mask_off, n)
		data_buf  := buffer_make(out_count * size_of(T), allocator) or_return
		src  := cast([^]T)arr.buffers[1].data
		dst  := cast([^]T)data_buf.data
		out_i := 0
		if mask_off == 0 && out_count * 4 < n {
			// Sparse: visit only the set bits via tzcnt and gather their values —
			// work ∝ survivors, no per-bit branch. (Same selectivity hybrid as
			// compute_select; dense uses the byte loop below.)
			words   := cast([^]u64)mask_bits
			n_words := n / 64
			for w in 0..<n_words {
				bits := words[w]
				base := w * 64
				for bits != 0 {
					t := intrinsics.count_trailing_zeros(bits)
					dst[out_i] = src[arr_off + base + int(t)]
					out_i += 1
					bits &= bits - 1
				}
			}
			for i := n_words * 64; i < n; i += 1 {
				if bitmap_get(mask_bits, i) { dst[out_i] = src[arr_off + i]; out_i += 1 }
			}
		} else if mask_off == 0 {
			// Dense: byte-at-a-time, skipping empty bytes.
			n_full := (n / 8) * 8
			for i := 0; i < n_full; i += 8 {
				byte := mask_bits[i >> 3]
				if byte == 0 { continue }
				for bit in u8(0)..<8 {
					if (byte >> bit) & 1 == 1 {
						dst[out_i] = src[arr_off + i + int(bit)]
						out_i += 1
					}
				}
			}
			for i := n_full; i < n; i += 1 {
				if bitmap_get(mask_bits, i) {
					dst[out_i] = src[arr_off + i]
					out_i += 1
				}
			}
		} else {
			for i in 0..<n {
				if bitmap_get(mask_bits, mask_off + i) {
					dst[out_i] = src[arr_off + i]
					out_i += 1
				}
			}
		}
		result = Array{
			type    = _data_type_for(T),
			length  = out_count,
			buffers = {{}, data_buf, {}},
		}
		return
	}

	// General path (source or mask has nulls): builder with upper-bound capacity.
	b := builder_make(T, n, allocator)
	defer builder_destroy(&b)
	for i in 0..<n {
		if _mask_passes(mask, i) {
			if array_is_null(arr, i) {
				builder_append_null(&b)
			} else {
				builder_append(&b, array_get(arr, i, T))
			}
		}
	}
	return builder_finish(&b, allocator)
}

// Count set bits in mask_bits in the range [off, off+n).
_filter_count_bits :: proc "contextless" (mask_bits: [^]u8, off, n: int) -> int {
	if off == 0 {
		return bitmap_popcount(mask_bits, n)
	}
	count := 0
	for i in 0..<n {
		if bitmap_get(mask_bits, off + i) { count += 1 }
	}
	return count
}

_filter_bool :: proc(arr, mask: ^Array, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	b := builder_make(bool, 0, allocator)
	defer builder_destroy(&b)
	for i in 0..<arr.length {
		if _mask_passes(mask, i) {
			if array_is_null(arr, i) {
				builder_append_null(&b)
			} else {
				builder_append(&b, array_get(arr, i, bool))
			}
		}
	}
	return builder_finish(&b, allocator)
}

_filter_string :: proc(arr, mask: ^Array, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	b := string_builder_make(0, allocator)
	defer string_builder_destroy(&b)
	for i in 0..<arr.length {
		if _mask_passes(mask, i) {
			if array_is_null(arr, i) {
				string_builder_append_null(&b)
			} else {
				string_builder_append(&b, array_get_string(arr, i))
			}
		}
	}
	return string_builder_finish(&b, allocator)
}

_filter_binary :: proc(arr, mask: ^Array, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	b := binary_builder_make(0, allocator)
	defer binary_builder_destroy(&b)
	for i in 0..<arr.length {
		if _mask_passes(mask, i) {
			if array_is_null(arr, i) {
				binary_builder_append_null(&b)
			} else {
				binary_builder_append(&b, array_get_binary(arr, i))
			}
		}
	}
	return binary_builder_finish(&b, allocator)
}

// ── take ──────────────────────────────────────────────────────────────────────

// Return a new array containing arr[indices[i]] for each i.
// indices must be Int64_Type; out-of-range indices panic.
compute_take :: proc(arr, indices: ^Array, allocator := context.allocator) -> (result: Array, err: mem.Allocator_Error) {
	_, ok := indices.type.(Int64_Type)
	assert(ok, "compute_take: indices must be Int64_Type")
	switch _ in arr.type {
	case Int8_Type:    return _take_primitive(arr, indices, i8,  allocator)
	case Int16_Type:   return _take_primitive(arr, indices, i16, allocator)
	case Int32_Type:   return _take_primitive(arr, indices, i32, allocator)
	case Int64_Type:   return _take_primitive(arr, indices, i64, allocator)
	case UInt8_Type:   return _take_primitive(arr, indices, u8,  allocator)
	case UInt16_Type:  return _take_primitive(arr, indices, u16, allocator)
	case UInt32_Type:  return _take_primitive(arr, indices, u32, allocator)
	case UInt64_Type:  return _take_primitive(arr, indices, u64, allocator)
	case Float32_Type: return _take_primitive(arr, indices, f32, allocator)
	case Float64_Type: return _take_primitive(arr, indices, f64, allocator)
	case Bool_Type:    return _take_bool(arr, indices, allocator)
	case String_Type:  return _take_string(arr, indices, allocator)
	case Binary_Type:  return _take_binary(arr, indices, allocator)
	case Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		return _take_primitive(arr, indices, i64, allocator)
	case Null_Type, Large_String_Type, Large_Binary_Type:
		panic("compute_take: unsupported type")
	}
	return
}

// Fixed-width gather: dst[i] = src[idx[i]], straight into the output buffer with
// no builder. The no-null source path is a tight branchless loop; when the
// source has nulls, the value is still gathered (garbage tolerated for null
// slots) and the output validity bit is gathered alongside it. Index bounds are
// the caller's responsibility (same contract as array_get).
_take_primitive :: proc(arr, indices: ^Array, $T: typeid, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	n   := indices.length
	idx := (cast([^]i64)indices.buffers[1].data)[indices.offset:]
	src := (cast([^]T)arr.buffers[1].data)[arr.offset:]

	data_buf := buffer_make_uninit(n * size_of(T), allocator) or_return
	dst := cast([^]T)data_buf.data

	src_bm := arr.buffers[0].data
	bitmap_buf: Buffer
	null_count := 0
	if src_bm == nil {
		for i in 0..<n {
			dst[i] = src[idx[i]]
		}
	} else {
		bitmap_buf = buffer_make(bitmap_byte_count(n), allocator) or_return
		for i in 0..<n {
			j := idx[i]
			dst[i] = src[j]
			if bitmap_get(src_bm, arr.offset + int(j)) {
				bitmap_set(bitmap_buf.data, i)
			} else {
				null_count += 1
			}
		}
	}

	result = Array{
		type       = _data_type_for(T),
		length     = n,
		null_count = null_count,
		buffers    = {bitmap_buf, data_buf, {}},
	}
	return
}

// Bool gather stays on the builder path: values are bit-packed, so there is no
// contiguous [^]T to index into.
_take_bool :: proc(arr, indices: ^Array, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	n := indices.length
	b := builder_make(bool, n, allocator)
	defer builder_destroy(&b)
	for i in 0..<n {
		idx := int(array_get(indices, i, i64))
		if array_is_null(arr, idx) {
			builder_append_null(&b)
		} else {
			builder_append(&b, array_get(arr, idx, bool))
		}
	}
	return builder_finish(&b, allocator)
}

_take_string :: proc(arr, indices: ^Array, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	n := indices.length
	b := string_builder_make(n, allocator)
	defer string_builder_destroy(&b)
	for i in 0..<n {
		idx := int(array_get(indices, i, i64))
		if array_is_null(arr, idx) {
			string_builder_append_null(&b)
		} else {
			string_builder_append(&b, array_get_string(arr, idx))
		}
	}
	return string_builder_finish(&b, allocator)
}

_take_binary :: proc(arr, indices: ^Array, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	n := indices.length
	b := binary_builder_make(n, allocator)
	defer binary_builder_destroy(&b)
	for i in 0..<n {
		idx := int(array_get(indices, i, i64))
		if array_is_null(arr, idx) {
			binary_builder_append_null(&b)
		} else {
			binary_builder_append(&b, array_get_binary(arr, idx))
		}
	}
	return binary_builder_finish(&b, allocator)
}

// ── cast ──────────────────────────────────────────────────────────────────────

// Cast arr to a different numeric type.  Supports all numeric → numeric conversions.
// String and Bool types are not supported.
compute_cast :: proc(arr: ^Array, to: DataType, allocator := context.allocator) -> (result: Array, err: mem.Allocator_Error) {
	switch _ in to {
	case Int8_Type:    return _cast_to(arr, i8,  allocator)
	case Int16_Type:   return _cast_to(arr, i16, allocator)
	case Int32_Type:   return _cast_to(arr, i32, allocator)
	case Int64_Type:   return _cast_to(arr, i64, allocator)
	case UInt8_Type:   return _cast_to(arr, u8,  allocator)
	case UInt16_Type:  return _cast_to(arr, u16, allocator)
	case UInt32_Type:  return _cast_to(arr, u32, allocator)
	case UInt64_Type:  return _cast_to(arr, u64, allocator)
	case Float32_Type: return _cast_to(arr, f32, allocator)
	case Float64_Type: return _cast_to(arr, f64, allocator)
	case Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		return _cast_to(arr, i64, allocator)
	case Null_Type, Bool_Type,
	     String_Type, Large_String_Type,
	     Binary_Type, Large_Binary_Type:
		panic("compute_cast: unsupported target type")
	}
	return
}

// Dispatch on the source type so the inner conversion loop is monomorphised on
// both From and To — a tight, vectorisable `dst[i] = To(src[i])` with no
// per-element type switch.
_cast_to :: proc(arr: ^Array, $To: typeid, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	switch _ in arr.type {
	case Int8_Type:    return _cast_impl(arr, i8,  To, allocator)
	case Int16_Type:   return _cast_impl(arr, i16, To, allocator)
	case Int32_Type:   return _cast_impl(arr, i32, To, allocator)
	case Int64_Type:   return _cast_impl(arr, i64, To, allocator)
	case UInt8_Type:   return _cast_impl(arr, u8,  To, allocator)
	case UInt16_Type:  return _cast_impl(arr, u16, To, allocator)
	case UInt32_Type:  return _cast_impl(arr, u32, To, allocator)
	case UInt64_Type:  return _cast_impl(arr, u64, To, allocator)
	case Float32_Type: return _cast_impl(arr, f32, To, allocator)
	case Float64_Type: return _cast_impl(arr, f64, To, allocator)
	case Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		return _cast_impl(arr, i64, To, allocator)
	case Null_Type, Bool_Type,
	     String_Type, Large_String_Type,
	     Binary_Type, Large_Binary_Type:
		panic("compute_cast: unsupported source type")
	}
	return
}

// Outputs at/above this size are written with non-temporal (streaming) stores:
// such a buffer won't stay cache-resident, so the usual read-for-ownership of its
// cache lines is wasted bandwidth. Smaller outputs keep normal stores so a
// consumer can read them while still hot. Only taken when inputs start at offset
// 0 (so the SIMD loads are naturally aligned).
_NT_STORE_MIN_BYTES :: 1 << 23 // 8 MiB

_cast_impl :: proc(arr: ^Array, $From, $To: typeid, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	n := arr.length
	data_buf := buffer_make_uninit(n * size_of(To), allocator) or_return
	dst := cast([^]To)data_buf.data
	src := (cast([^]From)arr.buffers[1].data)[arr.offset:]
	start := 0
	if arr.offset == 0 && n * size_of(To) >= _NT_STORE_MIN_BYTES {
		K  :: 8
		VS :: #simd[K]From
		VD :: #simd[K]To
		svp := cast([^]VS)src
		dvp := cast([^]VD)dst
		nv  := n / K
		for i in 0..<nv {
			intrinsics.non_temporal_store(&dvp[i], cast(VD)svp[i])
		}
		intrinsics.atomic_thread_fence(.Seq_Cst)
		start = nv * K
	}
	for i in start..<n {
		dst[i] = To(src[i])
	}
	// Cast preserves null positions exactly — copy the validity bitmap wholesale.
	bitmap_buf := _validity_copy(arr, allocator) or_return
	result = Array{
		type       = _data_type_for(To),
		length     = n,
		null_count = array_null_count(arr),
		buffers    = {bitmap_buf, data_buf, {}},
	}
	return
}

// Build a fresh validity buffer for [arr.offset, arr.offset+length). Returns a
// zero-value Buffer when the input has no validity bitmap (all valid).
_validity_copy :: proc(arr: ^Array, allocator: mem.Allocator) -> (buf: Buffer, err: mem.Allocator_Error) {
	src := arr.buffers[0].data
	if src == nil do return
	n := arr.length
	buf = buffer_make(bitmap_byte_count(n), allocator) or_return
	if arr.offset & 7 == 0 {
		mem.copy(rawptr(buf.data), rawptr(src[arr.offset >> 3:]), bitmap_byte_count(n))
	} else {
		for i in 0..<n {
			if bitmap_get(src, arr.offset + i) do bitmap_set(buf.data, i)
		}
	}
	return
}

// ── arithmetic (element-wise) ─────────────────────────────────────────────────

Arithmetic_Op :: enum { Add, Sub, Mul, Div }

// Element-wise arithmetic on two numeric arrays of the same type.
// Result type matches the input type.  Null propagates: if either element is
// null the output element is null.
compute_arithmetic :: proc(left, right: ^Array, op: Arithmetic_Op, allocator := context.allocator) -> (result: Array, err: mem.Allocator_Error) {
	assert(left.length == right.length, "compute_arithmetic: length mismatch")
	switch op {
	case .Add: return _arith_op(left, right, .Add, allocator)
	case .Sub: return _arith_op(left, right, .Sub, allocator)
	case .Mul: return _arith_op(left, right, .Mul, allocator)
	case .Div: return _arith_op(left, right, .Div, allocator)
	}
	return
}

_arith_op :: proc(left, right: ^Array, $OP: Arithmetic_Op, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	switch _ in left.type {
	case Int8_Type:    return _arith_impl(left, right, i8,  OP, allocator)
	case Int16_Type:   return _arith_impl(left, right, i16, OP, allocator)
	case Int32_Type:   return _arith_impl(left, right, i32, OP, allocator)
	case Int64_Type:   return _arith_impl(left, right, i64, OP, allocator)
	case UInt8_Type:   return _arith_impl(left, right, u8,  OP, allocator)
	case UInt16_Type:  return _arith_impl(left, right, u16, OP, allocator)
	case UInt32_Type:  return _arith_impl(left, right, u32, OP, allocator)
	case UInt64_Type:  return _arith_impl(left, right, u64, OP, allocator)
	case Float32_Type: return _arith_impl(left, right, f32, OP, allocator)
	case Float64_Type: return _arith_impl(left, right, f64, OP, allocator)
	case Null_Type, Bool_Type,
	     String_Type, Large_String_Type,
	     Binary_Type, Large_Binary_Type,
	     Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		panic("compute_arithmetic: unsupported type")
	}
	return
}

// Monomorphised on both value type and op: the inner loop is a tight,
// vectorisable `dst[i] = l[i] OP r[i]` with no per-element branching. The
// validity bitmap is the AND of the two inputs (built only when nulls exist).
_arith_impl :: proc(left, right: ^Array, $T: typeid, $OP: Arithmetic_Op, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	n := left.length
	data_buf := buffer_make_uninit(n * size_of(T), allocator) or_return
	dst := cast([^]T)data_buf.data
	l := (cast([^]T)left.buffers[1].data)[left.offset:]
	r := (cast([^]T)right.buffers[1].data)[right.offset:]
	start := 0
	// Div is excluded: integer SIMD division isn't available, and it isn't a
	// hot path. Add/Sub/Mul stream large outputs past the cache.
	when OP != .Div {
		if left.offset == 0 && right.offset == 0 && n * size_of(T) >= _NT_STORE_MIN_BYTES {
			K  :: 64 / size_of(T) // one cache line per store
			V  :: #simd[K]T
			lv := cast([^]V)l
			rv := cast([^]V)r
			dv := cast([^]V)dst
			nv := n / K
			for i in 0..<nv {
				v: V
				when      OP == .Add { v = lv[i] + rv[i] }
				else when OP == .Sub { v = lv[i] - rv[i] }
				else when OP == .Mul { v = lv[i] * rv[i] }
				intrinsics.non_temporal_store(&dv[i], v)
			}
			intrinsics.atomic_thread_fence(.Seq_Cst)
			start = nv * K
		}
	}
	for i in start..<n {
		when      OP == .Add { dst[i] = l[i] + r[i] }
		else when OP == .Sub { dst[i] = l[i] - r[i] }
		else when OP == .Mul { dst[i] = l[i] * r[i] }
		else when OP == .Div { dst[i] = l[i] / r[i] }
	}

	// Validity: null propagates if either side is null. Skip entirely when
	// both inputs are fully valid (the common, hot path).
	bitmap_buf: Buffer
	null_count := 0
	lbm := left.buffers[0].data
	rbm := right.buffers[0].data
	if lbm != nil || rbm != nil {
		bitmap_buf = buffer_make(bitmap_byte_count(n), allocator) or_return
		for i in 0..<n {
			if array_is_valid(left, i) && array_is_valid(right, i) {
				bitmap_set(bitmap_buf.data, i)
			} else {
				null_count += 1
			}
		}
	}

	result = Array{
		type       = _data_type_for(T),
		length     = n,
		null_count = null_count,
		buffers    = {bitmap_buf, data_buf, {}},
	}
	return
}

// Convenience wrappers.
compute_add :: proc(left, right: ^Array, allocator := context.allocator) -> (Array, mem.Allocator_Error) {
	return compute_arithmetic(left, right, .Add, allocator)
}
compute_sub :: proc(left, right: ^Array, allocator := context.allocator) -> (Array, mem.Allocator_Error) {
	return compute_arithmetic(left, right, .Sub, allocator)
}
compute_mul :: proc(left, right: ^Array, allocator := context.allocator) -> (Array, mem.Allocator_Error) {
	return compute_arithmetic(left, right, .Mul, allocator)
}
compute_div :: proc(left, right: ^Array, allocator := context.allocator) -> (Array, mem.Allocator_Error) {
	return compute_arithmetic(left, right, .Div, allocator)
}

// ── sort_indices ────────────────────────────────────────────────────────────────

// Return an Int64 array of indices that stably sorts `arr` in ascending order.
// Nulls are ordered last (Arrow's default). The result can be fed directly into
// compute_take to materialise the sorted array.
compute_sort_indices :: proc(arr: ^Array, allocator := context.allocator) -> (result: Array, err: mem.Allocator_Error) {
	switch _ in arr.type {
	case Int8_Type:    return _sort_indices_typed(arr, i8,  allocator)
	case Int16_Type:   return _sort_indices_typed(arr, i16, allocator)
	case Int32_Type:   return _sort_indices_typed(arr, i32, allocator)
	case Int64_Type:   return _sort_indices_typed(arr, i64, allocator)
	case UInt8_Type:   return _sort_indices_typed(arr, u8,  allocator)
	case UInt16_Type:  return _sort_indices_typed(arr, u16, allocator)
	case UInt32_Type:  return _sort_indices_typed(arr, u32, allocator)
	case UInt64_Type:  return _sort_indices_typed(arr, u64, allocator)
	case Float32_Type: return _sort_indices_typed(arr, f32, allocator)
	case Float64_Type: return _sort_indices_typed(arr, f64, allocator)
	case String_Type:  return _sort_indices_string(arr, allocator)
	case Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		return _sort_indices_typed(arr, i64, allocator)
	case Bool_Type, Null_Type, Binary_Type,
	     Large_String_Type, Large_Binary_Type:
		panic("compute_sort_indices: unsupported type")
	}
	return
}

// Stable bottom-up merge sort over the index array. `less[i] < less[j]` is
// resolved by the caller-supplied ordering captured in `idx`; ties (including
// null/null) keep the lower original index, giving a stable result.
_sort_finish_indices :: proc(idx: []i64, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	b := builder_make(i64, max(len(idx), 1), allocator)
	defer builder_destroy(&b)
	for v in idx { builder_append(&b, v) }
	return builder_finish(&b, allocator)
}

// LSD radix sort of indices by integer key — O(n · size_of(T)) with cache-
// friendly sequential passes, versus the comparison merge sort's O(n log n)
// indirect comparisons. Used for null-free fixed-width integer columns. Keys and
// indices are permuted together each pass so reads stay sequential; signed keys
// are bias-encoded (flip the sign bit) so an unsigned byte radix yields signed
// order. Passes whose radix byte is constant are skipped (small-range columns).
_radix_sort_indices :: proc(arr: ^Array, $T: typeid, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	n    := arr.length
	data := (cast([^]T)arr.buffers[1].data)[arr.offset:]

	enc :: #force_inline proc(v: T) -> u64 {
		when      T == i8  { return u64((cast(u8) v) ~ 0x80) }
		else when T == i16 { return u64((cast(u16)v) ~ 0x8000) }
		else when T == i32 { return u64((cast(u32)v) ~ 0x8000_0000) }
		else when T == i64 { return      (cast(u64)v) ~ 0x8000_0000_0000_0000 }
		else               { return u64(v) } // u8/u16/u32/u64 already byte-ordered
	}

	keys := make([]u64, n, context.temp_allocator)
	idx  := make([]i64, n, context.temp_allocator)
	kbuf := make([]u64, n, context.temp_allocator)
	ibuf := make([]i64, n, context.temp_allocator)
	for i in 0..<n { keys[i] = enc(data[i]); idx[i] = i64(i) }

	sk, dk := keys, kbuf
	si, di := idx,  ibuf
	for d in 0..<size_of(T) {
		shift := uint(d) * 8
		count: [256]int
		for i in 0..<n { count[int((sk[i] >> shift) & 0xFF)] += 1 }
		skip := false
		for c in 0..<256 { if count[c] == n { skip = true; break } }
		if skip { continue }
		sum := 0
		for c in 0..<256 { t := count[c]; count[c] = sum; sum += t }
		for i in 0..<n {
			b := int((sk[i] >> shift) & 0xFF)
			p := count[b]; count[b] += 1
			dk[p] = sk[i]
			di[p] = si[i]
		}
		sk, dk = dk, sk
		si, di = di, si
	}

	data_buf := buffer_make_uninit(n * size_of(i64), allocator) or_return
	if n > 0 { mem.copy(rawptr(data_buf.data), raw_data(si), n * size_of(i64)) }
	result = Array{type = Int64_Type{}, length = n, null_count = 0, buffers = {{}, data_buf, {}}}
	return
}

_sort_indices_typed :: proc(arr: ^Array, $T: typeid, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	n := arr.length
	when T != f32 && T != f64 {
		// Null-free integer columns: radix sort (floats keep the comparison
		// sort to avoid NaN / signed-zero ordering subtleties).
		if n > 1 && array_null_count(arr) == 0 {
			return _radix_sort_indices(arr, T, allocator)
		}
	}
	idx := make([]i64, n, context.temp_allocator)
	for i in 0..<n { idx[i] = i64(i) }
	if n <= 1 { return _sort_finish_indices(idx, allocator) }

	data      := cast([^]T)arr.buffers[1].data
	off       := arr.offset
	has_nulls := arr.null_count != 0
	vbits     := arr.buffers[0].data

	// a ≤ b under "nulls last, then ascending value, ties keep order".
	le :: #force_inline proc(data: [^]T, off: int, has_nulls: bool, vbits: [^]u8, ai, bi: i64) -> bool {
		if has_nulls && vbits != nil {
			an := !bitmap_get(vbits, off + int(ai))
			bn := !bitmap_get(vbits, off + int(bi))
			if an && bn { return true }   // both null → keep order
			if an       { return false }  // a null → after b
			if bn       { return true }   // b null → a before
		}
		return data[off + int(ai)] <= data[off + int(bi)]
	}

	tmp := make([]i64, n, context.temp_allocator)
	width := 1
	for width < n {
		i := 0
		for i < n {
			lo  := i
			mid := min(i + width, n)
			hi  := min(i + 2*width, n)
			a, b, k := lo, mid, lo
			for a < mid && b < hi {
				if le(data, off, has_nulls, vbits, idx[a], idx[b]) { tmp[k] = idx[a]; a += 1 }
				else                                               { tmp[k] = idx[b]; b += 1 }
				k += 1
			}
			for a < mid { tmp[k] = idx[a]; a += 1; k += 1 }
			for b < hi  { tmp[k] = idx[b]; b += 1; k += 1 }
			i += 2*width
		}
		copy(idx, tmp)
		width *= 2
	}
	return _sort_finish_indices(idx, allocator)
}

_sort_indices_string :: proc(arr: ^Array, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	n := arr.length
	idx := make([]i64, n, context.temp_allocator)
	for i in 0..<n { idx[i] = i64(i) }
	if n <= 1 { return _sort_finish_indices(idx, allocator) }

	off       := arr.offset
	has_nulls := arr.null_count != 0
	vbits     := arr.buffers[0].data
	offsets   := cast([^]i32)arr.buffers[1].data
	bytes     := arr.buffers[2].data

	get :: #force_inline proc(offsets: [^]i32, bytes: [^]u8, off: int, i: i64) -> string {
		s := int(offsets[off + int(i)])
		e := int(offsets[off + int(i) + 1])
		if s == e { return "" }
		return string(bytes[s:e])
	}
	le :: #force_inline proc(offsets: [^]i32, bytes: [^]u8, off: int, has_nulls: bool, vbits: [^]u8, ai, bi: i64) -> bool {
		if has_nulls && vbits != nil {
			an := !bitmap_get(vbits, off + int(ai))
			bn := !bitmap_get(vbits, off + int(bi))
			if an && bn { return true }
			if an       { return false }
			if bn       { return true }
		}
		return get(offsets, bytes, off, ai) <= get(offsets, bytes, off, bi)
	}

	tmp := make([]i64, n, context.temp_allocator)
	width := 1
	for width < n {
		i := 0
		for i < n {
			lo  := i
			mid := min(i + width, n)
			hi  := min(i + 2*width, n)
			a, b, k := lo, mid, lo
			for a < mid && b < hi {
				if le(offsets, bytes, off, has_nulls, vbits, idx[a], idx[b]) { tmp[k] = idx[a]; a += 1 }
				else                                                         { tmp[k] = idx[b]; b += 1 }
				k += 1
			}
			for a < mid { tmp[k] = idx[a]; a += 1; k += 1 }
			for b < hi  { tmp[k] = idx[b]; b += 1; k += 1 }
			i += 2*width
		}
		copy(idx, tmp)
		width *= 2
	}
	return _sort_finish_indices(idx, allocator)
}
