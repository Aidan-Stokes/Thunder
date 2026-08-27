package odinarrow

import "core:mem"
import "base:intrinsics"

// Comparison predicates and selection refinement. The point is conjunctive
// (AND) queries: with selections, each successive predicate is evaluated only on
// the rows that survived the previous one — so a selective first predicate makes
// every later predicate (and the final gather) touch far fewer rows.
//
// The comparison itself is monomorphised on the operator ($OP, a comptime
// constant) so the inner loop is a single branchless comparison the compiler can
// vectorise — a runtime `op` switch in the loop would block that entirely.

Compare_Op :: enum { Gt, Ge, Lt, Le, Eq, Ne }

@(private="file")
_apply :: #force_inline proc "contextless" (a, b: f64, $OP: Compare_Op) -> bool {
	when      OP == .Gt { return a >  b }
	else when OP == .Ge { return a >= b }
	else when OP == .Lt { return a <  b }
	else when OP == .Le { return a <= b }
	else when OP == .Eq { return a == b }
	else                { return a != b }
}

// Compare 8 contiguous elements against the scalar → packed mask byte (branchless).
@(private="file")
_cmp8 :: #force_inline proc "contextless" (data: [^]$T, base: int, s: f64, $OP: Compare_Op) -> u8 {
	b: u8 = 0
	#unroll for k in 0..<8 {
		b |= (u8(1) if _apply(f64(data[base + k]), s, OP) else u8(0)) << uint(k)
	}
	return b
}

// ── compute_compare → Bool mask ─────────────────────────────────────────────

compute_compare :: proc(arr: ^Array, op: Compare_Op, scalar: f64, allocator := context.allocator) -> (mask: Array, err: mem.Allocator_Error) {
	switch _ in arr.type {
	case Int8_Type:    return _compare_typed(arr, op, scalar, i8,  allocator)
	case Int16_Type:   return _compare_typed(arr, op, scalar, i16, allocator)
	case Int32_Type:   return _compare_typed(arr, op, scalar, i32, allocator)
	case Int64_Type:   return _compare_typed(arr, op, scalar, i64, allocator)
	case UInt8_Type:   return _compare_typed(arr, op, scalar, u8,  allocator)
	case UInt16_Type:  return _compare_typed(arr, op, scalar, u16, allocator)
	case UInt32_Type:  return _compare_typed(arr, op, scalar, u32, allocator)
	case UInt64_Type:  return _compare_typed(arr, op, scalar, u64, allocator)
	case Float32_Type: return _compare_typed(arr, op, scalar, f32, allocator)
	case Float64_Type: return _compare_typed(arr, op, scalar, f64, allocator)
	case Null_Type, Bool_Type, String_Type, Large_String_Type, Binary_Type, Large_Binary_Type,
	     Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		panic("compute_compare: type does not support ordering")
	}
	return
}

@(private="file")
_compare_typed :: proc(arr: ^Array, op: Compare_Op, scalar: f64, $T: typeid, allocator: mem.Allocator) -> (mask: Array, err: mem.Allocator_Error) {
	n    := arr.length
	data := cast([^]T)arr.buffers[1].data
	off  := arr.offset
	buf  := buffer_make(bitmap_byte_count(n), allocator) or_return   // zeroed

	if arr.buffers[0].data == nil {
		switch op {
		case .Gt: _compare_mono(data, off, n, scalar, buf.data, .Gt)
		case .Ge: _compare_mono(data, off, n, scalar, buf.data, .Ge)
		case .Lt: _compare_mono(data, off, n, scalar, buf.data, .Lt)
		case .Le: _compare_mono(data, off, n, scalar, buf.data, .Le)
		case .Eq: _compare_mono(data, off, n, scalar, buf.data, .Eq)
		case .Ne: _compare_mono(data, off, n, scalar, buf.data, .Ne)
		}
	} else {
		for i in 0..<n {
			if array_is_valid(arr, i) && _apply_rt(f64(data[off + i]), op, scalar) { buf.data[i >> 3] |= 1 << u8(i & 7) }
		}
	}
	mask = Array{ type = Bool_Type{}, length = n, buffers = {{}, buf, {}} }
	return
}

@(private="file")
_compare_mono :: proc "contextless" (data: [^]$T, off, n: int, s: f64, out: [^]u8, $OP: Compare_Op) {
	when T == f64 {
		// True SIMD compare → movemask: each 8-lane compare yields a mask vector
		// whose per-lane MSB packs directly into one output byte (LSB = lane 0).
		V  :: #simd[8]f64
		sv := V(s)
		vp := cast([^]V)(data[off:])
		nv := n / 8
		for i in 0..<nv {
			v := vp[i]
			m: #simd[8]u64
			when      OP == .Gt { m = intrinsics.simd_lanes_gt(v, sv) }
			else when OP == .Ge { m = intrinsics.simd_lanes_ge(v, sv) }
			else when OP == .Lt { m = intrinsics.simd_lanes_lt(v, sv) }
			else when OP == .Le { m = intrinsics.simd_lanes_le(v, sv) }
			else when OP == .Eq { m = intrinsics.simd_lanes_eq(v, sv) }
			else                { m = intrinsics.simd_lanes_ne(v, sv) }
			out[i] = transmute(u8)intrinsics.simd_extract_msbs(m)
		}
		for i := nv * 8; i < n; i += 1 {
			if _apply(data[off + i], s, OP) { out[i >> 3] |= 1 << u8(i & 7) }
		}
	} else {
		n_full := (n / 8) * 8
		for i := 0; i < n_full; i += 8 { out[i >> 3] = _cmp8(data, off + i, s, OP) }
		for i := n_full; i < n; i += 1 {
			if _apply(f64(data[off + i]), s, OP) { out[i >> 3] |= 1 << u8(i & 7) }
		}
	}
}

// Runtime-op fallback (null path only — rare, correctness over speed).
@(private="file")
_apply_rt :: #force_inline proc "contextless" (a: f64, op: Compare_Op, b: f64) -> bool {
	switch op {
	case .Gt: return a >  b
	case .Ge: return a >= b
	case .Lt: return a <  b
	case .Le: return a <= b
	case .Eq: return a == b
	case .Ne: return a != b
	}
	return false
}

// ── first predicate → selection ─────────────────────────────────────────────

// Evaluate the predicate into a temporary Bool mask with the vectorised compare,
// then compress that mask into a selection. This avoids the per-element dynamic
// append (whose repeated reallocations dominated, especially when many rows
// pass), and inherits compute_select's selectivity-aware tzcnt/byte compress.
compute_select_compare :: proc(arr: ^Array, op: Compare_Op, scalar: f64, allocator := context.allocator) -> Selection {
	mask, err := compute_compare(arr, op, scalar, allocator)
	if err != nil { return Selection{ allocator = allocator } }
	defer array_free(&mask, allocator)
	return compute_select(&mask, allocator)
}

// ── refine a selection by another predicate (evaluated at survivors only) ───

selection_refine_compare :: proc(sel: ^Selection, arr: ^Array, op: Compare_Op, scalar: f64, allocator := context.allocator) -> Selection {
	switch _ in arr.type {
	case Int8_Type:    return _refine_compare_typed(sel, arr, op, scalar, i8,  allocator)
	case Int16_Type:   return _refine_compare_typed(sel, arr, op, scalar, i16, allocator)
	case Int32_Type:   return _refine_compare_typed(sel, arr, op, scalar, i32, allocator)
	case Int64_Type:   return _refine_compare_typed(sel, arr, op, scalar, i64, allocator)
	case UInt8_Type:   return _refine_compare_typed(sel, arr, op, scalar, u8,  allocator)
	case UInt16_Type:  return _refine_compare_typed(sel, arr, op, scalar, u16, allocator)
	case UInt32_Type:  return _refine_compare_typed(sel, arr, op, scalar, u32, allocator)
	case UInt64_Type:  return _refine_compare_typed(sel, arr, op, scalar, u64, allocator)
	case Float32_Type: return _refine_compare_typed(sel, arr, op, scalar, f32, allocator)
	case Float64_Type: return _refine_compare_typed(sel, arr, op, scalar, f64, allocator)
	case Null_Type, Bool_Type, String_Type, Large_String_Type, Binary_Type, Large_Binary_Type,
	     Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		panic("selection_refine_compare: type does not support ordering")
	}
	return {}
}

@(private="file")
_refine_compare_typed :: proc(sel: ^Selection, arr: ^Array, op: Compare_Op, scalar: f64, $T: typeid, allocator: mem.Allocator) -> Selection {
	data := cast([^]T)arr.buffers[1].data
	off  := arr.offset
	has_nulls := arr.buffers[0].data != nil
	out := make([]i32, sel.length, allocator)   // upper bound
	k := 0
	for j in 0..<sel.length {
		i := int(sel.indices[j])
		if has_nulls && !array_is_valid(arr, i) { continue }
		if _apply_rt(f64(data[off + i]), op, scalar) { out[k] = sel.indices[j]; k += 1 }
	}
	return Selection{ indices = out[:k], length = k, allocator = allocator }
}
