package odinarrow

import "base:intrinsics"
import "core:mem"

// Selection vectors (C2): the result of a predicate is a list of surviving row
// indices, not a freshly materialised array. Downstream kernels operate through
// the selection, so a query can filter once and then touch only the columns and
// rows it actually needs — instead of copying every column's surviving rows up
// front (what compute_filter / record-batch filter do).
//
// The payoff shows up with multi-column batches and multi-step pipelines: the
// selection is cheap (just indices), and materialisation/aggregation of each
// column happens lazily and independently.

Selection :: struct {
	indices:   []i32, // surviving row indices, ascending
	length:    int,
	allocator: mem.Allocator,
}

selection_free :: proc(s: ^Selection) {
	delete(s.indices, s.allocator)
	s^ = {}
}

// Build a selection from a Bool mask (null mask entries count as false). Does
// NOT touch any data column — only the mask.
compute_select :: proc(mask: ^Array, allocator := context.allocator) -> Selection {
	n     := mask.length
	mbits := mask.buffers[1].data
	moff  := mask.offset

	count := 0
	if mask.buffers[0].data == nil {
		count = _filter_count_bits(mbits, moff, n)
	} else {
		for i in 0..<n { if _mask_passes(mask, i) { count += 1 } }
	}

	idx := make([]i32, count, allocator)
	if mask.buffers[0].data == nil && moff == 0 {
		// Two compress strategies, chosen by selectivity (we already counted):
		//  - sparse: visit only set bits via tzcnt (work ∝ survivors)
		//  - dense:  the byte-at-a-time loop, which skips empty bytes and has
		//            more ILP than the tzcnt loop's serial `bits &= bits-1` chain
		//            when ~half the bits are set.
		if count * 4 < n { _compress_sparse(mbits, n, idx) }
		else             { _compress_dense(mbits, n, idx) }
	} else {
		k := 0
		for i in 0..<n {
			if _mask_passes(mask, i) { idx[k] = i32(i); k += 1 }
		}
	}
	return Selection{ indices = idx, length = count, allocator = allocator }
}

// Sparse compress: read 64 bits at a time and visit ONLY the set bits via
// count-trailing-zeros (tzcnt). Work ∝ number of survivors. The bitmap is
// u64-addressable from byte 0 (Arrow buffers are 64-byte aligned).
_compress_sparse :: proc "contextless" (mbits: [^]u8, n: int, out: []i32) {
	k := 0
	words   := cast([^]u64)mbits
	n_words := n / 64
	for w in 0..<n_words {
		bits := words[w]
		base := w * 64
		for bits != 0 {
			t := intrinsics.count_trailing_zeros(bits)
			out[k] = i32(base + int(t))
			k += 1
			bits &= bits - 1
		}
	}
	for i := n_words * 64; i < n; i += 1 {
		if bitmap_get(mbits, i) { out[k] = i32(i); k += 1 }
	}
}

// Dense compress: byte-at-a-time, skipping empty bytes. Writes only the set
// positions (no scratch slot needed) and keeps the per-byte loops independent,
// which beats the tzcnt loop once a large fraction of bits are set.
_compress_dense :: proc "contextless" (mbits: [^]u8, n: int, out: []i32) {
	k := 0
	n_full := (n / 8) * 8
	for i := 0; i < n_full; i += 8 {
		b := mbits[i >> 3]
		if b == 0 { continue }
		for bit in u8(0)..<8 {
			if (b >> bit) & 1 == 1 { out[k] = i32(i + int(bit)); k += 1 }
		}
	}
	for i := n_full; i < n; i += 1 {
		if bitmap_get(mbits, i) { out[k] = i32(i); k += 1 }
	}
}

// ── aggregate through a selection (no materialisation) ──────────────────────

compute_sum_selection :: proc(arr: ^Array, sel: ^Selection) -> (sum: f64, valid_count: int) {
	switch _ in arr.type {
	case Int8_Type:    return _sum_selection_typed(arr, sel, i8)
	case Int16_Type:   return _sum_selection_typed(arr, sel, i16)
	case Int32_Type:   return _sum_selection_typed(arr, sel, i32)
	case Int64_Type:   return _sum_selection_typed(arr, sel, i64)
	case UInt8_Type:   return _sum_selection_typed(arr, sel, u8)
	case UInt16_Type:  return _sum_selection_typed(arr, sel, u16)
	case UInt32_Type:  return _sum_selection_typed(arr, sel, u32)
	case UInt64_Type:  return _sum_selection_typed(arr, sel, u64)
	case Float32_Type: return _sum_selection_typed(arr, sel, f32)
	case Float64_Type: return _sum_selection_typed(arr, sel, f64)
	case Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		return _sum_selection_typed(arr, sel, i64)
	case Null_Type, Bool_Type, String_Type, Large_String_Type, Binary_Type, Large_Binary_Type:
		panic("compute_sum_selection: type does not support numeric summation")
	}
	return
}

_sum_selection_typed :: #force_inline proc(arr: ^Array, sel: ^Selection, $T: typeid) -> (sum: f64, valid_count: int) {
	data := cast([^]T)arr.buffers[1].data
	off  := arr.offset
	if arr.buffers[0].data == nil {
		for idx in sel.indices { sum += f64(data[off + int(idx)]) }
		valid_count = sel.length
	} else {
		for idx in sel.indices {
			i := int(idx)
			if array_is_valid(arr, i) { sum += f64(data[off + i]); valid_count += 1 }
		}
	}
	return
}

// Number of selected rows.
compute_count_selection :: proc(sel: ^Selection) -> int { return sel.length }

compute_min_max_selection :: proc(arr: ^Array, sel: ^Selection) -> (min_val: f64, max_val: f64, valid_count: int) {
	switch _ in arr.type {
	case Int8_Type:    return _min_max_selection_typed(arr, sel, i8)
	case Int16_Type:   return _min_max_selection_typed(arr, sel, i16)
	case Int32_Type:   return _min_max_selection_typed(arr, sel, i32)
	case Int64_Type:   return _min_max_selection_typed(arr, sel, i64)
	case UInt8_Type:   return _min_max_selection_typed(arr, sel, u8)
	case UInt16_Type:  return _min_max_selection_typed(arr, sel, u16)
	case UInt32_Type:  return _min_max_selection_typed(arr, sel, u32)
	case UInt64_Type:  return _min_max_selection_typed(arr, sel, u64)
	case Float32_Type: return _min_max_selection_typed(arr, sel, f32)
	case Float64_Type: return _min_max_selection_typed(arr, sel, f64)
	case Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		return _min_max_selection_typed(arr, sel, i64)
	case Null_Type, Bool_Type, String_Type, Large_String_Type, Binary_Type, Large_Binary_Type:
		panic("compute_min_max_selection: type does not support ordering")
	}
	return
}

_min_max_selection_typed :: #force_inline proc(arr: ^Array, sel: ^Selection, $T: typeid) -> (min_val: f64, max_val: f64, valid_count: int) {
	data := cast([^]T)arr.buffers[1].data
	off  := arr.offset
	has_nulls := arr.buffers[0].data != nil
	lo, hi: T
	seen := false
	for j in 0..<sel.length {
		i := int(sel.indices[j])
		if has_nulls && !array_is_valid(arr, i) { continue }
		v := data[off + i]
		if seen { if v < lo { lo = v }; if v > hi { hi = v } } else { lo = v; hi = v; seen = true }
		valid_count += 1
	}
	if seen { min_val = f64(lo); max_val = f64(hi) }
	return
}

// ── materialise (the escape hatch: gather one column at the selected rows) ──

selection_take :: proc(arr: ^Array, sel: ^Selection, allocator := context.allocator) -> (result: Array, err: mem.Allocator_Error) {
	switch _ in arr.type {
	case Int8_Type:    return _selection_take_typed(arr, sel, i8,  allocator)
	case Int16_Type:   return _selection_take_typed(arr, sel, i16, allocator)
	case Int32_Type:   return _selection_take_typed(arr, sel, i32, allocator)
	case Int64_Type:   return _selection_take_typed(arr, sel, i64, allocator)
	case UInt8_Type:   return _selection_take_typed(arr, sel, u8,  allocator)
	case UInt16_Type:  return _selection_take_typed(arr, sel, u16, allocator)
	case UInt32_Type:  return _selection_take_typed(arr, sel, u32, allocator)
	case UInt64_Type:  return _selection_take_typed(arr, sel, u64, allocator)
	case Float32_Type: return _selection_take_typed(arr, sel, f32, allocator)
	case Float64_Type: return _selection_take_typed(arr, sel, f64, allocator)
	case String_Type:  return _selection_take_string(arr, sel, allocator)
	case Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		return _selection_take_typed(arr, sel, i64, allocator)
	case Bool_Type, Null_Type, Large_String_Type, Binary_Type, Large_Binary_Type:
		panic("selection_take: unsupported type")
	}
	return
}

_selection_take_typed :: proc(arr: ^Array, sel: ^Selection, $T: typeid, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	n    := sel.length
	data := cast([^]T)arr.buffers[1].data
	off  := arr.offset
	if arr.buffers[0].data == nil {
		buf := buffer_make(n * size_of(T), allocator) or_return
		dst := cast([^]T)buf.data
		for j in 0..<n { dst[j] = data[off + int(sel.indices[j])] }
		result = Array{ type = _data_type_for(T), length = n, buffers = {{}, buf, {}} }
		return
	}
	b := builder_make(T, n, allocator)
	defer builder_destroy(&b)
	for j in 0..<n {
		i := int(sel.indices[j])
		if array_is_null(arr, i) { builder_append_null(&b) } else { builder_append(&b, data[off + i]) }
	}
	return builder_finish(&b, allocator)
}

_selection_take_string :: proc(arr: ^Array, sel: ^Selection, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	b := string_builder_make(sel.length, allocator)
	defer string_builder_destroy(&b)
	for j in 0..<sel.length {
		i := int(sel.indices[j])
		if array_is_null(arr, i) { string_builder_append_null(&b) } else { string_builder_append(&b, array_get_string(arr, i)) }
	}
	return string_builder_finish(&b, allocator)
}

// Materialise every column of a batch at the selection (the eager equivalent of
// keeping a selection and gathering lazily — provided for comparison/when a
// dense batch is genuinely required).
record_batch_take :: proc(batch: ^Record_Batch, sel: ^Selection, allocator := context.allocator) -> (result: Record_Batch, ok: bool) {
	cols := make([]Array, len(batch.columns), allocator)
	for ci in 0..<len(batch.columns) {
		c, e := selection_take(&batch.columns[ci], sel, allocator)
		if e != nil {
			for k in 0..<ci { array_free(&cols[k], allocator) }
			delete(cols, allocator)
			return {}, false
		}
		cols[ci] = c
	}
	rb, made := record_batch_make(batch.schema, cols, allocator)
	delete(cols, allocator)   // record_batch_make copied the slice (shares buffers)
	return rb, made
}
