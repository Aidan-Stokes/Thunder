package odinarrow

import "core:mem"

// Run-end encoding (REE): a numeric column of `length` logical elements stored
// as `k` runs, where run i spans logical indices [run_ends[i-1], run_ends[i])
// and holds the value `values[i]`. run_ends is strictly increasing and
// run_ends[k-1] == length.
//
// This breaks the plain Arrow value-buffer layout, but it is the foundation for
// *encoding-aware* kernels: a sum/min/max over an REE column is O(runs), not
// O(length), and the column itself is k·(4+sizeof(T)) bytes instead of
// length·sizeof(T). For data with long runs (sorted, grouped, sparse, or
// constant-heavy columns) that is a large reduction in both time and memory —
// the kernels never touch the elements that the encoding already collapsed.
//
// Nulls are not yet modelled here (a future REE-with-validity would add a
// run-level validity bitmap); this v1 targets dense numeric columns.

RLE_Array :: struct($T: typeid) {
	run_ends:  []i32, // strictly increasing exclusive ends; last == length
	values:    []T,   // one value per run
	length:    int,   // logical length
	allocator: mem.Allocator,
}

// ── builder ─────────────────────────────────────────────────────────────────

RLE_Builder :: struct($T: typeid) {
	run_ends: [dynamic]i32,
	values:   [dynamic]T,
	length:   int,
}

rle_builder_make :: proc($T: typeid, initial_runs := 64, allocator := context.allocator) -> RLE_Builder(T) {
	return RLE_Builder(T){
		run_ends = make([dynamic]i32, 0, initial_runs, allocator),
		values   = make([dynamic]T,   0, initial_runs, allocator),
	}
}

// Append one logical value, coalescing with the current run if it is equal.
rle_append :: proc(b: ^RLE_Builder($T), v: T) {
	n := len(b.values)
	b.length += 1
	if n > 0 && b.values[n-1] == v {
		b.run_ends[n-1] = i32(b.length)   // extend the current run
	} else {
		append(&b.values, v)
		append(&b.run_ends, i32(b.length))
	}
}

rle_builder_finish :: proc(b: ^RLE_Builder($T), allocator := context.allocator) -> RLE_Array(T) {
	re := make([]i32, len(b.run_ends), allocator)
	vs := make([]T,   len(b.values),   allocator)
	copy(re, b.run_ends[:])
	copy(vs, b.values[:])
	return RLE_Array(T){ run_ends = re, values = vs, length = b.length, allocator = allocator }
}

rle_builder_destroy :: proc(b: ^RLE_Builder($T)) {
	delete(b.run_ends)
	delete(b.values)
	b^ = {}
}

rle_free :: proc(a: ^RLE_Array($T)) {
	delete(a.run_ends, a.allocator)
	delete(a.values,   a.allocator)
	a^ = {}
}

// ── encoding-aware kernels (O(runs)) ────────────────────────────────────────

rle_run_count :: proc(a: ^RLE_Array($T)) -> int { return len(a.values) }

// Number of bytes the encoded column occupies (vs length·sizeof(T) decoded).
rle_encoded_bytes :: proc(a: ^RLE_Array($T)) -> int {
	return len(a.run_ends) * size_of(i32) + len(a.values) * size_of(T)
}

// Sum of the logical column: Σ value_i · run_length_i, one multiply per run.
rle_sum :: proc(a: ^RLE_Array($T)) -> f64 {
	sum: f64
	prev: i32 = 0
	for i in 0..<len(a.values) {
		run_len := a.run_ends[i] - prev
		sum += f64(a.values[i]) * f64(run_len)
		prev = a.run_ends[i]
	}
	return sum
}

// Min/Max of the logical column == min/max over the distinct run values, since
// every logical element equals one of them.
rle_min_max :: proc(a: ^RLE_Array($T)) -> (lo: f64, hi: f64, ok: bool) {
	if len(a.values) == 0 { return }
	mn := a.values[0]; mx := a.values[0]
	for i in 1..<len(a.values) {
		v := a.values[i]
		if v < mn { mn = v }
		if v > mx { mx = v }
	}
	return f64(mn), f64(mx), true
}

// ── accessors / materialisation ─────────────────────────────────────────────

// Logical element at index i (binary search over run_ends).
rle_get :: proc(a: ^RLE_Array($T), i: int) -> T {
	assert(i >= 0 && i < a.length, "rle_get: out of bounds")
	lo, hi := 0, len(a.run_ends)
	for lo < hi {
		mid := (lo + hi) / 2
		if int(a.run_ends[mid]) <= i { lo = mid + 1 } else { hi = mid }
	}
	return a.values[lo]
}

// Expand the REE column back into a plain Arrow Array (the escape hatch when a
// dense representation is actually required).
rle_decode :: proc(a: ^RLE_Array($T), allocator := context.allocator) -> (arr: Array, err: mem.Allocator_Error) {
	data_buf := buffer_make(a.length * size_of(T), allocator) or_return
	dst := cast([^]T)data_buf.data
	prev: i32 = 0
	for i in 0..<len(a.values) {
		v := a.values[i]
		for j in prev..<a.run_ends[i] { dst[j] = v }
		prev = a.run_ends[i]
	}
	arr = Array{
		type    = _data_type_for(T),
		length  = a.length,
		buffers = {{}, data_buf, {}},
	}
	return
}

// Run-end-encode a plain numeric Array (coalescing equal neighbours).
rle_encode :: proc(src: ^Array, $T: typeid, allocator := context.allocator) -> RLE_Array(T) {
	b := rle_builder_make(T, 64, allocator)
	defer rle_builder_destroy(&b)
	data := cast([^]T)src.buffers[1].data
	off  := src.offset
	for i in 0..<src.length {
		rle_append(&b, data[off + i])
	}
	return rle_builder_finish(&b, allocator)
}
