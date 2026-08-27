package odinarrow

import "core:mem"
import "core:strings"

// Umbra / "German-style" strings: each element is a fixed 16-byte slot
//
//   { length: u32, data: [12]u8 }
//
// - length <= 12: the bytes live inline in `data` (the rest zero-padded). No
//   separate data buffer, no offset indirection, no pointer chase.
// - length  > 12: data[0:4] is a 4-byte prefix and data[4:12] is an i64 offset
//   into a side `data` buffer holding the full bytes.
//
// The payoff is comparison: the first 4 bytes (the prefix, zero-padded) sit in
// the slot, so the vast majority of string comparisons resolve in registers
// without ever touching the data buffer. For random-access comparison work like
// sorting — where Arrow's `[offsets][bytes]` layout takes a cache miss into the
// data buffer on every compare — this is a large win.
//
// This breaks the Arrow Utf8 layout by design; it is a separate type, and would
// be transcoded at an Arrow/IPC boundary.

Umbra_String :: struct {
	length: u32,
	data:   [12]u8,
}
#assert(size_of(Umbra_String) == 16)

Umbra_Array :: struct {
	slots:     []Umbra_String,
	data:      []u8, // backing bytes for strings longer than 12
	length:    int,
	allocator: mem.Allocator,
}

// ── builder ─────────────────────────────────────────────────────────────────

Umbra_Builder :: struct {
	slots:  [dynamic]Umbra_String,
	data:   [dynamic]u8,
	length: int,
}

umbra_builder_make :: proc(initial_cap := 64, allocator := context.allocator) -> Umbra_Builder {
	return Umbra_Builder{
		slots = make([dynamic]Umbra_String, 0, initial_cap, allocator),
		data  = make([dynamic]u8, 0, initial_cap * 8, allocator),
	}
}

umbra_append :: proc(b: ^Umbra_Builder, s: string) {
	us: Umbra_String
	us.length = u32(len(s))
	if len(s) <= 12 {
		copy(us.data[:], transmute([]u8)s)              // inline; tail stays zero
	} else {
		copy(us.data[0:4], transmute([]u8)s[:4])        // prefix
		_put_i64le(us.data[4:12], i64(len(b.data)))     // offset into the side buffer
		append(&b.data, ..transmute([]u8)s)
	}
	append(&b.slots, us)
	b.length += 1
}

umbra_builder_finish :: proc(b: ^Umbra_Builder, allocator := context.allocator) -> Umbra_Array {
	slots := make([]Umbra_String, len(b.slots), allocator)
	data  := make([]u8, len(b.data), allocator)
	copy(slots, b.slots[:])
	copy(data, b.data[:])
	return Umbra_Array{ slots = slots, data = data, length = b.length, allocator = allocator }
}

umbra_builder_destroy :: proc(b: ^Umbra_Builder) {
	delete(b.slots)
	delete(b.data)
	b^ = {}
}

umbra_free :: proc(a: ^Umbra_Array) {
	delete(a.slots, a.allocator)
	delete(a.data, a.allocator)
	a^ = {}
}

// ── accessors ───────────────────────────────────────────────────────────────

// Zero-copy view of element i. Valid while the array is alive.
umbra_get :: proc(a: ^Umbra_Array, i: int) -> string {
	s := &a.slots[i]
	if s.length <= 12 {
		return string(s.data[:s.length])
	}
	off := _get_i64le(s.data[4:12])
	return string(a.data[off : off + i64(s.length)])
}

umbra_length :: #force_inline proc(a: ^Umbra_Array, i: int) -> int { return int(a.slots[i].length) }

// ── comparison-heavy kernels ────────────────────────────────────────────────

// First 4 bytes as a big-endian u32 so a numeric compare equals a lexicographic
// compare of the (zero-padded) prefix.
_umbra_prefix :: #force_inline proc "contextless" (s: ^Umbra_String) -> u32 {
	return u32(s.data[0]) << 24 | u32(s.data[1]) << 16 | u32(s.data[2]) << 8 | u32(s.data[3])
}

// Lexicographic compare of elements i and j: -1, 0, or 1. The prefix resolves
// most pairs without touching the data buffer; equal prefixes fall back to a
// full byte compare.
umbra_compare :: proc(a: ^Umbra_Array, i, j: int) -> int {
	pi := _umbra_prefix(&a.slots[i])
	pj := _umbra_prefix(&a.slots[j])
	if pi != pj { return -1 if pi < pj else 1 }
	return strings.compare(umbra_get(a, i), umbra_get(a, j))
}

// Count elements equal to `needle`. Rejects on length, then prefix, before any
// data-buffer access.
umbra_count_eq :: proc(a: ^Umbra_Array, needle: string) -> int {
	nlen := u32(len(needle))
	npfx: u32
	{
		tmp: Umbra_String
		copy(tmp.data[0:4], transmute([]u8)needle[:min(len(needle), 4)])
		npfx = _umbra_prefix(&tmp)
	}
	count := 0
	for i in 0..<a.length {
		s := &a.slots[i]
		if s.length != nlen { continue }
		if _umbra_prefix(s) != npfx { continue }
		if umbra_get(a, i) == needle { count += 1 }
	}
	return count
}

// Stable ascending sort; returns an Int64 index array (feeds compute_take-style
// gathers). Uses umbra_compare, so the prefix fast path carries most of the work.
umbra_sort_indices :: proc(a: ^Umbra_Array, allocator := context.allocator) -> Array {
	n := a.length
	idx := make([]i64, max(n, 1), context.temp_allocator)
	for i in 0..<n { idx[i] = i64(i) }
	if n > 1 {
		tmp := make([]i64, n, context.temp_allocator)
		width := 1
		for width < n {
			i := 0
			for i < n {
				lo  := i
				mid := min(i + width, n)
				hi  := min(i + 2*width, n)
				x, y, k := lo, mid, lo
				for x < mid && y < hi {
					// stable: take left on ties (<= 0)
					if umbra_compare(a, int(idx[x]), int(idx[y])) <= 0 { tmp[k] = idx[x]; x += 1 }
					else                                               { tmp[k] = idx[y]; y += 1 }
					k += 1
				}
				for x < mid { tmp[k] = idx[x]; x += 1; k += 1 }
				for y < hi  { tmp[k] = idx[y]; y += 1; k += 1 }
				i += 2*width
			}
			copy(idx, tmp)
			width *= 2
		}
	}
	b := builder_make(i64, max(n, 1), allocator)
	defer builder_destroy(&b)
	for i in 0..<n { builder_append(&b, idx[i]) }
	arr, _ := builder_finish(&b, allocator)
	return arr
}

// ── little-endian i64 helpers (avoid unaligned i64 access into slot bytes) ───

_put_i64le :: #force_inline proc "contextless" (dst: []u8, v: i64) {
	u := u64(v)
	for k in 0..<8 { dst[k] = u8(u >> uint(8*k)) }
}
_get_i64le :: #force_inline proc "contextless" (src: []u8) -> i64 {
	u: u64
	for k in 0..<8 { u |= u64(src[k]) << uint(8*k) }
	return i64(u)
}
