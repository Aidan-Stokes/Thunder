package odinarrow

import "core:mem"

// Dictionary encoding for low-cardinality string columns: the distinct values
// are stored once in a `dict` string array, and each logical element is an i32
// `code` indexing into it.
//
// The encoding-aware payoff is group-by / value_counts: the codes ARE the group
// ids, so counting occurrences is an integer histogram over the codes — O(n)
// integer increments with no string hashing or comparison at all. The same idea
// extends to grouped aggregation (sum a second column bucketed by code). This is
// what columnar engines exploit when a column arrives dictionary-encoded from
// storage (Parquet stores columns this way), where the encode cost is already
// paid and amortised across many queries.
//
// Separate type (not in the Arrow Array union); transcoded at an Arrow boundary.

Str_Dict_Array :: struct {
	dict:      Array, // string Array of the distinct values
	codes:     []i32, // one dictionary index per logical element
	length:    int,
	allocator: mem.Allocator,
}

str_dict_free :: proc(d: ^Str_Dict_Array) {
	array_free(&d.dict, d.allocator)
	delete(d.codes, d.allocator)
	d^ = {}
}

n_dict :: proc(d: ^Str_Dict_Array) -> int { return d.dict.length }

// Dictionary-encode a plain Arrow string array (deduplicating values).
str_dict_encode :: proc(src: ^Array, allocator := context.allocator) -> Str_Dict_Array {
	n := src.length
	// Keys are zero-copy views into `src`'s data buffer, valid for this proc.
	lookup := make(map[string]i32, 1 << 8, context.temp_allocator)
	defer delete(lookup)

	codes := make([]i32, n, allocator)
	db := string_builder_make(64, allocator)
	for i in 0..<n {
		s := array_get_string(src, i)
		code, ok := lookup[s]
		if !ok {
			code = i32(db.length)
			string_builder_append(&db, s)
			lookup[s] = code
		}
		codes[i] = code
	}
	dict, _ := string_builder_finish(&db, allocator)
	string_builder_destroy(&db)
	return Str_Dict_Array{ dict = dict, codes = codes, length = n, allocator = allocator }
}

// Zero-copy view of logical element i.
str_dict_get :: proc(d: ^Str_Dict_Array, i: int) -> string {
	return array_get_string(&d.dict, int(d.codes[i]))
}

// ── encoding-aware kernels ──────────────────────────────────────────────────

// value_counts: occurrences of each dictionary entry. counts[c] is the number
// of logical elements whose value is dict[c]. Pure integer histogram over the
// codes — no string work.
str_dict_value_counts :: proc(d: ^Str_Dict_Array, allocator := context.allocator) -> []i64 {
	counts := make([]i64, d.dict.length, allocator)
	for c in d.codes {
		counts[c] += 1
	}
	return counts
}

// group-by sum: sum the parallel numeric column `weights` bucketed by this
// column's dictionary code. out[c] = Σ weights[i] where codes[i] == c.
str_dict_group_sum :: proc(d: ^Str_Dict_Array, weights: ^Array, allocator := context.allocator) -> []f64 {
	out := make([]f64, d.dict.length, allocator)
	wdata := cast([^]f64)weights.buffers[1].data
	woff  := weights.offset
	n := min(d.length, weights.length)
	for i in 0..<n {
		out[d.codes[i]] += wdata[woff + i]
	}
	return out
}

// Expand back into a plain Arrow string array.
str_dict_decode :: proc(d: ^Str_Dict_Array, allocator := context.allocator) -> (arr: Array, err: mem.Allocator_Error) {
	b := string_builder_make(d.length, allocator)
	defer string_builder_destroy(&b)
	for i in 0..<d.length {
		string_builder_append(&b, str_dict_get(d, i))
	}
	return string_builder_finish(&b, allocator)
}

// ── generic numeric dictionary ──────────────────────────────────────────────
//
// Dict_Array(T): the distinct values once in `dictionary`, an i32 `code` per
// element. As well as the group-by win (the codes are group ids), this is a
// bandwidth win for wide types — aggregation reads the narrow i32 codes instead
// of the full T values — and makes min/max O(n_dict): every logical element is
// one of the dictionary values, so min/max of the column is min/max of the
// (tiny) dictionary.

Dict_Array :: struct($T: typeid) {
	dictionary: []T,
	codes:      []i32,
	length:     int,
	allocator:  mem.Allocator,
}

dict_free :: proc(d: ^Dict_Array($T)) {
	delete(d.dictionary, d.allocator)
	delete(d.codes, d.allocator)
	d^ = {}
}

dict_size :: proc(d: ^Dict_Array($T)) -> int { return len(d.dictionary) }

dict_encode :: proc(src: ^Array, $T: typeid, allocator := context.allocator) -> Dict_Array(T) {
	n    := src.length
	data := cast([^]T)src.buffers[1].data
	off  := src.offset
	lookup := make(map[T]i32, 1 << 8, context.temp_allocator)
	defer delete(lookup)

	codes := make([]i32, n, allocator)
	dyn   := make([dynamic]T, 0, 64, context.temp_allocator)
	defer delete(dyn)
	for i in 0..<n {
		v := data[off + i]
		code, ok := lookup[v]
		if !ok {
			code = i32(len(dyn))
			append(&dyn, v)
			lookup[v] = code
		}
		codes[i] = code
	}
	dict := make([]T, len(dyn), allocator)
	copy(dict, dyn[:])
	return Dict_Array(T){ dictionary = dict, codes = codes, length = n, allocator = allocator }
}

dict_get :: proc(d: ^Dict_Array($T), i: int) -> T { return d.dictionary[d.codes[i]] }

// value_counts: integer histogram over the codes.
dict_value_counts :: proc(d: ^Dict_Array($T), allocator := context.allocator) -> []i64 {
	counts := make([]i64, len(d.dictionary), allocator)
	for c in d.codes { counts[c] += 1 }
	return counts
}

// Sum via a histogram over the (narrow) codes plus a weighted sum over the
// dictionary — reads sizeof(i32)·n instead of sizeof(T)·n.
dict_sum :: proc(d: ^Dict_Array($T)) -> f64 {
	counts := make([]i64, len(d.dictionary), context.temp_allocator)
	defer delete(counts, context.temp_allocator)
	for c in d.codes { counts[c] += 1 }
	sum: f64
	for v, ci in d.dictionary { sum += f64(v) * f64(counts[ci]) }
	return sum
}

// Min/Max over the dictionary values — O(n_dict), independent of length.
dict_min_max :: proc(d: ^Dict_Array($T)) -> (lo: f64, hi: f64, ok: bool) {
	if len(d.dictionary) == 0 { return }
	mn := d.dictionary[0]; mx := d.dictionary[0]
	for i in 1..<len(d.dictionary) {
		v := d.dictionary[i]
		if v < mn { mn = v }
		if v > mx { mx = v }
	}
	return f64(mn), f64(mx), true
}

dict_decode :: proc(d: ^Dict_Array($T), allocator := context.allocator) -> (arr: Array, err: mem.Allocator_Error) {
	buf := buffer_make(d.length * size_of(T), allocator) or_return
	dst := cast([^]T)buf.data
	for c, i in d.codes { dst[i] = d.dictionary[c] }
	arr = Array{ type = _data_type_for(T), length = d.length, buffers = {{}, buf, {}} }
	return
}
