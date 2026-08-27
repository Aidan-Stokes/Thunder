package dataframe

// Row-key encoding for grouping operations (unique, partition_by).
//
// A row's key is a canonical byte encoding of its key-column values, used as
// a map[string] key so groups can be tracked with the built-in hash map
// (Odin maps only accept comparable keys, so composite keys are serialized).
//
// Encoding per column (in order):
//   - 0xFF NULL marker — the column's value at this row is NULL
//   - 0x01 strings — 4-byte little-endian length prefix + content bytes
//   - 0x00 everything else — elem_size raw bytes (injective for the supported
//     column types)
//
// The tag + length-prefix scheme keeps the concatenation unambiguous across
// rows, including multi-string keys, and makes NULL a distinct per-column
// value: two rows share a key iff, for every column, both are NULL or both
// are non-NULL with equal values. Float values are canonicalized so that
// +0.0 == -0.0 and every NaN bit pattern hashes alike (matching `==` on real
// values and polars grouping); non-NaN equal values therefore share a key iff
// they compare equal with `==`.

import "core:math"

// encode_row appends the canonical key of row across key_cols to buf
// (buf is cleared first and reused).
@(private)
encode_row :: proc(key_cols: []^Column, row: int, buf: ^[dynamic]byte) -> Error {
	clear(buf)
	for c in key_cols {
		if !row_valid(c.valid, row) {
			append(buf, 0xFF)
			continue
		}
		if is_categorical(c) {
			// group on the category string, never the code: two tables that
			// happen to share codes must not merge incorrectly.
			append(buf, 0x01)
			s, _ := categorical_value(c, row)
			len32 := u32(len(s))
			for k in 0 ..< 4 {
				append(buf, u8((len32 >> (8 * u32(k))) & 0xFF))
			}
			for b in transmute([]byte)s {
				append(buf, b)
			}
			continue
		}
		if c.dtype == typeid_of(string) {
			append(buf, 0x01)
			s := column_typed_view(c, string)[row]
			len32 := u32(len(s))
			for k in 0 ..< 4 {
				append(buf, u8((len32 >> (8 * u32(k))) & 0xFF))
			}
			for b in transmute([]byte)s {
				append(buf, b)
			}
		} else {
			append(buf, 0x00)
			value_ptr := ptr_offset(c.data, row * c.elem_size)
			switch c.dtype {
			case typeid_of(f32):
				v := column_typed_view(c, f32)[row]
				v = canonical_float32(v)
				write_fixed(buf, &v, size_of(f32))
			case typeid_of(f64):
				v := column_typed_view(c, f64)[row]
				v = canonical_float64(v)
				write_fixed(buf, &v, size_of(f64))
			case:
				write_fixed(buf, value_ptr, c.elem_size)
			}
		}
	}
	return .None
}

// write_fixed appends the raw bytes at p to buf.
@(private)
write_fixed :: proc(buf: ^[dynamic]byte, p: rawptr, size: int) {
	for i in 0 ..< size {
		append(buf, (^u8)(ptr_offset(p, i))^)
	}
}

// canonical_float32 maps -0.0 to +0.0 and every NaN to one canonical NaN.
@(private)
canonical_float32 :: proc(v: f32) -> f32 {
	if v != v {
		return math.nan_f32()
	}
	if v == 0.0 {
		return 0.0
	}
	return v
}

// canonical_float64 maps -0.0 to +0.0 and every NaN to one canonical NaN.
@(private)
canonical_float64 :: proc(v: f64) -> f64 {
	if v != v {
		return math.nan_f64()
	}
	if v == 0.0 {
		return 0.0
	}
	return v
}
