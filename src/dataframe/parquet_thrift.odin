package dataframe

// Thrift compact protocol encoder/decoder for Parquet file metadata.
// Only the subset needed by Parquet is implemented: i64, i32, string,
// bool, list, struct, and enum fields.

import "core:mem"

// ── Encoder ────────────────────────────────────────────────────────────────

Thrift_Writer :: struct {
	buf:  []u8,
	pos:  int,
	last_fid: int,
	alloc: mem.Allocator,
	// Per-struct last_fid scope stack: thrift_write_struct_begin pushes the
	// enclosing struct's last_fid, struct_end pops and restores it, so field
	// id deltas after a nested struct stay relative to the outer scope.
	fid_stack: [32]int,
	fid_sp:    int,
}

thrift_writer_init :: proc(w: ^Thrift_Writer, initial_cap := 4096, allocator := context.allocator) {
	w^ = Thrift_Writer{
		buf   = make([]u8, initial_cap, allocator),
		pos   = 0,
		last_fid = 0,
		alloc = allocator,
	}
}

thrift_writer_destroy :: proc(w: ^Thrift_Writer) {
	delete(w.buf, w.alloc)
	w^ = {}
}

thrift_writer_bytes :: proc(w: ^Thrift_Writer) -> []u8 {
	return w.buf[:w.pos]
}

@(private)
thrift_ensure :: proc(w: ^Thrift_Writer, n: int) {
	if w.pos + n <= len(w.buf) { return }
	new_cap := max(len(w.buf) * 2, w.pos + n + 256)
	nb := make([]u8, new_cap, w.alloc)
	copy(nb, w.buf[:w.pos])
	delete(w.buf, w.alloc)
	w.buf = nb
}

@(private)
thrift_write_varint_i64 :: proc(w: ^Thrift_Writer, v: i64) {
	uv := u64(v)
	uv = (uv << 1) ~ (uv >> 63)  // zigzag encode: positive*2, negative*2+1
	for uv > 0x7F {
		thrift_ensure(w, 1)
		w.buf[w.pos] = u8(uv & 0x7F) | 0x80
		w.pos += 1
		uv >>= 7
	}
	thrift_ensure(w, 1)
	w.buf[w.pos] = u8(uv)
	w.pos += 1
}

@(private)
thrift_write_varint_i32 :: proc(w: ^Thrift_Writer, v: i32) {
	thrift_write_varint_i64(w, i64(v))
}

// write_field_header writes a compact protocol field header.
// delta is fid - last_fid; if delta == 1 it's omitted (byte encoding).
thrift_write_field_header :: proc(w: ^Thrift_Writer, ttype: u8, fid: int, delta := 0) {
	d := delta
	if d == 0 { d = fid - w.last_fid }
	if d >= 1 && d <= 15 {
		thrift_ensure(w, 1)
		w.buf[w.pos] = u8(d) << 4 | (ttype & 0x0F)
		w.pos += 1
	} else {
		// Long form: bare type byte (zero delta nibble) + zigzag varint delta
		thrift_ensure(w, 1)
		w.buf[w.pos] = ttype & 0x0F
		w.pos += 1
		thrift_write_varint_i32(w, i32(d))
	}
	w.last_fid = fid
}

// thrift_write_bool_field writes a bool as field header + value byte.
thrift_write_bool_field :: proc(w: ^Thrift_Writer, v: bool, fid: int, delta := 0) {
	if v {
		thrift_write_field_header(w, 1, fid, delta) // type 1 = true
	} else {
		thrift_write_field_header(w, 2, fid, delta) // type 2 = false
	}
}

thrift_write_i64_field :: proc(w: ^Thrift_Writer, v: i64, fid: int, delta := 0, def := i64(0)) {
	if v == def { return }
	thrift_write_field_header(w, 7, fid, delta) // type 7 = i64
	thrift_write_varint_i64(w, v)
}

thrift_write_i32_field :: proc(w: ^Thrift_Writer, v: i32, fid: int, delta := 0, def := i32(0)) {
	if v == def { return }
	thrift_write_field_header(w, 6, fid, delta) // type 6 = i32
	thrift_write_varint_i32(w, v)
}

thrift_write_string_field :: proc(w: ^Thrift_Writer, v: string, fid: int, delta := 0) {
	if v == "" { return }
	thrift_write_field_header(w, 8, fid, delta) // type 8 = binary/string
	thrift_write_varint_i32(w, i32(len(v)))
	thrift_ensure(w, len(v))
	copy(w.buf[w.pos:], v)
	w.pos += len(v)
}

thrift_write_double_field :: proc(w: ^Thrift_Writer, v: f64, fid: int, delta := 0, def := f64(0)) {
	if v == def { return }
	thrift_write_field_header(w, 4, fid, delta) // type 4 = double
	thrift_ensure(w, 8)
	v_copy := v
	mem.copy(rawptr(&w.buf[w.pos]), rawptr(&v_copy), 8)
	w.pos += 8
}

// Begin a struct: pushes the enclosing scope's last_fid (restored by
// thrift_write_struct_end) and resets the delta base for the new scope.
thrift_write_struct_begin :: proc(w: ^Thrift_Writer) {
	if w.fid_sp < len(w.fid_stack) {
		w.fid_stack[w.fid_sp] = w.last_fid
		w.fid_sp += 1
	}
	w.last_fid = 0
}

thrift_write_struct_end :: proc(w: ^Thrift_Writer) {
	thrift_ensure(w, 1)
	w.buf[w.pos] = 0 // stop field
	w.pos += 1
	if w.fid_sp > 0 {
		w.fid_sp -= 1
		w.last_fid = w.fid_stack[w.fid_sp]
	} else {
		w.last_fid = 0
	}
}

// Begin a list field. count must be known up front.
thrift_write_list_begin :: proc(w: ^Thrift_Writer, elem_type: u8, count: int) {
	if count <= 14 {
		thrift_ensure(w, 1)
		w.buf[w.pos] = u8(count) << 4 | (elem_type & 0x0F)
		w.pos += 1
	} else {
		thrift_ensure(w, 1)
		w.buf[w.pos] = 0xF0 | (elem_type & 0x0F)
		w.pos += 1
		thrift_write_varint_i32(w, i32(count))
	}
}

thrift_write_list_end :: proc(w: ^Thrift_Writer) {
	// lists don't have an explicit end marker
}

// ── Decoder ────────────────────────────────────────────────────────────────

Thrift_Reader :: struct {
	data: []u8,
	pos:  int,
	last_fid: int,
	has_error: bool,
	// Per-struct last_fid scope stack: thrift_read_struct_begin pushes the
	// enclosing struct's last_fid, thrift_read_struct_end pops and restores
	// it, mirroring the writer side. Every deserialize function must keep
	// begin/end calls balanced (defer is the easy way).
	fid_stack: [32]int,
	fid_sp:    int,
}

thrift_reader_init :: proc(r: ^Thrift_Reader, data: []u8) {
	r^ = Thrift_Reader{data = data, pos = 0, last_fid = 0, has_error = false}
}

@(private)
thrift_read_varint_i64 :: proc(r: ^Thrift_Reader) -> (v: i64, ok: bool) {
	shift := u64(0)
	v = 0
	for {
		if r.pos >= len(r.data) { return 0, false }
		b := r.data[r.pos]
		r.pos += 1
		v |= i64(u64(b & 0x7F) << shift)
		if b & 0x80 == 0 {
			// Zigzag decode: (v >> 1) ^ -(v & 1)
			decoded := u64(v) >> 1
			if u64(v) & 1 == 1 { decoded = ~decoded }
			return i64(decoded), true
		}
		shift += 7
		if shift > 63 { return 0, false }
	}
}

thrift_read_varint_i32 :: proc(r: ^Thrift_Reader) -> (v: i32, ok: bool) {
	v64, ok64 := thrift_read_varint_i64(r)
	return i32(v64), ok64
}

thrift_read_byte :: proc(r: ^Thrift_Reader) -> (u8, bool) {
	if r.pos >= len(r.data) { return 0, false }
	b := r.data[r.pos]
	r.pos += 1
	return b, true
}

// Read a field header: returns (fid, field_type, ok).
// Short form: delta in the high nibble, type in the low nibble.
// Long form (delta nibble == 0): zigzag varint delta follows the type byte.
thrift_read_field_header :: proc(r: ^Thrift_Reader) -> (fid: int, field_type: u8, ok: bool) {
	b, bok := thrift_read_byte(r)
	if !bok { return 0, 0, false }

	if b == 0 { return 0, 0, true } // stop

	delta := int(b >> 4)
	field_type = b & 0x0F

	if delta == 0 {
		d, dok := thrift_read_varint_i32(r)
		if !dok { return 0, 0, false }
		fid = r.last_fid + int(d)
	} else {
		fid = r.last_fid + delta
	}

	r.last_fid = fid
	return fid, field_type, true
}

// Begin a struct: pushes the enclosing scope's last_fid (restored by
// thrift_read_struct_end) and resets the delta base for the new scope.
// Compact protocol structs have no begin marker on the wire.
thrift_read_struct_begin :: proc(r: ^Thrift_Reader) {
	if r.fid_sp < len(r.fid_stack) {
		r.fid_stack[r.fid_sp] = r.last_fid
		r.fid_sp += 1
	}
	r.last_fid = 0
}

// End a struct: pops the pushed last_fid back into place so subsequent
// field deltas in the outer scope stay relative to it.
thrift_read_struct_end :: proc(r: ^Thrift_Reader) {
	if r.fid_sp > 0 {
		r.fid_sp -= 1
		r.last_fid = r.fid_stack[r.fid_sp]
	} else {
		r.last_fid = 0
	}
}

thrift_read_i32 :: proc(r: ^Thrift_Reader) -> (i32, bool) {
	return thrift_read_varint_i32(r)
}

thrift_read_i64 :: proc(r: ^Thrift_Reader) -> (i64, bool) {
	return thrift_read_varint_i64(r)
}

thrift_read_double :: proc(r: ^Thrift_Reader) -> (f64, bool) {
	if r.pos + 8 > len(r.data) { return 0, false }
	v: f64
	mem.copy(rawptr(&v), rawptr(&r.data[r.pos]), 8)
	r.pos += 8
	return v, true
}

thrift_read_string :: proc(r: ^Thrift_Reader, allocator := context.allocator) -> (string, bool) {
	length, lok := thrift_read_varint_i32(r)
	if !lok || int(length) < 0 || r.pos + int(length) > len(r.data) { return "", false }
	s := make([]u8, int(length), allocator)
	if s == nil && length > 0 { return "", false }
	copy(s, r.data[r.pos : r.pos + int(length)])
	r.pos += int(length)
	return string(s), true
}

thrift_read_list_begin :: proc(r: ^Thrift_Reader) -> (elem_type: u8, count: int, ok: bool) {
	b, bok := thrift_read_byte(r)
	if !bok { return 0, 0, false }
	size := int(b >> 4)
	et := b & 0x0F
	if size == 15 {
		s, sok := thrift_read_varint_i32(r)
		if !sok { return 0, 0, false }
		size = int(s)
	}
	return et, size, true
}
