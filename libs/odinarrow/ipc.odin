package odinarrow

// Arrow IPC file format (Feather v2).
//
// File layout:
//   "ARROW1\0\0"  (8 bytes)
//   Schema message
//   RecordBatch message(s)
//   EOS marker: 0xFFFFFFFF + 0x00000000
//   Footer FlatBuffer
//   Footer length (i32 LE)
//   "ARROW1"      (6 bytes)

import "core:mem"
import "core:os"

IPC_MAGIC_HEAD :: "ARROW1\x00\x00"  // 8 bytes
IPC_MAGIC_TAIL :: "ARROW1"          // 6 bytes
IPC_CONTINUATION :: i32(-1)          // 0xFFFFFFFF

// ── FlatBuffers builder (back-to-front / prepend) ─────────────────────────────
//
// FlatBuffers stores every internal reference as an UNSIGNED forward offset: a
// field referencing a sub-object must point at a HIGHER address.  The standard
// way to guarantee that is to assemble the buffer back-to-front — leaf objects
// are written first (ending up nearest the end), parents afterwards (nearer the
// front) — so every reference points forward.  This keeps the same emission
// order our encoders use (leaves first); only the primitives differ from a
// naive append buffer.  `off` handles are measured as distance-from-end (i.e.
// FlatBuffers' uoffset space): a smaller handle is a higher physical address.

FBB :: struct {
    backing:   []u8,         // live bytes are backing[head:]
    head:      int,
    minalign:  int,
    obj_start: int,          // size at the most recent _fbb_start_table
    vtable:    [dynamic]int, // per-field distance-from-end during table build
    allocator: mem.Allocator,
}

_fbb_make :: proc(initial_cap := 256, allocator := context.allocator) -> FBB {
    return FBB{
        backing   = make([]u8, max(initial_cap, 16), allocator),
        head      = max(initial_cap, 16),
        minalign  = 1,
        vtable    = make([dynamic]int, 0, 16, allocator),
        allocator = allocator,
    }
}

_fbb_destroy :: proc(b: ^FBB) {
    delete(b.backing, b.allocator)
    delete(b.vtable)
}

// Bytes written so far (== distance from current front to the end).
_fbb_size :: #force_inline proc(b: ^FBB) -> int { return len(b.backing) - b.head }

// Ensure room to prepend at least `n` more bytes.
_fbb_reserve :: proc(b: ^FBB, n: int) {
    if b.head >= n { return }
    used    := _fbb_size(b)
    new_len := max(len(b.backing) * 2, used + n + 16)
    nb := make([]u8, new_len, b.allocator)
    copy(nb[new_len-used:], b.backing[b.head:])
    delete(b.backing, b.allocator)
    b.backing = nb
    b.head    = new_len - used
}

// Pad so that, after `additional` more bytes are prepended, the next element of
// alignment `align` starts on an aligned boundary.
_fbb_pre_align :: proc(b: ^FBB, additional, align: int) {
    if align > b.minalign { b.minalign = align }
    sz  := _fbb_size(b) + additional
    pad := ((~sz) + 1) & (align - 1)
    if pad == 0 { return }
    _fbb_reserve(b, pad)
    for _ in 0..<pad { b.head -= 1; b.backing[b.head] = 0 }
}

_fbb_prepend_bytes :: proc(b: ^FBB, src: []u8) {
    _fbb_reserve(b, len(src))
    b.head -= len(src)
    copy(b.backing[b.head:], src)
}

_fbb_push_u8 :: proc(b: ^FBB, v: u8) {
    _fbb_reserve(b, 1); b.head -= 1; b.backing[b.head] = v
}
_fbb_push_u16 :: proc(b: ^FBB, v: u16) {
    _fbb_pre_align(b, 0, 2); _fbb_reserve(b, 2); b.head -= 2
    b.backing[b.head] = u8(v); b.backing[b.head+1] = u8(v>>8)
}
_fbb_push_i16 :: #force_inline proc(b: ^FBB, v: i16) { _fbb_push_u16(b, u16(v)) }
_fbb_push_u32 :: proc(b: ^FBB, v: u32) {
    _fbb_pre_align(b, 0, 4); _fbb_reserve(b, 4); b.head -= 4
    b.backing[b.head]   = u8(v)
    b.backing[b.head+1] = u8(v>>8)
    b.backing[b.head+2] = u8(v>>16)
    b.backing[b.head+3] = u8(v>>24)
}
_fbb_push_i32 :: #force_inline proc(b: ^FBB, v: i32) { _fbb_push_u32(b, u32(v)) }
_fbb_push_i64 :: proc(b: ^FBB, v: i64) {
    _fbb_pre_align(b, 0, 8); _fbb_reserve(b, 8); b.head -= 8
    u := u64(v)
    for i in 0..<8 { b.backing[b.head+i] = u8(u >> uint(8*i)) }
}

// ── table construction ────────────────────────────────────────────────────────

_fbb_start_table :: proc(b: ^FBB, num_fields: int) {
    clear(&b.vtable)
    resize(&b.vtable, num_fields)  // zero-filled
    b.obj_start = _fbb_size(b)
}

_fbb_slot :: #force_inline proc(b: ^FBB, field: int) { b.vtable[field] = _fbb_size(b) }

_fbb_add_i16 :: proc(b: ^FBB, field: int, v, def: i16) {
    if v == def { return }
    _fbb_push_i16(b, v); _fbb_slot(b, field)
}
_fbb_add_i32 :: proc(b: ^FBB, field: int, v, def: i32) {
    if v == def { return }
    _fbb_push_i32(b, v); _fbb_slot(b, field)
}
_fbb_add_i64 :: proc(b: ^FBB, field: int, v, def: i64) {
    if v == def { return }
    _fbb_push_i64(b, v); _fbb_slot(b, field)
}
_fbb_add_u8 :: proc(b: ^FBB, field: int, v, def: u8) {
    if v == def { return }
    _fbb_push_u8(b, v); _fbb_slot(b, field)
}
_fbb_add_bool :: proc(b: ^FBB, field: int, v: bool, def: bool) {
    _fbb_add_u8(b, field, u8(1) if v else u8(0), u8(1) if def else u8(0))
}

// Add a forward (uoffset) reference to a previously-built object.
_fbb_add_offset :: proc(b: ^FBB, field: int, off: int) {
    if off == 0 { return }
    _fbb_pre_align(b, 0, 4)
    val := _fbb_size(b) - off + 4
    _fbb_push_u32(b, u32(val))
    _fbb_slot(b, field)
}

// Close the current table; returns its distance-from-end handle.
_fbb_end_table :: proc(b: ^FBB) -> int {
    _fbb_push_i32(b, 0)          // soffset placeholder
    table_off := _fbb_size(b)    // table start (== soffset location)

    num := len(b.vtable)
    for num > 0 && b.vtable[num-1] == 0 { num -= 1 }  // trim trailing absent fields

    // vtable entries (one voffset per field) in reverse field order.
    for i := num - 1; i >= 0; i -= 1 {
        voff := u16(0)
        if b.vtable[i] != 0 { voff = u16(table_off - b.vtable[i]) }
        _fbb_push_u16(b, voff)
    }
    _fbb_push_u16(b, u16(table_off - b.obj_start))  // object size
    _fbb_push_u16(b, u16(4 + 2 * num))              // vtable size
    vt_off := _fbb_size(b)

    // Patch the table's soffset to point back to this vtable (positive: vtable
    // sits at a lower address than the table).
    soffset := i32(vt_off - table_off)
    pos := len(b.backing) - table_off
    b.backing[pos]   = u8(soffset)
    b.backing[pos+1] = u8(u32(soffset)>>8)
    b.backing[pos+2] = u8(u32(soffset)>>16)
    b.backing[pos+3] = u8(u32(soffset)>>24)
    return table_off
}

// ── strings & vectors ─────────────────────────────────────────────────────────

_fbb_create_string :: proc(b: ^FBB, s: string) -> int {
    _fbb_pre_align(b, len(s) + 1, 4)
    _fbb_push_u8(b, 0)                          // null terminator
    _fbb_prepend_bytes(b, transmute([]u8)s)
    _fbb_push_u32(b, u32(len(s)))               // length prefix (4-aligned)
    return _fbb_size(b)
}

// Vector of forward offsets (e.g. a fields vector).  `nil`/empty → count 0.
_fbb_create_offset_vector :: proc(b: ^FBB, offs: []int) -> int {
    n := len(offs)
    _fbb_pre_align(b, n * 4, 4)
    for i := n - 1; i >= 0; i -= 1 {
        _fbb_pre_align(b, 0, 4)
        val := _fbb_size(b) - offs[i] + 4
        _fbb_push_u32(b, u32(val))
    }
    _fbb_push_u32(b, u32(n))
    return _fbb_size(b)
}

// Vector of inline fixed-size structs (already in correct little-endian layout).
// `structs` may be empty → count 0.
_fbb_create_struct_vector :: proc(b: ^FBB, structs: [][]u8, elem_align: int) -> int {
    n := len(structs)
    elem_size := n > 0 ? len(structs[0]) : 0
    _fbb_pre_align(b, n * elem_size, 4)
    _fbb_pre_align(b, n * elem_size, elem_align)
    for i := n - 1; i >= 0; i -= 1 {
        _fbb_prepend_bytes(b, structs[i])
    }
    _fbb_push_u32(b, u32(n))
    return _fbb_size(b)
}

// Prepend the root offset and emit the finished buffer (caller owns it).
_fbb_finish :: proc(b: ^FBB, root: int, allocator := context.allocator) -> []u8 {
    _fbb_pre_align(b, 4, b.minalign)
    val := _fbb_size(b) - root + 4
    _fbb_push_u32(b, u32(val))
    used := _fbb_size(b)
    out := make([]u8, used, allocator)
    copy(out, b.backing[b.head:])
    return out
}

// Little-endian scalar writers for inline struct payloads.
_le_i64 :: #force_inline proc(dst: []u8, v: i64) {
    u := u64(v)
    for i in 0..<8 { dst[i] = u8(u >> uint(8*i)) }
}
_le_i32 :: #force_inline proc(dst: []u8, v: i32) {
    u := u32(v)
    for i in 0..<4 { dst[i] = u8(u >> uint(8*i)) }
}

// ── Arrow type encoding ───────────────────────────────────────────────────────

// Arrow IPC Type enum (format/Schema.fbs): None=0, Null=1, Int=2, FloatingPoint=3,
// Binary=4, Utf8=5, Bool=6, ..., LargeBinary=19, LargeUtf8=20
_IPC_TYPE_NONE        :: u8(0)
_IPC_TYPE_INT         :: u8(2)
_IPC_TYPE_FLOATINGPT  :: u8(3)
_IPC_TYPE_BINARY      :: u8(4)
_IPC_TYPE_UTF8        :: u8(5)
_IPC_TYPE_BOOL        :: u8(6)
_IPC_TYPE_DATE        :: u8(8)
_IPC_TYPE_TIME        :: u8(9)
_IPC_TYPE_TIMESTAMP   :: u8(10)
_IPC_TYPE_DURATION    :: u8(11)
_IPC_TYPE_LARGEBINARY :: u8(19)
_IPC_TYPE_LARGEUTF8   :: u8(20)

// MetadataVersion (V4=4 and V5=5 are both widely supported; use V5)
_IPC_VERSION :: i16(4)  // MetadataVersion.V5 = enum value 4

// Header type (union type tag in Message table)
_IPC_HEADER_SCHEMA      :: u8(1)
_IPC_HEADER_RECORD_BATCH :: u8(3)

// Endianness
_IPC_ENDIAN_LITTLE :: i16(0)

// Build an Arrow type table for the given DataType.
// Returns (type_discriminant, type_table_offset).  The offset is 0 only for the
// NONE discriminant; every concrete Arrow type — including the bodyless ones
// (Bool, Utf8, LargeUtf8) — must still carry a (possibly empty) type table, or
// Arrow rejects the field with "null field Field.type".
_ipc_write_type :: proc(b: ^FBB, dt: DataType) -> (disc: u8, off: int) {
    switch _ in dt {
    case Bool_Type:         return _IPC_TYPE_BOOL, _ipc_empty_type(b)
    case String_Type:       return _IPC_TYPE_UTF8, _ipc_empty_type(b)
    case Large_String_Type: return _IPC_TYPE_LARGEUTF8, _ipc_empty_type(b)
    case Binary_Type:       return _IPC_TYPE_BINARY, _ipc_empty_type(b)
    case Large_Binary_Type: return _IPC_TYPE_LARGEBINARY, _ipc_empty_type(b)
    case Int8_Type:   return _ipc_int_type(b, 8, true)
    case Int16_Type:  return _ipc_int_type(b, 16, true)
    case Int32_Type:  return _ipc_int_type(b, 32, true)
    case Int64_Type:  return _ipc_int_type(b, 64, true)
    case UInt8_Type:  return _ipc_int_type(b, 8, false)
    case UInt16_Type: return _ipc_int_type(b, 16, false)
    case UInt32_Type: return _ipc_int_type(b, 32, false)
    case UInt64_Type: return _ipc_int_type(b, 64, false)
    case Float32_Type: return _ipc_fp_type(b, 1)  // precision SINGLE
    case Float64_Type: return _ipc_fp_type(b, 2)  // precision DOUBLE
    case Null_Type:
        return _IPC_TYPE_NONE, 0
    case Date64_Type:    return _ipc_date_type(b, 1)    // DateUnit MILLISECOND
    case Time64_Type:    return _ipc_time_type(b, 2)    // TimeUnit MICROSECOND
    case Timestamp_Type: return _ipc_timestamp_type(b, 2)  // MICROSECOND
    case Duration_Type:  return _ipc_duration_type(b, 2) // TimeUnit MICROSECOND
    }
    return _IPC_TYPE_NONE, 0
}

// Empty type table (Bool, Utf8, LargeUtf8 carry no fields but must still exist).
_ipc_empty_type :: proc(b: ^FBB) -> int {
    _fbb_start_table(b, 0)
    return _fbb_end_table(b)
}

// Int type table: { bitWidth: i32 @0, is_signed: bool @1 }
_ipc_int_type :: proc(b: ^FBB, bit_width: i32, is_signed: bool) -> (u8, int) {
    _fbb_start_table(b, 2)
    _fbb_add_i32(b, 0, bit_width, 0)
    _fbb_add_bool(b, 1, is_signed, false)
    return _IPC_TYPE_INT, _fbb_end_table(b)
}

// FloatingPoint type table: { precision: i16 @0 }
_ipc_fp_type :: proc(b: ^FBB, precision: i16) -> (u8, int) {
    _fbb_start_table(b, 1)
    _fbb_add_i16(b, 0, precision, 0)
    return _IPC_TYPE_FLOATINGPT, _fbb_end_table(b)
}

// Date type table: { unit: i16 @0 }  0=DAY, 1=MILLISECOND
_ipc_date_type :: proc(b: ^FBB, unit: i16) -> (u8, int) {
    _fbb_start_table(b, 1)
    _fbb_add_i16(b, 0, unit, 0)
    return _IPC_TYPE_DATE, _fbb_end_table(b)
}

// Time type table: { unit: i16 @0, bitWidth: i32 @1 }  unit: 0=SECOND,1=MILLI,2=MICRO,3=NANO
_ipc_time_type :: proc(b: ^FBB, unit: i16) -> (u8, int) {
    _fbb_start_table(b, 2)
    _fbb_add_i16(b, 0, unit, 0)
    _fbb_add_i32(b, 1, 64, 0)
    return _IPC_TYPE_TIME, _fbb_end_table(b)
}

// Timestamp type table: { unit: i16 @0 }
_ipc_timestamp_type :: proc(b: ^FBB, unit: i16) -> (u8, int) {
    _fbb_start_table(b, 1)
    _fbb_add_i16(b, 0, unit, 0)
    return _IPC_TYPE_TIMESTAMP, _fbb_end_table(b)
}

// Duration type table: { unit: i16 @0 }
_ipc_duration_type :: proc(b: ^FBB, unit: i16) -> (u8, int) {
    _fbb_start_table(b, 1)
    _fbb_add_i16(b, 0, unit, 0)
    return _IPC_TYPE_DURATION, _fbb_end_table(b)
}

// ── Schema FlatBuffer encoding ────────────────────────────────────────────────

// Build the full Schema sub-tree (field names, type tables, children vectors,
// Field tables, fields vector, Schema table) into `b`.
// Returns the offset handle of the Schema table.  Shared by the standalone
// schema message encoder and the footer encoder so the Field layout stays in
// lock-step with what pyarrow's verifier expects.
//
// Arrow Field table fields: name@0, nullable@1, type_type@2 (union tag),
// type@3 (union value), dictionary@4, children@5, custom_metadata@6.
_ipc_write_schema_subtree :: proc(b: ^FBB, schema: ^Schema) -> int {
    n := len(schema.fields)
    field_offs := make([]int, n, context.temp_allocator)

    for i in 0..<n {
        f := schema.fields[i]
        name_off := _fbb_create_string(b, f.name)
        disc, type_off := _ipc_write_type(b, f.type)
        children_off := _fbb_create_offset_vector(b, nil)  // empty (leaf field)

        _fbb_start_table(b, 7)
        _fbb_add_offset(b, 0, name_off)
        _fbb_add_bool(b, 1, f.nullable, false)
        _fbb_add_u8(b, 2, disc, _IPC_TYPE_NONE)
        if type_off != 0 { _fbb_add_offset(b, 3, type_off) }
        _fbb_add_offset(b, 5, children_off)
        field_offs[i] = _fbb_end_table(b)
    }

    fields_vec := _fbb_create_offset_vector(b, field_offs)

    // Schema table: endianness@0 (default Little=0, omitted), fields@1.
    _fbb_start_table(b, 4)
    _fbb_add_offset(b, 1, fields_vec)
    return _fbb_end_table(b)
}

// Returns the encoded schema message bytes (ready to write as IPC metadata).
_ipc_encode_schema :: proc(schema: ^Schema, allocator := context.allocator) -> []u8 {
    b := _fbb_make(256, context.allocator)
    defer _fbb_destroy(&b)

    schema_off := _ipc_write_schema_subtree(&b, schema)

    // Message table: version@0, header_type@1, header@2, bodyLength@3 (omitted).
    _fbb_start_table(&b, 5)
    _fbb_add_i16(&b, 0, _IPC_VERSION, 0)
    _fbb_add_u8(&b, 1, _IPC_HEADER_SCHEMA, 0)
    _fbb_add_offset(&b, 2, schema_off)
    msg := _fbb_end_table(&b)

    return _fbb_finish(&b, msg, allocator)
}

// ── RecordBatch FlatBuffer + body ─────────────────────────────────────────────

IPC_Body_Buffer :: struct {
    data:   [^]u8,
    length: i64,
}

// Encode a record batch metadata FlatBuffer.  body_buf_offset is the running
// byte offset within the IPC message body (tracks where each buffer starts).
// Returns the FlatBuffer bytes.  body_descs is populated with (data, length)
// for each column buffer, in the Arrow IPC buffer order.
_ipc_encode_record_batch :: proc(
    batch:         ^Record_Batch,
    body_descs:    ^[dynamic]IPC_Body_Buffer,
    allocator      := context.allocator,
) -> []u8 {
    n_cols := len(batch.columns)

    // Count total buffers and collect body data.
    // Each column: [validity bitmap?] + [data or offsets + data for strings]
    n_buffers := 0
    buf_offsets := make([dynamic]i64, 0, n_cols*3, context.temp_allocator)
    buf_lengths := make([dynamic]i64, 0, n_cols*3, context.temp_allocator)
    running_off := i64(0)

    for ci in 0..<n_cols {
        col := &batch.columns[ci]
        // Validity bitmap buffer (always present as a slot; nil if no nulls)
        validity_data := col.buffers[0].data
        validity_len  := i64(0)
        if col.null_count != 0 && validity_data != nil {
            validity_len = i64(bitmap_byte_count(col.length))
        }
        append(&buf_offsets, running_off)
        append(&buf_lengths, validity_len)
        if validity_len > 0 {
            append(body_descs, IPC_Body_Buffer{validity_data, validity_len})
        } else {
            append(body_descs, IPC_Body_Buffer{nil, 0})
        }
        running_off += validity_len
        _align_i64(&running_off, 8)

        switch _ in col.type {
        case String_Type, Binary_Type:
            // i32 offsets buffer
            off_len := i64((col.length + 1) * size_of(i32))
            append(&buf_offsets, running_off)
            append(&buf_lengths, off_len)
            append(body_descs, IPC_Body_Buffer{col.buffers[1].data, off_len})
            running_off += off_len
            _align_i64(&running_off, 8)
            // values buffer (data length = last offset)
            offsets_ptr := cast([^]i32)col.buffers[1].data
            data_len := i64(offsets_ptr[col.length + col.offset])
            append(&buf_offsets, running_off)
            append(&buf_lengths, data_len)
            append(body_descs, IPC_Body_Buffer{col.buffers[2].data, data_len})
            running_off += data_len
            _align_i64(&running_off, 8)
            n_buffers += 3
        case Large_String_Type, Large_Binary_Type:
            // i64 offsets buffer
            off_len := i64((col.length + 1) * size_of(i64))
            append(&buf_offsets, running_off)
            append(&buf_lengths, off_len)
            append(body_descs, IPC_Body_Buffer{col.buffers[1].data, off_len})
            running_off += off_len
            _align_i64(&running_off, 8)
            // values buffer (data length = last offset)
            offsets_ptr := cast([^]i64)col.buffers[1].data
            data_len := offsets_ptr[col.length + col.offset]
            append(&buf_offsets, running_off)
            append(&buf_lengths, data_len)
            append(body_descs, IPC_Body_Buffer{col.buffers[2].data, data_len})
            running_off += data_len
            _align_i64(&running_off, 8)
            n_buffers += 3
        case Bool_Type:
            data_len := i64(bitmap_byte_count(col.length))
            append(&buf_offsets, running_off)
            append(&buf_lengths, data_len)
            append(body_descs, IPC_Body_Buffer{col.buffers[1].data, data_len})
            running_off += data_len
            _align_i64(&running_off, 8)
            n_buffers += 2
        case Null_Type:
            n_buffers += 1
        case Int8_Type, Int16_Type, Int32_Type, Int64_Type,
             UInt8_Type, UInt16_Type, UInt32_Type, UInt64_Type,
             Float32_Type, Float64_Type,
             Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
            // fixed-width
            w := i64(type_byte_width(col.type))
            data_len := i64(col.length) * w
            append(&buf_offsets, running_off)
            append(&buf_lengths, data_len)
            append(body_descs, IPC_Body_Buffer{col.buffers[1].data, data_len})
            running_off += data_len
            _align_i64(&running_off, 8)
            n_buffers += 2
        }
    }

    body_length := running_off
    total_bufs := len(buf_offsets)

    b := _fbb_make(256, context.allocator)
    defer _fbb_destroy(&b)

    // ── nodes vector: FieldNode struct = {length:i64, null_count:i64} (16 bytes)
    node_structs := make([][]u8, n_cols, context.temp_allocator)
    node_storage := make([]u8, n_cols*16, context.temp_allocator)
    for ci in 0..<n_cols {
        col := &batch.columns[ci]
        nc := i64(col.null_count)
        if nc < 0 { nc = i64(array_null_count(col)) }
        s := node_storage[ci*16:ci*16+16]
        _le_i64(s[0:],  i64(col.length))
        _le_i64(s[8:],  nc)
        node_structs[ci] = s
    }
    nodes_vec := _fbb_create_struct_vector(&b, node_structs, 8)

    // ── buffers vector: Buffer struct = {offset:i64, length:i64} (16 bytes)
    buf_structs := make([][]u8, total_bufs, context.temp_allocator)
    buf_storage := make([]u8, total_bufs*16, context.temp_allocator)
    for i in 0..<total_bufs {
        s := buf_storage[i*16:i*16+16]
        _le_i64(s[0:], buf_offsets[i])
        _le_i64(s[8:], buf_lengths[i])
        buf_structs[i] = s
    }
    bufs_vec := _fbb_create_struct_vector(&b, buf_structs, 8)

    // ── RecordBatch table: length@0, nodes@1, buffers@2
    _fbb_start_table(&b, 3)
    _fbb_add_i64(&b, 0, i64(batch.length), 0)
    _fbb_add_offset(&b, 1, nodes_vec)
    _fbb_add_offset(&b, 2, bufs_vec)
    rb_off := _fbb_end_table(&b)

    // ── Message table: version@0, header_type@1, header@2, bodyLength@3
    _fbb_start_table(&b, 5)
    _fbb_add_i16(&b, 0, _IPC_VERSION, 0)
    _fbb_add_u8(&b, 1, _IPC_HEADER_RECORD_BATCH, 0)
    _fbb_add_offset(&b, 2, rb_off)
    _fbb_add_i64(&b, 3, body_length, 0)
    msg := _fbb_end_table(&b)

    _ = n_buffers
    return _fbb_finish(&b, msg, allocator)
}

_align_i64 :: #force_inline proc "contextless" (v: ^i64, align: i64) {
    v^ = (v^ + align - 1) & ~(align - 1)
}

// ── Footer FlatBuffer ─────────────────────────────────────────────────────────

IPC_Block :: struct { offset: i64, meta_len: i32, body_len: i64 }

_ipc_encode_footer :: proc(schema: ^Schema, blocks: []IPC_Block, allocator := context.allocator) -> []u8 {
    b := _fbb_make(512, context.allocator)
    defer _fbb_destroy(&b)

    // schema sub-tree (shared encoder, keeps Field layout in lock-step).
    schema_off := _ipc_write_schema_subtree(&b, schema)

    // recordBatches vector: Block struct =
    //   {offset:i64@0, metaDataLength:i32@8, pad:i32@12, bodyLength:i64@16} (24 bytes, 8-align)
    blk_structs := make([][]u8, len(blocks), context.temp_allocator)
    blk_storage := make([]u8, len(blocks)*24, context.temp_allocator)
    for blk, i in blocks {
        s := blk_storage[i*24:i*24+24]
        _le_i64(s[0:],  blk.offset)
        _le_i32(s[8:],  blk.meta_len)
        _le_i32(s[12:], 0)            // struct padding
        _le_i64(s[16:], blk.body_len)
        blk_structs[i] = s
    }
    rb_vec   := _fbb_create_struct_vector(&b, blk_structs, 8)
    dict_vec := _fbb_create_struct_vector(&b, nil, 8)  // empty dictionaries vector

    // Footer table: version@0, schema@1, dictionaries@2, recordBatches@3.
    _fbb_start_table(&b, 4)
    _fbb_add_i16(&b, 0, _IPC_VERSION, 0)
    _fbb_add_offset(&b, 1, schema_off)
    _fbb_add_offset(&b, 2, dict_vec)
    _fbb_add_offset(&b, 3, rb_vec)
    ft := _fbb_end_table(&b)

    return _fbb_finish(&b, ft, allocator)
}

// ── Public writer ─────────────────────────────────────────────────────────────

// Write all batches to an Arrow IPC file.
ipc_write_file :: proc(path: string, schema: ^Schema, batches: []Record_Batch) -> bool {
    f, err := os.open(path, {.Write, .Create, .Trunc}, os.Permissions_Default_File)
    if err != nil { return false }
    defer os.close(f)

    // Header magic
    os.write(f, transmute([]u8)string(IPC_MAGIC_HEAD))
    offset := i64(8)

    // Schema message
    schema_meta := _ipc_encode_schema(schema)
    defer delete(schema_meta)
    _ipc_write_message(f, schema_meta, nil, &offset)

    // Record batch messages + block tracking
    blocks := make([dynamic]IPC_Block, 0, len(batches))
    defer delete(blocks)

    for i in 0..<len(batches) {
        body_descs := make([dynamic]IPC_Body_Buffer, 0, 32)
        defer delete(body_descs)
        rb_meta := _ipc_encode_record_batch(&batches[i], &body_descs)
        defer delete(rb_meta)

        blk_start := offset
        body_len := _ipc_write_message(f, rb_meta, body_descs[:], &offset)
        // Block.metaDataLength spans the whole message metadata region: the
        // 8-byte prefix (continuation + size) plus the flatbuffer padded to 8.
        meta_pad   := (8 - (len(rb_meta) % 8)) % 8
        meta_total := i32(8 + len(rb_meta) + meta_pad)
        append(&blocks, IPC_Block{offset=blk_start, meta_len=meta_total, body_len=body_len})
    }

    // EOS marker
    _ipc_write_u32(f, 0xFFFF_FFFF)
    _ipc_write_u32(f, 0)
    offset += 8

    // Footer
    footer := _ipc_encode_footer(schema, blocks[:])
    defer delete(footer)
    os.write(f, footer)
    _ipc_write_i32(f, i32(len(footer)))

    // Tail magic
    os.write(f, transmute([]u8)string(IPC_MAGIC_TAIL))
    return true
}

// Write one encapsulated IPC message.  Returns body_length (rounded up to 8).
_ipc_write_message :: proc(f: ^os.File, meta: []u8, body_descs: []IPC_Body_Buffer, offset: ^i64) -> i64 {
    // Arrow spec: the on-wire metadata_size INCLUDES the padding that follows
    // the flatbuffer, so the body begins exactly at (here + 8 + metadata_size)
    // and metadata_size is a multiple of 8.  Writing the unpadded length instead
    // makes a stream reader mistake the padding zeros for the EOS marker.
    pad := i64((8 - (len(meta) % 8)) % 8)
    meta_size := i32(i64(len(meta)) + pad)

    _ipc_write_i32(f, IPC_CONTINUATION)
    _ipc_write_i32(f, meta_size)
    os.write(f, meta)
    for _ in i64(0)..<pad { _ipc_write_u8(f, 0) }
    offset^ += 8 + i64(meta_size)

    if body_descs == nil { return 0 }

    body_start := offset^
    for desc in body_descs {
        if desc.data != nil && desc.length > 0 {
            os.write(f, desc.data[:desc.length])
        }
        // Pad to 8-byte alignment
        padded := (desc.length + 7) & ~i64(7)
        for _ in desc.length..<padded { _ipc_write_u8(f, 0) }
        offset^ += padded
    }
    return offset^ - body_start
}

_ipc_write_u8  :: proc(f: ^os.File, v: u8)  { b: [1]u8 = {v}; os.write(f, b[:]) }
_ipc_write_u32 :: proc(f: ^os.File, v: u32) { b: [4]u8; b[0]=u8(v); b[1]=u8(v>>8); b[2]=u8(v>>16); b[3]=u8(v>>24); os.write(f, b[:]) }
_ipc_write_i32 :: proc(f: ^os.File, v: i32) { _ipc_write_u32(f, u32(v)) }

// ── Public reader ─────────────────────────────────────────────────────────────

// Read all record batches from an Arrow IPC file.
// The caller owns returned batches (call record_batch_free on each).
// `validate` (default on) runs cheap structural guards plus an offset-buffer
// integrity check on variable-length columns — pass false to skip the offset
// check for files you produced yourself (trusted, max throughput).
// `use_mmap` (default true) memory-maps the file for zero-copy reads — fast, but
// assumes the file stays stable: if another process truncates it underneath, a
// later page access SIGBUSes (A5). Pass use_mmap=false in hostile/shared-storage
// environments to read the file into a private heap buffer instead (a copy, but
// truncation-safe). `validate` runs the structural + offset integrity checks.
ipc_read_file :: proc(path: string, allocator := context.allocator, validate := true, use_mmap := true) -> (schema: ^Schema, batches: []Record_Batch, ok: bool) {
    // mmap so pages fault in lazily and columns are decoded zero-copy as views
    // into the mapping; otherwise (disabled or unavailable) read into memory.
    // `release` is the matching deleter (munmap, or nil → heap delete).
    data: []u8
    release: proc(_: []u8)
    if use_mmap {
        d, unmap, mapped := _ipc_map_file(path)
        if mapped { data = d; release = unmap }
    }
    if data == nil {
        d, read_err := os.read_entire_file(path, allocator)
        if read_err != nil { return }
        data    = d
        release = nil
    }
    // Ownership transfers to the first batch below; on any early return or when
    // there are no batches the backing is freed here.
    keep_data := false
    defer if !keep_data {
        if release != nil { release(data) }
        else              { delete(data, allocator) }
    }

    n := len(data)
    if n < 14 { return }
    if string(data[:8]) != IPC_MAGIC_HEAD { return }

    // ── Read footer (file-format approach: footer is at the end) ─────────────
    // Layout tail: [footer_fb][footer_len:i32][ARROW1:6]
    tail_magic_len := 6
    footer_len_off := n - tail_magic_len - 4
    if footer_len_off < 8 { return }
    footer_len := int(_rd_u32(data, footer_len_off))
    footer_start := footer_len_off - footer_len
    if footer_start < 8 || footer_len <= 0 { return }
    footer_fb := data[footer_start : footer_start+footer_len]

    // Parse footer root → Footer table
    if len(footer_fb) < 8 { return }
    ft_root := int(_rd_u32(footer_fb, 0))
    if !_fits(footer_fb, ft_root, 4) { return }
    ft_soff := _rd_i32(footer_fb, ft_root)
    ft_vt   := ft_root - int(ft_soff)
    if !_fits(footer_fb, ft_vt, 4) { return }

    // Footer vtable: [version@0, schema@1, dictionaries@2, record_batches@3].
    // Returns a field's offset-within-table, or 0 (absent / out of range).
    ft_get := proc(fb: []u8, vt, idx: int) -> int {
        if !_fits(fb, vt, 2) { return 0 }
        nf := (int(_rd_i16(fb, vt)) - 4) / 2
        if idx < 0 || idx >= nf { return 0 }
        p := vt + 4 + idx*2
        if !_fits(fb, p, 2) { return 0 }
        return int(_rd_i16(fb, p))
    }

    // Schema is at vtable index 1 (index 0 is version i16).
    schema_fo := ft_get(footer_fb, ft_vt, 1)
    if schema_fo == 0 { return }
    if !_fits(footer_fb, ft_root + schema_fo, 4) { return }
    schema_tbl := ft_root + schema_fo + int(_rd_i32(footer_fb, ft_root+schema_fo))

    schema = new(Schema, allocator)
    if !_ipc_read_schema_table(footer_fb, schema_tbl, schema, allocator) {
        free(schema, allocator); schema = nil; return
    }

    // Record batch blocks are at vtable index 3.
    rb_fo := ft_get(footer_fb, ft_vt, 3)
    batch_list := make([dynamic]Record_Batch, 0, 8, allocator)

    if rb_fo > 0 && _fits(footer_fb, ft_root + rb_fo, 4) {
        bv_ref_pos := ft_root + rb_fo
        bv_ref     := _rd_i32(footer_fb, bv_ref_pos)
        bv         := bv_ref_pos + int(bv_ref)
        if _fits(footer_fb, bv, 4) {
            n_blocks := int(_rd_u32(footer_fb, bv))
            for i in 0..<n_blocks {
                // Block: {offset:i64(8), metaDataLength:i32(4), pad:4, bodyLength:i64(8)} = 24 bytes
                bp := bv + 4 + i * 24
                if !_fits(footer_fb, bp, 24) { break }
                blk_off  := int(_rd_i64(footer_fb, bp))
                blk_meta := int(_rd_i32(footer_fb, bp+8))
                blk_body := int(_rd_i64(footer_fb, bp+16))

                // Message at blk_off: [continuation:4][size:4][flatbuffer][pad][body].
                // metaDataLength spans the whole metadata region (prefix included);
                // the body begins blk_meta bytes after blk_off.
                if blk_off < 0 || blk_meta < 8 || blk_body < 0 { break }
                if blk_off > n - blk_meta { break }     // overflow-safe: blk_off+blk_meta <= n
                body_start := blk_off + blk_meta
                if body_start > n - blk_body { break }   // body within file
                meta      := data[blk_off+8 : blk_off+blk_meta]
                body_end  := body_start + blk_body

                batch, batch_ok := _ipc_decode_record_batch(meta, schema, data[body_start:body_end], allocator, validate)
                if batch_ok { append(&batch_list, batch) }
            }
        }
    }

    batches = batch_list[:]
    // Hand the backing block to the first batch (its columns are views into it);
    // record_batch_free releases it once.  Other batches reference it but never
    // free it, and view buffers are no-ops at free time, so free order is safe.
    if len(batches) > 0 {
        batches[0]._owned_backing = data
        batches[0]._backing_free  = release   // munmap, or nil → heap delete
        keep_data = true
    }
    ok = true
    return
}

// ── Stream format ─────────────────────────────────────────────────────────────
//
// The IPC stream format is the same encapsulated messages as the file format
// but without the "ARROW1" magic or the trailing footer/Block index: just a
// Schema message, the RecordBatch messages, and an end-of-stream marker.  It is
// read strictly sequentially (no random batch access), which suits sockets,
// pipes, and stdout.

// Write a schema + batches as an Arrow IPC stream.
ipc_write_stream :: proc(path: string, schema: ^Schema, batches: []Record_Batch) -> bool {
    f, err := os.open(path, {.Write, .Create, .Trunc}, os.Permissions_Default_File)
    if err != nil { return false }
    defer os.close(f)

    offset := i64(0)

    schema_meta := _ipc_encode_schema(schema)
    defer delete(schema_meta)
    _ipc_write_message(f, schema_meta, nil, &offset)

    for i in 0..<len(batches) {
        body_descs := make([dynamic]IPC_Body_Buffer, 0, 32)
        defer delete(body_descs)
        rb_meta := _ipc_encode_record_batch(&batches[i], &body_descs)
        defer delete(rb_meta)
        _ipc_write_message(f, rb_meta, body_descs[:], &offset)
    }

    // End-of-stream: continuation marker followed by a zero metadata length.
    _ipc_write_u32(f, 0xFFFF_FFFF)
    _ipc_write_u32(f, 0)
    return true
}

// Read an Arrow IPC stream.  The caller owns the returned schema and batches
// exactly as with ipc_read_file (free batches, then schema_free + free(schema)).
// See ipc_read_file for `validate` / `use_mmap` (use_mmap=false is the
// truncation-safe copy mode).
ipc_read_stream :: proc(path: string, allocator := context.allocator, validate := true, use_mmap := true) -> (schema: ^Schema, batches: []Record_Batch, ok: bool) {
    data: []u8
    release: proc(_: []u8)
    if use_mmap {
        d, unmap, mapped := _ipc_map_file(path)
        if mapped { data = d; release = unmap }
    }
    if data == nil {
        d, read_err := os.read_entire_file(path, allocator)
        if read_err != nil { return }
        data    = d
        release = nil
    }
    keep_data := false
    defer if !keep_data {
        if release != nil { release(data) }
        else              { delete(data, allocator) }
    }

    n := len(data)
    schema = new(Schema, allocator)
    schema_ok := false
    batch_list := make([dynamic]Record_Batch, 0, 8, allocator)

    pos := 0
    for pos + 8 <= n {
        cont := _rd_i32(data, pos)
        meta_size:  int
        meta_start: int
        if cont == IPC_CONTINUATION {        // 0xFFFFFFFF + size
            meta_size  = int(_rd_i32(data, pos+4))
            meta_start = pos + 8
        } else {                              // legacy framing: bare size
            meta_size  = int(cont)
            meta_start = pos + 4
        }
        if meta_size <= 0 { break }           // end-of-stream
        if meta_start + meta_size > n { break }

        meta := data[meta_start : meta_start+meta_size]
        header_type, body_length := _ipc_parse_message_header(meta)
        body_start := (meta_start + meta_size + 7) & ~int(7)

        switch header_type {
        case _IPC_HEADER_SCHEMA:
            if _ipc_decode_schema(meta, schema, allocator) { schema_ok = true }
            pos = body_start  // schema message carries no body
        case _IPC_HEADER_RECORD_BATCH:
            body_padded := (int(body_length) + 7) & ~int(7)
            body_end := body_start + int(body_length)
            if body_end > n { body_end = n }
            batch, bok := _ipc_decode_record_batch(meta, schema, data[body_start:body_end], allocator, validate)
            if bok { append(&batch_list, batch) }
            pos = body_start + body_padded
        case:
            // Unknown/dictionary message — cannot advance safely.
            pos = n
        }
    }

    if !schema_ok {
        free(schema, allocator)
        delete(batch_list)
        schema = nil
        return
    }

    batches = batch_list[:]
    if len(batches) > 0 {
        batches[0]._owned_backing = data
        batches[0]._backing_free  = release   // munmap, or nil → heap delete
        keep_data = true
    }
    ok = true
    return
}

// Non-owning Buffer view into `body` (a sub-slice of the file block).  The
// zero allocator marks it as not-owned, so buffer_free leaves it alone — the
// memory is released once when the owning batch frees its backing block.
_ipc_view_buffer :: proc(body: []u8, off, length: int) -> Buffer {
    if length <= 0 || off < 0 || off+length > len(body) { return {} }
    return Buffer{ data = cast([^]u8)&body[off], size = length, capacity = length }
}

// A4: validate a variable-length column's offset buffer before it is trusted by
// array_get_string (which slices the data buffer at offsets[i]..offsets[i+1]
// through an unchecked multi-pointer). Offsets must start at 0, be monotonically
// non-decreasing, and stay within the data buffer.
// A1/A3: the data buffer of a fixed-width / bool column must actually hold
// col_len elements, or array_get would read past the view (overflow-safe).
_ipc_fixed_ok :: proc(dt: DataType, col_len: int, data_buf: Buffer) -> bool {
    #partial switch _ in dt {
    case Bool_Type: return data_buf.size >= bitmap_byte_count(col_len)
    }
    w := type_byte_width(dt)
    return w > 0 && col_len <= data_buf.size / w
}

_ipc_offsets_valid :: proc(off_buf, data_buf: Buffer, col_len, off_w: int) -> bool {
    if col_len == 0 { return true }
    // Overflow-safe "need col_len+1 entries of off_w bytes": col_len < size/off_w.
    if off_buf.data == nil || col_len < 0 || off_w <= 0 || col_len >= off_buf.size / off_w { return false }
    data_len := i64(data_buf.size)
    if off_w == size_of(i32) {
        o := cast([^]i32)off_buf.data
        if o[0] != 0 { return false }
        prev: i32 = 0
        for k in 1..=col_len {
            cur := o[k]
            if cur < prev || i64(cur) > data_len { return false }
            prev = cur
        }
    } else {
        o := cast([^]i64)off_buf.data
        if o[0] != 0 { return false }
        prev: i64 = 0
        for k in 1..=col_len {
            cur := o[k]
            if cur < prev || cur > data_len { return false }
            prev = cur
        }
    }
    return true
}

// A1/A3: true iff a `size`-byte read at `pos` lies within `data`. Written as
// `pos <= len - size` (not `pos + size <= len`) so it can't overflow for a
// hostile `pos`, and rejects negative pos/size.
_fits :: #force_inline proc "contextless" (data: []u8, pos, size: int) -> bool {
    return pos >= 0 && size >= 0 && pos <= len(data) - size
}

// A1: the structural-metadata readers are bounds-safe by construction — an
// out-of-range position (computed from attacker-controlled offsets) yields 0
// rather than reading out of bounds. They only ever read the small FlatBuffer
// metadata, never bulk data, so the per-read check costs nothing measurable.
_rd_i32 :: #force_inline proc "contextless" (data: []u8, pos: int) -> i32 {
    if !_fits(data, pos, 4) { return 0 }
    return i32(data[pos]) | i32(data[pos+1])<<8 | i32(data[pos+2])<<16 | i32(data[pos+3])<<24
}
_rd_u32 :: #force_inline proc "contextless" (data: []u8, pos: int) -> u32 { return u32(_rd_i32(data, pos)) }
_rd_i16 :: #force_inline proc "contextless" (data: []u8, pos: int) -> i16 {
    if !_fits(data, pos, 2) { return 0 }
    return i16(data[pos]) | i16(data[pos+1])<<8
}
_rd_i64 :: #force_inline proc "contextless" (data: []u8, pos: int) -> i64 {
    if !_fits(data, pos, 8) { return 0 }
    lo := u32(_rd_i32(data, pos)); hi := u32(_rd_i32(data, pos+4))
    return i64(u64(lo) | u64(hi)<<32)
}

// Bounds-safe single-byte read (for direct metadata byte fields).
_rd_u8 :: #force_inline proc "contextless" (data: []u8, pos: int) -> u8 {
    if !_fits(data, pos, 1) { return 0 }
    return data[pos]
}

// Returns (header_type, body_length) from a Message FlatBuffer.
_ipc_parse_message_header :: proc(meta: []u8) -> (header_type: u8, body_length: i64) {
    if len(meta) < 8 { return }
    root_off := _rd_u32(meta, 0)
    if int(root_off)+4 > len(meta) { return }
    soff := _rd_i32(meta, int(root_off))
    vt := int(root_off) - int(soff)
    if vt < 0 || vt+6 > len(meta) { return }
    vt_size := _rd_i16(meta, vt)
    n_fields := (int(vt_size) - 4) / 2
    if n_fields < 3 { return }
    f1 := int(_rd_i16(meta, vt+6))  // header_type offset
    f2 := int(_rd_i16(meta, vt+8))  // header offset
    if f1 > 0 && int(root_off)+f1 < len(meta) {
        header_type = meta[int(root_off)+f1]
    }
    // body_length (field 3 in some versions): look for it
    if n_fields >= 4 {
        f3 := int(_rd_i16(meta, vt+10))
        if f3 > 0 && int(root_off)+f3+8 <= len(meta) {
            body_length = _rd_i64(meta, int(root_off)+f3)
        }
    }
    _ = f2
    return
}

_ipc_decode_schema :: proc(meta: []u8, schema: ^Schema, allocator: mem.Allocator) -> bool {
    if len(meta) < 8 { return false }
    root_off := _rd_u32(meta, 0)
    tbl := int(root_off)
    soff := _rd_i32(meta, tbl)
    vt := tbl - int(soff)
    vt_size := _rd_i16(meta, vt)
    n_fields := (int(vt_size) - 4) / 2
    if n_fields < 3 { return false }

    // header field = field[2] in Message, points to Schema table
    f2 := int(_rd_i16(meta, vt+8))
    if f2 == 0 { return false }
    header_ref_pos := tbl + f2
    header_ref := _rd_i32(meta, header_ref_pos)
    schema_tbl := header_ref_pos + int(header_ref)

    return _ipc_read_schema_table(meta, schema_tbl, schema, allocator)
}

_ipc_read_schema_table :: proc(data: []u8, tbl: int, schema: ^Schema, allocator: mem.Allocator) -> bool {
    if !_fits(data, tbl, 4) { return false }
    soff := _rd_i32(data, tbl)
    vt := tbl - int(soff)
    if !_fits(data, vt, 8) { return false }
    vt_size := _rd_i16(data, vt)
    n := (int(vt_size) - 4) / 2
    if n < 2 { return false }

    // fields ref is field[1] in Schema
    f1 := int(_rd_i16(data, vt+6))
    if f1 == 0 { return false }
    fv_ref_pos := tbl + f1
    fv_ref := _rd_i32(data, fv_ref_pos)
    fv := fv_ref_pos + int(fv_ref)
    if !_fits(data, fv, 4) { return false }
    n_fields := int(_rd_u32(data, fv))
    // A2/A3: a field ref is 4 bytes, so the count can't exceed the bytes left.
    if n_fields < 0 || fv < 0 || n_fields > (len(data) - fv - 4) / 4 { return false }

    fields := make([]Field, n_fields, allocator)
    for i in 0..<n_fields {
        elem_pos := fv + 4 + i * 4
        if !_fits(data, elem_pos, 4) { continue }
        fref := _rd_i32(data, elem_pos)
        ftbl := elem_pos + int(fref)
        fields[i] = _ipc_read_field(data, ftbl, allocator)
    }
    schema.fields    = fields
    schema.allocator = allocator
    return true
}

_ipc_read_field :: proc(data: []u8, tbl: int, allocator: mem.Allocator) -> Field {
    f: Field
    if !_fits(data, tbl, 4) { return f }
    soff := _rd_i32(data, tbl)
    vt := tbl - int(soff)
    if !_fits(data, vt, 4) { return f }
    vt_size := _rd_i16(data, vt)
    n := (int(vt_size) - 4) / 2

    get_fo := proc(data: []u8, vt, idx: int) -> int {
        if idx+1 > (int(_rd_i16(data, vt))-4)/2 { return 0 }
        return int(_rd_i16(data, vt+4+idx*2))
    }

    if fo := get_fo(data, vt, 0); fo > 0 {
        name_ref_pos := tbl + fo
        name_ref     := _rd_i32(data, name_ref_pos)
        name_tbl     := name_ref_pos + int(name_ref)
        if _fits(data, name_tbl, 4) {
            name_len := int(_rd_u32(data, name_tbl))
            if name_len >= 0 && _fits(data, name_tbl+4, name_len) {
                name_bytes := make([]u8, name_len, allocator)
                copy(name_bytes, data[name_tbl+4:name_tbl+4+name_len])
                f.name = string(name_bytes)
            }
        }
    }
    if fo := get_fo(data, vt, 1); fo > 0 {
        f.nullable = _rd_u8(data, tbl+fo) != 0
    } else {
        f.nullable = true
    }
    type_disc := u8(0)
    if fo := get_fo(data, vt, 2); fo > 0 {
        type_disc = _rd_u8(data, tbl+fo)
    }
    _ = n

    type_tbl := 0
    if fo := get_fo(data, vt, 3); fo > 0 {
        tr_pos := tbl + fo
        tr := _rd_i32(data, tr_pos)
        type_tbl = tr_pos + int(tr)
    }

    f.type = _ipc_disc_to_datatype(data, type_disc, type_tbl)
    return f
}

_ipc_disc_to_datatype :: proc(data: []u8, disc: u8, tbl: int) -> DataType {
    switch disc {
    case _IPC_TYPE_BOOL:        return Bool_Type{}
    case _IPC_TYPE_UTF8:        return String_Type{}
    case _IPC_TYPE_LARGEUTF8:   return Large_String_Type{}
    case _IPC_TYPE_BINARY:      return Binary_Type{}
    case _IPC_TYPE_LARGEBINARY: return Large_Binary_Type{}
    case _IPC_TYPE_INT:
        if _fits(data, tbl, 4) {
            soff := _rd_i32(data, tbl)
            vt   := tbl - int(soff)
            bw_off := int(_rd_i16(data, vt+4))  // bitWidth   (reads are bounds-safe)
            sg_off := int(_rd_i16(data, vt+6))  // isSigned
            bw := i32(0); sg := true
            if bw_off > 0 { bw = _rd_i32(data, tbl+bw_off) }
            if sg_off > 0 { sg = _rd_u8(data, tbl+sg_off) != 0 }
            if sg {
                switch bw {
                case 8:  return Int8_Type{}
                case 16: return Int16_Type{}
                case 32: return Int32_Type{}
                case 64: return Int64_Type{}
                }
            } else {
                switch bw {
                case 8:  return UInt8_Type{}
                case 16: return UInt16_Type{}
                case 32: return UInt32_Type{}
                case 64: return UInt64_Type{}
                }
            }
        }
        return Int64_Type{}
    case _IPC_TYPE_FLOATINGPT:
        if _fits(data, tbl, 4) {
            soff := _rd_i32(data, tbl)
            vt := tbl - int(soff)
            pr_off := int(_rd_i16(data, vt+4))
            pr := i16(0)
            if pr_off > 0 { pr = _rd_i16(data, tbl+pr_off) }
            if pr == 1 { return Float32_Type{} }
            return Float64_Type{}
        }
        return Float64_Type{}
    case _IPC_TYPE_DATE:       return Date64_Type{}
    case _IPC_TYPE_TIME:       return Time64_Type{}
    case _IPC_TYPE_TIMESTAMP:  return Timestamp_Type{}
    case _IPC_TYPE_DURATION:   return Duration_Type{}
    }
    return Null_Type{}
}

_ipc_decode_record_batch :: proc(
    meta:   []u8,
    schema: ^Schema,
    body:   []u8,
    allocator: mem.Allocator,
    validate: bool = true,
) -> (batch: Record_Batch, ok: bool) {
    if len(meta) < 8 { return }
    root_off := _rd_u32(meta, 0)
    tbl := int(root_off)
    soff := _rd_i32(meta, tbl)
    vt := tbl - int(soff)
    vt_size := _rd_i16(meta, vt)
    n_vf := (int(vt_size) - 4) / 2

    get_fo := proc(data: []u8, vt, idx: int) -> int {
        n := (int(_rd_i16(data, vt))-4)/2
        if idx >= n { return 0 }
        return int(_rd_i16(data, vt+4+idx*2))
    }

    // header_type is at Message.field[1]
    header_type := u8(0)
    if fo := get_fo(meta, vt, 1); fo > 0 { header_type = _rd_u8(meta, tbl+fo) }
    // Find Message.header (field[2])
    if fo := get_fo(meta, vt, 2); fo == 0 { return }
    else {
        hpos := tbl + fo
        href := _rd_i32(meta, hpos)
        rb_tbl := hpos + int(href)

        // rb_tbl is the RecordBatch table
        rb_soff := _rd_i32(meta, rb_tbl)
        rb_vt   := rb_tbl - int(rb_soff)
        _ = n_vf

        // RecordBatch fields: length(i64 at field[0]), nodes(ref at field[1]), buffers(ref at field[2])
        n_cols := len(schema.fields)

        length := i64(0)
        if fo2 := get_fo(meta, rb_vt, 0); fo2 > 0 {
            length = _rd_i64(meta, rb_tbl+fo2)
        }

        // nodes
        n_nodes := 0
        node_lengths    := make([dynamic]i64, 0, n_cols, context.temp_allocator)
        node_nullcounts := make([dynamic]i64, 0, n_cols, context.temp_allocator)
        if fo2 := get_fo(meta, rb_vt, 1); fo2 > 0 {
            nv_pos := rb_tbl + fo2
            nv_ref := _rd_i32(meta, nv_pos)
            nv := nv_pos + int(nv_ref)
            n_nodes = int(_rd_u32(meta, nv))
            // A2/A3: reject counts that can't fit (overflow-safe: each node 16 bytes).
            if n_nodes < 0 || nv < 0 || nv+4 > len(meta) || n_nodes > (len(meta)-nv-4)/16 { return }
            for i in 0..<n_nodes {
                p := nv + 4 + i * 16
                append(&node_lengths,    _rd_i64(meta, p))
                append(&node_nullcounts, _rd_i64(meta, p+8))
            }
        }

        // buffers
        n_bufs := 0
        buf_offsets := make([dynamic]i64, 0, n_cols*3, context.temp_allocator)
        buf_lengths := make([dynamic]i64, 0, n_cols*3, context.temp_allocator)
        if fo2 := get_fo(meta, rb_vt, 2); fo2 > 0 {
            bv_pos := rb_tbl + fo2
            bv_ref := _rd_i32(meta, bv_pos)
            bv := bv_pos + int(bv_ref)
            n_bufs = int(_rd_u32(meta, bv))
            // A2/A3: reject counts that can't fit (overflow-safe: each buffer 16 bytes).
            if n_bufs < 0 || bv < 0 || bv+4 > len(meta) || n_bufs > (len(meta)-bv-4)/16 { return }
            for i in 0..<n_bufs {
                p := bv + 4 + i * 16
                append(&buf_offsets, _rd_i64(meta, p))
                append(&buf_lengths, _rd_i64(meta, p+8))
            }
        }

        // Build columns — buffers are zero-copy views into `body` (which lives
        // in the file block the batch takes ownership of).
        columns := make([]Array, n_cols, allocator)
        buf_idx := 0
        for ci in 0..<min(n_cols, n_nodes) {
            col_len    := int(node_lengths[ci])
            null_count := int(node_nullcounts[ci])
            dt := schema.fields[ci].type

            if validate && (col_len < 0 || null_count < 0 || null_count > col_len) {
                delete(columns, allocator); return
            }

            // validity bitmap buffer
            var_buf: Buffer
            if buf_idx < n_bufs && buf_lengths[buf_idx] > 0 && null_count > 0 {
                var_buf = _ipc_view_buffer(body, int(buf_offsets[buf_idx]), int(buf_lengths[buf_idx]))
            }
            buf_idx += 1

            // A1/A3: a validity bitmap must cover col_len bits.
            if validate && null_count > 0 && (var_buf.data == nil || var_buf.size < bitmap_byte_count(col_len)) {
                delete(columns, allocator); return
            }

            switch _ in dt {
            case String_Type, Large_String_Type, Binary_Type, Large_Binary_Type:
                off_w := size_of(i32)
                #partial switch _ in dt {
                case Large_String_Type, Large_Binary_Type: off_w = size_of(i64)
                }
                off_len := (col_len + 1) * off_w
                off_buf: Buffer
                if buf_idx < n_bufs {
                    off_buf = _ipc_view_buffer(body, int(buf_offsets[buf_idx]), off_len)
                }
                buf_idx += 1
                data_buf: Buffer
                if buf_idx < n_bufs {
                    data_buf = _ipc_view_buffer(body, int(buf_offsets[buf_idx]), int(buf_lengths[buf_idx]))
                }
                buf_idx += 1
                // A4: the offset buffer is attacker-controlled and is later used
                // to slice the (unchecked) data buffer in array_get_string — so
                // verify it is monotonic and in-bounds before handing it out.
                if validate && !_ipc_offsets_valid(off_buf, data_buf, col_len, off_w) {
                    delete(columns, allocator)
                    return
                }
                columns[ci] = Array{type=dt, length=col_len, null_count=null_count, buffers={var_buf, off_buf, data_buf}}
            case Null_Type:
                columns[ci] = Array{type=dt, length=col_len}
            case Bool_Type,
                 Int8_Type, Int16_Type, Int32_Type, Int64_Type,
                 UInt8_Type, UInt16_Type, UInt32_Type, UInt64_Type,
                 Float32_Type, Float64_Type,
                 Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
                // fixed-width or bool
                data_buf: Buffer
                if buf_idx < n_bufs {
                    data_buf = _ipc_view_buffer(body, int(buf_offsets[buf_idx]), int(buf_lengths[buf_idx]))
                }
                buf_idx += 1
                if validate && !_ipc_fixed_ok(dt, col_len, data_buf) {
                    delete(columns, allocator); return
                }
                columns[ci] = Array{type=dt, length=col_len, null_count=null_count, buffers={var_buf, data_buf, {}}}
            }
        }

        cols_copy := make([]Array, n_cols, allocator)
        copy(cols_copy, columns)
        delete(columns, allocator)
        batch = Record_Batch{schema=schema, columns=cols_copy, length=int(length), allocator=allocator}
        ok = true
        _ = header_type
    }
    return
}
