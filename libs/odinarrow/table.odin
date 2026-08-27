package odinarrow

import "core:mem"

// Chunked_Array is a logical column split across one or more Array chunks.
// It does NOT own the individual chunk Arrays — the source Record_Batches do.
// It only owns the `chunks` slice itself.
Chunked_Array :: struct {
	type:      DataType,
	chunks:    []Array,  // borrowed from source batches
	length:    int,
	allocator: mem.Allocator,
}

// Table borrows chunk data from its source Record_Batches.
// Those batches must outlive the table.
Table :: struct {
	schema:    ^Schema,
	columns:   []Chunked_Array,
	length:    int,
	allocator: mem.Allocator,
}

// Build a Table by stacking Record_Batches. All batches must share the same
// schema (field count and types must match). Chunk Arrays are borrowed.
table_from_record_batches :: proc(
	batches: []Record_Batch,
	allocator := context.allocator,
) -> (t: Table, ok: bool) {
	if len(batches) == 0 do return {}, false

	schema := batches[0].schema
	n_cols := len(schema.fields)
	for i in 1..<len(batches) {
		if len(batches[i].schema.fields) != n_cols do return {}, false
	}

	columns, err := make([]Chunked_Array, n_cols, allocator)
	if err != nil do return {}, false

	for col_i in 0..<n_cols {
		chunks, cerr := make([]Array, len(batches), allocator)
		if cerr != nil do return {}, false
		total := 0
		for bi in 0..<len(batches) {
			chunks[bi] = batches[bi].columns[col_i]
			total += batches[bi].columns[col_i].length
		}
		columns[col_i] = Chunked_Array{
			type      = schema.fields[col_i].type,
			chunks    = chunks,
			length    = total,
			allocator = allocator,
		}
	}

	total_rows := 0
	for i in 0..<len(batches) { total_rows += batches[i].length }

	return Table{
		schema    = schema,
		columns   = columns,
		length    = total_rows,
		allocator = allocator,
	}, true
}

// Free only the allocated slices — does NOT free the chunk Array data.
table_free :: proc(t: ^Table) {
	for i in 0..<len(t.columns) {
		delete(t.columns[i].chunks, t.columns[i].allocator)
	}
	delete(t.columns, t.allocator)
	t^ = {}
}

// Look up a column by name.
table_column :: proc(t: ^Table, name: string) -> (col: ^Chunked_Array, ok: bool) {
	idx, found := schema_field_index(t.schema, name)
	if !found do return nil, false
	return &t.columns[idx], true
}

table_column_at :: proc(t: ^Table, i: int) -> ^Chunked_Array {
	return &t.columns[i]
}

// Get element i across all chunks of a ChunkedArray.
chunked_array_get :: proc(ca: ^Chunked_Array, i: int, $T: typeid) -> T {
	pos := i
	for ci in 0..<len(ca.chunks) {
		if pos < ca.chunks[ci].length {
			return array_get(&ca.chunks[ci], pos, T)
		}
		pos -= ca.chunks[ci].length
	}
	panic("chunked_array_get: index out of bounds")
}

chunked_array_is_null :: proc(ca: ^Chunked_Array, i: int) -> bool {
	pos := i
	for ci in 0..<len(ca.chunks) {
		if pos < ca.chunks[ci].length {
			return array_is_null(&ca.chunks[ci], pos)
		}
		pos -= ca.chunks[ci].length
	}
	panic("chunked_array_is_null: index out of bounds")
}
