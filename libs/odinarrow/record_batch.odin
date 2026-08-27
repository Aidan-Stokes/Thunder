package odinarrow

import "core:mem"

// RecordBatch owns its column arrays. It borrows the Schema (does not free it).
// All columns must have the same length and match the schema's field count.
Record_Batch :: struct {
	schema:    ^Schema,
	columns:   []Array,
	length:    int,
	allocator: mem.Allocator,
	// Optional backing block this batch owns (e.g. the file bytes an IPC reader
	// parsed zero-copy).  When set, the column buffers are views into it and
	// record_batch_free releases it after the columns.  nil when each column
	// owns its buffers individually.
	_owned_backing: []u8,
	// Custom releaser for _owned_backing (e.g. munmap for a memory-mapped file).
	// nil → the block is heap memory, freed with `delete(_, allocator)`.
	_backing_free: proc(_: []u8),
}

// Shallow-copies the columns slice (Arrays share their underlying buffers).
// Returns ok=false if column count or lengths do not match.
record_batch_make :: proc(
	schema:  ^Schema,
	columns: []Array,
	allocator := context.allocator,
) -> (rb: Record_Batch, ok: bool) {
	if len(columns) != len(schema.fields) do return {}, false
	n := 0
	if len(columns) > 0 {
		n = columns[0].length
		for i in 1..<len(columns) {
			if columns[i].length != n do return {}, false
		}
	}
	cols, err := make([]Array, len(columns), allocator)
	if err != nil do return {}, false
	copy(cols, columns)
	return Record_Batch{
		schema    = schema,
		columns   = cols,
		length    = n,
		allocator = allocator,
	}, true
}

// Free owned column arrays and the columns slice. Does NOT free the Schema.
// View buffers (those that point into _owned_backing) are no-ops in
// array_free; the shared backing block is released once, here.
record_batch_free :: proc(rb: ^Record_Batch) {
	for i in 0..<len(rb.columns) {
		array_free(&rb.columns[i], rb.allocator)
	}
	delete(rb.columns, rb.allocator)
	if rb._owned_backing != nil {
		if rb._backing_free != nil {
			rb._backing_free(rb._owned_backing)   // e.g. munmap
		} else {
			delete(rb._owned_backing, rb.allocator)
		}
	}
	rb^ = {}
}

// Return a pointer to the column named `name`, or nil.
record_batch_column :: proc(rb: ^Record_Batch, name: string) -> ^Array {
	idx, ok := schema_field_index(rb.schema, name)
	if !ok do return nil
	return &rb.columns[idx]
}

record_batch_column_at :: proc(rb: ^Record_Batch, i: int) -> ^Array {
	return &rb.columns[i]
}
