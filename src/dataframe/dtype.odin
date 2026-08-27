package dataframe

import "core:mem"

// Field pairs a column name with its element type. The type is an Odin
// typeid — the physical type is the type (DESIGN.md §2.1). A distinct
// logical-type layer (Date, Categorical, Enum, …) is deferred until such
// types exist; see DESIGN.md §2.1.
Field :: struct {
	name:  string,
	dtype: typeid,
}

// Schema is an ordered list of fields. A Schema owns its field slice and
// must be destroyed with schema_destroy when built via schema_create.
Schema :: struct {
	fields: []Field,
	alloc:  mem.Allocator,
}

// schema_create builds an owned copy of fields. The returned Schema owns
// its backing slice and must be released with schema_destroy.
schema_create :: proc(fields: []Field, allocator := context.allocator) -> (Schema, Error) {
	out := make([]Field, len(fields), allocator)
	if out == nil && len(fields) != 0 {
		return {}, .Allocator_Failure
	}
	copy(out, fields)
	return Schema{fields = out, alloc = allocator}, .None
}

// schema_destroy releases the field slice owned by schema. The Field name
// strings are borrowed from the source columns and are not freed.
schema_destroy :: proc(schema: ^Schema) {
	if schema.fields != nil {
		delete(schema.fields, schema.alloc)
	}
	schema^ = {}
}

// schema_len returns the number of fields in schema.
schema_len :: proc(schema: ^Schema) -> int {
	return len(schema.fields)
}

// schema_field_at returns the field at index i, or an error when out of range.
schema_field_at :: proc(schema: ^Schema, i: int) -> (Field, Error) {
	if i < 0 || i >= len(schema.fields) {
		return {}, .Out_Of_Bounds
	}
	return schema.fields[i], .None
}

// schema_has_column reports whether a field with the given name exists.
schema_has_column :: proc(schema: ^Schema, name: string) -> bool {
	for field in schema.fields {
		if field.name == name {
			return true
		}
	}
	return false
}

// schema_index_of returns the index of the field named name, or an error
// when it does not exist.
schema_index_of :: proc(schema: ^Schema, name: string) -> (int, Error) {
	for field, i in schema.fields {
		if field.name == name {
			return i, .None
		}
	}
	return -1, .Column_Not_Found
}
