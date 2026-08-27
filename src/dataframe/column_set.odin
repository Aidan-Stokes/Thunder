package dataframe

import "core:mem"
import "base:runtime"

// ColumnMeta holds cold metadata for a column. Not accessed on hot paths.
// dtype and elem_size live in ColumnSet's hot arrays (dtypes, sizes).
ColumnMeta :: struct {
	align:           int,
	alloc:           mem.Allocator,
	payload:         rawptr,
	payload_size:    int,
	categories:      []string,
	categorical_kind: Categorical_Kind,
	inner_dtype:     typeid,
	inner_valid:     []u64,
}

// ColumnSet stores columns in SoA layout: separate arrays for each field
// so tight loops that touch only one field stride over sizeof(field) per
// column instead of sizeof(Column).
ColumnSet :: struct {
	names:  [dynamic]string,
	dtypes: [dynamic]typeid,
	sizes:  [dynamic]int,
	rows:   [dynamic]int,
	data:   [dynamic]rawptr,
	valids: [dynamic][]u64,
	metas:  [dynamic]ColumnMeta,
	count:  int,
	alloc:  mem.Allocator,
}

column_set_create :: proc(alloc := context.allocator) -> ColumnSet {
	return ColumnSet{
		names  = make([dynamic]string, alloc),
		dtypes = make([dynamic]typeid, alloc),
		sizes  = make([dynamic]int, alloc),
		rows   = make([dynamic]int, alloc),
		data   = make([dynamic]rawptr, alloc),
		valids = make([dynamic][]u64, alloc),
		metas  = make([dynamic]ColumnMeta, alloc),
		count  = 0,
		alloc  = alloc,
	}
}

column_set_destroy :: proc(cs: ^ColumnSet) {
	for i in 0 ..< cs.count {
		column_set_destroy_entry(cs, i)
	}
	delete(cs.names)
	delete(cs.dtypes)
	delete(cs.sizes)
	delete(cs.rows)
	delete(cs.data)
	delete(cs.valids)
	delete(cs.metas)
	cs^ = {}
}

@(private)
column_set_destroy_entry :: proc(cs: ^ColumnSet, i: int) {
	meta := &cs.metas[i]
	if cs.data[i] != nil {
		mem.free_with_size(cs.data[i], cs.sizes[i] * cs.rows[i], meta.alloc)
	}
	if cs.valids[i] != nil {
		delete(cs.valids[i], meta.alloc)
	}
	if meta.payload != nil {
		mem.free_with_size(meta.payload, meta.payload_size, meta.alloc)
	}
	if meta.categories != nil {
		for s in meta.categories {
			delete_string(s, meta.alloc)
		}
		delete(meta.categories, meta.alloc)
	}
	if meta.inner_valid != nil {
		delete(meta.inner_valid, meta.alloc)
	}
	if cs.names[i] != "" {
		delete_string(cs.names[i], meta.alloc)
	}
}

// column_set_add transfers ownership of a Column into the ColumnSet.
// The source Column is zeroed on success.
column_set_add :: proc(cs: ^ColumnSet, col: ^Column) -> Error {
	if col.name == "" {
		return .Column_Name_Empty
	}
	for i in 0 ..< cs.count {
		if cs.names[i] == col.name {
			return .Duplicate_Column_Name
		}
	}
	if cs.count != 0 && cs.rows[0] != col.count {
		return .Length_Mismatch
	}

	meta := ColumnMeta{
		align           = col.align,
		alloc           = col.alloc,
		payload         = col.payload,
		payload_size    = col.payload_size,
		categories      = col.categories,
		categorical_kind = col.categorical_kind,
		inner_dtype     = col.inner_dtype,
		inner_valid     = col.inner_valid,
	}

	append(&cs.names, col.name)
	append(&cs.dtypes, col.dtype)
	append(&cs.sizes, col.elem_size)
	append(&cs.rows, col.count)
	append(&cs.data, col.data)
	append(&cs.valids, col.valid)
	append(&cs.metas, meta)
	cs.count += 1

	col^ = {}
	return .None
}

// column_set_get returns the index and true if a column with the given name exists.
column_set_get :: proc(cs: ^ColumnSet, name: string) -> (int, bool) {
	for i in 0 ..< cs.count {
		if cs.names[i] == name {
			return i, true
		}
	}
	return -1, false
}

// column_set_remove destroys the column at index i and removes it from the set.
column_set_remove :: proc(cs: ^ColumnSet, i: int) -> Error {
	if i < 0 || i >= cs.count {
		return .Out_Of_Bounds
	}
	column_set_destroy_entry(cs, i)
	ordered_remove(&cs.names, i)
	ordered_remove(&cs.dtypes, i)
	ordered_remove(&cs.sizes, i)
	ordered_remove(&cs.rows, i)
	ordered_remove(&cs.data, i)
	ordered_remove(&cs.valids, i)
	ordered_remove(&cs.metas, i)
	cs.count -= 1
	return .None
}

// column_set_rename replaces the column name at index i with an owned copy.
column_set_rename :: proc(cs: ^ColumnSet, i: int, new_name: string) -> Error {
	if new_name == "" {
		return .Column_Name_Empty
	}
	if i < 0 || i >= cs.count {
		return .Out_Of_Bounds
	}
	for j in 0 ..< cs.count {
		if j != i && cs.names[j] == new_name {
			return .Duplicate_Column_Name
		}
	}
	meta := &cs.metas[i]
	new_copy := clone_name(meta.alloc, new_name) or_return
	if cs.names[i] != "" {
		delete_string(cs.names[i], meta.alloc)
	}
	cs.names[i] = new_copy
	return .None
}

// column_set_copy returns a deep, independent copy of the ColumnSet.
column_set_copy :: proc(cs: ^ColumnSet, allocator := context.allocator) -> (out: ColumnSet, err: Error) {
	out = column_set_create(allocator)
	for i in 0 ..< cs.count {
		meta := &cs.metas[i]
		col := Column{
			name            = cs.names[i],
			dtype           = cs.dtypes[i],
			elem_size       = cs.sizes[i],
			align           = meta.align,
			count           = cs.rows[i],
			data            = cs.data[i],
			valid           = cs.valids[i],
			alloc           = meta.alloc,
			payload         = meta.payload,
			payload_size    = meta.payload_size,
			categories      = meta.categories,
			categorical_kind = meta.categorical_kind,
			inner_dtype     = meta.inner_dtype,
			inner_valid     = meta.inner_valid,
		}
		cc, c_err := column_copy(&col, allocator)
		if c_err != .None {
			column_set_destroy(&out)
			return {}, c_err
		}
		if add_err := column_set_add(&out, &cc); add_err != .None {
			column_destroy(&cc)
			column_set_destroy(&out)
			return {}, add_err
		}
	}
	return out, .None
}

// column_set_to_column reconstructs a Column from the ColumnSet entry at index i.
// The returned Column borrows data/valid from the ColumnSet and owns nothing.
column_set_to_column :: proc(cs: ^ColumnSet, i: int) -> Column {
	meta := &cs.metas[i]
	return Column{
		name            = cs.names[i],
		dtype           = cs.dtypes[i],
		elem_size       = cs.sizes[i],
		align           = meta.align,
		count           = cs.rows[i],
		data            = cs.data[i],
		valid           = cs.valids[i],
		alloc           = meta.alloc,
		payload         = meta.payload,
		payload_size    = meta.payload_size,
		categories      = meta.categories,
		categorical_kind = meta.categorical_kind,
		inner_dtype     = meta.inner_dtype,
		inner_valid     = meta.inner_valid,
	}
}

cs_typed_view :: proc(cs: ^ColumnSet, i: int, $T: typeid) -> []T {
	return transmute([]T)(runtime.Raw_Slice{data = cs.data[i], len = cs.rows[i]})
}

cs_data :: proc(cs: ^ColumnSet, i: int) -> rawptr     { return cs.data[i] }
cs_dtype :: proc(cs: ^ColumnSet, i: int) -> typeid     { return cs.dtypes[i] }
cs_name :: proc(cs: ^ColumnSet, i: int) -> string      { return cs.names[i] }
