package dataframe

import oa "../../libs/odinarrow"
import "core:mem"
import "base:runtime"

// ── Arrow IPC bridge ────────────────────────────────────────────────────────
//
// dataframe_write_arrow / dataframe_read_arrow provide Arrow IPC file I/O.
// Thin wrappers around OdinArrow translating between the dataframe's
// runtime-typed Column representation and OdinArrow's Array buffers.

// ── dtype ↔ Arrow DataType ─────────────────────────────────────────────────

arrow_type_for_dtype :: proc(dtype: typeid) -> (oa.DataType, bool) {
	switch dtype {
	case typeid_of(bool):       return oa.Bool_Type{}, true
	case typeid_of(i8):         return oa.Int8_Type{}, true
	case typeid_of(i16):        return oa.Int16_Type{}, true
	case typeid_of(i32):        return oa.Int32_Type{}, true
	case typeid_of(i64):        return oa.Int64_Type{}, true
	case typeid_of(u8):         return oa.UInt8_Type{}, true
	case typeid_of(u16):        return oa.UInt16_Type{}, true
	case typeid_of(u32):        return oa.UInt32_Type{}, true
	case typeid_of(u64):        return oa.UInt64_Type{}, true
	case typeid_of(f32):        return oa.Float32_Type{}, true
	case typeid_of(f64):        return oa.Float64_Type{}, true
	case typeid_of(string):     return oa.String_Type{}, true
	case typeid_of(Date):       return oa.Date64_Type{}, true
	case typeid_of(Time):       return oa.Time64_Type{}, true
	case typeid_of(Datetime):   return oa.Timestamp_Type{}, true
	case typeid_of(Duration):   return oa.Duration_Type{}, true
	}
	return {}, false
}

dtype_for_arrow_type :: proc(dt: oa.DataType) -> (typeid, bool) {
	switch _ in dt {
	case oa.Bool_Type:         return typeid_of(bool), true
	case oa.Int8_Type:         return typeid_of(i8), true
	case oa.Int16_Type:        return typeid_of(i16), true
	case oa.Int32_Type:        return typeid_of(i32), true
	case oa.Int64_Type:        return typeid_of(i64), true
	case oa.UInt8_Type:        return typeid_of(u8), true
	case oa.UInt16_Type:       return typeid_of(u16), true
	case oa.UInt32_Type:       return typeid_of(u32), true
	case oa.UInt64_Type:       return typeid_of(u64), true
	case oa.Float32_Type:      return typeid_of(f32), true
	case oa.Float64_Type:      return typeid_of(f64), true
	case oa.String_Type:       return typeid_of(string), true
	case oa.Date64_Type:       return typeid_of(Date), true
	case oa.Time64_Type:       return typeid_of(Time), true
	case oa.Timestamp_Type:    return typeid_of(Datetime), true
	case oa.Duration_Type:     return typeid_of(Duration), true
	case oa.Null_Type, oa.Large_String_Type, oa.Binary_Type, oa.Large_Binary_Type:
		return {}, false
	}
	return {}, false
}

// ── Write: DataFrame → Arrow IPC ───────────────────────────────────────────

dataframe_write_arrow :: proc(df: ^DataFrame, path: string, allocator := context.allocator) -> Error {
	n := df.columns.count
	if n == 0 {
		return .Invalid_Argument
	}

	fields := make([]oa.Field, n, allocator)
	if fields == nil {
		return .Allocator_Failure
	}
	defer delete(fields)

	for i in 0 ..< n {
		col := column_set_to_column(&df.columns, i)
		arrow_dt, ok := arrow_type_for_dtype(col.dtype)
		if !ok {
			return .Unsupported_Operation
		}
		nullable := false
		if col.valid != nil {
			nullable = true
		} else {
			for j in 0 ..< col.count {
				if !column_is_valid(&col, j) {
					nullable = true
					break
				}
			}
		}
		fields[i] = oa.field_make(col.name, arrow_dt, nullable)
	}

	schema, s_err := oa.schema_make(fields, allocator)
	if s_err != .None {
		return .Allocator_Failure
	}
	defer oa.schema_free(&schema)

	arrays := make([]oa.Array, n, allocator)
	if arrays == nil {
		return .Allocator_Failure
	}
	defer delete(arrays)

	for i in 0 ..< n {
		col := column_set_to_column(&df.columns, i)
		arr, a_err := _arrow_array_from_column(&col, allocator)
		if a_err != .None {
			return .Allocator_Failure
		}
		arrays[i] = arr
	}

	batch, b_ok := oa.record_batch_make(&schema, arrays, allocator)
	if !b_ok {
		return .Invalid_Argument
	}
	defer oa.record_batch_free(&batch)

	if !oa.ipc_write_file(path, &schema, []oa.Record_Batch{batch}) {
		return .CSV_Error
	}
	return .None
}

// ── Write: DataFrame → Arrow IPC stream ────────────────────────────────────

dataframe_write_arrow_stream :: proc(df: ^DataFrame, path: string, allocator := context.allocator) -> Error {
	n := df.columns.count
	if n == 0 {
		return .Invalid_Argument
	}

	fields := make([]oa.Field, n, allocator)
	if fields == nil {
		return .Allocator_Failure
	}
	defer delete(fields)

	for i in 0 ..< n {
		col := column_set_to_column(&df.columns, i)
		arrow_dt, ok := arrow_type_for_dtype(col.dtype)
		if !ok {
			return .Unsupported_Operation
		}
		nullable := false
		if col.valid != nil {
			nullable = true
		} else {
			for j in 0 ..< col.count {
				if !column_is_valid(&col, j) {
					nullable = true
					break
				}
			}
		}
		fields[i] = oa.field_make(col.name, arrow_dt, nullable)
	}

	schema, s_err := oa.schema_make(fields, allocator)
	if s_err != .None {
		return .Allocator_Failure
	}
	defer oa.schema_free(&schema)

	arrays := make([]oa.Array, n, allocator)
	if arrays == nil {
		return .Allocator_Failure
	}
	defer delete(arrays)

	for i in 0 ..< n {
		col := column_set_to_column(&df.columns, i)
		arr, a_err := _arrow_array_from_column(&col, allocator)
		if a_err != .None {
			return .Allocator_Failure
		}
		arrays[i] = arr
	}

	batch, b_ok := oa.record_batch_make(&schema, arrays, allocator)
	if !b_ok {
		return .Invalid_Argument
	}
	defer oa.record_batch_free(&batch)

	if !oa.ipc_write_stream(path, &schema, []oa.Record_Batch{batch}) {
		return .CSV_Error
	}
	return .None
}

// ── Read: Arrow IPC stream → DataFrame ────────────────────────────────────

dataframe_read_arrow_stream :: proc(path: string, allocator := context.allocator) -> (DataFrame, Error) {
	schema, batches, ok := oa.ipc_read_stream(path, allocator, true, false)
	if !ok {
		return {}, .Invalid_Schema
	}
	defer {
		oa.schema_free(schema)
		free(schema)
		for bx in batches { bc := bx; oa.record_batch_free(&bc) }
		delete(batches)
	}

	if len(batches) == 0 {
		return {}, .Invalid_Schema
	}

	df := dataframe_create(allocator)

	for fi in 0 ..< len(schema.fields) {
		field := &schema.fields[fi]

		dtype, d_ok := dtype_for_arrow_type(field.type)
		if !d_ok {
			dataframe_destroy(&df)
			return {}, .Unsupported_Operation
		}

		total := 0
		for bx in batches {
			if fi < len(bx.columns) {
				total += bx.columns[fi].length
			}
		}

		if total == 0 {
			col, c_err := column_empty(field.name, dtype, allocator)
			if c_err != .None {
				dataframe_destroy(&df)
				return {}, c_err
			}
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col)
				dataframe_destroy(&df)
				return {}, a_err
			}
			continue
		}

		switch dtype {
		case typeid_of(bool):
			col, err := _arrow_read_bool(batches, fi, field.name, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(i8):
			col, err := _arrow_read_primitive(batches, fi, field.name, i8, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(i16):
			col, err := _arrow_read_primitive(batches, fi, field.name, i16, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(i32):
			col, err := _arrow_read_primitive(batches, fi, field.name, i32, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(i64):
			col, err := _arrow_read_primitive(batches, fi, field.name, i64, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(u8):
			col, err := _arrow_read_primitive(batches, fi, field.name, u8, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(u16):
			col, err := _arrow_read_primitive(batches, fi, field.name, u16, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(u32):
			col, err := _arrow_read_primitive(batches, fi, field.name, u32, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(u64):
			col, err := _arrow_read_primitive(batches, fi, field.name, u64, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(f32):
			col, err := _arrow_read_primitive(batches, fi, field.name, f32, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(f64):
			col, err := _arrow_read_primitive(batches, fi, field.name, f64, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(string):
			col, err := _arrow_read_string(batches, fi, field.name, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(Date):
			col, err := _arrow_read_temporal(batches, fi, field.name, Date, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(Time):
			col, err := _arrow_read_temporal(batches, fi, field.name, Time, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(Datetime):
			col, err := _arrow_read_temporal(batches, fi, field.name, Datetime, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(Duration):
			col, err := _arrow_read_temporal(batches, fi, field.name, Duration, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case:
			dataframe_destroy(&df)
			return {}, .Unsupported_Operation
		}
	}

	return df, .None
}

// ── column → Arrow Array ────────────────────────────────────────────────────

_arrow_array_from_column :: proc(col: ^Column, allocator: mem.Allocator) -> (oa.Array, mem.Allocator_Error) {
	switch col.dtype {
	case typeid_of(bool):
		return _arrow_bool_array(col, allocator)
	case typeid_of(i8):
		return _arrow_primitive_array(col, i8, allocator)
	case typeid_of(i16):
		return _arrow_primitive_array(col, i16, allocator)
	case typeid_of(i32):
		return _arrow_primitive_array(col, i32, allocator)
	case typeid_of(i64):
		return _arrow_primitive_array(col, i64, allocator)
	case typeid_of(u8):
		return _arrow_primitive_array(col, u8, allocator)
	case typeid_of(u16):
		return _arrow_primitive_array(col, u16, allocator)
	case typeid_of(u32):
		return _arrow_primitive_array(col, u32, allocator)
	case typeid_of(u64):
		return _arrow_primitive_array(col, u64, allocator)
	case typeid_of(f32):
		return _arrow_primitive_array(col, f32, allocator)
	case typeid_of(f64):
		return _arrow_primitive_array(col, f64, allocator)
	case typeid_of(string):
		return _arrow_string_array_write(col, allocator)
	case typeid_of(Date):
		return _arrow_temporal_array(col, Date, oa.Date64_Type{}, allocator)
	case typeid_of(Time):
		return _arrow_temporal_array(col, Time, oa.Time64_Type{}, allocator)
	case typeid_of(Datetime):
		return _arrow_temporal_array(col, Datetime, oa.Timestamp_Type{}, allocator)
	case typeid_of(Duration):
		return _arrow_temporal_array(col, Duration, oa.Duration_Type{}, allocator)
	}
	return {}, .None
}

_arrow_primitive_array :: proc(col: ^Column, $T: typeid, allocator: mem.Allocator) -> (oa.Array, mem.Allocator_Error) {
	n := col.count
	if n == 0 {
		return oa.Array{type = oa._data_type_for(T)}, nil
	}

	dim := size_of(T) * n
	data_buf, err := oa.buffer_make(dim, allocator)
	if err != .None {
		return {}, err
	}
	if col.data != nil {
		mem.copy(rawptr(data_buf.data), col.data, dim)
	}

	bitmap_buf: oa.Buffer
	null_count := 0
	has_validity := false
	for i in 0 ..< n {
		if !column_is_valid(col, i) {
			has_validity = true
			null_count += 1
		}
	}

	if has_validity {
		bm_size := oa.bitmap_byte_count(n)
		bitmap_buf, err = oa.buffer_make(bm_size, allocator)
		if err != .None {
			oa.buffer_free(&data_buf)
			return {}, err
		}
		oa.bitmap_set_all(bitmap_buf.data, n)
		for i in 0 ..< n {
			if !column_is_valid(col, i) {
				oa.bitmap_clear(bitmap_buf.data, i)
			}
		}
	}

	return oa.Array{
		type       = oa._data_type_for(T),
		length     = n,
		null_count = null_count,
		offset     = 0,
		buffers    = {bitmap_buf, data_buf, {}},
	}, nil
}

_arrow_bool_array :: proc(col: ^Column, allocator: mem.Allocator) -> (oa.Array, mem.Allocator_Error) {
	n := col.count
	if n == 0 {
		return oa.Array{type = oa.Bool_Type{}}, nil
	}

	// Arrow bool arrays store values as a bit-packed bitmap in buffers[1].
	val_size := oa.bitmap_byte_count(n)
	val_buf, err := oa.buffer_make(val_size, allocator)
	if err != .None {
		return {}, err
	}

	null_count := 0
	for i in 0 ..< n {
		if column_is_valid(col, i) {
			// Set value bit: Column bools are byte-per-element.
			src := cast([^]bool)col.data
			if src[i] {
				oa.bitmap_set(val_buf.data, i)
			}
		} else {
			null_count += 1
		}
	}

	bitmap_buf: oa.Buffer
	if null_count > 0 {
		bm_size := oa.bitmap_byte_count(n)
		bitmap_buf, err = oa.buffer_make(bm_size, allocator)
		if err != .None {
			oa.buffer_free(&val_buf)
			return {}, err
		}
		oa.bitmap_set_all(bitmap_buf.data, n)
		for i in 0 ..< n {
			if !column_is_valid(col, i) {
				oa.bitmap_clear(bitmap_buf.data, i)
			}
		}
	}

	return oa.Array{
		type       = oa.Bool_Type{},
		length     = n,
		null_count = null_count,
		offset     = 0,
		buffers    = {bitmap_buf, val_buf, {}},
	}, nil
}

_arrow_string_array_write :: proc(col: ^Column, allocator: mem.Allocator) -> (oa.Array, mem.Allocator_Error) {
	n := col.count
	if n == 0 {
		return oa.Array{type = oa.String_Type{}}, nil
	}

	strs := column_typed_view(col, string)

	// Compute total data size
	data_size := 0
	for s in strs { data_size += len(s) }

	// Build offsets buffer: (n + 1) i32 values
	off_buf, err := oa.buffer_make((n + 1) * size_of(i32), allocator)
	if err != .None {
		return {}, err
	}
	off := cast([^]i32)off_buf.data
	off[0] = 0

	// Build data buffer
	data_buf, d_err := oa.buffer_make(data_size, allocator)
	if d_err != .None {
		oa.buffer_free(&off_buf)
		return {}, d_err
	}

	pos := 0
	for i in 0 ..< n {
		s := strs[i]
		if len(s) > 0 {
			mem.copy(rawptr(data_buf.data[pos:]), rawptr(raw_data(s)), len(s))
		}
		pos += len(s)
		off[i + 1] = i32(pos)
	}

	// Validity bitmap — build from column validity
	bitmap_buf: oa.Buffer
	null_count := 0
	for i in 0 ..< n {
		if !column_is_valid(col, i) { null_count += 1 }
	}
	if null_count > 0 {
		bm_size := oa.bitmap_byte_count(n)
		bitmap_buf, err = oa.buffer_make(bm_size, allocator)
		if err != .None {
			oa.buffer_free(&off_buf)
			oa.buffer_free(&data_buf)
			return {}, err
		}
		oa.bitmap_set_all(bitmap_buf.data, n)
		for i in 0 ..< n {
			if !column_is_valid(col, i) {
				oa.bitmap_clear(bitmap_buf.data, i)
			}
		}
	}

	return oa.Array{
		type       = oa.String_Type{},
		length     = n,
		null_count = null_count,
		offset     = 0,
		buffers    = {bitmap_buf, off_buf, data_buf},
	}, nil
}

// ── temporal column → Arrow Array ──────────────────────────────────────────
//
// All temporal types (Date, Time, Datetime, Duration) are distinct i64.
// We transmute the raw data to []i64 and use the same primitive array layout.

_arrow_temporal_array :: proc(col: ^Column, $T: typeid, arrow_dt: oa.DataType, allocator: mem.Allocator) -> (oa.Array, mem.Allocator_Error) {
	n := col.count
	if n == 0 {
		return oa.Array{type = arrow_dt}, nil
	}

	dim := size_of(i64) * n
	data_buf, err := oa.buffer_make(dim, allocator)
	if err != .None {
		return {}, err
	}
	if col.data != nil {
		mem.copy(rawptr(data_buf.data), col.data, dim)
	}

	bitmap_buf: oa.Buffer
	null_count := 0
	has_validity := false
	for i in 0 ..< n {
		if !column_is_valid(col, i) {
			has_validity = true
			null_count += 1
		}
	}

	if has_validity {
		bm_size := oa.bitmap_byte_count(n)
		bitmap_buf, err = oa.buffer_make(bm_size, allocator)
		if err != .None {
			oa.buffer_free(&data_buf)
			return {}, err
		}
		oa.bitmap_set_all(bitmap_buf.data, n)
		for i in 0 ..< n {
			if !column_is_valid(col, i) {
				oa.bitmap_clear(bitmap_buf.data, i)
			}
		}
	}

	return oa.Array{
		type       = arrow_dt,
		length     = n,
		null_count = null_count,
		offset     = 0,
		buffers    = {bitmap_buf, data_buf, {}},
	}, nil
}

// ── Read: Arrow IPC → DataFrame ────────────────────────────────────────────

dataframe_read_arrow :: proc(path: string, allocator := context.allocator) -> (DataFrame, Error) {
	schema, batches, ok := oa.ipc_read_file(path, allocator)
	if !ok {
		return {}, .Invalid_Schema
	}
	defer {
		oa.schema_free(schema)
		free(schema)
		for bx in batches { bc := bx; oa.record_batch_free(&bc) }
		delete(batches)
	}

	if len(batches) == 0 {
		return {}, .Invalid_Schema
	}

	df := dataframe_create(allocator)

	for fi in 0 ..< len(schema.fields) {
		field := &schema.fields[fi]

		dtype, d_ok := dtype_for_arrow_type(field.type)
		if !d_ok {
			dataframe_destroy(&df)
			return {}, .Unsupported_Operation
		}

		total := 0
		for bx in batches {
			if fi < len(bx.columns) {
				total += bx.columns[fi].length
			}
		}

		if total == 0 {
			col, c_err := column_empty(field.name, dtype, allocator)
			if c_err != .None {
				dataframe_destroy(&df)
				return {}, c_err
			}
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col)
				dataframe_destroy(&df)
				return {}, a_err
			}
			continue
		}

		switch dtype {
		case typeid_of(bool):
			col, err := _arrow_read_bool(batches, fi, field.name, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(i8):
			col, err := _arrow_read_primitive(batches, fi, field.name, i8, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(i16):
			col, err := _arrow_read_primitive(batches, fi, field.name, i16, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(i32):
			col, err := _arrow_read_primitive(batches, fi, field.name, i32, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(i64):
			col, err := _arrow_read_primitive(batches, fi, field.name, i64, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(u8):
			col, err := _arrow_read_primitive(batches, fi, field.name, u8, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(u16):
			col, err := _arrow_read_primitive(batches, fi, field.name, u16, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(u32):
			col, err := _arrow_read_primitive(batches, fi, field.name, u32, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(u64):
			col, err := _arrow_read_primitive(batches, fi, field.name, u64, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(f32):
			col, err := _arrow_read_primitive(batches, fi, field.name, f32, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(f64):
			col, err := _arrow_read_primitive(batches, fi, field.name, f64, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(string):
			col, err := _arrow_read_string(batches, fi, field.name, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(Date):
			col, err := _arrow_read_temporal(batches, fi, field.name, Date, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(Time):
			col, err := _arrow_read_temporal(batches, fi, field.name, Time, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(Datetime):
			col, err := _arrow_read_temporal(batches, fi, field.name, Datetime, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case typeid_of(Duration):
			col, err := _arrow_read_temporal(batches, fi, field.name, Duration, allocator)
			if err != .None { dataframe_destroy(&df); return {}, err }
			if a_err := dataframe_add_column(&df, &col); a_err != .None {
				column_destroy(&col); dataframe_destroy(&df); return {}, a_err
			}
		case:
			dataframe_destroy(&df)
			return {}, .Unsupported_Operation
		}
	}

	return df, .None
}

// ── Arrow Array → Column (primitives) ──────────────────────────────────────

_arrow_read_primitive :: proc(
	batches: []oa.Record_Batch,
	col_idx: int,
	col_name: string,
	$T: typeid,
	allocator: mem.Allocator,
) -> (Column, Error) {
	total := 0
	for b in batches {
		if col_idx < len(b.columns) {
			total += b.columns[col_idx].length
		}
	}

	values := make([]T, total, allocator)
	if values == nil && total != 0 {
		return {}, .Allocator_Failure
	}

	valid_bools := make([]bool, total, allocator)
	if valid_bools == nil && total != 0 {
		delete(values)
		return {}, .Allocator_Failure
	}

	has_nulls := false
	pos := 0
	for b in batches {
		if col_idx >= len(b.columns) { continue }
		arr := &b.columns[col_idx]
		for i in 0 ..< arr.length {
			if oa.array_is_valid(arr, i) {
				values[pos] = oa.array_get(arr, i, T)
				valid_bools[pos] = true
			} else {
				valid_bools[pos] = false
				has_nulls = true
			}
			pos += 1
		}
	}

	valid_arg: []bool
	if has_nulls {
		valid_arg = valid_bools
	}

	col, err := column_from_with_valid(col_name, values, valid_arg, allocator)
	delete(values)
	delete(valid_bools)
	return col, err
}

// ── Arrow Array → Column (string) ──────────────────────────────────────────
//
// Arrow UTF-8 arrays store i32 offsets + byte data. We build a contiguous
// blob, construct Odin string headers that point into it, call
// column_from_with_valid to copy the headers, then transfer blob ownership
// into col.payload.

_arrow_read_string :: proc(
	batches: []oa.Record_Batch,
	col_idx: int,
	col_name: string,
	allocator: mem.Allocator,
) -> (Column, Error) {
	total := 0
	for b in batches {
		if col_idx < len(b.columns) {
			total += b.columns[col_idx].length
		}
	}

	if total == 0 {
		col, c_err := column_empty(col_name, typeid_of(string), allocator)
		if c_err != .None {
			return {}, c_err
		}
		return col, .None
	}

	// Pass 1: compute total blob size
	blob_size := 0
	for b in batches {
		if col_idx >= len(b.columns) { continue }
		arr := &b.columns[col_idx]
		for i in 0 ..< arr.length {
			if oa.array_is_valid(arr, i) {
				blob_size += len(oa.array_get_string(arr, i))
			}
		}
	}

	// Allocate the contiguous byte blob
	blob: rawptr
	if blob_size > 0 {
		raw, b_err := mem.alloc(blob_size, 1, allocator)
		if b_err != .None || raw == nil {
			return {}, .Allocator_Failure
		}
		blob = raw
	}

	// Allocate string headers (will be passed to column_from_with_valid)
	strs := make([]string, total, allocator)
	if strs == nil {
		if blob != nil { mem.free_with_size(blob, blob_size, allocator) }
		return {}, .Allocator_Failure
	}

	valid_bools := make([]bool, total, allocator)
	if valid_bools == nil {
		if blob != nil { mem.free_with_size(blob, blob_size, allocator) }
		delete(strs)
		return {}, .Allocator_Failure
	}

	// Pass 2: copy string bytes into blob, build headers pointing into it
	blob_pos := 0
	pos := 0
	has_nulls := false
	for b in batches {
		if col_idx >= len(b.columns) { continue }
		arr := &b.columns[col_idx]
		for i in 0 ..< arr.length {
			if oa.array_is_valid(arr, i) {
				s := oa.array_get_string(arr, i)
				if len(s) > 0 && blob != nil {
					mem.copy(
						rawptr(uintptr(blob) + uintptr(blob_pos)),
						rawptr(raw_data(s)),
						len(s),
					)
				}
				strs[pos] = transmute(string)runtime.Raw_String{
					data = (^u8)(uintptr(blob) + uintptr(blob_pos)),
					len  = len(s),
				}
				valid_bools[pos] = true
				blob_pos += len(s)
			} else {
				strs[pos] = ""
				valid_bools[pos] = false
				has_nulls = true
			}
			pos += 1
		}
	}

	valid_arg: []bool
	if has_nulls {
		valid_arg = valid_bools
	}

	// column_from_with_valid copies the string headers into a new data buffer.
	// The copied headers still reference our blob.
	col, c_err := column_from_with_valid(col_name, strs, valid_arg, allocator)
	delete(valid_bools)
	if c_err != .None {
		delete(strs)
		if blob != nil { mem.free_with_size(blob, blob_size, allocator) }
		return {}, c_err
	}

	// Transfer blob ownership into col.payload so column_destroy frees it.
	col.payload = blob
	col.payload_size = blob_size

	// Free the temporary strs slice. Each string in strs is a non-owning
	// view into blob; delete(slice) only frees the backing array memory,
	// not the string byte data (Odin strings are not individually heap-allocated).
	delete(strs)

	return col, .None
}

// ── Arrow Array → Column (bool, bit-packed) ────────────────────────────────
//
// Arrow Bool arrays store values as a bit-packed bitmap in buffers[1].
// Odin Column bools are byte-per-element. We unpack on read.

_arrow_read_bool :: proc(
	batches: []oa.Record_Batch,
	col_idx: int,
	col_name: string,
	allocator: mem.Allocator,
) -> (Column, Error) {
	total := 0
	for b in batches {
		if col_idx < len(b.columns) {
			total += b.columns[col_idx].length
		}
	}

	values := make([]bool, total, allocator)
	if values == nil && total != 0 {
		return {}, .Allocator_Failure
	}

	valid_bools := make([]bool, total, allocator)
	if valid_bools == nil && total != 0 {
		delete(values)
		return {}, .Allocator_Failure
	}

	has_nulls := false
	pos := 0
	for b in batches {
		if col_idx >= len(b.columns) { continue }
		arr := &b.columns[col_idx]
		for i in 0 ..< arr.length {
			if oa.array_is_valid(arr, i) {
				values[pos] = oa.array_get(arr, i, bool)
				valid_bools[pos] = true
			} else {
				valid_bools[pos] = false
				has_nulls = true
			}
			pos += 1
		}
	}

	valid_arg: []bool
	if has_nulls {
		valid_arg = valid_bools
	}

	col, err := column_from_with_valid(col_name, values, valid_arg, allocator)
	delete(values)
	delete(valid_bools)
	return col, err
}

// ── Arrow Array → Column (temporal, distinct i64) ──────────────────────────
//
// All temporal types (Date, Time, Datetime, Duration) are distinct i64 in the
// dataframe. Arrow stores them as i64 buffers. We read as i64 then swap the
// dtype to the target temporal type.

_arrow_read_temporal :: proc(
	batches: []oa.Record_Batch,
	col_idx: int,
	col_name: string,
	$T: typeid,
	allocator: mem.Allocator,
) -> (Column, Error) {
	total := 0
	for b in batches {
		if col_idx < len(b.columns) {
			total += b.columns[col_idx].length
		}
	}

	values := make([]i64, total, allocator)
	if values == nil && total != 0 {
		return {}, .Allocator_Failure
	}

	valid_bools := make([]bool, total, allocator)
	if valid_bools == nil && total != 0 {
		delete(values)
		return {}, .Allocator_Failure
	}

	has_nulls := false
	pos := 0
	for b in batches {
		if col_idx >= len(b.columns) { continue }
		arr := &b.columns[col_idx]
		for i in 0 ..< arr.length {
			if oa.array_is_valid(arr, i) {
				values[pos] = oa.array_get(arr, i, i64)
				valid_bools[pos] = true
			} else {
				valid_bools[pos] = false
				has_nulls = true
			}
			pos += 1
		}
	}

	valid_arg: []bool
	if has_nulls {
		valid_arg = valid_bools
	}

	// Build as i64 column, then swap dtype to T (all temporal types are distinct i64)
	col, c_err := column_from_with_valid(col_name, values, valid_arg, allocator)
	delete(values)
	delete(valid_bools)
	if c_err != .None {
		return {}, c_err
	}

	// Swap the dtype from i64 to the target temporal type
	// (all are distinct i64 with the same elem_size and align)
	col.dtype = T
	return col, .None
}
