package dataframe

// Error is the single public error type for the dataframe API (DESIGN.md §12).
// It is returned as the last result of procs that can fail; .None means success.
Error :: enum {
	None,

	// Generic argument problems.
	Invalid_Argument,

	// Column access.
	Column_Not_Found,
	Duplicate_Column_Name,
	Column_Name_Empty,

	// Index/length problems.
	Out_Of_Bounds,
	Length_Mismatch,

	// Type problems. Conversions are never silent (principle 6).
	Type_Mismatch,
	Unsupported_Operation, // op or cast not supported for a given type

	// Data problems. NULL is a data condition, not an error; this is used
	// when an operation requires a non-NULL value.
	Null_Value,

	// Resource problems.
	Allocator_Failure,

	// I/O problems.
	CSV_Error, // CSV I/O or parse failure (missing file, ragged rows, bad quotes)
	JSON_Error, // JSON/NDJSON I/O or parse failure (missing file, malformed document)
	Parquet_Error, // Parquet I/O or format failure

	// Schema / plan problems.
	Invalid_Schema,
}

// error_to_string returns a human readable description of err.
error_to_string :: proc(err: Error) -> string {
	switch err {
	case .None:               return "no error"
	case .Invalid_Argument:   return "invalid argument"
	case .Column_Not_Found:   return "column not found"
	case .Duplicate_Column_Name: return "duplicate column name"
	case .Column_Name_Empty:  return "column name is empty"
	case .Out_Of_Bounds:      return "index out of bounds"
	case .Length_Mismatch:    return "length mismatch"
	case .Type_Mismatch:      return "type mismatch"
	case .Unsupported_Operation: return "operation not supported for this type"
	case .Null_Value:         return "value is NULL"
	case .Allocator_Failure:  return "allocator failure"
	case .CSV_Error:          return "csv read or write failure"
	case .JSON_Error:         return "json or ndjson read or write failure"
	case .Parquet_Error:      return "parquet read or write failure"
	case .Invalid_Schema:     return "invalid schema"
	}
	return "unknown error"
}
