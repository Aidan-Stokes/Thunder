package odinarrow

// --- Type tags -----------------------------------------------------------
// Empty structs — the tagged union discriminant IS the type identity.
// Types with parameters (Timestamp, FixedSizeList, etc.) will grow fields.

Null_Type         :: struct{}
Bool_Type         :: struct{}
Int8_Type         :: struct{}
Int16_Type        :: struct{}
Int32_Type        :: struct{}
Int64_Type        :: struct{}
UInt8_Type        :: struct{}
UInt16_Type       :: struct{}
UInt32_Type       :: struct{}
UInt64_Type       :: struct{}
Float32_Type      :: struct{}
Float64_Type      :: struct{}
String_Type       :: struct{} // UTF-8, i32 offsets
Large_String_Type :: struct{} // UTF-8, i64 offsets
Binary_Type       :: struct{} // raw bytes, i32 offsets
Large_Binary_Type :: struct{} // raw bytes, i64 offsets

// Temporal types
Date64_Type   :: struct{} // i64 days since epoch (matching Odin Date)
Time64_Type   :: struct{} // i64 microseconds since midnight (matching Odin Time)
Timestamp_Type :: struct{} // i64 microseconds since epoch (matching Odin Datetime)
Duration_Type  :: struct{} // i64 signed microseconds (matching Odin Duration)

DataType :: union {
	Null_Type,
	Bool_Type,
	Int8_Type, Int16_Type, Int32_Type, Int64_Type,
	UInt8_Type, UInt16_Type, UInt32_Type, UInt64_Type,
	Float32_Type, Float64_Type,
	String_Type, Large_String_Type,
	Binary_Type, Large_Binary_Type,
	Date64_Type, Time64_Type, Timestamp_Type, Duration_Type,
}

// --- Metadata helpers (exhaustive switches required by the compiler) ------

// Fixed byte width per element; -1 for variable-length; 0 for Null/Bool.
type_byte_width :: proc(dt: DataType) -> int {
	switch _ in dt {
	case Null_Type:                           return 0
	case Bool_Type:                           return 0
	case Int8_Type:                           return 1
	case Int16_Type:                          return 2
	case Int32_Type:                          return 4
	case Int64_Type:                          return 8
	case UInt8_Type:                          return 1
	case UInt16_Type:                         return 2
	case UInt32_Type:                         return 4
	case UInt64_Type:                         return 8
	case Float32_Type:                        return 4
	case Float64_Type:                        return 8
	case String_Type, Large_String_Type,
	     Binary_Type, Large_Binary_Type:      return -1
	case Date64_Type:                         return 8
	case Time64_Type:                         return 8
	case Timestamp_Type:                      return 8
	case Duration_Type:                       return 8
	}
	return 0
}

type_is_bit_packed :: proc(dt: DataType) -> bool {
	_, ok := dt.(Bool_Type)
	return ok
}

type_is_variable_length :: proc(dt: DataType) -> bool {
	switch _ in dt {
	case String_Type, Large_String_Type, Binary_Type, Large_Binary_Type:
		return true
	case Null_Type, Bool_Type,
	     Int8_Type, Int16_Type, Int32_Type, Int64_Type,
	     UInt8_Type, UInt16_Type, UInt32_Type, UInt64_Type,
	     Float32_Type, Float64_Type,
	     Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		return false
	}
	return false
}

type_is_integer :: proc(dt: DataType) -> bool {
	switch _ in dt {
	case Int8_Type, Int16_Type, Int32_Type, Int64_Type,
	     UInt8_Type, UInt16_Type, UInt32_Type, UInt64_Type:
		return true
	case Null_Type, Bool_Type, Float32_Type, Float64_Type,
	     String_Type, Large_String_Type, Binary_Type, Large_Binary_Type,
	     Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		return false
	}
	return false
}

type_is_signed :: proc(dt: DataType) -> bool {
	switch _ in dt {
	case Int8_Type, Int16_Type, Int32_Type, Int64_Type:
		return true
	case Null_Type, Bool_Type,
	     UInt8_Type, UInt16_Type, UInt32_Type, UInt64_Type,
	     Float32_Type, Float64_Type,
	     String_Type, Large_String_Type, Binary_Type, Large_Binary_Type,
	     Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		return false
	}
	return false
}

type_is_floating :: proc(dt: DataType) -> bool {
	switch _ in dt {
	case Float32_Type, Float64_Type:
		return true
	case Null_Type, Bool_Type,
	     Int8_Type, Int16_Type, Int32_Type, Int64_Type,
	     UInt8_Type, UInt16_Type, UInt32_Type, UInt64_Type,
	     String_Type, Large_String_Type, Binary_Type, Large_Binary_Type,
	     Date64_Type, Time64_Type, Timestamp_Type, Duration_Type:
		return false
	}
	return false
}

type_name :: proc(dt: DataType) -> string {
	switch _ in dt {
	case Null_Type:         return "null"
	case Bool_Type:         return "bool"
	case Int8_Type:         return "int8"
	case Int16_Type:        return "int16"
	case Int32_Type:        return "int32"
	case Int64_Type:        return "int64"
	case UInt8_Type:        return "uint8"
	case UInt16_Type:       return "uint16"
	case UInt32_Type:       return "uint32"
	case UInt64_Type:       return "uint64"
	case Float32_Type:      return "float32"
	case Float64_Type:      return "float64"
	case String_Type:       return "string"
	case Large_String_Type: return "large_string"
	case Binary_Type:       return "binary"
	case Large_Binary_Type: return "large_binary"
	case Date64_Type:       return "date64"
	case Time64_Type:       return "time64"
	case Timestamp_Type:    return "timestamp"
	case Duration_Type:     return "duration"
	}
	return "unknown"
}
