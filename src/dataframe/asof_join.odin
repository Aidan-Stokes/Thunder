package dataframe

// As-of join (Stage 14.6, DESIGN.md §18.3): dataframe_asof_join.
//
// Semantics:
//   - `on` selects, per left row, the nearest right row by the strategy:
//     Backward picks the greatest right `on` <= left `on`; Forward picks the
//     smallest right `on` >= left `on`. Right `on` ties resolve to the last
//     (Backward) or first (Forward) row in the bucket.
//   - `by` columns, when given, must match exactly; a NULL `by` or `on` value
//     on the left never matches, and right rows with a NULL `by`/`on` are
//     excluded from matching (SQL semantics, as in join.odin).
//   - Output is left-major like a left join: all left columns, then the right
//     columns that are neither `by` nor `on` columns (join.odin `_right`
//     collision-suffix rule). Unmatched left rows carry NULL right columns.
//   - Precondition (documented, not validated): `right` is sorted ascending
//     by (`by`, `on`).
//
// Implementation: right rows are bucketed by `by` (`encode_row`); the
// precondition guarantees each bucket's `on` values are ascending, so every
// left row matches by binary search. Materialization reuses join_materialize.

import "core:mem"

// Asof_Strategy selects the matching direction of an as-of join.
Asof_Strategy :: enum {
	Backward,
	Forward,
}

dataframe_asof_join :: proc(
	left, right: ^DataFrame,
	left_on, right_on: string,
	left_by, right_by: []string,
	strategy: Asof_Strategy,
	allocator := context.allocator,
) -> (out: DataFrame, err: Error) {
	if len(left_by) != len(right_by) {
		return {}, .Invalid_Argument
	}

	left_on_col, l_err := dataframe_get_column(left, left_on)
	if l_err != .None {
		return {}, l_err
	}
	right_on_col, r_err := dataframe_get_column(right, right_on)
	if r_err != .None {
		return {}, r_err
	}
	if left_on_col.dtype != right_on_col.dtype {
		return {}, .Type_Mismatch
	}
	if is_categorical(left_on_col) != is_categorical(right_on_col) {
		return {}, .Type_Mismatch
	}
	if !sortable_dtype(left_on_col.dtype) {
		return {}, .Unsupported_Operation
	}

	left_by_cols: []^Column
	left_by_cols, err = resolve_named_columns(left, allocator, left_by)
	if err != .None {
		return {}, err
	}
	defer delete(left_by_cols, allocator)
	right_by_cols: []^Column
	right_by_cols, err = resolve_named_columns(right, allocator, right_by)
	if err != .None {
		return {}, err
	}
	defer delete(right_by_cols, allocator)
	for i in 0 ..< len(left_by_cols) {
		if left_by_cols[i].dtype != right_by_cols[i].dtype {
			return {}, .Type_Mismatch
		}
	}

	// Combined key columns (by, then on) pair positionally for
	// join_materialize, which drops the right keys from the output.
	left_keys := make([dynamic]^Column, allocator)
	defer delete(left_keys)
	right_keys := make([dynamic]^Column, allocator)
	defer delete(right_keys)
	for c in left_by_cols {
		append(&left_keys, c)
	}
	append(&left_keys, left_on_col)
	for c in right_by_cols {
		append(&right_keys, c)
	}
	append(&right_keys, right_on_col)

	left_rows := dataframe_num_rows(left)
	right_rows := dataframe_num_rows(right)

	buckets: map[string][dynamic]int
	buckets, err = asof_bucket_build(right_by_cols, right_on_col, right_rows, allocator)
	if err != .None {
		return {}, err
	}
	defer join_hash_destroy(buckets, allocator)

	pairs := make([dynamic]Pair, allocator)
	defer delete(pairs)
	buf := make([dynamic]byte, allocator)
	defer delete(buf)
	for l in 0 ..< left_rows {
		if !join_key_valid(left_by_cols, l) || !row_valid(left_on_col.valid, l) {
			append(&pairs, Pair{l = l, r = -1})
			continue
		}
		if enc_err := encode_row(left_by_cols, l, &buf); enc_err != .None {
			return {}, enc_err
		}
		bucket, ok := buckets[string(buf[:])]
		if !ok {
			append(&pairs, Pair{l = l, r = -1})
			continue
		}
		match, found := asof_binary_search(left_on_col, l, right_on_col, bucket, strategy)
		if found {
			append(&pairs, Pair{l = l, r = match})
		} else {
			append(&pairs, Pair{l = l, r = -1})
		}
	}

	return join_materialize(.Left, left, right, left_keys[:], right_keys[:], allocator, pairs)
}

// resolve_named_columns returns borrowed pointers to the columns named in
// names (empty by lists resolve to an empty slice).
@(private)
resolve_named_columns :: proc(df: ^DataFrame, allocator: mem.Allocator, names: []string) -> ([]^Column, Error) {
	out := make([]^Column, len(names), allocator)
	if out == nil && len(names) != 0 {
		return nil, .Allocator_Failure
	}
	for name, i in names {
		for j in 0 ..< i {
			if names[j] == name {
				delete(out, allocator)
				return nil, .Invalid_Argument
			}
		}
		c, get_err := dataframe_get_column(df, name)
		if get_err != .None {
			delete(out, allocator)
			return nil, get_err
		}
		out[i] = c
	}
	return out, .None
}

// asof_bucket_build groups the right rows by `by` into a hash map. Rows with a
// NULL `by` or `on` value are excluded (they can never match). Within a bucket
// the row indices are ascending, and the as-of precondition guarantees their
// `on` values are ascending too.
@(private)
asof_bucket_build :: proc(by_cols: []^Column, on_col: ^Column, n_rows: int, allocator: mem.Allocator) -> (m: map[string][dynamic]int, err: Error) {
	m = make(map[string][dynamic]int, 0, allocator)
	buf := make([dynamic]byte, allocator)
	defer delete(buf)
	for row in 0 ..< n_rows {
		if !join_key_valid(by_cols, row) || !row_valid(on_col.valid, row) {
			continue
		}
		if enc_err := encode_row(by_cols, row, &buf); enc_err != .None {
			join_hash_destroy(m, allocator)
			return nil, enc_err
		}
		key := string(buf[:])
		if _, exists := m[key]; !exists {
			owned, o_err := clone_name(allocator, key)
			if o_err != .None {
				join_hash_destroy(m, allocator)
				return nil, o_err
			}
			m[owned] = make([dynamic]int, allocator)
		}
		append(&m[key], row)
	}
	return m, .None
}

// asof_binary_search finds the matching right row for left row `l` inside one
// `on`-sorted bucket, per the strategy, comparing values across the two
// columns. Returns found=false when no right row qualifies.
@(private)
asof_binary_search :: proc(left_on: ^Column, l: int, right_on: ^Column, bucket: [dynamic]int, strategy: Asof_Strategy) -> (match: int, found: bool) {
	lo, hi := 0, len(bucket)
	switch strategy {
	case .Backward:
		// greatest right `on` <= left `on`: first index with right > left.
		for lo < hi {
			mid := (lo + hi) >> 1
			if asof_compare(left_on, l, right_on, bucket[mid]) >= 0 {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		if lo == 0 {
			return 0, false
		}
		return bucket[lo - 1], true
	case .Forward:
		// smallest right `on` >= left `on`: first index with right >= left.
		for lo < hi {
			mid := (lo + hi) >> 1
			if asof_compare(left_on, l, right_on, bucket[mid]) > 0 {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		if lo == len(bucket) {
			return 0, false
		}
		return bucket[lo], true
	}
	return 0, false
}

// asof_compare orders the non-NULL value of left_on at l against right_on at
// r. Callers guarantee equal dtypes and non-NULL rows.
@(private)
asof_compare :: proc(left: ^Column, l: int, right: ^Column, r: int) -> int {
	if is_categorical(left) {
		ls, _ := categorical_value(left, l)
		rs, _ := categorical_value(right, r)
		return ordered_compare(ls, rs)
	}
	switch right.dtype {
	case typeid_of(bool):
		lb: u8 = 0
		if column_typed_view(left, bool)[l] {
			lb = 1
		}
		rb: u8 = 0
		if column_typed_view(right, bool)[r] {
			rb = 1
		}
		return ordered_compare(lb, rb)
	case typeid_of(string):
		return ordered_compare(column_typed_view(left, string)[l], column_typed_view(right, string)[r])
	case typeid_of(i8):
		return ordered_compare(column_typed_view(left, i8)[l], column_typed_view(right, i8)[r])
	case typeid_of(i16):
		return ordered_compare(column_typed_view(left, i16)[l], column_typed_view(right, i16)[r])
	case typeid_of(i32):
		return ordered_compare(column_typed_view(left, i32)[l], column_typed_view(right, i32)[r])
	case typeid_of(i64):
		return ordered_compare(column_typed_view(left, i64)[l], column_typed_view(right, i64)[r])
	case typeid_of(u8):
		return ordered_compare(column_typed_view(left, u8)[l], column_typed_view(right, u8)[r])
	case typeid_of(u16):
		return ordered_compare(column_typed_view(left, u16)[l], column_typed_view(right, u16)[r])
	case typeid_of(u32):
		return ordered_compare(column_typed_view(left, u32)[l], column_typed_view(right, u32)[r])
	case typeid_of(u64):
		return ordered_compare(column_typed_view(left, u64)[l], column_typed_view(right, u64)[r])
	case typeid_of(int):
		return ordered_compare(column_typed_view(left, int)[l], column_typed_view(right, int)[r])
	case typeid_of(uint):
		return ordered_compare(column_typed_view(left, uint)[l], column_typed_view(right, uint)[r])
	case typeid_of(f32):
		return ordered_compare(
			float32_bits(canonical_float32(column_typed_view(left, f32)[l])),
			float32_bits(canonical_float32(column_typed_view(right, f32)[r])),
		)
	case typeid_of(f64):
		return ordered_compare(
			float64_bits(canonical_float64(column_typed_view(left, f64)[l])),
			float64_bits(canonical_float64(column_typed_view(right, f64)[r])),
		)
	case typeid_of(Date):
		return ordered_compare(column_typed_view(left, Date)[l], column_typed_view(right, Date)[r])
	case typeid_of(Datetime):
		return ordered_compare(column_typed_view(left, Datetime)[l], column_typed_view(right, Datetime)[r])
	case typeid_of(Time):
		return ordered_compare(column_typed_view(left, Time)[l], column_typed_view(right, Time)[r])
	case typeid_of(Duration):
		return ordered_compare(column_typed_view(left, Duration)[l], column_typed_view(right, Duration)[r])
	case:
		return 0 // unreachable: sortable_dtype validated this already
	}
}
