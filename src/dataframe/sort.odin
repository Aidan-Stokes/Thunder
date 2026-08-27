package dataframe

// Sorting (Stage 5): argsort produces a stable permutation; sort reorders
// rows into a new DataFrame; sort_by sorts by computed (expression) keys.
//
// Ordering rules (documented, tested):
//   - Keys are compared in order; the first key that differs decides. Ties
//     (including rows that are NULL in the same keys) fall through to the
//     next key; a full tie keeps source order (the sort is stable).
//   - NULL ordering is explicit and direction-independent: NULLs sort after
//     all non-NULL values when nulls_first is false (the default), before
//     them when true. Rows NULL in a key column are never equal to non-NULL
//     rows, even if the underlying buffer holds zero/NaN/empty-string.
//   - Ordering is total for the supported scalar types (bool, the integer
//     families, f32/f64, string). Strings compare lexicographically by byte.
//     Float NaN compares equal to NaN (it sorts as its canonical bit value,
//     after every finite number); -0.0 equals +0.0. Anything else is an
//     error (Unsupported_Operation).
//
// Sort produces a permutation; physical reordering is a separate explicit op
// (DESIGN.md §5): argsort returns the permutation, sort materializes it.

import "core:mem"
import "core:sort"
import "expr"

// Sort_Order selects the direction for a sort key.
Sort_Order :: enum {
	Asc,
	Desc,
}

// Sort_Key names one sort key: a column, an order, and where NULLs go.
Sort_Key :: struct {
	name:        string,
	order:       Sort_Order,
	nulls_first: bool,
}

// sort_key builds a Sort_Key; the common single-column case reads naturally:
//
//	sort_key("age")              // ascending, NULLs last
//	sort_key("age", .Desc)       // descending, NULLs last
//	sort_key("age", .Asc, true)  // ascending, NULLs first
sort_key :: proc(name: string, order: Sort_Order = .Asc, nulls_first: bool = false) -> Sort_Key {
	return Sort_Key{name = name, order = order, nulls_first = nulls_first}
}

// dataframe_argsort returns a stable permutation ordering df's rows by the
// key columns. The returned slice is owned by the caller. An empty by is an
// error; unknown columns are .Column_Not_Found; unsortable dtypes are
// .Unsupported_Operation.
dataframe_argsort :: proc(df: ^DataFrame, by: []Sort_Key, allocator := context.allocator) -> (perm: []int, err: Error) {
	keys, k_err := resolve_sort_keys(df, allocator, by)
	if k_err != .None {
		return nil, k_err
	}
	defer delete(keys, allocator)
	return argsort_key_columns(keys, allocator)
}

// dataframe_sort returns a new DataFrame holding df's rows in key order. The
// permutation is stable and the result owns its columns; df is borrowed.
dataframe_sort :: proc(df: ^DataFrame, by: []Sort_Key, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	keys, k_err := resolve_sort_keys(df, allocator, by)
	if k_err != .None {
		return {}, k_err
	}
	defer delete(keys, allocator)

	perm, p_err := argsort_key_columns(keys, allocator)
	if p_err != .None {
		return {}, p_err
	}
	defer delete(perm, allocator)

	return take_columns(df, allocator, perm)
}

// dataframe_sort_by sorts df by computed keys: each expression is evaluated
// to a column and used as a key in order. Every result must be a scalar
// column of a sortable dtype. orders may be empty (all ascending) or must
// match len(by). NULL keys sort last (as if nulls_first were false).
dataframe_sort_by :: proc(df: ^DataFrame, by: []^expr.Expr, orders: []Sort_Order, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	if len(by) == 0 {
		return {}, .Invalid_Argument
	}
	if len(orders) != 0 && len(orders) != len(by) {
		return {}, .Invalid_Argument
	}

	key_cols := make([]Column, len(by), allocator)
	if key_cols == nil && len(by) != 0 {
		return {}, .Allocator_Failure
	}
	defer {
		for &c in key_cols {
			column_destroy(&c)
		}
		delete(key_cols, allocator)
	}
	oa: OpArena
	op_arena_init(&oa, allocator)
	defer op_arena_destroy(&oa)
	for e, i in by {
		c, eval_err := expr_eval(allocator, df, e, &oa)
		if eval_err != .None {
			return {}, eval_err
		}
		key_cols[i] = c
	}

	keys := make([]sort_key_internal, len(by), allocator)
	if keys == nil && len(by) != 0 {
		return {}, .Allocator_Failure
	}
	defer delete(keys, allocator)
	for i in 0 ..< len(by) {
		order := Sort_Order.Asc
		if len(orders) != 0 {
			order = orders[i]
		}
		keys[i] = sort_key_internal{col = &key_cols[i], order = order}
	}

	perm, p_err := argsort_key_columns(keys, allocator)
	if p_err != .None {
		return {}, p_err
	}
	defer delete(perm, allocator)

	return take_columns(df, allocator, perm)
}

// --- private machinery ------------------------------------------------------

// sort_key_internal is a resolved sort key: a key column plus its order and
// NULL placement.
@(private)
sort_key_internal :: struct {
	col:         ^Column,
	order:       Sort_Order,
	nulls_first: bool,
}

// resolve_sort_keys looks up each Sort_Key by name and validates sortability.
@(private)
resolve_sort_keys :: proc(df: ^DataFrame, allocator: mem.Allocator, by: []Sort_Key) -> ([]sort_key_internal, Error) {
	if len(by) == 0 {
		return nil, .Invalid_Argument
	}
	keys := make([]sort_key_internal, len(by), allocator)
	if keys == nil {
		return nil, .Allocator_Failure
	}
	for k, i in by {
		c, get_err := dataframe_get_column(df, k.name)
		if get_err != .None {
			delete(keys, allocator)
			return nil, get_err
		}
		if !sortable_dtype(c.dtype) {
			delete(keys, allocator)
			return nil, .Unsupported_Operation
		}
		keys[i] = sort_key_internal{col = c, order = k.order, nulls_first = k.nulls_first}
	}
	return keys, .None
}

// sortable_dtype reports whether dtype has a total order (see file doc).
@(private)
sortable_dtype :: proc(dtype: typeid) -> bool {
	switch dtype {
	case typeid_of(bool), typeid_of(string),
	     typeid_of(i8), typeid_of(i16), typeid_of(i32), typeid_of(i64),
	     typeid_of(u8), typeid_of(u16), typeid_of(u32), typeid_of(u64),
	     typeid_of(int), typeid_of(uint),
	     typeid_of(f32), typeid_of(f64),
	     typeid_of(Date), typeid_of(Datetime), typeid_of(Time), typeid_of(Duration):
		return true
	case:
		return false
	}
}

// argsort_key_columns returns a stable permutation ordering rows by keys.
// All key columns must share the same row count (guaranteed by callers).
//
// core:sort is used with the Interface form: its `collection` rawptr carries
// the key context (this Odin build rejects closures, so merge_sort_proc's
// plain proc(T,T) -> int comparator cannot see the keys, and sort.sort is the
// only context-capable entry point). Stability: `sort.sort` is introsort
// (documented unstable), but perm[i] holds the original row index, so
// comparing (key, then index) is a strict total order whose result is
// identical to a stable key-only sort.
@(private)
argsort_key_columns :: proc(keys: []sort_key_internal, allocator: mem.Allocator) -> ([]int, Error) {
	n := keys[0].col.count

	// S21.2: a single non-categorical key with a u64 order-key mapping (any
	// sortable dtype except string) takes the radix kernel above
	// RADIX_SORT_THRESHOLD rows — identical ordering and stability semantics,
	// no per-comparison dispatch.
	if len(keys) == 1 && n >= RADIX_SORT_THRESHOLD {
		k := &keys[0]
		if !is_categorical(k.col) && radix_key_supported(k.col.dtype) {
			return radix_argsort_single(k^, allocator)
		}
	}

	perm := make([]int, n, allocator)
	if perm == nil && n != 0 {
		return nil, .Allocator_Failure
	}
	for i in 0 ..< n {
		perm[i] = i
	}

	view := sorter_view{perm = perm, keys = keys}
	it := sort.Interface {
		collection = &view,
		len        = proc(it: sort.Interface) -> int {
			v := (^sorter_view)(it.collection)
			return len(v.perm)
		},
		less = proc(it: sort.Interface, i, j: int) -> bool {
			v := (^sorter_view)(it.collection)
			if c := compare_rows(v.keys, v.perm[i], v.perm[j]); c != 0 {
				return c < 0
			}
			return v.perm[i] < v.perm[j] // index tiebreaker -> stable
		},
		swap = proc(it: sort.Interface, i, j: int) {
			v := (^sorter_view)(it.collection)
			v.perm[i], v.perm[j] = v.perm[j], v.perm[i]
		},
	}
	sort.sort(it)
	return perm, .None
}

// sorter_view carries the sortable state for core:sort.Interface. All members
// are borrowed; nothing is owned.
@(private)
sorter_view :: struct {
	perm: []int,
	keys: []sort_key_internal,
}

// compare_rows returns <0 if row a sorts before row b across keys, 0 on a
// tie, >0 otherwise.
@(private)
compare_rows :: proc(keys: []sort_key_internal, a, b: int) -> int {
	for k in keys {
		a_valid := row_valid(k.col.valid, a)
		b_valid := row_valid(k.col.valid, b)
		if a_valid != b_valid {
			if !a_valid {
				return k.nulls_first ? -1 : 1
			}
			return k.nulls_first ? 1 : -1
		}
		if !a_valid {
			continue // both NULL in this key: tie, try next key
		}
		if c := compare_values(k.col, a, b); c != 0 {
			return k.order == .Asc ? c : -c
		}
	}
	return 0
}

// compare_values compares the non-NULL values of column c at rows a and b.
// Callers ensure a and b are both valid. Categoricals order by their category
// string, not their code.
@(private)
compare_values :: proc(c: ^Column, a, b: int) -> int {
	if is_categorical(c) {
		sa, _ := categorical_value(c, a)
		sb, _ := categorical_value(c, b)
		return ordered_compare(sa, sb)
	}
	switch c.dtype {
	case typeid_of(bool):
		return ordered_compare(cmp_bool(c, a), cmp_bool(c, b))
	case typeid_of(string):
		return ordered_compare(column_typed_view(c, string)[a], column_typed_view(c, string)[b])
	case typeid_of(i8):
		return ordered_compare(column_typed_view(c, i8)[a], column_typed_view(c, i8)[b])
	case typeid_of(i16):
		return ordered_compare(column_typed_view(c, i16)[a], column_typed_view(c, i16)[b])
	case typeid_of(i32):
		return ordered_compare(column_typed_view(c, i32)[a], column_typed_view(c, i32)[b])
	case typeid_of(i64):
		return ordered_compare(column_typed_view(c, i64)[a], column_typed_view(c, i64)[b])
	case typeid_of(u8):
		return ordered_compare(column_typed_view(c, u8)[a], column_typed_view(c, u8)[b])
	case typeid_of(u16):
		return ordered_compare(column_typed_view(c, u16)[a], column_typed_view(c, u16)[b])
	case typeid_of(u32):
		return ordered_compare(column_typed_view(c, u32)[a], column_typed_view(c, u32)[b])
	case typeid_of(u64):
		return ordered_compare(column_typed_view(c, u64)[a], column_typed_view(c, u64)[b])
	case typeid_of(int):
		return ordered_compare(column_typed_view(c, int)[a], column_typed_view(c, int)[b])
	case typeid_of(uint):
		return ordered_compare(column_typed_view(c, uint)[a], column_typed_view(c, uint)[b])
	case typeid_of(f32):
		return ordered_compare(float32_bits(canonical_float32(column_typed_view(c, f32)[a])), float32_bits(canonical_float32(column_typed_view(c, f32)[b])))
	case typeid_of(f64):
		return ordered_compare(float64_bits(canonical_float64(column_typed_view(c, f64)[a])), float64_bits(canonical_float64(column_typed_view(c, f64)[b])))
	case typeid_of(Date):
		return ordered_compare(column_typed_view(c, Date)[a], column_typed_view(c, Date)[b])
	case typeid_of(Datetime):
		return ordered_compare(column_typed_view(c, Datetime)[a], column_typed_view(c, Datetime)[b])
	case typeid_of(Time):
		return ordered_compare(column_typed_view(c, Time)[a], column_typed_view(c, Time)[b])
	case typeid_of(Duration):
		return ordered_compare(column_typed_view(c, Duration)[a], column_typed_view(c, Duration)[b])
	case:
		return 0 // unreachable: sortable_dtype validated this already
	}
}

// cmp_bool maps false -> 0, true -> 1 for a total order on bools.
@(private)
cmp_bool :: proc(c: ^Column, i: int) -> u8 {
	if column_typed_view(c, bool)[i] {
		return 1
	}
	return 0
}

// float32_bits maps a float to an unsigned bit key with a total order:
// -inf < negatives < -0.0 == +0.0 < positives < +inf < NaN. NaN (canonical,
// from any NaN bit pattern) sorts after every finite value and +inf.
@(private)
float32_bits :: proc(v: f32) -> u32 {
	u := transmute(u32)v
	if (u & 0x8000_0000) != 0 {
		return ~u
	}
	return u | 0x8000_0000
}

// float64_bits is float32_bits for f64 (IEEE-754 total-order bit trick).
@(private)
float64_bits :: proc(v: f64) -> u64 {
	u := transmute(u64)v
	if (u & 0x8000_0000_0000_0000) != 0 {
		return ~u
	}
	return u | 0x8000_0000_0000_0000
}

// ordered_compare is the three-way comparator for any `<=>`-able type.
@(private)
ordered_compare :: proc(a, b: $T) -> int {
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}
