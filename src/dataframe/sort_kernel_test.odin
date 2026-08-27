package dataframe

// Stage 21 tests: S21.2 radix argsort kernel (sort_radix.odin).
//
// The oracle everywhere is an O(n^2) insertion sort driven by compare_rows —
// the exact comparator that defines the documented ordering — so any
// divergence between the u64 order-key extraction (order_key_u64) and the
// legacy semantics fails loudly. Private kernels are called directly so the
// paths are exercised regardless of RADIX_SORT_THRESHOLD.

import "core:testing"
import "core:fmt"
import "core:math"
import "core:mem"

// radix_oracle computes the stable permutation with insertion sort +
// compare_rows (plus the source-index tiebreaker, matching the legacy path).
@(private)
radix_oracle :: proc(keys: []sort_key_internal, allocator: mem.Allocator) -> []int {
	n := keys[0].col.count
	perm := make([]int, n, allocator)
	for i in 0 ..< n {
		perm[i] = i
	}
	for i in 1 ..< n {
		j := i
		for j > 0 {
			c := compare_rows(keys, perm[j - 1], perm[j])
			if c < 0 || (c == 0 && perm[j - 1] < perm[j]) {
				break
			}
			perm[j - 1], perm[j] = perm[j], perm[j - 1]
			j -= 1
		}
	}
	return perm
}

// radix_expect_perm asserts got matches want element-wise.
@(private)
radix_expect_perm :: proc(t: ^testing.T, what: string, got, want: []int) {
	testing.expectf(t, len(got) == len(want), "%s: len %d != %d", what, len(got), len(want))
	for i in 0 ..< len(want) {
		testing.expectf(
			t,
			got[i] == want[i],
			"%s: perm[%d] = %d, want %d",
			what,
			i,
			got[i],
			want[i],
		)
	}
}

// radix_u64_oracle stably sorts (key, index) pairs with insertion sort and
// returns the index permutation — reference for radix_sort_u64 itself.
@(private)
radix_u64_oracle :: proc(keys: []u64, allocator: mem.Allocator) -> []int {
	n := len(keys)
	idx := make([]int, n, allocator)
	for i in 0 ..< n {
		idx[i] = i
	}
	for i in 1 ..< n {
		j := i
		for j > 0 && keys[idx[j]] < keys[idx[j - 1]] {
			idx[j], idx[j - 1] = idx[j - 1], idx[j]
			j -= 1
		}
	}
	return idx
}

// radix_lcg is a tiny deterministic PRNG for reproducible test data.
@(private)
radix_lcg :: proc(state: ^u64) -> u64 {
	state^ = state^ * 6364136223846793005 + 1442695040888963407
	return state^ >> 33
}

@(test)
radix_sort_u64_matches_reference :: proc(t: ^testing.T) {
	cases := [][]u64{
		{},
		{7},
		{5, 5, 5, 5, 5},
		{0, 1, 2, 3, 4, 5}, // already sorted
		{9, 8, 7, 6, 5, 4}, // reverse
		{0x0102030405060708, 1, 0xFF00FF00FF00FF00, 42, 42, 1 << 63, (1 << 63) - 1},
		make([]u64, 300), // filled below: small-range values (few active bytes)
		make([]u64, 257), // filled below: full-width random values
	}
	seed := u64(0xDEADBEEF)
	for i in 0 ..< 300 {
		cases[6][i] = radix_lcg(&seed) % 1000 // top bytes zero: skip-digit path
	}
	for i in 0 ..< 257 {
		cases[7][i] = radix_lcg(&seed) // all bytes active
	}
	defer delete(cases[6], context.allocator)
	defer delete(cases[7], context.allocator)

	for cases_arr, ci in cases {
		want := radix_u64_oracle(cases_arr, context.allocator)
		got_idx := make([]int, len(cases_arr))
		for i in 0 ..< len(got_idx) {
			got_idx[i] = i
		}
		err := radix_sort_u64(cases_arr, got_idx, context.allocator)
		testing.expectf(t, err == .None, "case %d: radix error %v", ci, err)
		radix_expect_perm(t, fmt.tprintf("u64 case %d", ci), got_idx, want)
		delete(got_idx, context.allocator)
		delete(want, context.allocator)
		// Keys must come out ascending and aligned with idx.
		for i in 1 ..< len(cases_arr) {
			testing.expectf(
				t,
				cases_arr[i - 1] <= cases_arr[i],
				"case %d: keys[%d]=%d > keys[%d]=%d",
				ci,
				i - 1,
				cases_arr[i - 1],
				i,
				cases_arr[i],
			)
		}
	}
}

// radix_dtype_column builds one column of n rows per dtype with duplicated,
// negative, and boundary-ish values (deterministic).
@(private)
radix_dtype_column :: proc(t: ^testing.T, dtype: typeid, n: int, seed: u64) -> (c: Column) {
	switch dtype {
	case typeid_of(i64):
		vals := make([]i64, n)
		s := seed
		for i in 0 ..< n {
			vals[i] = i64(radix_lcg(&s) % 2000) - 1000
		}
		col, err := column_from("k", vals)
		delete(vals)
		testing.expect(t, err == .None, "col i64")
		return col
	case:
		panic("unsupported dtype in test helper")
	}
	return
}

@(test)
radix_argsort_i64_matches_legacy :: proc(t: ^testing.T) {
	col := radix_dtype_column(t, typeid_of(i64), 5000, 42)
	defer column_destroy(&col)

	for ord in ([]Sort_Order{.Asc, .Desc}) {
		for nf in ([]bool{false, true}) {
			keys := []sort_key_internal{{col = &col, order = ord, nulls_first = nf}}
			want := radix_oracle(keys, context.allocator)
			got, err := radix_argsort_single(keys[0], context.allocator)
			testing.expectf(t, err == .None, "radix error %v", err)
			radix_expect_perm(
				t,
				fmt.tprintf("i64 %v nulls_first=%t", ord, nf),
				got,
				want,
			)
			delete(got, context.allocator)
			delete(want, context.allocator)
		}
	}
}

@(test)
radix_argsort_signed_unsigned_bool :: proc(t: ^testing.T) {
	seed := u64(1234)

	i8v := make([]i8, 3000)
	for i in 0 ..< len(i8v) {
		i8v[i] = i8(radix_lcg(&seed) % 200) - 100
	}
	c_i8, e := column_from("k", i8v)
	delete(i8v)
	testing.expect(t, e == .None, "i8")
	defer column_destroy(&c_i8)

	u16v := make([]u16, 3000)
	for i in 0 ..< len(u16v) {
		u16v[i] = u16(radix_lcg(&seed) % 60000)
	}
	c_u16, e2 := column_from("k", u16v)
	delete(u16v)
	testing.expect(t, e2 == .None, "u16")
	defer column_destroy(&c_u16)

	u64v := make([]u64, 3000)
	for i in 0 ..< len(u64v) {
		u64v[i] = radix_lcg(&seed) | (1 << 63) // high bit set: complement traps
	}
	c_u64, e3 := column_from("k", u64v)
	delete(u64v)
	testing.expect(t, e3 == .None, "u64")
	defer column_destroy(&c_u64)

	bv := make([]bool, 3000)
	for i in 0 ..< len(bv) {
		bv[i] = radix_lcg(&seed) % 2 == 0
	}
	c_b, e4 := column_from("k", bv)
	delete(bv)
	testing.expect(t, e4 == .None, "bool")
	defer column_destroy(&c_b)

	colset := [4]^Column{&c_i8, &c_u16, &c_u64, &c_b}
	for cols in colset {
		keys := []sort_key_internal{{col = cols, order = .Asc}}
		want := radix_oracle(keys, context.allocator)
		got, err := radix_argsort_single(keys[0], context.allocator)
		testing.expectf(t, err == .None, "radix error %v", err)
		radix_expect_perm(t, fmt.tprintf("dtype %v", cols.dtype), got, want)
		delete(got, context.allocator)
		delete(want, context.allocator)
	}
}

@(test)
radix_argsort_float_specials :: proc(t: ^testing.T) {
	nan_bits := transmute(f32)u32(0x7F80_0001)
	f32v := []f32{
		1.5, -0.0, 0.0, nan_bits, math.inf_f32(1), -math.inf_f32(1), 7.25,
		-3.5, transmute(f32)u32(0xFFC0_0001), 0.25, -1e30, 1e30,
		0.0, -0.0, 42.0,
	}
	c_f32, e1 := column_from("k", f32v)
	testing.expect(t, e1 == .None, "f32")
	defer column_destroy(&c_f32)

	f64v := []f64{
		math.nan_f64(), -0.0, 0.0, math.inf_f64(1), -math.inf_f64(1),
		-2.5, 2.5, transmute(f64)u64(0xFFF8_0000_0000_0001), 0.125, -0.0,
		math.min(f64), math.max(f64), 0.0,
	}
	c_f64, e2 := column_from("k", f64v)
	testing.expect(t, e2 == .None, "f64")
	defer column_destroy(&c_f64)

	colset := [2]^Column{&c_f32, &c_f64}
	for cols in colset {
		for ord in ([]Sort_Order{.Asc, .Desc}) {
			keys := []sort_key_internal{{col = cols, order = ord}}
			want := radix_oracle(keys, context.allocator)
			got, err := radix_argsort_single(keys[0], context.allocator)
			testing.expectf(t, err == .None, "radix error %v", err)
			radix_expect_perm(
				t,
				fmt.tprintf("%v %v", cols.dtype, ord),
				got,
				want,
			)
			delete(got, context.allocator)
			delete(want, context.allocator)
		}
	}
}

@(test)
radix_argsort_temporal_types :: proc(t: ^testing.T) {
	dt := make([]Datetime, 2500)
	for i in 0 ..< len(dt) {
		dt[i] = Datetime(i64(i * 7919 % 100000) - 50000) // negatives included
	}
	c_dt, e1 := column_from("k", dt)
	delete(dt)
	testing.expect(t, e1 == .None, "datetime")
	defer column_destroy(&c_dt)

	dur := make([]Duration, 2500)
	for i in 0 ..< len(dur) {
		dur[i] = Duration(i64(2500 - i)) // strictly descending source order
	}
	c_dur, e2 := column_from("k", dur)
	delete(dur)
	testing.expect(t, e2 == .None, "duration")
	defer column_destroy(&c_dur)

	days := make([]Date, 2500)
	for i in 0 ..< len(days) {
		days[i] = Date(i64(i % 365)) // many ties: stability visible
	}
	c_d, e3 := column_from("k", days)
	delete(days)
	testing.expect(t, e3 == .None, "date")
	defer column_destroy(&c_d)

	colset := [3]^Column{&c_dt, &c_dur, &c_d}
	for cols in colset {
		for ord in ([]Sort_Order{.Asc, .Desc}) {
			keys := []sort_key_internal{{col = cols, order = ord}}
			want := radix_oracle(keys, context.allocator)
			got, err := radix_argsort_single(keys[0], context.allocator)
			testing.expectf(t, err == .None, "radix error %v", err)
			radix_expect_perm(t, fmt.tprintf("temporal %v", cols.dtype), got, want)
			delete(got, context.allocator)
			delete(want, context.allocator)
		}
	}
}

@(test)
radix_argsort_nulls :: proc(t: ^testing.T) {
	vals := make([]i64, 4000)
	for i in 0 ..< len(vals) {
		vals[i] = i64(i % 37) // heavy duplication
	}
	col, e := column_from("k", vals)
	delete(vals)
	testing.expect(t, e == .None, "col")
	defer column_destroy(&col)
	for i in 0 ..< len(vals) {
		if i % 5 == 0 {
			testing.expect(t, column_set_null(&col, i) == .None, "set null")
		}
	}

	for nf in ([]bool{false, true}) {
		keys := []sort_key_internal{{col = &col, order = .Asc, nulls_first = nf}}
		want := radix_oracle(keys, context.allocator)
		got, err := radix_argsort_single(keys[0], context.allocator)
		testing.expectf(t, err == .None, "radix error %v", err)
		radix_expect_perm(t, fmt.tprintf("nulls_first=%t", nf), got, want)
		delete(got, context.allocator)
		delete(want, context.allocator)
	}
}

@(test)
radix_argsort_degenerate :: proc(t: ^testing.T) {
	empty, e0 := column_from("k", []i64{})
	testing.expect(t, e0 == .None, "empty")
	defer column_destroy(&empty)

	one, e1 := column_from("k", []i64{9})
	testing.expect(t, e1 == .None, "one")
	defer column_destroy(&one)
	testing.expect(t, column_set_null(&one, 0) == .None, "null it")

	all_eq, e2 := column_from("k", []i64{3, 3, 3, 3})
	testing.expect(t, e2 == .None, "all eq")
	defer column_destroy(&all_eq)

	all_null, e3 := column_from("k", []i64{1, 2, 3})
	testing.expect(t, e3 == .None, "all null")
	defer column_destroy(&all_null)
	for i in 0 ..< 3 {
		testing.expect(t, column_set_null(&all_null, i) == .None, "null")
	}

	colset := [4]^Column{&empty, &one, &all_eq, &all_null}
	for cols in colset {
		for nf in ([]bool{false, true}) {
			keys := []sort_key_internal{{col = cols, order = .Asc, nulls_first = nf}}
			want := radix_oracle(keys, context.allocator)
			got, err := radix_argsort_single(keys[0], context.allocator)
			testing.expectf(t, err == .None, "radix error %v", err)
			radix_expect_perm(t, fmt.tprintf("degenerate %v", cols.dtype), got, want)
			delete(got, context.allocator)
			delete(want, context.allocator)
		}
	}
}

// radix_dispatch_threshold_equivalence drives the real dispatch in
// argsort_key_columns just above RADIX_SORT_THRESHOLD and compares with the
// oracle: the kernel selection must not change results.
@(test)
radix_dispatch_threshold_equivalence :: proc(t: ^testing.T) {
	n := RADIX_SORT_THRESHOLD + 128
	vals := make([]i64, n)
	ids := make([]i32, n)
	seed := u64(777)
	for i in 0 ..< n {
		vals[i] = i64(radix_lcg(&seed) % 5000) - 2500
		ids[i] = i32(i)
	}
	c_id, e := column_from("id", ids)
	delete(ids)
	testing.expect(t, e == .None, "id")
	kcol, e2 := column_from("k", vals)
	delete(vals)
	testing.expect(t, e2 == .None, "k")

	// dataframe_from_columns takes ownership of the columns.
	df, derr := dataframe_from_columns([]^Column{&c_id, &kcol})
	testing.expect(t, derr == .None, "df")
	defer dataframe_destroy(&df)

	kref, kerr := dataframe_get_column(&df, "k")
	testing.expect(t, kerr == .None, "get k")

	for ord in ([]Sort_Order{.Asc, .Desc}) {
		perm, perr := dataframe_argsort(&df, []Sort_Key{sort_key("k", ord)})
		testing.expectf(t, perr == .None, "argsort error %v", perr)
		defer delete(perm, context.allocator)
		keys := []sort_key_internal{{col = kref, order = ord}}
		want := radix_oracle(keys, context.allocator)
		defer delete(want, context.allocator)
		radix_expect_perm(t, fmt.tprintf("dispatch %v", ord), perm, want)
	}
}

// radix_end_to_end_sorted_output runs dataframe_sort on a large frame and
// verifies the materialized key column is non-decreasing.
@(test)
radix_end_to_end_sorted_output :: proc(t: ^testing.T) {
	n := 200_000
	vals := make([]i64, n)
	ids := make([]i64, n)
	seed := u64(20260824)
	for i in 0 ..< n {
		vals[i] = i64(radix_lcg(&seed) % 1_000_000)
		ids[i] = i64(i)
	}
	c_id, e1 := column_from("id", ids)
	delete(ids)
	testing.expect(t, e1 == .None, "id")
	defer column_destroy(&c_id)
	c_v, e2 := column_from("v", vals)
	delete(vals)
	testing.expect(t, e2 == .None, "v")
	defer column_destroy(&c_v)
	df, derr := dataframe_from_columns([]^Column{&c_id, &c_v})
	testing.expect(t, derr == .None, "df")
	defer dataframe_destroy(&df)

	out, serr := dataframe_sort(&df, []Sort_Key{sort_key("v")})
	testing.expect(t, serr == .None, "sort")
	defer dataframe_destroy(&out)

	v_out, gerr := dataframe_get_column(&out, "v")
	testing.expect(t, gerr == .None, "v out")
	view := column_typed_view(v_out, i64)
	for i in 1 ..< n {
		testing.expectf(
			t,
			view[i - 1] <= view[i],
			"out[%d]=%d > out[%d]=%d",
			i - 1,
			view[i - 1],
			i,
			view[i],
		)
	}
}
