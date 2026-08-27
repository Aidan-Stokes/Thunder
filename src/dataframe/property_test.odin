package dataframe

// Stage 10 property tests (S10.6): invariants checked over pseudo-random
// data. The PRNG is seeded so runs are deterministic.

import "core:testing"
import "core:fmt"
import "expr"

// --- deterministic PRNG ------------------------------------------------------

prng :: struct { state: u64 }

prng_next :: proc(r: ^prng) -> u64 {
	r.state ~= r.state << 13
	r.state ~= r.state >> 7
	r.state ~= r.state << 17
	return r.state
}

// prng_int returns a value in [0, n).
prng_int :: proc(r: ^prng, n: int) -> int {
	return int(prng_next(r) % u64(n))
}

// prng_f64 returns a value in [lo, hi).
prng_f64 :: proc(r: ^prng, lo, hi: f64) -> f64 {
	return lo + f64(prng_next(r) % 1_000_000) / 999_999.0 * (hi - lo)
}

// --- comparison helpers ------------------------------------------------------

// df_equal compares two DataFrames by schema, row count, values, and NULL
// validity. Only values on valid rows are compared (NULL value slots are
// unspecified).
df_equal :: proc(a, b: ^DataFrame) -> bool {
	if a == b {
		return true
	}
	if dataframe_num_rows(a) != dataframe_num_rows(b) {
		return false
	}
	if dataframe_num_cols(a) != dataframe_num_cols(b) {
		return false
	}
	for i in 0 ..< dataframe_num_cols(a) {
		ca, ca_err := dataframe_column_at(a, i)
		cb, cb_err := dataframe_column_at(b, i)
		if ca_err != .None || cb_err != .None || !column_equal(ca, cb) {
			return false
		}
	}
	return true
}

column_equal :: proc(a, b: ^Column) -> bool {
	if a.name != b.name || a.dtype != b.dtype || a.count != b.count {
		return false
	}
	for i in 0 ..< a.count {
		if column_is_valid(a, i) != column_is_valid(b, i) {
			return false
		}
		if column_is_valid(a, i) && !column_slot_equal(a, b, i) {
			return false
		}
	}
	return true
}

// column_slot_equal compares the typed value of two columns at row i.
column_slot_equal :: proc(a, b: ^Column, i: int) -> bool {
	switch a.dtype {
	case typeid_of(bool):   return column_typed_view(a, bool)[i] == column_typed_view(b, bool)[i]
	case typeid_of(i8):     return column_typed_view(a, i8)[i] == column_typed_view(b, i8)[i]
	case typeid_of(i16):    return column_typed_view(a, i16)[i] == column_typed_view(b, i16)[i]
	case typeid_of(i32):    return column_typed_view(a, i32)[i] == column_typed_view(b, i32)[i]
	case typeid_of(i64):    return column_typed_view(a, i64)[i] == column_typed_view(b, i64)[i]
	case typeid_of(u8):     return column_typed_view(a, u8)[i] == column_typed_view(b, u8)[i]
	case typeid_of(u16):    return column_typed_view(a, u16)[i] == column_typed_view(b, u16)[i]
	case typeid_of(u32):    return column_typed_view(a, u32)[i] == column_typed_view(b, u32)[i]
	case typeid_of(u64):    return column_typed_view(a, u64)[i] == column_typed_view(b, u64)[i]
	case typeid_of(int):    return column_typed_view(a, int)[i] == column_typed_view(b, int)[i]
	case typeid_of(uint):   return column_typed_view(a, uint)[i] == column_typed_view(b, uint)[i]
	case typeid_of(f32):    return column_typed_view(a, f32)[i] == column_typed_view(b, f32)[i]
	case typeid_of(f64):    return column_typed_view(a, f64)[i] == column_typed_view(b, f64)[i]
	case typeid_of(string): return column_typed_view(a, string)[i] == column_typed_view(b, string)[i]
	}
	return false
}

// --- fixture -----------------------------------------------------------------

// prop_df builds an n-row frame with id (i64), grp (i32), cat (string), and
// val (f64 with ~14% NULLs), plus a fresh expression context.
prop_df :: proc(t: ^testing.T, r: ^prng, n: int) -> (df: DataFrame, ctx: expr.Context) {
	id := make([]i64, n)
	grp := make([]i32, n)
	cat := make([]string, n)
	val := make([]f64, n)
	valid := make([]bool, n)
	names := []string{"a", "b", "c"}
	for i in 0 ..< n {
		id[i] = i64(prng_next(r))
		grp[i] = i32(prng_int(r, 4))
		cat[i] = names[prng_int(r, len(names))]
		val[i] = prng_f64(r, -10.0, 10.0)
		valid[i] = prng_int(r, 7) != 0
	}

	id_c, e1 := column_from("id", id)
	grp_c, e2 := column_from("grp", grp)
	cat_c, e3 := column_from("cat", cat)
	val_c, e4 := column_from_with_valid("val", val, valid)
	delete(id)
	delete(grp)
	delete(cat)
	delete(val)
	delete(valid)
	testing.expect(t, e1 == .None, "id column in prop_df")
	testing.expect(t, e2 == .None, "grp column in prop_df")
	testing.expect(t, e3 == .None, "cat column in prop_df")
	testing.expect(t, e4 == .None, "val column in prop_df")

	d, df_err := dataframe_from_columns([]^Column{&id_c, &grp_c, &cat_c, &val_c})
	testing.expect(t, df_err == .None, "from_columns in prop_df")
	df = d
	ctx = expr.context_create(context.allocator)
	return
}

// --- sort idempotence --------------------------------------------------------

@(test)
property_sort_idempotent :: proc(t: ^testing.T) {
	r := prng{state = 0x9E3779B97F4A7C15}
	for iter in 0 ..< 6 {
		df, ctx := prop_df(t, &r, 60 + iter * 30)
		defer ops_test_destroy(t, &df, &ctx)

		by := []Sort_Key{sort_key("grp"), sort_key("cat"), sort_key("val")}
		s1, err1 := dataframe_sort(&df, by)
		defer dataframe_destroy(&s1)
		s2, err2 := dataframe_sort(&s1, by)
		defer dataframe_destroy(&s2)
		testing.expectf(t, err1 == .None && err2 == .None, "iter %d: sort errors %v %v", iter, err1, err2)
		testing.expectf(t, df_equal(&s1, &s2), "iter %d: sort(sort(df)) != sort(df)", iter)
	}
}

// --- filter invariants -------------------------------------------------------

@(test)
property_filter_row_count_monotonic :: proc(t: ^testing.T) {
	r := prng{state = 0x6A09E667F3BCC909}
	for iter in 0 ..< 6 {
		df, ctx := prop_df(t, &r, 80 + iter * 20)
		defer ops_test_destroy(t, &df, &ctx)

		pred := expr.and_(
			&ctx,
			expr.gt(&ctx, expr.col(&ctx, "val"), expr.lit(&ctx, f64(0))),
			expr.is_not_null(&ctx, expr.col(&ctx, "val")),
		)
		out, err := dataframe_filter(&df, pred)
		defer dataframe_destroy(&out)
		testing.expectf(t, err == .None, "iter %d: filter error", iter)
		testing.expectf(t, dataframe_num_rows(&out) <= dataframe_num_rows(&df),
			"iter %d: filter grew the frame", iter)
	}
}

@(test)
property_filter_idempotent :: proc(t: ^testing.T) {
	r := prng{state = 0xBB67AE8584CAA73B}
	for iter in 0 ..< 6 {
		df, ctx := prop_df(t, &r, 70 + iter * 25)
		defer ops_test_destroy(t, &df, &ctx)

		pred := expr.eq(&ctx, expr.col(&ctx, "grp"), expr.lit(&ctx, i32(1)))
		f1, err1 := dataframe_filter(&df, pred)
		defer dataframe_destroy(&f1)
		f2, err2 := dataframe_filter(&f1, pred)
		defer dataframe_destroy(&f2)
		testing.expectf(t, err1 == .None && err2 == .None, "iter %d: filter errors", iter)
		testing.expectf(t, df_equal(&f1, &f2), "iter %d: filter(filter(df, p), p) != filter(df, p)", iter)
	}
}

// --- select idempotence ------------------------------------------------------

@(test)
property_select_idempotent :: proc(t: ^testing.T) {
	r := prng{state = 0x3C6EF372FE94F82B}
	for iter in 0 ..< 6 {
		df, ctx := prop_df(t, &r, 60 + iter * 20)
		defer ops_test_destroy(t, &df, &ctx)

		exprs := []^expr.Expr{expr.col(&ctx, "cat"), expr.col(&ctx, "val")}
		s1, err1 := dataframe_select(&df, exprs)
		defer dataframe_destroy(&s1)
		s2, err2 := dataframe_select(&s1, exprs)
		defer dataframe_destroy(&s2)
		testing.expectf(t, err1 == .None && err2 == .None, "iter %d: select errors", iter)
		testing.expectf(t, df_equal(&s1, &s2), "iter %d: select(select(df, e), e) != select(df, e)", iter)
	}
}
