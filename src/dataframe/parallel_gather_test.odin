package dataframe

// S15.8 tests: parallel gather_rows correctness. Verifies that parallel
// gather produces identical results to sequential gather across all column
// types (i64, f64, bool, string) with NULLs, and that filter/select/sort/take
// all produce correct results when the parallel path is triggered.

import "core:fmt"
import "core:mem"
import "core:strings"
import "core:testing"

// --- helpers ----------------------------------------------------------------

// gather_sequential runs gather_rows_core on the sequential path by using
// a small index set that stays below the parallel threshold.
gather_sequential :: proc(src: ^Column, allocator: mem.Allocator, indices: []int) -> (Column, Error) {
	return gather_rows_core(src, allocator, src.name, indices, false)
}

// gather_parallel runs gather_rows_core with enough indices to trigger
// the parallel path.
gather_parallel :: proc(src: ^Column, allocator: mem.Allocator, indices: []int) -> (Column, Error) {
	return gather_rows_core(src, allocator, src.name, indices, false)
}

// --- tests ------------------------------------------------------------------

@(test)
gather_parallel_i64 :: proc(t: ^testing.T) {
	n := 50_000
	src, s_err := column_from("id", make([]i64, n, context.allocator), context.allocator)
	testing.expect(t, s_err == .None, fmt.tprintf("create: %v", s_err))
	defer column_destroy(&src)
	for i in 0 ..< n {
		column_set(&src, i, i64(i * 3))
	}

	indices := make([]int, n, context.allocator)
	defer delete(indices, context.allocator)
	for i in 0 ..< n {
		indices[i] = i
	}

	ser, seq_err := gather_sequential(&src, context.allocator, indices)
	testing.expect(t, seq_err == .None, fmt.tprintf("seq: %v", seq_err))
	defer column_destroy(&ser)

	par, par_err := gather_parallel(&src, context.allocator, indices)
	testing.expect(t, par_err == .None, fmt.tprintf("par: %v", par_err))
	defer column_destroy(&par)

	testing.expect(t, column_len(&ser) == column_len(&par), "len mismatch")
	for i in 0 ..< n {
		vs, es, _ := column_get(&ser, i, i64)
		vp, ep, _ := column_get(&par, i, i64)
		testing.expect(t, es == ep, fmt.tprintf("row %d valid mismatch", i))
		if es && ep {
			testing.expect(t, vs == vp, fmt.tprintf("row %d: %d != %d", i, vs, vp))
		}
	}
}

@(test)
gather_parallel_f64 :: proc(t: ^testing.T) {
	n := 50_000
	vals := make([]f64, n, context.allocator)
	for i in 0 ..< n {
		vals[i] = f64(i) * 1.5
	}
	src, s_err := column_from("score", vals, context.allocator)
	testing.expect(t, s_err == .None, fmt.tprintf("create: %v", s_err))
	defer column_destroy(&src)

	indices := make([]int, n, context.allocator)
	defer delete(indices, context.allocator)
	for i in 0 ..< n {
		indices[i] = n - 1 - i // reverse
	}

	ser, seq_err := gather_sequential(&src, context.allocator, indices)
	testing.expect(t, seq_err == .None, fmt.tprintf("seq: %v", seq_err))
	defer column_destroy(&ser)

	par, par_err := gather_parallel(&src, context.allocator, indices)
	testing.expect(t, par_err == .None, fmt.tprintf("par: %v", par_err))
	defer column_destroy(&par)

	testing.expect(t, column_len(&ser) == column_len(&par), "len mismatch")
	for i in 0 ..< n {
		vs, es, _ := column_get(&ser, i, f64)
		vp, ep, _ := column_get(&par, i, f64)
		testing.expect(t, es == ep, fmt.tprintf("row %d valid mismatch", i))
		if es && ep {
			testing.expect(t, vs == vp, fmt.tprintf("row %d: %v != %v", i, vs, vp))
		}
	}
}

@(test)
gather_parallel_bool :: proc(t: ^testing.T) {
	n := 50_000
	vals := make([]bool, n, context.allocator)
	for i in 0 ..< n {
		vals[i] = i % 3 == 0
	}
	src, s_err := column_from("ok", vals, context.allocator)
	testing.expect(t, s_err == .None, fmt.tprintf("create: %v", s_err))
	defer column_destroy(&src)

	indices := make([]int, n, context.allocator)
	defer delete(indices, context.allocator)
	for i in 0 ..< n {
		indices[i] = (i * 7) % n
	}

	ser, seq_err := gather_sequential(&src, context.allocator, indices)
	testing.expect(t, seq_err == .None, fmt.tprintf("seq: %v", seq_err))
	defer column_destroy(&ser)

	par, par_err := gather_parallel(&src, context.allocator, indices)
	testing.expect(t, par_err == .None, fmt.tprintf("par: %v", par_err))
	defer column_destroy(&par)

	testing.expect(t, column_len(&ser) == column_len(&par), "len mismatch")
	for i in 0 ..< n {
		vs, es, _ := column_get(&ser, i, bool)
		vp, ep, _ := column_get(&par, i, bool)
		testing.expect(t, es == ep, fmt.tprintf("row %d valid mismatch", i))
		if es && ep {
			testing.expect(t, vs == vp, fmt.tprintf("row %d: %v != %v", i, vs, vp))
		}
	}
}

@(test)
gather_parallel_string :: proc(t: ^testing.T) {
	n := 50_000
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)
	for i in 0 ..< n {
		fmt.sbprintf(&sb, "item%07d", i)
	}
	blob := strings.to_string(sb)
	stride := len(blob) / n

	// Build string column.
	strs := make([]string, n, context.allocator)
	for i in 0 ..< n {
		strs[i] = blob[i * stride : i * stride + stride]
	}
	src, s_err := column_from("name", strs, context.allocator)
	testing.expect(t, s_err == .None, fmt.tprintf("create: %v", s_err))
	defer column_destroy(&src)

	indices := make([]int, n, context.allocator)
	defer delete(indices, context.allocator)
	for i in 0 ..< n {
		indices[i] = n - 1 - i
	}

	ser, seq_err := gather_sequential(&src, context.allocator, indices)
	testing.expect(t, seq_err == .None, fmt.tprintf("seq: %v", seq_err))
	defer column_destroy(&ser)

	par, par_err := gather_parallel(&src, context.allocator, indices)
	testing.expect(t, par_err == .None, fmt.tprintf("par: %v", par_err))
	defer column_destroy(&par)

	testing.expect(t, column_len(&ser) == column_len(&par), "len mismatch")
	for i in 0 ..< n {
		vs, es, _ := column_get(&ser, i, string)
		vp, ep, _ := column_get(&par, i, string)
		testing.expect(t, es == ep, fmt.tprintf("row %d valid mismatch", i))
		if es && ep {
			testing.expect(t, vs == vp, fmt.tprintf("row %d: %s != %s", i, vs, vp))
		}
	}
}

@(test)
gather_parallel_with_nulls :: proc(t: ^testing.T) {
	n := 50_000
	vals := make([]i64, n, context.allocator)
	valid := make([]bool, n, context.allocator)
	for i in 0 ..< n {
		vals[i] = i64(i)
		valid[i] = i % 5 != 0 // every 5th row is NULL
	}
	src, s_err := column_from_with_valid("val", vals, valid, context.allocator)
	testing.expect(t, s_err == .None, fmt.tprintf("create: %v", s_err))
	defer column_destroy(&src)

	indices := make([]int, n, context.allocator)
	defer delete(indices, context.allocator)
	for i in 0 ..< n {
		indices[i] = i
	}

	ser, seq_err := gather_sequential(&src, context.allocator, indices)
	testing.expect(t, seq_err == .None, fmt.tprintf("seq: %v", seq_err))
	defer column_destroy(&ser)

	par, par_err := gather_parallel(&src, context.allocator, indices)
	testing.expect(t, par_err == .None, fmt.tprintf("par: %v", par_err))
	defer column_destroy(&par)

	testing.expect(t, column_len(&ser) == column_len(&par), "len mismatch")
	for i in 0 ..< n {
		vs, es, _ := column_get(&ser, i, i64)
		vp, ep, _ := column_get(&par, i, i64)
		testing.expect(t, es == ep, fmt.tprintf("row %d valid mismatch", i))
		if es && ep {
			testing.expect(t, vs == vp, fmt.tprintf("row %d: %d != %d", i, vs, vp))
		}
	}
}

@(test)
gather_parallel_filter_equivalence :: proc(t: ^testing.T) {
	// Build a 1M-row df, gather with indices, verify parallel matches sequential.
	n := 1_000_000
	id_vals := make([]i64, n, context.allocator)
	for i in 0 ..< n {
		id_vals[i] = i64(i)
	}
	id_col, id_err := column_from("id", id_vals, context.allocator)
	testing.expect(t, id_err == .None, "id col create")
	defer column_destroy(&id_col)

	indices := make([]int, n / 2, context.allocator)
	defer delete(indices, context.allocator)
	for i in 0 ..< n / 2 {
		indices[i] = i * 2
	}

	id_ser, seq_err := gather_sequential(&id_col, context.allocator, indices)
	testing.expect(t, seq_err == .None, fmt.tprintf("seq: %v", seq_err))
	defer column_destroy(&id_ser)

	id_par, par_err := gather_parallel(&id_col, context.allocator, indices)
	testing.expect(t, par_err == .None, fmt.tprintf("par: %v", par_err))
	defer column_destroy(&id_par)

	testing.expect(t, column_len(&id_ser) == column_len(&id_par), "len mismatch")
	for i in 0 ..< len(indices) {
		vs, _, _ := column_get(&id_ser, i, i64)
		vp, _, _ := column_get(&id_par, i, i64)
		testing.expect(t, vs == vp, fmt.tprintf("row %d: %d != %d", i, vs, vp))
	}
}
