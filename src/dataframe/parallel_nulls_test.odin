package dataframe

// S15.10 tests: parallel fill_null and coalesce correctness for non-string
// types. Verifies that parallel versions produce identical results to the
// sequential paths at sizes above and below the threshold.

import "core:fmt"
import "core:testing"

@(test)
test_parallel_fill_null_i64 :: proc(t: ^testing.T) {
	n := 100_000
	iv := make([]i64, n, context.allocator)
	valid := make([]bool, n, context.allocator)
	defer {
		delete(iv, context.allocator)
		delete(valid, context.allocator)
	}
	for i in 0 ..< n {
		iv[i] = i64(i)
		valid[i] = i % 2 != 0
	}
	col, err := column_from_with_valid("v", iv, valid, context.allocator)
	testing.expect(t, err == .None, fmt.tprintf("create: %v", err))
	defer column_destroy(&col)

	fill_val: i64 = -1
	parallel_fill_null(&col, &fill_val, n)

	view := column_typed_view(&col, i64)
	for i in 0 ..< n {
		if i % 2 == 0 {
			testing.expectf(t, view[i] == -1, "fill_null_i64[%d]: got %v want -1", i, view[i])
		} else {
			testing.expectf(t, view[i] == i64(i), "fill_null_i64[%d]: got %v want %v", i, view[i], i)
		}
	}
}

@(test)
test_parallel_fill_null_f64 :: proc(t: ^testing.T) {
	n := 100_000
	iv := make([]f64, n, context.allocator)
	valid := make([]bool, n, context.allocator)
	defer {
		delete(iv, context.allocator)
		delete(valid, context.allocator)
	}
	for i in 0 ..< n {
		iv[i] = f64(i) * 0.5
		valid[i] = i % 3 != 0
	}
	col, err := column_from_with_valid("v", iv, valid, context.allocator)
	testing.expect(t, err == .None, fmt.tprintf("create: %v", err))
	defer column_destroy(&col)

	fill_val: f64 = -99.0
	parallel_fill_null(&col, &fill_val, n)

	view := column_typed_view(&col, f64)
	for i in 0 ..< n {
		if i % 3 == 0 {
			testing.expectf(t, view[i] == -99.0, "fill_null_f64[%d]: got %v want -99", i, view[i])
		} else {
			testing.expectf(t, view[i] == f64(i) * 0.5, "fill_null_f64[%d]: got %v want %v", i, view[i], f64(i) * 0.5)
		}
	}
}

@(test)
test_parallel_fill_null_below_threshold :: proc(t: ^testing.T) {
	n := 1000
	iv := make([]i32, n, context.allocator)
	valid := make([]bool, n, context.allocator)
	defer {
		delete(iv, context.allocator)
		delete(valid, context.allocator)
	}
	for i in 0 ..< n {
		iv[i] = i32(i)
		valid[i] = i % 2 != 0
	}
	col, err := column_from_with_valid("v", iv, valid, context.allocator)
	testing.expect(t, err == .None, fmt.tprintf("create: %v", err))
	defer column_destroy(&col)

	fill_val: i32 = -1
	parallel_fill_null(&col, &fill_val, n)

	view := column_typed_view(&col, i32)
	for i in 0 ..< n {
		if i % 2 == 0 {
			testing.expectf(t, view[i] == -1, "fill_null_below[%d]: got %v want -1", i, view[i])
		} else {
			testing.expectf(t, view[i] == i32(i), "fill_null_below[%d]: got %v want %v", i, view[i], i32(i))
		}
	}
}

@(test)
test_parallel_coalesce_i64 :: proc(t: ^testing.T) {
	n := 100_000

	// col_a: all NULL
	iv_a := make([]i64, n, context.allocator)
	valid_a := make([]bool, n, context.allocator)
	for i in 0 ..< n {
		iv_a[i] = 100
		valid_a[i] = false
	}
	col_a, err_a := column_from_with_valid("a", iv_a, valid_a, context.allocator)
	testing.expect(t, err_a == .None, fmt.tprintf("create a: %v", err_a))
	defer column_destroy(&col_a)

	// col_b: even rows valid = i*10, odd rows NULL
	iv_b := make([]i64, n, context.allocator)
	valid_b := make([]bool, n, context.allocator)
	for i in 0 ..< n {
		iv_b[i] = i64(i * 10)
		valid_b[i] = i % 2 == 0
	}
	col_b, err_b := column_from_with_valid("b", iv_b, valid_b, context.allocator)
	testing.expect(t, err_b == .None, fmt.tprintf("create b: %v", err_b))
	defer column_destroy(&col_b)

	cols := []Column{col_a, col_b}
	count := col_a.count
	size := col_a.elem_size
	out, oerr := column_alloc(context.allocator, "out", typeid_of(i64), size, align_of(i64), count)
	testing.expect(t, oerr == .None, fmt.tprintf("alloc: %v", oerr))
	defer column_destroy(&out)

	parallel_coalesce_non_string(&out, cols, count)

	view := column_typed_view(&out, i64)
	for i in 0 ..< n {
		if i % 2 == 0 {
			// col_b has valid data = i*10
			testing.expectf(t, view[i] == i64(i * 10), "coalesce_i64[%d]: got %v want %v", i, view[i], i64(i * 10))
		} else {
			// col_a is NULL, col_b is NULL -> NULL (column_set_null)
			testing.expectf(t, !bm_get(out.valid, i), "coalesce_i64[%d]: should be NULL", i)
		}
	}
}

@(test)
test_parallel_coalesce_falls_through :: proc(t: ^testing.T) {
	n := 100_000

	// col_a: only first 50% valid
	iv_a := make([]f64, n, context.allocator)
	valid_a := make([]bool, n, context.allocator)
	for i in 0 ..< n {
		iv_a[i] = 1.0
		valid_a[i] = i < n / 2
	}
	col_a, err_a := column_from_with_valid("a", iv_a, valid_a, context.allocator)
	testing.expect(t, err_a == .None, fmt.tprintf("create a: %v", err_a))
	defer column_destroy(&col_a)

	// col_b: only second 50% valid
	iv_b := make([]f64, n, context.allocator)
	valid_b := make([]bool, n, context.allocator)
	for i in 0 ..< n {
		iv_b[i] = 2.0
		valid_b[i] = i >= n / 2
	}
	col_b, err_b := column_from_with_valid("b", iv_b, valid_b, context.allocator)
	testing.expect(t, err_b == .None, fmt.tprintf("create b: %v", err_b))
	defer column_destroy(&col_b)

	cols := []Column{col_a, col_b}
	count := col_a.count
	size := col_a.elem_size
	out, oerr := column_alloc(context.allocator, "out", typeid_of(f64), size, align_of(f64), count)
	testing.expect(t, oerr == .None, fmt.tprintf("alloc: %v", oerr))
	defer column_destroy(&out)

	parallel_coalesce_non_string(&out, cols, count)

	view := column_typed_view(&out, f64)
	for i in 0 ..< n {
		if i < n / 2 {
			testing.expectf(t, view[i] == 1.0, "coalesce_fall[%d]: got %v want 1.0", i, view[i])
		} else {
			testing.expectf(t, view[i] == 2.0, "coalesce_fall[%d]: got %v want 2.0", i, view[i])
		}
	}
}
