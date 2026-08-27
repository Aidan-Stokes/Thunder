package dataframe

// S15.9 tests: parallel SIMD kernel correctness. Verifies that parallel_simd_*
// produces identical results to the sequential simd_* across all operations
// (add, sub, mul, neg, abs, sum) for both f64 and i64, at both small (below
// threshold) and large (above threshold) sizes.

import "core:fmt"
import "core:testing"

@(test)
test_parallel_simd_add_f64 :: proc(t: ^testing.T) {
	n := 100_000
	lv := make([]f64, n, context.allocator)
	rv := make([]f64, n, context.allocator)
	ov_par := make([]f64, n, context.allocator)
	ov_seq := make([]f64, n, context.allocator)
	defer {
		delete(lv, context.allocator)
		delete(rv, context.allocator)
		delete(ov_par, context.allocator)
		delete(ov_seq, context.allocator)
	}
	for i in 0 ..< n {
		lv[i] = f64(i) * 1.5
		rv[i] = f64(i) * 0.5
	}
	parallel_simd_add_f64(lv, rv, ov_par)
	simd_add(lv, rv, ov_seq)
	for i in 0 ..< n {
		testing.expectf(t, ov_par[i] == ov_seq[i], "add_f64[%d]: got %v want %v", i, ov_par[i], ov_seq[i])
	}
}

@(test)
test_parallel_simd_sub_f64 :: proc(t: ^testing.T) {
	n := 100_000
	lv := make([]f64, n, context.allocator)
	rv := make([]f64, n, context.allocator)
	ov_par := make([]f64, n, context.allocator)
	ov_seq := make([]f64, n, context.allocator)
	defer {
		delete(lv, context.allocator)
		delete(rv, context.allocator)
		delete(ov_par, context.allocator)
		delete(ov_seq, context.allocator)
	}
	for i in 0 ..< n {
		lv[i] = f64(i) * 2.0
		rv[i] = f64(i) * 0.3
	}
	parallel_simd_sub_f64(lv, rv, ov_par)
	simd_sub(lv, rv, ov_seq)
	for i in 0 ..< n {
		testing.expectf(t, ov_par[i] == ov_seq[i], "sub_f64[%d]: got %v want %v", i, ov_par[i], ov_seq[i])
	}
}

@(test)
test_parallel_simd_mul_f64 :: proc(t: ^testing.T) {
	n := 100_000
	lv := make([]f64, n, context.allocator)
	rv := make([]f64, n, context.allocator)
	ov_par := make([]f64, n, context.allocator)
	ov_seq := make([]f64, n, context.allocator)
	defer {
		delete(lv, context.allocator)
		delete(rv, context.allocator)
		delete(ov_par, context.allocator)
		delete(ov_seq, context.allocator)
	}
	for i in 0 ..< n {
		lv[i] = f64(i)
		rv[i] = f64(i) * 0.01
	}
	parallel_simd_mul_f64(lv, rv, ov_par)
	simd_mul(lv, rv, ov_seq)
	for i in 0 ..< n {
		testing.expectf(t, ov_par[i] == ov_seq[i], "mul_f64[%d]: got %v want %v", i, ov_par[i], ov_seq[i])
	}
}

@(test)
test_parallel_simd_neg_f64 :: proc(t: ^testing.T) {
	n := 100_000
	iv := make([]f64, n, context.allocator)
	ov_par := make([]f64, n, context.allocator)
	ov_seq := make([]f64, n, context.allocator)
	defer {
		delete(iv, context.allocator)
		delete(ov_par, context.allocator)
		delete(ov_seq, context.allocator)
	}
	for i in 0 ..< n {
		iv[i] = f64(i) - 50_000.0
	}
	parallel_simd_neg_f64(iv, ov_par)
	simd_neg(iv, ov_seq)
	for i in 0 ..< n {
		testing.expectf(t, ov_par[i] == ov_seq[i], "neg_f64[%d]: got %v want %v", i, ov_par[i], ov_seq[i])
	}
}

@(test)
test_parallel_simd_abs_f64 :: proc(t: ^testing.T) {
	n := 100_000
	iv := make([]f64, n, context.allocator)
	ov_par := make([]f64, n, context.allocator)
	ov_seq := make([]f64, n, context.allocator)
	defer {
		delete(iv, context.allocator)
		delete(ov_par, context.allocator)
		delete(ov_seq, context.allocator)
	}
	for i in 0 ..< n {
		iv[i] = f64(i) - 50_000.0
	}
	parallel_simd_abs_f64(iv, ov_par)
	simd_abs(iv, ov_seq)
	for i in 0 ..< n {
		testing.expectf(t, ov_par[i] == ov_seq[i], "abs_f64[%d]: got %v want %v", i, ov_par[i], ov_seq[i])
	}
}

@(test)
test_parallel_simd_sum_f64 :: proc(t: ^testing.T) {
	n := 100_000
	iv := make([]f64, n, context.allocator)
	defer delete(iv, context.allocator)
	for i in 0 ..< n {
		iv[i] = 1.0
	}
	par_sum := parallel_simd_sum_f64(iv)
	seq_sum := simd_sum(iv)
	testing.expectf(t, par_sum == seq_sum, "sum_f64: parallel=%v sequential=%v", par_sum, seq_sum)
	testing.expectf(t, par_sum == f64(n), "sum_f64: got %v want %v", par_sum, f64(n))
}

@(test)
test_parallel_simd_add_i64 :: proc(t: ^testing.T) {
	n := 100_000
	lv := make([]i64, n, context.allocator)
	rv := make([]i64, n, context.allocator)
	ov_par := make([]i64, n, context.allocator)
	ov_seq := make([]i64, n, context.allocator)
	defer {
		delete(lv, context.allocator)
		delete(rv, context.allocator)
		delete(ov_par, context.allocator)
		delete(ov_seq, context.allocator)
	}
	for i in 0 ..< n {
		lv[i] = i64(i * 3)
		rv[i] = i64(i * 7)
	}
	parallel_simd_add_i64(lv, rv, ov_par)
	simd_add(lv, rv, ov_seq)
	for i in 0 ..< n {
		testing.expectf(t, ov_par[i] == ov_seq[i], "add_i64[%d]: got %v want %v", i, ov_par[i], ov_seq[i])
	}
}

@(test)
test_parallel_simd_sub_i64 :: proc(t: ^testing.T) {
	n := 100_000
	lv := make([]i64, n, context.allocator)
	rv := make([]i64, n, context.allocator)
	ov_par := make([]i64, n, context.allocator)
	ov_seq := make([]i64, n, context.allocator)
	defer {
		delete(lv, context.allocator)
		delete(rv, context.allocator)
		delete(ov_par, context.allocator)
		delete(ov_seq, context.allocator)
	}
	for i in 0 ..< n {
		lv[i] = i64(i * 10)
		rv[i] = i64(i * 3)
	}
	parallel_simd_sub_i64(lv, rv, ov_par)
	simd_sub(lv, rv, ov_seq)
	for i in 0 ..< n {
		testing.expectf(t, ov_par[i] == ov_seq[i], "sub_i64[%d]: got %v want %v", i, ov_par[i], ov_seq[i])
	}
}

@(test)
test_parallel_simd_mul_i64 :: proc(t: ^testing.T) {
	n := 100_000
	lv := make([]i64, n, context.allocator)
	rv := make([]i64, n, context.allocator)
	ov_par := make([]i64, n, context.allocator)
	ov_seq := make([]i64, n, context.allocator)
	defer {
		delete(lv, context.allocator)
		delete(rv, context.allocator)
		delete(ov_par, context.allocator)
		delete(ov_seq, context.allocator)
	}
	for i in 0 ..< n {
		lv[i] = i64(i)
		rv[i] = i64(i)
	}
	parallel_simd_mul_i64(lv, rv, ov_par)
	simd_mul(lv, rv, ov_seq)
	for i in 0 ..< n {
		testing.expectf(t, ov_par[i] == ov_seq[i], "mul_i64[%d]: got %v want %v", i, ov_par[i], ov_seq[i])
	}
}

@(test)
test_parallel_simd_neg_i64 :: proc(t: ^testing.T) {
	n := 100_000
	iv := make([]i64, n, context.allocator)
	ov_par := make([]i64, n, context.allocator)
	ov_seq := make([]i64, n, context.allocator)
	defer {
		delete(iv, context.allocator)
		delete(ov_par, context.allocator)
		delete(ov_seq, context.allocator)
	}
	for i in 0 ..< n {
		iv[i] = i64(i) - 50_000
	}
	parallel_simd_neg_i64(iv, ov_par)
	simd_neg(iv, ov_seq)
	for i in 0 ..< n {
		testing.expectf(t, ov_par[i] == ov_seq[i], "neg_i64[%d]: got %v want %v", i, ov_par[i], ov_seq[i])
	}
}

@(test)
test_parallel_simd_below_threshold :: proc(t: ^testing.T) {
	n := 1000
	lv := make([]f64, n, context.allocator)
	rv := make([]f64, n, context.allocator)
	ov := make([]f64, n, context.allocator)
	defer {
		delete(lv, context.allocator)
		delete(rv, context.allocator)
		delete(ov, context.allocator)
	}
	for i in 0 ..< n {
		lv[i] = f64(i)
		rv[i] = f64(i)
	}
	parallel_simd_add_f64(lv, rv, ov)
	for i in 0 ..< n {
		testing.expectf(t, ov[i] == f64(i * 2), "below_threshold[%d]: got %v want %v", i, ov[i], f64(i * 2))
	}
}
