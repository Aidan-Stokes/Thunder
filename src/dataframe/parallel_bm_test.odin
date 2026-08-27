package dataframe

// S15.11 tests: parallel bm_from_bools and bm_count_false correctness.

import "core:fmt"
import "core:testing"

@(test)
test_parallel_bm_from_bools_all_true :: proc(t: ^testing.T) {
	n := 100_000
	src := make([]bool, n, context.allocator)
	defer delete(src, context.allocator)
	for i in 0 ..< n {
		src[i] = true
	}
	bits := parallel_bm_from_bools(src, context.allocator)
	defer delete(bits, context.allocator)
	testing.expect(t, bits != nil, "bits should not be nil")
	for i in 0 ..< n {
		testing.expectf(t, bm_get(bits, i), "bm_from_bools_all_true[%d]: should be true", i)
	}
}

@(test)
test_parallel_bm_from_bools_all_false :: proc(t: ^testing.T) {
	n := 100_000
	src := make([]bool, n, context.allocator)
	defer delete(src, context.allocator)
	for i in 0 ..< n {
		src[i] = false
	}
	bits := parallel_bm_from_bools(src, context.allocator)
	defer delete(bits, context.allocator)
	testing.expect(t, bits != nil, "bits should not be nil")
	for i in 0 ..< n {
		testing.expectf(t, !bm_get(bits, i), "bm_from_bools_all_false[%d]: should be false", i)
	}
}

@(test)
test_parallel_bm_from_bools_mixed :: proc(t: ^testing.T) {
	n := 100_000
	src := make([]bool, n, context.allocator)
	defer delete(src, context.allocator)
	for i in 0 ..< n {
		src[i] = i % 3 != 0
	}
	bits := parallel_bm_from_bools(src, context.allocator)
	defer delete(bits, context.allocator)
	testing.expect(t, bits != nil, "bits should not be nil")
	for i in 0 ..< n {
		expected := i % 3 != 0
		testing.expectf(t, bm_get(bits, i) == expected, "bm_from_bools_mixed[%d]: got %v want %v", i, bm_get(bits, i), expected)
	}
}

@(test)
test_parallel_bm_from_bools_nil :: proc(t: ^testing.T) {
	bits := parallel_bm_from_bools(nil, context.allocator)
	testing.expect(t, bits == nil, "nil src should return nil")
}

@(test)
test_parallel_bm_count_false_all_valid :: proc(t: ^testing.T) {
	n := 100_000
	valid := make([]u64, bm_words(n), context.allocator)
	defer delete(valid, context.allocator)
	bm_init_all(valid)
	cnt := parallel_bm_count_false(valid, n)
	testing.expect(t, cnt == 0, fmt.tprintf("all valid: got %v want 0", cnt))
}

@(test)
test_parallel_bm_count_false_all_null :: proc(t: ^testing.T) {
	n := 100_000
	valid := make([]u64, bm_words(n), context.allocator)
	defer delete(valid, context.allocator)
	// all zeros by default
	cnt := parallel_bm_count_false(valid, n)
	testing.expect(t, cnt == n, fmt.tprintf("all null: got %v want %v", cnt, n))
}

@(test)
test_parallel_bm_count_false_mixed :: proc(t: ^testing.T) {
	n := 100_000
	valid := make([]u64, bm_words(n), context.allocator)
	defer delete(valid, context.allocator)
	expected := 0
	for i in 0 ..< n {
		v := i % 3 != 0
		bm_set(valid, i, v)
		if !v {
			expected += 1
		}
	}
	cnt := parallel_bm_count_false(valid, n)
	testing.expectf(t, cnt == expected, "mixed: got %v want %v", cnt, expected)
}

@(test)
test_parallel_bm_count_false_below_threshold :: proc(t: ^testing.T) {
	n := 1000
	valid := make([]u64, bm_words(n), context.allocator)
	defer delete(valid, context.allocator)
	for i in 0 ..< n {
		bm_set(valid, i, i % 2 == 0)
	}
	cnt := parallel_bm_count_false(valid, n)
	expected := n / 2
	testing.expectf(t, cnt == expected, "below_threshold: got %v want %v", cnt, expected)
}
