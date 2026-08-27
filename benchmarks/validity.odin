// validity.odin — Stage 15.1 benchmark: []bool vs packed-bitmap validity.
//
// Run with: odin run benchmarks/validity.odin -file
//
// Question (ROADMAP S15.1): replace Column.valid ([]bool) with a packed
// bitmap? Per principle 11, only if a benchmark justifies it. This file
// isolates the two representations and compares them on the access patterns
// that the dataframe hot paths actually use:
//
//   sum-scan     agg kernels (sum/mean/var/...):  if valid: sum += v[i]
//   count-scan   count / filter row counting
//   gather       join / groupby / filter on a permuted row order
//   scatter-set  building validity while parsing (column_set_valid)
//
// Three representations are timed:
//   nil      valid == nil (no array at all, all rows valid) — the ideal case
//            the current design already exploits (zero memory, one branch)
//   bool     []bool, current Column.valid representation
//   bitmap   bits packed in []u64
//
// NULL rates: 1% (sparse, e.g. a mostly-clean CSV column) and 50% (dense,
// e.g. a join result). Measurements are best-of-3; results in milliseconds.
// Memory is reported as bytes/row of validity storage.
//
// The nil row shows what a fully-valid column costs today; the bool-vs-bitmap
// columns show the cost of *having* NULLs under each representation.

package main

import "core:fmt"
import "core:time"

SIZES := []int{100_000, 1_000_000, 10_000_000}

bm_get :: proc(bits: []u64, i: int) -> bool {
	return (bits[i >> 6] & (u64(1) << uint(i & 63))) != 0
}
bm_set :: proc(bits: []u64, i: int, v: bool) {
	if v {
		bits[i >> 6] |= u64(1) << uint(i & 63)
	} else {
		bits[i >> 6] &= ~(u64(1) << uint(i & 63))
	}
}

// --- timing helpers (no closures — call directly) ----------------------------

sum_scan_nil :: proc(v: []f64) -> f64 {
	start := time.now()
	s := 0.0
	for x in v {
		s += x
	}
	return time.duration_milliseconds(time.since(start))
}

sum_scan_bool :: proc(v: []f64, valid: []bool) -> f64 {
	start := time.now()
	s := 0.0
	for i in 0 ..< len(v) {
		if valid[i] {
			s += v[i]
		}
	}
	return time.duration_milliseconds(time.since(start))
}

sum_scan_bitmap :: proc(v: []f64, bits: []u64) -> f64 {
	start := time.now()
	s := 0.0
	for i in 0 ..< len(v) {
		if bm_get(bits, i) {
			s += v[i]
		}
	}
	return time.duration_milliseconds(time.since(start))
}

gather_bool :: proc(valid: []bool, n: int) -> f64 {
	start := time.now()
	sum := 0
	idx := 0
	for _ in 0 ..< n {
		if valid[idx] { sum += idx }
		idx = (idx + 7919) % n
	}
	return time.duration_milliseconds(time.since(start))
}

gather_bitmap :: proc(bits: []u64, n: int) -> f64 {
	start := time.now()
	sum := 0
	idx := 0
	for _ in 0 ..< n {
		if bm_get(bits, idx) { sum += idx }
		idx = (idx + 7919) % n
	}
	return time.duration_milliseconds(time.since(start))
}

scatter_set_bool :: proc(n: int) -> f64 {
	valid := make([]bool, n)
	defer delete(valid)
	start := time.now()
	for i := 0; i < n; i += 100 {
		valid[i] = false
	}
	return time.duration_milliseconds(time.since(start))
}

scatter_set_bitmap :: proc(n: int) -> f64 {
	bits := make([]u64, (n + 63) / 64)
	defer delete(bits)
	for i in 0 ..< len(bits) {
		bits[i] = ~u64(0)
	}
	start := time.now()
	for i := 0; i < n; i += 100 {
		bm_set(bits, i, false)
	}
	return time.duration_milliseconds(time.since(start))
}

// --- report functions (best-of-3, inline) ------------------------------------

report_sum_scan :: proc(null_pct: int) {
	fmt.printf("\nsum scan, %d%% NULLs — ms (best of 3)\n", null_pct)
	fmt.printf("%12s | %10s | %10s | %10s\n", "rows", "nil", "bool", "bitmap")
	for n in SIZES {
		v := make([]f64, n)
		defer delete(v)
		valid := make([]bool, n)
		defer delete(valid)
		bits := make([]u64, (n + 63) / 64)
		defer delete(bits)

		stride := 100 if null_pct == 1 else 2
		for i in 0 ..< n {
			v[i] = f64(i)
			valid[i] = i % stride != 0
			if valid[i] { bm_set(bits, i, true) }
		}

		best_nil  := min(sum_scan_nil(v),  min(sum_scan_nil(v),  sum_scan_nil(v)))
		best_bool := min(sum_scan_bool(v, valid), min(sum_scan_bool(v, valid), sum_scan_bool(v, valid)))
		best_bm   := min(sum_scan_bitmap(v, bits), min(sum_scan_bitmap(v, bits), sum_scan_bitmap(v, bits)))
		fmt.printf("%12d | %10.3f | %10.3f | %10.3f\n", n, best_nil, best_bool, best_bm)
	}
}

report_gather :: proc(null_pct: int) {
	fmt.printf("\ngather (permuted order), %d%% NULLs — ms (best of 3)\n", null_pct)
	fmt.printf("%12s | %10s | %10s\n", "rows", "bool", "bitmap")
	for n in SIZES {
		valid := make([]bool, n)
		defer delete(valid)
		bits := make([]u64, (n + 63) / 64)
		defer delete(bits)

		stride := 100 if null_pct == 1 else 2
		for i in 0 ..< n {
			valid[i] = i % stride != 0
			if valid[i] { bm_set(bits, i, true) }
		}

		best_bool := min(gather_bool(valid, n), min(gather_bool(valid, n), gather_bool(valid, n)))
		best_bm   := min(gather_bitmap(bits, n), min(gather_bitmap(bits, n), gather_bitmap(bits, n)))
		fmt.printf("%12d | %10.3f | %10.3f\n", n, best_bool, best_bm)
	}
}

report_set :: proc() {
	fmt.printf("\nscatter-set every 100th row invalid — ms (best of 3)\n")
	fmt.printf("%12s | %10s | %10s\n", "rows", "bool", "bitmap")
	for n in SIZES {
		best_bool := min(scatter_set_bool(n), min(scatter_set_bool(n), scatter_set_bool(n)))
		best_bm   := min(scatter_set_bitmap(n), min(scatter_set_bitmap(n), scatter_set_bitmap(n)))
		fmt.printf("%12d | %10.3f | %10.3f\n", n, best_bool, best_bm)
	}
}

report_memory :: proc() {
	fmt.printf("\nvalidity memory (bytes)\n")
	fmt.printf("%12s | %10s | %10s | %10s\n", "rows", "bool", "bitmap", "saved")
	for n in SIZES {
		b := n * size_of(bool)
		bm := (n + 63) / 64 * size_of(u64)
		fmt.printf("%12d | %10d | %10d | %7.1f%%\n", n, b, bm, 100.0 * f64(b - bm) / f64(b))
	}
}

main :: proc() {
	fmt.println("Stage 15.1 — validity representation: []bool vs packed bitmap")
	report_sum_scan(1)
	report_sum_scan(50)
	report_gather(1)
	report_gather(50)
	report_set()
	report_memory()
}
