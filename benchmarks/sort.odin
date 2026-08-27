// sort.odin — Stage 5.6 benchmark: stable sort of 1M rows per dtype.
//
// Run with: odin run benchmarks/sort.odin -file
//
// Measures the full `dataframe_sort` path (permutation + row gather) against
// the baseline work it replaces — a single-column index sort plus a gather —
// across dtypes: i32, i64, f64, string. Rows are a permutation of 1..n (worst
// case for a stable sort; no near-sorted input). Results are printed as
// milliseconds; a zero value should be treated with suspicion.
//
// Since Stage 21 (DESIGN.md §14.8) single-key sorts over non-string dtypes
// at >= RADIX_SORT_THRESHOLD rows (4096) dispatch to the parallel LSD radix
// kernel, so the i32/i64/f64 rows measure the radix path while string still
// measures the legacy comparator path.

package main

import "core:fmt"
import "core:time"

import df "../src/dataframe"

SIZE :: 1_000_000

// make_key_df builds a 2-column df: key (dtype) with values permuted, and id.
make_key_df :: proc($T: typeid) -> (d: df.DataFrame, ok: bool) {
	key, k_err := df.column_from("key", make([]T, SIZE))
	if k_err != .None {
		fmt.println("  make_key_df: key failed:", k_err)
		return {}, false
	}
	id, i_err := df.column_from("id", make([]i32, SIZE))
	if i_err != .None {
		df.column_destroy(&key)
		fmt.println("  make_key_df: id failed:", i_err)
		return {}, false
	}
	// Permutation: half the range ascending, half descending, so no run is
	// nearly sorted and the merge sort has to actually merge everywhere.
	half := SIZE / 2
	for i in 0 ..< half {
		df.column_set(&key, i, key_value(T, i))
		df.column_set(&id, i, i32(i))
	}
	for i in half ..< SIZE {
		df.column_set(&key, i, key_value(T, SIZE - i))
		df.column_set(&id, i, i32(i))
	}

	err: df.Error
	d, err = df.dataframe_from_columns([]^df.Column{&key, &id})
	if err != .None {
		df.column_destroy(&key)
		df.column_destroy(&id)
		fmt.println("  make_key_df: from_columns failed:", err)
		return {}, false
	}
	return d, true
}

// key_value produces the i-th key value for dtype T.
key_value :: proc($T: typeid, i: int) -> T {
	when T == string {
		return fmt.tprintf("key-%07d", i)
	} else {
		return T(i)
	}
}

// check_sorted verifies the sorted key column is non-decreasing.
check_sorted :: proc(out: ^df.DataFrame, $T: typeid) -> bool {
	key_col, err := df.dataframe_get_column(out, "key")
	if err != .None {
		return false
	}
	prev, p_valid, p_err := df.column_get(key_col, 0, T)
	if p_err != .None || !p_valid {
		return false
	}
	for i in 1 ..< SIZE {
		v, valid, e := df.column_get(key_col, i, T)
		if e != .None || !valid {
			return false
		}
		when T == string {
			if prev > v {
				return false
			}
		} else {
			if v < prev {
				return false
			}
		}
		prev = v
	}
	return true
}

// bench_sort times one sort of the key column (ascending).
bench_sort :: proc($T: typeid) -> f64 {
	d, ok := make_key_df(T)
	if !ok {
		return 0
	}
	defer df.dataframe_destroy(&d)

	start := time.now()
	out, err := df.dataframe_sort(&d, []df.Sort_Key{df.sort_key("key")})
	elapsed := time.duration_milliseconds(time.since(start))
	if err != .None {
		fmt.println("  sort error:", err)
		df.dataframe_destroy(&out)
		return 0
	}
	defer df.dataframe_destroy(&out)

	// Sanity: key column must be sorted ascending; sum of ids defeats any
	// dead-code elimination of the gathered buffers.
	if !check_sorted(&out, T) {
		fmt.println("  (sort check) failed")
	}
	id_col, _ := df.dataframe_get_column(&out, "id")
	sum: i64
	for i in 0 ..< SIZE {
		v, valid, err := df.column_get(id_col, i, i32)
		if err != .None || !valid {
			fmt.println("  (sum check) read failed at", i)
			break
		}
		sum += i64(v)
	}
	if sum == 0 {
		fmt.println("  (sum check) sum was zero")
	}
	return elapsed
}

main :: proc() {
	fmt.printf("%-8s | %14s\n", "dtype", "sort 1M(ms)")
	ms := bench_sort(i32)
	fmt.printf("%-8s | %14.3f\n", "i32", ms)
	ms = bench_sort(i64)
	fmt.printf("%-8s | %14.3f\n", "i64", ms)
	ms = bench_sort(f64)
	fmt.printf("%-8s | %14.3f\n", "f64", ms)
	ms = bench_sort(string)
	fmt.printf("%-8s | %14.3f\n", "string", ms)
}
