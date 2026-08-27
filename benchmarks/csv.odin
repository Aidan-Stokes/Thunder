// csv.odin — Stage 9.7 benchmark: CSV parse over 1K / 100K / 1M rows.
//
// Run with: odin run benchmarks/csv.odin -file
//
// Workload: a synthetic 4-column CSV (id i64, score f64, ok bool, name
// string) with ~1% NULL rows. Each size is generated to a scratch file,
// parsed with dataframe_read_csv, and timed end-to-end. Reported as ms and
// rows/s; also shows the file size for MB/s. The parse is single-threaded
// stdlib glue (core:encoding/csv), so this is the baseline S15.7 (parallel
// chunked CSV parsing) must beat. No stdlib harness — manual timing, same
// as the other benchmarks in this directory.

package main

import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"

import df "../src/dataframe"

SIZES := []int{1_000, 100_000, 1_000_000, 10_000_000}

// write_csv builds a 4-column CSV of n rows and returns path + byte size.
write_csv :: proc(n: int) -> (path: string, size: int) {
	path = fmt.tprintf("/tmp/thunder_bench_csv_%d.csv", n)

	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)
	strings.write_string(&sb, "id,score,ok,name\n")
	for i in 0 ..< n {
		score := f64(i % 1000) / 8.0
		ok := i % 2 == 0
		if i % 97 == 0 {
			// empty field -> NULL on read
			fmt.sbprintf(&sb, "%d,,%t,name%07d\n", i, ok, i)
		} else {
			fmt.sbprintf(&sb, "%d,%.4f,%t,name%07d\n", i, score, ok, i)
		}
	}
	text := strings.to_string(sb)
	if os_err := os.write_entire_file_from_string(path, text); os_err != os.ERROR_NONE {
		fmt.println("  write", path, "failed:", os_err)
	}
	return path, len(text)
}

// bench_parse reads path three times (after a warm-up pass) and returns the
// best (minimum) elapsed time in ms.
bench_parse :: proc(path: string) -> f64 {
	warm, w_err := df.dataframe_read_csv(path)
	if w_err != .None {
		fmt.println("  warmup failed:", w_err)
		return 0
	}
	df.dataframe_destroy(&warm)

	best := f64(1e18)
	for _ in 0 ..< 3 {
		start := time.now()
		data, r_err := df.dataframe_read_csv(path)
		elapsed := time.duration_milliseconds(time.since(start))
		df.dataframe_destroy(&data)
		if r_err != .None {
			fmt.println("  parse failed:", r_err)
			return 0
		}
		best = min(best, elapsed)
	}
	return best
}

main :: proc() {
	fmt.printf("%12s | %12s | %12s | %12s | %10s\n", "rows", "time(ms)", "rows/s", "size(B)", "MB/s")
	for n in SIZES {
		path, size := write_csv(n)
		defer os.remove(path)

		ms := bench_parse(path)
		secs := ms / 1000.0
		fmt.printf("%12d | %12.3f | %12.0f | %12d | %10.2f\n", n, ms, f64(n) / secs, size, f64(size) / secs / 1e6)
	}
}
