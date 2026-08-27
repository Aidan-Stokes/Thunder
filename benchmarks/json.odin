// json.odin — Stage 13.4 benchmark: JSON parse and write over 1K / 100K / 1M
// rows.
//
// Run with: odin run benchmarks/json.odin -file
//
// Workload: a synthetic 4-column JSON array (id i64, score f64, ok bool,
// name string) with ~1% NULL rows. Each size is generated to a scratch file,
// parsed with dataframe_read_json, then re-written with dataframe_write_json,
// and each timed end-to-end. Reported as ms and rows/s; also shows the file
// size for MB/s. Both directions are single-threaded stdlib glue
// (core:encoding/json), so this is the baseline parallel JSON parsing must
// beat. No stdlib harness — manual timing, same as the other benchmarks in
// this directory.

package main

import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"

import df "../src/dataframe"

SIZES := []int{1_000, 100_000, 1_000_000}

// write_json builds a 4-column JSON file of n rows and returns path + size.
write_json :: proc(n: int) -> (path: string, size: int) {
	path = fmt.tprintf("/tmp/thunder_bench_json_%d.json", n)

	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)
	strings.write_string(&sb, "[")
	for i in 0 ..< n {
		if i != 0 {
			strings.write_string(&sb, ",")
		}
		if i % 97 == 0 {
			fmt.sbprintf(&sb, `{{"id": %d, "score": null, "ok": %t, "name": "name%07d"}}`, i, i % 2 == 0, i)
		} else {
			score := f64(i % 1000) / 8.0
			fmt.sbprintf(&sb, `{{"id": %d, "score": %.4f, "ok": %t, "name": "name%07d"}}`, i, score, i % 2 == 0, i)
		}
	}
	strings.write_string(&sb, "]")
	text := strings.to_string(sb)
	if os_err := os.write_entire_file_from_string(path, text); os_err != os.ERROR_NONE {
		fmt.println("  write", path, "failed:", os_err)
	}
	return path, len(text)
}

// bench_parse reads path three times (after a warm-up pass) and returns the
// best (minimum) elapsed time in ms.
bench_parse :: proc(path: string) -> f64 {
	warm, w_err := df.dataframe_read_json(path)
	if w_err != .None {
		fmt.println("  warmup failed:", w_err)
		return 0
	}
	df.dataframe_destroy(&warm)

	best := f64(1e18)
	for _ in 0 ..< 3 {
		start := time.now()
		data, r_err := df.dataframe_read_json(path)
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

// bench_write writes df to a scratch path three times and returns the best
// elapsed time in ms.
bench_write :: proc(data: ^df.DataFrame) -> f64 {
	best := f64(1e18)
	for _ in 0 ..< 3 {
		path := fmt.tprintf("/tmp/thunder_bench_json_out_%d.json", len(data.columns))
		start := time.now()
		w_err := df.dataframe_write_json(data, path)
		elapsed := time.duration_milliseconds(time.since(start))
		os.remove(path)
		if w_err != .None {
			fmt.println("  write failed:", w_err)
			return 0
		}
		best = min(best, elapsed)
	}
	return best
}

main :: proc() {
	fmt.printf("%12s | %12s | %12s | %12s | %10s | %12s | %12s\n", "rows", "read(ms)", "read rows/s", "write(ms)", "write rows/s", "size(B)", "MB/s")
	for n in SIZES {
		path, size := write_json(n)
		defer os.remove(path)

		r_ms := bench_parse(path)
		r_secs := r_ms / 1000.0

		data, r_err := df.dataframe_read_json(path)
		if r_err != .None {
			fmt.println("  load for write bench failed:", r_err)
			continue
		}
		w_ms := bench_write(&data)
		df.dataframe_destroy(&data)
		w_secs := w_ms / 1000.0

		fmt.printf("%12d | %12.3f | %12.0f | %12.3f | %12.0f | %12d | %10.2f\n",
			n, r_ms, f64(n) / r_secs, w_ms, f64(n) / w_secs, size, f64(size) / r_secs / 1e6)
	}
}
