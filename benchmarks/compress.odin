// compress.odin — Compression benchmark: LZ4 block, LZ4 frame over 1KB / 64KB / 1MB.
//
// Run with: odin run benchmarks/compress.odin -file

package main

import "core:fmt"
import "core:time"

import df "../src/dataframe"

BENCH_SIZES :: []int{1_024, 65_536, 1_048_576}

fill_repetitive :: proc(buf: []byte) {
	for i in 0 ..< len(buf) {
		buf[i] = byte(i % 128)
	}
}

fill_randomish :: proc(buf: []byte) {
	seed: u64 = 0xDEADBEEF
	for i in 0 ..< len(buf) {
		seed = seed * 6364136223846793005 + 1442695040888963407
		buf[i] = byte(seed >> 56)
	}
}

bench_compress_decompress :: proc(label: string, algo: df.Compression_Algorithm, data: []byte) {
	warm, _ := df.compress(data, algo)
	df.compressed_buffer_destroy(&warm)

	N := 20

	t0 := time.now()
	for _ in 0 ..< N {
		buf, _ := df.compress(data, algo)
		if buf.data != nil {
			delete(buf.data, buf.alloc)
		}
	}
	elapsed_compress_ms := time.duration_milliseconds(time.since(t0))

	compressed, _ := df.compress(data, algo)
	defer df.compressed_buffer_destroy(&compressed)

	t1 := time.now()
	for _ in 0 ..< N {
		result, _ := df.decompress(&compressed, len(data))
		if result != nil {
			delete(result)
		}
	}
	elapsed_decompress_ms := time.duration_milliseconds(time.since(t1))

	src_mb := f64(len(data)) / (1024.0 * 1024.0)
	avg_compress := elapsed_compress_ms / f64(N)
	avg_decompress := elapsed_decompress_ms / f64(N)

	fmt.printf(
		"  %-12s %8d B -> %8d B (%.1f%%)  compress: %6.2f ms (%6.1f MB/s)  decompress: %6.2f ms (%6.1f MB/s)\n",
		label,
		len(data), len(compressed.data),
		f64(len(compressed.data)) / f64(len(data)) * 100.0,
		avg_compress,
		src_mb / (avg_compress / 1000.0),
		avg_decompress,
		src_mb / (avg_decompress / 1000.0),
	)
}

main :: proc() {
	fmt.println("Compression benchmark — LZ4 block vs LZ4 frame")
	fmt.println("================================================")

	for size in BENCH_SIZES {
		data := make([]byte, size)
		defer delete(data)
		fill_repetitive(data)

		fmt.printf("\n--- %d bytes (%.1f KB, repetitive) ---\n", size, f64(size) / 1024.0)
		bench_compress_decompress("LZ4_Block", .LZ4_Block, data)
		bench_compress_decompress("LZ4_Frame", .LZ4_Frame, data)
	}

	for size in BENCH_SIZES {
		data := make([]byte, size)
		defer delete(data)
		fill_randomish(data)

		fmt.printf("\n--- %d bytes (%.1f KB, random) ---\n", size, f64(size) / 1024.0)
		bench_compress_decompress("LZ4_Block", .LZ4_Block, data)
		bench_compress_decompress("LZ4_Frame", .LZ4_Frame, data)
	}
}
