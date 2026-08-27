package dataframe

import "core:testing"

// --- .None (identity) -------------------------------------------------------

@(test)
compress_none_round_trip :: proc(t: ^testing.T) {
	src := "Hello, compression world!"
	src_bytes := transmute([]byte)(src)

	buf, err := compress(src_bytes, .None)
	testing.expect(t, err == .None, "compress .None should succeed")
	defer compressed_buffer_destroy(&buf)

	testing.expect(t, len(buf.data) == len(src_bytes), "compressed size should match")
	testing.expect(t, buf.algo == .None, "algo should be .None")

	result, err2 := decompress(&buf, len(src_bytes))
	defer delete(result)
	testing.expect(t, err2 == .None, "decompress .None should succeed")
	testing.expect(t, string(result) == src, "round-trip should match")
}

@(test)
compress_none_empty :: proc(t: ^testing.T) {
	buf, err := compress(nil, .None)
	testing.expect(t, err == .None, "compress empty .None should succeed")
	defer compressed_buffer_destroy(&buf)
	testing.expect(t, buf.data == nil, "empty compressed buffer should be nil")

	result, err2 := decompress(&buf, 0)
	testing.expect(t, err2 == .None, "decompress empty .None should succeed")
	testing.expect(t, result == nil, "decompressed empty should be nil")
}

// --- LZ4 Block (raw) --------------------------------------------------------

@(test)
compress_lz4_block_round_trip :: proc(t: ^testing.T) {
	src := "The quick brown fox jumps over the lazy dog. "
	src_bytes := transmute([]byte)(src)

	buf, err := compress(src_bytes, .LZ4_Block)
	testing.expect(t, err == .None, "compress LZ4_Block should succeed")
	defer compressed_buffer_destroy(&buf)

	testing.expect(t, buf.algo == .LZ4_Block, "algo should be .LZ4_Block")
	testing.expect(t, len(buf.data) > 0, "compressed data should be non-empty")

	result, err2 := decompress(&buf, len(src_bytes))
	defer delete(result)
	testing.expect(t, err2 == .None, "decompress LZ4_Block should succeed")
	testing.expect(t, string(result) == src, "round-trip should match")
}

@(test)
compress_lz4_block_empty :: proc(t: ^testing.T) {
	buf, err := compress(nil, .LZ4_Block)
	testing.expect(t, err == .None, "compress empty LZ4_Block should succeed")
	defer compressed_buffer_destroy(&buf)
	testing.expect(t, len(buf.data) == 0, "empty compressed should be zero-length")
}

@(test)
compress_lz4_block_compressed_smaller :: proc(t: ^testing.T) {
	// Repetitive data should compress well.
	src_bytes := make([]byte, 10000)
	defer delete(src_bytes)
	for i in 0 ..< len(src_bytes) {
		src_bytes[i] = byte(i % 26 + 'a')
	}

	buf, err := compress(src_bytes, .LZ4_Block)
	testing.expect(t, err == .None, "compress should succeed")
	defer compressed_buffer_destroy(&buf)

	testing.expect(t, len(buf.data) < len(src_bytes), "compressed should be smaller than source")

	result, err2 := decompress(&buf, len(src_bytes))
	defer delete(result)
	testing.expect(t, err2 == .None, "decompress should succeed")
	testing.expect(t, len(result) == len(src_bytes), "decompressed length should match")

	match := true
	for i in 0 ..< len(src_bytes) {
		if result[i] != src_bytes[i] {
			match = false
			break
		}
	}
	testing.expect(t, match, "decompressed bytes should match source")
}

@(test)
compress_lz4_block_decompress_wrong_capacity :: proc(t: ^testing.T) {
	src := "Some data for testing capacity errors."
	src_bytes := transmute([]byte)(src)

	buf, err := compress(src_bytes, .LZ4_Block)
	testing.expect(t, err == .None, "compress should succeed")
	defer compressed_buffer_destroy(&buf)

	// Too small dst_capacity — should fail.
	result, err2 := decompress(&buf, 1)
	defer delete(result)
	testing.expect(t, err2 != .None, "decompress with too-small capacity should fail")
	testing.expect(t, len(result) == 0, "failed decompress should return empty")
}

// --- LZ4 Frame --------------------------------------------------------------

@(test)
compress_lz4_frame_round_trip :: proc(t: ^testing.T) {
	src := "LZ4 frame compression test with repeated patterns. Patterns repeated!"
	src_bytes := transmute([]byte)(src)

	buf, err := compress(src_bytes, .LZ4_Frame)
	testing.expect(t, err == .None, "compress LZ4_Frame should succeed")
	defer compressed_buffer_destroy(&buf)

	testing.expect(t, buf.algo == .LZ4_Frame, "algo should be .LZ4_Frame")
	testing.expect(t, len(buf.data) > 0, "compressed data should be non-empty")

	result, err2 := decompress(&buf, 0)
	defer delete(result)
	testing.expect(t, err2 == .None, "decompress LZ4_Frame should succeed")
	testing.expect(t, string(result) == src, "round-trip should match")
}

@(test)
compress_lz4_frame_empty :: proc(t: ^testing.T) {
	buf, err := compress(nil, .LZ4_Frame)
	testing.expect(t, err == .None, "compress empty LZ4_Frame should succeed")
	defer compressed_buffer_destroy(&buf)
	testing.expect(t, len(buf.data) == 0, "empty compressed should be zero-length")

	result, err2 := decompress(&buf, 0)
	testing.expect(t, err2 == .None, "decompress empty LZ4_Frame should succeed")
	testing.expect(t, len(result) == 0, "decompressed empty should be zero-length")
}

@(test)
compress_lz4_frame_large_data :: proc(t: ^testing.T) {
	// 64KB of repetitive data.
	src_bytes := make([]byte, 65536)
	defer delete(src_bytes)
	for i in 0 ..< len(src_bytes) {
		src_bytes[i] = byte(i % 128)
	}

	buf, err := compress(src_bytes, .LZ4_Frame)
	testing.expect(t, err == .None, "compress should succeed")
	defer compressed_buffer_destroy(&buf)

	testing.expect(t, len(buf.data) < len(src_bytes), "compressed should be smaller")

	result, err2 := decompress(&buf, 0)
	defer delete(result)
	testing.expect(t, err2 == .None, "decompress should succeed")
	testing.expect(t, len(result) == len(src_bytes), "decompressed length should match")

	match := true
	for i in 0 ..< len(src_bytes) {
		if result[i] != src_bytes[i] {
			match = false
			break
		}
	}
	testing.expect(t, match, "decompressed bytes should match source")
}
