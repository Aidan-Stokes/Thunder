package dataframe

import "core:mem"
import "core:c"
import lz4 "../../libs/odin-lz4/lz4"

// Compression_Algorithm enumerates supported codecs. Each maps to a
// specific wire format (LZ4 block for Parquet, LZ4 frame for Arrow IPC).
Compression_Algorithm :: enum {
	None,
	LZ4_Block,
	LZ4_Frame,
}

// Compressed_Buffer is an owned, immutable byte buffer produced by compress().
// The caller must destroy it with compressed_buffer_destroy when done.
// algo records which codec was used, so decompress() can dispatch.
Compressed_Buffer :: struct {
	data:  []byte,
	algo:  Compression_Algorithm,
	alloc: mem.Allocator,
}

// compressed_buffer_destroy releases the owned buffer.
compressed_buffer_destroy :: proc(buf: ^Compressed_Buffer) {
	if buf.data != nil {
		delete(buf.data, buf.alloc)
	}
	buf^ = {}
}

// compress compresses src using the given algorithm.
// Returns an owned Compressed_Buffer on success.
compress :: proc(
	src: []byte,
	algo: Compression_Algorithm,
	allocator := context.allocator,
) -> (Compressed_Buffer, Error) {
	switch algo {
	case .None:
		out := make([]byte, len(src), allocator)
		if out == nil && len(src) != 0 {
			return {}, .Allocator_Failure
		}
		copy(out, src)
		return Compressed_Buffer{data = out, algo = .None, alloc = allocator}, .None

	case .LZ4_Block:
		return compress_lz4_block(src, allocator)

	case .LZ4_Frame:
		return compress_lz4_frame(src, allocator)
	}

	return {}, .Invalid_Argument
}

// decompress decompresses buf into an owned byte slice.
// For .None buffers the result is a verbatim copy.
// For .LZ4_Block the caller must supply the expected uncompressed size
// via dst_capacity.
// For .LZ4_Frame dst_capacity is ignored; the frame header provides the
// information needed.
decompress :: proc(
	buf: ^Compressed_Buffer,
	dst_capacity: int,
	allocator := context.allocator,
) -> ([]byte, Error) {
	switch buf.algo {
	case .None:
		out := make([]byte, len(buf.data), allocator)
		if out == nil && len(buf.data) != 0 {
			return {}, .Allocator_Failure
		}
		copy(out, buf.data)
		return out, .None

	case .LZ4_Block:
		return decompress_lz4_block(buf.data, dst_capacity, allocator)

	case .LZ4_Frame:
		return decompress_lz4_frame(buf.data, allocator)
	}

	return {}, .Invalid_Argument
}

// --- LZ4 Block (raw) -------------------------------------------------------

compress_lz4_block :: proc(
	src: []byte,
	allocator := context.allocator,
) -> (Compressed_Buffer, Error) {
	if len(src) == 0 {
		return Compressed_Buffer{data = nil, algo = .LZ4_Block, alloc = allocator}, .None
	}

	bound := lz4.compressBound(c.int(len(src)))
	if bound <= 0 {
		return {}, .Invalid_Argument
	}

	dst := make([]byte, int(bound), allocator)
	if dst == nil {
		return {}, .Allocator_Failure
	}

	written := lz4.compress_default(
		src = &src[0],
		dst = &dst[0],
		srcSize = c.int(len(src)),
		dstCapacity = c.int(len(dst)),
	)

	if written <= 0 {
		delete(dst, allocator)
		return {}, .Invalid_Argument
	}

	return Compressed_Buffer{
		data  = dst[:written],
		algo  = .LZ4_Block,
		alloc = allocator,
	}, .None
}

decompress_lz4_block :: proc(
	compressed: []byte,
	dst_capacity: int,
	allocator := context.allocator,
) -> ([]byte, Error) {
	if len(compressed) == 0 && dst_capacity == 0 {
		return nil, .None
	}

	dst := make([]byte, dst_capacity, allocator)
	if dst == nil && dst_capacity != 0 {
		return {}, .Allocator_Failure
	}

	written := lz4.decompress_safe(
		src = &compressed[0],
		dst = &dst[0],
		compressedSize = c.int(len(compressed)),
		dstCapacity = c.int(dst_capacity),
	)

	if written < 0 {
		delete(dst, allocator)
		return {}, .Invalid_Argument
	}

	return dst[:written], .None
}

// --- LZ4 Frame -------------------------------------------------------------

compress_lz4_frame :: proc(
	src: []byte,
	allocator := context.allocator,
) -> (Compressed_Buffer, Error) {
	if len(src) == 0 {
		return Compressed_Buffer{data = nil, algo = .LZ4_Frame, alloc = allocator}, .None
	}

	bound := lz4.LZ4F_compressFrameBound(c.size_t(len(src)), nil)
	if lz4.LZ4F_isError(bound) != 0 {
		return {}, .Invalid_Argument
	}

	dst := make([]byte, int(bound), allocator)
	if dst == nil {
		return {}, .Allocator_Failure
	}

	written := lz4.LZ4F_compressFrame(
		rawptr(&dst[0]), c.size_t(len(dst)),
		rawptr(&src[0]), c.size_t(len(src)),
		nil,
	)

	if lz4.LZ4F_isError(written) != 0 {
		delete(dst, allocator)
		return {}, .Invalid_Argument
	}

	return Compressed_Buffer{
		data  = dst[:int(written)],
		algo  = .LZ4_Frame,
		alloc = allocator,
	}, .None
}

decompress_lz4_frame :: proc(
	compressed: []byte,
	allocator := context.allocator,
) -> ([]byte, Error) {
	if len(compressed) == 0 {
		return nil, .None
	}

	dctx: lz4.LZ4F_DCtx
	err_code := lz4.LZ4F_createDecompressionContext(&dctx, lz4.LZ4F_VERSION)
	if lz4.LZ4F_isError(err_code) != 0 {
		return {}, .Invalid_Argument
	}
	defer lz4.LZ4F_freeDecompressionContext(dctx)

	// Initial guess: 4x compressed size, minimum 256 bytes.
	cap := len(compressed) * 4
	if cap < 256 {
		cap = 256
	}

	dst := make([]byte, cap, allocator)
	if dst == nil {
		return {}, .Allocator_Failure
	}

	src_remaining := len(compressed)
	src_pos := 0
	dst_pos := 0

	for {
		// Available space in dst buffer.
		avail_dst := cap - dst_pos
		if avail_dst == 0 {
			// Grow the buffer.
			new_cap := cap * 2
			new_dst := make([]byte, new_cap, allocator)
			if new_dst == nil {
				delete(dst, allocator)
				return {}, .Allocator_Failure
			}
			copy(new_dst[:cap], dst)
			delete(dst, allocator)
			dst = new_dst
			cap = new_cap
			avail_dst = cap - dst_pos
		}

		remaining_in_chunk := c.size_t(src_remaining)
		dst_written := c.size_t(avail_dst)

		ret := lz4.LZ4F_decompress(
			dctx,
			rawptr(&dst[dst_pos]), &dst_written,
			rawptr(&compressed[src_pos]), &remaining_in_chunk,
			nil,
		)

		if lz4.LZ4F_isError(ret) != 0 {
			delete(dst, allocator)
			return {}, .Invalid_Argument
		}

		src_pos += int(remaining_in_chunk)
		src_remaining -= int(remaining_in_chunk)
		dst_pos += int(dst_written)

		if ret == 0 {
			// Frame fully decoded.
			return dst[:dst_pos], .None
		}
	}
}
