package dataframe

// Parallel CSV parsing (S15.7): csv_find_chunks, csv_build_columns_parallel,
// csv_merge_buffers.  For files >= CSV_PARALLEL_THRESHOLD (1 MB) the hot
// record-to-column loop is split across N threads via src/parallel; each thread
// gets its own csv.Reader over a pre-computed byte range, fills independent
// CSV_Column_Buffers, and the main thread merges the results into a single
// DataFrame.
//
// Type inference (csv_infer_types) and header parsing stay sequential — only
// the bulk data parse is parallelized.  Comment mode (options.comment != 0)
// falls back to sequential because comment lines break chunk boundaries.

import "core:encoding/csv"
import "core:mem"
import "core:os"
import "core:thread"
import "../../libs/parallel"

// CSV_PARALLEL_THRESHOLD is the minimum file size (bytes) to use parallel
// parsing.  Below this the sequential path is faster (no thread overhead).
CSV_PARALLEL_THRESHOLD :: 1_048_576 // 1 MB

// CSV_PARALLEL_DEFAULT_THREADS is the default pool size for parallel CSV parse.
CSV_PARALLEL_DEFAULT_THREADS :: 8

// csv_find_chunks scans data (everything after the header line) for record
// boundaries and returns nthreads chunk descriptors.  Each chunk is a byte
// range [start, end) that starts and ends at a record boundary (\n or \r\n).
// Quoted fields containing newlines are respected via an in_quote state
// machine.
@(private)
csv_find_chunks :: proc(data: string, nthreads: int) -> []CSV_Chunk {
	if nthreads <= 0 || len(data) == 0 {
		return nil
	}

	// Single pass: find all record boundary positions.
	boundaries := make([dynamic]int, context.allocator)
	defer delete(boundaries)
	append(&boundaries, 0) // first chunk starts at 0

	in_quote := false
	i := 0
	for i < len(data) {
		b := data[i]
		if b == '"' {
			in_quote = !in_quote
			i += 1
		} else if in_quote {
			i += 1
		} else if b == '\r' {
			// \r\n or \r — advance past both if \n follows.
			next := i + 1
			if next < len(data) && data[next] == '\n' {
				next += 1
			}
			if next < len(data) {
				append(&boundaries, next)
			}
			i = next
		} else if b == '\n' {
			next := i + 1
			if next < len(data) {
				append(&boundaries, next)
			}
			i = next
		} else {
			i += 1
		}
	}

	nrecs := len(boundaries)
	if nrecs == 0 {
		return nil
	}

	// Distribute records across nthreads as evenly as possible.
	chunks := make([]CSV_Chunk, nthreads, context.allocator)
	if chunks == nil {
		return nil
	}
	each := nrecs / nthreads
	rest := nrecs % nthreads
	pos := 0
	for t in 0 ..< nthreads {
		n := each
		if t < rest {
			n += 1
		}
		if n == 0 {
			// This thread gets nothing.
			chunks[t] = CSV_Chunk{start = len(data), end = len(data)}
			continue
		}
		start := boundaries[pos]
		pos += n
		end: int
		if pos < nrecs {
			end = boundaries[pos]
		} else {
			end = len(data)
		}
		chunks[t] = CSV_Chunk{start = start, end = end}
	}
	return chunks
}

// CSV_Chunk is a byte range [start, end) within the raw CSV data, representing
// a contiguous set of records assigned to one thread.
CSV_Chunk :: struct {
	start: int,
	end:   int,
}

// CSV_Thread_Context is the shared state for all parallel parse threads.
// Each thread writes into its own thread_bufs[thread_index].
CSV_Thread_Context :: struct {
	data:         string,
	chunks:       []CSV_Chunk,
	kinds:        []CSV_Kind,
	num_cols:     int,
	options:      CSV_Options,
	allocator:    mem.Allocator,
	thread_bufs:  [][]CSV_Column_Buffer, // [nthreads][ncols]
	has_error:    bool,
	error:        Error,
	field_idx:    []int, // for _keep variant; nil means all columns
	total_fields: int,  // total fields per record (for _keep width check)
}

// csv_parse_chunk_task is the per-thread task proc.  It parses all records in
// the thread's chunk into independent CSV_Column_Buffers.  Matches
// thread.Task_Proc signature.
@(private)
csv_parse_chunk_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(CSV_Thread_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	l0 := info.a_start_off[tid]
	l1 := info.a_end_off[tid]
	chunk := ctx.chunks[tid]
	if chunk.start >= chunk.end {
		return
	}

	// Each thread gets its own csv.Reader over its byte range.
	r: csv.Reader
	csv.reader_init_with_string(&r, ctx.data[chunk.start:chunk.end], ctx.allocator)
	r.comma = csv_delimiter(ctx.options)
	r.comment = ctx.options.comment
	r.reuse_record = true
	r.reuse_record_buffer = true
	defer csv.reader_destroy(&r)

	// The first chunk starts at offset 0 which includes the header row;
	// read and discard it.  All other chunks start at a record boundary.
	if chunk.start == 0 {
		if _, r_err := csv.read(&r); r_err != nil {
			return
		}
	}

	bufs := ctx.thread_bufs[tid]
	for {
		rec, rec_err := csv.read(&r)
		if csv.is_io_error(rec_err, .EOF) {
			break
		}
		if rec_err != nil {
			if !ctx.has_error {
				ctx.has_error = true
				ctx.error = .CSV_Error
			}
			return
		}
		if len(rec) != ctx.num_cols {
			if !ctx.has_error {
				ctx.has_error = true
				ctx.error = .CSV_Error
			}
			return
		}

		append_err: Error
		if ctx.field_idx != nil {
			append_err = csv_append_record_keep(bufs, ctx.allocator, ctx.options, rec, ctx.field_idx)
		} else {
			append_err = csv_append_record(bufs, ctx.allocator, ctx.options, rec)
		}
		if append_err != .None {
			if !ctx.has_error {
				ctx.has_error = true
				ctx.error = append_err
			}
			return
		}
	}
}

// csv_build_columns_parallel is the parallel version of csv_build_columns.
// It splits data across nthreads, each thread fills independent buffers, and
// the main thread merges them.
@(private)
csv_build_columns_parallel :: proc(data: string, allocator: mem.Allocator, options: CSV_Options, names: []string, kinds: []CSV_Kind, nthreads: int) -> (out: DataFrame, err: Error) {
	chunks := csv_find_chunks(data, nthreads)
	if chunks == nil {
		return csv_build_columns(data, allocator, options, names, kinds)
	}
	defer delete(chunks)

	ncols := len(kinds)
	actual := nthreads
	for t in 0 ..< nthreads {
		if chunks[t].start == chunks[t].end {
			actual = t
			break
		}
	}
	if actual < 2 {
		return csv_build_columns(data, allocator, options, names, kinds)
	}

	// Allocate per-thread column buffers.
	thread_bufs := make([][]CSV_Column_Buffer, actual, allocator)
	if thread_bufs == nil {
		return {}, .Allocator_Failure
	}
	defer {
		for t in 0 ..< actual {
			csv_column_buffers_destroy(thread_bufs[t], allocator)
		}
		delete(thread_bufs, allocator)
	}
	for t in 0 ..< actual {
		thread_bufs[t] = make([]CSV_Column_Buffer, ncols, allocator)
		if thread_bufs[t] == nil {
			return {}, .Allocator_Failure
		}
		for i in 0 ..< ncols {
			thread_bufs[t][i] = csv_column_buffer_new(kinds[i], allocator)
		}
	}

	// Build shared context.
	ctx := new(CSV_Thread_Context, allocator)
	if ctx == nil {
		return {}, .Allocator_Failure
	}
	defer free(ctx, allocator)

	ctx.data = data
	ctx.chunks = chunks
	ctx.kinds = kinds
	ctx.num_cols = ncols
	ctx.options = options
	ctx.allocator = allocator
	ctx.thread_bufs = thread_bufs

	// Dispatch.
	pool: thread.Pool
	thread.pool_init(&pool, allocator, actual)
	defer thread.pool_destroy(&pool)

	parallel.do_parallel(&pool, csv_parse_chunk_task, ctx, actual, actual)

	if ctx.has_error {
		return {}, ctx.error
	}

	// Merge per-thread buffers into the final DataFrame.
	return csv_merge_buffers(thread_bufs, actual, ncols, kinds, names, allocator)
}

// csv_build_columns_keep_parallel is the parallel version of
// csv_build_columns_keep.  Only the columns listed in field_idx are parsed.
@(private)
csv_build_columns_keep_parallel :: proc(data: string, allocator: mem.Allocator, options: CSV_Options, names: []string, kinds: []CSV_Kind, field_idx: []int, width: int, nthreads: int) -> (out: DataFrame, err: Error) {
	chunks := csv_find_chunks(data, nthreads)
	if chunks == nil {
		return csv_build_columns_keep(data, allocator, options, names, kinds, field_idx, width)
	}
	defer delete(chunks)

	ncols := len(kinds)
	actual := nthreads
	for t in 0 ..< nthreads {
		if chunks[t].start == chunks[t].end {
			actual = t
			break
		}
	}
	if actual < 2 {
		return csv_build_columns_keep(data, allocator, options, names, kinds, field_idx, width)
	}

	thread_bufs := make([][]CSV_Column_Buffer, actual, allocator)
	if thread_bufs == nil {
		return {}, .Allocator_Failure
	}
	defer {
		for t in 0 ..< actual {
			csv_column_buffers_destroy(thread_bufs[t], allocator)
		}
		delete(thread_bufs, allocator)
	}
	for t in 0 ..< actual {
		thread_bufs[t] = make([]CSV_Column_Buffer, ncols, allocator)
		if thread_bufs[t] == nil {
			return {}, .Allocator_Failure
		}
		for i in 0 ..< ncols {
			thread_bufs[t][i] = csv_column_buffer_new(kinds[i], allocator)
		}
	}

	ctx := new(CSV_Thread_Context, allocator)
	if ctx == nil {
		return {}, .Allocator_Failure
	}
	defer free(ctx, allocator)

	ctx.data = data
	ctx.chunks = chunks
	ctx.kinds = kinds
	ctx.num_cols = width
	ctx.options = options
	ctx.allocator = allocator
	ctx.thread_bufs = thread_bufs
	ctx.field_idx = field_idx
	ctx.total_fields = width

	pool: thread.Pool
	thread.pool_init(&pool, allocator, actual)
	defer thread.pool_destroy(&pool)

	parallel.do_parallel(&pool, csv_parse_chunk_task, ctx, actual, actual)

	if ctx.has_error {
		return {}, ctx.error
	}

	return csv_merge_buffers(thread_bufs, actual, ncols, kinds, names, allocator)
}

// csv_merge_buffers merges per-thread CSV_Column_Buffers into a single
// DataFrame.  Non-string columns are merged via memcpy.  String columns
// merge the byte blobs, re-base the CSV_String_Seg offsets, and materialize
// string headers into the merged blob.
@(private)
csv_merge_buffers :: proc(thread_bufs: [][]CSV_Column_Buffer, nthreads: int, ncols: int, kinds: []CSV_Kind, names: []string, allocator: mem.Allocator) -> (out: DataFrame, err: Error) {
	// Count total rows per thread for validity merging.
	total_rows := 0
	for t in 0 ..< nthreads {
		if len(thread_bufs[t]) > 0 {
			total_rows += len(thread_bufs[t][0].valid)
		}
	}

	out = dataframe_create(allocator)

	for col_i in 0 ..< ncols {
		kind := kinds[col_i]

		// Collect per-thread row counts and check for NULLs.
		has_any_null := false
		thread_counts := make([]int, nthreads, allocator)
		defer delete(thread_counts, allocator)
		for t in 0 ..< nthreads {
			thread_counts[t] = len(thread_bufs[t][col_i].valid)
			if thread_bufs[t][col_i].has_null {
				has_any_null = true
			}
		}

		switch kind {
		case .Bool:
			merged := make([]bool, total_rows, allocator)
			if merged == nil && total_rows != 0 {
				dataframe_destroy(&out)
				return {}, .Allocator_Failure
			}
			defer delete(merged)
			offset := 0
			for t in 0 ..< nthreads {
				n := thread_counts[t]
				if n > 0 {
					copy(merged[offset:offset + n], thread_bufs[t][col_i].bools[:])
				}
				offset += n
			}
			valid_arg: []bool
			if has_any_null {
				valid_arg = csv_merge_validity(thread_bufs, col_i, nthreads, allocator)
			}
			col, c_err := column_from_with_valid(names[col_i], merged, valid_arg, allocator)
			if has_any_null {
				delete(valid_arg, allocator)
			}
			if c_err != .None {
				dataframe_destroy(&out)
				return {}, c_err
			}
			if a_err := dataframe_add_column(&out, &col); a_err != .None {
				column_destroy(&col)
				dataframe_destroy(&out)
				return {}, .Allocator_Failure
			}

		case .I64:
			merged := make([]i64, total_rows, allocator)
			if merged == nil && total_rows != 0 {
				dataframe_destroy(&out)
				return {}, .Allocator_Failure
			}
			defer delete(merged)
			offset := 0
			for t in 0 ..< nthreads {
				n := thread_counts[t]
				if n > 0 {
					copy(merged[offset:offset + n], thread_bufs[t][col_i].i64s[:])
				}
				offset += n
			}
			valid_arg: []bool
			if has_any_null {
				valid_arg = csv_merge_validity(thread_bufs, col_i, nthreads, allocator)
			}
			col, c_err := column_from_with_valid(names[col_i], merged, valid_arg, allocator)
			if has_any_null {
				delete(valid_arg, allocator)
			}
			if c_err != .None {
				dataframe_destroy(&out)
				return {}, c_err
			}
			if a_err := dataframe_add_column(&out, &col); a_err != .None {
				column_destroy(&col)
				dataframe_destroy(&out)
				return {}, .Allocator_Failure
			}

		case .F64:
			merged := make([]f64, total_rows, allocator)
			if merged == nil && total_rows != 0 {
				dataframe_destroy(&out)
				return {}, .Allocator_Failure
			}
			defer delete(merged)
			offset := 0
			for t in 0 ..< nthreads {
				n := thread_counts[t]
				if n > 0 {
					copy(merged[offset:offset + n], thread_bufs[t][col_i].f64s[:])
				}
				offset += n
			}
			valid_arg: []bool
			if has_any_null {
				valid_arg = csv_merge_validity(thread_bufs, col_i, nthreads, allocator)
			}
			col, c_err := column_from_with_valid(names[col_i], merged, valid_arg, allocator)
			if has_any_null {
				delete(valid_arg, allocator)
			}
			if c_err != .None {
				dataframe_destroy(&out)
				return {}, c_err
			}
			if a_err := dataframe_add_column(&out, &col); a_err != .None {
				column_destroy(&col)
				dataframe_destroy(&out)
				return {}, .Allocator_Failure
			}

		case .String:
			// Compute total blob size and segment count.
			total_blob := 0
			total_segs := 0
			for t in 0 ..< nthreads {
				total_blob += len(thread_bufs[t][col_i].blob)
				total_segs += len(thread_bufs[t][col_i].segs)
			}

			merged_blob := make([]byte, total_blob, allocator)
			if merged_blob == nil && total_blob != 0 {
				dataframe_destroy(&out)
				return {}, .Allocator_Failure
			}

			// Copy thread blobs into the merged blob and re-base segments.
			merged_segs := make([]CSV_String_Seg, total_segs, allocator)
			if merged_segs == nil && total_segs != 0 {
				delete(merged_blob, allocator)
				dataframe_destroy(&out)
				return {}, .Allocator_Failure
			}

			blob_offset := 0
			seg_offset := 0
			for t in 0 ..< nthreads {
				tblob := thread_bufs[t][col_i].blob[:]
				tsegs := thread_bufs[t][col_i].segs[:]
				if len(tblob) > 0 {
					copy(merged_blob[blob_offset:], tblob)
				}
				for s, i in tsegs {
					merged_segs[seg_offset + i] = CSV_String_Seg{
						start = tsegs[i].start + blob_offset,
						len   = tsegs[i].len,
					}
				}
				blob_offset += len(tblob)
				seg_offset += len(tsegs)
			}

			// Materialize string headers into the merged blob.
			strs := make([]string, total_segs, allocator)
			if strs == nil && total_segs != 0 {
				delete(merged_segs, allocator)
				delete(merged_blob, allocator)
				dataframe_destroy(&out)
				return {}, .Allocator_Failure
			}
			for s, i in merged_segs {
				strs[i] = string(merged_blob[merged_segs[i].start:merged_segs[i].start + merged_segs[i].len])
			}

			valid_arg: []bool
			if has_any_null {
				valid_arg = csv_merge_validity(thread_bufs, col_i, nthreads, allocator)
			}

			col, c_err := column_from_with_valid(names[col_i], strs, valid_arg, allocator)
			if has_any_null {
				delete(valid_arg, allocator)
			}
			delete(strs, allocator)
			if c_err != .None {
				delete(merged_segs, allocator)
				delete(merged_blob, allocator)
				dataframe_destroy(&out)
				return {}, c_err
			}
			if len(merged_blob) != 0 {
				col.payload = raw_data(merged_blob)
				col.payload_size = len(merged_blob)
				// Don't delete merged_blob — column owns it now.
			}
			// Segments are only needed during merge; free them.
			delete(merged_segs, allocator)

			if a_err := dataframe_add_column(&out, &col); a_err != .None {
				column_destroy(&col)
				dataframe_destroy(&out)
				return {}, .Allocator_Failure
			}

		}
	}

	return out, .None
}

// csv_merge_validity concatenates per-thread validity arrays into a single
// []bool of length total_rows.  The caller must delete the result.
@(private)
csv_merge_validity :: proc(thread_bufs: [][]CSV_Column_Buffer, col_i: int, nthreads: int, allocator: mem.Allocator) -> []bool {
	total := 0
	for t in 0 ..< nthreads {
		total += len(thread_bufs[t][col_i].valid)
	}
	merged := make([]bool, total, allocator)
	if merged == nil && total != 0 {
		return nil
	}
	offset := 0
	for t in 0 ..< nthreads {
		n := len(thread_bufs[t][col_i].valid)
		if n > 0 {
			copy(merged[offset:offset + n], thread_bufs[t][col_i].valid[:])
		}
		offset += n
	}
	return merged
}
