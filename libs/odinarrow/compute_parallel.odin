package odinarrow

import "core:mem"
import "core:os"

// ── parallel compute kernels ──────────────────────────────────────────────────
//
// Multi-threaded variants of the compute kernels. Each one chunks the input
// with array_slice (zero-copy), runs the serial kernel per chunk on its own
// thread, and combines the partial results. Arrays are immutable, so workers
// share the underlying buffers without synchronisation.
//
// Arrays shorter than PARALLEL_MIN_LENGTH fall back to the serial kernel —
// thread spawn/join overhead (~50µs) swamps the work below that size.

PARALLEL_MIN_LENGTH :: 262_144

// 0 → one thread per logical core.
_resolve_threads :: proc(requested: int, length: int) -> int {
	n := requested if requested > 0 else os.get_processor_core_count()
	// Never spawn threads that would get less than half a minimum chunk
	max_useful := max(length / (PARALLEL_MIN_LENGTH / 2), 1)
	return clamp(n, 1, max_useful)
}

// ── sum ───────────────────────────────────────────────────────────────────────

_Sum_Task :: struct {
	slice: Array,
	sum:   f64,
	valid: int,
}

compute_sum_parallel :: proc(arr: ^Array, num_threads := 0) -> (sum: f64, valid_count: int) {
	nt := _resolve_threads(num_threads, arr.length)
	if arr.length < PARALLEL_MIN_LENGTH || nt <= 1 {
		return compute_sum(arr)
	}

	tasks := make([]_Sum_Task, nt)
	defer delete(tasks)

	// Round the chunk up to a multiple of 8 so each sub-slice starts on a
	// validity-byte boundary — that keeps the null-aware bulk aggregation path
	// (which needs offset % 8 == 0) active inside every worker.
	chunk := (((arr.length + nt - 1) / nt) + 7) &~ 7
	for i in 0..<nt {
		from := min(i * chunk, arr.length)
		to   := min(from + chunk, arr.length)
		tasks[i].slice = array_slice(arr^, from, to)
	}
	_pool_run(nt, _sum_job, raw_data(tasks))
	for i in 0..<nt {
		sum         += tasks[i].sum
		valid_count += tasks[i].valid
	}
	return
}

_sum_job :: proc(ctx: rawptr, idx: int) {
	t := &(cast([^]_Sum_Task)ctx)[idx]
	t.sum, t.valid = compute_sum(&t.slice)
}

// ── min / max ─────────────────────────────────────────────────────────────────

_MinMax_Task :: struct {
	slice:   Array,
	val:     f64,
	valid:   int,
	is_max:  bool,
}

_min_max_parallel :: proc(arr: ^Array, is_max: bool, num_threads: int) -> (val: f64, valid_count: int) {
	nt := _resolve_threads(num_threads, arr.length)
	if arr.length < PARALLEL_MIN_LENGTH || nt <= 1 {
		if is_max {
			return compute_max(arr)
		}
		return compute_min(arr)
	}

	tasks := make([]_MinMax_Task, nt)
	defer delete(tasks)

	// Round the chunk up to a multiple of 8 so each sub-slice starts on a
	// validity-byte boundary — that keeps the null-aware bulk aggregation path
	// (which needs offset % 8 == 0) active inside every worker.
	chunk := (((arr.length + nt - 1) / nt) + 7) &~ 7
	for i in 0..<nt {
		from := min(i * chunk, arr.length)
		to   := min(from + chunk, arr.length)
		tasks[i].slice  = array_slice(arr^, from, to)
		tasks[i].is_max = is_max
	}
	_pool_run(nt, _min_max_job, raw_data(tasks))
	for i in 0..<nt {
		if tasks[i].valid > 0 {
			if valid_count == 0 {
				val = tasks[i].val
			} else if is_max {
				val = max(val, tasks[i].val)
			} else {
				val = min(val, tasks[i].val)
			}
			valid_count += tasks[i].valid
		}
	}
	return
}

_min_max_job :: proc(ctx: rawptr, idx: int) {
	t := &(cast([^]_MinMax_Task)ctx)[idx]
	if t.is_max {
		t.val, t.valid = compute_max(&t.slice)
	} else {
		t.val, t.valid = compute_min(&t.slice)
	}
}

compute_min_parallel :: proc(arr: ^Array, num_threads := 0) -> (min_val: f64, valid_count: int) {
	return _min_max_parallel(arr, false, num_threads)
}

compute_max_parallel :: proc(arr: ^Array, num_threads := 0) -> (max_val: f64, valid_count: int) {
	return _min_max_parallel(arr, true, num_threads)
}

compute_mean_parallel :: proc(arr: ^Array, num_threads := 0) -> (mean: f64, valid_count: int) {
	sum: f64
	sum, valid_count = compute_sum_parallel(arr, num_threads)
	if valid_count > 0 {
		mean = sum / f64(valid_count)
	}
	return
}

// ── min_max parallel ──────────────────────────────────────────────────────────

_MM_Task :: struct {
	slice: Array,
	lo:    f64,
	hi:    f64,
	valid: int,
}

compute_min_max_parallel :: proc(arr: ^Array, num_threads := 0) -> (min_val: f64, max_val: f64, valid_count: int) {
	nt := _resolve_threads(num_threads, arr.length)
	if arr.length < PARALLEL_MIN_LENGTH || nt <= 1 {
		return compute_min_max(arr)
	}

	tasks := make([]_MM_Task, nt)
	defer delete(tasks)

	// Round the chunk up to a multiple of 8 so each sub-slice starts on a
	// validity-byte boundary — that keeps the null-aware bulk aggregation path
	// (which needs offset % 8 == 0) active inside every worker.
	chunk := (((arr.length + nt - 1) / nt) + 7) &~ 7
	for i in 0..<nt {
		from := min(i * chunk, arr.length)
		to   := min(from + chunk, arr.length)
		tasks[i].slice = array_slice(arr^, from, to)
	}
	_pool_run(nt, _mm_job, raw_data(tasks))

	for i in 0..<nt {
		if tasks[i].valid > 0 {
			if valid_count == 0 {
				min_val = tasks[i].lo
				max_val = tasks[i].hi
			} else {
				if tasks[i].lo < min_val { min_val = tasks[i].lo }
				if tasks[i].hi > max_val { max_val = tasks[i].hi }
			}
			valid_count += tasks[i].valid
		}
	}
	return
}

_mm_job :: proc(ctx: rawptr, idx: int) {
	t := &(cast([^]_MM_Task)ctx)[idx]
	t.lo, t.hi, t.valid = compute_min_max(&t.slice)
}

// ── filter ────────────────────────────────────────────────────────────────────

_Filter_Task :: struct {
	slice:      Array,
	mask_slice: Array,
	result:     Array,
	err:        mem.Allocator_Error,
	allocator:  mem.Allocator,
}

// Parallel filter: each worker filters its chunk with the serial kernel, then
// the chunk results are concatenated. Works for every type the serial
// compute_filter supports except String/Binary (their offset buffers would
// need rebasing on merge — those fall back to the serial kernel).
compute_filter_parallel :: proc(arr, mask: ^Array, num_threads := 0, allocator := context.allocator) -> (result: Array, err: mem.Allocator_Error) {
	nt := _resolve_threads(num_threads, arr.length)
	#partial switch _ in arr.type {
	case String_Type, Binary_Type:
		return compute_filter(arr, mask, allocator)
	}
	if arr.length < PARALLEL_MIN_LENGTH || nt <= 1 {
		return compute_filter(arr, mask, allocator)
	}

	tasks := make([]_Filter_Task, nt)
	defer delete(tasks)

	// Round the chunk up to a multiple of 8 so each sub-slice starts on a
	// validity-byte boundary — that keeps the null-aware bulk aggregation path
	// (which needs offset % 8 == 0) active inside every worker.
	chunk := (((arr.length + nt - 1) / nt) + 7) &~ 7
	for i in 0..<nt {
		from := min(i * chunk, arr.length)
		to   := min(from + chunk, arr.length)
		tasks[i].slice      = array_slice(arr^, from, to)
		tasks[i].mask_slice = array_slice(mask^, from, to)
		tasks[i].allocator  = allocator
	}
	_pool_run(nt, _filter_job, raw_data(tasks))
	for i in 0..<nt {
		if tasks[i].err != nil do err = tasks[i].err
	}
	defer for &t in tasks {
		array_free(&t.result)
	}
	if err != nil do return

	return _concat_fixed_width(arr.type, tasks, allocator)
}

_filter_job :: proc(ctx: rawptr, idx: int) {
	t := &(cast([^]_Filter_Task)ctx)[idx]
	t.result, t.err = compute_filter(&t.slice, &t.mask_slice, t.allocator)
}

// Concatenate the per-chunk filter results into one Array.
// Fixed-width values are memcpy'd; Bool data and validity bitmaps are merged
// bit-by-bit because chunk lengths are rarely byte-aligned.
_concat_fixed_width :: proc(dt: DataType, tasks: []_Filter_Task, allocator: mem.Allocator) -> (result: Array, err: mem.Allocator_Error) {
	total_len   := 0
	total_nulls := 0
	for &t in tasks {
		total_len   += t.result.length
		total_nulls += array_null_count(&t.result)
	}

	is_bool := type_is_bit_packed(dt)
	width   := type_byte_width(dt)

	bitmap_buf: Buffer
	if total_nulls > 0 {
		bitmap_buf = buffer_make(bitmap_byte_count(total_len), allocator) or_return
	}

	data_bytes := bitmap_byte_count(total_len) if is_bool else total_len * width
	data_buf, derr := buffer_make(data_bytes, allocator)
	if derr != nil {
		buffer_free(&bitmap_buf)
		return {}, derr
	}

	pos := 0
	for &t in tasks {
		r := &t.result
		if is_bool {
			_copy_bits(data_buf.data, pos, r.buffers[1].data, r.offset, r.length)
		} else {
			mem.copy(rawptr(data_buf.data[pos * width:]),
			         rawptr(r.buffers[1].data[r.offset * width:]),
			         r.length * width)
		}
		if total_nulls > 0 {
			if r.buffers[0].data != nil {
				_copy_bits(bitmap_buf.data, pos, r.buffers[0].data, r.offset, r.length)
			} else {
				for i in 0..<r.length {
					bitmap_set(bitmap_buf.data, pos + i)
				}
			}
		}
		pos += r.length
	}

	result = Array{
		type       = dt,
		length     = total_len,
		null_count = total_nulls,
		buffers    = {bitmap_buf, data_buf, {}},
	}
	return
}

_copy_bits :: proc "contextless" (dst: [^]u8, dst_off: int, src: [^]u8, src_off, n: int) {
	for i in 0..<n {
		if bitmap_get(src, src_off + i) {
			bitmap_set(dst, dst_off + i)
		}
	}
}
