package dataframe

// Specialized single-key sort kernel (Stage 21, S21.2): stable LSD radix
// argsort over u64 order keys, dispatched from argsort_key_columns for the
// common single-column sort (any sortable dtype except string/categorical).
//
// Why radix: the legacy path compares permutation entries through
// core:sort's Interface — every comparison pays a closure hop plus a typeid
// switch plus per-row validity checks. This kernel extracts each row's total-
// order key ONCE into a u64 (the exact ordering compare_values defines:
// signed ints flip their sign bit, floats use the canonical IEEE-754
// total-order bit key with NaN after +inf and -0.0 == +0.0, temporal types
// are their i64 storage) and then sorts fixed-width integers, which removes
// all per-comparison dispatch.
//
// Stability: LSD radix with ascending-index scatter is stable, so equal keys
// (including duplicate transformed values like ±0.0) keep source order — the
// same guarantee as the legacy index-tiebreaker path.
//
// NULLs: rows NULL in the key column never compare equal to any value row;
// they tie only with each other and fall through to source order. The kernel
// therefore stitches them (in source order) at the head (nulls_first) or tail
// before sorting the valid rows — direction-independent, as documented.
//
// Descending folds into the key by complementing it (~k), preserving both the
// total order and stability.
//
// Parallelism: each digit pass runs a private-per-thread histogram followed
// by a sequential exclusive prefix-sum combine (T x 256 entries) and a
// parallel scatter whose per-thread bucket offsets make the pass race-free
// and stable across chunk boundaries. Passes whose byte is 0 in every key
// (cumulative OR mask) are skipped entirely — small-range integer data often
// sorts in 2-3 passes instead of 8.

import "core:mem"
import "core:thread"
import "../../libs/parallel"

// RADIX_SORT_THRESHOLD is the minimum row count for the radix kernel; below
// it the legacy comparator path is cheaper than the extraction passes.
RADIX_SORT_THRESHOLD :: 4096

// RADIX_SORT_THREADS is the max thread count for the radix kernel.
RADIX_SORT_THREADS :: 8

// RADIX_SIGN_FLIP maps signed integers onto an unsigned total order:
// k = u64(i64_value) ~ RADIX_SIGN_FLIP is monotone in i64_value.
@(private)
RADIX_SIGN_FLIP :: u64(0x8000_0000_0000_0000)

// radix_key_supported reports whether dtype has a u64 order-key mapping
// (every sortable dtype except string; categoricals are excluded separately).
@(private)
radix_key_supported :: proc(dtype: typeid) -> bool {
	switch dtype {
	case typeid_of(bool),
	     typeid_of(i8), typeid_of(i16), typeid_of(i32), typeid_of(i64),
	     typeid_of(u8), typeid_of(u16), typeid_of(u32), typeid_of(u64),
	     typeid_of(int), typeid_of(uint),
	     typeid_of(f32), typeid_of(f64),
	     typeid_of(Date), typeid_of(Datetime), typeid_of(Time), typeid_of(Duration):
		return true
	case:
		return false
	}
}

// order_key_u64 maps the value at row i to its unsigned total-order key
// (identical ordering to compare_values). Row must be valid.
@(private)
order_key_u64 :: proc(c: ^Column, i: int, desc: bool) -> u64 {
	k: u64
	switch c.dtype {
	case typeid_of(bool):
		k = column_typed_view(c, bool)[i] ? 1 : 0
	case typeid_of(i8):
		k = u64(i64(column_typed_view(c, i8)[i])) ~ RADIX_SIGN_FLIP
	case typeid_of(i16):
		k = u64(i64(column_typed_view(c, i16)[i])) ~ RADIX_SIGN_FLIP
	case typeid_of(i32):
		k = u64(i64(column_typed_view(c, i32)[i])) ~ RADIX_SIGN_FLIP
	case typeid_of(i64):
		k = u64(column_typed_view(c, i64)[i]) ~ RADIX_SIGN_FLIP
	case typeid_of(int):
		k = u64(column_typed_view(c, int)[i]) ~ RADIX_SIGN_FLIP
	case typeid_of(u8):
		k = u64(column_typed_view(c, u8)[i])
	case typeid_of(u16):
		k = u64(column_typed_view(c, u16)[i])
	case typeid_of(u32):
		k = u64(column_typed_view(c, u32)[i])
	case typeid_of(u64):
		k = column_typed_view(c, u64)[i]
	case typeid_of(uint):
		k = u64(column_typed_view(c, uint)[i])
	case typeid_of(f32):
		k = u64(float32_bits(canonical_float32(column_typed_view(c, f32)[i])))
	case typeid_of(f64):
		k = float64_bits(canonical_float64(column_typed_view(c, f64)[i]))
	case typeid_of(Date):
		k = u64(i64(column_typed_view(c, Date)[i])) ~ RADIX_SIGN_FLIP
	case typeid_of(Datetime):
		k = u64(i64(column_typed_view(c, Datetime)[i])) ~ RADIX_SIGN_FLIP
	case typeid_of(Time):
		k = u64(i64(column_typed_view(c, Time)[i])) ~ RADIX_SIGN_FLIP
	case typeid_of(Duration):
		k = u64(i64(column_typed_view(c, Duration)[i])) ~ RADIX_SIGN_FLIP
	case:
		return 0 // unreachable: radix_key_supported gated the dispatch
	}
	return desc ? ~k : k
}

// Radix_Context carries the shared state for the per-pass tasks.
@(private)
Radix_Context :: struct {
	keys:     []u64,   // current-order keys (aliased into caller storage)
	idx:      []int,   // current-order permutation
	dst_keys: []u64,   // scratch targets for the pass
	dst_idx:  []int,
	hist:     [][]i32, // [tid][256] digit counts
	starts:   [][]i32, // [tid][256] global scatter offsets
	bounds:   []int,   // [tid] chunk starts (len T+1, bounds[T] == n)
	shift:    u32,     // this pass's digit position
}

// radix_hist_task zeroes the thread's histogram row and counts its chunk.
@(private)
radix_hist_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(Radix_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	h := ctx.hist[tid]
	for j in 0 ..< len(h) {
		h[j] = 0
	}
	lo := ctx.bounds[tid]
	hi := ctx.bounds[tid + 1]
	mask := u64(0xFF) << ctx.shift
	for i in lo ..< hi {
		b := u8((ctx.keys[i] & mask) >> ctx.shift)
		h[b] += 1
	}
}

// radix_scatter_task scatters its chunk into dst at the thread's exclusive
// bucket offsets (ascending index => stable).
@(private)
radix_scatter_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(Radix_Context))(t.data)
	ctx := info.sl
	tid := t.user_index
	s := ctx.starts[tid]
	lo := ctx.bounds[tid]
	hi := ctx.bounds[tid + 1]
	mask := u64(0xFF) << ctx.shift
	for i in lo ..< hi {
		b := u8((ctx.keys[i] & mask) >> ctx.shift)
		d := s[b]
		s[b] = d + 1
		ctx.dst_keys[d] = ctx.keys[i]
		ctx.dst_idx[d] = ctx.idx[i]
	}
}

// radix_argsort_single is the S21.2 fast path: a stable permutation of all
// rows ordered by the single resolved key k. Returns nil only on allocation
// failure.
@(private)
radix_argsort_single :: proc(k: sort_key_internal, allocator: mem.Allocator) -> ([]int, Error) {
	n := k.col.count
	perm := make([]int, n, allocator)
	if perm == nil && n != 0 {
		return nil, .Allocator_Failure
	}
	if n == 0 {
		return perm, .None
	}

	// Partition rows into NULLs (source order, stitched head or tail) and
	// valid rows (extracted to order keys).
	valid_n := 0
	null_n := 0
	for i in 0 ..< n {
		if row_valid(k.col.valid, i) {
			valid_n += 1
		} else {
			null_n += 1
		}
	}
	// Partition rows: NULLs (source order) stitched at head or tail, valid
	// rows occupying the remaining contiguous range in source order.
	head_nulls := k.nulls_first
	null_base := 0
	if !head_nulls {
		null_base = valid_n
	}
	valid_base := 0
	if head_nulls {
		valid_base = null_n
	}
	vi, ni := valid_base, null_base
	for i in 0 ..< n {
		if row_valid(k.col.valid, i) {
			perm[vi] = i
			vi += 1
		} else {
			perm[ni] = i
			ni += 1
		}
	}

	if valid_n <= 1 {
		return perm, .None
	}

	// Extract order keys for the valid rows (they occupy perm's valid slot
	// range contiguously).
	keys := make([]u64, valid_n, allocator)
	if keys == nil {
		delete(perm, allocator)
		return nil, .Allocator_Failure
	}
	defer delete(keys, allocator)
	for r in 0 ..< valid_n {
		keys[r] = order_key_u64(k.col, perm[valid_base + r], k.order == .Desc)
	}

	if rerr := radix_sort_u64(keys, perm[valid_base : valid_base + valid_n], allocator); rerr != .None {
		delete(perm, allocator)
		return nil, rerr
	}
	return perm, .None
}

// radix_sort_u64 stably sorts keys ascending, carrying idx alongside (both
// slices, same length, modified in place). Parallel histogram/scatter per
// active digit pass; passes whose byte vanishes in the cumulative OR are
// skipped.
@(private)
radix_sort_u64 :: proc(keys: []u64, idx: []int, allocator: mem.Allocator) -> Error {
	n := len(keys)
	if n < 2 {
		return .None
	}

	or_mask: u64
	for k in keys {
		or_mask |= k
	}
	active: [8]u32
	n_active := 0
	for p in 0 ..< 8 {
		if (or_mask >> u32(8 * p)) & 0xFF != 0 {
			active[n_active] = u32(8 * p)
			n_active += 1
		}
	}
	if n_active == 0 {
		return .None // all keys equal: already (stably) sorted
	}

	T := min(RADIX_SORT_THREADS, n)

	bounds := make([]int, T + 1, allocator)
	if bounds == nil {
		return .Allocator_Failure
	}
	defer delete(bounds, allocator)
	for t in 0 ..< T {
		bounds[t] = (n * t) / T
	}
	bounds[T] = n

	dst_keys := make([]u64, n, allocator)
	dst_idx := make([]int, n, allocator)
	hist_rows := make([][]i32, T, allocator)
	starts_rows := make([][]i32, T, allocator)
	hist_back := make([]i32, T * 256, allocator)
	starts_back := make([]i32, T * 256, allocator)
	if dst_keys == nil || dst_idx == nil || hist_rows == nil || starts_rows == nil ||
		hist_back == nil || starts_back == nil {
		return .Allocator_Failure // defers release everything allocated
	}
	defer delete(dst_keys, allocator)
	defer delete(dst_idx, allocator)
	defer delete(hist_rows, allocator)
	defer delete(starts_rows, allocator)
	defer delete(hist_back, allocator)
	defer delete(starts_back, allocator)
	for t in 0 ..< T {
		hist_rows[t] = hist_back[t * 256 : (t + 1) * 256]
		starts_rows[t] = starts_back[t * 256 : (t + 1) * 256]
	}

	ctx := Radix_Context{
		keys     = keys,
		idx      = idx,
		dst_keys = dst_keys,
		dst_idx  = dst_idx,
		hist     = hist_rows,
		starts   = starts_rows,
		bounds   = bounds,
	}

	pool: thread.Pool
	thread.pool_init(&pool, context.allocator, T)
	defer thread.pool_destroy(&pool)

	for a in 0 ..< n_active {
		ctx.shift = active[a]

		parallel.do_parallel(&pool, radix_hist_task, &ctx, n, T)

		// Sequential combine: exclusive prefix over (bucket-major) counts,
		// recording each thread's global starting offset per bucket.
		run := i32(0)
		for b in 0 ..< 256 {
			for t in 0 ..< T {
				starts_rows[t][b] = run
				run += hist_rows[t][b]
			}
		}

		parallel.do_parallel(&pool, radix_scatter_task, &ctx, n, T)

		ctx.keys, ctx.dst_keys = ctx.dst_keys, ctx.keys
		ctx.idx, ctx.dst_idx = ctx.dst_idx, ctx.idx
	}

	// Each pass flips which buffer pair holds the sorted-so-far state; after
	// an odd number of passes it lives in the scratch buffers — copy back.
	if n_active % 2 == 1 {
		copy(keys, ctx.keys)
		copy(idx, ctx.idx)
	}
	return .None
}
