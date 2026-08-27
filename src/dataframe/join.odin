package dataframe

// Hash-based joins (Stage 8, DESIGN.md §10): dataframe_inner_join,
// dataframe_left_join, dataframe_right_join, dataframe_full_join,
// dataframe_semi_join, dataframe_anti_join, dataframe_cross_join.
//
// Keys are column names on each side. Matching uses the same canonical key
// encoding as group_by/unique (group_keys.odin encode_row), so multi-column
// keys and every supported column type work. A NULL in any key column never
// matches — not even another NULL (SQL semantics, polars join_nulls=false);
// NULL-keyed rows only appear as unmatched rows (left/right/full) or are
// dropped (inner).
//
// Output schema (S8.5): left columns then right non-key columns. The join key
// appears once, coalesced from the left side and filled from the right for
// rows that exist only on the right. A right column whose name collides with
// an already-emitted name gets the suffix "_right", appended repeatedly until
// unique. semi/anti return the left columns only; cross returns all columns
// of both sides.
//
// Row order is deterministic per kind (DESIGN.md §10): inner is left-major,
// left preserves left order, right is right-major, full is left-major then
// unmatched right rows, semi/anti preserve left order, cross is left-major.
// The result owns its columns; both inputs are borrowed.

import "core:mem"
import "core:thread"
import "../../libs/parallel"
import "base:runtime"

// Join_Kind selects the join semantics.
Join_Kind :: enum {
	Inner,
	Left,
	Right,
	Full,
	Semi,
	Anti,
	Cross,
}

// Pair is one output row: the left and right source row indices (-1 = none).
Pair :: struct {
	l: int,
	r: int,
}

dataframe_inner_join :: proc(left, right: ^DataFrame, left_keys, right_keys: []string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	return join_run(.Inner, left, right, allocator, left_keys, right_keys)
}

dataframe_left_join :: proc(left, right: ^DataFrame, left_keys, right_keys: []string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	return join_run(.Left, left, right, allocator, left_keys, right_keys)
}

dataframe_right_join :: proc(left, right: ^DataFrame, left_keys, right_keys: []string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	return join_run(.Right, left, right, allocator, left_keys, right_keys)
}

dataframe_full_join :: proc(left, right: ^DataFrame, left_keys, right_keys: []string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	return join_run(.Full, left, right, allocator, left_keys, right_keys)
}

dataframe_semi_join :: proc(left, right: ^DataFrame, left_keys, right_keys: []string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	return join_run(.Semi, left, right, allocator, left_keys, right_keys)
}

dataframe_anti_join :: proc(left, right: ^DataFrame, left_keys, right_keys: []string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	return join_run(.Anti, left, right, allocator, left_keys, right_keys)
}

// dataframe_cross_join returns the cartesian product of left x right (no
// keys). The result owns its columns; both inputs are borrowed.
dataframe_cross_join :: proc(left, right: ^DataFrame, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	return join_run(.Cross, left, right, allocator, nil, nil)
}

// --- implementation ----------------------------------------------------------

// PARALLEL_DEFAULT_THREADS is the default pool size for parallel join probe.
PARALLEL_DEFAULT_THREADS :: 8

// JOIN_PARALLEL_THRESHOLD is the minimum probe-side row count to justify
// spawning threads.  Below this, the sequential path avoids overhead.
JOIN_PARALLEL_THRESHOLD :: 256_000

// join_run resolves and validates the key columns, computes the row pairs,
// and materializes the result DataFrame.
@(private)
join_run :: proc(kind: Join_Kind, left, right: ^DataFrame, allocator: mem.Allocator, left_keys, right_keys: []string) -> (out: DataFrame, err: Error) {
	left_cols, right_cols: []^Column
	defer delete(left_cols, allocator)
	defer delete(right_cols, allocator)
	if kind != .Cross {
		if len(left_keys) == 0 || len(left_keys) != len(right_keys) {
			return {}, .Invalid_Argument
		}
		left_cols, err = resolve_key_columns(left, allocator, left_keys)
		if err != .None {
			return {}, err
		}
		right_cols, err = resolve_key_columns(right, allocator, right_keys)
		if err != .None {
			return {}, err
		}
		for i in 0 ..< len(left_cols) {
			if left_cols[i].dtype != right_cols[i].dtype {
				return {}, .Type_Mismatch
			}
		}
	}

	left_rows := dataframe_num_rows(left)
	right_rows := dataframe_num_rows(right)

	pairs := make([dynamic]Pair, allocator)
	defer delete(pairs)

	// Build the hash on the side that is probed (right for the left-driven
	// joins so output is naturally left-major; left for right_join).
	switch kind {
	case .Right:
		hash, h_err := join_hash_build(left_cols, left_rows, allocator)
		if h_err != .None {
			return {}, h_err
		}
		defer join_hash_destroy(hash, allocator)
		buf := make([dynamic]byte, allocator)
		defer delete(buf)
		reserve(&pairs, right_rows)
		for r in 0 ..< right_rows {
			if !join_key_valid(right_cols, r) {
				append(&pairs, Pair{l = -1, r = r})
				continue
			}
			if enc_err := encode_row(right_cols, r, &buf); enc_err != .None {
				return {}, enc_err
			}
			matches, ok := hash[string(buf[:])]
			if !ok {
				append(&pairs, Pair{l = -1, r = r})
				continue
			}
			for l in matches {
				append(&pairs, Pair{l = l, r = r})
			}
		}
	case .Cross:
		reserve(&pairs, left_rows * right_rows)
		for l in 0 ..< left_rows {
			for r in 0 ..< right_rows {
				append(&pairs, Pair{l = l, r = r})
			}
		}
	case .Inner, .Left, .Full, .Semi, .Anti:
		hash, h_err := join_hash_build(right_cols, right_rows, allocator)
		if h_err != .None {
			return {}, h_err
		}
		defer join_hash_destroy(hash, allocator)

		if left_rows >= JOIN_PARALLEL_THRESHOLD {
			err = join_probe_parallel(kind, hash, left_cols, left_rows, right_rows, &pairs, allocator)
		} else {
			err = join_probe_sequential(kind, hash, left_cols, left_rows, right_rows, &pairs, allocator)
		}
		if err != .None {
			return {}, err
		}
	}

	return join_materialize(kind, left, right, left_cols, right_cols, allocator, pairs)
}

// join_probe_sequential is the single-threaded probe path.
@(private)
join_probe_sequential :: proc(kind: Join_Kind, hash: map[string][dynamic]int, left_cols: []^Column, left_rows, right_rows: int, pairs: ^[dynamic]Pair, allocator: mem.Allocator) -> Error {
	buf := make([dynamic]byte, allocator)
	defer delete(buf)
	matched_right := make([]bool, right_rows, allocator) if kind == .Full else nil
	defer delete(matched_right)
	reserve(pairs, left_rows)
	for l in 0 ..< left_rows {
		if !join_key_valid(left_cols, l) {
			if kind == .Left || kind == .Full || kind == .Anti {
				append(pairs, Pair{l = l, r = -1})
			}
			continue
		}
		if enc_err := encode_row(left_cols, l, &buf); enc_err != .None {
			return enc_err
		}
		matches, ok := hash[string(buf[:])]
		if !ok {
			if kind == .Left || kind == .Full || kind == .Anti {
				append(pairs, Pair{l = l, r = -1})
			}
			continue
		}
		for r in matches {
			if matched_right != nil {
				matched_right[r] = true
			}
			if kind == .Semi || kind == .Anti {
				continue
			}
			append(pairs, Pair{l = l, r = r})
		}
		if kind == .Semi {
			append(pairs, Pair{l = l, r = -1})
		}
	}
	if kind == .Full {
		for r in 0 ..< right_rows {
			if !matched_right[r] {
				append(pairs, Pair{l = -1, r = r})
			}
		}
	}
	return .None
}

// join_probe_ctx is the shared context passed to each parallel probe thread.
@(private)
join_probe_ctx :: struct {
	kind:       Join_Kind,
	hash:       map[string][dynamic]int,
	left_cols:  []^Column,
	left_rows:  int,
	right_rows: int,
	local_pairs: []^[dynamic]Pair,
	local_bufs:  []^[dynamic]byte,
	// For Full join: per-thread matched_right bitmaps.
	local_mr:   [][]bool,
}

// join_probe_task is the per-thread task for parallel hash join probe.
// It matches thread.Task_Proc: t.data is ^parallel.ParallelInfo(join_probe_ctx).
@(private)
join_probe_task :: proc(t: thread.Task) {
	info := (^parallel.ParallelInfo(join_probe_ctx))(t.data)
	ctx := info.sl
	tid := t.user_index
	l0 := info.a_start_off[tid]
	l1 := info.a_end_off[tid]
	lst := ctx.local_pairs[tid]
	buf := ctx.local_bufs[tid]

	mr: []bool
	if ctx.local_mr != nil {
		mr = ctx.local_mr[tid]
	}

	for l in l0 ..< l1 {
		if !join_key_valid(ctx.left_cols, l) {
			if ctx.kind == .Left || ctx.kind == .Full || ctx.kind == .Anti {
				append(lst, Pair{l = l, r = -1})
			}
			continue
		}
		if enc_err := encode_row(ctx.left_cols, l, buf); enc_err != .None {
			// Error handling not possible from thread; skip row.
			continue
		}
		matches, ok := ctx.hash[string(buf^[:])]
		if !ok {
			if ctx.kind == .Left || ctx.kind == .Full || ctx.kind == .Anti {
				append(lst, Pair{l = l, r = -1})
			}
			continue
		}
		for r in matches {
			if mr != nil {
				mr[r] = true
			}
			if ctx.kind == .Semi || ctx.kind == .Anti {
				continue
			}
			append(lst, Pair{l = l, r = r})
		}
		if ctx.kind == .Semi {
			append(lst, Pair{l = l, r = -1})
		}
	}
}

// join_probe_parallel splits the left side across threads and probes in
// parallel.  The hash map is read-only so no synchronisation is needed.
@(private)
join_probe_parallel :: proc(kind: Join_Kind, hash: map[string][dynamic]int, left_cols: []^Column, left_rows, right_rows: int, pairs: ^[dynamic]Pair, allocator: mem.Allocator) -> Error {
	nthreads := min(PARALLEL_DEFAULT_THREADS, left_rows)
	if nthreads < 2 {
		return join_probe_sequential(kind, hash, left_cols, left_rows, right_rows, pairs, allocator)
	}

	ctx := new(join_probe_ctx, allocator)
	defer free(ctx, allocator)

	ctx.kind = kind
	ctx.hash = hash
	ctx.left_cols = left_cols
	ctx.left_rows = left_rows
	ctx.right_rows = right_rows

	// Allocate per-thread pair lists and buffers.
	ctx.local_pairs = make([]^[dynamic]Pair, nthreads, allocator)
	ctx.local_bufs = make([]^[dynamic]byte, nthreads, allocator)
	defer {
		for i in 0 ..< nthreads {
			delete(ctx.local_pairs[i]^)
			delete(ctx.local_bufs[i]^)
		}
		delete(ctx.local_pairs, allocator)
		delete(ctx.local_bufs, allocator)
	}
	for i in 0 ..< nthreads {
		ctx.local_pairs[i] = new([dynamic]Pair, allocator)
		ctx.local_bufs[i] = new([dynamic]byte, allocator)
	}

	// For Full join: per-thread matched_right bitmaps.
	if kind == .Full {
		ctx.local_mr = make([][]bool, nthreads, allocator)
		for i in 0 ..< nthreads {
			ctx.local_mr[i] = make([]bool, right_rows, allocator)
		}
	}

	// Dispatch threads via parallel.do_parallel (S15.4).
	pool: thread.Pool
	thread.pool_init(&pool, allocator, nthreads)
	defer thread.pool_destroy(&pool)

	parallel.do_parallel(&pool, join_probe_task, ctx, left_rows, nthreads)

	// Merge local pair lists into the shared pairs array.
	total := 0
	for i in 0 ..< nthreads {
		total += len(ctx.local_pairs[i]^)
	}
	reserve(pairs, len(pairs^) + total)
	for i in 0 ..< nthreads {
		for p in ctx.local_pairs[i]^ {
			append(pairs, p)
		}
	}

	// For Full join: merge matched_right bitmaps and emit unmatched right rows.
	if kind == .Full {
		matched_right := make([]bool, right_rows, allocator)
		defer delete(matched_right, allocator)
		for r in 0 ..< right_rows {
			for t in 0 ..< nthreads {
				if ctx.local_mr[t][r] {
					matched_right[r] = true
					break
				}
			}
		}
		for r in 0 ..< right_rows {
			if !matched_right[r] {
				append(pairs, Pair{l = -1, r = r})
			}
		}
		// Free per-thread matched_right bitmaps.
		for i in 0 ..< nthreads {
			delete(ctx.local_mr[i], allocator)
		}
		delete(ctx.local_mr, allocator)
	}

	return .None
}

// join_hash_build returns a map from encoded key to the source rows with that
// key. Rows with a NULL in any key column are skipped (they can never match).
@(private)
join_hash_build :: proc(key_cols: []^Column, n_rows: int, allocator: mem.Allocator) -> (m: map[string][dynamic]int, err: Error) {
	// Pre-size map to reduce rehashing: assume ~1 unique key per 4 rows as a
	// heuristic.  The map grows automatically if we overshoot.
	est_keys := max(n_rows / 4, 64)
	m = make(map[string][dynamic]int, est_keys, allocator)
	buf := make([dynamic]byte, allocator)
	defer delete(buf)
	for row in 0 ..< n_rows {
		if !join_key_valid(key_cols, row) {
			continue
		}
		if enc_err := encode_row(key_cols, row, &buf); enc_err != .None {
			join_hash_destroy(m, allocator)
			return nil, enc_err
		}
		key := string(buf[:])
		if _, exists := m[key]; !exists {
			owned, o_err := clone_name(allocator, key)
			if o_err != .None {
				join_hash_destroy(m, allocator)
				return nil, o_err
			}
			m[owned] = make([dynamic]int, allocator)
		}
		append(&m[key], row)
	}
	return m, .None
}

// join_hash_destroy releases every owned key string and row list, then the
// map itself. Keys are collected first because deleting a map entry while
// ranging the map is unsafe.
@(private)
join_hash_destroy :: proc(m: map[string][dynamic]int, allocator: mem.Allocator) {
	keys := make([dynamic]string, allocator)
	defer delete(keys)
	for k in m {
		append(&keys, k)
	}
	for k in keys {
		delete_string(k, allocator)
		delete(m[k])
	}
	delete(m)
}

// join_key_valid reports whether every key column is non-NULL at row.
@(private)
join_key_valid :: proc(cols: []^Column, row: int) -> bool {
	for c in cols {
		if !row_valid(c.valid, row) {
			return false
		}
	}
	return true
}

// join_materialize builds the result columns from the row pairs.
@(private)
join_materialize :: proc(kind: Join_Kind, left, right: ^DataFrame, left_cols, right_cols: []^Column, allocator: mem.Allocator, pairs: [dynamic]Pair) -> (out: DataFrame, err: Error) {
	out = dataframe_create(allocator)
	n := len(pairs)
	if n == 0 {
		// Emit the schema with 0 rows so the column layout is still correct.
		for i in 0 ..< left.columns.count {
			if err = join_emit_left(&out, &left.col_views[i], left_cols, right_cols, allocator, nil, nil); err != .None {
				dataframe_destroy(&out)
				return {}, err
			}
		}
		if kind != .Semi && kind != .Anti {
			if err = join_emit_right(&out, right, right_cols, allocator, nil); err != .None {
				dataframe_destroy(&out)
				return {}, err
			}
		}
		return out, .None
	}

	l_idx := make([]int, n, allocator)
	r_idx := make([]int, n, allocator)
	if l_idx == nil || r_idx == nil {
		dataframe_destroy(&out)
		return {}, .Allocator_Failure
	}
	defer delete(l_idx, allocator)
	defer delete(r_idx, allocator)
	for p, i in pairs {
		l_idx[i] = p.l
		r_idx[i] = p.r
	}

	for i in 0 ..< left.columns.count {
		if err = join_emit_left(&out, &left.col_views[i], left_cols, right_cols, allocator, l_idx, r_idx); err != .None {
			dataframe_destroy(&out)
			return {}, err
		}
	}
	if kind != .Semi && kind != .Anti {
		if err = join_emit_right(&out, right, right_cols, allocator, r_idx); err != .None {
			dataframe_destroy(&out)
			return {}, err
		}
	}
	return out, .None
}

// join_emit_left appends one left column. A left key column is coalesced with
// the matching right key column (so right/full-only rows keep the key value);
// other left columns are gathered directly.
@(private)
join_emit_left :: proc(out: ^DataFrame, col: ^Column, left_cols, right_cols: []^Column, allocator: mem.Allocator, l_idx: []int, r_idx: []int) -> (err: Error) {
	if l_idx == nil {
		c, c_err := column_copy_empty(col, allocator, col.name)
		if c_err != .None {
			return c_err
		}
		return dataframe_add_column(out, &c)
	}
	for i in 0 ..< len(left_cols) {
		if left_cols[i] == col {
			c, c_err := gather_coalesced(col, right_cols[i], allocator, col.name, l_idx, r_idx)
			if c_err != .None {
				return c_err
			}
			return dataframe_add_column(out, &c)
		}
	}
	c, c_err := gather_rows_core(col, allocator, col.name, l_idx, true)
	if c_err != .None {
		return c_err
	}
	return dataframe_add_column(out, &c)
}

// join_emit_right appends the right non-key columns (right key columns are
// dropped; the key appears once on the left). Colliding names get the
// "_right" suffix.
@(private)
join_emit_right :: proc(out: ^DataFrame, right: ^DataFrame, right_cols: []^Column, allocator: mem.Allocator, r_idx: []int) -> (err: Error) {
	for i in 0 ..< right.columns.count {
		col := &right.col_views[i]
		is_key := false
		for rc in right_cols {
			if rc == col {
				is_key = true
				break
			}
		}
		if is_key {
			continue
		}
		name := col.name
		owned := false
		for dataframe_has_column(out, name) {
			if owned {
				delete_string(name, allocator)
			}
			suffixed, s_err := join_append_suffix(allocator, name)
			if s_err != .None {
				return s_err
			}
			name = suffixed
			owned = true
		}
		c: Column
		if r_idx == nil {
			c, err = column_copy_empty(col, allocator, name)
		} else {
			c, err = gather_rows_core(col, allocator, name, r_idx, true)
		}
		if owned {
			delete_string(name, allocator)
		}
		if err != .None {
			return err
		}
		if a_err := dataframe_add_column(out, &c); a_err != .None {
			column_destroy(&c)
			return a_err
		}
	}
	return .None
}

// gather_coalesced builds a column of len(l_idx) rows. Row i's element comes
// from left_src at l_idx[i] when present, else from right_src at r_idx[i]
// when present, else NULL. Used for the coalesced join key column: matched
// rows take the left value and right/full-only rows take the right value.
@(private)
gather_coalesced :: proc(left_src, right_src: ^Column, allocator: mem.Allocator, name: string, l_idx: []int, r_idx: []int) -> (out: Column, err: Error) {
	out, err = column_alloc(allocator, name, left_src.dtype, left_src.elem_size, left_src.align, len(l_idx))
	if err != .None {
		return {}, err
	}
	if len(l_idx) == 0 {
		return out, .None
	}

	has_null := false
	for i in 0 ..< len(l_idx) {
		if l_idx[i] >= 0 {
			mem.copy(ptr_offset(out.data, i * left_src.elem_size), ptr_offset(left_src.data, l_idx[i] * left_src.elem_size), left_src.elem_size)
		} else if r_idx != nil && r_idx[i] >= 0 {
			mem.copy(ptr_offset(out.data, i * right_src.elem_size), ptr_offset(right_src.data, r_idx[i] * right_src.elem_size), right_src.elem_size)
		} else {
			has_null = true
		}
	}

	if left_src.valid != nil || right_src.valid != nil || has_null {
		v := bm_make(len(l_idx), true, allocator)
		if v == nil {
			column_destroy(&out)
			return {}, .Allocator_Failure
		}
		for i in 0 ..< len(l_idx) {
			if l_idx[i] >= 0 {
				bm_set(v, i, row_valid(left_src.valid, l_idx[i]))
			} else if r_idx != nil && r_idx[i] >= 0 {
				bm_set(v, i, row_valid(right_src.valid, r_idx[i]))
			} else {
				bm_set(v, i, false)
			}
		}
		out.valid = v
	}

	if out.dtype == typeid_of(string) && (left_src.payload != nil || right_src.payload != nil) {
		if p_err := join_merge_payload(&out, left_src, right_src, allocator); p_err != .None {
			column_destroy(&out)
			return {}, p_err
		}
	}
	return out, .None
}

// join_merge_payload concatenates both sources' string payloads into one blob
// owned by out and re-points every header that referenced either payload.
// Borrowed strings (not owned by either payload) are left untouched.
@(private)
join_merge_payload :: proc(out: ^Column, left_src, right_src: ^Column, allocator: mem.Allocator) -> (err: Error) {
	lsz := left_src.payload_size
	rsz := right_src.payload_size
	total := lsz + rsz
	if total == 0 {
		return .None
	}
	blob, a_err := mem.alloc(total, 1, allocator)
	if a_err != .None || blob == nil {
		return .Allocator_Failure
	}
	if lsz != 0 {
		mem.copy(blob, left_src.payload, lsz)
	}
	if rsz != 0 {
		mem.copy(ptr_offset(blob, lsz), right_src.payload, rsz)
	}

	lb := uintptr(left_src.payload)
	rb := uintptr(right_src.payload)
	nb := uintptr(blob)
	ov := column_typed_view(out, string)
	for &s in ov {
		raw := transmute(runtime.Raw_String)s
		p := uintptr(raw.data)
		switch {
		case lsz != 0 && p >= lb && p < lb + uintptr(lsz):
			s = transmute(string)runtime.Raw_String {
				data = (^u8)(nb + (p - lb)),
				len  = raw.len,
			}
		case rsz != 0 && p >= rb && p < rb + uintptr(rsz):
			s = transmute(string)runtime.Raw_String {
				data = (^u8)(nb + uintptr(lsz) + (p - rb)),
				len  = raw.len,
			}
		}
	}

	out.payload = blob
	out.payload_size = total
	return .None
}

// column_copy_empty returns an owned 0-row column with col's name and dtype
// (used to materialize the join schema when there are no pairs).
@(private)
column_copy_empty :: proc(col: ^Column, allocator: mem.Allocator, name: string) -> (out: Column, err: Error) {
	return column_alloc(allocator, name, col.dtype, col.elem_size, col.align, 0)
}

// join_append_suffix returns name + "_right".
@(private)
join_append_suffix :: proc(allocator: mem.Allocator, name: string) -> (string, Error) {
	sfx := "_right"
	out := make([]byte, len(name) + len(sfx), allocator)
	if out == nil && len(name) + len(sfx) != 0 {
		return "", .Allocator_Failure
	}
	dst := raw_data(out)
	mem.copy(dst, raw_data(name), len(name))
	mem.copy(ptr_offset(dst, len(name)), raw_data(sfx), len(sfx))
	return string(out), .None
}
