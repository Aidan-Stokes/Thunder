package dataframe

// Window function evaluation (Stage 14.4/14.5, DESIGN.md §18.2).
//
// A Window node computes a per-partition, order-preserving result column.
// `over` partitions the rows by evaluated key expressions (empty = one
// partition of all rows); the rows of each partition keep source order, so
// every kernel is an order-dependent loop over the partition's row indices.
//
// Semantics (each partition independent):
//   - Row_Number: 1-based source position (i64, always valid).
//   - Rank: f64; non-NULL rows ranked with the given tie method; NULL rows
//     yield NULL.
//   - Cum_Sum/Cum_Min/Cum_Max: running aggregate in source order; output dtype
//     = child dtype; NULL input rows yield NULL output and the running value
//     continues.
//   - Shift: out[i] = in[i-n] within the partition; out-of-partition and NULL
//     source positions are NULL.
//   - Rolling: agg over the trailing window [i-n+1, i] via the Stage 6/7
//     kernels (run_group_agg). Count is always valid; value aggs are NULL when
//     the window has no valid rows.
//   - Cumulative_Eval: like Rolling with the growing window [0, i].
//   - Ewma: y_i = alpha·x_i + (1-alpha)·y_{i-1}; the first valid row seeds
//     y = x; NULL rows yield NULL and the recursion continues. Result is f64.
//
// String-typed outputs (Shift / Rolling Min/Max/First/Last/Mode over string
// children) copy the child's owned contents after the kernels so the result
// survives the child column's destruction.

import "core:mem"
import "core:sort"
import "expr"

// window_result_dtype maps a window function and its child dtype to the result
// column type.
@(private)
window_result_dtype :: proc(n: expr.Window, child_dtype: typeid) -> (typeid, bool) {
	switch n.func {
	case .Row_Number:
		return typeid_of(i64), true
	case .Rank, .Ewma:
		return typeid_of(f64), true
	case .Cum_Sum, .Cum_Min, .Cum_Max, .Shift:
		return child_dtype, true
	case .Rolling, .Cumulative_Eval:
		return agg_result_dtype(n.agg, child_dtype)
	case:
		return {}, false
	}
}

// window_eval implements the expr.Window node.
@(private)
window_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Window) -> (Column, Error) {
	rows := dataframe_num_rows(df)

	child: Column
	if n.expr != nil {
		c, cerr := expr_eval(allocator, df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		child = c
	}
	defer column_destroy(&child)

	if n.func != .Row_Number && !column_supported_window_type(child.dtype) {
		return {}, .Unsupported_Operation
	}
	#partial switch n.func {
	case .Rank:
		if !sortable_dtype(child.dtype) {
			return {}, .Unsupported_Operation
		}
	case .Cum_Sum, .Cum_Min, .Cum_Max, .Ewma:
		if !is_numeric_type(child.dtype) {
			return {}, .Unsupported_Operation
		}
	case .Rolling, .Cumulative_Eval:
		if v_err := validate_agg(n.agg, child.dtype); v_err != .None {
			return {}, v_err
		}
		if n.agg == .Quantile {
			return {}, .Invalid_Argument
		}
		if n.func == .Rolling && n.n < 1 {
			return {}, .Invalid_Argument
		}
	case:
	}

	res_dtype, ok := window_result_dtype(n, child.dtype)
	if !ok {
		return {}, .Unsupported_Operation
	}
	out, aerr := column_alloc(allocator, child.name, res_dtype, size_of_ty(res_dtype), align_of_ty(res_dtype), rows)
	if aerr != .None {
		return {}, aerr
	}
	out.valid = bm_make(rows, true, allocator)
	if out.valid == nil && rows != 0 {
		column_destroy(&out)
		return {}, .Allocator_Failure
	}

	parts, key_cols, perr := window_partitions(allocator, df, n.over, rows)
	if perr != .None {
		column_destroy(&out)
		return {}, perr
	}
	defer {
		for p in parts {
			delete(p, allocator)
		}
		delete(parts)
		for &c in key_cols {
			column_destroy(&c)
		}
		delete(key_cols, allocator)
	}

	switch n.func {
	case .Row_Number:
		ov := column_typed_view(&out, i64)
		for part in parts {
			for r, j in part {
				ov[r] = i64(j + 1)
			}
		}
	case .Rank:
		for part in parts {
			if kerr := window_rank_kernel(allocator, &child, part, &out, n.method); kerr != .None {
				column_destroy(&out)
				return {}, kerr
			}
		}
	case .Cum_Sum, .Cum_Min, .Cum_Max:
		for part in parts {
			window_cum_run(child.dtype, &child, part, &out, n.func)
		}
	case .Shift:
		for part in parts {
			window_shift_kernel(&child, part, &out, n.n)
		}
	case .Rolling, .Cumulative_Eval:
		for part in parts {
			if a_err := window_agg_run(allocator, &child, part, &out, n); a_err != .None {
				column_destroy(&out)
				return {}, a_err
			}
		}
	case .Ewma:
		for part in parts {
			window_ewma_run(child.dtype, &child, part, &out, n.alpha)
		}
	case:
		column_destroy(&out)
		return {}, .Invalid_Argument
	}

	// String outputs own a copy of the child's contents so they survive it.
	if out.dtype == typeid_of(string) && child.payload != nil {
		if ferr := finalize_string_contents(&out, uintptr(child.payload), child.payload_size, allocator); ferr != .None {
			column_destroy(&out)
			return {}, ferr
		}
	}
	return out, .None
}

// column_supported_window_type reports whether a column type can be the child
// of a value-copying window function (everything but the numeric-only kernels).
@(private)
column_supported_window_type :: proc(dtype: typeid) -> bool {
	switch dtype {
	case typeid_of(bool), typeid_of(string),
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

// window_partitions evaluates the over-key expressions and returns the row
// indices of each partition in source order (first-seen partition order).
// key_cols holds the evaluated key columns (owned by the caller).
@(private)
window_partitions :: proc(allocator: mem.Allocator, df: ^DataFrame, over: []^expr.Expr, rows: int) -> (parts: [dynamic][]int, key_cols: []Column, err: Error) {
	if len(over) == 0 {
		all := make([]int, rows, allocator)
		if all == nil && rows != 0 {
			return {}, nil, .Allocator_Failure
		}
		for i in 0 ..< rows {
			all[i] = i
		}
		append(&parts, all)
		return parts, nil, .None
	}

	key_cols = make([]Column, len(over), allocator)
	if key_cols == nil && len(over) != 0 {
		return {}, nil, .Allocator_Failure
	}
	for e, i in over {
		c, cerr := expr_eval(allocator, df, e)
		if cerr != .None {
			window_destroy_partial(&parts, &key_cols, allocator)
			return {}, nil, cerr
		}
		key_cols[i] = c
	}
	key_ptrs := make([]^Column, len(over), allocator)
	if key_ptrs == nil && len(over) != 0 {
		window_destroy_partial(&parts, &key_cols, allocator)
		return {}, nil, .Allocator_Failure
	}
	defer delete(key_ptrs, allocator)
	for i in 0 ..< len(over) {
		key_ptrs[i] = &key_cols[i]
	}

	groups := make(map[string][dynamic]int, 0, allocator)
	order := make([dynamic]string, allocator)
	buf := make([dynamic]byte, allocator)
	defer delete(buf)
	for row in 0 ..< rows {
		if enc_err := encode_row(key_ptrs, row, &buf); enc_err != .None {
			window_destroy_groups(&groups, &order, allocator)
			window_destroy_partial(&parts, &key_cols, allocator)
			return {}, nil, enc_err
		}
		key := string(buf[:])
		if _, exists := groups[key]; !exists {
			owned, o_err := clone_name(allocator, key)
			if o_err != .None {
				window_destroy_groups(&groups, &order, allocator)
				window_destroy_partial(&parts, &key_cols, allocator)
				return {}, nil, o_err
			}
			groups[owned] = make([dynamic]int, allocator)
			append(&order, owned)
		}
		append(&groups[key], row)
	}

	for gk in order {
		part := make([]int, len(groups[gk]), allocator)
		if part == nil && len(groups[gk]) != 0 {
			window_destroy_groups(&groups, &order, allocator)
			window_destroy_partial(&parts, &key_cols, allocator)
			return {}, nil, .Allocator_Failure
		}
		copy(part, groups[gk][:])
		append(&parts, part)
	}
	window_destroy_groups(&groups, &order, allocator)
	return parts, key_cols, .None
}

// window_destroy_partial frees key_cols and any partition slices built so far
// (error path of window_partitions).
@(private)
window_destroy_partial :: proc(parts: ^[dynamic][]int, key_cols: ^[]Column, allocator: mem.Allocator) {
	for p in parts^ {
		delete(p, allocator)
	}
	delete(parts^)
	parts^ = {}
	for &c in key_cols^ {
		column_destroy(&c)
	}
	delete(key_cols^, allocator)
	key_cols^ = nil
}

// window_destroy_groups frees the group map and its order strings. The map and
// dynamic arrays were allocated with `allocator`; this version of the runtime
// deletes them via context.allocator, so callers are expected to use the
// default context (see window_partitions' allocator convention).
@(private)
window_destroy_groups :: proc(groups: ^map[string][dynamic]int, order: ^[dynamic]string, allocator: mem.Allocator) {
	for _, g in groups^ {
		delete(g)
	}
	delete(groups^)
	for k in order^ {
		delete_string(k, allocator)
	}
	delete(order^)
}

// --- rank --------------------------------------------------------------------

// rank_view carries the sortable state for the rank kernel's core:sort.Interface.
@(private)
rank_view :: struct {
	idx: []int,
	col: ^Column,
}

// window_rank_kernel ranks the partition's valid rows by value (ascending,
// total order via compare_values) with the given tie method, writing f64 ranks
// into out at the rows' original indices. NULL child rows yield NULL output.
@(private)
window_rank_kernel :: proc(allocator: mem.Allocator, child: ^Column, part: []int, out: ^Column, method: expr.Rank_Method) -> Error {
	for r in part {
		if !column_is_valid(child, r) {
			bm_set(out.valid, r, false)
		}
	}
	valid_rows := make([dynamic]int, 0, len(part), allocator)
	defer delete(valid_rows)
	for r in part {
		if column_is_valid(child, r) {
			append(&valid_rows, r)
		}
	}
	n := len(valid_rows)
	if n == 0 {
		return .None
	}

	view := rank_view{idx = valid_rows[:], col = child}
	it := sort.Interface {
		collection = &view,
		len        = proc(it: sort.Interface) -> int {
			v := (^rank_view)(it.collection)
			return len(v.idx)
		},
		less = proc(it: sort.Interface, i, j: int) -> bool {
			v := (^rank_view)(it.collection)
			if c := compare_values(v.col, v.idx[i], v.idx[j]); c != 0 {
				return c < 0
			}
			return v.idx[i] < v.idx[j]
		},
		swap = proc(it: sort.Interface, i, j: int) {
			v := (^rank_view)(it.collection)
			v.idx[i], v.idx[j] = v.idx[j], v.idx[i]
		},
	}
	sort.sort(it)

	ov := column_typed_view(out, f64)
	dense := 0
	i := 0
	for i < n {
		j := i
		for j + 1 < n && compare_values(child, valid_rows[i], valid_rows[j + 1]) == 0 {
			j += 1
		}
		rank: f64
		switch method {
		case .Min:
			rank = f64(i + 1)
		case .Max:
			rank = f64(j + 1)
		case .Average:
			rank = (f64(i + 1) + f64(j + 1)) / 2
		case .Dense:
			dense += 1
			rank = f64(dense)
		case:
		}
		for k in i ..= j {
			ov[valid_rows[k]] = rank
		}
		i = j + 1
	}
	return .None
}

// --- cum_sum / cum_min / cum_max --------------------------------------------

@(private)
window_cum_run :: proc(t: typeid, child: ^Column, part: []int, out: ^Column, kind: expr.Window_Func) {
	switch t {
	case typeid_of(i8):   window_cum_typed(child, part, out, i8, kind)
	case typeid_of(i16):  window_cum_typed(child, part, out, i16, kind)
	case typeid_of(i32):  window_cum_typed(child, part, out, i32, kind)
	case typeid_of(i64):  window_cum_typed(child, part, out, i64, kind)
	case typeid_of(u8):   window_cum_typed(child, part, out, u8, kind)
	case typeid_of(u16):  window_cum_typed(child, part, out, u16, kind)
	case typeid_of(u32):  window_cum_typed(child, part, out, u32, kind)
	case typeid_of(u64):  window_cum_typed(child, part, out, u64, kind)
	case typeid_of(int):  window_cum_typed(child, part, out, int, kind)
	case typeid_of(uint): window_cum_typed(child, part, out, uint, kind)
	case typeid_of(f32):  window_cum_typed(child, part, out, f32, kind)
	case typeid_of(f64):  window_cum_typed(child, part, out, f64, kind)
	}
}

// window_cum_typed runs the running aggregate in source order. NULL child rows
// yield NULL output (validity stays false) and the running value continues.
@(private)
window_cum_typed :: proc(child: ^Column, part: []int, out: ^Column, $T: typeid, kind: expr.Window_Func) {
	iv := column_typed_view(child, T)
	ov := column_typed_view(out, T)
	started := false
	acc: T
	for r in part {
		if !column_is_valid(child, r) {
			bm_set(out.valid, r, false)
			continue
		}
		if !started {
			acc = iv[r]
			started = true
		} else {
			#partial switch kind {
			case .Cum_Sum: acc += iv[r]
			case .Cum_Min: acc = min(acc, iv[r])
			case .Cum_Max: acc = max(acc, iv[r])
			}
		}
		ov[r] = acc
	}
}

// --- shift -------------------------------------------------------------------

// window_shift_kernel computes out[r] = in[part[j-n]] (bytewise copy, works
// for any element type). Out-of-partition positions and NULL source rows are
// NULL.
@(private)
window_shift_kernel :: proc(child: ^Column, part: []int, out: ^Column, n: i64) {
	for r, j in part {
		src_j := j - int(n)
		if src_j < 0 || src_j >= len(part) {
			bm_set(out.valid, r, false)
			continue
		}
		src := part[src_j]
		if !column_is_valid(child, src) {
			bm_set(out.valid, r, false)
			continue
		}
		mem.copy(
			ptr_offset(out.data, r * child.elem_size),
			ptr_offset(child.data, src * child.elem_size),
			child.elem_size,
		)
	}
}

// --- rolling / cumulative_eval ----------------------------------------------

// window_agg_run aggregates over the trailing window [i-n+1, i] (Rolling) or
// the growing window [0, i] (Cumulative_Eval) using the shared Stage 6/7
// kernels.
@(private)
window_agg_run :: proc(allocator: mem.Allocator, child: ^Column, part: []int, out: ^Column, n: expr.Window) -> Error {
	win := make([dynamic]int, 0, int(n.n), allocator)
	defer delete(win)
	for r, j in part {
		clear(&win)
		start := 0
		if n.func == .Rolling {
			start = max(0, j - int(n.n) + 1)
		}
		for k in start ..= j {
			append(&win, part[k])
		}
		if agg_err := run_group_agg(allocator, child, n.agg, 0, out, win[:], r); agg_err != .None {
			return agg_err
		}
	}
	return .None
}

// --- ewma --------------------------------------------------------------------

@(private)
window_ewma_run :: proc(t: typeid, child: ^Column, part: []int, out: ^Column, alpha: f64) {
	switch t {
	case typeid_of(i8):   window_ewma_typed(child, part, out, i8, alpha)
	case typeid_of(i16):  window_ewma_typed(child, part, out, i16, alpha)
	case typeid_of(i32):  window_ewma_typed(child, part, out, i32, alpha)
	case typeid_of(i64):  window_ewma_typed(child, part, out, i64, alpha)
	case typeid_of(u8):   window_ewma_typed(child, part, out, u8, alpha)
	case typeid_of(u16):  window_ewma_typed(child, part, out, u16, alpha)
	case typeid_of(u32):  window_ewma_typed(child, part, out, u32, alpha)
	case typeid_of(u64):  window_ewma_typed(child, part, out, u64, alpha)
	case typeid_of(int):  window_ewma_typed(child, part, out, int, alpha)
	case typeid_of(uint): window_ewma_typed(child, part, out, uint, alpha)
	case typeid_of(f32):  window_ewma_typed(child, part, out, f32, alpha)
	case typeid_of(f64):  window_ewma_typed(child, part, out, f64, alpha)
	}
}

// window_ewma_typed computes the exponential weighted moving average in source
// order. NULL child rows yield NULL output and the recursion continues from the
// last computed value.
@(private)
window_ewma_typed :: proc(child: ^Column, part: []int, out: ^Column, $T: typeid, alpha: f64) {
	iv := column_typed_view(child, T)
	ov := column_typed_view(out, f64)
	started := false
	y: f64
	for r in part {
		if !column_is_valid(child, r) {
			bm_set(out.valid, r, false)
			continue
		}
		if !started {
			y = f64(iv[r])
			started = true
		} else {
			y = alpha * f64(iv[r]) + (1 - alpha) * y
		}
		ov[r] = y
	}
}
