package dataframe

// Dynamic and rolling groupby (Stage 14.7, DESIGN.md §18.4).
//
// Dynamic:
//
//	gdb := dataframe_group_by_dynamic(df, "ts", {"sym"}, every, offset, closed)
//	defer dataframe_dynamic_group_by_destroy(&gdb)
//	out := dataframe_dynamic_group_by_agg(&gdb, agg_exprs) or_return
//
// Rows are grouped by the optional `by` columns and a window
// `[start, start+every)` (adjusted per `closed`) where
// `start = floor((t − offset)/every)·every + offset` (floor division, so
// pre-epoch datetimes are exact). Rows with a NULL time value belong to no
// window and are dropped; NULL `by` values form their own group (SQL
// NULL-as-key semantics, as in group_by). Output windows are ordered by
// (window start ascending, by-group first-seen), one row per window: the by
// columns, the window-start Datetime column (named `time_col`), then the
// Stage 6 aggregation results.
//
// Rolling:
//
//	rgb := dataframe_group_by_rolling(df, "ts", {"sym"}, period, offset, closed)
//	defer dataframe_rolling_group_by_destroy(&rgb)
//	out := dataframe_rolling_group_by_agg(&rgb, agg_exprs) or_return
//
// One *trailing* window per source row over the same by-group's rows:
// `[t_i − period − offset, t_i − offset]`, with the window endpoints closed per
// `closed`. Windows are computed per by-group with the group's rows sorted by
// time, each window found by binary search. The agg returns one row per source
// row (source order): the by columns, the time column, then the aggregation
// results; a NULL-time source row yields an empty window (NULL value aggs,
// count 0).
//
// Ownership: both builders borrow df — do not destroy or restructure df
// between building and destroying the group object. The result DataFrame owns
// its columns and is destroyed with dataframe_destroy.

import "core:mem"
import "core:sort"
import "expr"

// Window_Key identifies one dynamic window: its start plus the encoded
// by-group key. The `by` string is owned (via Dynamic_Group_By.by_order).
Window_Key :: struct {
	start: i64,
	by:    string,
}

// Dynamic_Group_By holds the windows of a dataframe_group_by_dynamic call. It
// borrows df and releases everything with dataframe_dynamic_group_by_destroy.
Dynamic_Group_By :: struct {
	df:        ^DataFrame,
	alloc:     mem.Allocator,
	time_name: string,             // borrowed from df; the output time column name
	by_ptrs:   []^Column,          // borrowed; empty when no by columns
	windows:   map[Window_Key][dynamic]int, // owned window start + by key -> row indices
	starts:    [dynamic]i64,       // distinct window starts, sorted ascending
	by_order:  [dynamic]string,    // owned by keys in first-seen order
}

// Rolling_Group_By holds the per-by-group time-sorted row indices of a
// dataframe_group_by_rolling call. It borrows df and releases everything with
// dataframe_rolling_group_by_destroy.
Rolling_Group_By :: struct {
	df:       ^DataFrame,
	alloc:    mem.Allocator,
	time_ptr: ^Column,     // borrowed, Datetime
	by_ptrs:  []^Column,   // borrowed
	period:   Duration,    // stored for the agg-time window computation
	offset:   Duration,
	closed:   Closed_Interval,
	groups:   map[string][dynamic]int, // owned by key -> source rows sorted by time
}

dataframe_group_by_dynamic :: proc(
	df: ^DataFrame,
	time_col: string,
	by: []string,
	every: Duration,
	offset: Duration = Duration(0),
	closed: Closed_Interval = .Left,
	allocator := context.allocator,
) -> (out: Dynamic_Group_By, err: Error) {
	if every <= Duration(0) {
		return {}, .Invalid_Argument
	}
	tcol, get_err := dataframe_get_column(df, time_col)
	if get_err != .None {
		return {}, get_err
	}
	if tcol.dtype != typeid_of(Datetime) {
		return {}, .Type_Mismatch
	}
	by_ptrs: []^Column
	by_ptrs, err = resolve_named_columns(df, allocator, by)
	if err != .None {
		return {}, err
	}

	out.df = df
	out.alloc = allocator
	out.time_name = time_col
	out.by_ptrs = by_ptrs
	out.windows = make(map[Window_Key][dynamic]int, 0, allocator)
	out.starts = make([dynamic]i64, allocator)
	out.by_order = make([dynamic]string, allocator)

	rows := dataframe_num_rows(df)
	tv := column_typed_view(tcol, Datetime)
	every_us := i64(every)
	off_us := i64(offset)
	by_first := make(map[string]int, 0, allocator)
	defer delete(by_first)
	start_seen := make(map[i64]bool, 0, allocator)
	defer delete(start_seen)
	buf := make([dynamic]byte, allocator)
	defer delete(buf)

	for row in 0 ..< rows {
		if !row_valid(tcol.valid, row) {
			continue
		}
		t := i64(tv[row])
		start := floor_div(t - off_us, every_us) * every_us + off_us
		switch closed {
		case .Right, .None:
			// A row exactly on a boundary belongs to the window *ending* at it.
			if (t - off_us) % every_us == 0 {
				start -= every_us
			}
		case .Both, .Left:
			// A boundary row belongs to the window starting at it.
		}
		if enc_err := encode_row(by_ptrs, row, &buf); enc_err != .None {
			dataframe_dynamic_group_by_destroy(&out)
			return {}, enc_err
		}
		by_key := string(buf[:])
		if _, exists := out.windows[Window_Key { start = start, by = by_key }]; !exists {
			by_owned: string
			if bi, has := by_first[by_key]; has {
				by_owned = out.by_order[bi]
			} else {
				owned, o_err := clone_name(allocator, by_key)
				if o_err != .None {
					dataframe_dynamic_group_by_destroy(&out)
					return {}, o_err
				}
				by_first[owned] = len(out.by_order)
				append(&out.by_order, owned)
				by_owned = owned
			}
			out.windows[Window_Key { start = start, by = by_owned }] = make([dynamic]int, allocator)
			if !start_seen[start] {
				start_seen[start] = true
				append(&out.starts, start)
			}
		}
		append(&out.windows[Window_Key { start = start, by = by_key }], row)
	}

	if len(out.starts) > 1 {
		view := starts_view { arr = out.starts[:] }
		it := sort.Interface {
			collection = &view,
			len        = proc(it: sort.Interface) -> int {
				v := (^starts_view)(it.collection)
				return len(v.arr)
			},
			less = proc(it: sort.Interface, i, j: int) -> bool {
				v := (^starts_view)(it.collection)
				return v.arr[i] < v.arr[j]
			},
			swap = proc(it: sort.Interface, i, j: int) {
				v := (^starts_view)(it.collection)
				v.arr[i], v.arr[j] = v.arr[j], v.arr[i]
			},
		}
		sort.sort(it)
	}
	return out, .None
}

// dataframe_dynamic_group_by_destroy releases the windows, the owned by keys,
// and the borrowed-column slice of gdb.
dataframe_dynamic_group_by_destroy :: proc(g: ^Dynamic_Group_By) {
	for _, rows in g.windows {
		delete(rows)
	}
	delete(g.windows)
	for k in g.by_order {
		delete_string(k, g.alloc)
	}
	delete(g.by_order)
	delete(g.starts)
	delete(g.by_ptrs, g.alloc)
	g^ = {}
}

dataframe_group_by_rolling :: proc(
	df: ^DataFrame,
	time_col: string,
	by: []string,
	period: Duration,
	offset: Duration = Duration(0),
	closed: Closed_Interval = .Both,
	allocator := context.allocator,
) -> (out: Rolling_Group_By, err: Error) {
	if period <= Duration(0) {
		return {}, .Invalid_Argument
	}
	tcol, get_err := dataframe_get_column(df, time_col)
	if get_err != .None {
		return {}, get_err
	}
	if tcol.dtype != typeid_of(Datetime) {
		return {}, .Type_Mismatch
	}
	by_ptrs: []^Column
	by_ptrs, err = resolve_named_columns(df, allocator, by)
	if err != .None {
		return {}, err
	}

	out.df = df
	out.alloc = allocator
	out.time_ptr = tcol
	out.by_ptrs = by_ptrs
	out.period = period
	out.offset = offset
	out.closed = closed
	out.groups = make(map[string][dynamic]int, 0, allocator)

	rows := dataframe_num_rows(df)
	buf := make([dynamic]byte, allocator)
	defer delete(buf)
	for row in 0 ..< rows {
		if !row_valid(tcol.valid, row) {
			continue
		}
		if enc_err := encode_row(by_ptrs, row, &buf); enc_err != .None {
			dataframe_rolling_group_by_destroy(&out)
			return {}, enc_err
		}
		key := string(buf[:])
		if _, exists := out.groups[key]; !exists {
			owned, o_err := clone_name(allocator, key)
			if o_err != .None {
				dataframe_rolling_group_by_destroy(&out)
				return {}, o_err
			}
			out.groups[owned] = make([dynamic]int, allocator)
		}
		append(&out.groups[key], row)
	}

	// Sort each group's rows by time ascending (the per-row windows below
	// binary-search these arrays).
	tv := column_typed_view(tcol, Datetime)
	for _, g in out.groups {
		if len(g) < 2 {
			continue
		}
		view := time_idx_view { idx = g[:], tv = tv }
		it := sort.Interface {
			collection = &view,
			len        = proc(it: sort.Interface) -> int {
				v := (^time_idx_view)(it.collection)
				return len(v.idx)
			},
			less = proc(it: sort.Interface, i, j: int) -> bool {
				v := (^time_idx_view)(it.collection)
				return v.tv[v.idx[i]] < v.tv[v.idx[j]]
			},
			swap = proc(it: sort.Interface, i, j: int) {
				v := (^time_idx_view)(it.collection)
				v.idx[i], v.idx[j] = v.idx[j], v.idx[i]
			},
		}
		sort.sort(it)
	}
	return out, .None
}

// dataframe_rolling_group_by_destroy releases the per-group row arrays and the
// owned by keys of rgb.
dataframe_rolling_group_by_destroy :: proc(g: ^Rolling_Group_By) {
	for _, rows in g.groups {
		delete(rows)
	}
	for k in g.groups {
		delete_string(k, g.alloc)
	}
	delete(g.groups)
	delete(g.by_ptrs, g.alloc)
	g^ = {}
}

// dataframe_dynamic_group_by_agg aggregates each window of gdb into one output
// row, in (window start, by-group first-seen) order: the by columns, the
// window-start Datetime column named time_col, then one result column per agg
// expression (naming and errors as in dataframe_group_by_agg).
dataframe_dynamic_group_by_agg :: proc(g: ^Dynamic_Group_By, aggs: []^expr.Expr, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	if len(aggs) == 0 {
		return {}, .Invalid_Argument
	}
	out = dataframe_create(allocator)

	win_rows := make([dynamic][]int, allocator)
	defer delete(win_rows)
	win_starts := make([dynamic]Datetime, allocator)
	defer delete(win_starts)
	for si in 0 ..< len(g.starts) {
		start := g.starts[si]
		for bi in 0 ..< len(g.by_order) {
			key := g.by_order[bi]
			if rows, ok := g.windows[Window_Key { start = start, by = key }]; ok {
				append(&win_rows, rows[:])
				append(&win_starts, Datetime(start))
			}
		}
	}
	n_win := len(win_rows)

	// By columns: gathered from each window's first row.
	if len(g.by_ptrs) != 0 {
		first_rows := make([]int, n_win, allocator)
		defer delete(first_rows)
		for wi in 0 ..< n_win {
			first_rows[wi] = win_rows[wi][0]
		}
		for bi in 0 ..< len(g.by_ptrs) {
			by_col, kerr := gather_rows(g.by_ptrs[bi], allocator, first_rows)
			if kerr != .None {
				dataframe_destroy(&out)
				return {}, kerr
			}
			if a_err := dataframe_add_column(&out, &by_col); a_err != .None {
				column_destroy(&by_col)
				dataframe_destroy(&out)
				return {}, a_err
			}
		}
	}

	// Window-start time column.
	tcol, terr := column_alloc(allocator, g.time_name, typeid_of(Datetime), size_of(Datetime), align_of(Datetime), n_win)
	if terr != .None {
		dataframe_destroy(&out)
		return {}, terr
	}
	dv := column_typed_view(&tcol, Datetime)
	for wi in 0 ..< n_win {
		dv[wi] = win_starts[wi]
	}
	if a_err := dataframe_add_column(&out, &tcol); a_err != .None {
		column_destroy(&tcol)
		dataframe_destroy(&out)
		return {}, a_err
	}

	if aerr := group_agg_run(allocator, g.df, aggs, &out, win_rows[:]); aerr != .None {
		dataframe_destroy(&out)
		return {}, aerr
	}
	return out, .None
}

// dataframe_rolling_group_by_agg aggregates each source row's trailing window
// into one output row, in source order: the by columns, the time column, then
// one result column per agg expression. A NULL-time source row produces an
// empty window (NULL value aggs, count 0).
dataframe_rolling_group_by_agg :: proc(g: ^Rolling_Group_By, aggs: []^expr.Expr, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	if len(aggs) == 0 {
		return {}, .Invalid_Argument
	}
	rows := dataframe_num_rows(g.df)
	out = dataframe_create(allocator)

	// Per-row windows over the by-group's time-sorted rows.
	window_arrs := make([dynamic][dynamic]int, rows, allocator)
	defer {
		for &w in window_arrs {
			delete(w)
		}
		delete(window_arrs)
	}
	tv := column_typed_view(g.time_ptr, Datetime)
	period_us := i64(g.period)
	off_us := i64(g.offset)
	buf := make([dynamic]byte, allocator)
	defer delete(buf)
	for row in 0 ..< rows {
		if !row_valid(g.time_ptr.valid, row) {
			continue
		}
		if enc_err := encode_row(g.by_ptrs, row, &buf); enc_err != .None {
			dataframe_destroy(&out)
			return {}, enc_err
		}
		group, ok := g.groups[string(buf[:])]
		if !ok {
			continue
		}
		t := i64(tv[row])
		hi := t - off_us
		lo := hi - period_us
		lb, ub := rolling_window_bounds(tv, group, lo, hi, g.closed)
		for k in lb ..< ub {
			append(&window_arrs[row], group[k])
		}
	}

	// Key columns gathered per source row.
	all_idx := make([]int, rows, allocator)
	defer delete(all_idx)
	for i in 0 ..< rows {
		all_idx[i] = i
	}
	for bi in 0 ..< len(g.by_ptrs) {
		by_col, kerr := gather_rows(g.by_ptrs[bi], allocator, all_idx)
		if kerr != .None {
			dataframe_destroy(&out)
			return {}, kerr
		}
		if a_err := dataframe_add_column(&out, &by_col); a_err != .None {
			column_destroy(&by_col)
			dataframe_destroy(&out)
			return {}, a_err
		}
	}
	tcol, terr := gather_rows(g.time_ptr, allocator, all_idx)
	if terr != .None {
		dataframe_destroy(&out)
		return {}, terr
	}
	if a_err := dataframe_add_column(&out, &tcol); a_err != .None {
		column_destroy(&tcol)
		dataframe_destroy(&out)
		return {}, a_err
	}

	win_slices := make([][]int, rows, allocator)
	defer delete(win_slices)
	for i in 0 ..< rows {
		win_slices[i] = window_arrs[i][:]
	}
	if aerr := group_agg_run(allocator, g.df, aggs, &out, win_slices); aerr != .None {
		dataframe_destroy(&out)
		return {}, aerr
	}
	return out, .None
}

// group_agg_run materializes the aggregation result columns shared by the
// group-by-family agg procs. It decomposes the agg expressions (Alias/Agg),
// evaluates each child once against df, and runs the Stage 6 kernel over every
// window's row indices. On error out is left partially built for the caller to
// destroy.
@(private)
group_agg_run :: proc(allocator: mem.Allocator, df: ^DataFrame, aggs: []^expr.Expr, out: ^DataFrame, windows: [][]int) -> (err: Error) {
	if len(aggs) == 0 {
		return .Invalid_Argument
	}
	n := len(windows)

	specs := make([]agg_spec, len(aggs), allocator)
	if specs == nil && len(aggs) != 0 {
		return .Allocator_Failure
	}
	defer delete(specs, allocator)
	for e, i in aggs {
		s, ok := decompose_agg(e)
		if !ok {
			return .Unsupported_Operation
		}
		if s.name == "" {
			return .Invalid_Argument
		}
		specs[i] = s
	}

	children := make([]Column, len(aggs), allocator)
	if children == nil && len(aggs) != 0 {
		return .Allocator_Failure
	}
	defer {
		for &c in children {
			column_destroy(&c)
		}
		delete(children, allocator)
	}
	oa: OpArena
	op_arena_init(&oa, allocator)
	defer op_arena_destroy(&oa)
	for s, i in specs {
		c, cerr := expr_eval(allocator, df, s.child, &oa)
		if cerr != .None {
			return cerr
		}
		children[i] = c
	}

	// An empty window arrives as a nil slice, which run_group_agg treats as
	// "all rows"; substitute a shared non-nil empty slice so empty windows
	// aggregate to count 0 / NULL value results.
	empty_win := make([dynamic]int, 0, 1)
	defer delete(empty_win)
	non_nil_empty := empty_win[:]

	for s, i in specs {
		child := &children[i]
		if v_err := validate_agg(s.kind, child.dtype); v_err != .None {
			return v_err
		}
		res_dtype, ok := agg_result_dtype(s.kind, child.dtype)
		if !ok {
			return .Invalid_Argument
		}
		col, aerr := column_alloc(allocator, s.name, res_dtype, size_of_ty(res_dtype), align_of_ty(res_dtype), n)
		if aerr != .None {
			return aerr
		}
		for wi in 0 ..< n {
			w := windows[wi]
			if w == nil {
				w = non_nil_empty
			}
			if rerr := run_group_agg(allocator, child, s.kind, s.q, &col, w, wi); rerr != .None {
				column_destroy(&col)
				return rerr
			}
		}
		if a_err := dataframe_add_column(out, &col); a_err != .None {
			column_destroy(&col)
			return a_err
		}
	}
	return .None
}

// rolling_window_bounds returns the [lb, ub) slice of `group` (rows sorted by
// time) whose times fall in [lo, hi], endpoints closed per `closed`.
@(private)
rolling_window_bounds :: proc(tv: []Datetime, group: [dynamic]int, lo, hi: i64, closed: Closed_Interval) -> (lb, ub: int) {
	switch closed {
	case .Both:
		return rolling_lower_bound(tv, group, lo), rolling_upper_bound(tv, group, hi)
	case .Left:
		return rolling_lower_bound(tv, group, lo), rolling_lower_bound(tv, group, hi)
	case .Right:
		return rolling_upper_bound(tv, group, lo), rolling_upper_bound(tv, group, hi)
	case .None:
		return rolling_upper_bound(tv, group, lo), rolling_lower_bound(tv, group, hi)
	}
	return 0, 0
}

// rolling_lower_bound returns the first index k with tv[group[k]] >= target.
@(private)
rolling_lower_bound :: proc(tv: []Datetime, group: [dynamic]int, target: i64) -> int {
	lo, hi := 0, len(group)
	for lo < hi {
		mid := (lo + hi) >> 1
		if i64(tv[group[mid]]) < target {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// rolling_upper_bound returns the first index k with tv[group[k]] > target.
@(private)
rolling_upper_bound :: proc(tv: []Datetime, group: [dynamic]int, target: i64) -> int {
	lo, hi := 0, len(group)
	for lo < hi {
		mid := (lo + hi) >> 1
		if i64(tv[group[mid]]) <= target {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// starts_view and time_idx_view carry the sort.Interface context.
@(private)
starts_view :: struct {
	arr: []i64,
}

@(private)
time_idx_view :: struct {
	idx: []int,
	tv:  []Datetime,
}
