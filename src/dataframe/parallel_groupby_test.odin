package dataframe

// Stage 21 tests: S21.1 parallel partitioned hash grouping
// (parallel_groupby.odin).
//
// The reference is group_rows_sequential — the original single-threaded loop,
// whose semantics (first-appearance group order, source-order rows within a
// group) are pinned by the Stage 7 suite. The parallel path must reproduce it
// exactly; private kernels are called directly so the paths run regardless of
// PARALLEL_GROUPBY_THRESHOLD.

import "core:testing"
import "core:mem"
import "core:fmt"
import "expr"

// groupby_shell builds an empty Group_By over deep copies of src_cols, so two
// shells can be destroyed independently of each other and of any df.
@(private)
groupby_shell :: proc(
	t: ^testing.T,
	src_cols: []^Column,
	allocator: mem.Allocator,
) -> Group_By {
	gb: Group_By
	gb.alloc = allocator
	all := make([]int, src_cols[0].count, allocator)
	for i in 0 ..< len(all) {
		all[i] = i
	}
	defer delete(all, allocator)
	gb.key_cols = make([]Column, len(src_cols), allocator)
	for c, i in src_cols {
		cp, err := gather_rows(c, allocator, all)
		testing.expectf(t, err == .None, "shell copy %d: %v", i, err)
		gb.key_cols[i] = cp
	}
	gb.key_ptrs = make([]^Column, len(src_cols), allocator)
	for i in 0 ..< len(src_cols) {
		gb.key_ptrs[i] = &gb.key_cols[i]
	}
	gb.groups = make(map[string][dynamic]int, 0, allocator)
	gb.order = make([dynamic]string, 0, allocator)
	return gb
}

// expect_groups_equal asserts both shells grouped identically: same key
// sequence (first-appearance order) and identical row arrays per group.
@(private)
expect_groups_equal :: proc(t: ^testing.T, what: string, a, b: ^Group_By) {
	testing.expectf(
		t,
		len(a.order) == len(b.order),
		"%s: %d groups != %d",
		what,
		len(a.order),
		len(b.order),
	)
	for k, i in a.order {
		ra := a.groups[k]
		rb, exists := b.groups[k]
		testing.expectf(t, exists, "%s: group %d missing", what, i)
		if !exists {
			continue
		}
		testing.expectf(
			t,
			len(ra) == len(rb),
			"%s: group %d len %d != %d",
			what,
			i,
			len(ra),
			len(rb),
		)
		for j in 0 ..< min(len(ra), len(rb)) {
			testing.expectf(
				t,
				ra[j] == rb[j],
				"%s: group %d row %d: %d != %d",
				what,
				i,
				j,
				ra[j],
				rb[j],
			)
		}
	}
}

// groupby_parallel_matches_sequential runs both grouping loops over 60k rows
// with two key columns (i32 + string), NULLs sprinkled in both, so chunks
// overlap heavily in key space. Results must be identical.
@(test)
groupby_parallel_matches_sequential :: proc(t: ^testing.T) {
	n := 60_000
	kvals := make([]i32, n)
	svals := make([]string, n)
	words := []string{"alpha", "beta", "gamma", "delta"}
	for i in 0 ..< n {
		kvals[i] = i32(i % 97)
		svals[i] = words[i % 4]
	}
	kc, e1 := column_from("k", kvals)
	delete(kvals)
	testing.expect(t, e1 == .None, "k column")
	sc, e2 := column_from("s", svals)
	delete(svals)
	testing.expect(t, e2 == .None, "s column")
	for i in 0 ..< n {
		if i % 11 == 3 {
			testing.expect(t, column_set_valid(&kc, i, false) == .None, "k null")
		}
		if i % 7 == 5 {
			testing.expect(t, column_set_valid(&sc, i, false) == .None, "s null")
		}
	}
	df, derr := dataframe_from_columns([]^Column{&kc, &sc})
	testing.expect(t, derr == .None, "df")
	defer dataframe_destroy(&df)

	cols := make([]^Column, 2)
	defer delete(cols)
	cols[0], _ = dataframe_get_column(&df, "k")
	cols[1], _ = dataframe_get_column(&df, "s")

	seq := groupby_shell(t, cols[:], context.allocator)
	defer dataframe_group_by_destroy(&seq)
	par := groupby_shell(t, cols[:], context.allocator)
	defer dataframe_group_by_destroy(&par)

	e3 := group_rows_sequential(&seq, n)
	testing.expectf(t, e3 == .None, "sequential error %v", e3)
	e4 := group_rows_parallel(&par, n)
	testing.expectf(t, e4 == .None, "parallel error %v", e4)

	expect_groups_equal(t, "large multi-key", &seq, &par)

	total := 0
	for _, g in par.groups {
		total += len(g)
	}
	testing.expectf(t, total == n, "partition property: %d rows != %d", total, n)
}

// groupby_parallel_small_forced runs the parallel kernel far below the
// dispatch threshold with a hand-checkable pattern: k = i % 3 yields
// first-appearance groups {rows ≡ 0}, {rows ≡ 1}, {rows ≡ 2}, each holding
// ascending rows in arithmetic progression.
@(test)
groupby_parallel_small_forced :: proc(t: ^testing.T) {
	n := 137
	vals := make([]i32, n)
	for i in 0 ..< n {
		vals[i] = i32(i % 3)
	}
	kc, err := column_from("k", vals)
	delete(vals)
	testing.expect(t, err == .None, "k column")

	df, derr := dataframe_from_columns([]^Column{&kc})
	testing.expect(t, derr == .None, "df")
	defer dataframe_destroy(&df)

	cols := make([]^Column, 1)
	defer delete(cols)
	cols[0], _ = dataframe_get_column(&df, "k")

	gb := groupby_shell(t, cols[:], context.allocator)
	defer dataframe_group_by_destroy(&gb)
	perr := group_rows_parallel(&gb, n)
	testing.expectf(t, perr == .None, "parallel error %v", perr)

	testing.expectf(t, len(gb.order) == 3, "expected 3 groups, got %d", len(gb.order))
	for key, g in gb.order {
		rows := gb.groups[key]
		testing.expectf(t, len(rows) == (n - g + 2) / 3, "group %d size", g)
		for j in 0 ..< len(rows) {
			testing.expectf(
				t,
				rows[j] == j * 3 + g,
				"group %d[%d] = %d, want %d",
				g,
				j,
				rows[j],
				j * 3 + g,
			)
		}
	}

	seq := groupby_shell(t, cols[:], context.allocator)
	defer dataframe_group_by_destroy(&seq)
	serr := group_rows_sequential(&seq, n)
	testing.expectf(t, serr == .None, "sequential error %v", serr)
	expect_groups_equal(t, "small mod-3", &seq, &gb)
}

// groupby_agg_end_to_end_large drives the public API past the threshold so
// dataframe_group_by takes the parallel path, then checks the aggregated
// output against expectations recomputed from the fixture pattern.
@(test)
groupby_agg_end_to_end_large :: proc(t: ^testing.T) {
	n := 150_000
	depts := make([]string, n)
	vs := make([]f64, n)
	for i in 0 ..< n {
		depts[i] = fmt.tprintf("dept%d", i % 5)
		vs[i] = f64(i % 1000) + 0.5
	}
	dc, e1 := column_from("dept", depts)
	delete(depts)
	testing.expect(t, e1 == .None, "dept")
	vc, e2 := column_from("v", vs)
	delete(vs)
	testing.expect(t, e2 == .None, "v")
	for i in 0 ..< n {
		if i % 13 == 8 {
			testing.expect(t, column_set_valid(&dc, i, false) == .None, "dept null")
		}
		if i % 17 == 4 {
			testing.expect(t, column_set_valid(&vc, i, false) == .None, "v null")
		}
	}
	df, derr := dataframe_from_columns([]^Column{&dc, &vc})
	testing.expect(t, derr == .None, "df")
	defer dataframe_destroy(&df)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	gb, gerr := dataframe_group_by(
		&df,
		[]^expr.Expr{expr.col(&ctx, "dept")},
	)
	testing.expectf(t, gerr == .None, "group_by error %v", gerr)
	defer dataframe_group_by_destroy(&gb)

	// 6 groups: dept0..dept4 in first-appearance order (rows 0..4), then the
	// NULL-dept group (first NULL dept row is 8).
	testing.expectf(t, len(gb.order) == 6, "got %d groups", len(gb.order))

	out, aerr := dataframe_group_by_agg(
		&gb,
		[]^expr.Expr{
			expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "v")), "cnt"),
			expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "v")), "total"),
		},
	)
	testing.expectf(t, aerr == .None, "agg error %v", aerr)
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 6, "agg row count")
	cnt_out, _ := dataframe_get_column(&out, "cnt")
	sum_out, _ := dataframe_get_column(&out, "total")

	for key, gi in gb.order {
		rows := gb.groups[key]
		first := rows[0]
		is_null_group := first % 13 == 8
		exp_cnt := 0
		exp_sum := 0.0
		for i in 0 ..< n {
			matches := false
			if is_null_group {
				matches = i % 13 == 8
			} else if i % 13 != 8 && i % 5 == first % 5 {
				matches = true
			}
			if !matches || i % 17 == 4 {
				continue
			}
			exp_cnt += 1
			exp_sum += f64(i % 1000) + 0.5
		}
		cv := column_typed_view(cnt_out, i64)[gi]
		sv := column_typed_view(sum_out, f64)[gi]
		testing.expectf(
			t,
			cv == i64(exp_cnt),
			"group %d: count %d != %d",
			gi,
			cv,
			exp_cnt,
		)
		testing.expectf(
			t,
			sv == exp_sum,
			"group %d: sum %v != %v",
			gi,
			sv,
			exp_sum,
		)
	}
}
