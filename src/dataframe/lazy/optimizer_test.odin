package lazy

// Stage 12 optimizer tests (S12.6): each rewrite rule is checked twice — the
// plan-before result equals the plan-after result on the same data, and the
// pruned Scan_CSV node carries exactly the columns the plan references.

import "core:os"
import "core:strings"
import "core:testing"
import "../../dataframe"
import "../../dataframe/expr"

// --- fixtures -----------------------------------------------------------------

// opt_fixture_csv writes a 4-column frame (id, grp, val with NULLs, ok) to a
// unique scratch CSV and reads it back; the round-tripped frame is what eager
// reference pipelines run against.
opt_fixture_csv :: proc(t: ^testing.T, name: string) -> (path: string, full: dataframe.DataFrame) {
	df := dataframe.dataframe_create(context.allocator)
	defer dataframe.dataframe_destroy(&df)
	id, _ := dataframe.column_from("id", []i64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	dataframe.dataframe_add_column(&df, &id)
	grp, _ := dataframe.column_from("grp", []string{"a", "b", "c", "a", "b", "c", "a", "b", "c", "a"})
	dataframe.dataframe_add_column(&df, &grp)
	val, _ := dataframe.column_from_with_valid(
		"val",
		[]f64{0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5},
		[]bool{false, true, true, true, false, true, true, true, false, true},
	)
	dataframe.dataframe_add_column(&df, &val)
	ok, _ := dataframe.column_from("ok", []bool{true, false, true, true, false, true, false, true, true, false})
	dataframe.dataframe_add_column(&df, &ok)

	path = strings.concatenate([]string{"/tmp/thunder_opt_", name, ".csv"})
	testing.expect(t, dataframe.dataframe_write_csv(&df, path) == .None, "write fixture csv")

	f, f_err := dataframe.dataframe_read_csv(path)
	testing.expect(t, f_err == .None, "read fixture csv")
	return path, f
}

opt_cleanup :: proc(t: ^testing.T, path: string) {
	if err := os.remove(path); err != os.ERROR_NONE {
		testing.expect(t, false, "remove scratch csv")
	}
	delete_string(path, context.allocator)
}

// opt_collect_scans gathers copies of every Scan_CSV node in a plan tree.
opt_collect_scans :: proc(plan: ^Logical_Plan, out: ^[dynamic]Scan_CSV) {
	switch n in plan^ {
	case Scan_CSV:
		append(out, n)
	case Scan_DF:
	case Filter:
		opt_collect_scans(n.child, out)
	case Projection:
		opt_collect_scans(n.child, out)
	case Sort:
		opt_collect_scans(n.child, out)
	case Group_By:
		opt_collect_scans(n.child, out)
	case Limit:
		opt_collect_scans(n.child, out)
	case Slice:
		opt_collect_scans(n.child, out)
	case Join:
		opt_collect_scans(n.left, out)
		opt_collect_scans(n.right, out)
	}
}

// opt_expect_columns asserts the single scan of a plan was pruned to exactly
// want (in order). An empty want asserts no pruning happened.
opt_expect_columns :: proc(t: ^testing.T, root: ^Logical_Plan, want: []string, msg: string) {
	scans := make([dynamic]Scan_CSV, context.allocator)
	defer delete(scans)
	opt_collect_scans(root, &scans)
	_ = msg
	testing.expect(t, len(scans) == 1, "one scan")
	got := scans[0].columns
	testing.expect(t, len(got) == len(want), "pruned column count")
	for i in 0 ..< len(want) {
		if i < len(got) {
			testing.expect(t, got[i] == want[i], "pruned column order")
		}
	}
}

// --- S12.1 rewrite rules ------------------------------------------------------

@(test)
opt_filter_only_no_prune :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "filter_only")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	pred := expr.ge(&ctx, expr.col(&ctx, "val"), expr.lit(&ctx, 2.0))

	lf := scan_csv(path)
	defer destroy(&lf)
	lf = filter(lf, pred)

	optimize(lf.root, lf.plan.alloc)
	opt_expect_columns(t, lf.root, []string{}, "filter alone")

	eager, e_err := dataframe.dataframe_filter(&full, pred)
	testing.expect(t, e_err == .None, "eager filter")
	defer dataframe.dataframe_destroy(&eager)
	out, c_err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "filter-only == eager")
}

@(test)
opt_projection_prunes :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "proj")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	exprs := []^expr.Expr{expr.col(&ctx, "id"), expr.col(&ctx, "val")}

	lf := scan_csv(path)
	defer destroy(&lf)
	lf = select(lf, exprs)

	optimize(lf.root, lf.plan.alloc)
	opt_expect_columns(t, lf.root, []string{"id", "val"}, "projection prunes")

	eager, e_err := dataframe.dataframe_select(&full, exprs)
	testing.expect(t, e_err == .None, "eager select")
	defer dataframe.dataframe_destroy(&eager)
	out, c_err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "projection == eager select")
}

@(test)
opt_filter_select_prunes :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "filter_select")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	exprs := []^expr.Expr{expr.col(&ctx, "id"), expr.col(&ctx, "val")}
	pred := expr.ge(&ctx, expr.col(&ctx, "val"), expr.lit(&ctx, 2.0))

	lf := scan_csv(path)
	defer destroy(&lf)
	lf = select(filter(lf, pred), exprs)

	optimize(lf.root, lf.plan.alloc)
	opt_expect_columns(t, lf.root, []string{"id", "val"}, "filter+select prunes to referenced columns")

	f, e_err := dataframe.dataframe_filter(&full, pred)
	testing.expect(t, e_err == .None, "eager filter")
	defer dataframe.dataframe_destroy(&f)
	eager, s_err := dataframe.dataframe_select(&f, exprs)
	testing.expect(t, s_err == .None, "eager select")
	defer dataframe.dataframe_destroy(&eager)
	out, c_err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "filter+select == eager")
}

@(test)
opt_sort_select_prunes :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "sort_select")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	exprs := []^expr.Expr{expr.col(&ctx, "id"), expr.col(&ctx, "val")}
	keys := []dataframe.Sort_Key{dataframe.sort_key("id")}

	lf := scan_csv(path)
	defer destroy(&lf)
	lf = sort(select(lf, exprs), keys)

	optimize(lf.root, lf.plan.alloc)
	opt_expect_columns(t, lf.root, []string{"id", "val"}, "sort+select prunes to referenced columns")

	s, e_err := dataframe.dataframe_select(&full, exprs)
	testing.expect(t, e_err == .None, "eager select")
	defer dataframe.dataframe_destroy(&s)
	eager, so_err := dataframe.dataframe_sort(&s, keys)
	testing.expect(t, so_err == .None, "eager sort")
	defer dataframe.dataframe_destroy(&eager)
	out, c_err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "sort+select == eager")
}

@(test)
opt_group_agg_prunes :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "group_agg")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	keys := []^expr.Expr{expr.col(&ctx, "grp")}
	aggs := []^expr.Expr{
		expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "val")), "sum_val"),
		expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "val")), "n"),
	}

	lf := scan_csv(path)
	lf = group_by(lf, keys)
	agg_lf, a_err := agg(lf, aggs)
	testing.expect(t, a_err == .None, "agg attaches")
	defer destroy(&agg_lf)

	optimize(agg_lf.root, agg_lf.plan.alloc)
	opt_expect_columns(t, agg_lf.root, []string{"grp", "val"}, "group+agg prunes to key and agg columns")

	gb, g_err := dataframe.dataframe_group_by(&full, keys)
	testing.expect(t, g_err == .None, "eager group_by")
	defer dataframe.dataframe_group_by_destroy(&gb)
	eager, ea_err := dataframe.dataframe_group_by_agg(&gb, aggs)
	testing.expect(t, ea_err == .None, "eager group agg")
	defer dataframe.dataframe_destroy(&eager)
	out, c_err := collect(agg_lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "group+agg == eager")
}

@(test)
opt_limit_select_prunes :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "limit_select")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	exprs := []^expr.Expr{expr.col(&ctx, "id")}

	lf := scan_csv(path)
	defer destroy(&lf)
	lf = limit(select(lf, exprs), 5)

	optimize(lf.root, lf.plan.alloc)
	opt_expect_columns(t, lf.root, []string{"id"}, "limit+select prunes")

	s, e_err := dataframe.dataframe_select(&full, exprs)
	testing.expect(t, e_err == .None, "eager select")
	defer dataframe.dataframe_destroy(&s)
	eager, l_err := dataframe.dataframe_limit(&s, 5)
	testing.expect(t, l_err == .None, "eager limit")
	defer dataframe.dataframe_destroy(&eager)
	out, c_err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "limit+select == eager")
}

@(test)
opt_no_refs_no_prune :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "no_refs")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	lf := scan_csv(path)
	defer destroy(&lf)
	lf = limit(lf, 3)

	optimize(lf.root, lf.plan.alloc)
	opt_expect_columns(t, lf.root, []string{}, "no referenced columns does not prune")

	eager, e_err := dataframe.dataframe_limit(&full, 3)
	testing.expect(t, e_err == .None, "eager limit")
	defer dataframe.dataframe_destroy(&eager)
	out, c_err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "no-refs == eager")
}

@(test)
opt_alias_projection_reads_source_column :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "alias_proj")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	exprs := []^expr.Expr{expr.alias(&ctx, expr.col(&ctx, "val"), "v")}
	pred := expr.gt(&ctx, expr.col(&ctx, "v"), expr.lit(&ctx, 0.0))

	lf := scan_csv(path)
	defer destroy(&lf)
	lf = filter(select(lf, exprs), pred)

	optimize(lf.root, lf.plan.alloc)
	opt_expect_columns(t, lf.root, []string{"val"}, "alias output name is not a scan column")

	s, e_err := dataframe.dataframe_select(&full, exprs)
	testing.expect(t, e_err == .None, "eager select")
	defer dataframe.dataframe_destroy(&s)
	eager, f_err := dataframe.dataframe_filter(&s, pred)
	testing.expect(t, f_err == .None, "eager filter")
	defer dataframe.dataframe_destroy(&eager)
	out, c_err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "alias pipeline == eager")
}

@(test)
opt_join_is_barrier :: proc(t: ^testing.T) {
	left_df := dataframe.dataframe_create(context.allocator)
	defer dataframe.dataframe_destroy(&left_df)
	id, _ := dataframe.column_from("id", []i64{1, 2, 3, 4})
	dataframe.dataframe_add_column(&left_df, &id)
	dept, _ := dataframe.column_from("dept", []string{"eng", "eng", "sales", "hr"})
	dataframe.dataframe_add_column(&left_df, &dept)
	score, _ := dataframe.column_from("score", []f64{10, 20, 30, 40})
	dataframe.dataframe_add_column(&left_df, &score)

	right_df := dataframe.dataframe_create(context.allocator)
	defer dataframe.dataframe_destroy(&right_df)
	rdept, _ := dataframe.column_from("dept", []string{"eng", "sales", "hr"})
	dataframe.dataframe_add_column(&right_df, &rdept)
	mgr, _ := dataframe.column_from("mgr", []string{"ada", "grace", "lin"})
	dataframe.dataframe_add_column(&right_df, &mgr)

	lp := strings.concatenate([]string{"/tmp/thunder_opt_join_left.csv"})
	rp := strings.concatenate([]string{"/tmp/thunder_opt_join_right.csv"})
	defer opt_cleanup(t, lp)
	defer opt_cleanup(t, rp)
	testing.expect(t, dataframe.dataframe_write_csv(&left_df, lp) == .None, "write left csv")
	testing.expect(t, dataframe.dataframe_write_csv(&right_df, rp) == .None, "write right csv")
	left_full, le := dataframe.dataframe_read_csv(lp)
	defer dataframe.dataframe_destroy(&left_full)
	right_full, re := dataframe.dataframe_read_csv(rp)
	defer dataframe.dataframe_destroy(&right_full)
	testing.expect(t, le == .None && re == .None, "read join fixtures")

	eager, ej_err := dataframe.dataframe_inner_join(&left_full, &right_full, []string{"dept"}, []string{"dept"})
	testing.expect(t, ej_err == .None, "eager inner join")
	defer dataframe.dataframe_destroy(&eager)

	l := scan_csv(lp)
	defer destroy(&l)
	r := scan_csv(rp)
	defer destroy(&r)
	j := join(l, r, .Inner, []string{"dept"}, []string{"dept"})
	defer destroy(&j)

	optimize(j.root, j.plan.alloc)
	scans := make([dynamic]Scan_CSV, context.allocator)
	defer delete(scans)
	opt_collect_scans(j.root, &scans)
	testing.expect(t, len(scans) == 2, "join has two scans")
	for s in scans {
		testing.expect(t, len(s.columns) == 0, "join children are not pruned")
	}

	out, c_err := collect(j)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "join == eager")
}

@(test)
opt_repeat_collect_stable :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "repeat")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	exprs := []^expr.Expr{expr.col(&ctx, "id"), expr.col(&ctx, "val")}

	lf := scan_csv(path)
	defer destroy(&lf)
	lf = select(lf, exprs)

	out1, e1 := collect(lf)
	defer dataframe.dataframe_destroy(&out1)
	out2, e2 := collect(lf)
	defer dataframe.dataframe_destroy(&out2)
	testing.expect(t, e1 == .None && e2 == .None, "repeat collects succeed")
	testing.expect(t, df_equal(&out1, &out2), "repeat collect stable")

	opt_expect_columns(t, lf.root, []string{"id", "val"}, "re-optimized scan still pruned")
}

// --- S12.3 column pruning (intermediate projections) -------------------------

@(test)
opt_intermediate_projection_prunes :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "intermediate")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	inner_exprs := []^expr.Expr{expr.col(&ctx, "id"), expr.col(&ctx, "val"), expr.col(&ctx, "grp")}
	outer_exprs := []^expr.Expr{expr.col(&ctx, "id"), expr.col(&ctx, "val")}

	lf := scan_csv(path)
	defer destroy(&lf)
	lf = select(select(lf, inner_exprs), outer_exprs)

	optimize(lf.root, lf.plan.alloc)
	opt_expect_columns(t, lf.root, []string{"id", "val"}, "intermediate projection prunes to referenced columns")

	// The inner projection must no longer produce the unused "grp" output.
	#partial switch outer in lf.root^ {
	case Projection:
		#partial switch inner in outer.child^ {
		case Projection:
			testing.expect(t, len(inner.exprs) == 2, "inner projection dropped unused expr")
			testing.expect(t, expr_output_name(inner.exprs[0]) == "id", "inner keeps id")
			testing.expect(t, expr_output_name(inner.exprs[1]) == "val", "inner keeps val")
		case:
			testing.expect(t, false, "outer projection child is not a projection")
		}
	case:
		testing.expect(t, false, "root is not the outer projection")
	}

	eager1, e1 := dataframe.dataframe_select(&full, inner_exprs)
	defer dataframe.dataframe_destroy(&eager1)
	testing.expect(t, e1 == .None, "eager inner select")
	eager, e2 := dataframe.dataframe_select(&eager1, outer_exprs)
	defer dataframe.dataframe_destroy(&eager)
	testing.expect(t, e2 == .None, "eager outer select")
	out, c_err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "pruned intermediate == eager")
}

// --- S12.4 constant folding -------------------------------------------------

@(test)
opt_fold_constants_equivalence :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "fold")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	// val + (2 * 3): the literal product folds to 6 before execution, but the
	// eager reference still evaluates the unfolded tree.
	exprs := []^expr.Expr{
		expr.alias(&ctx, expr.add(&ctx, expr.col(&ctx, "val"), expr.mul(&ctx, expr.lit(&ctx, 2.0), expr.lit(&ctx, 3.0))), "v"),
	}

	lf := scan_csv(path)
	defer destroy(&lf)
	lf = select(lf, exprs)

	eager, e_err := dataframe.dataframe_select(&full, exprs)
	testing.expect(t, e_err == .None, "eager select")
	defer dataframe.dataframe_destroy(&eager)
	out, c_err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "folded plan == eager")
}

// --- S12.2 predicate pushdown ----------------------------------------------

@(test)
opt_pushdown_filter_through_sort :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "push_sort")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	keys := []dataframe.Sort_Key{dataframe.sort_key("id")}
	pred := expr.gt(&ctx, expr.col(&ctx, "val"), expr.lit(&ctx, 2.0))

	lf := scan_csv(path)
	defer destroy(&lf)
	lf = sort(lf, keys)
	lf = filter(lf, pred)

	optimize(lf.root, lf.plan.alloc)
	#partial switch r in lf.root^ {
	case Filter:
		// The top filter stays; a copy is pushed below the sort.
		#partial switch s in r.child^ {
		case Sort:
			#partial switch sc in s.child^ {
			case Filter:
			case:
				testing.expect(t, false, "no filter below sort")
			}
		case:
			testing.expect(t, false, "filter did not cross the sort")
		}
	case:
		testing.expect(t, false, "root filter lost")
	}

	s, e_err := dataframe.dataframe_sort(&full, keys)
	testing.expect(t, e_err == .None, "eager sort")
	defer dataframe.dataframe_destroy(&s)
	eager, f_err := dataframe.dataframe_filter(&s, pred)
	testing.expect(t, f_err == .None, "eager filter")
	defer dataframe.dataframe_destroy(&eager)
	out, c_err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "pushed filter == eager")
}

@(test)
opt_pushdown_into_join_left :: proc(t: ^testing.T) {
	left_df := dataframe.dataframe_create(context.allocator)
	defer dataframe.dataframe_destroy(&left_df)
	id, _ := dataframe.column_from("id", []i64{1, 2, 3, 4})
	dataframe.dataframe_add_column(&left_df, &id)
	dept, _ := dataframe.column_from("dept", []string{"eng", "eng", "sales", "hr"})
	dataframe.dataframe_add_column(&left_df, &dept)
	score, _ := dataframe.column_from("score", []f64{10, 20, 30, 40})
	dataframe.dataframe_add_column(&left_df, &score)

	right_df := dataframe.dataframe_create(context.allocator)
	defer dataframe.dataframe_destroy(&right_df)
	rdept, _ := dataframe.column_from("dept", []string{"eng", "sales", "hr"})
	dataframe.dataframe_add_column(&right_df, &rdept)
	mgr, _ := dataframe.column_from("mgr", []string{"ada", "grace", "lin"})
	dataframe.dataframe_add_column(&right_df, &mgr)

	lp := strings.concatenate([]string{"/tmp/thunder_opt_pushjoin_left.csv"})
	rp := strings.concatenate([]string{"/tmp/thunder_opt_pushjoin_right.csv"})
	defer opt_cleanup(t, lp)
	defer opt_cleanup(t, rp)
	testing.expect(t, dataframe.dataframe_write_csv(&left_df, lp) == .None, "write left csv")
	testing.expect(t, dataframe.dataframe_write_csv(&right_df, rp) == .None, "write right csv")
	left_full, le := dataframe.dataframe_read_csv(lp)
	defer dataframe.dataframe_destroy(&left_full)
	right_full, re := dataframe.dataframe_read_csv(rp)
	defer dataframe.dataframe_destroy(&right_full)
	testing.expect(t, le == .None && re == .None, "read join fixtures")

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	left_exprs := []^expr.Expr{expr.col(&ctx, "dept"), expr.col(&ctx, "score")}
	right_exprs := []^expr.Expr{expr.col(&ctx, "dept")}
	pred := expr.gt(&ctx, expr.col(&ctx, "score"), expr.lit(&ctx, 20.0))

	ls, le_err := dataframe.dataframe_select(&left_full, left_exprs)
	testing.expect(t, le_err == .None, "eager left select")
	defer dataframe.dataframe_destroy(&ls)
	rs, re_err := dataframe.dataframe_select(&right_full, right_exprs)
	testing.expect(t, re_err == .None, "eager right select")
	defer dataframe.dataframe_destroy(&rs)
	j, j_err := dataframe.dataframe_inner_join(&ls, &rs, []string{"dept"}, []string{"dept"})
	testing.expect(t, j_err == .None, "eager inner join")
	defer dataframe.dataframe_destroy(&j)
	eager, f_err := dataframe.dataframe_filter(&j, pred)
	testing.expect(t, f_err == .None, "eager filter")
	defer dataframe.dataframe_destroy(&eager)

	l := select(scan_csv(lp), left_exprs)
	defer destroy(&l)
	r := select(scan_csv(rp), right_exprs)
	defer destroy(&r)
	jl := join(l, r, .Inner, []string{"dept"}, []string{"dept"})
	defer destroy(&jl)
	jl = filter(jl, pred)

	optimize(jl.root, jl.plan.alloc)
	#partial switch fl in jl.root^ {
	case Filter:
		#partial switch jn in fl.child^ {
		case Join:
			// The pushed filter wraps the left side; the rewrite below the
			// projection lands under the projection's child.
			#partial switch pushed in jn.left^ {
			case Filter:
				#partial switch pn in pushed.child^ {
				case Projection:
					#partial switch pc in pn.child^ {
					case Filter:
					case:
						testing.expect(t, false, "no filter below projection")
					}
				case:
					testing.expect(t, false, "pushed filter child is not the projection")
				}
			case:
				testing.expect(t, false, "join left is not a pushed filter")
			}
		case:
			testing.expect(t, false, "filter did not cross the join")
		}
	case:
		testing.expect(t, false, "root filter lost")
	}

	out, c_err := collect(jl)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "join-pushed filter == eager")
}

// --- S12.5 common subexpression elimination --------------------------------

@(test)
opt_cse_structurally_equal :: proc(t: ^testing.T) {
	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	add1 := expr.add(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, 1.0))
	add2 := expr.add(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, 1.0))
	testing.expect(t, exprs_structurally_equal(add1, add2), "identical trees equal")
	testing.expect(t, exprs_structurally_equal(add1, add1), "same node equal")

	other := expr.add(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, 2.0))
	testing.expect(t, !exprs_structurally_equal(add1, other), "different literal not equal")

	mul1 := expr.mul(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, 1.0))
	testing.expect(t, !exprs_structurally_equal(add1, mul1), "different op not equal")

	col_x := expr.col(&ctx, "x")
	testing.expect(t, !exprs_structurally_equal(col_x, expr.col(&ctx, "y")), "different column not equal")

	al1 := expr.alias(&ctx, col_x, "n")
	testing.expect(t, exprs_structurally_equal(al1, expr.alias(&ctx, col_x, "n")), "same alias equal")
	testing.expect(t, !exprs_structurally_equal(al1, expr.alias(&ctx, col_x, "m")), "different alias name not equal")

	testing.expect(t, exprs_structurally_equal(nil, nil), "nil == nil")
	testing.expect(t, !exprs_structurally_equal(add1, nil), "nil != tree")
}

@(test)
opt_cse_share_computed :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "cse_share")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	// Two aliases of one computed expression: CSE evaluates it once and copies
	// the column into both output positions.
	comp := expr.mul(&ctx, expr.col(&ctx, "val"), expr.lit(&ctx, 2.0))
	exprs := []^expr.Expr{
		expr.alias(&ctx, comp, "dbl1"),
		expr.alias(&ctx, comp, "dbl2"),
		expr.col(&ctx, "id"),
	}

	lf := scan_csv(path)
	defer destroy(&lf)
	lf = select(lf, exprs)

	eager, e_err := dataframe.dataframe_select(&full, exprs)
	testing.expect(t, e_err == .None, "eager select")
	defer dataframe.dataframe_destroy(&eager)
	out, c_err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "cse projection == eager")
}

@(test)
opt_cse_col_alias_share :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "cse_col_alias")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)
	// A bare column and an alias of the same column compute one value; the
	// plan must still emit both columns under their own names.
	exprs := []^expr.Expr{
		expr.col(&ctx, "val"),
		expr.alias(&ctx, expr.col(&ctx, "val"), "v2"),
	}

	lf := scan_csv(path)
	defer destroy(&lf)
	lf = select(lf, exprs)

	eager, e_err := dataframe.dataframe_select(&full, exprs)
	testing.expect(t, e_err == .None, "eager select")
	defer dataframe.dataframe_destroy(&eager)
	out, c_err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, c_err == .None, "collect")
	testing.expect(t, df_equal(&eager, &out), "col+alias share == eager")
}

@(test)
opt_cse_error_semantics :: proc(t: ^testing.T) {
	path, full := opt_fixture_csv(t, "cse_err")
	defer dataframe.dataframe_destroy(&full)
	defer opt_cleanup(t, path)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	dup := []^expr.Expr{
		expr.alias(&ctx, expr.col(&ctx, "val"), "a"),
		expr.alias(&ctx, expr.col(&ctx, "val"), "a"),
	}
	_, e_err := dataframe.dataframe_select(&full, dup)
	testing.expect(t, e_err == .Duplicate_Column_Name, "eager rejects duplicate names")
	ld := select(scan_csv(path), dup)
	defer destroy(&ld)
	_, c_err := collect(ld)
	testing.expect(t, c_err == .Duplicate_Column_Name, "cse preserves duplicate-name error")

	unnamed := []^expr.Expr{
		expr.alias(&ctx, expr.col(&ctx, "val"), "a"),
		expr.mul(&ctx, expr.col(&ctx, "val"), expr.lit(&ctx, 2.0)),
	}
	_, u_err := dataframe.dataframe_select(&full, unnamed)
	testing.expect(t, u_err == .Invalid_Argument, "eager rejects unnamed expr")
	lu := select(scan_csv(path), unnamed)
	defer destroy(&lu)
	_, uc_err := collect(lu)
	testing.expect(t, uc_err == .Invalid_Argument, "cse preserves unnamed-expr error")
}
