package lazy

// Stage 11 lazy-engine tests (S11.5): plan building is side-effect free, and
// collect() on a plan equals the equivalent eager pipeline.

import "core:testing"
import "../../dataframe"
import "../../dataframe/expr"

// --- comparison helpers (mirror property_test.odin for the lazy package) -----

df_equal :: proc(a, b: ^dataframe.DataFrame) -> bool {
	if a == b {
		return true
	}
	if dataframe.dataframe_num_rows(a) != dataframe.dataframe_num_rows(b) {
		return false
	}
	if dataframe.dataframe_num_cols(a) != dataframe.dataframe_num_cols(b) {
		return false
	}
	for i in 0 ..< dataframe.dataframe_num_cols(a) {
		ca, a_err := dataframe.dataframe_column_at(a, i)
		cb, b_err := dataframe.dataframe_column_at(b, i)
		if a_err != .None || b_err != .None || !column_equal(ca, cb) {
			return false
		}
	}
	return true
}

column_equal :: proc(a, b: ^dataframe.Column) -> bool {
	if dataframe.column_name(a) != dataframe.column_name(b) {
		return false
	}
	if dataframe.column_dtype(a) != dataframe.column_dtype(b) {
		return false
	}
	if dataframe.column_len(a) != dataframe.column_len(b) {
		return false
	}
	for i in 0 ..< dataframe.column_len(a) {
		if dataframe.column_is_valid(a, i) != dataframe.column_is_valid(b, i) {
			return false
		}
		if dataframe.column_is_valid(a, i) && !column_slot_equal(a, b, i) {
			return false
		}
	}
	return true
}

column_slot_equal :: proc(a, b: ^dataframe.Column, i: int) -> bool {
	switch dataframe.column_dtype(a) {
	case typeid_of(bool):
		va, _, _ := dataframe.column_get(a, i, bool)
		vb, _, _ := dataframe.column_get(b, i, bool)
		return va == vb
	case typeid_of(i32):
		va, _, _ := dataframe.column_get(a, i, i32)
		vb, _, _ := dataframe.column_get(b, i, i32)
		return va == vb
	case typeid_of(i64):
		va, _, _ := dataframe.column_get(a, i, i64)
		vb, _, _ := dataframe.column_get(b, i, i64)
		return va == vb
	case typeid_of(f64):
		va, _, _ := dataframe.column_get(a, i, f64)
		vb, _, _ := dataframe.column_get(b, i, f64)
		return va == vb
	case typeid_of(string):
		va, _, _ := dataframe.column_get(a, i, string)
		vb, _, _ := dataframe.column_get(b, i, string)
		return va == vb
	}
	return false
}

// --- fixture -----------------------------------------------------------------

// make_fixture builds a 10-row frame: id i64, grp string (a/b/c), val f64
// with NULL on rows 0, 4, 8, ok bool.
make_fixture :: proc() -> dataframe.DataFrame {
	df := dataframe.dataframe_create(context.allocator)
	id, _ := dataframe.column_from("id", []i64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	dataframe.dataframe_add_column(&df, &id)
	grp, _ := dataframe.column_from("grp", []string{"a", "b", "c", "a", "b", "c", "a", "b", "c", "a"})
	dataframe.dataframe_add_column(&df, &grp)
	val, _ := dataframe.column_from_with_valid(
		"val",
		[]f64{0.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5},
		[]bool{false, true, true, true, false, true, true, true, false, true},
	)
	dataframe.dataframe_add_column(&df, &val)
	ok, _ := dataframe.column_from("ok", []bool{true, false, true, true, false, true, false, true, true, false})
	dataframe.dataframe_add_column(&df, &ok)
	return df
}

// --- S11.5: plan building is side-effect free --------------------------------

@(test)
lazy_plan_building_side_effect_free :: proc(t: ^testing.T) {
	// A nonexistent CSV is not read during building; only collect fails.
	lf := scan_csv("/nonexistent/definitely_not_here.csv")
	defer destroy(&lf)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	lf2 := filter(lf, expr.ge(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, 0)))
	lf2 = sort(lf2, []dataframe.Sort_Key{dataframe.sort_key("x")})
	lf2 = limit(lf2, 5)
	// Building must not have raised anything (builders do not error).

	_, err := collect(lf2)
	testing.expect(t, err == .CSV_Error, "collect on missing CSV file")
}

@(test)
lazy_filter_unknown_column_collect_error :: proc(t: ^testing.T) {
	df := make_fixture()
	defer dataframe.dataframe_destroy(&df)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	lf := scan_dataframe(&df)
	defer destroy(&lf)
	lf = filter(lf, expr.eq(&ctx, expr.col(&ctx, "nope"), expr.lit(&ctx, 1)))

	_, err := collect(lf)
	testing.expect(t, err == .Column_Not_Found, "unknown column surfaces at collect")
}

@(test)
lazy_agg_requires_group_by :: proc(t: ^testing.T) {
	df := make_fixture()
	defer dataframe.dataframe_destroy(&df)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	lf := scan_dataframe(&df)
	defer destroy(&lf)
	lf = limit(lf, 3)

	_, err := agg(lf, []^expr.Expr{expr.sum_(&ctx, expr.col(&ctx, "val"))})
	testing.expect(t, err == .Invalid_Argument, "agg on a non-group_by plan root")
}

@(test)
lazy_group_by_without_agg_collect_error :: proc(t: ^testing.T) {
	df := make_fixture()
	defer dataframe.dataframe_destroy(&df)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	lf := scan_dataframe(&df)
	defer destroy(&lf)
	lf = group_by(lf, []^expr.Expr{expr.col(&ctx, "grp")})

	_, err := collect(lf)
	testing.expect(t, err == .Invalid_Argument, "collecting a group_by with no agg")
}

// --- S11.5: collect equals eager ---------------------------------------------

@(test)
lazy_collect_scan_dataframe_root_copies :: proc(t: ^testing.T) {
	df := make_fixture()
	defer dataframe.dataframe_destroy(&df)

	lf := scan_dataframe(&df)
	defer destroy(&lf)

	out, err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, err == .None, "collect scan root")
	testing.expect(t, df_equal(&df, &out), "root scan copies the source")

	// The copy is deep: mutating the result must not touch the source.
	c, c_err := dataframe.dataframe_column_at(&out, 0)
	testing.expect(t, c_err == .None, "mutate result column")
	dataframe.column_set(c, 0, i64(99))
	testing.expect(t, !df_equal(&df, &out), "result is an owned deep copy")
}

@(test)
lazy_collect_pipeline_equals_eager :: proc(t: ^testing.T) {
	df := make_fixture()
	defer dataframe.dataframe_destroy(&df)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	pred := expr.and_(
		&ctx,
		expr.ge(&ctx, expr.col(&ctx, "val"), expr.lit(&ctx, 2.0)),
		expr.not_(&ctx, expr.eq(&ctx, expr.col(&ctx, "ok"), expr.lit(&ctx, false))),
	)
	keys := []dataframe.Sort_Key{dataframe.sort_key("val", .Desc)}

	eager_f, ef_err := dataframe.dataframe_filter(&df, pred)
	testing.expect(t, ef_err == .None, "eager filter")
	defer dataframe.dataframe_destroy(&eager_f)
	eager_s, es_err := dataframe.dataframe_sort(&eager_f, keys)
	testing.expect(t, es_err == .None, "eager sort")
	defer dataframe.dataframe_destroy(&eager_s)
	eager_h, eh_err := dataframe.dataframe_limit(&eager_s, 4)
	testing.expect(t, eh_err == .None, "eager limit")
	defer dataframe.dataframe_destroy(&eager_h)

	lf := scan_dataframe(&df)
	defer destroy(&lf)
	lf = filter(lf, pred)
	lf = sort(lf, keys)
	lf = limit(lf, 4)
	out, err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, err == .None, "pipeline collect")
	testing.expect(t, df_equal(&eager_h, &out), "lazy pipeline == eager pipeline")
}

@(test)
lazy_collect_select_equals_eager :: proc(t: ^testing.T) {
	df := make_fixture()
	defer dataframe.dataframe_destroy(&df)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	exprs := []^expr.Expr{
		expr.col(&ctx, "id"),
		expr.alias(&ctx, expr.mul(&ctx, expr.col(&ctx, "val"), expr.lit(&ctx, 2.0)), "double"),
		expr.col(&ctx, "grp"),
	}

	eager_s, e_err := dataframe.dataframe_select(&df, exprs)
	testing.expect(t, e_err == .None, "eager select")
	defer dataframe.dataframe_destroy(&eager_s)

	lf := scan_dataframe(&df)
	defer destroy(&lf)
	lf = select(lf, exprs)
	out, err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, err == .None, "select collect")
	testing.expect(t, df_equal(&eager_s, &out), "lazy select == eager select")
}

@(test)
lazy_collect_slice_equals_eager :: proc(t: ^testing.T) {
	df := make_fixture()
	defer dataframe.dataframe_destroy(&df)

	eager_s, e_err := dataframe.dataframe_slice(&df, 2, 5)
	testing.expect(t, e_err == .None, "eager slice")
	defer dataframe.dataframe_destroy(&eager_s)

	lf := scan_dataframe(&df)
	defer destroy(&lf)
	lf = slice(lf, 2, 5)
	out, err := collect(lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, err == .None, "slice collect")
	testing.expect(t, df_equal(&eager_s, &out), "lazy slice == eager slice")
}

@(test)
lazy_collect_group_agg_equals_eager :: proc(t: ^testing.T) {
	df := make_fixture()
	defer dataframe.dataframe_destroy(&df)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	keys := []^expr.Expr{expr.col(&ctx, "grp")}
	aggs := []^expr.Expr{
		expr.alias(&ctx, expr.sum_(&ctx, expr.col(&ctx, "val")), "sum_val"),
		expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "val")), "n"),
	}

	gb, g_err := dataframe.dataframe_group_by(&df, keys)
	testing.expect(t, g_err == .None, "eager group_by")
	defer dataframe.dataframe_group_by_destroy(&gb)
	eager_a, ea_err := dataframe.dataframe_group_by_agg(&gb, aggs)
	testing.expect(t, ea_err == .None, "eager group agg")
	defer dataframe.dataframe_destroy(&eager_a)

	lf := scan_dataframe(&df)
	defer destroy(&lf)
	lf = group_by(lf, keys)
	agg_lf, a_err := agg(lf, aggs)
	testing.expect(t, a_err == .None, "lazy agg attaches")
	out, err := collect(agg_lf)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, err == .None, "group agg collect")
	testing.expect(t, df_equal(&eager_a, &out), "lazy group agg == eager group agg")
}

@(test)
lazy_collect_join_equals_eager :: proc(t: ^testing.T) {
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

	eager_j, ej_err := dataframe.dataframe_inner_join(&left_df, &right_df, []string{"dept"}, []string{"dept"})
	testing.expect(t, ej_err == .None, "eager inner join")
	defer dataframe.dataframe_destroy(&eager_j)

	// Two independent plans: join must clone the right tree into the left
	// arena (exercises the cross-plan path).
	l := scan_dataframe(&left_df)
	defer destroy(&l)
	r := scan_dataframe(&right_df)
	defer destroy(&r)
	j := join(l, r, .Inner, []string{"dept"}, []string{"dept"})
	defer destroy(&j)
	out, err := collect(j)
	defer dataframe.dataframe_destroy(&out)
	testing.expect(t, err == .None, "join collect")
	testing.expect(t, df_equal(&eager_j, &out), "lazy join == eager join")
}

@(test)
lazy_collect_repeatable :: proc(t: ^testing.T) {
	df := make_fixture()
	defer dataframe.dataframe_destroy(&df)

	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	lf := scan_dataframe(&df)
	defer destroy(&lf)
	lf = filter(lf, expr.ge(&ctx, expr.col(&ctx, "val"), expr.lit(&ctx, 2.0)))

	out1, e1 := collect(lf)
	defer dataframe.dataframe_destroy(&out1)
	out2, e2 := collect(lf)
	defer dataframe.dataframe_destroy(&out2)
	testing.expect(t, e1 == .None && e2 == .None, "repeat collects succeed")
	testing.expect(t, df_equal(&out1, &out2), "collect does not consume the plan")
}
