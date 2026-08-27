package dataframe

// Stage 10 tests for the chain-friendly aliases (S10.2): the short names
// must resolve to the dataframe_* procs and compose into pipelines.

import "core:testing"
import "expr"

@(test)
chain_aliases_pipeline :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	// qualified short names: dataframe.filter / dataframe.select /
	// dataframe.sort / dataframe.head — compose left to right.
	filtered, f_err := filter(&df, expr.gt(&ctx, expr.col(&ctx, "x"), expr.lit(&ctx, i32(20))))
	testing.expect(t, f_err == .None, "filter alias")
	defer dataframe_destroy(&filtered)

	sorted, s_err := sort(&filtered, []Sort_Key{sort_key("id", .Desc)})
	testing.expect(t, s_err == .None, "sort alias")
	defer dataframe_destroy(&sorted)

	selected, sel_err := select(&sorted, []^expr.Expr{expr.col(&ctx, "id")})
	testing.expect(t, sel_err == .None, "select alias")
	defer dataframe_destroy(&selected)

	top, h_err := head(&selected, 2)
	testing.expect(t, h_err == .None, "head alias")
	defer dataframe_destroy(&top)

	id_col, _ := dataframe_get_column(&top, "id")
	expect_i32_col(t, "id", id_col, []i32{6, 5})
}

@(test)
chain_aliases_group_by_agg :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	gb, g_err := group_by(&df, []^expr.Expr{expr.col(&ctx, "g")})
	testing.expect(t, g_err == .None, "group_by alias")
	defer dataframe_group_by_destroy(&gb)

	out, a_err := agg(&gb, []^expr.Expr{expr.sum_(&ctx, expr.col(&ctx, "id"))})
	testing.expect(t, a_err == .None, "agg alias")
	defer dataframe_destroy(&out)

	sum_col, _ := dataframe_get_column(&out, "id")
	sv := column_typed_view(sum_col, f64)
	wants := []f64{3, 7, 11}
	for i in 0 ..< len(wants) {
		testing.expect(t, sv[i] == wants[i], "sum id value")
	}
}

@(test)
chain_aliases_unique_partition :: proc(t: ^testing.T) {
	df, ctx := ops_test_df(t)
	defer ops_test_destroy(t, &df, &ctx)

	u, u_err := unique(&df, []string{"s"})
	testing.expect(t, u_err == .None, "unique alias")
	defer dataframe_destroy(&u)
	testing.expect(t, dataframe_num_rows(&u) == 3, "unique rows")

	parts, p_err := partition_by(&df, []^expr.Expr{expr.col(&ctx, "g")})
	testing.expect(t, p_err == .None, "partition_by alias")
	defer dataframe_partitions_destroy(parts, context.allocator)
	testing.expect(t, len(parts) == 3, "partition count")

	_ = []^expr.Expr{expr.col(&ctx, "id")}
}
