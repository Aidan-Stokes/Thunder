package dataframe

// Grouped aggregation (Stage 7, DESIGN.md §9): dataframe_group_by then
// dataframe_group_by_agg.
//
//   gb := dataframe_group_by(df, []^expr.Expr{expr.col(&ctx, "dept")})
//   defer dataframe_group_by_destroy(&gb)
//   out := dataframe_group_by_agg(&gb, []^expr.Expr{...}) or_return
//
// Grouping (S7.1): key expressions are evaluated once against df, then every
// row is hashed on a canonical byte encoding of its key values (group_keys.odin
// encode_row, the same scheme as unique/partition_by). NULL is a per-column key
// value (S7.4): rows that are NULL in the same key columns form one group.
// Groups appear in first-appearance order; each group preserves source row
// order.
//
// Aggregation (S7.2–S7.3): each agg expression is an Agg node (optionally
// wrapped in Alias). Its child is evaluated once against df and the Stage 6
// kernel is run over each group's row indices (agg.odin run_group_agg). NULL
// values inside a group are skipped by every kernel, matching the Stage 6
// NULL semantics: count/n_unique are always valid (0 when no row is valid),
// value-based aggs produce a NULL row when the group has no valid rows.
//
// Ownership: Group_By borrows df — do not destroy or restructure df between
// dataframe_group_by and dataframe_group_by_destroy. The result DataFrame owns
// its columns and is destroyed with dataframe_destroy.

import "core:mem"
import "expr"

// Group_By holds the row-index groups of a dataframe_group_by call. It borrows
// df (which must outlive it) and owns its key columns, the group map, and the
// first-appearance key order. Release with dataframe_group_by_destroy.
Group_By :: struct {
	df:       ^DataFrame,
	alloc:    mem.Allocator,
	key_cols: []Column,              // owned; one evaluated column per key expr
	key_ptrs: []^Column,             // borrowed; points into key_cols
	groups:   map[string][dynamic]int, // owned key bytes -> group row indices
	order:    [dynamic]string,       // owned key strings in first-appearance order
}

// agg_spec is a decomposed agg-list expression: the Agg node's child (to
// evaluate once over the full df), the aggregation kind, and the result column
// name.
agg_spec :: struct {
	child: ^expr.Expr, // Agg node's child expression
	kind:  expr.Agg_Kind,
	q:     f64,        // Quantile only
	name:  string,     // Alias name, else the aggregated column's name
}

// decompose_agg splits an agg-list expression into its aggregation spec. The
// root must be an Agg node, optionally wrapped in an Alias. ok is false for
// any other expression shape (Cov/Corr are not supported as grouped aggs in
// the MVP).
@(private)
decompose_agg :: proc(e: ^expr.Expr) -> (spec: agg_spec, ok: bool) {
	#partial switch n in e^ {
	case expr.Alias:
		agg, is_agg := n.expr^.(expr.Agg)
		if !is_agg {
			return {}, false
		}
		return agg_spec{child = agg.expr, kind = agg.kind, q = agg.q, name = n.name}, true
	case expr.Agg:
		spec = agg_spec{child = n.expr, kind = n.kind, q = n.q}
		if c, is_col := n.expr^.(expr.Col); is_col {
			spec.name = c.name
		}
		return spec, true
	case:
		return {}, false
	}
	return {}, false
}
// dataframe_group_by groups df by the evaluated key expressions, returning the
// row-index groups. Every key expression's result must be named (a Col keeps
// its column name, an Alias supplies one); unnamed results are an error. An
// empty exprs is an error. On success the caller must release gb with
// dataframe_group_by_destroy.
dataframe_group_by :: proc(df: ^DataFrame, exprs: []^expr.Expr, allocator := context.allocator) -> (out: Group_By, err: Error) {
	if len(exprs) == 0 {
		return {}, .Invalid_Argument
	}
	out.df = df
	out.alloc = allocator

	out.key_cols = make([]Column, len(exprs), allocator)
	if out.key_cols == nil && len(exprs) != 0 {
		return {}, .Allocator_Failure
	}
	oa: OpArena
	op_arena_init(&oa, allocator)
	defer op_arena_destroy(&oa)
	for e, i in exprs {
		c, cerr := expr_eval(allocator, df, e, &oa)
		if cerr != .None {
			dataframe_group_by_destroy(&out)
			return {}, cerr
		}
		if c.name == "" {
			column_destroy(&c)
			dataframe_group_by_destroy(&out)
			return {}, .Invalid_Argument
		}
		out.key_cols[i] = c
	}

	out.key_ptrs = make([]^Column, len(exprs), allocator)
	if out.key_ptrs == nil && len(exprs) != 0 {
		dataframe_group_by_destroy(&out)
		return {}, .Allocator_Failure
	}
	for i in 0 ..< len(exprs) {
		out.key_ptrs[i] = &out.key_cols[i]
	}

	rows := dataframe_num_rows(df)
	out.groups = make(map[string][dynamic]int, 0, allocator)
	out.order = make([dynamic]string, 0, allocator)

	// S21.1: the per-row encode+hash loop parallelizes above
	// PARALLEL_GROUPBY_THRESHOLD (partitioned hash grouping; identical
	// first-appearance/source-order semantics).
	g_err: Error
	if rows >= PARALLEL_GROUPBY_THRESHOLD {
		g_err = group_rows_parallel(&out, rows)
	} else {
		g_err = group_rows_sequential(&out, rows)
	}
	if g_err != .None {
		dataframe_group_by_destroy(&out)
		return {}, g_err
	}
	return out, .None
}

// dataframe_group_by_destroy releases the key columns, the group map, and the
// order strings of gb. The borrowed df is not touched.
dataframe_group_by_destroy :: proc(gb: ^Group_By) {
	for &c in gb.key_cols {
		column_destroy(&c)
	}
	delete(gb.key_cols, gb.alloc)
	delete(gb.key_ptrs, gb.alloc)
	for k in gb.order {
		delete_string(k, gb.alloc)
	}
	delete(gb.order)
	for _, g in gb.groups {
		delete(g)
	}
	delete(gb.groups)
	gb^ = {}
}

// dataframe_group_by_agg aggregates each group of gb, returning a new
// DataFrame with one row per group (in first-appearance order): the key
// columns followed by one result column per agg expression. An empty df (no
// groups) yields the correct 0-row schema.
//
// Naming: each agg's result is named by its Alias, or by the aggregated
// column's name when the Agg node's child is a Col (mirroring polars). An
// agg over any other expression shape must be aliased. Result names must be
// unique across keys and aggs (dataframe_add_column's rule); a collision is
// .Duplicate_Column_Name.
//
// Error cases: empty aggs (.Invalid_Argument), a non-Agg expression
// (.Unsupported_Operation), an unnamed agg result (.Invalid_Argument), a kind
// unsupported for the child dtype (.Unsupported_Operation), and any
// evaluation/typecheck error from the child expressions.
dataframe_group_by_agg :: proc(gb: ^Group_By, aggs: []^expr.Expr, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	if len(aggs) == 0 {
		return {}, .Invalid_Argument
	}
	n_groups := len(gb.order)

	specs := make([]agg_spec, len(aggs), allocator)
	if specs == nil && len(aggs) != 0 {
		return {}, .Allocator_Failure
	}
	defer delete(specs, allocator)
	for e, i in aggs {
		s, ok := decompose_agg(e)
		if !ok {
			return {}, .Unsupported_Operation
		}
		if s.name == "" {
			return {}, .Invalid_Argument
		}
		specs[i] = s
	}

	children := make([]Column, len(aggs), allocator)
	if children == nil && len(aggs) != 0 {
		return {}, .Allocator_Failure
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
		c, cerr := expr_eval(allocator, gb.df, s.child, &oa)
		if cerr != .None {
			return {}, cerr
		}
		children[i] = c
	}

	out = dataframe_create(allocator)

	// Key columns: one row per group. The first row of each group carries the
	// group's key values and NULL flags.
	first_rows := make([]int, n_groups, allocator)
	if first_rows == nil && n_groups != 0 {
		dataframe_destroy(&out)
		return {}, .Allocator_Failure
	}
	for gi in 0 ..< n_groups {
		rows := gb.groups[gb.order[gi]]
		first_rows[gi] = rows[0]
	}
	for &kc in gb.key_cols {
		key_col, kerr := gather_rows(&kc, allocator, first_rows)
		if kerr != .None {
			delete(first_rows, allocator)
			dataframe_destroy(&out)
			return {}, kerr
		}
		if a_err := dataframe_add_column(&out, &key_col); a_err != .None {
			column_destroy(&key_col)
			delete(first_rows, allocator)
			dataframe_destroy(&out)
			return {}, a_err
		}
	}
	delete(first_rows, allocator)

	// Aggregation result columns.
	for s, i in specs {
		child := &children[i]
		if v_err := validate_agg(s.kind, child.dtype); v_err != .None {
			dataframe_destroy(&out)
			return {}, v_err
		}
		res_dtype, ok := agg_result_dtype(s.kind, child.dtype)
		if !ok {
			dataframe_destroy(&out)
			return {}, .Invalid_Argument
		}
		col, aerr := column_alloc(allocator, s.name, res_dtype, size_of_ty(res_dtype), align_of_ty(res_dtype), n_groups)
		if aerr != .None {
			dataframe_destroy(&out)
			return {}, aerr
		}
		for gi in 0 ..< n_groups {
			rows := gb.groups[gb.order[gi]]
			if rerr := run_group_agg(allocator, child, s.kind, s.q, &col, rows[:], gi); rerr != .None {
				column_destroy(&col)
				dataframe_destroy(&out)
				return {}, rerr
			}
		}
		if a_err := dataframe_add_column(&out, &col); a_err != .None {
			column_destroy(&col)
			dataframe_destroy(&out)
			return {}, a_err
		}
	}
	return out, .None
}
