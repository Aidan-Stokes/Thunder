package lazy

// Physical operators (Stage 11, DESIGN.md §7.3).
//
// In Stage 11 the executor is eager-backed (S11.3): every logical node
// compiles to exactly one eager-engine call that materializes the node's
// output. The procs below ARE the physical operators — each wraps a single
// dataframe package call. The Stage 12 optimizer (optimizer.odin) rewrites
// the logical plan before this layer runs, so scan_csv_op honors the pruned
// Scan_CSV.columns. No streaming/parallelism yet (Stage 15 swaps these calls
// for parallel kernels without touching the logical layer).
//
// Every operator returns an owned DataFrame (the caller destroys it), except
// scan_df_borrow which hands the scanned frame through untouched (borrowed:
// the caller must not destroy it).
//
// Contract: a child passed to a non-scan operator is borrowed for the
// duration of the call only — operators never retain it.

import "core:mem"
import "core:strings"
import "../../dataframe"
import "../../dataframe/expr"

// scan_csv_op materializes a Scan_CSV node. When the optimizer (S12.1) has
// pruned the scan to a column subset, only those columns are read.
scan_csv_op :: proc(n: Scan_CSV, allocator: mem.Allocator) -> (out: dataframe.DataFrame, err: dataframe.Error) {
	if len(n.columns) > 0 {
		return dataframe.dataframe_read_csv_with_columns(n.path, n.columns, n.options, allocator)
	}
	return dataframe.dataframe_read_csv(n.path, n.options, allocator)
}

// scan_df_op hands the scanned DataFrame through without copying. The result
// is borrowed; the executor treats it specially (it is only ever a child, or
// deep-copied when it is the plan root).
scan_df_op :: proc(n: Scan_DF) -> (out: dataframe.DataFrame, borrowed: bool) {
	return n.df^, true
}

// filter_op runs dataframe_filter over the materialized child.
filter_op :: proc(child: ^dataframe.DataFrame, n: Filter, allocator: mem.Allocator) -> (out: dataframe.DataFrame, err: dataframe.Error) {
	return dataframe.dataframe_filter(child, n.predicate, allocator)
}

// projection_op runs dataframe_select over the materialized child, with CSE
// (S12.5): positions whose expressions are structurally equal after stripping
// the top-level Alias share one evaluation, and the evaluated column is
// deep-copied into each output position under that position's own name.
// Evaluation is pure, so plan-before equals plan-after (S12.6). Error order is
// preserved too: positions are processed in order, and a reused position can
// only exist where the first occurrence already evaluated successfully, so
// eval failures, unnamed expressions, and duplicate output names surface at
// exactly the position eager dataframe_select reports them.
projection_op :: proc(child: ^dataframe.DataFrame, n: Projection, allocator: mem.Allocator) -> (out: dataframe.DataFrame, err: dataframe.Error) {
	classes := make([dynamic]cse_class, context.allocator)
	defer delete(classes)
	defer projection_cse_cleanup(&classes)
	position_class := make([]int, len(n.exprs), context.allocator)
	defer delete(position_class)

	for i in 0 ..< len(n.exprs) {
		canonical := unwrap_alias(n.exprs[i])
		ci := -1
		for j in 0 ..< len(classes) {
			if exprs_structurally_equal(classes[j].canonical, canonical) {
				ci = j
				break
			}
		}
		if ci == -1 {
			append(&classes, cse_class{canonical = canonical})
			ci = len(classes) - 1
		}
		classes[ci].count += 1
		position_class[i] = ci
	}

	out = dataframe.dataframe_create(allocator)
	for i in 0 ..< len(n.exprs) {
		class := &classes[position_class[i]]
		col: dataframe.Column
		if class.evaluated {
			col, err = dataframe.column_copy(&class.col, allocator)
			if err != .None {
				dataframe.dataframe_destroy(&out)
				return {}, err
			}
		} else {
			oa: dataframe.OpArena
			dataframe.op_arena_init(&oa, allocator)
			defer dataframe.op_arena_destroy(&oa)
			col, err = dataframe.expr_eval(allocator, child, class.canonical, &oa)
			if err != .None {
				dataframe.dataframe_destroy(&out)
				return {}, err
			}
			if class.count > 1 {
				cached, c_err := dataframe.column_copy(&col, allocator)
				if c_err != .None {
					dataframe.column_destroy(&col)
					dataframe.dataframe_destroy(&out)
					return {}, c_err
				}
				class.col = cached
				class.evaluated = true
			}
		}
		if r_err := column_rename(&col, expr_output_name(n.exprs[i])); r_err != .None {
			dataframe.column_destroy(&col)
			dataframe.dataframe_destroy(&out)
			return {}, r_err
		}
		if col.name == "" {
			dataframe.column_destroy(&col)
			dataframe.dataframe_destroy(&out)
			return {}, .Invalid_Argument
		}
		if a_err := dataframe.dataframe_add_column(&out, &col); a_err != .None {
			dataframe.column_destroy(&col)
			dataframe.dataframe_destroy(&out)
			return {}, a_err
		}
	}
	return out, .None
}

// cse_class tracks one canonical (top-level-alias-stripped) expression in a
// projection. The first position of the class evaluates it; when the class has
// more than one position the evaluated column is cached in col and reused.
cse_class :: struct {
	canonical: ^expr.Expr,
	count:     int,
	evaluated: bool,
	col:       dataframe.Column,
}

@(private)
projection_cse_cleanup :: proc(classes: ^[dynamic]cse_class) {
	for &c in classes {
		if c.evaluated {
			dataframe.column_destroy(&c.col)
		}
	}
}

// column_rename replaces a column's name with an owned copy in the column's
// own allocator (the same way expr_eval renames an Alias result).
@(private)
column_rename :: proc(col: ^dataframe.Column, name: string) -> dataframe.Error {
	if col.name == name {
		return .None
	}
	new_name := make([]byte, len(name), col.alloc)
	if new_name == nil && len(name) != 0 {
		return .Allocator_Failure
	}
	copy(new_name, transmute([]byte)name)
	delete_string(col.name, col.alloc)
	col.name = string(new_name)
	return .None
}

// sort_op runs dataframe_sort over the materialized child.
sort_op :: proc(child: ^dataframe.DataFrame, n: Sort, allocator: mem.Allocator) -> (out: dataframe.DataFrame, err: dataframe.Error) {
	return dataframe.dataframe_sort(child, n.keys, allocator)
}

// group_agg_op runs the two-step group_by + agg over the materialized child.
// An empty aggs list is an .Invalid_Argument error (a group_by with no
// aggregation is not a query result).
group_agg_op :: proc(child: ^dataframe.DataFrame, n: Group_By, allocator: mem.Allocator) -> (out: dataframe.DataFrame, err: dataframe.Error) {
	if len(n.aggs) == 0 {
		return {}, .Invalid_Argument
	}
	gb, g_err := dataframe.dataframe_group_by(child, n.keys, allocator)
	if g_err != .None {
		return {}, g_err
	}
	defer dataframe.dataframe_group_by_destroy(&gb)
	return dataframe.dataframe_group_by_agg(&gb, n.aggs, allocator)
}

// limit_op runs dataframe_limit over the materialized child.
limit_op :: proc(child: ^dataframe.DataFrame, n: Limit, allocator: mem.Allocator) -> (out: dataframe.DataFrame, err: dataframe.Error) {
	return dataframe.dataframe_limit(child, n.n, allocator)
}

// slice_op runs dataframe_slice over the materialized child.
slice_op :: proc(child: ^dataframe.DataFrame, n: Slice, allocator: mem.Allocator) -> (out: dataframe.DataFrame, err: dataframe.Error) {
	return dataframe.dataframe_slice(child, n.offset, n.length, allocator)
}

// join_op runs the eager join matching n.kind. Both children are materialized
// first by the executor.
join_op :: proc(left, right: ^dataframe.DataFrame, n: Join, allocator: mem.Allocator) -> (out: dataframe.DataFrame, err: dataframe.Error) {
	switch n.kind {
	case .Inner:
		return dataframe.dataframe_inner_join(left, right, n.left_keys, n.right_keys, allocator)
	case .Left:
		return dataframe.dataframe_left_join(left, right, n.left_keys, n.right_keys, allocator)
	case .Right:
		return dataframe.dataframe_right_join(left, right, n.left_keys, n.right_keys, allocator)
	case .Full:
		return dataframe.dataframe_full_join(left, right, n.left_keys, n.right_keys, allocator)
	case .Semi:
		return dataframe.dataframe_semi_join(left, right, n.left_keys, n.right_keys, allocator)
	case .Anti:
		return dataframe.dataframe_anti_join(left, right, n.left_keys, n.right_keys, allocator)
	case .Cross:
		return dataframe.dataframe_cross_join(left, right, allocator)
	}
	return {}, .Unsupported_Operation
}
