package lazy

// The executor (Stage 11, DESIGN.md §7.4): runs the Stage 12 optimizer over
// the logical plan, then walks it bottom-up, compiling each node to its
// physical operator (physical.odin) and feeding each operator's output into
// the next. collect() is the only entry point.

import "core:mem"
import "../../dataframe"

// collect executes lf's plan and returns an owned DataFrame. The plan itself
// is not consumed: lf can be collected again, and destroy(lf) still frees
// the plan afterwards. Errors (missing CSV file, unknown column, type
// mismatch, malformed plan) are returned from here, never from the builders.
collect :: proc(lf: LazyFrame, allocator := context.allocator) -> (out: dataframe.DataFrame, err: dataframe.Error) {
	if lf.root == nil {
		return {}, .Invalid_Argument
	}
	optimize(lf.root, lf.plan.alloc)
	df, borrowed, exec_err := exec_node(lf.root, allocator)
	if exec_err != .None {
		return {}, exec_err
	}
	if borrowed {
		// The plan root was a Scan_DF: hand back an owned deep copy so the
		// caller never shares buffers with the scanned frame.
		return dataframe.dataframe_copy(&df, allocator)
	}
	return df, .None
}

// exec_node materializes plan into an owned DataFrame. `borrowed` is true
// only for a bare Scan_DF node, in which case out shares the scanned frame's
// buffers and must NOT be destroyed by the caller.
exec_node :: proc(plan: ^Logical_Plan, allocator: mem.Allocator) -> (out: dataframe.DataFrame, borrowed: bool, err: dataframe.Error) {
	switch n in plan^ {
	case Scan_CSV:
		df, exec_err := scan_csv_op(n, allocator)
		return df, false, exec_err
	case Scan_DF:
		df, borrowed := scan_df_op(n)
		return df, borrowed, .None
	case Filter:
		child, cb, c_err := exec_node(n.child, allocator)
		if c_err != .None {
			return {}, false, c_err
		}
		defer if !cb { dataframe.dataframe_destroy(&child) }
		df, exec_err := filter_op(&child, n, allocator)
		return df, false, exec_err
	case Projection:
		child, cb, c_err := exec_node(n.child, allocator)
		if c_err != .None {
			return {}, false, c_err
		}
		defer if !cb { dataframe.dataframe_destroy(&child) }
		df, exec_err := projection_op(&child, n, allocator)
		return df, false, exec_err
	case Sort:
		child, cb, c_err := exec_node(n.child, allocator)
		if c_err != .None {
			return {}, false, c_err
		}
		defer if !cb { dataframe.dataframe_destroy(&child) }
		df, exec_err := sort_op(&child, n, allocator)
		return df, false, exec_err
	case Group_By:
		child, cb, c_err := exec_node(n.child, allocator)
		if c_err != .None {
			return {}, false, c_err
		}
		defer if !cb { dataframe.dataframe_destroy(&child) }
		df, exec_err := group_agg_op(&child, n, allocator)
		return df, false, exec_err
	case Limit:
		child, cb, c_err := exec_node(n.child, allocator)
		if c_err != .None {
			return {}, false, c_err
		}
		defer if !cb { dataframe.dataframe_destroy(&child) }
		df, exec_err := limit_op(&child, n, allocator)
		return df, false, exec_err
	case Slice:
		child, cb, c_err := exec_node(n.child, allocator)
		if c_err != .None {
			return {}, false, c_err
		}
		defer if !cb { dataframe.dataframe_destroy(&child) }
		df, exec_err := slice_op(&child, n, allocator)
		return df, false, exec_err
	case Join:
		l, lb, l_err := exec_node(n.left, allocator)
		if l_err != .None {
			return {}, false, l_err
		}
		defer if !lb { dataframe.dataframe_destroy(&l) }
		r, rb, r_err := exec_node(n.right, allocator)
		if r_err != .None {
			return {}, false, r_err
		}
		defer if !rb { dataframe.dataframe_destroy(&r) }
		df, exec_err := join_op(&l, &r, n, allocator)
		return df, false, exec_err
	}
	return {}, false, .None
}
