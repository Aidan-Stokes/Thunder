package lazy

// Stage 12 optimizer (DESIGN.md §7.4): rewrites the Logical_Plan before the
// executor runs. The rewrite is column-count-preserving and equivalence-
// preserving on valid plans (S12.6); invalid plans still surface the same
// errors unless the offending code was dead after the rewrite (documented
// below for each rule). optimize runs three passes over the tree, in order:
//
//   S12.4 constant folding  — literal-only sub-expressions collapse to a Lit.
//   S12.2 predicate pushdown — filters move down past operators they commute
//                              with.
//   S12.3 column pruning     — scans read only referenced columns (S12.1) and
//                              intermediate projections keep only referenced
//                              outputs.
//
// It runs inside collect() and is idempotent: collecting the same LazyFrame
// repeatedly re-optimizes the same tree with the same result. Rewritten nodes
// are allocated from alloc (the plan arena), which the LazyFrame frees
// wholesale; the optimizer never mutates borrowed expr nodes.

import "core:mem"
import "../../dataframe"
import "../../dataframe/expr"

// column_set is the running set of column names a scan must provide. all
// marks "every column" (no pruning); names are kept in first-seen order so
// pruned scans read deterministic, caller-visible columns.
column_set :: struct {
	all:   bool,
	names: [dynamic]string,
}

// optimize rewrites root in place. Allocations come from alloc (the plan
// arena), which the LazyFrame frees wholesale.
optimize :: proc(root: ^Logical_Plan, alloc: mem.Allocator) {
	// A 1-row frame gives literal evaluation a row count for S12.4 without
	// touching any plan data.
	temp := dataframe.dataframe_create(context.allocator)
	defer dataframe.dataframe_destroy(&temp)
	dummy, _ := dataframe.column_from("__fold_dummy", []bool{true})
	dataframe.dataframe_add_column(&temp, &dummy)

	fold_constants(root, &temp, alloc)
	push_predicates(root, alloc)

	req := column_set{all = true}
	defer delete(req.names)
	prune_node(root, &req, false, alloc)
}

// ===========================================================================
// S12.4 constant folding
// ===========================================================================
//
// Only elementwise, position-independent nodes fold: Lit, Binary, Unary, Cast,
// Not_Null, Func (Abs/Sign/Round), Is_Between, Is_In. Row-shaping nodes
// (Arange, Diff/Pct_Change/Cum_Sum, Arg_Where, Agg, Cov, Corr, Dot_Product,
// Concat_Str, Search_Sorted, Distinct) never fold — their row counts or NULL
// patterns depend on the context — but their children still fold. Integer
// division/modulo by zero produces an all-NULL column, which a scalar Lit
// cannot represent, so those subtrees stay unfolded (collect yields the
// identical all-NULL result). String-valued results are never folded (a folded
// Lit would need to own a payload; input strings stay borrowed either way).

@(private)
fold_constants :: proc(plan: ^Logical_Plan, temp: ^dataframe.DataFrame, alloc: mem.Allocator) {
	switch &n in plan^ {
	case Scan_CSV, Scan_DF:
	case Filter:
		n.predicate = fold_expr(n.predicate, temp, alloc)
		fold_constants(n.child, temp, alloc)
	case Projection:
		for i in 0 ..< len(n.exprs) {
			n.exprs[i] = fold_expr(n.exprs[i], temp, alloc)
		}
		fold_constants(n.child, temp, alloc)
	case Sort:
		fold_constants(n.child, temp, alloc)
	case Group_By:
		for i in 0 ..< len(n.keys) {
			n.keys[i] = fold_expr(n.keys[i], temp, alloc)
		}
		for i in 0 ..< len(n.aggs) {
			n.aggs[i] = fold_expr(n.aggs[i], temp, alloc)
		}
		fold_constants(n.child, temp, alloc)
	case Limit:
		fold_constants(n.child, temp, alloc)
	case Slice:
		fold_constants(n.child, temp, alloc)
	case Join:
		fold_constants(n.left, temp, alloc)
		fold_constants(n.right, temp, alloc)
	}
}

// fold_expr folds the constant parts of e. Returns e itself when nothing
// changed, a new arena node otherwise. The returned tree is never mutated.
@(private)
fold_expr :: proc(e: ^expr.Expr, temp: ^dataframe.DataFrame, alloc: mem.Allocator) -> ^expr.Expr {
	if e == nil {
		return e
	}
	switch v in e^ {
	case expr.Lit, expr.Col:
		return e
	case expr.Binary:
		lhs := fold_expr(v.lhs, temp, alloc)
		rhs := fold_expr(v.rhs, temp, alloc)
		_, l_is_lit := lhs^.(expr.Lit)
		_, r_is_lit := rhs^.(expr.Lit)
		if l_is_lit && r_is_lit {
			mini := expr.Expr(expr.Binary{op = v.op, lhs = lhs, rhs = rhs})
			if folded, ok := fold_value(&mini, temp, alloc); ok {
				return folded
			}
		}
		if lhs != v.lhs || rhs != v.rhs {
			return plan_expr(alloc, expr.Binary{op = v.op, lhs = lhs, rhs = rhs})
		}
		return e
	case expr.Unary:
		child := fold_expr(v.expr, temp, alloc)
		if _, c_is_lit := child^.(expr.Lit); c_is_lit {
			mini := expr.Expr(expr.Unary{op = v.op, expr = child})
			if folded, ok := fold_value(&mini, temp, alloc); ok {
				return folded
			}
		}
		if child != v.expr {
			return plan_expr(alloc, expr.Unary{op = v.op, expr = child})
		}
		return e
	case expr.Cast:
		child := fold_expr(v.expr, temp, alloc)
		if _, c_is_lit := child^.(expr.Lit); c_is_lit {
			mini := expr.Expr(expr.Cast{expr = child, to = v.to})
			if folded, ok := fold_value(&mini, temp, alloc); ok {
				return folded
			}
		}
		if child != v.expr {
			return plan_expr(alloc, expr.Cast{expr = child, to = v.to})
		}
		return e
	case expr.Not_Null:
		child := fold_expr(v.expr, temp, alloc)
		if _, c_is_lit := child^.(expr.Lit); c_is_lit {
			return new_lit(alloc, true)
		}
		if child != v.expr {
			return plan_expr(alloc, expr.Not_Null{expr = child})
		}
		return e
	case expr.Is_Nan:
		child := fold_expr(v.expr, temp, alloc)
		if _, c_is_lit := child^.(expr.Lit); c_is_lit {
			mini := expr.Expr(expr.Is_Nan{expr = child})
			if folded, ok := fold_value(&mini, temp, alloc); ok {
				return folded
			}
		}
		if child != v.expr {
			return plan_expr(alloc, expr.Is_Nan{expr = child})
		}
		return e
	case expr.Fill_Null:
		child := fold_expr(v.expr, temp, alloc)
		value := fold_expr(v.value, temp, alloc)
		if child != v.expr || value != v.value {
			return plan_expr(alloc, expr.Fill_Null{expr = child, value = value})
		}
		return e
	case expr.Coalesce:
		changed := false
		parts := make([dynamic]^expr.Expr, 0, len(v.exprs), context.allocator)
		defer delete(parts)
		for p in v.exprs {
			f := fold_expr(p, temp, alloc)
			append(&parts, f)
			if f != p {
				changed = true
			}
		}
		if changed {
			owned := make([]^expr.Expr, len(parts), alloc)
			copy_slice(owned, parts[:])
			return plan_expr(alloc, expr.Coalesce{exprs = owned})
		}
		return e
	case expr.Forward_Fill:
		child := fold_expr(v.expr, temp, alloc)
		if child != v.expr {
			return plan_expr(alloc, expr.Forward_Fill{expr = child})
		}
		return e
	case expr.Backward_Fill:
		child := fold_expr(v.expr, temp, alloc)
		if child != v.expr {
			return plan_expr(alloc, expr.Backward_Fill{expr = child})
		}
		return e
	case expr.Interpolate:
		child := fold_expr(v.expr, temp, alloc)
		if child != v.expr {
			return plan_expr(alloc, expr.Interpolate{expr = child})
		}
		return e
	case expr.Func:
		child := fold_expr(v.expr, temp, alloc)
		if _, c_is_lit := child^.(expr.Lit); c_is_lit {
			#partial switch v.kind {
			case .Abs, .Sign, .Round:
				mini := expr.Expr(expr.Func{kind = v.kind, expr = child, decimals = v.decimals, n = v.n})
				if folded, ok := fold_value(&mini, temp, alloc); ok {
					return folded
				}
			case:
			}
		}
		if child != v.expr {
			return plan_expr(alloc, expr.Func{kind = v.kind, expr = child, decimals = v.decimals, n = v.n})
		}
		return e
	case expr.Is_Between:
		child := fold_expr(v.expr, temp, alloc)
		lo := fold_expr(v.lower, temp, alloc)
		hi := fold_expr(v.upper, temp, alloc)
		_, c_lit := child^.(expr.Lit)
		_, lo_lit := lo^.(expr.Lit)
		_, hi_lit := hi^.(expr.Lit)
		if c_lit && lo_lit && hi_lit {
			mini := expr.Expr(expr.Is_Between{expr = child, lower = lo, upper = hi})
			if folded, ok := fold_value(&mini, temp, alloc); ok {
				return folded
			}
		}
		if child != v.expr || lo != v.lower || hi != v.upper {
			return plan_expr(alloc, expr.Is_Between{expr = child, lower = lo, upper = hi})
		}
		return e
	case expr.Is_In:
		child := fold_expr(v.expr, temp, alloc)
		values := fold_expr(v.values, temp, alloc)
		_, c_lit := child^.(expr.Lit)
		_, v_lit := values^.(expr.Lit)
		if c_lit && v_lit {
			mini := expr.Expr(expr.Is_In{expr = child, values = values})
			if folded, ok := fold_value(&mini, temp, alloc); ok {
				return folded
			}
		}
		if child != v.expr || values != v.values {
			return plan_expr(alloc, expr.Is_In{expr = child, values = values})
		}
		return e
	case expr.Alias:
		child := fold_expr(v.expr, temp, alloc)
		if child != v.expr {
			return plan_expr(alloc, expr.Alias{expr = child, name = v.name})
		}
		return e
	case expr.Agg:
		child := fold_expr(v.expr, temp, alloc)
		if child != v.expr {
			return plan_expr(alloc, expr.Agg{kind = v.kind, expr = child, q = v.q})
		}
		return e
	case expr.Cov:
		lhs := fold_expr(v.lhs, temp, alloc)
		rhs := fold_expr(v.rhs, temp, alloc)
		if lhs != v.lhs || rhs != v.rhs {
			return plan_expr(alloc, expr.Cov{lhs = lhs, rhs = rhs})
		}
		return e
	case expr.Corr:
		lhs := fold_expr(v.lhs, temp, alloc)
		rhs := fold_expr(v.rhs, temp, alloc)
		if lhs != v.lhs || rhs != v.rhs {
			return plan_expr(alloc, expr.Corr{lhs = lhs, rhs = rhs})
		}
		return e
	case expr.Dot_Product:
		lhs := fold_expr(v.lhs, temp, alloc)
		rhs := fold_expr(v.rhs, temp, alloc)
		if lhs != v.lhs || rhs != v.rhs {
			return plan_expr(alloc, expr.Dot_Product{lhs = lhs, rhs = rhs})
		}
		return e
	case expr.Concat_Str:
		changed := false
		parts := make([dynamic]^expr.Expr, 0, len(v.exprs), context.allocator)
		defer delete(parts)
		for p in v.exprs {
			f := fold_expr(p, temp, alloc)
			append(&parts, f)
			if f != p {
				changed = true
			}
		}
		if changed {
			owned := make([]^expr.Expr, len(parts), alloc)
			copy_slice(owned, parts[:])
			return plan_expr(alloc, expr.Concat_Str{exprs = owned, separator = v.separator})
		}
		return e
	case expr.Arange:
		start := fold_expr(v.start, temp, alloc)
		end := fold_expr(v.end, temp, alloc)
		if start != v.start || end != v.end {
			return plan_expr(alloc, expr.Arange{start = start, end = end})
		}
		return e
	case expr.Arg_Where:
		child := fold_expr(v.expr, temp, alloc)
		if child != v.expr {
			return plan_expr(alloc, expr.Arg_Where{expr = child})
		}
		return e
	case expr.Distinct:
		child := fold_expr(v.expr, temp, alloc)
		if child != v.expr {
			return plan_expr(alloc, expr.Distinct{kind = v.kind, expr = child})
		}
		return e
	case expr.Search_Sorted:
		sorted := fold_expr(v.sorted, temp, alloc)
		values := fold_expr(v.values, temp, alloc)
		if sorted != v.sorted || values != v.values {
			return plan_expr(alloc, expr.Search_Sorted{sorted = sorted, values = values})
		}
		return e
	case expr.Window:
		return e // window computations are order-dependent; never fold
	}
	return e
}

// fold_value evaluates a constant sub-expression (built on the stack from
// already-folded children) against the 1-row temp frame and returns a folded
// Lit when the result is a single valid scalar. String results never fold.
@(private)
fold_value :: proc(e: ^expr.Expr, temp: ^dataframe.DataFrame, alloc: mem.Allocator) -> (^expr.Expr, bool) {
	oa: dataframe.OpArena
	dataframe.op_arena_init(&oa, context.allocator)
	defer dataframe.op_arena_destroy(&oa)
	col, err := dataframe.expr_eval(context.allocator, temp, e, &oa)
	if err != .None {
		return nil, false
	}
	defer dataframe.column_destroy(&col)
	return lit_from_column(&col, alloc)
}

// lit_from_column extracts row 0 of a 1-row column as a Lit node.
@(private)
lit_from_column :: proc(col: ^dataframe.Column, alloc: mem.Allocator) -> (^expr.Expr, bool) {
	if dataframe.column_len(col) != 1 {
		return nil, false
	}
	switch dataframe.column_dtype(col) {
	case typeid_of(i8):   return lit_from_typed(col, alloc, i8)
	case typeid_of(i16):  return lit_from_typed(col, alloc, i16)
	case typeid_of(i32):  return lit_from_typed(col, alloc, i32)
	case typeid_of(i64):  return lit_from_typed(col, alloc, i64)
	case typeid_of(u8):   return lit_from_typed(col, alloc, u8)
	case typeid_of(u16):  return lit_from_typed(col, alloc, u16)
	case typeid_of(u32):  return lit_from_typed(col, alloc, u32)
	case typeid_of(u64):  return lit_from_typed(col, alloc, u64)
	case typeid_of(int):  return lit_from_typed(col, alloc, int)
	case typeid_of(uint): return lit_from_typed(col, alloc, uint)
	case typeid_of(f32):  return lit_from_typed(col, alloc, f32)
	case typeid_of(f64):  return lit_from_typed(col, alloc, f64)
	case typeid_of(bool): return lit_from_typed(col, alloc, bool)
	}
	return nil, false
}

@(private)
lit_from_typed :: proc(col: ^dataframe.Column, alloc: mem.Allocator, $T: typeid) -> (^expr.Expr, bool) {
	v, valid, err := dataframe.column_get(col, 0, T)
	if err != .None || !valid {
		return nil, false
	}
	return new_lit(alloc, v), true
}

// new_lit builds a Lit node holding a scalar value in the plan arena.
@(private)
new_lit :: proc(alloc: mem.Allocator, value: $T) -> ^expr.Expr {
	l := expr.Lit{dtype = typeid_of(T)}
	v := value
	mem.copy(&l.data, &v, size_of(T))
	return plan_expr(alloc, l)
}

// ===========================================================================
// S12.2 predicate pushdown
// ===========================================================================
//
// A Filter node moves down the plan when it commutes with what it passes:
//   - through Sort (filter and sort commute),
//   - through a sibling Filter (merged with `and`),
//   - through a Projection when every column the predicate names is produced
//     by that projection (the names are rewritten to the projection's defining
//     expressions; a predicate referencing any other name is left in place so
//     it still fails identically at collect),
//   - into the left side of a left-major join (Inner/Semi/Anti/Cross) when
//     the predicate names only columns the left side provably produces — the
//     join output binds those names to the left columns. Right/Full joins and
//     outer-left rows cannot be pushed (unmatched right rows would survive),
//     and the side's columns must be statically known (Scan_DF, a pruned
//     Scan_CSV, or a pass-through/projection/group-by over them).
// Filters never cross Limit/Slice/Group_By (their row counts change).

@(private)
push_predicates :: proc(plan: ^Logical_Plan, alloc: mem.Allocator) {
	switch &n in plan^ {
	case Scan_CSV, Scan_DF:
	case Filter:
		push_filter(&n.child, n.predicate, alloc)
		push_predicates(n.child, alloc)
	case Projection:
		push_predicates(n.child, alloc)
	case Sort:
		push_predicates(n.child, alloc)
	case Group_By:
		push_predicates(n.child, alloc)
	case Limit:
		push_predicates(n.child, alloc)
	case Slice:
		push_predicates(n.child, alloc)
	case Join:
		push_predicates(n.left, alloc)
		push_predicates(n.right, alloc)
	}
}

// push_filter moves the filter at child^'s parent as far down as the plan
// allows, rebuilding nodes in the arena as it goes. pred is borrowed.
@(private)
push_filter :: proc(child_ptr: ^^Logical_Plan, pred: ^expr.Expr, alloc: mem.Allocator) {
	child := child_ptr
	to_push := pred
	for {
		#partial switch &c in child^ {
		case Sort:
			inner := plan_new(alloc, Filter{child = c.child, predicate = to_push})
			child^ = plan_new(alloc, Sort{child = inner, keys = c.keys})
			child = node_child_ptr(inner)
		case Filter:
			merged := plan_expr(alloc, expr.Binary{op = .And, lhs = to_push, rhs = c.predicate})
			inner := plan_new(alloc, Filter{child = c.child, predicate = merged})
			child^ = inner
			child = node_child_ptr(inner)
			to_push = merged
		case Projection:
			rewritten, ok := rewrite_through_projection(to_push, c.exprs, alloc)
			if !ok {
				return
			}
			inner := plan_new(alloc, Filter{child = c.child, predicate = rewritten})
			child^ = plan_new(alloc, Projection{child = inner, exprs = c.exprs})
			child = node_child_ptr(inner)
			to_push = rewritten
		case Join:
			if !join_pushable(c, to_push) {
				return
			}
			inner := plan_new(alloc, Filter{child = c.left, predicate = to_push})
			child^ = plan_new(alloc, Join{
				left = inner, right = c.right, kind = c.kind,
				left_keys = c.left_keys, right_keys = c.right_keys,
			})
			child = node_child_ptr(inner)
		case Scan_CSV, Scan_DF, Limit, Slice, Group_By:
			return
		}
	}
}

// node_child_ptr returns the mutable child pointer of a single-child node
// (for Join, the left child — filters never push into a right side).
@(private)
node_child_ptr :: proc(p: ^Logical_Plan) -> ^^Logical_Plan {
	#partial switch &n in p^ {
	case Filter:     return &n.child
	case Projection: return &n.child
	case Sort:       return &n.child
	case Group_By:   return &n.child
	case Limit:      return &n.child
	case Slice:      return &n.child
	case Join:       return &n.left
	}
	return nil
}

// rewrite_through_projection rewrites a predicate's column references from the
// projection's output names back to the projection's defining expressions, so
// the predicate can run against the projection's child. Fails (returns ok =
// false) when any referenced name is not produced by the projection — the
// filter then stays above it and still errors at collect exactly as before.
@(private)
rewrite_through_projection :: proc(pred: ^expr.Expr, exprs: []^expr.Expr, alloc: mem.Allocator) -> (^expr.Expr, bool) {
	return rewrite_expr(pred, exprs, alloc)
}

@(private)
rewrite_expr :: proc(e: ^expr.Expr, exprs: []^expr.Expr, alloc: mem.Allocator) -> (out: ^expr.Expr, ok: bool) {
	#partial switch v in e^ {
	case expr.Col:
		for ex in exprs {
			if expr_output_name(ex) == v.name {
				return unwrap_alias(ex), true
			}
		}
		return nil, false
	case expr.Lit:
		return e, true
	case expr.Binary:
		lhs, lok := rewrite_expr(v.lhs, exprs, alloc)
		if !lok {
			return nil, false
		}
		rhs, rok := rewrite_expr(v.rhs, exprs, alloc)
		if !rok {
			return nil, false
		}
		if lhs == v.lhs && rhs == v.rhs {
			return e, true
		}
		return plan_expr(alloc, expr.Binary{op = v.op, lhs = lhs, rhs = rhs}), true
	case expr.Unary:
		child, cok := rewrite_expr(v.expr, exprs, alloc)
		if !cok {
			return nil, false
		}
		if child == v.expr {
			return e, true
		}
		return plan_expr(alloc, expr.Unary{op = v.op, expr = child}), true
	case expr.Cast:
		child, cok := rewrite_expr(v.expr, exprs, alloc)
		if !cok {
			return nil, false
		}
		if child == v.expr {
			return e, true
		}
		return plan_expr(alloc, expr.Cast{expr = child, to = v.to}), true
	case expr.Not_Null:
		child, cok := rewrite_expr(v.expr, exprs, alloc)
		if !cok {
			return nil, false
		}
		if child == v.expr {
			return e, true
		}
		return plan_expr(alloc, expr.Not_Null{expr = child}), true
	case expr.Is_Nan:
		child, cok := rewrite_expr(v.expr, exprs, alloc)
		if !cok {
			return nil, false
		}
		if child == v.expr {
			return e, true
		}
		return plan_expr(alloc, expr.Is_Nan{expr = child}), true
	case expr.Fill_Null:
		child, c1 := rewrite_expr(v.expr, exprs, alloc)
		value, c2 := rewrite_expr(v.value, exprs, alloc)
		if !c1 || !c2 {
			return nil, false
		}
		if child == v.expr && value == v.value {
			return e, true
		}
		return plan_expr(alloc, expr.Fill_Null{expr = child, value = value}), true
	case expr.Coalesce:
		changed := false
		parts := make([dynamic]^expr.Expr, 0, len(v.exprs), context.allocator)
		defer delete(parts)
		for p in v.exprs {
			f, ok := rewrite_expr(p, exprs, alloc)
			if !ok {
				return nil, false
			}
			append(&parts, f)
			if f != p {
				changed = true
			}
		}
		if !changed {
			return e, true
		}
		owned := make([]^expr.Expr, len(parts), alloc)
		copy_slice(owned, parts[:])
		return plan_expr(alloc, expr.Coalesce{exprs = owned}), true
	case expr.Forward_Fill:
		child, cok := rewrite_expr(v.expr, exprs, alloc)
		if !cok {
			return nil, false
		}
		if child == v.expr {
			return e, true
		}
		return plan_expr(alloc, expr.Forward_Fill{expr = child}), true
	case expr.Backward_Fill:
		child, cok := rewrite_expr(v.expr, exprs, alloc)
		if !cok {
			return nil, false
		}
		if child == v.expr {
			return e, true
		}
		return plan_expr(alloc, expr.Backward_Fill{expr = child}), true
	case expr.Interpolate:
		child, cok := rewrite_expr(v.expr, exprs, alloc)
		if !cok {
			return nil, false
		}
		if child == v.expr {
			return e, true
		}
		return plan_expr(alloc, expr.Interpolate{expr = child}), true
	case expr.Func:
		child, cok := rewrite_expr(v.expr, exprs, alloc)
		if !cok {
			return nil, false
		}
		if child == v.expr {
			return e, true
		}
		return plan_expr(alloc, expr.Func{kind = v.kind, expr = child, decimals = v.decimals, n = v.n}), true
	case expr.Distinct:
		child, cok := rewrite_expr(v.expr, exprs, alloc)
		if !cok {
			return nil, false
		}
		if child == v.expr {
			return e, true
		}
		return plan_expr(alloc, expr.Distinct{kind = v.kind, expr = child}), true
	case expr.Is_Between:
		child, c1 := rewrite_expr(v.expr, exprs, alloc)
		lo, c2 := rewrite_expr(v.lower, exprs, alloc)
		hi, c3 := rewrite_expr(v.upper, exprs, alloc)
		if !c1 || !c2 || !c3 {
			return nil, false
		}
		if child == v.expr && lo == v.lower && hi == v.upper {
			return e, true
		}
		return plan_expr(alloc, expr.Is_Between{expr = child, lower = lo, upper = hi}), true
	case expr.Is_In:
		child, c1 := rewrite_expr(v.expr, exprs, alloc)
		values, c2 := rewrite_expr(v.values, exprs, alloc)
		if !c1 || !c2 {
			return nil, false
		}
		if child == v.expr && values == v.values {
			return e, true
		}
		return plan_expr(alloc, expr.Is_In{expr = child, values = values}), true
	case expr.Alias:
		child, cok := rewrite_expr(v.expr, exprs, alloc)
		if !cok {
			return nil, false
		}
		if child == v.expr {
			return e, true
		}
		return plan_expr(alloc, expr.Alias{expr = child, name = v.name}), true
	case:
		// row-shaping or scalar-reducing nodes in a predicate are never pushed
		return nil, false
	}
}

// join_pushable reports whether pred can be pushed into the join's left side:
// the join must be left-major and non-outer (Inner/Semi/Anti/Cross), and every
// column pred references must be a column the left side provably produces.
// When the left side's columns are not statically known (e.g. an unpruned CSV
// scan) the filter stays put.
@(private)
join_pushable :: proc(j: Join, pred: ^expr.Expr) -> bool {
	#partial switch j.kind {
	case .Inner, .Semi, .Anti, .Cross:
	case:
		return false
	}
	left_names := make([dynamic]string, context.allocator)
	defer delete(left_names)
	if !plan_known_columns(j.left, &left_names) {
		return false
	}
	return pred_columns_known(pred, &left_names)
}

// pred_columns_known reports whether every column reference in a predicate is
// present in known.
@(private)
pred_columns_known :: proc(e: ^expr.Expr, known: ^[dynamic]string) -> bool {
	#partial switch v in e^ {
	case expr.Col:
		for name in known {
			if name == v.name {
				return true
			}
		}
		return false
	case expr.Lit:
		return true
	case expr.Binary:
		return pred_columns_known(v.lhs, known) && pred_columns_known(v.rhs, known)
	case expr.Unary:
		return pred_columns_known(v.expr, known)
	case expr.Cast:
		return pred_columns_known(v.expr, known)
	case expr.Not_Null:
		return pred_columns_known(v.expr, known)
	case expr.Is_Nan:
		return pred_columns_known(v.expr, known)
	case expr.Fill_Null:
		return pred_columns_known(v.expr, known) && pred_columns_known(v.value, known)
	case expr.Coalesce:
		for p in v.exprs {
			if !pred_columns_known(p, known) {
				return false
			}
		}
		return true
	case expr.Forward_Fill:
		return pred_columns_known(v.expr, known)
	case expr.Backward_Fill:
		return pred_columns_known(v.expr, known)
	case expr.Interpolate:
		return pred_columns_known(v.expr, known)
	case expr.Func:
		return pred_columns_known(v.expr, known)
	case expr.Distinct:
		return pred_columns_known(v.expr, known)
	case expr.Is_Between:
		return pred_columns_known(v.expr, known) && pred_columns_known(v.lower, known) && pred_columns_known(v.upper, known)
	case expr.Is_In:
		return pred_columns_known(v.expr, known) && pred_columns_known(v.values, known)
	case expr.Alias:
		return pred_columns_known(v.expr, known)
	case:
		return false
	}
}

// plan_known_columns appends the column names a plan produces to names and
// reports whether the set is statically known. Projection/Group_By output
// names are structural; a Scan_DF exposes its frame's columns; a Scan_CSV is
// only known once S12.1 has pruned it to an explicit column list. Joins are
// not analyzed (output naming depends on schema resolution), so pushdown
// stops at a nested join.
@(private)
plan_known_columns :: proc(plan: ^Logical_Plan, names: ^[dynamic]string) -> bool {
	switch n in plan^ {
	case Scan_CSV:
		if len(n.columns) == 0 {
			return false
		}
		for c in n.columns {
			append(names, c)
		}
		return true
	case Scan_DF:
		for i in 0 ..< n.df.columns.count {
			append(names, dataframe.cs_name(&n.df.columns, i))
		}
		return true
	case Filter:
		return plan_known_columns(n.child, names)
	case Sort:
		return plan_known_columns(n.child, names)
	case Limit:
		return plan_known_columns(n.child, names)
	case Slice:
		return plan_known_columns(n.child, names)
	case Projection:
		for e in n.exprs {
			append(names, expr_output_name(e))
		}
		return true
	case Group_By:
		for e in n.keys {
			append(names, expr_output_name(e))
		}
		for e in n.aggs {
			append(names, expr_output_name(e))
		}
		return true
	case Join:
		return false
	}
	return false
}

// expr_output_name is the name a projection/group-by result column takes: the
// Alias target, else the source column name, else "" (unnamed).
@(private)
expr_output_name :: proc(e: ^expr.Expr) -> string {
	#partial switch v in e^ {
	case expr.Alias:
		return v.name
	case expr.Col:
		return v.name
	}
	return ""
}

// unwrap_alias strips the top-level Alias of an expression.
@(private)
unwrap_alias :: proc(e: ^expr.Expr) -> ^expr.Expr {
	#partial switch v in e^ {
	case expr.Alias:
		return v.expr
	}
	return e
}

// ===========================================================================
// S12.3 column pruning (and S12.1 scan pruning)
// ===========================================================================
//
// The required-column set flows top-down. `redefines` marks a requirement that
// was built by a column-redefining ancestor (Projection/Group_By): only those
// requirements may prune an intermediate Projection, because a pass-through
// ancestor (Filter/Sort/Limit/Slice) exposes the projection's full output as
// the query result. The root never prunes. A projection whose outputs are
// invalid as a select (unnamed or duplicate) is never pruned, so an invalid
// plan still fails at collect exactly as eager.

@(private)
prune_node :: proc(plan: ^Logical_Plan, req: ^column_set, redefines: bool, alloc: mem.Allocator) {
	switch &n in plan^ {
	case Scan_CSV:
		if req.all || len(req.names) == 0 {
			n.columns = nil
		} else {
			cols := make([]string, len(req.names), alloc)
			mem.copy(raw_data(cols), raw_data(req.names), len(req.names) * size_of(string))
			n.columns = cols
		}
	case Scan_DF:
		// In-memory frame: nothing to prune.
	case Filter:
		add_expr(req, n.predicate)
		prune_node(n.child, req, redefines, alloc)
	case Projection:
		if redefines && exprs_prunable(n.exprs) {
			n.exprs = prune_exprs(n.exprs, req, alloc)
		}
		child_req := column_set{}
		defer delete(child_req.names)
		add_exprs(&child_req, n.exprs)
		prune_node(n.child, &child_req, true, alloc)
	case Sort:
		for k in n.keys {
			add_name(req, k.name)
		}
		prune_node(n.child, req, redefines, alloc)
	case Group_By:
		child_req := column_set{}
		defer delete(child_req.names)
		add_exprs(&child_req, n.keys)
		add_exprs(&child_req, n.aggs)
		prune_node(n.child, &child_req, true, alloc)
	case Limit:
		prune_node(n.child, req, redefines, alloc)
	case Slice:
		prune_node(n.child, req, redefines, alloc)
	case Join:
		// Barrier: neither side is pruned (see header).
		all := column_set{all = true}
		defer delete(all.names)
		prune_node(n.left, &all, false, alloc)
		prune_node(n.right, &all, false, alloc)
	}
}

// exprs_prunable reports whether a projection is a valid standalone select
// (every output named, no duplicates). Invalid projections keep every expr so
// collect fails identically to eager.
@(private)
exprs_prunable :: proc(exprs: []^expr.Expr) -> bool {
	for i in 0 ..< len(exprs) {
		name := expr_output_name(exprs[i])
		if name == "" {
			return false
		}
		for j in i + 1 ..< len(exprs) {
			if expr_output_name(exprs[j]) == name {
				return false
			}
		}
	}
	return true
}

// prune_exprs keeps the projection expressions whose output names are in req,
// reordered to req's first-seen order. Names in req that no expression
// produces are skipped (an ancestor referencing them still errors at collect).
@(private)
prune_exprs :: proc(exprs: []^expr.Expr, req: ^column_set, alloc: mem.Allocator) -> []^expr.Expr {
	if req.all {
		return exprs
	}
	out := make([dynamic]^expr.Expr, 0, len(req.names), alloc)
	for name in req.names {
		for e in exprs {
			if expr_output_name(e) == name {
				append(&out, e)
				break
			}
		}
	}
	return out[:]
}

@(private)
add_name :: proc(req: ^column_set, name: string) {
	if req.all {
		return
	}
	for n in req.names {
		if n == name {
			return
		}
	}
	append(&req.names, name)
}

@(private)
add_exprs :: proc(req: ^column_set, exprs: []^expr.Expr) {
	for e in exprs {
		add_expr(req, e)
	}
}

// add_expr collects every column reference inside an expression tree into
// req. Expression output names (Alias targets, literal results) are never
// scan columns and are skipped.
@(private)
add_expr :: proc(req: ^column_set, e: ^expr.Expr) {
	if e == nil {
		return
	}
	switch v in e^ {
	case expr.Col:
		add_name(req, v.name)
	case expr.Lit:
	case expr.Binary:
		add_expr(req, v.lhs)
		add_expr(req, v.rhs)
	case expr.Unary:
		add_expr(req, v.expr)
	case expr.Cast:
		add_expr(req, v.expr)
	case expr.Alias:
		add_expr(req, v.expr)
	case expr.Not_Null:
		add_expr(req, v.expr)
	case expr.Is_Nan:
		add_expr(req, v.expr)
	case expr.Fill_Null:
		add_expr(req, v.expr)
		add_expr(req, v.value)
	case expr.Coalesce:
		for c in v.exprs {
			add_expr(req, c)
		}
	case expr.Forward_Fill:
		add_expr(req, v.expr)
	case expr.Backward_Fill:
		add_expr(req, v.expr)
	case expr.Interpolate:
		add_expr(req, v.expr)
	case expr.Func:
		add_expr(req, v.expr)
	case expr.Is_Between:
		add_expr(req, v.expr)
		add_expr(req, v.lower)
		add_expr(req, v.upper)
	case expr.Is_In:
		add_expr(req, v.expr)
		add_expr(req, v.values)
	case expr.Arange:
		add_expr(req, v.start)
		add_expr(req, v.end)
	case expr.Arg_Where:
		add_expr(req, v.expr)
	case expr.Distinct:
		add_expr(req, v.expr)
	case expr.Dot_Product:
		add_expr(req, v.lhs)
		add_expr(req, v.rhs)
	case expr.Concat_Str:
		for c in v.exprs {
			add_expr(req, c)
		}
	case expr.Search_Sorted:
		add_expr(req, v.sorted)
		add_expr(req, v.values)
	case expr.Agg:
		add_expr(req, v.expr)
	case expr.Cov:
		add_expr(req, v.lhs)
		add_expr(req, v.rhs)
	case expr.Corr:
		add_expr(req, v.lhs)
		add_expr(req, v.rhs)
	case expr.Window:
		if v.expr != nil {
			add_expr(req, v.expr)
		}
		for e in v.over {
			add_expr(req, e)
		}
	}
}

// ===========================================================================
// S12.5 common subexpression elimination
// ===========================================================================
//
// exprs_structurally_equal compares two expression trees by shape and value,
// not pointer identity. The executor (physical.odin projection_op) uses it to
// evaluate one canonical expression and deep-copy its column into every output
// position with the same canonical structure, so repeated computed columns do
// not recompute. Sharing a node between two trees is fine: evaluation is
// read-only.

exprs_structurally_equal :: proc(a, b: ^expr.Expr) -> bool {
	if a == nil || b == nil {
		return a == b
	}
	if a == b {
		return true
	}
	switch x in a^ {
	case expr.Col:
		y, ok := b^.(expr.Col)
		return ok && x.name == y.name
	case expr.Lit:
		y, ok := b^.(expr.Lit)
		// Comparing the whole inline buffer is conservative-safe: two equal
		// values always share the significant leading bytes, so a false match
		// is impossible; only padding differences cause a missed match. Copy
		// to locals so the arrays are addressable for slicing.
		if !ok || x.dtype != y.dtype {
			return false
		}
		xd, yd := x.data, y.data
		return mem.compare(xd[:], yd[:]) == 0
	case expr.Binary:
		y, ok := b^.(expr.Binary)
		return ok && x.op == y.op && exprs_structurally_equal(x.lhs, y.lhs) && exprs_structurally_equal(x.rhs, y.rhs)
	case expr.Unary:
		y, ok := b^.(expr.Unary)
		return ok && x.op == y.op && exprs_structurally_equal(x.expr, y.expr)
	case expr.Cast:
		y, ok := b^.(expr.Cast)
		return ok && x.to == y.to && exprs_structurally_equal(x.expr, y.expr)
	case expr.Alias:
		y, ok := b^.(expr.Alias)
		return ok && x.name == y.name && exprs_structurally_equal(x.expr, y.expr)
	case expr.Not_Null:
		y, ok := b^.(expr.Not_Null)
		return ok && exprs_structurally_equal(x.expr, y.expr)
	case expr.Is_Nan:
		y, ok := b^.(expr.Is_Nan)
		return ok && exprs_structurally_equal(x.expr, y.expr)
	case expr.Fill_Null:
		y, ok := b^.(expr.Fill_Null)
		return ok && exprs_structurally_equal(x.expr, y.expr) && exprs_structurally_equal(x.value, y.value)
	case expr.Coalesce:
		y, ok := b^.(expr.Coalesce)
		if !ok || len(x.exprs) != len(y.exprs) {
			return false
		}
		for i in 0 ..< len(x.exprs) {
			if !exprs_structurally_equal(x.exprs[i], y.exprs[i]) {
				return false
			}
		}
		return true
	case expr.Forward_Fill:
		y, ok := b^.(expr.Forward_Fill)
		return ok && exprs_structurally_equal(x.expr, y.expr)
	case expr.Backward_Fill:
		y, ok := b^.(expr.Backward_Fill)
		return ok && exprs_structurally_equal(x.expr, y.expr)
	case expr.Interpolate:
		y, ok := b^.(expr.Interpolate)
		return ok && exprs_structurally_equal(x.expr, y.expr)
	case expr.Func:
		y, ok := b^.(expr.Func)
		return ok && x.kind == y.kind && x.decimals == y.decimals && x.n == y.n && exprs_structurally_equal(x.expr, y.expr)
	case expr.Is_Between:
		y, ok := b^.(expr.Is_Between)
		return ok && exprs_structurally_equal(x.expr, y.expr) && exprs_structurally_equal(x.lower, y.lower) && exprs_structurally_equal(x.upper, y.upper)
	case expr.Is_In:
		y, ok := b^.(expr.Is_In)
		return ok && exprs_structurally_equal(x.expr, y.expr) && exprs_structurally_equal(x.values, y.values)
	case expr.Arange:
		y, ok := b^.(expr.Arange)
		return ok && exprs_structurally_equal(x.start, y.start) && exprs_structurally_equal(x.end, y.end)
	case expr.Arg_Where:
		y, ok := b^.(expr.Arg_Where)
		return ok && exprs_structurally_equal(x.expr, y.expr)
	case expr.Distinct:
		y, ok := b^.(expr.Distinct)
		return ok && x.kind == y.kind && exprs_structurally_equal(x.expr, y.expr)
	case expr.Dot_Product:
		y, ok := b^.(expr.Dot_Product)
		return ok && exprs_structurally_equal(x.lhs, y.lhs) && exprs_structurally_equal(x.rhs, y.rhs)
	case expr.Concat_Str:
		y, ok := b^.(expr.Concat_Str)
		if !ok || x.separator != y.separator || len(x.exprs) != len(y.exprs) {
			return false
		}
		for i in 0 ..< len(x.exprs) {
			if !exprs_structurally_equal(x.exprs[i], y.exprs[i]) {
				return false
			}
		}
		return true
	case expr.Search_Sorted:
		y, ok := b^.(expr.Search_Sorted)
		return ok && exprs_structurally_equal(x.sorted, y.sorted) && exprs_structurally_equal(x.values, y.values)
	case expr.Agg:
		y, ok := b^.(expr.Agg)
		return ok && x.kind == y.kind && x.q == y.q && exprs_structurally_equal(x.expr, y.expr)
	case expr.Cov:
		y, ok := b^.(expr.Cov)
		return ok && exprs_structurally_equal(x.lhs, y.lhs) && exprs_structurally_equal(x.rhs, y.rhs)
	case expr.Corr:
		y, ok := b^.(expr.Corr)
		return ok && exprs_structurally_equal(x.lhs, y.lhs) && exprs_structurally_equal(x.rhs, y.rhs)
	case expr.Window:
		y, ok := b^.(expr.Window)
		if !ok || x.func != y.func || x.n != y.n || x.agg != y.agg || x.alpha != y.alpha || x.method != y.method || len(x.over) != len(y.over) {
			return false
		}
		for i in 0 ..< len(x.over) {
			if !exprs_structurally_equal(x.over[i], y.over[i]) {
				return false
			}
		}
		return exprs_structurally_equal(x.expr, y.expr)
	}
	return false
}

// plan_expr allocates an expression node in the plan arena.
@(private)
plan_expr :: proc(alloc: mem.Allocator, v: expr.Expr) -> ^expr.Expr {
	n := new(expr.Expr, alloc)
	n^ = v
	return n
}
