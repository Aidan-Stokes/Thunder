package lazy

// LazyFrame and plan builders (Stage 11, DESIGN.md §7.1).
//
// A LazyFrame is a thin wrapper over a Logical_Plan tree. Every builder
// (lazy.filter, lazy.select, ...) allocates one new node in the frame's
// arena and returns a new LazyFrame sharing that arena — no data is read or
// evaluated. Odin has no method-call sugar for procs, so the idiomatic style
// is qualified calls composed left to right (S10.2, DESIGN.md §15):
//
//	lf := lazy.scan_csv("sales.csv")
//	lf  = lazy.filter(lf, pred)
//	lf  = lazy.sort(lf, []dataframe.Sort_Key{dataframe.sort_key("price", .Desc)})
//	lf  = lazy.limit(lf, 10)
//	out := lazy.collect(lf) or_return
//	lazy.destroy(lf)
//
// Ownership: a LazyFrame owns one arena holding its plan nodes (and the
// slices they copied in); `destroy` frees the whole arena. Linear builders
// (filter/select/sort/group_by/agg/limit/slice) all append to the SAME arena
// as their input, so the recommended style is to rebind one variable and
// destroy it exactly once:
//
//	lf := lazy.scan_csv("sales.csv")
//	lf  = lazy.filter(lf, pred)
//	lf  = lazy.sort(lf, keys)
//	out := lazy.collect(lf) or_return
//	lazy.destroy(lf) // frees the whole chain — destroy no other frame of it
//
// lazy.join is the exception: it builds an independent plan, so each of its
// three frames is destroyed once. expr nodes and scanned DataFrames are
// borrowed (see plan.odin); keep their owners alive until collect returns.

import "core:mem"
import "../../dataframe"
import "../../dataframe/expr"

// Plan is the shared arena behind one logical plan tree. A LazyFrame owns a
// ^Plan; every derived LazyFrame shares it.
Plan :: struct {
	arena:   mem.Dynamic_Arena,
	alloc:   mem.Allocator, // arena-backed; for plan nodes and node slices
	backing: mem.Allocator, // the allocator Plan and its arena blocks come from
}

// LazyFrame is a reference to one node of a plan tree. Copying a LazyFrame
// shares the plan; destroy one of the copies when done.
LazyFrame :: struct {
	root: ^Logical_Plan,
	plan: ^Plan,
}

// --- sources ----------------------------------------------------------------

// scan_csv opens a lazy plan that reads path at collect time. Nothing is
// read here (S11.5); a missing or malformed file is a .CSV_Error from
// collect.
scan_csv :: proc(path: string, options: dataframe.CSV_Options = {}, allocator := context.allocator) -> LazyFrame {
	p := plan_create(allocator)
	root := plan_new(p.alloc, Scan_CSV{path = path, options = options})
	return LazyFrame{root = root, plan = p}
}

// scan_dataframe wraps an in-memory DataFrame in a lazy plan. The frame is
// borrowed — it must outlive the plan and must not be restructured before
// collect. Collecting a bare scan_dataframe root returns an owned deep copy.
scan_dataframe :: proc(df: ^dataframe.DataFrame, allocator := context.allocator) -> LazyFrame {
	p := plan_create(allocator)
	root := plan_new(p.alloc, Scan_DF{df = df})
	return LazyFrame{root = root, plan = p}
}

// --- builders ---------------------------------------------------------------

// filter keeps the rows where predicate is true. `predicate` is borrowed
// from an expr context that must outlive collect.
filter :: proc(lf: LazyFrame, predicate: ^expr.Expr) -> LazyFrame {
	return with_root(lf, plan_new(lf.plan.alloc, Filter{child = lf.root, predicate = predicate}))
}

// select evaluates exprs against each row set and keeps the result columns.
select :: proc(lf: LazyFrame, exprs: []^expr.Expr) -> LazyFrame {
	return with_root(lf, plan_new(lf.plan.alloc, Projection{child = lf.root, exprs = clone_exprs(lf.plan.alloc, exprs)}))
}

// sort reorders rows by the key columns (stable; per-key NULL placement).
sort :: proc(lf: LazyFrame, keys: []dataframe.Sort_Key) -> LazyFrame {
	return with_root(lf, plan_new(lf.plan.alloc, Sort{child = lf.root, keys = clone_keys(lf.plan.alloc, keys)}))
}

// group_by groups rows by the key expressions. Follow with lazy.agg to set
// the aggregations.
group_by :: proc(lf: LazyFrame, keys: []^expr.Expr) -> LazyFrame {
	return with_root(lf, plan_new(lf.plan.alloc, Group_By{child = lf.root, keys = clone_exprs(lf.plan.alloc, keys)}))
}

// agg attaches the aggregation expressions to the group_by step at the root
// of the plan. The plan root must be a Group_By node (built by lazy.group_by)
// or an .Invalid_Argument error is returned.
agg :: proc(lf: LazyFrame, aggs: []^expr.Expr) -> (LazyFrame, dataframe.Error) {
	#partial switch n in lf.root^ {
	case Group_By:
		gb := n
		node := plan_new(lf.plan.alloc, Group_By{
			child = gb.child,
			keys  = gb.keys,
			aggs  = clone_exprs(lf.plan.alloc, aggs),
		})
		return with_root(lf, node), .None
	}
	return {}, .Invalid_Argument
}

// limit keeps the first n rows.
limit :: proc(lf: LazyFrame, n: int) -> LazyFrame {
	return with_root(lf, plan_new(lf.plan.alloc, Limit{child = lf.root, n = n}))
}

// slice keeps rows [offset, offset+length).
slice :: proc(lf: LazyFrame, offset: int, length: int) -> LazyFrame {
	return with_root(lf, plan_new(lf.plan.alloc, Slice{child = lf.root, offset = offset, length = length}))
}

// join combines two plans into an independent plan. Both child trees are
// cloned into a fresh arena, so the joined frame is owned independently of
// its inputs: destroy left, right, and the joined frame each exactly once.
// left and right may be built with different expr contexts — their expr
// nodes are borrowed either way.
join :: proc(left, right: LazyFrame, kind: dataframe.Join_Kind, left_keys, right_keys: []string) -> LazyFrame {
	p := plan_create(left.plan.backing)
	left_root := clone_plan(p.alloc, left.root)
	right_root := clone_plan(p.alloc, right.root)
	node := plan_new(p.alloc, Join{
		left      = left_root,
		right     = right_root,
		kind      = kind,
		left_keys = clone_strings(p.alloc, left_keys),
		right_keys = clone_strings(p.alloc, right_keys),
	})
	return LazyFrame{root = node, plan = p}
}

// --- lifecycle ---------------------------------------------------------------

// destroy frees the plan arena behind lf. Any LazyFrame sharing lf's plan
// is invalid afterwards. Destroy each source frame exactly once.
destroy :: proc(lf: ^LazyFrame) {
	if lf.plan == nil {
		return
	}
	mem.dynamic_arena_destroy(&lf.plan.arena)
	mem.free(lf.plan, lf.plan.backing)
	lf^ = {}
}

// --- arena plumbing ----------------------------------------------------------

@(private)
plan_create :: proc(allocator: mem.Allocator) -> ^Plan {
	p := new(Plan, allocator)
	mem.dynamic_arena_init(&p.arena, allocator, allocator)
	p.alloc = mem.dynamic_arena_allocator(&p.arena)
	p.backing = allocator
	return p
}

@(private)
plan_new :: proc(alloc: mem.Allocator, v: Logical_Plan) -> ^Logical_Plan {
	p := new(Logical_Plan, alloc)
	p^ = v
	return p
}

@(private)
with_root :: proc(lf: LazyFrame, root: ^Logical_Plan) -> LazyFrame {
	return LazyFrame{root = root, plan = lf.plan}
}

@(private)
clone_exprs :: proc(alloc: mem.Allocator, exprs: []^expr.Expr) -> []^expr.Expr {
	if len(exprs) == 0 {
		return nil
	}
	out := make([]^expr.Expr, len(exprs), alloc)
	mem.copy(raw_data(out), raw_data(exprs), len(exprs) * size_of(^expr.Expr))
	return out
}

@(private)
clone_keys :: proc(alloc: mem.Allocator, keys: []dataframe.Sort_Key) -> []dataframe.Sort_Key {
	if len(keys) == 0 {
		return nil
	}
	out := make([]dataframe.Sort_Key, len(keys), alloc)
	mem.copy(raw_data(out), raw_data(keys), len(keys) * size_of(dataframe.Sort_Key))
	return out
}

@(private)
clone_strings :: proc(alloc: mem.Allocator, src: []string) -> []string {
	if len(src) == 0 {
		return nil
	}
	out := make([]string, len(src), alloc)
	mem.copy(raw_data(out), raw_data(src), len(src) * size_of(string))
	return out
}

// clone_plan deep-copies a plan tree into alloc. Only the node structure and
// owned slices are copied; expr nodes, strings, and DataFrame pointers stay
// borrowed (they outlive the plan by contract).
@(private)
clone_plan :: proc(alloc: mem.Allocator, src: ^Logical_Plan) -> ^Logical_Plan {
	switch n in src^ {
	case Scan_CSV:
		return plan_new(alloc, Scan_CSV{
			path    = n.path,
			options = n.options,
			columns = clone_strings(alloc, n.columns),
		})
	case Scan_DF:
		return plan_new(alloc, n)
	case Filter:
		return plan_new(alloc, Filter{child = clone_plan(alloc, n.child), predicate = n.predicate})
	case Projection:
		return plan_new(alloc, Projection{child = clone_plan(alloc, n.child), exprs = clone_exprs(alloc, n.exprs)})
	case Sort:
		return plan_new(alloc, Sort{child = clone_plan(alloc, n.child), keys = clone_keys(alloc, n.keys)})
	case Group_By:
		return plan_new(alloc, Group_By{
			child = clone_plan(alloc, n.child),
			keys  = clone_exprs(alloc, n.keys),
			aggs  = clone_exprs(alloc, n.aggs),
		})
	case Limit:
		return plan_new(alloc, Limit{child = clone_plan(alloc, n.child), n = n.n})
	case Slice:
		return plan_new(alloc, Slice{child = clone_plan(alloc, n.child), offset = n.offset, length = n.length})
	case Join:
		return plan_new(alloc, Join{
			left      = clone_plan(alloc, n.left),
			right     = clone_plan(alloc, n.right),
			kind      = n.kind,
			left_keys = clone_strings(alloc, n.left_keys),
			right_keys = clone_strings(alloc, n.right_keys),
		})
	}
	return nil
}
