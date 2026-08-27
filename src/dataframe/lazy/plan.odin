package lazy

// Logical query plans (Stage 11, DESIGN.md §7).
//
// A Logical_Plan is a tree whose leaves are scans (Scan_CSV, Scan_DF) and
// whose interior nodes are relational operators (Filter, Projection, Sort,
// Group_By, Limit, Slice, Join). Pointers form the tree; every node lives in
// the arena owned by the LazyFrame that built it (lazyframe.odin) and is
// freed wholesale by lazy.destroy.
//
// Ownership (DESIGN.md §7.2): nodes own the slices they need copied into the
// arena ([]^expr.Expr, []Sort_Key, []string) and borrow everything else —
// expr nodes come from the caller's expr.Context, a Scan_DF borrows its
// DataFrame, and name/path strings are borrowed. The rule for users: keep
// every expr context (and any scanned DataFrame) alive until collect returns.
//
// Building a plan never touches data (S11.5): no I/O, no evaluation, no
// validation. Errors for missing files, unknown columns, type mismatches,
// and bad argument shapes all surface at collect() time.

import "../../dataframe"
import "../../dataframe/expr"

// Logical_Plan is the query tree. Dispatch with `switch n in plan^`.
Logical_Plan :: union {
	Scan_CSV,
	Scan_DF,
	Filter,
	Projection,
	Sort,
	Group_By,
	Limit,
	Slice,
	Join,
}

// Scan_CSV reads a CSV file at collect time. `path` and the option token are
// borrowed. `columns` is the S12.1 projection-pushdown result: when
// non-empty, only those columns are read (in this order); when empty, every
// column is read. The optimizer (optimizer.odin) fills it at collect time.
Scan_CSV :: struct {
	path:    string,
	options: dataframe.CSV_Options,
	columns: []string,
}

// Scan_DF wraps an existing in-memory DataFrame. The frame is borrowed: it
// must outlive this plan and must not be restructured before collect.
Scan_DF :: struct {
	df: ^dataframe.DataFrame,
}

// Filter keeps the rows of child where predicate evaluates to true (NULL
// predicates drop the row). `predicate` is borrowed from an expr context.
Filter :: struct {
	child:     ^Logical_Plan,
	predicate: ^expr.Expr,
}

// Projection evaluates each expr against child and keeps the result columns
// in order (eager dataframe_select semantics). exprs is owned by the arena.
Projection :: struct {
	child: ^Logical_Plan,
	exprs: []^expr.Expr,
}

// Sort reorders child's rows by the key columns (stable; NULL placement per
// key). keys is owned by the arena.
Sort :: struct {
	child: ^Logical_Plan,
	keys:  []dataframe.Sort_Key,
}

// Group_By groups child's rows by the key expressions and, once aggs is set
// (via lazy.agg), aggregates each group. Collecting a Group_By node with an
// empty aggs list is an .Invalid_Argument error. keys and aggs are owned by
// the arena.
Group_By :: struct {
	child: ^Logical_Plan,
	keys:  []^expr.Expr,
	aggs:  []^expr.Expr,
}

// Limit keeps the first n rows of child.
Limit :: struct {
	child: ^Logical_Plan,
	n:     int,
}

// Slice keeps rows [offset, offset+length) of child.
Slice :: struct {
	child:  ^Logical_Plan,
	offset: int,
	length: int,
}

// Join combines two child plans with the given join kind and key column
// names (empty keys with .Cross selects the cross join). left_keys and
// right_keys are owned by the arena.
Join :: struct {
	left:      ^Logical_Plan,
	right:     ^Logical_Plan,
	kind:      dataframe.Join_Kind,
	left_keys: []string,
	right_keys: []string,
}
