package expr

// Expression trees for the dataframe API (DESIGN.md §6).
//
// Ownership (DESIGN.md §6.2): an Expr is a recursive tree; interior nodes
// hold ^Expr children. Every node is allocated from a Context arena and freed
// in bulk by context_destroy. A ^Expr borrows its context — it must not be
// used after context_destroy and cannot move between contexts.
//
// Leaves carry no heap: Col/Alias store borrowed name strings, and Lit stores
// its value inline in a fixed-size buffer (#assert(size_of(T) <= 16) covers
// all scalars and the string header).

import "core:mem"

// Binary_Op is a two-operand operation. Arithmetic and comparison ops are
// elementwise over numeric columns; logical ops are bool-only (DESIGN.md
// §6.4).
Binary_Op :: enum byte {
	Add,
	Sub,
	Mul,
	Div,
	Mod,
	Eq,
	Ne,
	Lt,
	Le,
	Gt,
	Ge,
	And,
	Or,
}

// Unary_Op is a one-operand operation: Negates numeric values, Not flips bool.
Unary_Op :: enum byte {
	Neg,
	Not,
}

// Agg_Kind identifies an aggregation operation (Stage 6, DESIGN.md §6.6).
// Every kind reduces a child expression to a single row, skipping NULL rows.
Agg_Kind :: enum byte {
	Count,    // number of valid rows            -> i64
	N_Unique, // number of distinct valid values -> i64
	Sum,      // sum of valid rows               -> f64
	Mean,     // arithmetic mean of valid rows   -> f64
	Var,      // sample variance (n-1)           -> f64
	Std,      // sample standard deviation       -> f64
	Median,   // 0.5 quantile (linear interp)    -> f64
	Quantile, // q quantile (linear interp)      -> f64
	Min,      // minimum (value type preserved)
	Max,      // maximum (value type preserved)
	Product,  // product of valid rows           -> f64
	Mode,     // most frequent value (ties: first seen; type preserved)
	First,    // first valid value (type preserved)
	Last,     // last valid value (type preserved)
	Skew,     // sample skewness (G1)            -> f64
	Kurtosis, // sample excess kurtosis (G2)     -> f64
}

// Agg reduces a child expression to a single-row column (DESIGN.md §6.6).
Agg :: struct {
	kind: Agg_Kind,
	expr: ^Expr,
	q:    f64, // Quantile only
}

// Cov is the sample covariance of two equal-length numeric columns, yielding
// a single-row f64 column. Rows invalid in either column are skipped.
Cov :: struct {
	lhs: ^Expr,
	rhs: ^Expr,
}

// Corr is the Pearson correlation of two equal-length numeric columns,
// yielding a single-row f64 column. Rows invalid in either column are skipped.
Corr :: struct {
	lhs: ^Expr,
	rhs: ^Expr,
}

// Window_Func identifies a per-partition, order-preserving computation
// (Stage 14.4, DESIGN.md §18.2). `over` partitions the rows by evaluated key
// expressions (empty = one partition of all rows); rows within a partition
// keep source order.
Window_Func :: enum byte {
	Row_Number,
	Rank,
	Cum_Sum,
	Cum_Min,
	Cum_Max,
	Shift,
	Rolling,
	Cumulative_Eval,
	Ewma,
}

// Rank_Method selects the tie-breaking rule for Window_Func.Rank.
Rank_Method :: enum byte {
	Average, // tied rows share the mean of their positions (polars default)
	Min,     // tied rows all get the best (smallest) position
	Max,     // tied rows all get the worst (largest) position
	Dense,   // consecutive distinct values get consecutive ranks
}

// Window computes a window function over each partition of the rows. `expr` is
// the child (nil for Row_Number); `over` holds the partition-key expressions;
// `n` is the Shift lag / Rolling window size; `agg` and `alpha` configure
// Rolling / Cumulative_Eval / Ewma.
Window :: struct {
	func:   Window_Func,
	expr:   ^Expr,
	over:   []^Expr,
	n:      i64,
	agg:    Agg_Kind,
	alpha:  f64,
	method: Rank_Method,
}

// Expr is the expression tree node union. Dispatch with `switch v in e^`.
Expr :: union {
	Col,
	Lit,
	Binary,
	Unary,
	Cast,
	Alias,
	Not_Null,
	Is_Nan,
	Fill_Null,
	Coalesce,
	Forward_Fill,
	Backward_Fill,
	Interpolate,
	Func,
	Is_Between,
	Is_In,
	Arange,
	Arg_Where,
	Distinct,
	Dot_Product,
	Concat_Str,
	Search_Sorted,
	Agg,
	Cov,
	Corr,
	Window,
}

// Col references a column of the evaluated DataFrame by (borrowed) name.
Col :: struct {
	name: string,
}

// Lit is a typed constant stored inline (no heap). `data` holds the value
// bytes of type `dtype`; read it back with lit_as.
Lit :: struct {
	dtype: typeid,
	data:  [16]byte,
}

// Binary applies op to two child expressions.
Binary :: struct {
	op:   Binary_Op,
	lhs:  ^Expr,
	rhs:  ^Expr,
}

// Unary applies op to a child expression.
Unary :: struct {
	op:   Unary_Op,
	expr: ^Expr,
}

// Cast converts a child expression's result to `to` (numeric types only,
// DESIGN.md §6.4).
Cast :: struct {
	expr: ^Expr,
	to:   typeid,
}

// Alias names a child expression's result.
Alias :: struct {
	expr: ^Expr,
	name: string,
}

// Not_Null tests each row of a child expression for NULL.
Not_Null :: struct {
	expr: ^Expr,
}

// Is_Nan tests each row of a float child expression for NaN. NULL rows stay
// NULL.
Is_Nan :: struct {
	expr: ^Expr,
}

// Fill_Null replaces the NULL rows of expr with the constant scalar `value` (a
// Lit child of the same dtype as expr).
Fill_Null :: struct {
	expr:  ^Expr,
	value: ^Expr,
}

// Coalesce returns, per row, the first non-NULL value across parts. Every part
// must produce the same dtype; a row is NULL when every part is NULL.
Coalesce :: struct {
	exprs: []^Expr,
}

// Forward_Fill carries the last valid value across trailing NULL rows.
Forward_Fill :: struct {
	expr: ^Expr,
}

// Backward_Fill carries the next valid value across leading NULL rows.
Backward_Fill :: struct {
	expr: ^Expr,
}

// Interpolate linearly interpolates the NULL rows of a numeric child between
// the surrounding valid values (f64 result). Leading/trailing NULLs stay NULL.
Interpolate :: struct {
	expr: ^Expr,
}

// Func_Kind identifies a single-child numeric function (S3.10).
Func_Kind :: enum byte {
	Abs,        // |x|
	Sign,       // -1 / 0 / 1 (math.sign; f32/f64, int, i16/i32/i64)
	Round,      // round to `decimals` places (float columns only)
	Diff,       // x[i] - x[i-1]; first `n` rows are NULL
	Pct_Change, // (x[i] - x[i-1]) / x[i-1]; first `n` rows are NULL
	Cum_Sum,    // running sum (numeric only)
	First_Distinct, // true where the row is the first occurrence of its value
	Last_Distinct,  // true where the row is the last occurrence of its value
}

// Func applies a single-child numeric function to its child.
Func :: struct {
	kind:     Func_Kind,
	expr:     ^Expr,
	decimals: i32, // Round only
	n:        int, // Diff / Pct_Change: lag (default 1)
}

// Is_Between tests rows for lower <= x <= upper (inclusive bounds). The
// bounds are constant scalar expressions (Col-less) of the child's dtype.
Is_Between :: struct {
	expr:  ^Expr,
	lower: ^Expr,
	upper: ^Expr,
}

// Is_In tests each row for membership in a constant list literal (`Lit`
// holding a []T of the child's dtype).
Is_In :: struct {
	expr: ^Expr,
	values: ^Expr,
}

// Arange builds an int column from start (inclusive) to end (exclusive).
// Both bounds are constant scalar expressions.
Arange :: struct {
	start: ^Expr,
	end:   ^Expr,
}

// Arg_Where returns the row indices (as an int column) where expr is true.
Arg_Where :: struct {
	expr: ^Expr,
}

// Distinct marks the first or last occurrence of each distinct value
// (First_Distinct / Last_Distinct), including NULLs (a NULL row is marked).
Distinct :: struct {
	kind:     Func_Kind,
	expr:     ^Expr,
}

// Dot_Product sums lhs[i] * rhs[i] over two equal-length numeric columns,
// yielding a single-row f64 column. NULLs are skipped (treated as 0).
Dot_Product :: struct {
	lhs: ^Expr,
	rhs: ^Expr,
}

// Concat_Str joins string column rows with `separator` ("" for plain
// concatenation). The result is NULL where any input row is NULL.
Concat_Str :: struct {
	exprs:     []^Expr,
	separator: string,
}

// Search_Sorted binary-searches each value of `values` in the sorted column
// `sorted`, returning the insertion index as an int column. NULL values give
// NULL results.
Search_Sorted :: struct {
	sorted: ^Expr,
	values: ^Expr,
}

// Context is an arena that owns every node allocated from it. Create with
// context_create, release all nodes with context_destroy.
Context :: struct {
	alloc: mem.Allocator,
	nodes: [dynamic]^Expr,
	// extra owns out-of-line allocations made by constructors (e.g. the parts
	// array of Concat_Str); released by context_destroy.
	extra: [dynamic]rawptr,
}

// context_create returns an empty expression arena bound to allocator.
context_create :: proc(allocator := context.allocator) -> Context {
	return Context {
		alloc = allocator,
		nodes = make([dynamic]^Expr, allocator),
		extra = make([dynamic]rawptr, allocator),
	}
}

// context_destroy frees every node allocated from ctx. Any ^Expr obtained
// from ctx is invalid afterwards.
context_destroy :: proc(ctx: ^Context) {
	for n in ctx.nodes {
		mem.free(n, ctx.alloc)
	}
	delete(ctx.nodes)
	for p in ctx.extra {
		mem.free(p, ctx.alloc)
	}
	delete(ctx.extra)
	ctx^ = {}
}

// node allocates a heap node holding e and tracks it in the arena.
@(private)
node :: proc(ctx: ^Context, e: Expr) -> ^Expr {
	n := new(Expr, ctx.alloc)
	n^ = e
	append(&ctx.nodes, n)
	return n
}

// --- constructors -----------------------------------------------------------

// col builds a column-reference node.
col :: proc(ctx: ^Context, name: string) -> ^Expr {
	return node(ctx, Col{name = name})
}

// lit builds a constant node. Any type up to 16 bytes fits inline (scalars,
// string headers, small structs); larger types are a compile-time error.
lit :: proc(ctx: ^Context, value: $T) -> ^Expr {
	#assert(size_of(T) <= size_of([16]byte), "expr.lit: type too large for inline literal storage")
	l := Lit{dtype = typeid_of(T)}
	v := value
	mem.copy(&l.data, &v, size_of(T))
	return node(ctx, l)
}

// lit_as reads the inline value of a Lit node as T. Returns ok=false when the
// literal's type is not T (there is no silent conversion, principle 6).
lit_as :: proc(l: ^Lit, $T: typeid) -> (value: T, ok: bool) {
	if l.dtype != typeid_of(T) {
		return {}, false
	}
	mem.copy(&value, &l.data, size_of(T))
	return value, true
}

// binary builds a two-operand node.
@(private)
binary :: proc(ctx: ^Context, op: Binary_Op, lhs, rhs: ^Expr) -> ^Expr {
	return node(ctx, Binary{op = op, lhs = lhs, rhs = rhs})
}

add :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr { return binary(ctx, .Add, lhs, rhs) }
sub :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr { return binary(ctx, .Sub, lhs, rhs) }
mul :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr { return binary(ctx, .Mul, lhs, rhs) }
div :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr { return binary(ctx, .Div, lhs, rhs) }
mod :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr { return binary(ctx, .Mod, lhs, rhs) }
eq  :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr { return binary(ctx, .Eq, lhs, rhs) }
ne  :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr { return binary(ctx, .Ne, lhs, rhs) }
lt  :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr { return binary(ctx, .Lt, lhs, rhs) }
le  :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr { return binary(ctx, .Le, lhs, rhs) }
gt  :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr { return binary(ctx, .Gt, lhs, rhs) }
ge  :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr { return binary(ctx, .Ge, lhs, rhs) }
and_ :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr { return binary(ctx, .And, lhs, rhs) }
or_  :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr { return binary(ctx, .Or, lhs, rhs) }

// neg builds a numeric negation node.
neg :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Unary{op = .Neg, expr = e})
}

// not_ builds a logical-not node.
not_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Unary{op = .Not, expr = e})
}

// cast_ builds an explicit numeric conversion node. T is the target type,
// e.g. expr.cast_(&ctx, e, f64).
cast_ :: proc(ctx: ^Context, e: ^Expr, $T: typeid) -> ^Expr {
	return node(ctx, Cast{expr = e, to = typeid_of(T)})
}

// alias names the result of e.
alias :: proc(ctx: ^Context, e: ^Expr, name: string) -> ^Expr {
	return node(ctx, Alias{expr = e, name = name})
}

// is_not_null builds a NULL test node.
is_not_null :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Not_Null{expr = e})
}

// is_null_ is the complement of is_not_null.
is_null_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return not_(ctx, is_not_null(ctx, e))
}

// is_nan_ tests a float column for NaN rows.
is_nan_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Is_Nan{expr = e})
}

// fill_null_ replaces NULL rows of e with the constant scalar value.
fill_null_ :: proc(ctx: ^Context, e, value: ^Expr) -> ^Expr {
	return node(ctx, Fill_Null{expr = e, value = value})
}

// coalesce_ returns the first non-NULL value per row across parts. The part
// list is copied into the arena.
coalesce_ :: proc(ctx: ^Context, parts: []^Expr) -> ^Expr {
	copied := make([]^Expr, len(parts), ctx.alloc)
	copy_slice(copied, parts)
	append(&ctx.extra, raw_data(copied))
	return node(ctx, Coalesce{exprs = copied})
}

// ffill_ carries the last valid value across trailing NULL rows.
ffill_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Forward_Fill{expr = e})
}

// bfill_ carries the next valid value across leading NULL rows.
bfill_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Backward_Fill{expr = e})
}

// interpolate_ linearly interpolates NULL rows between surrounding valid
// values (numeric child, f64 result).
interpolate_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Interpolate{expr = e})
}

// --- S3.10 function constructors --------------------------------------------

// abs_ builds |e|.
abs_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Func{kind = .Abs, expr = e})
}

// sign_ builds the elementwise sign (-1 / 0 / 1) of e.
sign_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Func{kind = .Sign, expr = e})
}

// round_ builds round-to-decimals of e (float columns only).
round_ :: proc(ctx: ^Context, e: ^Expr, decimals: i32) -> ^Expr {
	return node(ctx, Func{kind = .Round, expr = e, decimals = decimals})
}

// diff_ builds the n-lag difference of e; the first n rows are NULL.
diff_ :: proc(ctx: ^Context, e: ^Expr, n: int) -> ^Expr {
	return node(ctx, Func{kind = .Diff, expr = e, n = n})
}

// pct_change_ builds the fractional change vs n rows back.
pct_change_ :: proc(ctx: ^Context, e: ^Expr, n: int) -> ^Expr {
	return node(ctx, Func{kind = .Pct_Change, expr = e, n = n})
}

// cum_sum_ builds the running sum of e (numeric only).
cum_sum_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Func{kind = .Cum_Sum, expr = e})
}

// is_between_ tests lower <= e <= upper (inclusive). Bounds are constant
// scalar expressions of the same dtype as e.
is_between_ :: proc(ctx: ^Context, e, lower, upper: ^Expr) -> ^Expr {
	return node(ctx, Is_Between{expr = e, lower = lower, upper = upper})
}

// is_in_ tests membership of e in a constant list literal values (a `Lit`
// node holding a []T).
is_in_ :: proc(ctx: ^Context, e, values: ^Expr) -> ^Expr {
	return node(ctx, Is_In{expr = e, values = values})
}

// arange_ builds an int column start ..< end (both constant scalar exprs).
arange_ :: proc(ctx: ^Context, start, end: ^Expr) -> ^Expr {
	return node(ctx, Arange{start = start, end = end})
}

// arg_where_ returns the row indices where e is true.
arg_where_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Arg_Where{expr = e})
}

// first_distinct_ marks rows that are the first occurrence of their value.
first_distinct_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Distinct{kind = .First_Distinct, expr = e})
}

// last_distinct_ marks rows that are the last occurrence of their value.
last_distinct_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Distinct{kind = .Last_Distinct, expr = e})
}

// dot_product_ builds the elementwise product sum of two columns (single-row
// f64 result).
dot_product_ :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr {
	return node(ctx, Dot_Product{lhs = lhs, rhs = rhs})
}

// concat_str_ joins the string column rows of parts with separator ("" joins
// without a separator). The part list is copied into the arena.
concat_str_ :: proc(ctx: ^Context, parts: []^Expr, separator: string) -> ^Expr {
	copied := make([]^Expr, len(parts), ctx.alloc)
	copy_slice(copied, parts)
	append(&ctx.extra, raw_data(copied))
	return node(ctx, Concat_Str{exprs = copied, separator = separator})
}

// search_sorted_ binary-searches the sorted column for each value.
search_sorted_ :: proc(ctx: ^Context, sorted, values: ^Expr) -> ^Expr {
	return node(ctx, Search_Sorted{sorted = sorted, values = values})
}

// --- Stage 6 aggregation constructors ----------------------------------------

// count_ builds the number of valid rows of e (i64 result).
count_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .Count, expr = e})
}

// n_unique_ builds the number of distinct valid values of e (i64 result).
n_unique_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .N_Unique, expr = e})
}

// sum_ builds the sum of the valid rows of e (f64 result).
sum_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .Sum, expr = e})
}

// mean_ builds the arithmetic mean of the valid rows of e (f64 result).
mean_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .Mean, expr = e})
}

// var_ builds the sample variance (n-1) of the valid rows of e.
var_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .Var, expr = e})
}

// std_ builds the sample standard deviation of the valid rows of e.
std_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .Std, expr = e})
}

// median_ builds the 0.5 quantile of the valid rows of e (f64 result).
median_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .Median, expr = e})
}

// quantile_ builds the q quantile (linear interpolation) of the valid rows of
// e. q must be in [0, 1].
quantile_ :: proc(ctx: ^Context, e: ^Expr, q: f64) -> ^Expr {
	return node(ctx, Agg{kind = .Quantile, expr = e, q = q})
}

// min_ builds the minimum of the valid rows of e (value type preserved).
min_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .Min, expr = e})
}

// max_ builds the maximum of the valid rows of e (value type preserved).
max_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .Max, expr = e})
}

// product_ builds the product of the valid rows of e (f64 result).
product_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .Product, expr = e})
}

// mode_ builds the most frequent value of the valid rows of e; ties resolve to
// the value seen first (value type preserved).
mode_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .Mode, expr = e})
}

// first_ builds the first valid value of e (value type preserved).
first_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .First, expr = e})
}

// last_ builds the last valid value of e (value type preserved).
last_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .Last, expr = e})
}

// skew_ builds the sample skewness (G1) of the valid rows of e.
skew_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .Skew, expr = e})
}

// kurtosis_ builds the sample excess kurtosis (G2) of the valid rows of e.
kurtosis_ :: proc(ctx: ^Context, e: ^Expr) -> ^Expr {
	return node(ctx, Agg{kind = .Kurtosis, expr = e})
}

// cov_ builds the sample covariance of two numeric columns (single-row f64).
cov_ :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr {
	return node(ctx, Cov{lhs = lhs, rhs = rhs})
}

// corr_ builds the Pearson correlation of two numeric columns (single-row f64).
corr_ :: proc(ctx: ^Context, lhs, rhs: ^Expr) -> ^Expr {
	return node(ctx, Corr{lhs = lhs, rhs = rhs})
}

// --- Stage 14.4 window constructors ------------------------------------------

// copy_over copies the partition-key expression slice into the arena (owned by
// the context, released by context_destroy).
@(private)
copy_over :: proc(ctx: ^Context, over: []^Expr) -> []^Expr {
	copied := make([]^Expr, len(over), ctx.alloc)
	copy_slice(copied, over)
	append(&ctx.extra, raw_data(copied))
	return copied
}

// window_row_number builds a 1-based per-partition position column (i64,
// always valid). The result has no source column name, so it must be aliased.
window_row_number :: proc(ctx: ^Context, over: []^Expr) -> ^Expr {
	return node(ctx, Window{func = .Row_Number, over = copy_over(ctx, over)})
}

// window_rank builds a per-partition rank (f64) of e with the given tie method
// (default Average). NULL rows yield NULL.
window_rank :: proc(ctx: ^Context, e: ^Expr, method: Rank_Method = .Average, over: []^Expr) -> ^Expr {
	return node(ctx, Window{func = .Rank, expr = e, over = copy_over(ctx, over), method = method})
}

// window_cum_sum builds the running sum of e per partition (output dtype =
// child dtype); NULL rows yield NULL and the running sum continues.
window_cum_sum :: proc(ctx: ^Context, e: ^Expr, over: []^Expr) -> ^Expr {
	return node(ctx, Window{func = .Cum_Sum, expr = e, over = copy_over(ctx, over)})
}

// window_cum_min builds the running minimum of e per partition.
window_cum_min :: proc(ctx: ^Context, e: ^Expr, over: []^Expr) -> ^Expr {
	return node(ctx, Window{func = .Cum_Min, expr = e, over = copy_over(ctx, over)})
}

// window_cum_max builds the running maximum of e per partition.
window_cum_max :: proc(ctx: ^Context, e: ^Expr, over: []^Expr) -> ^Expr {
	return node(ctx, Window{func = .Cum_Max, expr = e, over = copy_over(ctx, over)})
}

// window_shift builds the n-lag of e per partition: out[i] = in[i-n] (positive
// n moves values down; negative moves up). Out-of-partition positions are
// NULL.
window_shift :: proc(ctx: ^Context, e: ^Expr, n: i64, over: []^Expr) -> ^Expr {
	return node(ctx, Window{func = .Shift, expr = e, over = copy_over(ctx, over), n = n})
}

// window_rolling builds, for each row, the aggregation `agg` over the trailing
// window [i-n+1, i] within its partition (n >= 1).
window_rolling :: proc(ctx: ^Context, e: ^Expr, window_size: i64, agg: Agg_Kind, over: []^Expr) -> ^Expr {
	return node(ctx, Window{func = .Rolling, expr = e, over = copy_over(ctx, over), n = window_size, agg = agg})
}

// window_cumulative_eval builds, for each row, the aggregation `agg` over the
// growing window [0, i] within its partition.
window_cumulative_eval :: proc(ctx: ^Context, e: ^Expr, agg: Agg_Kind, over: []^Expr) -> ^Expr {
	return node(ctx, Window{func = .Cumulative_Eval, expr = e, over = copy_over(ctx, over), agg = agg})
}

// window_ewma builds the exponential weighted moving average of e per
// partition: y_i = alpha·x_i + (1-alpha)·y_{i-1} (f64 result). The first valid
// row seeds y = x; NULL rows yield NULL and the recursion continues.
window_ewma :: proc(ctx: ^Context, e: ^Expr, alpha: f64, over: []^Expr) -> ^Expr {
	return node(ctx, Window{func = .Ewma, expr = e, over = copy_over(ctx, over), alpha = alpha})
}
