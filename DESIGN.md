# DESIGN.md — Odin dataframe / query-engine library

Status: **Stage 15 — performance** (in progress). Stages 1–14
(columnar core, DataFrame, expressions, ops, sort, aggs, groupby, joins,
CSV/JSON I/O, ergonomics, lazy engine + optimizer, advanced functionality)
shipped. Stage 15 lands validity bitmap, specialized/SIMD kernels, parallel
join, parallel CSV parsing, memory reuse, and benchmarks (ROADMAP.md Stage 15).
Scope: Pandas/Polars-inspired, with Polars as the architectural reference
(columnar memory, expressions, lazy plans, optimizer) and Pandas as the feature
breadth reference.

## 1. Goals and non-goals

This library does not reinvent the wheel: it adds **dataframe support to
Odin** by wiring existing tooling into a data API. Odin's `core:` stdlib
(sort, math, encoding, thread/sync, container, simd, prof), the in-tree
`src/parallel` pool helper, and the language's own machinery (generics,
`typeid`, explicit allocators) are the building blocks; custom code is written
only where no existing tool fits, and then as thinly as possible so the pieces
stay swappable. This keeps the surface small, the maintenance burden low, and
the library extensible.

Goals:
- A native, fast, columnar dataframe library for Odin.
- Expression-based eager API (`DataFrame`) and a lazy query engine (`LazyFrame`).
- Explicit memory ownership via Odin allocator conventions.
- Eventual parallel execution using `core:thread`/`core:sync` and the existing
  `src/parallel` pool helper.

Non-goals (for now):
- Literal Pandas API compatibility (we design an idiomatic Odin API instead).
- Nested types (List/Array/Struct) as first-class logical dtypes. Columns of
  arbitrary Odin structs work via the generic column constructor, but nested
  *logical* types (dtype-aware nesting, `explode`, `unnest`) are deferred.
- A full SQL engine, Parquet interop, window functions, time series.
  These are on the roadmap, not in the MVP. Arrow IPC is done (S17).

### 1.1 Build on existing tooling, not from scratch

Everything a dataframe engine needs that Odin already ships is reused as-is;
custom code is the thin glue between existing pieces (principle 2). The
`core:` stdlib and in-tree helpers cover most of the engine:

| Need | Existing tool | Where used |
|---|---|---|
| Sorting / permutation | `core:sort` (Interface form) | sort.odin (S5) |
| Numeric functions | `core:math` (`abs`, `sign`, `round`, `pow`, …) | expr library (S3.10) |
| String joining / ops | `core:strings.join` + `core:strings` | `concat_str` (S3.10) |
| CSV parsing | `core:encoding/csv` | I/O (S9) |
| JSON / NDJSON | `core:encoding/json` | I/O (S13) |
| Threading / sync | `core:thread` / `core:sync` | parallel executor (S15.4) |
| Thread pool | in-tree `src/parallel` (`do_parallel`) | executor (S15.4) |
| Validity bitmap | packed `[]u64` in `column.odin` (bm_get/bm_set) | S15.1 |
| Profiling | `core:prof` | performance stage |
| SIMD (later) | `core:simd`, per-arch guarded | S15.5 |

Odin's own language features are equally reused instead of built around:
generics (`$T`) give typed kernels without codegen; `typeid` gives dynamic
typing without reflection; explicit allocators give ownership without a GC.
When no existing tool fits (e.g., stable sort over composite row keys, group
keys, row gather), the custom code is kept minimal and idiomatic, and the
documents note where a later `core:`/third-party tool could take over.

## 2. Type system

### 2.1 Types

A column's type is an Odin `typeid` — the physical type is the type. There is
no separate logical-type enum in the MVP. `Schema`/`Field` (used by the
DataFrame and I/O) carry the column name and a `typeid` directly; the
constructor establishes the pairing, and it is invariant afterwards
(principle 6: never silently convert).

When logical types arrive that do **not** map 1:1 onto a physical type
(`Date`, `Datetime`, `Time`, `Duration`, `Categorical`, `Enum`, `List`,
`Array`, `Struct`, `Null` — Stage 14), a `DType` logical layer is introduced
then, Polars-style, mapping logical types onto physical buffers. Building it
now would be speculative (principle 10).

### 2.2 Physical storage

The canonical physical buffer is a **type-tagged raw element buffer** that
supports **any Odin type** (including arbitrary user structs):

```odin
Column :: struct {
    name:      string,
    dtype:     typeid,   // logical + physical type of the elements
    elem_size: int,      // size_of(dtype)
    align:     int,      // align_of(dtype)
    count:     int,      // number of rows
    data:      rawptr,   // element buffer, owned; nil when count == 0
    valid:     []u64,    // packed bitmap; nil == all valid
    alloc:     mem.Allocator,
}
```

`dtype` is an Odin `typeid`, so columns are not restricted to a fixed type
set; the strict runtime equality check `col.dtype == typeid_of(T)` in the
typed accessors preserves principle 6 (no silent conversion). `Schema`/`Field`
carry these `typeid`s directly (see §2.1).

### 2.3 Why this representation (research result)

We verified the following mechanisms against `odin dev-2026-07`:

1. **`typeid` + `rawptr` element buffer (chosen)** — smallest code footprint
   (no per-type constructors/accessors; every op is one generic proc and the
   copy/destroy paths are a single `mem.copy` / `free_with_size`), and it
   supports *any* Odin type. Strict `dtype == typeid_of(T)` equality checks
   in the typed accessors are not silent conversions. Kernels obtain a typed
   view via `transmute([]T)(Raw_Slice{...})` — one cast, no per-element
   boxing. Cost: runtime `typeid` comparison instead of a compile-time enum
   switch, and no exhaustive enumeration of column types (later-stage kernels
   dispatch on a per-column `typeid` switch over the types they support).
2. **Tagged union of slices** — works cleanly and gives exhaustive
   compile-time dispatch, but it fixes the type set at 12 members, forces a
   per-type constructor/accessor/wrapper per type, and copies/destroys through
   a 12-case switch. Kept as a reference, not the primary model.
3. **`any` (rawptr + type_info)** — works for generic access, but the stored
   `type_info` has no stable identity for equality comparisons and dispatch
   goes through reflection; rejected.
4. **Generic `Column(T)`** — excellent for typed kernels internally, but it
   cannot live directly in a heterogeneous `DataFrame`; the union/any wrapper
   problem reappears. Used inside accessors via the `$T` polymorphic params.

Compiler bug note: `#partial switch` over a `Type_Info.variant` union crashes
`dev-2026-07` (llvm_backend assertion `TypeSwitch_Invalid`). Reflection is
therefore only acceptable on non-hot, optional paths until the bug is fixed.
The chosen design does not need it.

Toolchain note (dev-2026-07): `#partial switch` over an *enum* with grouped
case labels (`case .I8, .I16:`) is fine, but a grouped case followed by an
empty `case:` default triggers a false exhaustiveness error. Prefer
`#partial switch` + trailing `return false` for "does this belong to a
subset" helpers.

## 3. Nullable values

Follow Plan.md §4. Use an explicit **validity array** (`[]bool`), never
sentinel values (0 is not NULL).

- `valid == nil` means "all rows valid" (the common, allocation-free case).
- NULL/zero/empty-string must remain distinct (principle 7).
- `NaN` is a *value*, not NULL. `is_nan` and `is_null` are different.
- Sorting defines a total order (Stage 5): NULLs sort after all values by
  default (`nulls_first` selects before); ties (equal keys, including rows
  NULL in the same keys) fall through to the next key and a full tie keeps
  source order (stable). Floats use an IEEE-754 total-order bit key
  (‑0.0 == +0.0; NaN sorts after +inf).
- Aggregations skip invalid rows; `count` counts valid rows by default.
- A `bit_array` (available in `core:container/bit_array`) is a documented
  later optimization for the validity buffer, not an MVP requirement
  (principle 10).

## 4. Core data model

```
DataFrame
    ├── columns: [dynamic]Column   // owned by value, in insertion order
    └── alloc: mem.Allocator       // captured for the dynamic array

Schema :: struct { fields: []Field }
Field :: struct { name: string, dtype: typeid }

Series = single-column DataFrame view (Polars Series analog).
```

- `schema` and `num_rows` are **derived**, never stored: each `Column` already
  owns its name/dtype, so a stored `Schema` would be a second source of truth
  that can drift. `dataframe_schema(df)` materializes an owned `Schema` copy
  from the columns; `dataframe_num_rows` reads `columns[0].count`. Empty
  DataFrame ⇒ 0 rows / 0 cols.
- Column order is preserved; names are unique (duplicate names are an error).
- `DataFrame` owns its columns; `Column` owns its value buffer and validity.
- All buffers are created with an explicit `context.allocator` captured at
  construction time and stored on the struct for symmetric destroy.
- Ownership contract: `dataframe_destroy` frees everything; zero-copy views
  borrow and document their borrow.

### 4.1 Column ownership transfer

Odin has no move semantics, so ownership transfer into a DataFrame is
**explicit and zeroing**: `dataframe_add_column(df, &col)` and
`dataframe_from_columns(allocator, []^Column{...})` take `^Column` pointers,
run all validation first, and on success copy the struct into the DataFrame
and **zero the caller's `Column`** (`col^ = {}`). After the call the caller's
struct is empty — `column_destroy(&col)` is a safe no-op, and double-transfer
fails loudly with `.Column_Name_Empty`/`.Duplicate_Column_Name`. On any
validation error nothing is consumed; on allocation failure the partially
built DataFrame is destroyed and the error returned. Callers who want to keep
their column deep-copy it first (`column_copy`) or use `dataframe_copy`.

### 4.2 Borrowed accessors

`dataframe_get_column(df, name)` and `dataframe_column_at(df, i)` return a
`^Column` that **borrows** the DataFrame — the caller must not destroy it and
must not outlive the `DataFrame` with it. `Schema`'s field names likewise
borrow the columns' name strings; `schema_destroy` frees only the field slice.

### 4.3 Copy and views

`dataframe_copy(df, allocator)` is deep and independent. Row-subset ops
(`head`, `tail`, `take`, `slice`) land in Stage 4 and materialize new columns
from row indices rather than borrowing, keeping the single-owner model;
an explicit borrow-view type is only introduced if a benchmark justifies it
(principle 10).

## 5. Operations over row indices

Operations primarily manipulate **column buffers + row-index lists**, never
construct per-row objects on hot paths (Plan.md §15).

- `filter`/`take` produce a new `[]int` index array or a new set of column
  buffers (kernel decides; filters can materialize directly into buffers).
- `sort` produces a permutation; physical reordering is a separate explicit op.

## 6. Expression system

Expressions are a first-class architectural pillar (Plan.md §16), introduced
early but kept simple initially.

The expression tree lives in package `dataframe/expr`; **evaluation and type
checking live in the `dataframe` package** (`expr_eval.odin`,
`expr_typecheck.odin`) because they operate on `Column`/`DataFrame`. Odin
imports are one-directional (`dataframe -> expr`, never the reverse), so the
tree package is dependency-free and evaluation sits with the types it needs.

### 6.1 Nodes

```odin
Expr :: union {
    Col,       // column reference by name (borrowed)
    Lit,       // typed constant, stored inline (no heap)
    Binary,    // op(lhs, rhs): + - * / % == != < <= > >= and or
    Unary,     // -x, !x
    Cast,      // explicit numeric type conversion
    Alias,     // name the result
    Not_Null,  // is_not_null(x)
    Agg,       // reduce a child expression to a single row (§6.6)
    Cov,       // sample covariance of two numeric columns (single row)
    Corr,      // Pearson correlation of two numeric columns (single row)
}
```

Constructors: `col(ctx, name)`, `lit(ctx, value)`, `add/sub/mul/div/mod`,
`eq/ne/lt/le/gt/ge`, `and_/or_`, `neg/not_`, `cast_`, `alias`,
`is_not_null`, and the aggregation constructors in §6.6.

### 6.2 Ownership — arena

`Expr` is a recursive tree; interior nodes (`Binary`, `Unary`, `Cast`,
`Alias`, `Not_Null`) hold `^Expr` children, so every node is heap-allocated
from an `expr.Context` arena:

```odin
ctx := expr.context_create(context.allocator)
defer expr.context_destroy(&ctx)
e := expr.add(&ctx, expr.col(&ctx, "age"), expr.lit(&ctx, 30))
```

`context_destroy` frees every node at once — no per-node ownership, no
double-free risk. A `^Expr` borrows its context: it must not be used after
`context_destroy` and cannot move between contexts. `Col`/`Lit` leaves carry
no heap of their own; `Lit` stores its value inline in a 16-byte buffer
(`#assert(size_of(T) <= 16)` — all scalars and `string` headers fit), so
literals need no allocation, no destroy, and no reflection.

### 6.3 Literal coercion

`lit(30)` infers `int` from the untyped constant (Odin inference), which
almost never matches a column's exact `typeid`. Binary ops therefore allow
exactly one implicit conversion, and only for a `Lit` operand: when both
sides are numeric, the literal is converted to the other operand's type
(`T(value)`, e.g. `int -> i32`). This is a *constant*, not column data, and
is the only implicit conversion in the library (principle 6 applies to data).

### 6.4 Evaluation

`expr_eval(allocator, df, e) -> (Column, Error)` walks the tree:

- `Col` → deep copy of the source column (result is owned by the caller).
- `Lit` → constant column of `df`'s row count, all rows valid.
- `Binary`/`Unary`/`Cast`/`Not_Null` → typed kernel over the child columns.
- `Alias` → the child result, renamed.

Kernels require exact `typeid` equality between the two operand columns
(principle 6), except literal coercion (§6.3). Type matrix: numeric
(`i8..i64`, `u8..u64`, `int`, `uint`, `f32`, `f64`), `bool`, `string`.

- Arithmetic (`+ - * / %`): numeric only. Integer division truncates toward
  zero; `%` is the truncated remainder. Integer division/modulo by zero
  yields NULL; float division follows IEEE (inf/NaN).
- Comparison (`== != < <= > >=`): any same-type pair; strings compare
  lexicographically.
- Logical (`and or`): bool only. Negation: numeric; `not`: bool.
- NULL propagation is strict: a result row is NULL if either operand row is
  NULL (no Kleene logic in the MVP).

### 6.5 Type checking

`expr_typecheck(df, e) -> (typeid, Error)` infers the result type without
evaluating, rejecting bad expressions early (missing column, operand type
mismatch, operation not supported for a type, invalid cast). Eval performs
the same checks inline; typecheck is the public, pre-evaluation gate (used by
`select`/`filter` in later stages).

### 6.6 Aggregations (Stage 6)

`Agg` reduces a child expression (a column, or any expression) to a **single
row**. Kernels are the reuse point for Stage 7 `group_by` (they consume a
`Column` and will gain row-index subsets there). Results are new single-row
columns; use `alias` to name them for `select`.

```odin
Agg_Kind :: enum byte {
    Count,    // number of valid rows            -> i64
    N_Unique, // distinct valid values           -> i64
    Sum,      // sum of valid rows               -> f64
    Mean,     // arithmetic mean                 -> f64
    Var,      // sample variance, n-1            -> f64
    Std,      // sample standard deviation       -> f64
    Median,   // 0.5 quantile, linear interp     -> f64
    Quantile, // q quantile, linear interp       -> f64
    Min,      // minimum (value type preserved)
    Max,      // maximum (value type preserved)
    Product,  // product of valid rows           -> f64
    Mode,     // most frequent value (ties: first seen; type preserved)
    First,    // first valid value (type preserved)
    Last,     // last valid value (type preserved)
    Skew,     // sample skewness, G1             -> f64
    Kurtosis, // sample excess kurtosis, G2      -> f64
}
Agg :: struct { kind: Agg_Kind, expr: ^Expr, q: f64 } // q: Quantile only
```

Constructors: `count_`, `sum_`, `mean_`, `min_`, `max_`, `var_`, `std_`,
`median_`, `quantile_(ctx, e, q)`, `n_unique_`, `mode_`, `product_`,
`first_`, `last_`, `skew_`, `kurtosis_`, and two-column `cov_`, `corr_`.

**NULL semantics (S6.3):** every aggregation skips invalid (NULL) rows.
`count`/`n_unique` always return a valid row (`0` when nothing is valid);
every value-based aggregation returns a NULL row when there are no valid
rows (visible through `expr_eval`; the scalar convenience API in
`dataframe/stats.odin` reports it as the `.Null_Value` error). `count` works
on any column type, including struct columns (it reads validity only).

**Type matrix (S6.4):** `Sum`/`Mean`/`Var`/`Std`/`Median`/`Quantile`/
`Product`/`Skew`/`Kurtosis` accept numeric columns only and return `f64`
(single-row, no overflow). `Min`/`Max` accept numeric and `string` columns
(ordering types, matching binary `<`), preserving the input type. `Mode`,
`N_Unique`, `First`, `Last` accept numeric, `bool`, and `string` (First/Last
only need validity, so any type works); non-numeric use elsewhere is
`.Unsupported_Operation`.

**Definitions:** `median` is `quantile(0.5)`; quantile uses linear
interpolation between the two closest ranks (`h = q·(n-1)`, polars/numpy
`linear` method). `var`/`std` are sample statistics (`n-1` denominator;
single valid row gives NaN). `skew` is the sample skewness G1
(`(Σ(x−x̄)³/Σ(x−x̄)²^1.5)·√(n(n−1))/(n−2)`, pandas default, NaN for n<3);
`kurtosis` is the sample excess kurtosis G2 (pandas default, NaN for n<4).
`cov`/`corr` require equal-length numeric columns, use rows valid in **both**
columns, and are sample covariance / Pearson correlation (corr for constant
input is NaN).

**Ownership:** the result column is caller-owned (`column_destroy`).
`median`/`quantile`/`n_unique`/`mode` allocate temporary buffers from the
caller's allocator and free them before returning.

## 7. Lazy engine

Data flow (Polars-style):

```
DataFrame / scan_*  ->  LazyFrame
LazyFrame: holds Logical_Plan (no execution)
  -> Optimizer rewrites Logical_Plan (Stage 12)
  -> Physical_Plan is built from Logical_Plan
  -> Executor runs Physical_Plan (may be parallel)
  -> DataFrame
```

Implemented in Stage 11 (`src/dataframe/lazy/`):

### 7.1 LazyFrame and builders

`LazyFrame` wraps a `Logical_Plan` tree; each builder (`lazy.filter`,
`lazy.select`, `lazy.sort`, `lazy.group_by`/`lazy.agg`, `lazy.limit`,
`lazy.slice`, `lazy.join`) appends one node. Odin has no method-call sugar,
so the idiomatic chain rebinds one variable:

```odin
lf := lazy.scan_csv("sales.csv")          // or lazy.scan_dataframe(&df)
lf  = lazy.filter(lf, expr.ge(&ctx, expr.col(&ctx, "price"), expr.lit(&ctx, 100)))
lf  = lazy.sort(lf, []dataframe.Sort_Key{dataframe.sort_key("price", .Desc)})
lf  = lazy.limit(lf, 10)
out := lazy.collect(lf) or_return
lazy.destroy(lf)
```

### 7.2 Ownership

- A `LazyFrame` owns one arena holding its plan nodes and node slices;
  `destroy` frees the whole arena. Linear builders share their input's arena,
  so destroy the chain's final frame exactly once (the rebind style above).
  `lazy.join` is the exception: it clones both child trees into a fresh arena,
  so the joined frame is independent and left/right/join are each destroyed
  once.
- expr nodes (`^expr.Expr`) and `scan_dataframe` sources are **borrowed**:
  keep their expr contexts (and scanned frames) alive until `collect` returns.
- `collect` does not consume the plan; it can be called repeatedly.
- `collect` always returns an owned DataFrame (a bare `scan_dataframe` root is
  deep-copied).

### 7.3 Physical operators

In Stage 11 the executor is eager-backed (S11.3): every logical node compiles
to exactly one eager-engine call (`dataframe_filter`, `dataframe_select`, …)
that materializes the node's output. The operator procs live in
`lazy/physical.odin`; Stage 15 can swap them for parallel kernels without
touching the logical layer.

### 7.4 Optimizer

`lazy/optimizer.odin` rewrites the logical plan in place at the start of
`collect` (Stage 12). S12.1 projection pushdown: a `Scan_CSV` node is pruned
to the columns its ancestors actually reference, so the CSV reader
(`dataframe_read_csv_with_columns`) materializes and type-checks only those
fields. Pass-through nodes (filter/sort/limit/slice) keep every child column
so no pruning happens without a projection above the scan; join is a barrier
(it cannot resolve which side owns a required name without schemas). The
rewrite is idempotent and column-count-preserving, so plan-before equals
plan-after on data (S12.6). S12.7 measures the scan-cost win on a 20-column
CSV: pruning a 2-column projection makes collect ~2.6x faster.

### 7.5 Executor

`lazy/executor.odin` runs the optimizer over the logical plan, then walks it
bottom-up (`exec_node`), feeding each operator's owned output into the next.
`collect` is the only entry point. Plan building is side-effect free (S11.5):
no I/O, no evaluation, no validation — missing files, unknown columns, type
mismatches, and malformed plans all surface as `Error` values from `collect`.

`Logical_Plan` is a union of plan nodes:
`Scan_CSV, Scan_DF, Filter, Projection, Sort, Group_By, Limit, Slice, Join`
(grows with the roadmap). `Physical_Plan` mirrors logical nodes as executable
operators that stream columnar batches.

## 8. Optimizer

Stage 8 milestone. Initial rule set (each rule = one pass over the plan):
1. Projection pushdown — prune columns read from scans and intermediate nodes.
2. Predicate pushdown — apply filters as early as possible (before joins,
   before materialization).
3. Constant folding — evaluate literal-only sub-expressions at plan time.
4. Common subexpression elimination — reuse identical computed columns.
5. (Later) limit pushdown, sort/limit fusion, join reordering.

Rules must be independently testable and each must preserve plan equivalence.

## 9. GroupBy

Odin has no chaining, so the API is explicit two-step (as opposed to the
Polars-style `group_by(df, exprs).agg(aggs)` in the original plan):

```odin
gb, err := dataframe.group_by(df, allocator, keys)   // Group_By
out, err := dataframe.group_by_agg(&gb, allocator, aggs)
dataframe.group_by_destroy(&gb)                        // df owned by caller
```

`Group_By` borrows `df` (no copy); `dataframe_group_by_destroy` frees the
group index structures and must run before `dataframe_destroy(&df)`.
Aggregation results are owned by the caller, as usual.

- Keys and aggs are `^expr.Expr` lists. Keys may be any expression type
  (columns, literals, computed exprs); aggs must be `Agg` nodes (count/sum/
  mean/min/max/var/std/median/quantile/n_unique/first/last — §6.6). Any
  non-Agg expression in the agg list is `.Unsupported_Operation`.
- Result naming: an agg or key named by its `Alias`, else by its `Col` child
  name. Aggregating a computed expression therefore requires an `Alias`.
  Unnamed results and duplicate result names (including key/agg collisions)
  are errors (`.Invalid_Argument` / `.Duplicate_Column_Name`).
- Grouping is hash-based: each row's key values are encoded (`encode_row`,
  reused from partition) into a string map key, so NULL keys form their own
  group and multi-column keys work. First-seen order of groups is preserved.
- Aggregation runs the Stage 6 kernels over each group's row-index subset
  (`rows []int`, nil = whole column — §6). Values inside a group follow the
  Stage 6 NULL rules: kernels skip NULLs, an all-NULL group yields NULL for
  value aggs and 0 for count/n_unique, and an empty `df` produces a 0-row
  result with the correct schema.
- Since Stage 21 the grouping loop dispatches to parallel partitioned hash
  grouping at >= 100K rows (§14.7); below the threshold the original
  single-threaded loop runs. Semantics are unchanged and pinned by
  equivalence tests.

## 10. Joins

Keys are column names (`[]string` on each side), resolved by
`resolve_key_columns` (the same helper unique/group_by use). The two key
lists must be non-empty and equally long; key dtypes must match pairwise
(`.Type_Mismatch` otherwise — never a silent conversion, principle 6).
Every keyed join is a hash join: the probed side's keys are encoded with
`encode_row` (group_keys.odin, same canonical encoding as group_by/unique),
so multi-column keys and any column type work for free.

### NULL keys (SQL semantics)

A NULL in any key column **never matches** — not even another NULL. This is
the SQL convention and the polars default (`join_nulls=false`); it is what
makes joins different from group_by/unique, where NULL is a group key value.
NULL-keyed rows therefore appear only as unmatched rows (left/right/full) or
are dropped (inner).

### Output schema and name collisions (S8.5)

- inner/left/right/full: **all left columns**, then the right columns that
  are **not** right-key columns (the key appears once, coalesced from the left
  side). For rows that exist only on the right (right/full joins), the left
  key column is filled from the right key value (polars coalesce behavior).
- semi/anti: **left columns only**.
- cross: all left columns then all right columns.

A right column whose name already exists in the output gets the suffix
`_right` (polars default); if the suffixed name still collides, `_right` is
appended again until the name is unique. Right key columns are never emitted,
so they never collide.

### Row order (deterministic)

- inner: left-major — each left row in order, its right matches in right
  order.
- left: all left rows in order; unmatched rows carry NULL right columns.
- right: right-major — each right row in order, its left matches in left
  order; unmatched rows carry NULL left columns.
- full: left-major matched + unmatched-left rows, then unmatched right rows.
- semi/anti: left row order, one row per matching/non-matching left row.
- cross: left-major, every (left, right) pair.

### API

```odin
dataframe_inner_join(left, right: ^DataFrame, allocator, left_keys, right_keys: []string) -> (DataFrame, Error)
dataframe_left_join(...) / dataframe_right_join(...) / dataframe_full_join(...)
dataframe_semi_join(...) / dataframe_anti_join(...)
dataframe_cross_join(left, right: ^DataFrame, allocator) -> (DataFrame, Error)
```

The result owns its columns; both inputs are borrowed. Sort-merge and
broadcast join physical operators can be added behind the same API later
(Plan.md §8).

## 11. I/O

CSV is a separate subsystem (Plan.md §9), built on `core:encoding/csv`
(RFC 4180 reader/writer already in the stdlib — verified present).

- `read_csv(allocator, path, options) -> DataFrame` with type inference
  (sample first N records, then full validation). Inference picks the
  narrowest type that parses every sampled non-NULL value (bool > i64 >
  f64 > string); a post-sample value that fails the sampled type is a
  `.Type_Mismatch`, never a silent widening. Columns with no sampled
  non-NULL value fall back to string.
- `read_csv_with_schema(allocator, path, schema, options)` for explicit
  control: header names must match the schema in order (`.Invalid_Schema`),
  supported dtypes are bool/i64/f64/string.
- `write_csv(df, allocator, path, options)`; NULL rows written as the
  null token, f64 shortest-round-trip, quotes/escapes per RFC 4180.
- `CSV_Options`: `delimiter`, `comment`, `null_token` (default ""), and
  `sample_rows` (default 1000). The default empty token means empty fields
  read as NULL; a non-empty token makes the empty string a real value.
- JSON and NDJSON (S13) are a sibling subsystem built on `core:encoding/json`
  (parser + string escaping; finite f64 values are written shortest-round-trip
  as `%v`, matching CSV). JSON values are self-describing, so there is no
  inference sample: the type of each column is fixed by the first value and
  every later value must match exactly — `.Type_Mismatch`, never a silent
  widening (principle 7, same rule as CSV). `null` never constrains the type.
  Supported column dtypes are bool, i64, f64, string; a nested Object/Array
  value is `.Unsupported_Operation` until List/Struct land (S14.2). Mixed
  Integer/Float in one column is a `.Type_Mismatch` (no int→float widening).
- `read_json(path)` reads a top-level array of objects (records
  orientation). `core:encoding/json` parses objects into an unordered map, so
  column order is **key-alphabetical** (the stdlib's own `unparse` sorts maps
  the same way); a record missing a key is NULL for that cell.
  `read_json_with_schema(path, schema)`: an object key outside the
  schema is `.Invalid_Schema`, a missing key is NULL, and a schema field
  present in no record becomes an all-NULL column. JSON is self-describing, so
  reads take no options (relaxed-JSON/JSON5 dialect knobs are out of scope);
  `write_json(df, path, options)` writes an array of objects, NULL is `null`
  and f64 NaN/±Inf are written verbatim (like CSV) — a documented extension,
  since JSON has no NaN/Inf literal.
- `read_ndjson(path)` / `write_ndjson(df, path)`: one JSON object per line,
  columns are the union of keys in key-alphabetical order. Same type rules as
  JSON.
  Lazy-scan integration (a JSON/NDJSON `Scan` node feeding S12.1 pushdown) is
  deferred; S13 I/O is eager-only.
- Arrow IPC (S13.3/S17): vendored OdinArrow (`src/odinarrow/`) provides
  `ipc_write_file` / `ipc_read_file`; bridge in `dataframe/arrow.odin`
  converts between the runtime-typed `Column` and Arrow's `Array` buffers.
  Supported: i8–u64, f32/f64, bool (bit-packed), string (offsets+data).
  Parquet remains deferred (requires a Parquet code generator).

## 12. Errors

A single `Error` enum for public APIs, returned as the **last** result
(`value, err := ...`). The first enum value is `.None` (no error), so every
erroring proc is immediately usable with Odin's native `or_return` /
`or_else` / `or_continue` operators — callers never need to unroll
`if err != .None`:

```odin
v, ok := column_get(&col, i, i64) or_return   // (value, ok, err)
col := column_from(allocator, "x", xs) or_return
```

`runtime.Allocator_Error` from `mem.*` is mapped to `.Allocator_Failure` once
at the allocation boundary so the two error types never mix. No exceptions;
no silent fallbacks.

## 13. Memory and ownership

- Everything is allocated with `context.allocator` unless stated otherwise;
  structs that own memory capture the allocator in a field for symmetric
  destroy.
- Pair `create`/`destroy` procs (`column_create`, `column_destroy`,
  `dataframe_create`, `dataframe_destroy`).
- Copy semantics: `column_copy`, `dataframe_copy` are deep by default; views
  are explicit and borrow. Construction transfers column ownership into the
  DataFrame by zeroing the source (DESIGN.md §4.1).
- `Schema` owns only its field slice; field names borrow the columns'
  strings.
- The test runner already tracks memory leaks per test (verified: `odin test`
  reports bad frees / leaks), so ownership bugs surface in CI.
- **Arena allocation (S15.6)**: `OpArena` (op_arena.odin) wraps
  `mem.Dynamic_Arena` for per-operator bump allocation. Intermediate columns
  in expression evaluation (Binary/Unary/Cast/Not_Null children) are
  arena-allocated; final results use the caller's allocator. Individual frees
  (`column_destroy`, `delete`) are no-ops on the arena; bulk-free via
  `dynamic_arena_destroy` at the end of each operator. Top-level callers
  (`dataframe_select`, `dataframe_filter`, `dataframe_group_by`, etc.)
  create and destroy one arena per call.

## 14. Performance

Follow principles 9–10: benchmarks before optimization, never optimize without
a measured result. Planned benchmark matrix (1K / 100K / 1M / 10M rows):

- CSV parse, filter, sort, groupby, aggregation, join, column create
- Compare against naive per-row/boxed implementations to justify the design.

### 14.1 CSV parse baseline (S9.7)

Run: `odin run benchmarks/csv.odin -file`. Single-threaded stdlib
(`core:encoding/csv`) reader, 4-column synthetic data (`id,score,ok,name`)
with ~1% NULLs:

| rows    | time    | rows/s   | MB/s |
|---------|---------|----------|------|
| 1K      | ~1.1 ms | ~0.93 M/s| ~27  |
| 100K    | ~98 ms  | ~1.02 M/s| ~32  |
| 1M      | ~1.04 s | ~0.96 M/s| ~31  |

Best-of-3 runs, `odin run benchmarks/csv.odin -file`. The parse cost is
dominated by the stdlib record scanner; per-field typed parsing and column
building add ~10%. This is the sequential baseline; S15.7 parallelizes the
hot record-to-column loop for files >= 1 MB (see §14.6).

### 14.2 Validity bitmap vs `[]bool` (S15.1)

Run: `odin run benchmarks/validity.odin -file`. Compares the current packed
`[]u64` bitmap against the old `[]bool` on sum-scan, gather (permuted access),
and scatter-set, at 1% and 50% NULL rates:

| workload       | 10M rows, 1% NULL  | 10M rows, 50% NULL |
|----------------|--------------------|--------------------|
| sum-scan       | bool=51.9, bm=49.6 (-4.5%) | bool=51.8, bm=48.1 (-7.2%) |
| gather         | bool=59.1, bm=53.5 (-9.5%) | bool=59.6, bm=53.2 (-10.7%) |
| scatter-set    | bool=0.59, bm=0.24 (2.5x)  | — |

Memory: 87.5% reduction (10 MB → 1.25 MB for 10M rows).

Best-of-3 runs in milliseconds. Decision: adopt packed bitmap. Gains are
modest at 1M rows (~1 ms) but consistent at 10M rows; memory savings are the
primary win for large columns. The `nil == all valid` invariant is preserved,
so fully-valid columns have zero overhead.

### 14.3 All-valid fast paths for numeric kernels (S15.2)

Run: `odin run benchmarks/kernels.odin -file`. Adds an all-valid fast path to
the elementwise and aggregation kernels: when `Column.valid == nil` the
validity branch is hoisted out of the inner loop. Measured on 1M / 10M rows
with 0% and 1% NULL rates (best-of-3):

| operation | 10M, 0% NULL (before → after) | 10M, 1% NULL (before → after) |
|-----------|--------------------------------|-------------------------------|
| sum       | 55.5 ms → 34.0 ms (**-38%**)   | 75.4 ms → 74.1 ms (-2%)       |
| add       | 176.7 ms → 118.1 ms (**-33%**) | 219.6 ms → 207.1 ms (-6%)     |
| neg       | 96.0 ms → 67.9 ms (**-29%**)   | 116.7 ms → 100.1 ms (-14%)    |
| abs       | 89.2 ms → 69.3 ms (**-22%**)   | 150.1 ms → 123.0 ms (-18%)    |

Functions optimized: `binary_arith`, `binary_cmp`, `binary_bool`,
`neg_typed`, `func_abs_typed`, `func_sign_typed`, `func_round_typed`,
`numeric_reduce_typed` (sum/product). The 1% NULL path also benefits (6-18%)
from the branch-predictor split. The fast path is the first branch in each
kernel; the original validity-aware path handles all other cases.

### 14.4 SIMD kernels via `core:simd` (S15.5)

Run: `odin run benchmarks/kernels.odin -file`. Adds explicit SIMD
implementations to the all-valid fast paths using `core:simd` intrinsics,
guarded by `when runtime.HAS_HARDWARE_SIMD`. Lane counts target 256-bit
vectors (4 lanes for 8-byte types). Load/store uses `transmute` +
`mem.copy` through stack-aligned buffers.

| operation | S15.2 10M 0% NULL | S15.5 10M 0% NULL | Change |
|-----------|-------------------|-------------------|--------|
| sum       | 34.0 ms           | 11.1 ms           | **-67%** |
| add       | 118.1 ms          | 107.3 ms          | -9%    |
| neg       | 67.9 ms           | 68.9 ms           | ~0%    |
| abs       | 69.3 ms           | 66.6 ms           | -4%    |

The sum reduction benefits most (3x) because `simd.reduce_add_pairs`
accumulates 4 elements per iteration and reduces horizontally, which the
compiler's auto-vectorizer does not match. Elementwise ops get modest gains
because the compiler already auto-vectorizes the simple scalar loops, and
the `mem.copy` load/store overhead partially offsets the SIMD benefit.

SIMD kernels: `simd_kernels.odin` — `simd_neg`, `simd_abs`, `simd_add`,
`simd_sub`, `simd_mul`, `simd_sum` (f64 only). Integrated into
`binary_arith`, `neg_typed`, `func_abs_typed`, `numeric_reduce_typed`.
The 1% NULL paths are unchanged (they stay on the scalar validity-aware
path).

Optimization reserve (do NOT implement speculatively):
specialized numeric kernels (all-valid fast paths adopted, S15.2), hash joins,
parallel operators, SIMD (`core:simd` adopted S15.5), memory reuse (arena adopted
S15.6), zero-copy ops. Validity bitmap (S15.1) is adopted: `Column.valid` is now a packed
`[]u64` bitmap; see §14 benchmarks.

### 14.5 Per-operator arena allocation (S15.6)

Run: `odin run benchmarks/arena.odin -file`. `OpArena` (op_arena.odin) wraps
`mem.Dynamic_Arena` to provide bump-pointer allocation for intermediate columns
during expression evaluation. Individual `column_destroy`/`delete` calls are
no-ops on the arena; bulk-free happens in `dynamic_arena_destroy` at operator
exit. The result column is always allocated with the caller's allocator,
preserving ownership semantics.

Integrated into `expr_eval` (Binary/Unary/Cast/Not_Null children) and all
top-level callers (`dataframe_select`, `dataframe_filter`, `dataframe_group_by`,
`dataframe_sort`, `partition_by`, `groupby_dynamic`, lazy engine).

| workload | 10K rows | 100K rows | 1M rows |
|----------|----------|-----------|---------|
| chained (a + b*2 - c + 1) | 0.2 ms | 2.2 ms | 19.2 ms |
| filter ((value > 0.5) & (value < 0.8)) | 0.3 ms | 2.5 ms | 26.8 ms |
| select (3 computed columns) | 0.2 ms | 1.2 ms | 17.0 ms |

### 14.6 Parallel CSV parse (S15.7)

Run: `odin run benchmarks/csv.odin -file`. Parallel chunked CSV parsing using
`src/parallel` pool (8 threads). Files >= 1 MB are split into record-aligned
chunks; each thread gets its own `csv.Reader` and fills independent
`CSV_Column_Buffer`s; the main thread merges the results. Comment mode
(`options.comment != 0`) falls back to sequential. Type inference and header
parsing stay sequential.

| rows    | time    | rows/s   | MB/s | vs sequential |
|---------|---------|----------|------|---------------|
| 1K      | ~1.0 ms | ~0.98 M/s| ~29  | ~1x (below threshold) |
| 100K    | ~67 ms  | ~1.49 M/s| ~47  | ~1.5x |
| 1M      | ~668 ms | ~1.50 M/s| ~49  | ~1.5x |
| 10M     | ~7.2 s  | ~1.39 M/s| ~46  | ~1.5x |

Best-of-3 runs. The ~1.5x speedup at 1M+ rows reflects the parallel
record-to-column loop; the merge overhead (memcpy + string re-basing) is
small relative to per-field typed parsing. The stdlib record scanner
(`core:encoding/csv`) stays single-threaded per chunk; further gains would
require a parallel byte-scanner (reserved).

### 14.7 Parallel partitioned hash grouping (S21.1)

`dataframe_group_by`'s cost is the per-row `encode_row` (canonical key
bytes) plus a map probe/insert on one shared map. Above
`PARALLEL_GROUPBY_THRESHOLD` (100K rows) the loop is split into contiguous
row ranges processed in parallel (`src/parallel` pool, 8 threads); each task
builds a thread-local map, first-seen order, and encode buffer from the
caller's allocator. The main thread then merges partitions in ascending
row-range order: because chunk t only contains rows >= every row of chunk
t-1, merging in that order reproduces the sequential semantics bit-for-bit —
groups in first-appearance order, source-order rows within each group.
Merging adopts each chunk's new keys (string + row array move into the
global map) and appends+frees duplicates; nothing is boxed and no locks are
taken during the parallel phase. Below threshold the original loop runs
unchanged as `group_rows_sequential`.

Correctness is pinned by exact sequential-vs-parallel equivalence tests
(`parallel_groupby_test.odin`), including NULL keys and multi-key rows.

| rows | naive(ms) | groupby(ms) | vs sequential groupby |
|------|-----------|-------------|-----------------------|
| 100K | 2.7       | 6.6         | below/at threshold    |
| 1M   | 27.7      | 49.4        | 127 ms → ~2.6x faster |
| 10M  | 251.4     | 400.5       | ~3x faster            |

Run: `odin run benchmarks/groupby.odin -file`. Best-of-N. The remaining gap
to the naive slice-loop reference is `encode_row` itself (per-column tag +
copy per row); dictionary-encoded grouping for low-cardinality keys is the
reserved next lever.

### 14.8 Radix argsort kernel (S21.2)

The legacy sort compares permutation entries through `core:sort`'s
Interface — every comparison pays a closure hop plus a typeid switch plus
validity checks. For the common single-key case the total order can be
computed once per row instead: `order_key_u64` maps any sortable non-string
dtype to a u64 whose unsigned order equals the documented order (signed ints
and distinct-i64 temporals XOR the sign bit, floats use the canonical
IEEE-754 total-order bit key already used by `compare_values`, descending
complements the key). A stable LSD base-256 radix argsort over those keys
then replaces comparisons entirely; digit passes whose byte vanishes in the
cumulative OR mask are skipped (typical integer data sorts in 2–3 passes,
not 8). Each pass runs parallel histogram + scatter tasks with a sequential
prefix-sum combine (T×256 entries), so passes are race-free and stable
across chunk boundaries.

NULLs never compare equal to values, so they are stitched at head/tail in
source order before the valid range is sorted — identical to the legacy
index-tiebreaker outcome for both `nulls_first` settings.

Dispatch lives in `argsort_key_columns`: single key, non-string dtype, not
categorical, >= RADIX_SORT_THRESHOLD (4096) rows. Multi-key and string
sorts keep the legacy path; the oracle tests prove both paths produce the
same permutation at the threshold boundary.

| dtype (1M rows, gather included) | time    | vs legacy comparator path |
|----------------------------------|---------|---------------------------|
| i32                              | 94.4 ms | ~3.2x faster              |
| i64                              | 94.0 ms | ~3x faster                |
| f64                              | 98.8 ms | ~3x faster                |
| string                           | 694.8 ms| unchanged (legacy path)   |

Run: `odin run benchmarks/sort.odin -file`. Best-of-N. Further levers,
reserved until measured: groupwise key extraction, parallel combine, and
GDS-style permutation application during gather.

## 15. API sketch (blessed after Stage 10)

```odin
import "dataframe"
import "dataframe/expr"    // expression tree + arena Context

// column construction (values are copied; ownership moves into the frame)
age, age_err := dataframe.column_from("age", []i32{25, 30, 35})
name, n_err  := dataframe.column_from("name", []string{"ada", "grace", "katherine"})
sal,  s_err  := dataframe.column_from_with_valid("salary", []f64{150000, 0, 95000}, []bool{true, false, true})
df, df_err := dataframe.dataframe_from_columns([]^dataframe.Column{&age, &name, &sal})
defer dataframe.dataframe_destroy(&df)

// expressions are built in a context (arena) — every node is owned by ctx
ctx := expr.context_create(context.allocator)
defer expr.context_destroy(&ctx)
pred := expr.and_(&ctx,
    expr.ge(&ctx, expr.col(&ctx, "age"), expr.lit(&ctx, i32(30))),
    expr.is_not_null(&ctx, expr.col(&ctx, "salary")),
)

// operations return a new owned DataFrame (never mutate the input). The
// short aliases (dataframe.filter == dataframe.dataframe_filter) compose
// with `or_return` into pipelines — Odin has no method-call sugar for
// procs, so this is the idiomatic chain style (S10.2).
filtered := dataframe.filter(&df, pred) or_return
sorted   := dataframe.sort(&filtered, []dataframe.Sort_Key{dataframe.sort_key("age", .Desc)}) or_return
top      := dataframe.head(&sorted, 10) or_return

// group by + aggregate
gb      := dataframe.group_by(&df, []^expr.Expr{expr.col(&ctx, "dept")}) or_return
grouped := dataframe.agg(&gb, []^expr.Expr{expr.mean_(&ctx, expr.col(&ctx, "salary"))}) or_return

// CSV I/O
read := dataframe.dataframe_read_csv("sales.csv") or_return

// display (S10.3) — to_string owns the returned string; print writes stdout
s := dataframe.dataframe_to_string(&df) or_return
dataframe.dataframe_print(&df)
```

### 15.1 Conventions

- `dataframe_*` / `column_*` / `schema_*` procs are the unambiguous full
  names. Short aliases in `chain.odin` (`filter`, `select`, `sort`,
  `sort_by`, `head`, `tail`, `slice`, `take`, `limit`, `with_columns`,
  `drop`, `unique`, `group_by`, `agg`, `partition_by`) bind the blessed
  spellings (`dataframe.filter(...)`) so pipelines read left to right.
- The allocator is always the last, defaulted parameter
  (`allocator := context.allocator`); `Error` is always the last result
  value (§12), so `or_return` / `or_else` apply.
- Operations never mutate their input: `filter`/`select`/`sort`/… return
  new DataFrames the caller owns and destroys.

`dataframe_to_string` renders a polars-style table: a `shape: (rows, cols)`
line, a header row, then each row; columns are left-aligned to their widest
cell (header or value) and separated by two spaces. NULL renders as `null` so
it stays distinct from `0`/`""` (principle 8). The string is owned by the
caller (release with `delete`); `dataframe_print` is a thin stdout wrapper.

## 16. Open decisions to resolve during implementation

- Whether `select` returns a new DataFrame or a lazy view.
- Column count caps: `num_rows` as `int`; large-string columns need a
  dictionary (categorical) path — deferred.

## 17. Polars reference (feature mapping)

Polars is the functional reference for this library (architecturally Polars-
inspired, Pandas-breadth). This section maps the polars 0.55.2 surface to
Thunder's plan and tracks what is covered, deferred, or deliberately dropped.
Version: `docs.rs/polars/0.55.2` (2026-08).

### 17.1 Architectural mapping (validated)

| Polars concept | Thunder equivalent | Where |
|---|---|---|
| `DataFrame = Vec<Series>` (columnar) | `DataFrame { columns: [dynamic]Column }` | §4 |
| `Series` (type-agnostic column) | `Column` (type-tagged buffer + `valid`) | §2.2 |
| `ChunkedArray<T>` (typed heart) | typed kernels + `column_typed_view` | §2.2, §6.4 |
| `Expr = Fn(Series) -> Series`, composable | `Expr` union, eval walks the tree | §6 |
| eager API | `dataframe` package (stages 1–9) | §15 |
| lazy API + optimizer | `lazy/` package (stages 11–12) | §7–8 |
| opt-in dtypes via feature flags | `typeid`-based columns; no flag needed | §2.1 |
| custom allocator (~25% runtime) | explicit `mem.Allocator` throughout | §13–14 |

### 17.2 Feature coverage vs ROADMAP

Already planned (polars name -> Thunder task):
`group_by/agg` S7, `inner/left/right/full/semi/anti/cross_join` S8,
`sort/sort_by` S5, `pivot/melt` S14.1, `explode/unnest` S14.2,
`rank/row_number/cum_sum/rolling/shift` S14.3, `asof_join` S14.4,
`dynamic_group_by` S14.5, Date/Datetime dtypes + `date_range/truncate` S14.6,
`is_null/is_not_null/fill_null/drop_nulls/coalesce/interpolate` S14.7,
`categorical` S14.8, CSV S9, JSON/NDJSON S13, Arrow IPC S17, Parquet (deferred),
`unique/unique_counts` S4.4/S6.1, window/`over` S14.3.

Fold-in tasks (polars features worth explicit tasks):
- S3.x expression library: `abs`, `sign`, `diff`, `pct_change`, `dot_product`,
  `round_series`, `is_between`, `is_in`, `is_first_distinct`/
  `is_last_distinct`, `arange`, `concat_str`, string ops (`str_*`),
  `arg_where`, `search_sorted`.
  (`checked_arithmetic`, `zip_with`, `repeat_by` dropped: Rust-functional
  style; Odin is data-oriented and expresses them with plain loops.)
- S6.x aggregations: `mode`, `product`, `cov`, `corr`, `moment` (kurt/skew).
- S4.x: `partition_by` (split by group keys).
- S14.x: `cumulative_eval`, `ewma`, `list` dtype + `list_*` ops,
  `struct` dtype, `list_to_struct`, `unique_counts` on expressions.

Deliberately dropped for the MVP (revisit only if justified):
- `extract_jsonpath`, `extract_groups`, `find_many`, `regex` column
  selection (string-regex machinery; revisit with a strings milestone).
- `reinterpret` bit-casting (violates principle 6 unless explicitly typed).
- `sql` front end (syntax-to-expr translation, not core).

### 17.3 Performance notes worth adopting

- Polars recommends jemalloc/mimalloc (up to ~25%); Thunder's allocator-aware
  design can adopt a custom allocator without code changes (§14 reserve).
- Polars parallelizes across expressions and within groupby/join; matches
  Thunder's S15.4 parallel executor plan.
- Polars gates SIMD (`simd`, `performant`, `bigidx`) behind features;
  Thunder's S15.5 SIMD kernels should also be compile-time guarded.

## 18. Stage 14 — advanced functionality

Design for ROADMAP Stage 14: reshaping, nested dtypes, window functions,
as-of joins, dynamic/rolling groupby, time series, null handling, and
categorical/enum. Each subsection records the decisions the implementation
must honor (tested in the corresponding `*_test.odin`).

### 18.1 List and Struct dtypes (S14.2, S14.3)

A `List` column stores `List_Ref` elements (`List_Ref :: struct { off, len: int }`),
one per row, indexing into the column's inner element buffer:

- `dtype` = `typeid_of(List_Ref)`, `elem_size` = `size_of(List_Ref)`.
- `data` = the `List_Ref` array (one per row).
- `payload` = the inner element buffer (all rows' elements, concatenated).
  For string elements the inner string contents are appended to the payload
  behind the element bytes and the headers point into that region.
- `inner_dtype` = the element type; `inner_valid` = per-element validity
  (`nil` = all elements valid). A NULL *element* is distinct from a NULL *row*
  (the outer `valid` flag says the row's list is NULL).
- A NULL outer row leaves `List_Ref` zeroed (off/len 0) and is marked invalid.

Ownership: the payload, `inner_valid`, and any inner string blob are owned by
the Column and freed by `column_destroy`. `column_copy` deep-copies them and
re-points inner string headers (mirroring string columns). `gather_rows_core`
copies the payload wholesale (offsets stay valid because the whole buffer is
kept) and copies `inner_valid` wholesale; it does not re-slice.

Builders (all transfer/own their input where stated):
- `list_from_column(name, elems: ^Column, offsets: []int)` — transfers the
  element column into a List column. `offsets` has `len(elems)+1` entries
  (`offsets[0] == 0`, monotone). The element column's data becomes the
  payload; a string element column's own payload (string contents) is merged
  into one blob and the headers re-pointed. `elems` is zeroed on success.
- `list_from_slices(name, values: [][]$T)` / `..._with_valid` — convenience
  wrappers; string contents are borrowed (documented limitation, as in
  `column_from`).

Ops (`list_*.odin`), each returning a caller-owned Column:
- `list_count(col)` — `i64`, element count per row; NULL where the row is NULL.
- `list_get(col, index)` — a column of `inner_dtype`; NULL where the row is
  NULL, the index is out of range, or the element is NULL.
- `list_gather(col, indices)` — same, but the index is per-row.
- `list_unique(col)` — per-row deduplicated list (first-seen order, NULL
  elements skipped); NULL rows stay NULL.
- `list_to_struct(col)` — a DataFrame with one column per element position,
  named `field_0..field_k-1` (k = longest valid list). Rows with fewer
  elements (and NULL rows) get NULL in the missing positions.
- `dataframe_explode(df, col_name)` — one output row per element; a NULL row
  explodes to one all-NULL row. The other columns repeat the outer row.
- `dataframe_unnest(df, col_name)` — splits a struct-valued column into one
  column per field, named `<col>_<field>`. Field layout is read with
  `typeid` reflection (`#partial switch` over `Type_Info`, verified working in
  this Odin build). A NULL struct row yields NULL in every field column.

### 18.2 Window functions (S14.4, S14.5)

A new `expr.Window` node computes a per-partition, order-preserving result
column. `over` partitions by evaluated key expressions (empty = one partition
of all rows); the rows of each partition keep source order.

```odin
Window_Func :: enum byte { Rank, Row_Number, Cum_Sum, Cum_Min, Cum_Max,
                           Shift, Rolling, Cumulative_Eval, Ewma }
Rank_Method :: enum byte { Average, Min, Dense, Max }   // Rank ties
Window :: struct { func: Window_Func, expr: ^Expr, over: []^Expr,
                   n: i64, agg: Agg_Kind, alpha: f64, method: Rank_Method }
```

Constructors: `window_rank(ctx, e, method, over)`, `window_row_number(ctx, over)`,
`window_cum_sum/cum_min/cum_max(ctx, e, over)`, `window_shift(ctx, e, n, over)`,
`window_rolling(ctx, e, window, agg, over)`,
`window_cumulative_eval(ctx, e, agg, over)`, `window_ewma(ctx, e, alpha, over)`.
The `over` slice is copied into the context arena (as `Concat_Str.parts`).

Semantics (each partition independent; NULL rows keep NULL output where noted):
- `Row_Number` — 1-based source position; `i64`, always valid. Requires Alias.
- `Rank` — `f64`; non-NULL rows ranked by value with the given tie method
  (Average/Min/Max/Dense; Average is polars' default). NULL rows are not
  ranked and yield NULL.
- `Cum_Sum` / `Cum_Min` / `Cum_Max` — running aggregate over the partition in
  source order; output dtype = child dtype; NULL input rows yield NULL output
  and the running aggregate continues (matches the `Func.Cum_Sum` kernel).
- `Shift` — `out[i] = in[i-n]` within the partition (positive n moves values
  down; negative moves up); out-of-partition positions are NULL.
- `Rolling` — for row i, aggregate `agg` over the trailing window
  `[i-n+1, i]` (n ≥ 1) using the Stage 6 kernels
  (`run_group_agg` over the window's row indices). Result dtype follows
  `agg_result_dtype`; `count` is always valid (0 for an empty window), value
  aggs are NULL when the window has no valid rows.
- `Cumulative_Eval` — like Rolling with a growing window `[0, i]`.
- `Ewma` — `y_i = alpha·x_i + (1-alpha)·y_{i-1}` per partition; the first
  valid row seeds `y = x`. NULL input rows yield NULL output and the recursion
  continues from the last computed value. Result is `f64`.

Result naming follows the engine: the child column's name, else "" (so
`row_number` must be aliased). Type checking mirrors `Func`/`Agg` rules.

### 18.3 As-of joins (S14.6)

`dataframe_asof_join(left, right, left_on, right_on, left_by, right_by,
strategy, allocator)`. `strategy` is `Asof_Strategy { Backward, Forward }`.

- Precondition (documented, not validated): the right side is sorted ascending
  by `(by, on)`.
- A NULL `on`/`by` value on the left never matches (SQL semantics, as in
  `join.odin`). `by` columns, when given, must match exactly (NULL never
  matches); the `on` column selects the greatest right `on` <= left `on`
  (Backward) or the smallest right `on` >= left `on` (Forward). Right rows with
  NULL `on`/`by` are excluded from matching.
- Output is left-major like a left join: all left columns, then the right
  columns that are neither `by` nor `on` columns, with the join.odin `_right`
  collision-suffix rule. Unmatched left rows carry NULL right columns.
- Implementation: bucket the right rows by `by` (`encode_row`); within each
  bucket the rows are already `on`-sorted, so each left row is matched by
  binary search.

### 18.4 Dynamic and rolling groupby (S14.7)

`Closed_Interval :: enum byte { Both, Left, Right, None }` names the window
endpoint convention (polars `Closed_Window`).

- `dataframe_group_by_dynamic(df, time_col, by, every, offset, closed)` returns
  a `Dynamic_Group_By` (owned; `dataframe_dynamic_group_by_destroy`). The
  time column must be `Datetime`. Rows are grouped by `by` (optional) and a
  half-open window `[start, start+every)` where
  `start = floor((t − offset)/every)·every + offset` (floor division, so
  pre-epoch times work). Groups are ordered by (window start, by-group
  first-seen). `dataframe_dynamic_group_by_agg` returns one row per window:
  the by columns, the window-start `Datetime` column (named `time_col`), then
  the aggregation results (Stage 6 kernels over each window's row indices).
- `dataframe_group_by_rolling(df, time_col, by, period, offset, closed)`
  returns a `Rolling_Group_By` with one *trailing* window per row:
  `[t_i − period − offset, t_i − offset]` (closed per `closed`). Its agg
  returns one row per source row (source order): the by columns, the time
  column, then the aggregation results. Windows are computed per by-group with
  the group's rows sorted by time.

### 18.5 Time series dtypes (S14.8)

Four distinct `i64` types (storage = `i64`, so `encode_row` and byte-copy
kernels work unchanged):

```odin
Date     :: distinct i64 // days since 1970-01-01 (proleptic Gregorian)
Datetime :: distinct i64 // microseconds since 1970-01-01T00:00:00
Time     :: distinct i64 // microseconds since midnight
Duration :: distinct i64 // signed microseconds
```

- Calendar math uses Howard Hinnant's `days_from_civil`/`civil_from_days`
  (`calendar.odin`); `date_create(y,m,d)`, `datetime_create(...)`,
  `time_create(h,m,s,us)` validate ranges.
- Conversions/arithmetic: `date_to_datetime`, `datetime_to_date`,
  `datetime_add(dt, Duration)`, `datetime_diff(a, b)`, `datetime_time_of_day`,
  `duration_from_days/hours/minutes/seconds`.
- Accessors (`dt_year/month/day/weekday/hour/minute/second`) take a `Datetime`
  (year..day also accept `Date`) or `Time` (hour..second) column and return
  `i64` columns; `weekday` is 1=Monday..7=Sunday. NULL rows stay NULL.
- `date_range(start, end, every, closed)` builds a `Datetime` column;
  `truncate(col, every, offset)` floors each value to its window start.
  Both use floor division and closed-interval semantics as in §18.4.
- `type_layout`/`size_of_ty`/`align_of_ty` gain the four temporal types
  (size 8, align 8); `sortable_dtype`/`compare_values` treat them as `i64`;
  `is_numeric_type` does *not* (temporal arithmetic is not scalar arithmetic).
  Rendering (`print.odin`) uses ISO-8601 (`Date` = `YYYY-MM-DD`, `Datetime` =
  `YYYY-MM-DDTHH:MM:SS.ffffff`, `Time` = `HH:MM:SS.ffffff`, `Duration` =
  `±DDD:HH:MM:SS.ffffff`).

### 18.6 NULL handling (S14.9)

New `expr` nodes (constructors `is_null_`, `is_nan_`, `fill_null_`,
`coalesce_`, `ffill_`, `bfill_`, `interpolate_`):

- `is_null(e)` = `not(is_not_null(e))` (bool, all valid).
- `is_nan(e)` — float columns only; bool output; NULL rows stay NULL.
- `fill_null(e, value)` — replaces NULL rows with the constant scalar
  `value` (a `Lit` child of the same dtype). String fills copy the literal
  into a rebuilt payload so the result owns its data.
- `coalesce(parts)` — per row, the first non-NULL value across the parts
  (all parts must share a dtype); NULL when every part is NULL.
- `ffill(e)` / `bfill(e)` — carry the last/next valid value across NULL rows
  (whole column, no partitioning); leading/trailing NULLs stay NULL.
- `interpolate(e)` — linear interpolation of numeric columns between the
  surrounding valid values (`f64` output); leading/trailing NULLs stay NULL.
- `dataframe_drop_nulls(df, cols)` — a new DataFrame without the rows where
  any of `cols` is NULL (empty `cols` = every column).

### 18.7 Categorical and Enum dtypes (S14.10)

A categorical column is dictionary-encoded: `dtype = i64` codes into an owned
category table `categories: []string`, tagged with
`Categorical_Kind { None, Categorical, Enum }`:

- `categorical_from_strings(_with_valid)` assigns codes in first-seen order.
- `enum_from_strings(_with_valid)` validates every value against a caller
  `levels` list (a value outside the levels is `.Invalid_Argument`) and
  stores levels as the category table with `Categorical_Kind.Enum`.
- `categorical_categories` / `enum_levels` return the borrowed table;
  `categorical_value(col, i)` returns the string for row i (valid=false for a
  NULL row or an out-of-range code).
- Grouping/joining/unique key on the **category string**, not the code
  (`encode_row` encodes categorical columns by their category), so two
  tables that happen to share codes never merge incorrectly. Sorting
  (`compare_values`) orders categoricals by their category string. Printing
  renders the category.
- `column_copy`, `gather_rows_core`, and `column_destroy` carry the category
  table (owned deep copies).

### 18.8 Reshaping: pivot and melt (S14.1)

- `dataframe_melt(df, id_vars, value_vars, variable_name, value_name)` — one
  row per (id row × value column): the id columns, a `variable_name` column of
  column names, and a `value_name` column of values. All value columns must
  share one dtype (`.Type_Mismatch` otherwise). Empty `value_vars` = all
  non-id columns. Defaults: `"variable"` / `"value"`.
- `dataframe_pivot(df, index, columns, values, agg)` — one row per distinct
  index-key (empty `index` = a single group), and one column per distinct
  value of the `columns` column, named by that value's canonical string form
  (`scalar_to_string`: numbers via `fmt`, bools, strings verbatim, temporal
  types ISO). Each cell runs `agg` (Stage 6 kernel, `validate_agg`) over the
  rows of the `values` column whose (index, columns) cell matches; cells with
  no rows are NULL. Two distinct values that stringify identically are an
  `.Invalid_Argument`.
