# ROADMAP.md — implementation milestones

Work one milestone at a time, in order. Each stage ends with `odin test <dir>`
green and, where noted, a benchmark. Tasks are numbered `S<stage>.<n>`.

Stages mirror Plan.md §14 and the Plan's phased model. The first three
commits (per the Plan) are: skeleton + docs, typed Column + tests,
DataFrame + tests.

This project builds **on existing tooling**, not from scratch (AGENTS.md
principle 2): every stage first looks for what Odin already provides — the
`core:` stdlib (sort, math, encoding, thread/sync, container, simd, prof) and
the in-tree `src/parallel` pool helper — and wires it into the dataframe API.
Custom code is the thin glue where nothing fits; stage notes record which
existing tools were reused and where a future `core:`/third-party tool could
take over. Reusing the stdlib keeps each stage small, the surface predictable,
and the library extensible.

---

## Stage 0 — Skeleton and research artifacts (done)

- S0.1 `AGENTS.md` with project rules. — done
- S0.2 `DESIGN.md` with the column-representation decision and architecture. — done
- S0.3 `ARCHITECTURE.md` with layout and data flow. — done
- S0.4 `ROADMAP.md`. — done
- S0.5 Verify toolchain: `odin version`, `odin test`, `odin run src/`,
  `core:encoding/csv` availability. — done (dev-2026-07; note the
  `#partial switch` over `Type_Info.variant` compiler bug in DESIGN.md §2.3).

## Stage 1 — Columnar core (done)

Goal: `Column`, `Schema`, ownership, NULL semantics. No DataFrame.

- S1.1 Implement `src/dataframe/dtype.odin`: `Field` and `Schema` with
  `typeid`-based dtypes (no logical-type enum in the MVP; see DESIGN.md §2.1).
- S1.2 Implement `src/dataframe/error.odin`: `Error` enum (unknown column,
  length mismatch, type mismatch, null, etc.).
- S1.3 Implement `src/dataframe/column.odin`: `Column` struct per DESIGN.md §2.2
  (type-tagged raw element buffer + `valid`), `column_destroy`.
- S1.4 Generic constructor `column_from(allocator, name, values: []$T)` and
  `column_from_with_valid`; any Odin type is supported; allocator captured in
  the struct.
- S1.5 Accessors: `column_name`, `column_dtype`, `column_len`, `column_valid`,
  `column_is_all_valid`, `column_is_valid(col, i)`, `column_elem_size`,
  `column_data`.
- S1.6 Typed value accessor `column_get(col, i, $T)` with NULL semantics
  (returns `valid=false` for NULL; `Error` for wrong type / out of bounds).
- S1.7 Typed value setter `column_set(col, i, value)`; type-checked via strict
  `typeid` equality (no silent conversion).
- S1.8 `column_copy`, `column_clone_valid`, `column_set_all_valid`.
- S1.9 Tests: creation, retrieval, NULL vs zero distinction, empty column,
  destroy/leak-free (via `odin test` memory tracking).
- S1.10 Benchmarks: column create + typed get across 1K/100K/1M rows
  (`benchmarks/column.odin`).

## Stage 2 — DataFrame core (done)

Goal: ordered, typed, owned columns with row count; no ops yet.

- S2.1 Implement `src/dataframe/dataframe.odin`: `DataFrame` struct
  (owned `[dynamic]Column` + allocator), `dataframe_create`,
  `dataframe_destroy`. — done
- S2.2 `dataframe_add_column` (validates name uniqueness, length match). The
  source `^Column` is zeroed on success (ownership transfer, DESIGN.md §4.1);
  on validation error nothing is consumed. — done
- S2.3 `dataframe_remove_column`, `dataframe_rename_column`,
  `dataframe_get_column`, `dataframe_has_column`. — done
- S2.4 `dataframe_num_rows`, `dataframe_num_cols`,
  `dataframe_schema` (schema derived from columns, never stored). — done
- S2.5 `dataframe_from_columns` convenience constructor (validates all, then
  transfers by zeroing each source). — done
- S2.6 `dataframe_copy` (deep). The explicit borrow view is deferred to
  S4.3 `slice`/`take`, which materialize new columns from row indices
  (DESIGN.md §4.3); a borrow-view type is only introduced if a benchmark
  justifies it. — done (copy only)
- S2.7 Tests per Plan.md §10 matrix (normal/empty/one-row/many-rows,
  boundary, error cases). — done
- S2.8 Build a minimal demo in `src/main.odin` using the eager API. — done

## Stage 3 — Expressions (foundation)

- S3.1 Implement `src/dataframe/expr/expr.odin`: `Expr` union (`Col`, `Lit`,
  `Binary`, `Unary`, `Cast`, `Alias`, `Agg`, `Is_Not_Null`). — done
  (no `Agg` yet; deferred to Stage 6 — see note).
- S3.2 Constructors: `col(name)`, `lit(value)`, operator helpers
  (`add`, `mul`, `gt`, `eq`, `and`, `or`, `not`, …). — done
- S3.3 Implement `expr/eval.odin`: evaluate `Expr` against a DataFrame context
  -> typed `Column`; switch-dispatch over the union. — done (lives in
  `src/dataframe/expr_eval.odin`, dataframe package, per DESIGN.md §6).
- S3.4 Binary op kernels for the supported numeric/string/bool type matrix;
  NULL propagation rules. — done (incl. int div/mod-by-zero -> NULL).
- S3.5 `cast` expression with explicit type conversion. — done
- S3.6 `alias` naming of expression results. — done
- S3.7 Implement `expr/typecheck.odin`: static dtype inference for exprs;
  reject incompatible ops early with `Error`. — done (lives in
  `src/dataframe/typecheck.odin`; the expr package stays dependency-free, so
  typecheck sits beside eval in the dataframe package).
- S3.8 Tests: eval, NULL propagation, type errors, constant exprs. — done
  (`src/dataframe/expr_test.odin`; 70 tests green).
- S3.9 Benchmark: elementwise `a + b` expression over 1M rows vs naive loop.
  — done (`benchmarks/expr.odin`; see notes below)
- S3.10 Expression library (polars fold-in, DESIGN.md §17.2): `abs`, `sign`,
  `diff`, `pct_change`, `dot_product`, `round_series`, `is_between`, `is_in`,
  `is_first_distinct`/`is_last_distinct`, `arange`,
  `concat_str`, `arg_where`, `search_sorted`.
  — done (`src/dataframe/expr_eval.odin` + `typecheck.odin`; built on
  `core:math` where possible: `math.abs`, `math.sign`, `math.round`,
  `math.pow`; `concat_str` delegates the row join to `core:strings.join` and
  stores the owned string `payload` blob, and `expr.Context.extra` tracks
  out-of-line arena allocations). Removed from the original polars fold-in as
  Rust-functional-flavored (Odin is data-oriented): `checked_arithmetic`,
  `zip_with`, `repeat_by` — Odin expresses these with plain loops/arithmetic.

### Stage 3 notes (benchmarks/expr.odin, dev-2026-07)

`a + b` over i32 columns, full expression path (tree build + `expr_eval` +
result destroy) vs a raw typed loop. Manual timing, one run each:

```
      rows |    naive(ms) |     expr(ms) | x slower
       1K  |        0.004 |        0.016 |     3.78
     100K  |        0.488 |        1.180 |     2.42
       1M  |        5.271 |       10.979 |     2.08
```

The constant ~2x at scale is dominated by allocation: `Col` evaluation deep-
copies each operand, `binary_eval` allocates a fresh result buffer plus a
`[]bool` validity array, and the output is allocated before the kernel runs.
All are "eager, caller-owned" costs of DESIGN.md §6; no optimization without
a further profile (principle 10).

## Stage 4 — DataFrame operations (done)

- S4.1 `filter(df, Expr)` predicate -> new DataFrame (mask -> indices or
  direct kernel materialization). — done (`filter.odin`; mask -> indices via
  `mask_true_indices` then `take_columns`)
- S4.2 `select(df, []Expr)`, `select_by_name(df, names)`,
  `with_columns`, `drop`. — done (`select.odin`)
- S4.3 `head`, `tail`, `slice`, `take(indices)`, `limit`. — done
  (`filter.odin`, row-gathering via `gather_rows`)
- S4.4 `unique(df, cols)`. — done (`unique.odin` + `group_keys.odin`
  canonical row-key encoding; NULL is a per-column key value)
- S4.5 `partition_by(df, exprs)` — split into DataFrames by group keys
  (polars fold-in, DESIGN.md §17.2). — done (`partition_by.odin`)
- S4.6 Tests incl. empty df, single column, NULL rows, unknown column errors.
  — done (`ops_test.odin`; 110 tests green, leak-checked)
- S4.7 Benchmark filter + select over 1K/100K/1M/10M rows. — done
  (`benchmarks/ops.odin`; see notes below)

### Stage 4 notes (benchmarks/ops.odin, dev-2026-07)

Workload: filter a 3-col df (`id` i64, `group` i32, `value` f64) on
`value > 0.5` (~50% kept), then select `id` + `value * 2`. Manual timing,
one run each:

```
       rows |    baseline(ms) |       api(ms) | x slower
        1K  |         0.013 |         0.064 |     4.82
      100K  |         1.265 |         4.149 |     3.28
        1M  |        13.484 |        44.554 |     3.30
       10M  |       134.919 |       463.646 |     3.44
```

The stable ~3.4x at scale is the documented eager cost of DESIGN.md §6:
`dataframe_filter` materializes a full 3-column DataFrame (fresh element
buffers, per-column `[]bool` validity, per-row bytewise copies), then
`dataframe_select` deep-copies/allocates a second time. Each step also
allocates and frees intermediate index/validity arrays. The baseline is the
same logical work (mask + gather) with no intermediate materialization. No
optimization without a further profile (principle 10); Stage 11's lazy plan
will collapse filter+select into a single materialization.

## Stage 5 — Sorting (done)

- S5.1 `sort(df, by, direction)` and `sort_by(df, exprs)` using permutation.
  — done (`sort.odin`; `dataframe_sort`/`dataframe_sort_by` materialize via
  `take_columns`; keys are `Sort_Key` structs — `sort_key("age", .Desc)`)
- S5.2 `argsort(df, col)` returning `[]int` permutation. — done
  (`dataframe_argsort`)
- S5.3 Multi-column sort, mixed directions. — done (keys compared in order;
  each `Sort_Key` carries its own order)
- S5.4 NULL ordering rule (define: nulls first/last param, default documented).
  — done (`Sort_Key.nulls_first`, default false = NULLs after all values,
  direction-independent; rows NULL in the same keys tie into the next key)
- S5.5 Tests: stable sort property, multi-key, NULL ordering, idempotence
  `sort(sort(df)) == sort(df)`. — done (`sort_test.odin`; 18 tests incl.
  float total order, string/bool sorts, argsort permutation, sort_by exprs)
- S5.6 Benchmark sort 1M rows per dtype. — done (`benchmarks/sort.odin`;
  see notes below)

### Stage 5 notes (benchmarks/sort.odin, dev-2026-07)

Sort of 1M rows, full `dataframe_sort` path (permutation + row gather), keys
permuted worst-case (ascending first half, descending second), one run each:

```
  dtype  |  sort 1M(ms)
    i32  |       481.9
    i64  |       517.8
    f64  |       656.8
 string  |       708.0
```

Sorting uses the built-in `core:sort` via its `Interface` form: `collection`
carries the key context (this Odin build rejects closures, so
`merge_sort_proc`'s plain `proc(T, T) -> int` cannot see the keys). `sort.sort`
is documented unstable, so `argsort_key_columns` breaks ties with the original
row index — since the permutation elements *are* the original indices, the
result is identical to a stable key-only sort. Every comparison dispatches
through `Interface` plus a dtype switch; an earlier hand-rolled stable merge
sorted the same workload ~1.6x faster (i32 301ms) but was dropped in favor of
the stdlib. Stage 15's specialized kernels are the planned lever.

## Stage 6 — Statistics / aggregations

- S6.1 `count`, `sum`, `mean`, `min`, `max`, `var`, `std`, `median`,
  `quantile`, `n_unique` per column / over expressions.
- S6.2 Extend aggs (polars fold-in, DESIGN.md §17.2): `mode`, `product`,
  `cov`, `corr`, `moment` (kurtosis/skew).
- S6.3 NULL semantics: skip invalid rows; `count` counts valid by default.
- S6.4 Type matrix for numeric aggs; `Error` for non-numeric.
- S6.5 Tests incl. NULL-only column, single row, mixed NULL.
- S6.6 Benchmark aggregations over 1M/10M rows. — done
  (`benchmarks/agg.odin`; see notes below)

### Stage 6 notes (benchmarks/agg.odin, dev-2026-08)

Aggregations over a 2-column df (id i64 = 0..n-1, value f64 = (i % 1000)+0.5,
all rows valid), per-column scalar path (stats.odin: column lookup + kernel +
result column alloc/destroy), one run each:

```
reference: cost of `sum(value)` three ways
  rows       |  naive(ms)  |  api(ms)  |  expr(ms)
   1,000,000 |        3.19 |      4.73 |      7.25
  10,000,000 |       37.67 |     48.99 |     74.22

aggregations
  agg       |  1M(ms)  | 10M(ms)
  count     |     1.75 |    21.28
  sum       |     5.44 |    48.51
  mean      |     5.12 |    48.85
  var       |    14.24 |   125.24
  min       |     5.22 |    50.50
  max       |     4.96 |    51.72
  median    |   130.08 |  1528.69
  quantile  |   132.10 |  1520.98
  skew      |    12.38 |   139.06
  kurtosis  |    14.45 |   129.43
  n_unique  |   360.87 |  4322.14
  mode      |    42.10 |   387.57
  cov       |    15.33 |   154.72
  corr      |    14.34 |   163.07
```

The scalar API adds ~1.5x over the raw single-pass loop (result column
alloc/destroy + per-row dispatch); the expression path adds another ~1.5x
(expr tree walk + select materialization). Cheap one-pass kernels (count/sum/
mean/min/max) scale linearly with the data. Two-pass moment aggs (var/skew/
kurtosis) and cov/corr are ~2-3x the one-pass cost. The sort-based
median/quantile dominates at ~25x sum (the dynamic-array gather + `quick_sort`
of all valid rows), and the hash-set n_unique at ~70x sum over the
high-cardinality id column (1M-entry map). Stage 7 will reuse these kernels
over row-index subsets; the Stage 15 optimization reserve (validity bitmap,
specialized kernels) is where the sorting/hash paths would get attention.

## Stage 7 — GroupBy — COMPLETE

- S7.1 `dataframe_group_by(df, allocator, exprs)` -> `Group_By` (key -> row
  indices) via hash. Key values encoded per row (`encode_row`, reused from
  partition); first-seen group order preserved; multi-key and computed keys
  supported.
- S7.2 `dataframe_group_by_agg(gb, allocator, aggs)` — agg over count/sum/
  mean/min/max via the Stage 6 kernels run on each group's row subset.
- S7.3 Extended aggs: var/std/median/quantile/n_unique/first/last.
- S7.4 NULL keys form their own group; NULL values within groups follow the
  Stage 6 rules (skipped by kernels; all-NULL group -> NULL value aggs, 0
  count/n_unique).
- S7.5 Tests: grouping correctness, multiple keys, NULL keys, NULL-only
  groups, agg over computed exprs, consistency vs whole-column aggs, empty
  df (0 rows, schema preserved), result dtypes, error cases. 161 tests green
  (was 150).
- S7.6 Benchmark groupby over 1M rows (hash-based), per plan §7. `benchmarks/
  groupby.odin`: full group_by+agg over count+sum, 1000 groups. groupby is
  ~4.5-5x a hand-rolled single-pass map reduction (1M: 127.6ms vs 26.6ms;
  10M: 1184.5ms vs 270.6ms). The overhead is key string encode (encode_row,
  fmt-backed), per-row map insert, per-group kernel dispatch, and result
  materialization — all single-threaded, deferred to the Stage 15
  performance work (specialized keys, parallel groupby).

## Stage 8 — Joins

- S8.1 `inner_join(left, right, left_keys, right_keys)` via hash join.
- S8.2 `left_join` (MVP complete per Plan.md §8).
- S8.3 `right_join`, `full_join`.
- S8.4 `semi_join`, `anti_join`, `cross_join`.
- S8.5 Column-name collision handling (suffix rules, documented).
- S8.6 Tests: join stress (empty sides, NULL keys, duplicates, type mismatches).
- S8.7 Benchmark join 1M x 1M; document hash-build/probe sizing.

## Stage 9 — CSV I/O (done)

- S9.1 `dataframe_read_csv(allocator, path, options)` using `core:encoding/csv`:
  headers, delimiter, quoted fields, escaped quotes, empty fields. — done
- S9.2 Type inference (sample records -> dtype per column; then validate all).
  — done (strict: a post-sample value that fails the sampled type is a
  `.Type_Mismatch`, never a silent widening)
- S9.3 `dataframe_read_csv_with_schema(allocator, path, schema, options)`.
  — done
- S9.4 NULL value recognition (empty/unquoted token, configurable via
  `CSV_Options.null_token`). — done (default "" means empty fields are NULL;
  a non-empty token makes the empty string a real value)
- S9.5 `dataframe_write_csv(df, allocator, path, options)` round-trip.
  — done (f64 shortest-round-trip incl. NaN/±Inf; string contents deep-copied)
- S9.6 Tests: round-trip, inference vs explicit schema, malformed rows.
  — done (`csv_test.odin`; 24 tests, all 198 package tests green)
- S9.7 Benchmark CSV parse 1K/100K/1M rows; document vs single-thread baseline.
  — done (`benchmarks/csv.odin`; results in DESIGN.md §14.1)

### Stage 9 notes (benchmarks/csv.odin, dev-2026-08)

- Reused: `core:encoding/csv` RFC 4180 reader/writer (record split, quoting,
  escapes), `core:strconv` (`parse_i64`/`parse_f64`), `core:strings` (builder
  for write), `core:os` (file I/O).
- Gotcha (documented in code): `csv.Reader` must be initialized in place —
  `reader_init_with_string` wires bufio's `io.Reader` to an internal
  `strings.Reader` by pointer, so returning the reader struct by value leaves
  the copy dangling and every read fails with EOF.
- String columns parse into per-row byte offsets (`CSV_String_Seg`) plus a
  shared owned blob; headers are materialized after the last append and the
  blob is transferred into the column payload, so CSV-read strings are deeply
  owned (no aliasing of the file buffer).
- NULLs are skipped by inference and never constrain the dtype; a column with
  no sampled non-NULL value falls back to string (so header-only files read as
  string columns). Inference order is bool > i64 > f64 > string.
- Benchmark (best-of-3): 1K ~1.1ms, 100K ~98ms, 1M ~1.04s (~0.96 M rows/s,
  ~31 MB/s). Parse cost is dominated by the stdlib record scanner; this is the
  baseline S15.7 parallel parsing must beat.

## Stage 10 — Ergonomics

- S10.1 Revisit and finalize the public API (naming, argument order, errors).
  **done** (2026-08): allocator-last convention enforced everywhere
  (`schema_create(fields, allocator := …)`, was allocator-first); blessed
  names recorded in DESIGN.md §15.1.
- S10.2 Method-chaining-friendly helpers where they read naturally. **done**
  (2026-08): `chain.odin` binds short aliases (`dataframe.filter`, `select`,
  `sort`, `head`, …) so pipelines compose with `or_return`. Odin has no
  method-call sugar for procs (`->` only calls proc-typed fields; no `|>`
  pipe in this Odin version), so this is the idiomatic chain style.
- S10.3 Pretty-print `dataframe_print` (with NULL rendering). **done** (see
  `src/dataframe/print.odin`, `print_test.odin`).
- S10.4 Convenience constructors (empty with schema, from CSV). **done**:
  `column_empty`, `dataframe_create_with_schema`; `dataframe_read_csv` (S9)
  already covers the from-CSV constructor.
- S10.5 Update DESIGN.md §15 sketch to the blessed API; update demo. **done**
  (2026-08): §15 rewritten to the real API + §15.1 conventions; `src/main.odin`
  demonstrates pipeline, group_by/agg, and schema-shaped constructors.
- S10.6 Property tests: `sort(sort(df)) == sort(df)`,
  `row_count(filter(df, p)) <= row_count(df)`, filter/select idempotence.
  **done** (2026-08): `property_test.odin` (seeded PRNG, 4 invariants,
  6 iterations each).

## Stage 11 — Lazy engine

- S11.1 `lazy/plan.odin`: `Logical_Plan` union (Scan_CSV, Scan_DF, Filter,
  Projection, Sort, Group_By, Limit, Slice, Join). **done** (2026-08): node
  structs own their arena-copied slices; exprs/dfs/paths are borrowed.
- S11.2 `lazy/lazyframe.odin`: `LazyFrame` wrapper; builders
  `scan_csv().filter().select()...`. **done** (2026-08): Odin has no method
  sugar, so builders are `lazy.filter(lf, pred)` etc., rebinding one variable.
  `lazy.join` builds an independent plan (clones both child trees), so each
  of its three frames is destroyed once; linear builders share one arena.
- S11.3 `collect()` -> executes the plan via the eager engine (no optimizer
  yet). **done** (2026-08): `executor.odin` walks the plan bottom-up;
  `collect` is repeatable and always returns an owned frame.
- S11.4 `lazy/physical.odin`: operator structs; `lazy/executor.odin` walks
  them. **done** (2026-08): operators are the eager-backed procs in
  `physical.odin` (one per logical node); `executor.odin` is the walker.
- S11.5 Tests: lazy plan building is side-effect free; collect equals eager.
  **done** (2026-08): 11 tests in `lazy_test.odin` — building never touches
  data (missing CSV / unknown column / bad shapes surface at collect),
  collect == eager for filter/sort/limit, select, slice, group_by+agg, join,
  and a bare `scan_dataframe` root returns an owned deep copy.
- S11.6 Benchmark: lazy vs eager equivalence + timing on 1M rows. **done**
  (2026-08): `benchmarks/lazy.odin` — filter→sort→limit over a 1M-row CSV;
  results equal row-for-row, timing ratio ~0.98 (expected: the S11 executor
  is eager-backed, so it cannot yet beat eager).

## Stage 12 — Optimizer (done)

- S12.1 Projection pushdown (prune scan columns). **done** (2026-08):
  `lazy/optimizer.odin` rewrites the plan at collect time; a `Scan_CSV` reads
  only the columns its ancestors reference via `dataframe_read_csv_with_columns`
  (which now infers, type-checks, and parses only the requested fields).
  Pass-through nodes never prune; join is a barrier. Tests in
  `lazy/optimizer_test.odin` (pruning + equivalence vs eager).
- S12.2 Predicate pushdown (filters before joins/materialization). **done**
  (2026-08): `push_filter` threads one predicate down the tree, rebuilding
  `Sort`/`Filter`/`Projection`/`Join` nodes in the plan arena. Filters cross
  sorts unchanged, merge with existing filters via `And`, rewrite column
  references back through projections, and wrap the pushable (left) side of a
  join. The predicate pushed further is always the rewritten/merged one — the
  original plan keeps its top filter, so repeated `collect` is stable.
- S12.3 Column pruning (drop unused intermediate columns). **done** (2026-08):
  `prune_exprs` trims a projection under an ancestor to the names the ancestor
  references (reordered to first-seen order); a projection is only rewritten
  when it is a valid standalone select (every output named, no duplicates).
- S12.4 Constant folding. **done** (2026-08): `fold_constants`/`fold_expr`
  collapse literal-only sub-expressions (including single-argument funcs like
  `abs`/`sign`/`round`) to a `Lit` at collect time.
- S12.5 Common subexpression elimination. **done** (2026-08):
  `projection_op` (physical.odin) evaluates structurally equal projection
  expressions once (top-level `Alias` stripped) and deep-copies the column into
  each output position under that position's name; error order matches eager
  `dataframe_select`.
- S12.6 Equivalence tests per rule (plan before == plan after on data). **done**
  (2026-08): 18 tests in `lazy/optimizer_test.odin` — every rule has an
  equivalence case vs eager plus structure assertions (pruned scan columns,
  pushed-filter placement, dropped intermediate exprs, CSE column sharing).
  `opt_repeat_collect_stable` re-collects the same frame twice.
- S12.7 Benchmark: 20-column CSV, filter+select 2 cols; compare scan cost.
  **done** (2026-08): `benchmarks/optimizer.odin` — 1M rows x 20 cols
  (172 MB); pruned 2-col scan runs 2.6x faster than the full 20-col scan
  (2224 ms vs 5766 ms), and the lazy plan with auto-pushdown matches it.

### Stage 12 notes (lazy optimizer, 2026-08)

- This stage made the lazy package compile and test green for the first time:
  `odin test src/dataframe/lazy/` passes 29 tests (11 from S11.5, 18 S12.6
  optimizer tests); `odin test src/dataframe/` passes all 216 eager-engine
  tests, and `odin run src/` still builds.
- Non-exhaustive union/enum switches in the optimizer are `#partial switch` —
  the compiler requires it once the remaining cases are intentionally
  unhandled (row-shaping nodes, un-fusable funcs).
- Fix shipped in S12.2: `push_filter` previously kept pushing the original
  predicate after rewriting it through a projection, so a filter below merged
  `And{unrewritten, rewritten}` — referencing a column that does not exist at
  that depth (`Column_Not_Found` on the second `collect`). The loop now threads
  a `to_push` local that becomes the rewritten/merged predicate at each step.

## Stage 13 — More I/O (done)

- S13.1 JSON read/write. **done** (2026-08): `src/dataframe/json.odin`
  (`dataframe_read_json`, `dataframe_read_json_with_schema`,
  `dataframe_write_json`) built on `core:encoding/json` — the token scan,
  string unescaping, and number grammar are stdlib; the file only glues typed
  JSON values to columns (reusing the CSV column buffers). Reads are strict
  JSON (`json.Specification.JSON`, i64 for integral tokens); columns are
  key-alphabetical (objects are unordered maps; the stdlib unparse sorts the
  same way); a key seen only as null falls back to string (CSV's no-sample
  fallback). Type rules match CSV: first non-NULL value fixes the dtype,
  later mismatches are `.Type_Mismatch`, never a widening; nested
  Object/Array is `.Unsupported_Operation` until List/Struct land (S14.2).
  Writes keep the DataFrame's column order and force a decimal/exponent on
  f64 so integral floats read back as f64, not i64; NaN/±Inf are written
  verbatim (like CSV) and are write-only (strict JSON has no such literal).
  Design documented in DESIGN.md §11.
- S13.2 NDJSON. **done** (2026-08): `dataframe_read_ndjson` /
  `dataframe_write_ndjson`, one object per line, blank lines skipped, same
  type rules as JSON.
- S13.3 Arrow IPC / Parquet. **Arrow IPC done** (2026-08): vendored
  [OdinArrow](https://github.com/TimeLord/OdinArrow) as `src/odinarrow/`
  (Zlib license); bridge in `arrow.odin` provides `dataframe_write_arrow` /
  `dataframe_read_arrow`. Supports all primitive types (i8–u64, f32/f64, bool)
  and UTF-8 strings, with NULL round-trip. Parquet remains deferred (requires a
  Parquet-specific code generator). File: `arrow.odin`, tests: `arrow_test.odin`
  (9 round-trip tests).
- S13.4 Tests + benchmarks. **done** (2026-08): 21 tests in
  `json_test.odin` (round-trips incl. pretty + f64 specials, alphabetical
  order, strict typing, nested rejection, NULL/missing/all-null keys, schema
  mode incl. extra-key/all-null/unsupported-type, NDJSON, error cases);
  `benchmarks/json.odin` on 1K/100K/1M rows (read ~350K rows/s, write
  ~600K rows/s, ~24–27 MB/s).

### Stage 13 notes (JSON/NDJSON I/O, 2026-08)

- Full suite green: `odin test src/dataframe/` passes 237 tests (216 eager +
  21 JSON), `odin test src/dataframe/lazy/` passes 29, `odin run src/` builds.
- JSON parse is ~3x slower than CSV on the same 1M-row synthetic data
  (2841 ms vs 996 ms): the stdlib builds a full `Value` tree (per-key strings,
  map inserts) that must be walked twice (analysis + fill). That is the
  baseline parallel/columnar JSON parsing in a later stage (S15) must beat;
  the write path is the same order as CSV (~600K rows/s vs ~1M rows/s).
- Two Odin gotchas hit during S13: (1) a fresh `map` and a fresh `[dynamic]`
  array are `nil` (lazily backed), so `make(...) == nil` is not an allocation
  check for them — the analyzer initially misreported `.Allocator_Failure`;
  (2) Odin's `for v, i in slice` yields (value, index), not (index, value).
- Upstream quirk: `core:encoding/json` leaks a few bytes when it aborts
  mid-object (an already-cloned key never inserted into the object map). It
  only triggers on invalid input, which fails with `.JSON_Error` anyway.

## Stage 14 — Advanced functionality

- S14.1 `pivot`, `melt`/`unpivot`.
  — done (`src/dataframe/reshape.odin`: `dataframe_melt(df, id_vars, value_vars,
  variable_name, value_name)` with value-column-major stacking, owned strings,
  dtype checks, default value_vars = non-id columns; `dataframe_pivot(df, index,
  columns, values, agg, q)` grouping by index keys via `encode_row`, pivot
  columns named by `scalar_to_string`, per-cell Stage 6 kernels with NULL cells;
  `scalar_to_string` renders numbers/bools/strings/temporal/categorical; tests
  in `reshape_test.odin`).
- S14.2 List/struct dtypes (polars fold-in, DESIGN.md §17.2): `List` dtype +
  `list_*` ops (`list_gather`, `list_count`, `list_to_struct`, list sets),
  `Struct` dtype. `explode`/`unnest` depend on `List`.
  — done (`src/dataframe/list.odin` per DESIGN.md §18.1: `List_Ref` rows
  indexing a payload element buffer; `list_from_column` (transfers/zeroes the
  element column, merges an owned string payload into one blob with re-pointed
  headers), `list_from_slices(_with_valid)` (outer NULL rows keep a zeroed
  `List_Ref`); `Column.inner_dtype`/`inner_valid` owned and carried through
  `column_destroy`/`column_copy`/`gather_rows_core`; ops `list_count`,
  `list_get`, `list_gather` (per-row index), `list_unique` (first-seen, NULL
  elements skipped), `list_to_struct` (`field_0..`); `[a, b]` rendering; the
  Struct dtype itself lands with `unnest` in S14.3; tests in `list_test.odin`).
- S14.3 `explode`, `unnest` (list columns).
  — done (`src/dataframe/explode.odin`: `dataframe_explode(df, col)` widens the
  row count to the total list elements, repeating outer-row values for all
  element types (primitives via `mem.copy`, owned strings re-pointed into the
  output payload, nested Lists via recomputed offsets), outer NULL rows and NULL
  elements propagate as NULL with a `[0,0]` List_Ref; `dataframe_unnest(df,
  col)` flattens a struct column into `<col>_<field>` columns by reflected
  field offsets, NULL struct rows -> NULL fields, rejects List fields and
  non-struct input; tests in `explode_test.odin`).
- S14.4 Window functions: `rank`, `row_number`, `cum_sum`, `cum_min/max`,
  `shift`, `rolling` with `over(...)`.
  — done (`src/dataframe/window.odin` + `expr.Window` node: `over` partitions by
  evaluated key expressions via `encode_row`; kernels for Row_Number (i64),
  Rank (f64, Average/Min/Max/Dense ties via `core:sort`), Cum_Sum/Min/Max
  (running, NULL-continues), Shift (lag/lead, bytewise, NULL out-of-partition),
  Rolling (trailing window through `run_group_agg`), string-output ownership via
  `finalize_string_contents`; wired into `expr_eval`/`typecheck` and the lazy
  optimizer's fold/require/structural-equality passes; tests in
  `window_test.odin`).
- S14.5 Window/sequence extensions (polars fold-in, DESIGN.md §17.2):
  `cumulative_eval`, `ewma`.
  — done (same file: `Cumulative_Eval` reuses the Rolling kernel with a growing
  window `[0, i]`; `Ewma` runs `y_i = alpha·x_i + (1-alpha)·y_{i-1}` per
  partition, first valid row seeds `y = x`; tests in `window_test.odin`).
- S14.6 As-of joins.
  — done (`src/dataframe/asof_join.odin` per DESIGN.md §18.3:
  `dataframe_asof_join(left, right, left_on, right_on, left_by, right_by,
  strategy)` with `Asof_Strategy { Backward, Forward }`; right rows bucketed by
  `by` via `encode_row` (NULL `by`/`on` excluded) and, per the documented
  right-sorted precondition, each left row matched by binary search over the
  bucket's `on` values (`asof_compare` covers every sortable dtype incl. string
  and the temporal types; ties resolve to the last/first bucket row);
  output is left-major, right `by`/`on` dropped, join.odin `_right` collision
  rule; materialization reuses `join_materialize`; tests in
  `asof_join_test.odin`).
- S14.7 Dynamic groupby (group_by_dynamic) and rolling groupby.
  — done (`src/dataframe/groupby_dynamic.odin` per DESIGN.md §18.4:
  `dataframe_group_by_dynamic(df, time_col, by, every, offset, closed)` buckets
  rows into `[start, start+every)` windows via floor division (`start =
  floor((t-offset)/every)*every + offset`, exact pre-epoch), `closed`
  (Closed_Interval) decides boundary membership, NULL-time rows are dropped,
  NULL `by` values form their own group; `dataframe_group_by_rolling(df,
  time_col, by, period, offset, closed)` builds one trailing window per source
  row `[t-period-offset, t-offset]` per by-group (rows time-sorted, windows by
  binary search), NULL-time rows get an empty window; both aggs reuse the
  Stage 6/7 kernels via the shared `group_agg_run` (nil windows normalized so
  empty windows aggregate to count 0 / NULL value results); dynamic output is
  ordered (window start, by first-seen), rolling output preserves source order;
  tests in `groupby_dynamic_test.odin`).
- S14.8 Time series: Date/Datetime/Time/Duration dtypes + dt.* accessors,
  `date_range`, `truncate`.
  — done (`src/dataframe/calendar.odin`: distinct i64 dtypes, `date_create`/
  `datetime_create`/`time_create`, conversions, arithmetic, `dt_*` accessors,
  `date_range`, `truncate`; rendering in `print.odin`; dtype/compare support;
  tests in `calendar_test.odin`).
- S14.9 `is_null/is_not_null/is_nan`, `fill_null`, `drop_nulls`, `coalesce`,
  forward/backward fill, `interpolate`.
  — done (`src/dataframe/nulls.odin`; expr nodes per DESIGN.md §18.6, constructors
  `is_null_`/`is_nan_`/`fill_null_`/`coalesce_`/`ffill_`/`bfill_`/`interpolate_`,
  eval + typecheck + lazy-optimizer cases, `dataframe_drop_nulls(df, cols)` with
  empty cols = every column; tests in `nulls_test.odin`).
- S14.10 Categorical / Enum dtypes.
  — done (`src/dataframe/categorical.odin`: `categorical_from_strings(_with_valid)`,
  `enum_from_strings(_with_valid)`, `categorical_categories`, `enum_levels`,
  `categorical_value`; `Column.categories` + `Categorical_Kind`; category-table
  ownership through `column_copy`/`gather_rows_core`/`column_destroy`; string-based
  semantics in `encode_row`/`compare_values`/print; tests in `categorical_test.odin`).

## Stage 15 — Performance

Rule: every item must be justified by a benchmark from the corresponding
stage; profile before optimizing (`core:prof`).

- S15.1 Validity bitmap (replace `[]bool` only if a benchmark justifies it).
  **done** (2026-08): `benchmarks/validity.odin` benchmarked `[]bool` vs packed
  `[]u64` bitmap on sum-scan, gather, and scatter-set across 100K/1M/10M rows
  at 1% and 50% NULL rates. Bitmap wins: sum-scan -7% (48 vs 52 ms at 10M),
  gather -11% (53 vs 60 ms), scatter-set 2.5x faster (0.24 vs 0.59 ms), and
  87.5% memory reduction (1.25 MB vs 10 MB for 10M rows). Decision: adopt.
  `Column.valid` is now `[]u64` packed bitmap; `nil == all valid` invariant
  preserved. `bm_get`/`bm_set` helpers in column.odin; all consumers updated
  (expr_eval, agg, filter, window, nulls, join, reshape, list, explode).
- S15.2 Specialized numeric kernels (vectorized elementwise/agg).
  **done** (2026-08): all-valid fast paths added to `binary_arith`,
  `binary_cmp`, `binary_bool`, `neg_typed`, `func_abs_typed`,
  `func_sign_typed`, `func_round_typed`, and `numeric_reduce_typed`
  (sum/product). When `Column.valid == nil` the validity branch is hoisted
  out of the hot loop, giving 22-38% speedup at 10M rows on sum, add, neg,
  abs (benchmarks/kernels.odin). 1% NULL path also benefits (6-18%) from
  branch-predictor split. All 359 tests pass.
- S15.3 Hash join tuning; parallel probe.
  Pre-sized hash map, reserved pair array, parallel probe via
  `src/parallel` `do_parallel`. 100K/1M/10M inner join benchmark:
  69ms/445ms/4492ms (16-18% faster than sequential). All 359 tests pass.
- S15.4 Parallel executor: adopted `src/parallel` `do_parallel`
  (unmodified upstream); 5 correctness tests in `src/parallel/parallel_test.odin`
  (zero/one/large/uneven/tid-range). Parallel for_each benchmark shows
  2.8-3.1x speedup at 1-10M elements. All 364 tests pass.
- S15.5 SIMD kernels via `core:simd` (guarded per-arch).
  `simd_kernels.odin` provides `simd_neg`, `simd_abs`, `simd_add`,
  `simd_sub`, `simd_mul`, `simd_sum` with `when runtime.HAS_HARDWARE_SIMD`
  guard and scalar fallback. transmute + mem.copy load/store (aligned
  stack buffers). Integrated into `binary_arith`, `neg_typed`,
  `func_abs_typed`, `numeric_reduce_typed`. 10M 0% NULL:
  sum 34ms→11ms (-67%), add 118ms→107ms (-9%), abs 69ms→67ms (-4%).
  All 359 tests pass.
- S15.6 Memory reuse / arena per operator; zero-copy ops.
  **done** (2026-08): `OpArena` type in `op_arena.odin` wraps `mem.Dynamic_Arena`
  for per-operator bump allocation with bulk cleanup. Integrated into expression
  evaluation (`expr_eval`): intermediate child columns in Binary/Unary/Cast/
  Not_Null nodes are arena-allocated; final result uses caller's allocator.
  Top-level callers (`dataframe_select`, `dataframe_filter`, `dataframe_with_columns`,
  `dataframe_group_by`, `dataframe_group_by_agg`, `dataframe_sort`, `partition_by`,
  `groupby_dynamic`, lazy engine) create an arena per operation. Individual
  frees (`column_destroy`, `delete`) are no-ops on `Dynamic_Arena`; bulk-free
  via `dynamic_arena_destroy`. Benchmarks (benchmarks/arena.odin): chained
  expression (a + b*2 - c + 1) 1M rows: 19ms; filter (value > 0.5) & (value < 0.8)
  1M rows: 27ms; select with 3 computed columns 1M rows: 17ms. All 364 tests pass.
- S15.7 Parallel CSV parsing (chunked + merge).
  **done** (2026-08): `csv_parallel.odin` provides `csv_find_chunks` (O(n) byte
  scan respecting quotes), `csv_parse_chunk_task` (per-thread `csv.Reader` +
  `CSV_Column_Buffer`s), `csv_merge_buffers` (memcpy + string blob merge +
  re-base offsets), `csv_build_columns_parallel` / `csv_build_columns_keep_parallel`
  orchestrators. Dispatch in `csv.odin` for all three `read_csv` variants when
  `len(data) >= 1 MB && options.comment == 0`. Bug fixed: `csv_read_header` returned
  dangling aliases (reader destroyed before `csv_own_strings`); inlined header read
  in `dataframe_read_csv_with_schema` to keep reader alive. 14 tests in
  `csv_parallel_test.odin` (chunk splitting, parallel vs sequential correctness for
  100/10K/100K/1M rows, all-strings, all-nulls, quoted newlines, schema, keep).
  Benchmark: 1M rows 668ms (~1.5x sequential), 10M rows 7.2s. All 373 tests pass.
- S15.8 Parallel gather_rows_core (element-copy loop).
  **done** (2026-08): `parallel_gather.odin` provides `Gather_Context`,
  `gather_chunk_task`, `parallel_gather_rows_core` orchestrator. Dispatch in
  `filter.odin:205-215` when `len(indices) >= 10_000 && src.elem_size > 0`.
  Only element-copy loop parallelized; validity bitmap and string repointing
  sequential. 6 tests in `parallel_gather_test.odin` (i64, f64, bool, string,
  nulls, filter equivalence). Benchmark: filter+select 1M rows 29.6ms (was 44.6ms,
  -34%), 10M rows 332ms (was 464ms, -28%). All 379 tests pass.
- S15.9 Parallel SIMD kernels (neg/abs/add/sub/mul/sum).
  **done** (2026-08): `parallel_simd.odin` provides typed context structs,
  task procs, and orchestrators for f64 and i64. Threshold: 65K elements.
  Dispatch in `expr_eval.odin` (binary_arith neg abs) and `agg.odin` (sum)
  via `when T == f64/i64` compile-time branches. 11 tests in
  `parallel_simd_test.odin`. All 390 tests pass.
- S15.10 Parallel coalesce / fill_null (non-string).
  **done** (2026-08): `parallel_nulls.odin` provides `Fill_Null_Context`,
  `fill_null_chunk_task`, `parallel_fill_null` orchestrator; `Coalesce_Context`,
  `coalesce_chunk_task`, `parallel_coalesce_non_string` orchestrator.
  String paths remain sequential (sequential write_at / prefix-sum needed).
  Coalesce pre-allocates validity bitmap to avoid race on lazy allocation.
  5 tests in `parallel_nulls_test.odin`. All 395 tests pass.
- S15.11 Parallel bitmap utilities (bm_from_bools + bm_count_false).
  **done** (2026-08): Added to `parallel_nulls.odin` with `BM_From_Bools_Context`,
  `BM_Count_False_Context`, typed task procs, and orchestrators.
  `bm_from_bools` and `bm_count_false` in `column.odin` now delegate to
  parallel versions. 8 tests in `parallel_bm_test.odin`. All 403 tests pass.

## Stage 16 — Data-Oriented Column Storage (ColumnSet SoA)

- S16.1 ColumnSet core + DataFrame migration. **done** (2026-08): `column_set.odin`
  provides `ColumnSet` struct (SoA layout: `names`, `dtypes`, `sizes`, `rows`,
  `data`, `valids` hot arrays + `metas` cold array), CRUD (`column_set_create`,
  `column_set_destroy`, `column_set_add`, `column_set_get`, `column_set_remove`,
  `column_set_rename`, `column_set_copy`), `column_set_to_column` borrowed view,
  and `cs_typed_view`/`cs_data`/`cs_dtype`/`cs_name` hot-path accessors.
  `ColumnMeta` holds cold fields (align, alloc, payload, categories, inner_dtype,
  inner_valid). `DataFrame.columns` changed from `[dynamic]Column` to
  `ColumnSet`; a `col_views: [dynamic]Column` backward-compat array provides
  borrowed `^Column` pointers for existing callers. `dataframe_get_column` and
  `dataframe_column_at` return pointers into `col_views`. `dataframe_copy` deep-
  copies via `column_set_copy` then rebuilds views. All existing `df.columns[i]`
  patterns eliminated — type system change makes old-style access a compile error.
  18 tests in `column_set_test.odin` (create, add, get, remove, rename, copy,
  typed view, data accessors, round-trip, validation errors, empty/edge cases).
- S16.2 Hot loop optimization. **done** (2026-08): `take_columns` (filter.odin)
  builds `Column` structs directly from `ColumnSet` arrays inline instead of
  calling `column_set_to_column`. `dataframe_print` (print.odin) reads
  `ColumnSet` fields directly via `render_cell_cs` for all standard dtypes,
  bypassing `dataframe_column_at` reconstruction; only List_Ref and categorical
  columns reconstruct a minimal `Column` for their specialized renderers. All
  430 tests pass (18 new ColumnSet tests + 412 existing).

### Stage 16 notes (ColumnSet SoA, 2026-08)

- The ColumnSet split puts the 6 hot fields (`names`, `dtypes`, `sizes`, `rows`,
  `data`, `valids`) in separate arrays so tight loops that touch only one or two
  fields stride over `sizeof(field)` per column instead of `sizeof(Column)` (~168
  bytes). At 100 columns this reduces L1 footprint from ~16 KB to ~4.8 KB for the
  hot fields.
- The `col_views` backward-compat array is maintained because 6 production sites
  (join.odin, unique.odin, reshape.odin, sort.odin, groupby_dynamic.odin,
  asof_join.odin) store `^Column` pointers returned by `dataframe_get_column`,
  and 2 test sites mutate columns through the returned pointer. Removing
  `col_views` would require a dual API (value-return for transient access +
  pointer-return for stored/borrowed access) across ~130 call sites — deferred
  to a later stage if benchmarks justify it.
- `encode_row` (group_keys.odin) still takes `[]^Column` — the ColumnSet-native
  variant would require updating 10+ callers across join, groupby, window,
  partition, reshape, and asof_join. Deferred since `encode_row` is called once
  per row per group (not the dominant cost).
- Benchmark (ops.odin): filter+select 1M rows 28.2ms (~1.9x baseline), 10M rows
  332ms (~2.9x). Mostly unchanged from S15.8 — the ColumnSet inlining in
  `take_columns` is a minor win since `gather_rows_core` dominates.

## Stage 17 — Arrow IPC via Vendored OdinArrow (done, 2026-08)

### Goal
Provide Arrow IPC file read/write for DataFrames by wiring the vendored
OdinArrow library into the dataframe package through a thin bridge.

### What was done
- Vendored OdinArrow (`github.com/TimeLord/OdinArrow`, Zlib license) as
  `src/odinarrow/` — 24 source files covering Array, builders, compute,
  IPC (file + stream), record batches, schema, and string/binary support.
- Built a static library with LZ4 + LZ4 frame + xxhash support for
  compression-enabled IPC files.
- Wrote `src/dataframe/arrow.odin`: thin bridge between the dataframe's
  runtime-typed `Column` / `ColumnSet` and OdinArrow's compile-time-typed
  `Array` / `Record_Batch`.
  - `dataframe_write_arrow(df, path)`: iterates columns, builds Arrow
    schema + arrays via OdinArrow builders, writes via `ipc_write_file`.
  - `dataframe_read_arrow(path)`: reads Arrow IPC, maps each Arrow field
    back to a dataframe `Column` (primitive bulk copy, string blob
    reconstruction with payload ownership transfer).
  - Special-cases bool (Odin byte-per-element ↔ Arrow bit-packed) and
    string (Odin headers+payload ↔ Arrow offsets+data).
  - Supported dtypes: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64,
    bool, string. Date/Datetime/Time/Duration (distinct i64) and
    List_Ref are not yet bridged.
- Wrote `src/dataframe/arrow_test.odin`: 9 round-trip tests covering
  i64, f64, bool, string, string+null, i32+null, multi-column, empty
  DataFrame, and u8/u16.

### Notes
- OdinArrow's `ipc_read_file` uses mmap on Linux; the read bridge copies
  buffers into Column-owned memory so the mmap can be released.
- `record_batch_make` shallow-copies the columns slice; after passing
  arrays to a batch, the batch owns the buffer lifetimes.
- Bool arrays require special handling: Column stores `[]bool` (1 byte
  per element) while Arrow bit-packs them into a bitmap.
- String round-trip builds a contiguous byte blob, constructs Odin string
  headers pointing into it, then transfers blob ownership via
  `col.payload`.

---

## Stage 18 — Compression subsystem (done, 2026-08)

### Goal
A small codec layer so Parquet and Arrow IPC can share one compression
entry point instead of each format vendoring its own glue.

### What was done
- Vendored `github.com/judah-caruso/odin-lz4` as `src/odin-lz4/` (LZ4 block,
  LZ4 frame, LZ4HC; C sources + build scripts for linux/mac/windows).
- Wrote `src/dataframe/compress.odin`: `Compression_Algorithm`
  (`None`, `LZ4_Block`, `LZ4_Frame`), the owned `Compressed_Buffer` handle
  with `compressed_buffer_destroy`, and `compress` / `decompress` procs that
  dispatch on the recorded algorithm. Explicit allocator throughout.
- `src/dataframe/compress_test.odin`: 9 tests (round-trips per algorithm,
  empty input, incompressible data, error paths).

### Notes
- Pure wiring over the vendored C library — no custom compression code
  (AGENTS.md principle 2). Parquet uses the block codec, Arrow IPC the frame
  codec; both formats call the same `compress`/`decompress`.

---

## Stage 19 — Parquet writer (done, 2026-08)

### Goal
Write DataFrames as Parquet byte streams (PLAIN encoding, optional LZ4),
reusing the Stage 17 Arrow bridge instead of writing a second typed encoder.

### What was done
- Wrote `src/dataframe/parquet_types.odin`: the Parquet wire enums
  (types, encodings, codecs, page types) shared by reader and writer.
- Wrote the writer half of `src/dataframe/parquet_thrift.odin`: minimal
  compact-protocol encoder for the subset of Parquet thrift structures we
  emit (a struct-scoped field-id stack keeps nesting correct).
- Wrote `src/dataframe/parquet_writer.odin`: `dataframe_write_parquet(df,
  options)` produces an owned byte slice. Columns are converted to OdinArrow
  arrays via the existing bridge and their buffer data emitted as PLAIN
  pages; `Parquet_Write_Options` carries compression (default LZ4) and row
  group size (default 65536).

### Notes
- Reusing the Arrow conversion means dtype support matches the Arrow bridge
  exactly; no per-dtype Parquet encoder was written.
- Dictionary/RLE encodings are out of scope; PLAIN only.

---

## Stage 20 — Parquet reader (done, 2026-08)

### Goal
Read back what Stage 19 writes (and reasonable third-party files): footer
metadata, schema mapping, column chunks, and page decoding.

### What was done
- Extended `src/dataframe/parquet_thrift.odin` with the compact-protocol
  reader: varint/zigzag, `Thrift_Reader` with a field-id stack so nested
  structs restore the enclosing field id on exit (`thrift_read_struct_begin`
  / `thrift_read_struct_end`), skip procs for unknown fields, and one
  deserialize proc per structure (file meta, schema element, logical type,
  row group, column chunk, column meta data, page header).
- Wrote `src/dataframe/parquet_reader.odin`: `dataframe_read_parquet` walks
  the footer, maps Parquet physical types (+ logical types) to dataframe
  dtypes, decodes PLAIN pages (with LZ4 decompression via Stage 18), and
  rebuilds string/blob columns; validity is reconstructed from definition
  levels where present.
- Fixed a real bug found by fuzzing truncated/nested footers during this
  stage: every deserialize proc now pairs `thrift_read_struct_begin` /
  `thrift_read_struct_end` so `last_fid` is restored when leaving a nested
  struct; previously an inner struct's final field id leaked into the outer
  parse and silently skipped outer fields.
- `src/dataframe/parquet_test.odin`: 10 round-trip tests (per dtype family,
  nulls, multi-row-group, compression off/on); `parquet_debug_test.odin`
  dumps raw structures for manual inspection.

### Notes
- The reader trusts chunk offsets from the footer; no corruption recovery.
- Same dtype coverage as the writer (the Arrow bridge set); List/Struct
  Parquet schemas are not supported yet.

---

## Stage 21 — Performance optimizations (done, 2026-08)

### Goal
Two hot-path kernels flagged by the Stage 15 baselines: hash grouping
(per-row encode + single global map) and sorting (comparator dispatch per
comparison). Both must preserve documented semantics exactly — group order,
source-order rows within groups, stable total order incl. NULL placement —
verified against the sequential/reference implementations.

### S21.1 Parallel partitioned hash grouping

- New `src/dataframe/parallel_groupby.odin`: rows split into contiguous
  chunks across the `src/parallel` pool (8 threads at >= 100K rows);
  each task encodes and hashes its range into thread-local
  map + first-seen order + encode buffer, all allocated from the caller's
  allocator; the main thread merges chunks in ascending row-range order,
  which reproduces first-appearance group order and source-order rows
  exactly. Ownership transfers explicitly (adopt vs merge+free) and error
  paths tear down every local.
- `dataframe_group_by` dispatches above `PARALLEL_GROUPBY_THRESHOLD`;
  below it the original loop runs as `group_rows_sequential`.
- `src/dataframe/parallel_groupby_test.odin`: 3 tests — exact
  sequential-vs-parallel equivalence on 60k multi-key rows with NULLs, a
  hand-checkable mod-3 pattern run far below threshold, and an end-to-end
  group_by+agg at 150k rows checked against recomputed expectations.

### S21.2 Radix argsort kernel

- New `src/dataframe/sort_radix.odin`: stable LSD base-256 radix argsort
  over u64 order keys extracted once per row (`order_key_u64` mirrors the
  documented total order: sign-bit flip for signed/temporal, IEEE-754
  canonical bit keys for floats, complement for descending). NULLs are
  stitched head/tail in source order before sorting valid rows. Each digit
  pass runs parallel histogram + scatter tasks with a sequential prefix-sum
  combine; passes whose byte vanishes in the cumulative OR mask are skipped.
- `argsort_key_columns` dispatches to it for a single non-string key at
  >= RADIX_SORT_THRESHOLD (4096) rows; multi-key and string sorts keep the
  legacy comparator path.
- `src/dataframe/sort_kernel_test.odin`: 9 tests — an O(n²)
  compare_rows-driven oracle proves permutation equality per dtype
  (ints incl. high-bit u64, floats incl. NaN/±0/inf, bools, temporals),
  NULL placement both directions, degenerate sizes, dispatch-threshold
  equivalence through the public API, and a sorted-output check at 200k
  rows.

### Benchmark results (best-of-N, same machine as earlier tables)

groupby (`odin run benchmarks/groupby.odin -file`, count+sum over 1000
groups):

| rows    | naive(ms) | groupby(ms) | vs Stage 7 baseline |
|---------|-----------|-------------|---------------------|
| 100K    | 2.7       | 6.6         | —                   |
| 1M      | 27.7      | 49.4        | 127 ms → 2.6x faster|
| 10M     | 251.4     | 400.5       | ~3x faster          |

sort (`odin run benchmarks/sort.odin -file`, 1M rows, gather included):

| dtype  | sort(ms) | vs Stage 5 baseline      |
|--------|----------|--------------------------|
| i32    | 94.4     | 301 ms → 3.2x faster     |
| i64    | 94.0     | ~3x faster               |
| f64    | 98.8     | ~3x faster               |
| string | 694.8    | legacy path, unchanged   |

### Notes
- Tools reused: `src/parallel` pool, built-in maps/dynamic arrays,
  `core:sort` untouched for legacy paths. Custom code is confined to the
  partitioned merge and the digit-pass kernels.
- Remaining levers (not done, no benchmark pressure yet): SIMD/groupwise
  key hashing inside encode_row, parallel radix combine, dictionary-encoded
  grouping for low-cardinality keys.

---

## Definition of done for each stage

- All `*_test.odin` for the stage pass: `odin test <pkg-dir>`.
- `odin run src/` demo still builds.
- Benchmarks for marked items exist and are run once; results recorded in the
  stage notes.
- Stage notes record which existing `core:`/in-tree tools were reused and any
  custom code written because no existing tool fit (AGENTS.md principle 2).
- DESIGN.md/ARCHITECTURE.md updated where behavior changed.
