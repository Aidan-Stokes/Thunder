# ARCHITECTURE.md — repository layout and data flow

## 1. Repository layout

Odin packages are directories, and tests are conventionally colocated in the
package as `*_test.odin` files run with `odin test <pkg-dir>`. The library
therefore lives under `src/` as importable sibling packages rather than in a
Plan.md-style `tests/` top-level directory (the deviation is an Odin idiom,
not a scope change).

```
sim/
├── AGENTS.md
├── DESIGN.md
├── ARCHITECTURE.md
├── ROADMAP.md
├── LICENSE
│
├── libs/                          vendored + in-tree helper packages
│   ├── odinarrow/                 vendored OdinArrow (Arrow IPC, S17)
│   │                              Zlib license — github.com/TimeLord/OdinArrow
│   ├── odin-lz4/                  vendored odin-lz4 (LZ4 block/frame/HC, S18)
│   │                              Zlib license — github.com/judah-caruso/odin-lz4
│   └── parallel/                  existing in-tree do_parallel pool helper
│
├── src/                           package main — demo + entry point
│   ├── main.odin                  `odin run src/`
│   │
│   ├── dataframe/                 package dataframe (public root API)
│   │   ├── dtype.odin            Field, Schema (typeid-based)
│   │   ├── error.odin            Error
│   │   ├── column.odin           Column (any-type buffer) + generic accessors
│   │   ├── column_test.odin
│   │   ├── column_set.odin      ColumnSet (SoA layout) + CRUD + views
│   │   ├── column_set_test.odin
│   │   ├── dataframe.odin        DataFrame: create/destroy/add/remove/get
│   │   ├── dataframe_test.odin
│   │   ├── chain.odin            S10.2 short aliases (dataframe.filter, …)
│   │   ├── chain_test.odin
│   │   ├── filter.odin           filter / take / head / tail / slice / limit
│   │   ├── select.odin           select / select_by_name / with_columns / drop
│   │   ├── group_keys.odin       canonical row-key encoding for grouping
│   │   ├── unique.odin           unique by named columns (per-column NULL keys)
│   │   ├── partition_by.odin     split into DataFrames by group keys
│   │   ├── sort.odin             sort / argsort / sort_by (core:sort-driven)
│   │   ├── sort_test.odin
│   │   ├── property_test.odin    S10.6 seeded property tests
│   │   ├── print.odin            S10.3 dataframe_to_string / dataframe_print
│   │   ├── print_test.odin
│   │   ├── ops_test.odin         Stage 4 tests (filter/select/unique/partition)
│   │   ├── expr_eval.odin        expression -> Column evaluation
│   │   ├── typecheck.odin        static expression type checking
│   │   ├── expr_test.odin
│   │   ├── window.odin           S14.4/14.5 window functions (over partitions)
│   │   ├── window_test.odin
│   │   ├── agg.odin              column agg kernels (row-subset aware)
│   │   ├── agg_test.odin         Stage 6 tests
│   │   ├── groupby.odin          Group_By, group_by / group_by_agg / destroy
│   │   ├── groupby_test.odin     Stage 7 tests
│   │   ├── join.odin             hash joins (inner/left/right/full/semi/anti/cross)
│   │   ├── join_test.odin        Stage 8 tests
│   │   ├── asof_join.odin        as-of joins (S14.6, bucket + binary search)
│   │   ├── asof_join_test.odin
│   │   ├── csv.odin              read/write CSV + type inference (S9)
│   │   ├── csv_test.odin
│   │   ├── csv_parallel.odin      parallel CSV reader (S15)
│   │   ├── csv_parallel_test.odin
│   │   ├── json.odin              read/write JSON + schema mode (S13)
│   │   ├── json_test.odin
│   │   ├── compress.odin           LZ4 block + frame compression (S18)
│   │   ├── compress_test.odin
│   │   ├── arrow.odin              Arrow IPC bridge (S17)
│   │   ├── arrow_test.odin
│   │   ├── parquet_types.odin      Parquet wire enums (S19/S20)
│   │   ├── parquet_thrift.odin     compact-protocol reader/writer (S19/S20)
│   │   ├── parquet_writer.odin     dataframe_write_parquet, PLAIN + LZ4 (S19)
│   │   ├── parquet_reader.odin     dataframe_read_parquet (S20)
│   │   ├── parquet_test.odin       10 round-trip tests
│   │   ├── parallel_groupby.odin   S21.1 partitioned hash grouping kernel
│   │   ├── parallel_groupby_test.odin
│   │   ├── sort_radix.odin         S21.2 single-key radix argsort kernel
│   │   ├── sort_kernel_test.odin
│   │   ├── property_test.odin    S10.6 seeded property tests
│   │   ├── lazy/                 package lazy — Stage 11 lazy engine
│   │   │   ├── plan.odin         Logical_Plan union + node structs
│   │   │   ├── lazyframe.odin    LazyFrame + builders (scan/filter/select/…)
│   │   │   ├── physical.odin     eager-backed physical operators
│   │   │   ├── executor.odin     collect() + exec_node plan walk
│   │   │   └── lazy_test.odin    S11.5 tests (side-effect-free, equals-eager)
│   │   │
│   │   ├── expr/                 package expr — expression tree (dependency-free)
│   │   │   └── expr.odin         Expr union, col()/lit()/op constructors
│   │   │
│   │   (planned: stats.odin)
│   │
└── benchmarks/                   package bench — manual-timing benchmarks
    ├── column.odin               1K/100K/1M rows
    ├── expr.odin                 expression vs naive loop
    ├── ops.odin                  filter + select
    ├── sort.odin                 1M rows per dtype (radix kernel for numeric, S21)
    ├── agg.odin                  whole-column aggs (Stage 6)
    ├── groupby.odin              group_by + agg (parallel kernel at >= 100K, S21)
    ├── csv.odin                  CSV parse 1K/100K/1M (Stage 9)
    └── lazy.odin                 lazy vs eager 1M rows (Stage 11)
```

Import paths (relative to `src/`): `import "dataframe"`,
`import "dataframe/expr"`, `import "dataframe/lazy"`, `import "dataframe/io"`.
`arrow.odin` imports vendored OdinArrow as `import "../../libs/odinarrow"`.
Because Odin resolves imports relative to the importing package's directory,
a package nested under `dataframe/` (e.g. `dataframe/lazy`) imports its
siblings by walking up: `import "../../dataframe"`, `import "../../dataframe/expr"`.

This library **reuses existing Odin tooling rather than reinventing it**
(AGENTS.md principle 2, DESIGN.md §1.1): `core:sort` powers sorting,
`core:math`/`core:strings` power the expression library, `core:encoding/*`
powers I/O, `core:thread`/`core:sync` plus the in-tree `libs/parallel` pool
power the parallel executor, and vendored `libs/odinarrow` (OdinArrow, Zlib
license) powers Arrow IPC read/write. Custom files stay thin and focused
(`group_keys.odin` is the canonical encoding that makes `unique`/`partition_by`
work with Odin's built-in hash maps; `sort.odin` is the comparator glue over
`core:sort`'s `Interface`; `arrow.odin` is the type-mapping bridge between
the runtime-typed `Column` and OdinArrow's compile-time-typed `Array`).

## 2. Package responsibilities

| Package | Responsibility | Depends on |
|---|---|---|
| `dataframe` | Public eager API: DataFrame, Column, Schema, filter/sort/unique/partition/groupby/join, Arrow IPC | `dataframe/expr`, `core:sort`, `core:math`, `odinarrow` |
| `dataframe/expr` | Expr tree, evaluation, type checking | `core:math`, `core:strings` |
| `dataframe/lazy` | Logical plan, optimizer, physical plan, executor | `dataframe`, `dataframe/expr` |
| `dataframe/io` | CSV read/write + type inference | `dataframe`, `core:encoding/csv` |
| `odinarrow` | (vendored) Arrow IPC file/stream, Array builders, compute | `core:mem`, `core:encoding` |
| `parallel` | (existing) thread-pool do_parallel helper; adopted by executor in the performance stage | `core:thread` |

The "Depends on" column is the point of DESIGN.md §1.1: the dataframe package
glues the `core:` stdlib's sorted/math/encoding/thread machinery into a data
API. Dependency direction between our packages is strictly downward: `expr`
knows nothing about `lazy`; `lazy` builds on the eager engine for now (it can
materialize via it), and eventually the executor becomes the sole execution
path. `core:` packages sit below everything and are never wrapped unless a
benchmark justifies it.

## 3. Data flow

### 3.1 Columnar core

```
data:  rawptr + dtype:typeid + elem_size + count   ->  typed view via $T
valid: []bool                                       ->  NULL semantics
name + Schema (Field name, typeid)                  ->  logical contract
```

A column stores a type-tagged raw element buffer. Kernels obtain a typed
view once per column (`transmute([]T)(Raw_Slice{...})` after a `typeid`
equality check) and loop elementwise over contiguous data (no per-element
dispatch, no boxing). `Schema`/`Field` carry the column names and `typeid`s
as the DataFrame's logical contract.

### 3.2 Eager operation

```
DataFrame ──select/filter/sort──▶ indices or new buffers ──▶ new DataFrame
     ▲                                                                │
     └───────────────────── copy / materialize ◀──────────────────────┘
```

Filters and projections materialize new column buffers. Sorts produce a
permutation (`[]int`) which downstream ops consume without reordering unless
explicitly materialized.

### 3.3 Lazy path (full pipeline, Plan.md §16)

```
DataFrame / scan_csv
        │  .lazy()
        ▼
   LazyFrame ──filter/select/group_by/sort/join──▶ LazyFrame (builds plan)
        │  .collect()
        ▼
  Logical_Plan
        │  optimizer passes (pushdowns, folding, CSE)
        ▼
  Physical_Plan
        │  executor (per-operator kernels; parallel later)
        ▼
      DataFrame
```

`LazyFrame` methods never execute; each call appends a node to the plan.
`collect()` runs the optimizer then the executor.

## 4. Cross-cutting conventions

- Ownership: `create`/`destroy` pairs; allocator captured in owning structs;
  deep copies default, views are explicit and borrow.
- Errors: `dataframe.Error` returned as the last result value so Odin's
  `or_return`/`or_else` operators apply (DESIGN.md §12).
- Tests: `*_test.odin` colocated per package; `odin test src/dataframe`,
  `odin test src/dataframe/expr`, etc.
- Benchmarks: manual-timing `main` procs in `benchmarks/` (no stdlib harness),
  run with `odin run benchmarks/filter.odin`.

## 5. Notes on the existing `libs/parallel`

`libs/parallel` is a self-contained `do_parallel` thread-pool helper with a
generic `ParallelInfo` payload (upstream by Joao Nuno Carvalho, MIT license).
It is used **unmodified** — the file must not be edited.

API: `do_parallel(pool, procedure, data, num_elem, nthreads)` — allocates
`ParallelInfo(T)` internally, partitions `[0, num_elem)` across `nthreads`,
dispatches tasks via `pool_add_task`, calls `pool_start` + `pool_finish`,
then frees the info struct. The caller owns pool init/destroy.

**Important contract**: `data` is `^$T` and `ParallelInfo.sl` is `^T`.
The task proc receives `t.data` as `rawptr` and must cast to
`^ParallelInfo(T)`. Passing `&x` (pointer-to-pointer) changes the generic
`$T` — always pass the pointer directly.

Adopted in S15.4 for parallel join probe (`join.odin:334`) and available
for future parallel operators. The pool lifecycle stays in the caller
(`pool_init` / `pool_destroy`); `do_parallel` handles start/finish.

## 6. Build / test commands

- Build demo: `odin run src/`
- Check a library package: `odin check src/dataframe -build-mode:object`
  (`odin check` alone expects an executable entry point, so the library build
  mode is required).
- Tests: `odin test src/dataframe` (+ `/expr`, `/lazy`, `/io`)
- Benchmarks: single-file packages, run with
  `odin run benchmarks/column.odin -file` (the `-file` flag treats a standalone
  file as a package; relative imports resolve from the file's directory).
