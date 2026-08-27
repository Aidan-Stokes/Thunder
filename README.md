# Thunder

A native, columnar dataframe and query-engine library for the
[Odin](https://odin-lang.org/) programming language.  Inspired by the breadth
of Pandas and the columnar/lazy architecture of Polars, but designed as an
idiomatic Odin API around typed columnar data, expressions, explicit memory
ownership, and eventual parallel execution.

> **Status:** 469 passing tests · parallel CSV · expression engine · lazy
> query planner · Arrow IPC · Parquet read/write · parallel groupby + radix
> sort · temporal / calendar types.

---

## Features

| Area | What you get |
|------|-------------|
| **DataFrame & Column** | Typed, column-oriented DataFrames with explicit NULL tracking via validity bitmaps. Any Odin type is supported as a column dtype. |
| **Expressions** | A full expression tree: arithmetic, comparisons, boolean logic, aggregations (`sum`, `mean`, `var`, `std`, `median`, `quantile`, `min`, `max`, `mode`, `skew`, `kurtosis`, `product`), window functions (`row_number`, `rank`, `cum_sum`, `shift`, `rolling`, `ewm`), null handling (`fill_null`, `forward_fill`, `backward_fill`, `interpolate`, `coalesce`, `drop_nulls`), and string operations (`concat_str`, `search_sorted`). |
| **Eager API** | `filter`, `select`, `sort`, `head`/`tail`/`slice`, `group_by` + `agg`, `join` (inner / left / right / full / semi / anti / cross), `asof_join`, `unique`, `partition_by`, `melt`, `pivot`, `explode`, `unnest`, `with_columns`, `drop`. |
| **Lazy query planner** | Build a logical plan (`scan_csv` → `filter` → `select` → `sort` → `group_by` → `agg` → `join` → `limit`), then `collect` to execute. Predicate/projection push-down optimizer. |
| **CSV** | Read with automatic type inference, schema-constrained reads, column selection, and **parallel parsing** for files ≥ 1 MB (thread-pooled via `libs/parallel`). Write to CSV. |
| **JSON / NDJSON** | Read and write both JSON (array-of-objects) and newline-delimited JSON. Schema-constrained reads supported. |
| **Parquet** | Read and write Parquet with PLAIN + LZ4 compression, via vendored odinarrow + odin-lz4. |
| **Arrow IPC** | Read and write Arrow IPC (Feather v2) via vendored OdinArrow. |
| **Compression** | LZ4 block and frame compression, exposed as a standalone `compress`/`decompress` API. |
| **Temporal** | `Date`, `Datetime`, `Time`, `Duration` types with `date_range`, `truncate`, and `dt_*` column accessors (`dt_year`, `dt_month`, `dt_day`, `dt_hour`, etc.). |
| **Categorical** | Categorical and Enum columns from string arrays. |
| **List columns** | Variable-length list columns with typed offsets. |
| **Parallel kernels** | Parallel SIMD wrappers, parallel row-gather, parallel validity bitmap construction, parallel groupby, radix argsort — all automatically used when data exceeds tuned thresholds. |

---

## Quick start

### Prerequisites

- [Odin](https://odin-lang.org/) compiler (`dev-2026-08` or later)

### Run the demo

```sh
odin run src/
```

### Use as a library

Add the repo to your project and import the `dataframe` package:

```odin
package main

import "dataframe"
import "dataframe/expr"
import "core:fmt"

main :: proc() {
    // --- build a DataFrame from columns ---
    age, _    := dataframe.column_from("age", []i32{25, 30, 35, 40})
    name, _   := dataframe.column_from("name", []string{"ada", "grace", "katherine", "margaret"})
    dept, _   := dataframe.column_from("dept", []string{"eng", "sales", "eng", "eng"})
    salary, _ := dataframe.column_from("salary", []f64{150_000, 90_000, 0, 210_000})

    df, _ := dataframe.dataframe_from_columns([]^dataframe.Column{&age, &name, &dept, &salary})
    defer dataframe.dataframe_destroy(&df)

    // --- expression context (reusable for all expressions) ---
    ctx := expr.context_create(context.allocator)
    defer expr.context_destroy(&ctx)

    // --- filter → sort → head pipeline ---
    filtered := dataframe.filter(&df,
        expr.ge(&ctx, expr.col(&ctx, "age"), expr.lit(&ctx, i32(30)))) or_return
    defer dataframe.dataframe_destroy(&filtered)

    sorted := dataframe.sort(&filtered,
        []dataframe.Sort_Key{dataframe.sort_key("salary", .Desc)}) or_return
    defer dataframe.dataframe_destroy(&sorted)

    top := dataframe.head(&sorted, 2) or_return
    defer dataframe.dataframe_destroy(&top)

    dataframe.dataframe_print(&top)

    // --- group_by + aggregation ---
    gb := dataframe.group_by(&df, []^expr.Expr{expr.col(&ctx, "dept")}) or_return
    defer dataframe.dataframe_group_by_destroy(&gb)

    grouped := dataframe.agg(&gb, []^expr.Expr{
        expr.alias(&ctx, expr.mean_(&ctx, expr.col(&ctx, "salary")), "mean_salary"),
        expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "salary")), "n"),
    }) or_return
    defer dataframe.dataframe_destroy(&grouped)

    dataframe.dataframe_print(&grouped)
}
```

### Read from CSV

```odin
df, err := dataframe.dataframe_read_csv("data.csv")
defer dataframe.dataframe_destroy(&df)

// With an explicit schema:
schema, _ := dataframe.schema_create([]dataframe.Field{
    {"id",    typeid_of(i64)},
    {"score", typeid_of(f64)},
    {"name",  typeid_of(string)},
})
defer dataframe.schema_destroy(&schema)

df, err = dataframe.dataframe_read_csv_with_schema("data.csv", schema)
```

Files ≥ 1 MB are automatically parsed in parallel (8 threads by default).

### Lazy query planner

```odin
import "dataframe/lazy"

lf := lazy.scan_csv("sales.csv")
lf = lazy.filter(lf, expr.gt(&ctx, expr.col(&ctx, "amount"), expr.lit(&ctx, f64(100))))
lf = lazy.sort(lf, []dataframe.Sort_Key{dataframe.sort_key("date", .Desc)})
lf = lazy.limit(lf, 10)

result, err := lazy.collect(lf)
defer dataframe.dataframe_destroy(&result)
```

### Joins

```odin
inner, _ := dataframe.dataframe_inner_join(&employees, &departments,
    []string{"dept_id"}, []string{"id"})

asof, _ := dataframe.dataframe_asof_join(&ticks, &quotes,
    "timestamp", "timestamp",
    []string{"symbol"}, []string{"symbol"},
    .Backward)
```

---

## Supported column types

Any Odin type can back a column.  First-class support (I/O, expressions,
aggregations):

`bool`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `int`, `uint`,
`f32`, `f64`, `string`, `Date`, `Datetime`, `Time`, `Duration`

Arbitrary Odin structs are also supported via the generic `column_from`
constructor; expression evaluation and I/O work only over the types above.

---

## Platform support

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux x64** | Supported | Primary development platform |
| **macOS ARM64** (Apple Silicon) | Supported | |
| **macOS x64** (Intel) | Supported | AVX2 path requires Haswell+ CPU (2013+) |
| **Windows x64** | Supported | AVX2 path requires Haswell+ CPU (2013+) |
| **Linux ARM64** | Supported | Scalar fallback for i32 min/max kernel |

All SIMD-accelerated kernels degrade gracefully on non-x86 architectures —
the `core:simd` abstraction decomposes wide vectors into native NEON ops,
and the AVX2-only `_min_max_i32_simd` kernel falls back to a scalar loop on
ARM64.  LZ4 compression ships pre-built static archives for Linux, macOS,
and Windows.

---

## Project structure

```
Thunder/
├── libs/
│   ├── parallel/       in-tree thread-pool helper (do_parallel)
│   ├── odinarrow/      vendored OdinArrow (Arrow IPC)
│   └── odin-lz4/       vendored odin-lz4 (LZ4 compression)
├── src/
│   ├── main.odin       demo entry point
│   └── dataframe/      core library package
│       ├── expr/       expression tree (dependency-free)
│       └── lazy/       lazy query planner + optimizer + executor
└── benchmarks/         manual-timing benchmarks
```

Import paths from your own package:

```odin
import "dataframe"
import "dataframe/expr"
import "dataframe/lazy"
```

(Adjust the path to match where you place the library in your project.)

---

## Documentation

| Document | Contents |
|----------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Repository layout, package responsibilities, data flow, build/test commands |
| [DESIGN.md](DESIGN.md) | Design decisions, type system, memory model, performance strategy |
| [ROADMAP.md](ROADMAP.md) | Stage-by-stage implementation history and future plan |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development workflow, running tests, code conventions |

---

## Running tests

```sh
odin test src/dataframe          # full suite (469 tests)
odin test src/dataframe -define:ODIN_TEST_NAMES='csv_parallel_with_schema'  # single test
```

## Running benchmarks

```sh
odin run benchmarks/sort.odin -file
odin run benchmarks/groupby.odin -file
odin run benchmarks/csv.odin -file
```

---

## Vendored dependencies

| Library | License | Location |
|---------|---------|----------|
| [OdinArrow](https://github.com/TimeLord/OdinArrow) | Zlib | `libs/odinarrow/` |
| [odin-lz4](https://github.com/judah-caruso/odin-lz4) | BSD-2-Clause | `libs/odin-lz4/` |
| `libs/parallel` | MIT (Joao Nuno Carvalho) | `libs/parallel/` |

---

## License

The Thunder library itself is released under the **MIT License**.
Vendored dependencies retain their original licenses (see above).
