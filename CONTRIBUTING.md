# Contributing to Thunder

Thanks for your interest in contributing.  This document covers the development
workflow, test conventions, and code style used in this project.

---

## Prerequisites

- **Odin compiler** `dev-2026-08` or later — check with `odin version`
- **Git**
- No external build tools required (no C compiler, no Make)

---

## Development workflow

### 1. Read the project rules

`AGENTS.md` contains the principles that govern every change.  Read it before
submitting a patch — it covers allocator ownership, NULL semantics, testing
requirements, and the "build on existing tooling" philosophy.

### 2. Understand the architecture

| Document | What it covers |
|----------|----------------|
| `ARCHITECTURE.md` | Repository layout, package responsibilities, data flow |
| `DESIGN.md` | Type system, memory model, performance strategy, design decisions |
| `ROADMAP.md` | Stage-by-stage milestones and what's shipped vs planned |

### 3. Work one stage at a time

The project is organized into numbered stages in `ROADMAP.md`.  Pick the next
uncompleted stage, read its tasks, and implement them in order.  Each stage ends
with all tests green and, where noted, a benchmark.

### 4. Branch, implement, test, submit

```sh
git checkout -b my-feature
# ... make changes ...
odin test src/dataframe
git add -A && git commit -m "stage N.M: description"
git push -u origin my-feature
```

---

## Running tests

```sh
# Full suite (currently 469 tests)
odin test src/dataframe

# Single test by name
odin test src/dataframe -define:ODIN_TEST_NAMES='csv_parallel_with_schema'

# Lazy engine tests
odin test src/dataframe/lazy

# Expression tests
odin test src/dataframe/expr
```

All tests use `core:testing`.  The leak tracker reports per-test allocations
automatically — a `[WARN]` line means a leak in that test.

---

## Running benchmarks

```sh
odin run benchmarks/sort.odin -file
odin run benchmarks/groupby.odin -file
odin run benchmarks/csv.odin -file
odin run benchmarks/column.odin -file
odin run benchmarks/expr.odin -file
odin run benchmarks/ops.odin -file
odin run benchmarks/lazy.odin -file
odin run benchmarks/agg.odin -file
odin run benchmarks/parallel.odin -file
```

Benchmarks are manual-timing procs (no stdlib harness).  Do not optimize
without a benchmark or profiler result (`DESIGN.md §14`).

---

## Code conventions

### Naming

- Use Odin standard naming: `snake_case` for procs/fields, `PascalCase` for
  types/enums, `SCREAMING_SNAKE` for constants.
- Public API procs follow the pattern `dataframe_<verb>` (e.g.
  `dataframe_filter`, `dataframe_sort`).  Short aliases live in `chain.odin`.

### Memory

- Every public operation takes an explicit `allocator` parameter (defaulting
  to `context.allocator`).
- `dataframe_destroy` / `column_destroy` / `schema_destroy` are the ownership
  transfer points — callers must call them when done.
- `dataframe_from_columns` **takes ownership** and zeroes source Column structs;
  callers must not use source columns after passing them in.

### NULL handling

- NULL is distinct from zero/empty — stored in a packed `[]u64` validity bitmap.
- `column_get` returns a `(value, valid: bool, error)` triple.
- Never silently convert between incompatible types.

### Parallel code

- Parallelism uses `libs/parallel` (`do_parallel`) with a threshold +
  thread-count constant at the top of each parallel file.
- Callers must pass `nthreads = min(T, n)` to `do_parallel`.
- Parallel kernels are internal (`@(private)`); the public API picks them up
  automatically when thresholds are exceeded.

### Testing

- Every public operation requires at least one test.
- Test files are colocated in the package as `*_test.odin`.
- Use `testing.expect` for assertions; use `fmt.tprintf` for descriptive
  failure messages.
- Property tests use seeded PRNGs (`property_test.odin`).

---

## Adding a new feature

1. **Inspect** existing code for the closest analogy.
2. **Update** `DESIGN.md` if the feature touches the type system, memory model,
   or public API shape.
3. **Add tasks** to `ROADMAP.md` under the appropriate stage.
4. **Write tests first** — ensure they fail before the implementation.
5. **Implement** the feature, keeping it minimal and idiomatic.
6. **Run the full suite** — `odin test src/dataframe` must pass.
7. **Add benchmarks** if the feature touches hot paths.

---

## Bug reports

Open an issue on [GitHub](https://github.com/Aidan-Stokes/Thunder/issues) with:

- A minimal reproduction (code snippet or CSV file)
- Expected vs actual behavior
- Odin version (`odin version` output)
- OS / architecture

---

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.
