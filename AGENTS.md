# Project Rules

This project adds **dataframe support for the Odin programming language**: a
native, columnar dataframe and query-engine library inspired by the breadth of
Pandas and the columnar/lazy architecture of Polars.

This is not an exercise in reinventing the wheel. The goal is to wire existing
Odin tooling together into a data API: the `core:` standard library (sort,
math, encoding, thread/sync, container, simd, prof), the in-tree `libs/parallel`
thread-pool helper, and the Odin language itself (generics, `typeid`, explicit
allocators) are the building blocks. Custom code is written only where no
existing tool fits, and then as thinly as possible so the pieces stay
swappable. Reusing the stdlib keeps the surface small, the maintenance burden
low, and the library extensible.

Do not implement a Python-style dataframe API literally. Design an idiomatic
Odin API around typed columnar data, expressions, lazy query plans, explicit
memory ownership, and eventual parallel execution.

## Principles

1. Prefer simple, idiomatic Odin over abstractions copied from other languages.
2. **Prefer the existing `core:` stdlib and in-tree helpers; write custom
   code only where nothing fits, and keep it minimal so tools stay swappable.**
3. Data should be column-oriented.
4. Avoid boxing values on hot paths.
5. Avoid unnecessary allocations.
6. Preserve allocator ownership explicitly.
7. Never silently convert between incompatible types.
8. NULL and zero/empty values must remain distinct.
9. Every public operation requires tests.
10. Performance-sensitive operations require benchmarks.
11. Do not optimize without a benchmark or profiler result.

## API

Public APIs should:
- be predictable
- use Odin naming conventions
- expose ownership clearly
- return useful errors
- avoid hidden allocations

## Development

Before implementing a major subsystem:
1. inspect existing code
2. write/update DESIGN.md
3. propose the API
4. implement tests
5. implement the feature
6. run all tests
7. add benchmarks where appropriate

Never rewrite large portions of the project without explaining why.

## Working notes for this repo

- Layout and entry point are documented in ARCHITECTURE.md.
- Work one milestone at a time; tasks are numbered in ROADMAP.md.
- Tests run with: `odin test <dir>`.
- Build the demo with: `odin run src/`.
- Before any benchmark-driven optimization, read DESIGN.md §14 (Performance).
