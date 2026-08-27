// op_arena.odin — S15.6 per-operator arena for temporary allocations.
//
// An OpArena wraps mem.Dynamic_Arena to provide a bump allocator whose
// backing memory is freed in bulk at the end of an operator.  Individual
// free calls (column_destroy, delete) are silently ignored by the arena,
// so existing destroy code works without modification.
//
// Typical usage:
//
//     oa: OpArena
//     op_arena_init(&oa, context.allocator)
//     defer op_arena_destroy(&oa)
//     alloc := op_arena_allocator(&oa)
//     ... use alloc for temporaries ...

package dataframe

import "core:mem"

// OpArena is a per-operator arena allocator backed by Dynamic_Arena.
// Allocations are bump-pointer; individual frees are no-ops.
// Bulk-free happens in op_arena_destroy.
OpArena :: struct {
	arena: mem.Dynamic_Arena,
}

// op_arena_init initializes the arena with the given backing allocator.
// The arena grows automatically as needed.
op_arena_init :: proc(a: ^OpArena, backing: mem.Allocator) {
	mem.dynamic_arena_init(&a.arena, backing, backing)
}

// op_arena_destroy frees all memory allocated through the arena.
// After this call, any pointers obtained from op_arena_allocator are invalid.
op_arena_destroy :: proc(a: ^OpArena) {
	mem.dynamic_arena_destroy(&a.arena)
}

// op_arena_allocator returns a mem.Allocator that bump-allocates from the arena.
// Individual free calls are silently ignored.
op_arena_allocator :: proc(a: ^OpArena) -> mem.Allocator {
	return mem.dynamic_arena_allocator(&a.arena)
}
