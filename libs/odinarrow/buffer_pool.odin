package odinarrow

import "core:mem"

// Buffer_Pool is a free-list allocator that retains freed blocks (bucketed by
// size) and hands them back on the next allocation of the same size, instead of
// returning them to the OS. Reused blocks keep their pages resident, so a
// build → free → build loop avoids re-faulting fresh pages every time — the same
// reuse strategy Arrow's memory pool relies on to make repeated construction
// cheap.
//
// Use it by passing buffer_pool_allocator(&pool) as the builder's allocator.
//
// Not thread-safe: use one pool per thread (builders are single-threaded).
// All blocks are released by buffer_pool_destroy.

Buffer_Pool :: struct {
	backing:   mem.Allocator,
	free_list: map[int][dynamic]rawptr, // size -> stack of free user pointers
	bases:     [dynamic]rawptr,         // every backing base pointer (for destroy)
}

@(private="file")
_Pool_Header :: struct {
	size: int,    // user-requested size (bucket key)
	base: rawptr, // backing allocation start
}

@(private="file")
_POOL_HDR :: size_of(_Pool_Header)

buffer_pool_init :: proc(pool: ^Buffer_Pool, backing := context.allocator) {
	pool.backing = backing
	// Bookkeeping always lives in the backing allocator, never context.allocator:
	// callers commonly set context.allocator to this very pool, and routing the
	// pool's own metadata through itself is reentrant (use-after-free).
	pool.free_list = make(map[int][dynamic]rawptr, 16, backing)
	pool.bases     = make([dynamic]rawptr, 0, 0, backing)
}

// Release every block the pool ever allocated, plus its bookkeeping.
buffer_pool_destroy :: proc(pool: ^Buffer_Pool) {
	for base in pool.bases {
		mem.free(base, pool.backing)
	}
	delete(pool.bases)
	for _, stack in pool.free_list {
		delete(stack)
	}
	delete(pool.free_list)
	pool^ = {}
}

buffer_pool_allocator :: proc(pool: ^Buffer_Pool) -> mem.Allocator {
	return mem.Allocator{procedure = _buffer_pool_proc, data = pool}
}

@(private="file")
_buffer_pool_proc :: proc(
	allocator_data: rawptr, mode: mem.Allocator_Mode,
	size, alignment: int, old_memory: rawptr, old_size: int,
	loc := #caller_location,
) -> ([]byte, mem.Allocator_Error) {
	pool := cast(^Buffer_Pool)allocator_data

	switch mode {
	case .Alloc, .Alloc_Non_Zeroed:
		ptr, err := _pool_get(pool, size, alignment)
		if err != .None { return nil, err }
		if size == 0 { return nil, .None }
		if mode == .Alloc { mem.zero(ptr, size) }
		return (cast([^]byte)ptr)[:size], .None

	case .Free:
		if old_memory != nil { _pool_put(pool, old_memory) }
		return nil, .None

	case .Free_All:
		// A pool keeps memory owned for reuse; nothing is returned to the OS
		// until buffer_pool_destroy. Treat Free_All as a no-op.
		return nil, .None

	case .Resize, .Resize_Non_Zeroed:
		ptr, err := _pool_resize(pool, old_memory, old_size, size, alignment, mode == .Resize)
		if err != .None { return nil, err }
		if size == 0 { return nil, .None }
		return (cast([^]byte)ptr)[:size], .None

	case .Query_Features:
		set := (^mem.Allocator_Mode_Set)(old_memory)
		if set != nil {
			set^ = {.Alloc, .Alloc_Non_Zeroed, .Free, .Free_All, .Resize, .Resize_Non_Zeroed, .Query_Features}
		}
		return nil, .None

	case .Query_Info:
		return nil, .Mode_Not_Implemented
	}
	return nil, .None
}

// Hand back a block of exactly `size` bytes (reused if one is free), with the
// user pointer aligned to `alignment` and a header recording its size + base.
@(private="file")
_pool_get :: proc(pool: ^Buffer_Pool, size, alignment: int) -> (rawptr, mem.Allocator_Error) {
	if size == 0 { return nil, .None }
	if stack, ok := pool.free_list[size]; ok && len(stack) > 0 {
		s := stack
		user := pop(&s)
		pool.free_list[size] = s
		return user, .None
	}
	total := _POOL_HDR + alignment + size
	raw, err := mem.alloc_bytes_non_zeroed(total, alignment, pool.backing)
	if err != .None || raw == nil { return nil, .Out_Of_Memory }
	base := raw_data(raw)
	au   := uintptr(alignment)
	user := (uintptr(base) + uintptr(_POOL_HDR) + au - 1) & ~(au - 1)
	header := (^_Pool_Header)(rawptr(user - uintptr(_POOL_HDR)))
	header.size = size
	header.base = base
	append(&pool.bases, base)
	return rawptr(user), .None
}

@(private="file")
_pool_put :: proc(pool: ^Buffer_Pool, user: rawptr) {
	header := (^_Pool_Header)(rawptr(uintptr(user) - uintptr(_POOL_HDR)))
	sz := header.size
	s, ok := pool.free_list[sz]
	if !ok {
		// New bucket — back its stack with the pool's backing allocator so
		// appends never reenter the pool via context.allocator.
		s = make([dynamic]rawptr, 0, 0, pool.backing)
	}
	append(&s, user)
	pool.free_list[sz] = s
}

@(private="file")
_pool_resize :: proc(pool: ^Buffer_Pool, old_memory: rawptr, old_size, new_size, alignment: int, zeroed: bool) -> (rawptr, mem.Allocator_Error) {
	if old_memory == nil {
		p, err := _pool_get(pool, new_size, alignment)
		if err != .None { return nil, err }
		if zeroed && new_size > 0 { mem.zero(p, new_size) }
		return p, .None
	}
	if new_size == 0 {
		_pool_put(pool, old_memory)
		return nil, .None
	}
	header   := (^_Pool_Header)(rawptr(uintptr(old_memory) - uintptr(_POOL_HDR)))
	capacity := header.size
	if new_size <= capacity {
		if zeroed && new_size > old_size {
			mem.zero(rawptr(uintptr(old_memory) + uintptr(old_size)), new_size - old_size)
		}
		return old_memory, .None
	}
	// Grow: take a bigger block, copy the live bytes, recycle the old one.
	p, err := _pool_get(pool, new_size, alignment)
	if err != .None { return nil, err }
	keep := min(old_size, new_size)
	if keep > 0 { mem.copy(p, old_memory, keep) }
	if zeroed && new_size > keep {
		mem.zero(rawptr(uintptr(p) + uintptr(keep)), new_size - keep)
	}
	_pool_put(pool, old_memory)
	return p, .None
}
