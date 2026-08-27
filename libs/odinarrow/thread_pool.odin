package odinarrow

import "base:runtime"
import "core:os"
import "core:sync"
import "core:thread"

// A process-lifetime worker pool for the parallel compute kernels. Workers are
// created once (lazily) and parked on a condition variable, so each parallel
// call is a broadcast + barrier instead of spawning and tearing down threads
// (~50µs per call). One partition is assigned per worker (work is already
// chunked by the caller), so there is no task queue or work stealing.
//
// Usage model is single-caller: the compute kernels are invoked from one thread
// at a time (no nested or concurrent parallelism), which keeps the barrier logic
// simple — a job is fully complete before the next one starts.
//
// The pool's own threads/slices are allocated on the raw heap, not the caller's
// context allocator, so they never show up as "leaks" in a tracking allocator
// (e.g. the test runner): the pool is a singleton that lives until process exit.

_Pool_Worker :: struct {
	pool: ^_Compute_Pool,
	id:   int,
}

_Compute_Pool :: struct {
	workers:   []^thread.Thread,
	wdata:     []_Pool_Worker,
	n:         int, // number of worker threads (== logical cores)

	submit:    sync.Mutex, // serialises whole jobs so concurrent callers don't clash
	mutex:     sync.Mutex,
	work_cond: sync.Cond, // wakes workers when a job is posted
	done_cond: sync.Cond, // wakes the caller when the job is complete

	gen:       u64, // job generation; workers wait for it to advance
	job:       proc(ctx: rawptr, idx: int),
	ctx:       rawptr,
	job_n:     int, // number of partitions in the current job
	completed: int, // partitions finished in the current job
	shutdown:  bool,
}

@(private) g_pool:      _Compute_Pool
@(private) g_pool_once: sync.Once

_pool_get :: proc() -> ^_Compute_Pool {
	sync.once_do(&g_pool_once, _pool_init)
	return &g_pool
}

@(private)
_pool_init :: proc() {
	// Keep the pool's allocations off any caller's (tracking) allocator.
	context.allocator = runtime.heap_allocator()

	n := os.get_processor_core_count()
	if n < 1 { n = 1 }
	g_pool.n       = n
	g_pool.workers = make([]^thread.Thread, n)
	g_pool.wdata   = make([]_Pool_Worker, n)
	for i in 0..<n {
		g_pool.wdata[i] = _Pool_Worker{ pool = &g_pool, id = i }
		g_pool.workers[i] = thread.create_and_start_with_poly_data(&g_pool.wdata[i], _pool_worker)
	}
}

@(private)
_pool_worker :: proc(w: ^_Pool_Worker) {
	p := w.pool
	last_gen: u64 = 0
	for {
		sync.mutex_lock(&p.mutex)
		for p.gen == last_gen && !p.shutdown {
			sync.cond_wait(&p.work_cond, &p.mutex)
		}
		if p.shutdown {
			sync.mutex_unlock(&p.mutex)
			return
		}
		last_gen = p.gen
		job := p.job
		ctx := p.ctx
		jn  := p.job_n
		sync.mutex_unlock(&p.mutex)

		if w.id < jn {
			job(ctx, w.id)
			sync.mutex_lock(&p.mutex)
			p.completed += 1
			if p.completed == jn {
				sync.cond_signal(&p.done_cond)
			}
			sync.mutex_unlock(&p.mutex)
		}
	}
}

// Run job(ctx, i) for i in 0..<nt across the pool's workers and block until all
// nt partitions complete. nt must be <= the worker count (callers cap it via
// _resolve_threads). nt <= 1 runs inline on the calling thread.
_pool_run :: proc(nt: int, job: proc(ctx: rawptr, idx: int), ctx: rawptr) {
	if nt <= 1 {
		job(ctx, 0)
		return
	}
	p := _pool_get()
	n := min(nt, p.n)

	// One job uses every worker, so serialise whole jobs: concurrent callers
	// (e.g. the multi-threaded test runner) each take the pool in turn rather
	// than clobbering the shared job/gen/completed state.
	sync.mutex_lock(&p.submit)
	defer sync.mutex_unlock(&p.submit)

	sync.mutex_lock(&p.mutex)
	p.job       = job
	p.ctx       = ctx
	p.job_n     = n
	p.completed = 0
	p.gen      += 1
	sync.cond_broadcast(&p.work_cond)
	for p.completed < n {
		sync.cond_wait(&p.done_cond, &p.mutex)
	}
	sync.mutex_unlock(&p.mutex)
}

// The pool is a process-lifetime singleton: its workers park until the program
// exits, when the OS reclaims them along with the heap-allocated bookkeeping.
// The `shutdown` flag is read by the workers so an explicit teardown entry
// point could be added later.
