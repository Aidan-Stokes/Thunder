package odinarrow

import "base:intrinsics"

// SIMD-accelerated inner kernels.  All procs assume offset == 0; callers must
// advance the pointer by the array offset before calling.
//
// Strategy:
//   f64 sum : 4-wide SIMD with 4 unrolled accumulators (hides 4-cycle VADDPD latency)
//   i32 sum : 4-wide i64 SIMD (sign-extends i32 → i64 to prevent overflow)
//   min/max : 8-way scalar unroll so LLVM emits vpminsd/vpmaxsd ymm (AVX2)
//   min+max : single pass combined; ~2× faster than two separate passes

// ── f64 sum ───────────────────────────────────────────────────────────────────

_sum_f64_simd :: #force_inline proc "contextless" (data: [^]f64, n: int) -> f64 {
	V  :: #simd[4]f64
	vp := cast([^]V)data
	nv := n / 4

	// 4 independent accumulators break the 4-cycle VADDPD dependency chain.
	a, b, c, d: V = {}, {}, {}, {}
	v4 := nv / 4
	for i in 0..<v4 {
		j := i * 4
		a += vp[j]; b += vp[j+1]; c += vp[j+2]; d += vp[j+3]
	}
	acc := a + b + c + d
	for i := v4 * 4; i < nv; i += 1 { acc += vp[i] }

	sum := intrinsics.simd_reduce_add_ordered(acc)
	for i := nv * 4; i < n; i += 1 { sum += data[i] }
	return sum
}

// ── i32 sum ───────────────────────────────────────────────────────────────────

// 8-way unrolled i64 accumulation; LLVM emits vpmovsxdq + vpaddq (AVX2).
_sum_i32_simd :: #force_inline proc "contextless" (data: [^]i32, n: int) -> i64 {
	a0, a1, a2, a3, a4, a5, a6, a7: i64 = 0, 0, 0, 0, 0, 0, 0, 0
	i := 0
	for i + 8 <= n {
		a0 += i64(data[i]);   a1 += i64(data[i+1])
		a2 += i64(data[i+2]); a3 += i64(data[i+3])
		a4 += i64(data[i+4]); a5 += i64(data[i+5])
		a6 += i64(data[i+6]); a7 += i64(data[i+7])
		i += 8
	}
	sum := a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
	for ; i < n; i += 1 { sum += i64(data[i]) }
	return sum
}

// ── i32 min / max ─────────────────────────────────────────────────────────────

_min_i32_simd :: #force_inline proc "contextless" (data: [^]i32, n: int) -> i32 {
	a, b, c, d, e, f, g, h := data[0], data[0], data[0], data[0],
	                           data[0], data[0], data[0], data[0]
	i := 0
	for i + 8 <= n {
		a = min(a, data[i]);   b = min(b, data[i+1])
		c = min(c, data[i+2]); d = min(d, data[i+3])
		e = min(e, data[i+4]); f = min(f, data[i+5])
		g = min(g, data[i+6]); h = min(h, data[i+7])
		i += 8
	}
	best := min(min(min(a, b), min(c, d)), min(min(e, f), min(g, h)))
	for ; i < n; i += 1 { best = min(best, data[i]) }
	return best
}

_max_i32_simd :: #force_inline proc "contextless" (data: [^]i32, n: int) -> i32 {
	a, b, c, d, e, f, g, h := data[0], data[0], data[0], data[0],
	                           data[0], data[0], data[0], data[0]
	i := 0
	for i + 8 <= n {
		a = max(a, data[i]);   b = max(b, data[i+1])
		c = max(c, data[i+2]); d = max(d, data[i+3])
		e = max(e, data[i+4]); f = max(f, data[i+5])
		g = max(g, data[i+6]); h = max(h, data[i+7])
		i += 8
	}
	best := max(max(max(a, b), max(c, d)), max(max(e, f), max(g, h)))
	for ; i < n; i += 1 { best = max(best, data[i]) }
	return best
}

// ── i32 min+max combined ──────────────────────────────────────────────────────

// Single-pass min and max.  Returns (min, max); n must be > 0.
// Explicit #simd vectors (vs a scalar unroll) so the packed min/max lowers to
// real SIMD. Two vector pairs hide op latency.
//
// AVX2 is enabled for this proc only (the global build stays baseline x86-64, so
// other kernels aren't affected): with AVX2 each #simd[8]i32 min/max is a single
// vpminsd/vpmaxsd over a 256-bit ymm, removing the ~8% ALU tax the baseline
// (SSE2, no packed i32 min/max → pcmpgtd+blend) pays — the kernel then runs at
// the memory-read floor. NOT force-inlined so the target feature stays scoped to
// this function rather than leaking AVX2 codegen into a baseline caller.
//
// TRADEOFF: on x86, emits AVX2 unconditionally (requires Haswell+, 2013+).
// On ARM64 / other architectures, falls back to a scalar loop.
when ODIN_ARCH == .amd64 || ODIN_ARCH == .i386 {
	@(enable_target_feature = "avx2")
	_min_max_i32_simd :: proc "contextless" (data: [^]i32, n: int) -> (lo: i32, hi: i32) {
		V  :: #simd[8]i32
		vp := cast([^]V)data
		nv := n / 8
		if nv > 0 {
			lo0 := vp[0]; lo1 := vp[0]
			hi0 := vp[0]; hi1 := vp[0]
			i := 1
			for i + 1 < nv {
				lo0 = intrinsics.simd_min(lo0, vp[i]);   hi0 = intrinsics.simd_max(hi0, vp[i])
				lo1 = intrinsics.simd_min(lo1, vp[i+1]); hi1 = intrinsics.simd_max(hi1, vp[i+1])
				i += 2
			}
			for ; i < nv; i += 1 {
				lo0 = intrinsics.simd_min(lo0, vp[i]); hi0 = intrinsics.simd_max(hi0, vp[i])
			}
			lo = intrinsics.simd_reduce_min(intrinsics.simd_min(lo0, lo1))
			hi = intrinsics.simd_reduce_max(intrinsics.simd_max(hi0, hi1))
			for j := nv * 8; j < n; j += 1 { lo = min(lo, data[j]); hi = max(hi, data[j]) }
		} else {
			lo = data[0]; hi = data[0]
			for j in 1..<n { lo = min(lo, data[j]); hi = max(hi, data[j]) }
		}
		return
	}
} else {
	// Scalar fallback for ARM64 and other non-x86 architectures.
	_min_max_i32_simd :: proc "contextless" (data: [^]i32, n: int) -> (lo: i32, hi: i32) {
		lo = data[0]; hi = data[0]
		for j in 1..<n { lo = min(lo, data[j]); hi = max(hi, data[j]) }
		return
	}
}
