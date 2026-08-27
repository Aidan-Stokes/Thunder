// simd_kernels.odin — S15.5 SIMD-accelerated numeric kernels.
//
// All functions operate on raw typed slices and are guarded by
// `when runtime.HAS_HARDWARE_SIMD` so the scalar fallback compiles
// cleanly on targets without SIMD hardware.
//
// Lane counts target 256-bit vectors: 4 lanes for 8-byte types
// (f64, i64, u64), 8 lanes for 4-byte types (f32, i32, u32).
//
// Load/store uses transmute + mem.copy through stack-aligned buffers
// to avoid alignment faults and the element-by-element overhead of
// simd.from_slice.

package dataframe

import "core:mem"
import "core:simd"
import "base:runtime"

// --- unary kernels ----------------------------------------------------------

// simd_neg negates all elements: ov[i] = -iv[i].
simd_neg :: proc(iv, ov: []$T) {
	n := len(iv)
	when runtime.HAS_HARDWARE_SIMD {
		when size_of(T) == 8 { W :: 4 }
		else when size_of(T) == 4 { W :: 8 }
		else when size_of(T) == 2 { W :: 16 }
		else { W :: 32 }
		vec_bytes := W * size_of(T)
		i := 0
		for ; i + W <= n; i += W {
			buf: [W]T
			mem.copy(&buf, &iv[i], vec_bytes)
			result := simd.neg(transmute(#simd[W]T)(buf))
			arr: [W]T = transmute([W]T)(result)
			mem.copy(&ov[i], &arr, vec_bytes)
		}
		for ; i < n; i += 1 {
			ov[i] = -iv[i]
		}
	} else {
		for i in 0 ..< n {
			ov[i] = -iv[i]
		}
	}
}

// simd_abs computes elementwise absolute value: ov[i] = |iv[i]|.
simd_abs :: proc(iv, ov: []$T) {
	n := len(iv)
	when runtime.HAS_HARDWARE_SIMD {
		when size_of(T) == 8 { W :: 4 }
		else when size_of(T) == 4 { W :: 8 }
		else when size_of(T) == 2 { W :: 16 }
		else { W :: 32 }
		vec_bytes := W * size_of(T)
		i := 0
		for ; i + W <= n; i += W {
			buf: [W]T
			mem.copy(&buf, &iv[i], vec_bytes)
			result := simd.abs(transmute(#simd[W]T)(buf))
			arr: [W]T = transmute([W]T)(result)
			mem.copy(&ov[i], &arr, vec_bytes)
		}
		for ; i < n; i += 1 {
			ov[i] = iv[i] >= 0 ? iv[i] : -iv[i]
		}
	} else {
		for i in 0 ..< n {
			ov[i] = iv[i] >= 0 ? iv[i] : -iv[i]
		}
	}
}

// --- binary arithmetic kernels ----------------------------------------------

// simd_add computes ov[i] = lv[i] + rv[i].
simd_add :: proc(lv, rv, ov: []$T) {
	n := len(lv)
	when runtime.HAS_HARDWARE_SIMD {
		when size_of(T) == 8 { W :: 4 }
		else when size_of(T) == 4 { W :: 8 }
		else when size_of(T) == 2 { W :: 16 }
		else { W :: 32 }
		vec_bytes := W * size_of(T)
		i := 0
		for ; i + W <= n; i += W {
			buf_a: [W]T
			buf_b: [W]T
			mem.copy(&buf_a, &lv[i], vec_bytes)
			mem.copy(&buf_b, &rv[i], vec_bytes)
			result := simd.add(transmute(#simd[W]T)(buf_a), transmute(#simd[W]T)(buf_b))
			arr: [W]T = transmute([W]T)(result)
			mem.copy(&ov[i], &arr, vec_bytes)
		}
		for ; i < n; i += 1 {
			ov[i] = lv[i] + rv[i]
		}
	} else {
		for i in 0 ..< n {
			ov[i] = lv[i] + rv[i]
		}
	}
}

// simd_sub computes ov[i] = lv[i] - rv[i].
simd_sub :: proc(lv, rv, ov: []$T) {
	n := len(lv)
	when runtime.HAS_HARDWARE_SIMD {
		when size_of(T) == 8 { W :: 4 }
		else when size_of(T) == 4 { W :: 8 }
		else when size_of(T) == 2 { W :: 16 }
		else { W :: 32 }
		vec_bytes := W * size_of(T)
		i := 0
		for ; i + W <= n; i += W {
			buf_a: [W]T
			buf_b: [W]T
			mem.copy(&buf_a, &lv[i], vec_bytes)
			mem.copy(&buf_b, &rv[i], vec_bytes)
			result := simd.sub(transmute(#simd[W]T)(buf_a), transmute(#simd[W]T)(buf_b))
			arr: [W]T = transmute([W]T)(result)
			mem.copy(&ov[i], &arr, vec_bytes)
		}
		for ; i < n; i += 1 {
			ov[i] = lv[i] - rv[i]
		}
	} else {
		for i in 0 ..< n {
			ov[i] = lv[i] - rv[i]
		}
	}
}

// simd_mul computes ov[i] = lv[i] * rv[i].
simd_mul :: proc(lv, rv, ov: []$T) {
	n := len(lv)
	when runtime.HAS_HARDWARE_SIMD {
		when size_of(T) == 8 { W :: 4 }
		else when size_of(T) == 4 { W :: 8 }
		else when size_of(T) == 2 { W :: 16 }
		else { W :: 32 }
		vec_bytes := W * size_of(T)
		i := 0
		for ; i + W <= n; i += W {
			buf_a: [W]T
			buf_b: [W]T
			mem.copy(&buf_a, &lv[i], vec_bytes)
			mem.copy(&buf_b, &rv[i], vec_bytes)
			result := simd.mul(transmute(#simd[W]T)(buf_a), transmute(#simd[W]T)(buf_b))
			arr: [W]T = transmute([W]T)(result)
			mem.copy(&ov[i], &arr, vec_bytes)
		}
		for ; i < n; i += 1 {
			ov[i] = lv[i] * rv[i]
		}
	} else {
		for i in 0 ..< n {
			ov[i] = lv[i] * rv[i]
		}
	}
}

// --- reduction kernels ------------------------------------------------------

// simd_sum returns the sum of all elements in iv (f64 only).
simd_sum :: proc(iv: []f64) -> f64 {
	n := len(iv)
	when runtime.HAS_HARDWARE_SIMD {
		W :: 4
		vec_bytes := W * size_of(f64)
		accum: #simd[W]f64
		i := 0
		for ; i + W <= n; i += W {
			buf: [W]f64
			mem.copy(&buf, &iv[i], vec_bytes)
			a := transmute(#simd[W]f64)(buf)
			accum = simd.add(accum, a)
		}
		sum := simd.reduce_add_pairs(accum)
		for ; i < n; i += 1 {
			sum += iv[i]
		}
		return sum
	} else {
		sum: f64
		for i in 0 ..< n {
			sum += iv[i]
		}
		return sum
	}
}
