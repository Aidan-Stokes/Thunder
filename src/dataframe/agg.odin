package dataframe

// Aggregation kernels (Stage 6, DESIGN.md §6.6).
//
// agg_eval implements the expr.Agg node; cov_eval/corr_eval implement the
// expr.Cov/expr.Corr nodes. agg_column is the shared core: it runs a kernel
// over any Column and returns a new single-row result column (used by
// agg_eval and by the scalar per-column API in stats.odin). Kernels are the
// reuse point for group_by (Stage 7): they consume a ^Column and will gain
// row-index subsets there.
//
// Every kernel skips invalid (NULL) rows. count/n_unique always produce a
// valid row; value-based aggregations produce a NULL row when no input row is
// valid.

import "core:mem"
import "core:math"
import "core:sort"
import "expr"

// agg_eval evaluates an Agg node against df: a single-row result column.
@(private)
agg_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Agg) -> (Column, Error) {
	child, cerr := eval_child(allocator, df, n.expr)
	if cerr != .None {
		return {}, cerr
	}
	defer column_destroy(&child)
	return agg_column(allocator, &child, n.kind, n.q)
}

// agg_column runs aggregation kind over col, returning a caller-owned
// single-row column whose dtype follows agg_result_dtype.
@(private)
agg_column :: proc(allocator: mem.Allocator, col: ^Column, kind: expr.Agg_Kind, q: f64) -> (Column, Error) {
	if err := validate_agg(kind, col.dtype); err != .None {
		return {}, err
	}
	res_dtype, ok := agg_result_dtype(kind, col.dtype)
	if !ok {
		return {}, .Invalid_Argument
	}
	out, err := column_alloc(allocator, "", res_dtype, size_of_ty(res_dtype), align_of_ty(res_dtype), 1)
	if err != .None {
		return {}, err
	}
	if rerr := run_group_agg(allocator, col, kind, q, &out, nil, 0); rerr != .None {
		column_destroy(&out)
		return {}, rerr
	}
	return out, .None
}

// run_group_agg runs aggregation kind over col restricted to the row subset
// `rows` (nil = every row), writing the single result value into out at
// out_row. It is the shared kernel entry point: agg_column runs the whole
// column (rows = nil, out_row = 0); group_by_agg runs each group's row indices
// into successive out rows (DESIGN.md §9). Callers must have validated
// kind/dtype with validate_agg/agg_result_dtype and allocated out with the
// matching result dtype and at least out_row+1 rows.
@(private)
run_group_agg :: proc(allocator: mem.Allocator, col: ^Column, kind: expr.Agg_Kind, q: f64, out: ^Column, rows: []int, out_row: int) -> Error {
	if kind == .Quantile && (q < 0 || q > 1) {
		return .Invalid_Argument
	}
	switch kind {
	case .Count:
		agg_count(col, out, rows, out_row)
	case .N_Unique:
		agg_n_unique(col.dtype, col, out, rows, out_row)
	case .Sum, .Mean, .Var, .Std, .Product, .Skew, .Kurtosis:
		agg_numeric_reduce(col.dtype, col, out, rows, out_row, kind)
	case .Median:
		return agg_quantile(allocator, col, out, rows, out_row, 0.5)
	case .Quantile:
		return agg_quantile(allocator, col, out, rows, out_row, q)
	case .Min:
		agg_min(col.dtype, col, out, rows, out_row)
	case .Max:
		agg_max(col.dtype, col, out, rows, out_row)
	case .Mode:
		agg_mode(col.dtype, col, out, rows, out_row)
	case .First:
		agg_first_last(col, out, rows, out_row, false)
	case .Last:
		agg_first_last(col, out, rows, out_row, true)
	case:
		return .Invalid_Argument
	}
	return .None
}

// subset_total returns the number of iterations for a row-subset loop over a
// buffer of n rows: n when rows is nil (all rows), else len(rows).
@(private)
subset_total :: proc(n: int, rows: []int) -> int {
	if rows == nil {
		return n
	}
	return len(rows)
}

// subset_index returns the column row index of the k-th subset iteration.
@(private)
subset_index :: proc(rows: []int, k: int) -> int {
	if rows == nil {
		return k
	}
	return rows[k]
}

// validate_agg checks that kind is supported for a column of dtype (S6.4).
@(private)
validate_agg :: proc(kind: expr.Agg_Kind, dtype: typeid) -> Error {
	switch kind {
	case .Count, .First, .Last:
		// validity-only / byte-copy kernels work on any type.
		return .None
	case .Sum, .Mean, .Var, .Std, .Median, .Quantile, .Product, .Skew, .Kurtosis:
		if is_numeric_type(dtype) {
			return .None
		}
		return .Unsupported_Operation
	case .Min, .Max:
		// ordering types, matching binary <.
		if is_numeric_type(dtype) || dtype == typeid_of(string) {
			return .None
		}
		return .Unsupported_Operation
	case .N_Unique, .Mode:
		// hash-set kernels over the supported scalar set.
		if is_numeric_type(dtype) || dtype == typeid_of(bool) || dtype == typeid_of(string) {
			return .None
		}
		return .Unsupported_Operation
	}
	return .Unsupported_Operation
}

// agg_result_dtype maps an aggregation kind to its result column type (S6.4).
@(private)
agg_result_dtype :: proc(kind: expr.Agg_Kind, dtype: typeid) -> (typeid, bool) {
	switch kind {
	case .Count, .N_Unique:
		return typeid_of(i64), true
	case .Sum, .Mean, .Var, .Std, .Median, .Quantile, .Product, .Skew, .Kurtosis:
		return typeid_of(f64), true
	case .Min, .Max, .Mode, .First, .Last:
		return dtype, true
	}
	return {}, false
}

// --- count / n_unique / mode / first / last ---------------------------------

// agg_count counts the valid rows of col in the subset `rows` (nil = all),
// writing into out at out_row (works on any column type).
@(private)
agg_count :: proc(col, out: ^Column, rows: []int, out_row: int) {
	n := 0
	for k in 0 ..< subset_total(col.count, rows) {
		if row_valid(col.valid, subset_index(rows, k)) {
			n += 1
		}
	}
	column_typed_view(out, i64)[out_row] = i64(n)
}

// agg_n_unique counts the distinct valid values of col's subset (hash set).
@(private)
agg_n_unique :: proc(t: typeid, col, out: ^Column, rows: []int, out_row: int) {
	switch t {
	case typeid_of(i8):   n_unique_typed(col, out, rows, out_row, i8)
	case typeid_of(i16):  n_unique_typed(col, out, rows, out_row, i16)
	case typeid_of(i32):  n_unique_typed(col, out, rows, out_row, i32)
	case typeid_of(i64):  n_unique_typed(col, out, rows, out_row, i64)
	case typeid_of(u8):   n_unique_typed(col, out, rows, out_row, u8)
	case typeid_of(u16):  n_unique_typed(col, out, rows, out_row, u16)
	case typeid_of(u32):  n_unique_typed(col, out, rows, out_row, u32)
	case typeid_of(u64):  n_unique_typed(col, out, rows, out_row, u64)
	case typeid_of(int):  n_unique_typed(col, out, rows, out_row, int)
	case typeid_of(uint): n_unique_typed(col, out, rows, out_row, uint)
	case typeid_of(f32):  n_unique_typed(col, out, rows, out_row, f32)
	case typeid_of(f64):  n_unique_typed(col, out, rows, out_row, f64)
	case typeid_of(bool): n_unique_typed(col, out, rows, out_row, bool)
	case typeid_of(string): n_unique_typed(col, out, rows, out_row, string)
	}
}

@(private)
n_unique_typed :: proc(col, out: ^Column, rows: []int, out_row: int, $T: typeid) {
	iv := column_typed_view(col, T)
	seen := make(map[T]bool)
	defer delete(seen)
	n := 0
	for k in 0 ..< subset_total(col.count, rows) {
		i := subset_index(rows, k)
		if !row_valid(col.valid, i) {
			continue
		}
		if !seen[iv[i]] {
			seen[iv[i]] = true
			n += 1
		}
	}
	column_typed_view(out, i64)[out_row] = i64(n)
}

// agg_mode finds the most frequent valid value of col's subset; ties resolve
// to the value seen first. The result column preserves the input dtype.
@(private)
agg_mode :: proc(t: typeid, col, out: ^Column, rows: []int, out_row: int) {
	switch t {
	case typeid_of(i8):   mode_typed(col, out, rows, out_row, i8)
	case typeid_of(i16):  mode_typed(col, out, rows, out_row, i16)
	case typeid_of(i32):  mode_typed(col, out, rows, out_row, i32)
	case typeid_of(i64):  mode_typed(col, out, rows, out_row, i64)
	case typeid_of(u8):   mode_typed(col, out, rows, out_row, u8)
	case typeid_of(u16):  mode_typed(col, out, rows, out_row, u16)
	case typeid_of(u32):  mode_typed(col, out, rows, out_row, u32)
	case typeid_of(u64):  mode_typed(col, out, rows, out_row, u64)
	case typeid_of(int):  mode_typed(col, out, rows, out_row, int)
	case typeid_of(uint): mode_typed(col, out, rows, out_row, uint)
	case typeid_of(f32):  mode_typed(col, out, rows, out_row, f32)
	case typeid_of(f64):  mode_typed(col, out, rows, out_row, f64)
	case typeid_of(bool): mode_typed(col, out, rows, out_row, bool)
	case typeid_of(string): mode_typed(col, out, rows, out_row, string)
	}
}

@(private)
mode_typed :: proc(col, out: ^Column, rows: []int, out_row: int, $T: typeid) {
	iv := column_typed_view(col, T)
	counts := make(map[T]int)
	defer delete(counts)
	max_count := 0
	for k in 0 ..< subset_total(col.count, rows) {
		i := subset_index(rows, k)
		if !row_valid(col.valid, i) {
			continue
		}
		c := counts[iv[i]] + 1
		counts[iv[i]] = c
		if c > max_count {
			max_count = c
		}
	}
	if max_count == 0 {
		_ = column_set_valid(out, out_row, false)
		return
	}
	// ties resolve to the first value in column order among the
	// maximal-frequency values.
	best: T
	for k in 0 ..< subset_total(col.count, rows) {
		i := subset_index(rows, k)
		if !row_valid(col.valid, i) {
			continue
		}
		if counts[iv[i]] == max_count {
			best = iv[i]
			break
		}
	}
	column_typed_view(out, T)[out_row] = best
}

// agg_first_last copies the first (last=false) or last (last=true) valid value
// of col's subset into out at out_row. Byte-level copy: works for any column
// type.
@(private)
agg_first_last :: proc(col, out: ^Column, rows: []int, out_row: int, last: bool) {
	if last {
		for k := subset_total(col.count, rows) - 1; k >= 0; k -= 1 {
			i := subset_index(rows, k)
			if row_valid(col.valid, i) {
				mem.copy(ptr_offset(out.data, out_row * col.elem_size), ptr_offset(col.data, i * col.elem_size), col.elem_size)
				return
			}
		}
	} else {
		for k := 0; k < subset_total(col.count, rows); k += 1 {
			i := subset_index(rows, k)
			if row_valid(col.valid, i) {
				mem.copy(ptr_offset(out.data, out_row * col.elem_size), ptr_offset(col.data, i * col.elem_size), col.elem_size)
				return
			}
		}
	}
	_ = column_set_valid(out, out_row, false)
}

// --- numeric reductions ------------------------------------------------------

// agg_numeric_reduce runs a numeric reduction (Sum/Mean/Var/Std/Product/
// Skew/Kurtosis) over col's subset, dispatching once by dtype. All results are
// f64.
@(private)
agg_numeric_reduce :: proc(t: typeid, col, out: ^Column, rows: []int, out_row: int, kind: expr.Agg_Kind) {
	switch t {
	case typeid_of(i8):   numeric_reduce_typed(col, out, rows, out_row, i8, kind)
	case typeid_of(i16):  numeric_reduce_typed(col, out, rows, out_row, i16, kind)
	case typeid_of(i32):  numeric_reduce_typed(col, out, rows, out_row, i32, kind)
	case typeid_of(i64):  numeric_reduce_typed(col, out, rows, out_row, i64, kind)
	case typeid_of(u8):   numeric_reduce_typed(col, out, rows, out_row, u8, kind)
	case typeid_of(u16):  numeric_reduce_typed(col, out, rows, out_row, u16, kind)
	case typeid_of(u32):  numeric_reduce_typed(col, out, rows, out_row, u32, kind)
	case typeid_of(u64):  numeric_reduce_typed(col, out, rows, out_row, u64, kind)
	case typeid_of(int):  numeric_reduce_typed(col, out, rows, out_row, int, kind)
	case typeid_of(uint): numeric_reduce_typed(col, out, rows, out_row, uint, kind)
	case typeid_of(f32):  numeric_reduce_typed(col, out, rows, out_row, f32, kind)
	case typeid_of(f64):  numeric_reduce_typed(col, out, rows, out_row, f64, kind)
	}
}

// numeric_reduce_typed implements every numeric reduction for a column of
// concrete type T over the subset `rows`. Pass one computes count + mean;
// var/std/skew/kurtosis take a second pass over central moments.
@(private)
numeric_reduce_typed :: proc(col, out: ^Column, rows: []int, out_row: int, $T: typeid, kind: expr.Agg_Kind) {
	iv := column_typed_view(col, T)
	sum: f64
	n := 0

	// Fast path: all-valid full column. No subset indirection, no validity
	// checks — the most common aggregation case.
	if col.valid == nil && rows == nil {
		when T == f64 {
			sum = parallel_simd_sum_f64(iv)
			n = col.count
		} else {
			for i in 0 ..< col.count {
				sum += f64(iv[i])
				n += 1
			}
		}
	} else {
		for k in 0 ..< subset_total(col.count, rows) {
			i := subset_index(rows, k)
			if row_valid(col.valid, i) {
				sum += f64(iv[i])
				n += 1
			}
		}
	}

	if n == 0 {
		_ = column_set_valid(out, out_row, false)
		return
	}
	mean := sum / f64(n)
	ov := column_typed_view(out, f64)

	#partial switch kind {
	case .Sum:
		ov[out_row] = sum
	case .Mean:
		ov[out_row] = mean
	case .Product:
		prod: f64 = 1
		if col.valid == nil && rows == nil {
			for i in 0 ..< col.count {
				prod *= f64(iv[i])
			}
		} else {
			for k in 0 ..< subset_total(col.count, rows) {
				i := subset_index(rows, k)
				if row_valid(col.valid, i) {
					prod *= f64(iv[i])
				}
			}
		}
		ov[out_row] = prod
	case .Var, .Std:
		m2, _, _ := central_moments(iv, col.valid, rows, mean, 2)
		v := m2 / f64(n - 1)
		if kind == .Std {
			v = math.sqrt(v)
		}
		ov[out_row] = v
	case .Skew, .Kurtosis:
		m2, m3, m4 := central_moments(iv, col.valid, rows, mean, 4)
		if kind == .Skew {
			// sample skewness G1 (pandas/scipy default), NaN for n < 3.
			if n < 3 {
				ov[out_row] = math.nan_f64()
			} else {
				b := math.sqrt(m2)
				ov[out_row] = (m3 / (b * b * b)) * math.sqrt(f64(n) * f64(n - 1)) / f64(n - 2)
			}
		} else {
			// sample excess kurtosis G2 (pandas default), NaN for n < 4.
			if n < 4 {
				ov[out_row] = math.nan_f64()
			} else {
				g2 := f64(n) * m4 / (m2 * m2) - 3
				ov[out_row] = ((f64(n) + 1) * g2 + 6) * f64(n - 1) / (f64(n - 2) * f64(n - 3))
			}
		}
	case:
		_ = column_set_valid(out, out_row, false)
	}
}

// central_moments accumulates sum((x - mean)^k) over the valid rows of iv in
// the subset `rows` (nil = all) for k = 2 .. max_order (2 or 4), returning
// (m2, m3, m4).
@(private)
central_moments :: proc(iv: []$T, valid: []u64, rows: []int, mean: f64, max_order: int) -> (m2: f64, m3: f64, m4: f64) {
	for k in 0 ..< subset_total(len(iv), rows) {
		i := subset_index(rows, k)
		if row_valid(valid, i) {
			d := f64(iv[i]) - mean
			d2 := d * d
			m2 += d2
			if max_order >= 3 {
				m3 += d2 * d
			}
			if max_order >= 4 {
				m4 += d2 * d2
			}
		}
	}
	return
}

// --- min / max ---------------------------------------------------------------

// agg_min / agg_max find the smallest/largest valid value of col's subset. The
// result column preserves the input dtype (numeric or string).
@(private)
agg_min :: proc(t: typeid, col, out: ^Column, rows: []int, out_row: int) {
	switch t {
	case typeid_of(i8):   min_max_typed(col, out, rows, out_row, i8, true)
	case typeid_of(i16):  min_max_typed(col, out, rows, out_row, i16, true)
	case typeid_of(i32):  min_max_typed(col, out, rows, out_row, i32, true)
	case typeid_of(i64):  min_max_typed(col, out, rows, out_row, i64, true)
	case typeid_of(u8):   min_max_typed(col, out, rows, out_row, u8, true)
	case typeid_of(u16):  min_max_typed(col, out, rows, out_row, u16, true)
	case typeid_of(u32):  min_max_typed(col, out, rows, out_row, u32, true)
	case typeid_of(u64):  min_max_typed(col, out, rows, out_row, u64, true)
	case typeid_of(int):  min_max_typed(col, out, rows, out_row, int, true)
	case typeid_of(uint): min_max_typed(col, out, rows, out_row, uint, true)
	case typeid_of(f32):  min_max_typed(col, out, rows, out_row, f32, true)
	case typeid_of(f64):  min_max_typed(col, out, rows, out_row, f64, true)
	case typeid_of(string): min_max_typed(col, out, rows, out_row, string, true)
	}
}

@(private)
agg_max :: proc(t: typeid, col, out: ^Column, rows: []int, out_row: int) {
	switch t {
	case typeid_of(i8):   min_max_typed(col, out, rows, out_row, i8, false)
	case typeid_of(i16):  min_max_typed(col, out, rows, out_row, i16, false)
	case typeid_of(i32):  min_max_typed(col, out, rows, out_row, i32, false)
	case typeid_of(i64):  min_max_typed(col, out, rows, out_row, i64, false)
	case typeid_of(u8):   min_max_typed(col, out, rows, out_row, u8, false)
	case typeid_of(u16):  min_max_typed(col, out, rows, out_row, u16, false)
	case typeid_of(u32):  min_max_typed(col, out, rows, out_row, u32, false)
	case typeid_of(u64):  min_max_typed(col, out, rows, out_row, u64, false)
	case typeid_of(int):  min_max_typed(col, out, rows, out_row, int, false)
	case typeid_of(uint): min_max_typed(col, out, rows, out_row, uint, false)
	case typeid_of(f32):  min_max_typed(col, out, rows, out_row, f32, false)
	case typeid_of(f64):  min_max_typed(col, out, rows, out_row, f64, false)
	case typeid_of(string): min_max_typed(col, out, rows, out_row, string, false)
	}
}

@(private)
min_max_typed :: proc(col, out: ^Column, rows: []int, out_row: int, $T: typeid, is_min: bool) {
	iv := column_typed_view(col, T)
	ov := column_typed_view(out, T)
	have := false
	best: T
	for k in 0 ..< subset_total(col.count, rows) {
		i := subset_index(rows, k)
		if !row_valid(col.valid, i) {
			continue
		}
		if !have {
			have = true
			best = iv[i]
		} else if is_min {
			if iv[i] < best {
				best = iv[i]
			}
		} else {
			if iv[i] > best {
				best = iv[i]
			}
		}
	}
	if !have {
		_ = column_set_valid(out, out_row, false)
		return
	}
	ov[out_row] = best
}

// --- median / quantile -------------------------------------------------------

// agg_quantile returns the q quantile of the valid rows of col's subset using
// linear interpolation between the two closest ranks (h = q*(n-1)). q is
// validated by run_group_agg.
@(private)
agg_quantile :: proc(allocator: mem.Allocator, col, out: ^Column, rows: []int, out_row: int, q: f64) -> Error {
	values, err := collect_valid_f64(allocator, col, rows)
	if err != .None {
		return err
	}
	defer delete(values)
	if len(values) == 0 {
		_ = column_set_valid(out, out_row, false)
		return .None
	}
	sort.quick_sort(values)

	h := q * f64(len(values) - 1)
	lo := max(int(math.floor(h)), 0)
	hi := min(int(math.ceil(h)), len(values) - 1)
	value: f64
	if lo == hi {
		value = values[lo]
	} else {
		frac := h - f64(lo)
		value = values[lo] + (values[hi] - values[lo]) * frac
	}
	column_typed_view(out, f64)[out_row] = value
	return .None
}

// collect_valid_f64 gathers the valid rows of col's subset as f64 values.
@(private)
collect_valid_f64 :: proc(allocator: mem.Allocator, col: ^Column, rows: []int) -> ([]f64, Error) {
	cap := col.count
	if rows != nil {
		cap = len(rows)
	}
	values := make([dynamic]f64, 0, cap, allocator)
	if raw_data(values) == nil && cap != 0 {
		return nil, .Allocator_Failure
	}
	switch col.dtype {
	case typeid_of(i8):   collect_f64_typed(col, &values, rows, i8)
	case typeid_of(i16):  collect_f64_typed(col, &values, rows, i16)
	case typeid_of(i32):  collect_f64_typed(col, &values, rows, i32)
	case typeid_of(i64):  collect_f64_typed(col, &values, rows, i64)
	case typeid_of(u8):   collect_f64_typed(col, &values, rows, u8)
	case typeid_of(u16):  collect_f64_typed(col, &values, rows, u16)
	case typeid_of(u32):  collect_f64_typed(col, &values, rows, u32)
	case typeid_of(u64):  collect_f64_typed(col, &values, rows, u64)
	case typeid_of(int):  collect_f64_typed(col, &values, rows, int)
	case typeid_of(uint): collect_f64_typed(col, &values, rows, uint)
	case typeid_of(f32):  collect_f64_typed(col, &values, rows, f32)
	case typeid_of(f64):  collect_f64_typed(col, &values, rows, f64)
	}
	return values[:], .None
}

@(private)
collect_f64_typed :: proc(col: ^Column, out: ^[dynamic]f64, rows: []int, $T: typeid) {
	iv := column_typed_view(col, T)
	for k in 0 ..< subset_total(col.count, rows) {
		i := subset_index(rows, k)
		if row_valid(col.valid, i) {
			append(out, f64(iv[i]))
		}
	}
}

// --- cov / corr --------------------------------------------------------------

// cov_eval implements the Cov node: sample covariance of two numeric columns.
@(private)
cov_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Cov) -> (Column, Error) {
	return moment2_eval(allocator, df, n.lhs, n.rhs, false)
}

// corr_eval implements the Corr node: Pearson correlation of two numeric
// columns.
@(private)
corr_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Corr) -> (Column, Error) {
	return moment2_eval(allocator, df, n.lhs, n.rhs, true)
}

// moment2_eval reduces two equal-length numeric columns to a single-row f64
// column. Rows invalid in either column are skipped.
@(private)
moment2_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, lhs_e, rhs_e: ^expr.Expr, want_corr: bool) -> (Column, Error) {
	lhs, lerr := eval_child(allocator, df, lhs_e)
	if lerr != .None {
		return {}, lerr
	}
	defer column_destroy(&lhs)
	rhs, rerr := eval_child(allocator, df, rhs_e)
	if rerr != .None {
		return {}, rerr
	}
	defer column_destroy(&rhs)
	if !is_numeric_type(lhs.dtype) || !is_numeric_type(rhs.dtype) {
		return {}, .Unsupported_Operation
	}
	if lhs.count != rhs.count {
		return {}, .Length_Mismatch
	}

	out, err := column_alloc(allocator, "", f64, size_of(f64), align_of(f64), 1)
	if err != .None {
		return {}, err
	}
	moment2_pair_dispatch(&lhs, &rhs, &out, lhs.dtype, rhs.dtype, want_corr)
	return out, .None
}

// moment2_pair_dispatch routes a cov/corr kernel over the lhs dtype.
@(private)
moment2_pair_dispatch :: proc(lhs, rhs, out: ^Column, t, u: typeid, want_corr: bool) {
	switch t {
	case typeid_of(i8):   moment2_rhs(lhs, rhs, out, i8, u, want_corr)
	case typeid_of(i16):  moment2_rhs(lhs, rhs, out, i16, u, want_corr)
	case typeid_of(i32):  moment2_rhs(lhs, rhs, out, i32, u, want_corr)
	case typeid_of(i64):  moment2_rhs(lhs, rhs, out, i64, u, want_corr)
	case typeid_of(u8):   moment2_rhs(lhs, rhs, out, u8, u, want_corr)
	case typeid_of(u16):  moment2_rhs(lhs, rhs, out, u16, u, want_corr)
	case typeid_of(u32):  moment2_rhs(lhs, rhs, out, u32, u, want_corr)
	case typeid_of(u64):  moment2_rhs(lhs, rhs, out, u64, u, want_corr)
	case typeid_of(int):  moment2_rhs(lhs, rhs, out, int, u, want_corr)
	case typeid_of(uint): moment2_rhs(lhs, rhs, out, uint, u, want_corr)
	case typeid_of(f32):  moment2_rhs(lhs, rhs, out, f32, u, want_corr)
	case typeid_of(f64):  moment2_rhs(lhs, rhs, out, f64, u, want_corr)
	}
}

// moment2_rhs routes a cov/corr kernel over the rhs dtype.
@(private)
moment2_rhs :: proc(lhs, rhs, out: ^Column, $T: typeid, u: typeid, want_corr: bool) {
	switch u {
	case typeid_of(i8):   moment2_typed(lhs, rhs, out, T, i8, want_corr)
	case typeid_of(i16):  moment2_typed(lhs, rhs, out, T, i16, want_corr)
	case typeid_of(i32):  moment2_typed(lhs, rhs, out, T, i32, want_corr)
	case typeid_of(i64):  moment2_typed(lhs, rhs, out, T, i64, want_corr)
	case typeid_of(u8):   moment2_typed(lhs, rhs, out, T, u8, want_corr)
	case typeid_of(u16):  moment2_typed(lhs, rhs, out, T, u16, want_corr)
	case typeid_of(u32):  moment2_typed(lhs, rhs, out, T, u32, want_corr)
	case typeid_of(u64):  moment2_typed(lhs, rhs, out, T, u64, want_corr)
	case typeid_of(int):  moment2_typed(lhs, rhs, out, T, int, want_corr)
	case typeid_of(uint): moment2_typed(lhs, rhs, out, T, uint, want_corr)
	case typeid_of(f32):  moment2_typed(lhs, rhs, out, T, f32, want_corr)
	case typeid_of(f64):  moment2_typed(lhs, rhs, out, T, f64, want_corr)
	}
}

// moment2_typed computes cov (c/(n-1)) or corr (c/sqrt(vx*vy)) over the
// pairwise-valid rows of two typed columns.
@(private)
moment2_typed :: proc(lhs, rhs, out: ^Column, $T: typeid, $U: typeid, want_corr: bool) {
	lv := column_typed_view(lhs, T)
	rv := column_typed_view(rhs, U)
	n := 0
	mx: f64
	my: f64
	for i in 0 ..< len(lv) {
		if row_valid(lhs.valid, i) && row_valid(rhs.valid, i) {
			mx += f64(lv[i])
			my += f64(rv[i])
			n += 1
		}
	}
	if n == 0 {
		_ = column_set_valid(out, 0, false)
		return
	}
	mx /= f64(n)
	my /= f64(n)
	c: f64
	vx: f64
	vy: f64
	for i in 0 ..< len(lv) {
		if row_valid(lhs.valid, i) && row_valid(rhs.valid, i) {
			dx := f64(lv[i]) - mx
			dy := f64(rv[i]) - my
			c += dx * dy
			vx += dx * dx
			vy += dy * dy
		}
	}
	ov := column_typed_view(out, f64)
	if want_corr {
		ov[0] = c / (math.sqrt(vx) * math.sqrt(vy))
	} else {
		ov[0] = c / f64(n - 1)
	}
}
