package dataframe

// Per-column aggregations (Stage 6, DESIGN.md §6.6). A scalar convenience layer
// over the kernels in agg.odin: look up a column by name, run the aggregation,
// return the scalar value. The expression path (expr.agg_/cov_/corr_) covers
// "over expressions"; these procs cover "per column".
//
// NULL semantics: count/n_unique return 0 for a column with no valid rows;
// every value-based aggregation reports the `.Null_Value` error instead of a
// value (there is no silent NULL-as-zero, principle 7). var/std of a single
// valid row, and skew/kurtosis of too few rows, are NaN.

import "expr"

// Scalar is a single value returned by value-preserving per-column
// aggregations (min/max/mode/first/last). The member type mirrors the source
// column's value type.
Scalar :: union {
	i8, i16, i32, i64,
	u8, u16, u32, u64,
	int, uint,
	f32, f64,
	bool, string,
}

// agg_i64 runs an i64-valued aggregation (Count/N_Unique) over a column.
// The result is always valid (0 when no row is valid).
@(private)
agg_i64 :: proc(df: ^DataFrame, name: string, kind: expr.Agg_Kind) -> (i64, Error) {
	col, cerr := dataframe_get_column(df, name)
	if cerr != .None {
		return {}, cerr
	}
	out, aerr := agg_column(df.alloc, col, kind, 0)
	if aerr != .None {
		return {}, aerr
	}
	defer column_destroy(&out)
	return column_typed_view(&out, i64)[0], .None
}

// agg_f64 runs an f64-valued aggregation over a column.
@(private)
agg_f64 :: proc(df: ^DataFrame, name: string, kind: expr.Agg_Kind, q: f64) -> (f64, Error) {
	col, cerr := dataframe_get_column(df, name)
	if cerr != .None {
		return {}, cerr
	}
	out, aerr := agg_column(df.alloc, col, kind, q)
	if aerr != .None {
		return {}, aerr
	}
	defer column_destroy(&out)
	if !column_is_valid(&out, 0) {
		return {}, .Null_Value
	}
	return column_typed_view(&out, f64)[0], .None
}

// agg_scalar runs a value-preserving aggregation (Min/Max/Mode/First/Last)
// over a column.
@(private)
agg_scalar :: proc(df: ^DataFrame, name: string, kind: expr.Agg_Kind) -> (Scalar, Error) {
	col, cerr := dataframe_get_column(df, name)
	if cerr != .None {
		return {}, cerr
	}
	out, aerr := agg_column(df.alloc, col, kind, 0)
	if aerr != .None {
		return {}, aerr
	}
	defer column_destroy(&out)
	if !column_is_valid(&out, 0) {
		return {}, .Null_Value
	}
	switch out.dtype {
	case typeid_of(i8):   return column_typed_view(&out, i8)[0], .None
	case typeid_of(i16):  return column_typed_view(&out, i16)[0], .None
	case typeid_of(i32):  return column_typed_view(&out, i32)[0], .None
	case typeid_of(i64):  return column_typed_view(&out, i64)[0], .None
	case typeid_of(u8):   return column_typed_view(&out, u8)[0], .None
	case typeid_of(u16):  return column_typed_view(&out, u16)[0], .None
	case typeid_of(u32):  return column_typed_view(&out, u32)[0], .None
	case typeid_of(u64):  return column_typed_view(&out, u64)[0], .None
	case typeid_of(int):  return column_typed_view(&out, int)[0], .None
	case typeid_of(uint): return column_typed_view(&out, uint)[0], .None
	case typeid_of(f32):  return column_typed_view(&out, f32)[0], .None
	case typeid_of(f64):  return column_typed_view(&out, f64)[0], .None
	case typeid_of(bool): return column_typed_view(&out, bool)[0], .None
	case typeid_of(string): return column_typed_view(&out, string)[0], .None
	}
	return {}, .Unsupported_Operation
}

// moment2_named runs cov/corr over two named columns.
@(private)
moment2_named :: proc(df: ^DataFrame, x_name, y_name: string, want_corr: bool) -> (f64, Error) {
	x, xerr := dataframe_get_column(df, x_name)
	if xerr != .None {
		return {}, xerr
	}
	y, yerr := dataframe_get_column(df, y_name)
	if yerr != .None {
		return {}, yerr
	}
	if !is_numeric_type(x.dtype) || !is_numeric_type(y.dtype) {
		return {}, .Unsupported_Operation
	}
	if x.count != y.count {
		return {}, .Length_Mismatch
	}
	out, aerr := column_alloc(df.alloc, "", f64, size_of(f64), align_of(f64), 1)
	if aerr != .None {
		return {}, aerr
	}
	defer column_destroy(&out)
	moment2_pair_dispatch(x, y, &out, x.dtype, y.dtype, want_corr)
	if !column_is_valid(&out, 0) {
		return {}, .Null_Value
	}
	return column_typed_view(&out, f64)[0], .None
}

// --- public procs ------------------------------------------------------------

// dataframe_count returns the number of valid rows in column name.
dataframe_count :: proc(df: ^DataFrame, name: string) -> (i64, Error) {
	return agg_i64(df, name, .Count)
}

// dataframe_n_unique returns the number of distinct valid values in name.
dataframe_n_unique :: proc(df: ^DataFrame, name: string) -> (i64, Error) {
	return agg_i64(df, name, .N_Unique)
}

// dataframe_sum returns the sum of the valid rows of name. Reports
// .Unsupported_Operation for non-numeric columns.
dataframe_sum :: proc(df: ^DataFrame, name: string) -> (f64, Error) {
	return agg_f64(df, name, .Sum, 0)
}

// dataframe_mean returns the arithmetic mean of the valid rows of name.
dataframe_mean :: proc(df: ^DataFrame, name: string) -> (f64, Error) {
	return agg_f64(df, name, .Mean, 0)
}

// dataframe_var returns the sample variance (n-1) of the valid rows of name.
dataframe_var :: proc(df: ^DataFrame, name: string) -> (f64, Error) {
	return agg_f64(df, name, .Var, 0)
}

// dataframe_std returns the sample standard deviation of the valid rows of
// name.
dataframe_std :: proc(df: ^DataFrame, name: string) -> (f64, Error) {
	return agg_f64(df, name, .Std, 0)
}

// dataframe_median returns the 0.5 quantile (linear interpolation) of name.
dataframe_median :: proc(df: ^DataFrame, name: string) -> (f64, Error) {
	return agg_f64(df, name, .Median, 0)
}

// dataframe_quantile returns the q quantile (linear interpolation) of the
// valid rows of name. q must be in [0, 1].
dataframe_quantile :: proc(df: ^DataFrame, name: string, q: f64) -> (f64, Error) {
	return agg_f64(df, name, .Quantile, q)
}

// dataframe_product returns the product of the valid rows of name.
dataframe_product :: proc(df: ^DataFrame, name: string) -> (f64, Error) {
	return agg_f64(df, name, .Product, 0)
}

// dataframe_skew returns the sample skewness (G1) of the valid rows of name
// (NaN for fewer than 3 valid rows).
dataframe_skew :: proc(df: ^DataFrame, name: string) -> (f64, Error) {
	return agg_f64(df, name, .Skew, 0)
}

// dataframe_kurtosis returns the sample excess kurtosis (G2) of the valid rows
// of name (NaN for fewer than 4 valid rows).
dataframe_kurtosis :: proc(df: ^DataFrame, name: string) -> (f64, Error) {
	return agg_f64(df, name, .Kurtosis, 0)
}

// dataframe_min returns the minimum of the valid rows of name, preserving the
// column's value type.
dataframe_min :: proc(df: ^DataFrame, name: string) -> (Scalar, Error) {
	return agg_scalar(df, name, .Min)
}

// dataframe_max returns the maximum of the valid rows of name, preserving the
// column's value type.
dataframe_max :: proc(df: ^DataFrame, name: string) -> (Scalar, Error) {
	return agg_scalar(df, name, .Max)
}

// dataframe_mode returns the most frequent valid value of name (ties resolve
// to the first-seen value), preserving the column's value type.
dataframe_mode :: proc(df: ^DataFrame, name: string) -> (Scalar, Error) {
	return agg_scalar(df, name, .Mode)
}

// dataframe_first returns the first valid value of name.
dataframe_first :: proc(df: ^DataFrame, name: string) -> (Scalar, Error) {
	return agg_scalar(df, name, .First)
}

// dataframe_last returns the last valid value of name.
dataframe_last :: proc(df: ^DataFrame, name: string) -> (Scalar, Error) {
	return agg_scalar(df, name, .Last)
}

// dataframe_cov returns the sample covariance of columns x and y (rows valid
// in both columns).
dataframe_cov :: proc(df: ^DataFrame, x, y: string) -> (f64, Error) {
	return moment2_named(df, x, y, false)
}

// dataframe_corr returns the Pearson correlation of columns x and y (rows
// valid in both columns; NaN for constant input).
dataframe_corr :: proc(df: ^DataFrame, x, y: string) -> (f64, Error) {
	return moment2_named(df, x, y, true)
}
