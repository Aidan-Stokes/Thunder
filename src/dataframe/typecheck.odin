package dataframe

// Expression type inference (DESIGN.md §6.5). Returns the result type of e
// against df without evaluating, rejecting bad expressions early.
//
// Lives in the dataframe package (not dataframe/expr) because it resolves
// column types from a DataFrame; package imports are one-directional:
// dataframe -> expr.

import "expr"

// expr_typecheck infers the result type of e when evaluated against df.
expr_typecheck :: proc(df: ^DataFrame, e: ^expr.Expr) -> (typeid, Error) {
	switch n in e^ {
	case expr.Col:
		col, err := dataframe_get_column(df, n.name)
		if err != .None {
			return {}, err
		}
		return col.dtype, .None

	case expr.Lit:
		return n.dtype, .None

	case expr.Binary:
		lt, lerr := expr_typecheck(df, n.lhs)
		if lerr != .None {
			return {}, lerr
		}
		rt, rerr := expr_typecheck(df, n.rhs)
		if rerr != .None {
			return {}, rerr
		}
		// mirror literal coercion (§6.3): one side numeric Lit, other numeric
		if lt != rt {
			_, lhs_is_lit := n.lhs^.(expr.Lit)
			_, rhs_is_lit := n.rhs^.(expr.Lit)
			if lhs_is_lit && is_numeric_type(lt) && is_numeric_type(rt) {
				lt = rt
			} else if rhs_is_lit && is_numeric_type(rt) && is_numeric_type(lt) {
				rt = lt
			} else {
				return {}, .Type_Mismatch
			}
		}
		if err := validate_binary(n.op, lt); err != .None {
			return {}, err
		}
		if is_cmp_op(n.op) {
			return typeid_of(bool), .None
		}
		return lt, .None

	case expr.Unary:
		ct, cerr := expr_typecheck(df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		switch n.op {
		case .Neg:
			if !is_numeric_type(ct) {
				return {}, .Unsupported_Operation
			}
		case .Not:
			if ct != typeid_of(bool) {
				return {}, .Unsupported_Operation
			}
		}
		return ct, .None

	case expr.Cast:
		ct, cerr := expr_typecheck(df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		if !is_numeric_type(ct) || !is_numeric_type(n.to) {
			return {}, .Unsupported_Operation
		}
		return n.to, .None

	case expr.Alias:
		return expr_typecheck(df, n.expr)

	case expr.Not_Null:
		if _, cerr := expr_typecheck(df, n.expr); cerr != .None {
			return {}, cerr
		}
		return typeid_of(bool), .None

	case expr.Is_Nan:
		ct, cerr := expr_typecheck(df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		if ct != typeid_of(f32) && ct != typeid_of(f64) {
			return {}, .Unsupported_Operation
		}
		return typeid_of(bool), .None

	case expr.Fill_Null:
		ct, cerr := expr_typecheck(df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		vt, verr := expr_typecheck(df, n.value)
		if verr != .None {
			return {}, verr
		}
		_, value_is_lit := n.value^.(expr.Lit)
		if !value_is_lit || vt != ct {
			return {}, .Type_Mismatch
		}
		return ct, .None

	case expr.Coalesce:
		if len(n.exprs) == 0 {
			return {}, .Invalid_Argument
		}
		first, ferr := expr_typecheck(df, n.exprs[0])
		if ferr != .None {
			return {}, ferr
		}
		for i := 1; i < len(n.exprs); i += 1 {
			et, eerr := expr_typecheck(df, n.exprs[i])
			if eerr != .None {
				return {}, eerr
			}
			if et != first {
				return {}, .Type_Mismatch
			}
		}
		return first, .None

	case expr.Forward_Fill:
		ct, cerr := expr_typecheck(df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		return ct, .None

	case expr.Backward_Fill:
		ct, cerr := expr_typecheck(df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		return ct, .None

	case expr.Interpolate:
		ct, cerr := expr_typecheck(df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		if !is_numeric_type(ct) {
			return {}, .Unsupported_Operation
		}
		return typeid_of(f64), .None

	case expr.Func:
		ct, cerr := expr_typecheck(df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		if !is_numeric_type(ct) {
			return {}, .Unsupported_Operation
		}
		#partial switch n.kind {
		case .Round:
			if ct != typeid_of(f32) && ct != typeid_of(f64) {
				return {}, .Unsupported_Operation
			}
			return ct, .None
		case .Pct_Change:
			return typeid_of(f64), .None
		case:
			return ct, .None
		}

	case expr.Is_Between:
		ct, cerr := expr_typecheck(df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		lt, lerr := expr_typecheck(df, n.lower)
		if lerr != .None {
			return {}, lerr
		}
		ut, uerr := expr_typecheck(df, n.upper)
		if uerr != .None {
			return {}, uerr
		}
		if !is_numeric_type(ct) || !is_numeric_type(lt) || !is_numeric_type(ut) {
			return {}, .Unsupported_Operation
		}
		return typeid_of(bool), .None

	case expr.Is_In:
		ct, cerr := expr_typecheck(df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		lit, ok := n.values^.(expr.Lit)
		if !ok || lit.dtype != slice_typeid_of(ct) {
			return {}, .Type_Mismatch
		}
		return typeid_of(bool), .None

	case expr.Arange:
		st, serr := expr_typecheck(df, n.start)
		if serr != .None {
			return {}, serr
		}
		et, eerr := expr_typecheck(df, n.end)
		if eerr != .None {
			return {}, eerr
		}
		if st != typeid_of(int) || et != typeid_of(int) {
			return {}, .Type_Mismatch
		}
		return typeid_of(int), .None

	case expr.Arg_Where:
		ct, cerr := expr_typecheck(df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		if ct != typeid_of(bool) {
			return {}, .Type_Mismatch
		}
		return typeid_of(int), .None

	case expr.Distinct:
		ct, cerr := expr_typecheck(df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		switch ct {
		case typeid_of(i8), typeid_of(i16), typeid_of(i32), typeid_of(i64),
		     typeid_of(u8), typeid_of(u16), typeid_of(u32), typeid_of(u64),
		     typeid_of(int), typeid_of(uint), typeid_of(f32), typeid_of(f64),
		     typeid_of(bool), typeid_of(string):
			return typeid_of(bool), .None
		case:
			return {}, .Unsupported_Operation
		}
		return typeid_of(bool), .None

	case expr.Dot_Product:
		lt, lerr := expr_typecheck(df, n.lhs)
		if lerr != .None {
			return {}, lerr
		}
		rt, rerr := expr_typecheck(df, n.rhs)
		if rerr != .None {
			return {}, rerr
		}
		if !is_numeric_type(lt) || !is_numeric_type(rt) {
			return {}, .Unsupported_Operation
		}
		return typeid_of(f64), .None

	case expr.Concat_Str:
		if len(n.exprs) == 0 {
			return {}, .Invalid_Argument
		}
		for sub in n.exprs {
			et, eerr := expr_typecheck(df, sub)
			if eerr != .None {
				return {}, eerr
			}
			if et != typeid_of(string) {
				return {}, .Type_Mismatch
			}
		}
		return typeid_of(string), .None

	case expr.Search_Sorted:
		st, serr := expr_typecheck(df, n.sorted)
		if serr != .None {
			return {}, serr
		}
		vt, verr := expr_typecheck(df, n.values)
		if verr != .None {
			return {}, verr
		}
		switch st {
		case typeid_of(i8), typeid_of(i16), typeid_of(i32), typeid_of(i64),
		     typeid_of(u8), typeid_of(u16), typeid_of(u32), typeid_of(u64),
		     typeid_of(int), typeid_of(uint), typeid_of(f32), typeid_of(f64),
		     typeid_of(string):
		case:
			return {}, .Unsupported_Operation
		}
		if st != vt {
			return {}, .Type_Mismatch
		}
		return typeid_of(int), .None

	case expr.Agg:
		ct, cerr := expr_typecheck(df, n.expr)
		if cerr != .None {
			return {}, cerr
		}
		if n.kind == .Quantile && (n.q < 0 || n.q > 1) {
			return {}, .Invalid_Argument
		}
		switch n.kind {
		case .Count:
			return typeid_of(i64), .None
		case .N_Unique:
			if !is_supported_agg_value_type(ct) {
				return {}, .Unsupported_Operation
			}
			return typeid_of(i64), .None
		case .Sum, .Mean, .Var, .Std, .Median, .Quantile, .Product, .Skew, .Kurtosis:
			if !is_numeric_type(ct) {
				return {}, .Unsupported_Operation
			}
			return typeid_of(f64), .None
		case .Min, .Max:
			if !is_numeric_type(ct) && ct != typeid_of(string) {
				return {}, .Unsupported_Operation
			}
			return ct, .None
		case .Mode:
			if !is_supported_agg_value_type(ct) {
				return {}, .Unsupported_Operation
			}
			return ct, .None
		case .First, .Last:
			return ct, .None
		case:
			return {}, .Invalid_Argument
		}

	case expr.Cov:
		return two_column_stat_typecheck(df, n.lhs, n.rhs)

	case expr.Corr:
		return two_column_stat_typecheck(df, n.lhs, n.rhs)

	case expr.Window:
		ct: typeid
		if n.expr != nil {
			child_t, cerr := expr_typecheck(df, n.expr)
			if cerr != .None {
				return {}, cerr
			}
			ct = child_t
		}
		if n.func != .Row_Number && !column_supported_window_type(ct) {
			return {}, .Unsupported_Operation
		}
		switch n.func {
		case .Row_Number:
			return typeid_of(i64), .None
		case .Rank, .Ewma:
			if !is_numeric_type(ct) {
				return {}, .Unsupported_Operation
			}
			return typeid_of(f64), .None
		case .Cum_Sum, .Cum_Min, .Cum_Max:
			if !is_numeric_type(ct) {
				return {}, .Unsupported_Operation
			}
			return ct, .None
		case .Shift:
			return ct, .None
		case .Rolling, .Cumulative_Eval:
			if n.agg == .Quantile {
				return {}, .Invalid_Argument // no q channel on Window
			}
			if v_err := validate_agg(n.agg, ct); v_err != .None {
				return {}, v_err
			}
			if n.func == .Rolling && n.n < 1 {
				return {}, .Invalid_Argument
			}
			rt, rok := agg_result_dtype(n.agg, ct)
			if !rok {
				return {}, .Unsupported_Operation
			}
			return rt, .None
		case:
			return {}, .Invalid_Argument
		}

	case:
		return {}, .Invalid_Argument
	}
}

// two_column_stat_typecheck validates the common shape of Cov/Corr: two
// numeric, type-consistent expression results.
@(private)
two_column_stat_typecheck :: proc(df: ^DataFrame, lhs_e, rhs_e: ^expr.Expr) -> (typeid, Error) {
	lt, lerr := expr_typecheck(df, lhs_e)
	if lerr != .None {
		return {}, lerr
	}
	rt, rerr := expr_typecheck(df, rhs_e)
	if rerr != .None {
		return {}, rerr
	}
	if !is_numeric_type(lt) || !is_numeric_type(rt) {
		return {}, .Unsupported_Operation
	}
	return typeid_of(f64), .None
}

// is_supported_agg_value_type reports whether a column type supports the
// value-preserving aggregations (mode/n_unique, which hash their values).
@(private)
is_supported_agg_value_type :: proc(dtype: typeid) -> bool {
	switch dtype {
	case typeid_of(i8), typeid_of(i16), typeid_of(i32), typeid_of(i64),
	     typeid_of(u8), typeid_of(u16), typeid_of(u32), typeid_of(u64),
	     typeid_of(int), typeid_of(uint), typeid_of(f32), typeid_of(f64),
	     typeid_of(bool), typeid_of(string):
		return true
	}
	return false
}
