package dataframe

// Expression evaluation (DESIGN.md §6.4). Evaluation lives in the dataframe
// package (not dataframe/expr) because kernels operate on Column/DataFrame;
// package imports are one-directional: dataframe -> expr.

import "core:mem"
import "core:math"
import "core:strings"
import "base:runtime"
import "expr"

// expr_eval evaluates e against df and returns a new, caller-owned Column.
// The result column is named: the alias name when the root is Alias, the
// source name when the root is Col, else "" (rename before adding to a
// DataFrame).
//
// When arena is non-nil, intermediate child columns are allocated from the
// arena (bump-pointer, bulk-freed later).  The result column is always
// allocated from allocator, preserving ownership semantics.
expr_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, e: ^expr.Expr, arena: ^OpArena = nil) -> (Column, Error) {
	inter_alloc := allocator
	if arena != nil {
		inter_alloc = op_arena_allocator(arena)
	}

	switch n in e^ {
	case expr.Col:
		col, err := dataframe_get_column(df, n.name)
		if err != .None {
			return {}, err
		}
		return column_copy(col, allocator)

	case expr.Lit:
		return literal_column(allocator, df, n)

	case expr.Binary:
		lhs, lerr := expr_eval(inter_alloc, df, n.lhs, arena)
		if lerr != .None {
			return {}, lerr
		}
		defer column_destroy(&lhs)
		rhs, rerr := expr_eval(inter_alloc, df, n.rhs, arena)
		if rerr != .None {
			return {}, rerr
		}
		defer column_destroy(&rhs)
		_, lhs_is_lit := n.lhs^.(expr.Lit)
		_, rhs_is_lit := n.rhs^.(expr.Lit)
		return binary_eval(allocator, n.op, &lhs, &rhs, lhs_is_lit, rhs_is_lit)

	case expr.Unary:
		child, cerr := expr_eval(inter_alloc, df, n.expr, arena)
		if cerr != .None {
			return {}, cerr
		}
		defer column_destroy(&child)
		return unary_eval(allocator, n.op, &child)

	case expr.Cast:
		child, cerr := expr_eval(inter_alloc, df, n.expr, arena)
		if cerr != .None {
			return {}, cerr
		}
		defer column_destroy(&child)
		return cast_eval(allocator, &child, n.to)

	case expr.Alias:
		out, aerr := expr_eval(allocator, df, n.expr)
		if aerr != .None {
			return {}, aerr
		}
		name_copy, nerr := clone_name(out.alloc, n.name)
		if nerr != .None {
			column_destroy(&out)
			return {}, nerr
		}
		delete_string(out.name, out.alloc)
		out.name = name_copy
		return out, .None

	case expr.Not_Null:
		child, cerr := expr_eval(inter_alloc, df, n.expr, arena)
		if cerr != .None {
			return {}, cerr
		}
		defer column_destroy(&child)
		return not_null_eval(allocator, &child)

	case expr.Is_Nan:
		return is_nan_eval(allocator, df, n)

	case expr.Fill_Null:
		return fill_null_eval(allocator, df, n)

	case expr.Coalesce:
		return coalesce_eval(allocator, df, n)

	case expr.Forward_Fill:
		return fill_forward_eval(allocator, df, n)

	case expr.Backward_Fill:
		return fill_backward_eval(allocator, df, n)

	case expr.Interpolate:
		return interpolate_eval(allocator, df, n)

	case expr.Func:
		return func_eval(allocator, df, n)

	case expr.Is_Between:
		return is_between_eval(allocator, df, n)

	case expr.Is_In:
		return is_in_eval(allocator, df, n)

	case expr.Arange:
		return arange_eval(allocator, df, n)

	case expr.Arg_Where:
		return arg_where_eval(allocator, df, n)

	case expr.Distinct:
		return distinct_eval(allocator, df, n)

	case expr.Dot_Product:
		return dot_product_eval(allocator, df, n)

	case expr.Concat_Str:
		return concat_str_eval(allocator, df, n)

	case expr.Search_Sorted:
		return search_sorted_eval(allocator, df, n)

	case expr.Agg:
		return agg_eval(allocator, df, n)

	case expr.Cov:
		return cov_eval(allocator, df, n)

	case expr.Corr:
		return corr_eval(allocator, df, n)

	case expr.Window:
		return window_eval(allocator, df, n)

	case:
		return {}, .Invalid_Argument
	}
}

// --- type helpers ------------------------------------------------------------

// is_numeric_type reports whether dtype is one of the supported numeric types.
@(private)
is_numeric_type :: proc(dtype: typeid) -> bool {
	switch dtype {
	case typeid_of(i8), typeid_of(i16), typeid_of(i32), typeid_of(i64),
	     typeid_of(u8), typeid_of(u16), typeid_of(u32), typeid_of(u64),
	     typeid_of(int), typeid_of(uint),
	     typeid_of(f32), typeid_of(f64):
		return true
	}
	return false
}

// is_int_type reports whether dtype is a supported integer type.
@(private)
is_int_type :: proc(dtype: typeid) -> bool {
	switch dtype {
	case typeid_of(i8), typeid_of(i16), typeid_of(i32), typeid_of(i64),
	     typeid_of(u8), typeid_of(u16), typeid_of(u32), typeid_of(u64),
	     typeid_of(int), typeid_of(uint):
		return true
	}
	return false
}

// type_layout returns the element size and alignment of a supported column
// type. ok is false for unsupported types.
@(private)
type_layout :: proc(dtype: typeid) -> (size: int, align: int, ok: bool) {
	switch dtype {
	case typeid_of(i8), typeid_of(u8):
		return size_of(i8), align_of(i8), true
	case typeid_of(i16), typeid_of(u16):
		return size_of(i16), align_of(i16), true
	case typeid_of(i32), typeid_of(u32), typeid_of(f32):
		return size_of(i32), align_of(i32), true
	case typeid_of(i64), typeid_of(u64), typeid_of(int), typeid_of(uint), typeid_of(f64):
		return size_of(i64), align_of(i64), true
	case typeid_of(Date), typeid_of(Datetime), typeid_of(Time), typeid_of(Duration):
		return size_of(i64), align_of(i64), true
	case typeid_of(bool):
		return size_of(bool), align_of(bool), true
	case typeid_of(string):
		return size_of(string), align_of(string), true
	case typeid_of(List_Ref):
		return size_of(List_Ref), align_of(List_Ref), true
	}
	return 0, 0, false
}

// row_valid reports whether row i is non-NULL (nil validity == all valid).
@(private)
row_valid :: proc(valid: []u64, i: int) -> bool {
	return valid == nil || bm_get(valid, i)
}

// copy_validity mirrors src's NULL rows onto out, allocating out.valid only
// when src has NULLs.
@(private)
copy_validity :: proc(allocator: mem.Allocator, src, out: ^Column) -> Error {
	if src.valid == nil {
		return .None
	}
	w := bm_words(out.count)
	v := make([]u64, w, allocator)
	if v == nil && w != 0 {
		return .Allocator_Failure
	}
	copy(v, src.valid)
	out.valid = v
	return .None
}

// --- literals ----------------------------------------------------------------

// literal_column builds a constant column of the literal value, with as many
// rows as df has. Slice literals (e.g. []i32 for search_sorted) build a column
// of len(slice) rows holding the slice's elements. Unsupported literal types
// return an error.
literal_column :: proc(allocator: mem.Allocator, df: ^DataFrame, l: expr.Lit) -> (Column, Error) {
	if elem := slice_elem_typeid(l.dtype); elem != {} {
		return literal_slice_column(allocator, l, l.dtype, elem)
	}
	size, align, ok := type_layout(l.dtype)
	if !ok {
		return {}, .Unsupported_Operation
	}
	out, err := column_alloc(allocator, "", l.dtype, size, align, dataframe_num_rows(df))
	if err != .None {
		return {}, err
	}
	local := l
	for i in 0 ..< out.count {
		mem.copy(ptr_offset(out.data, i * size), &local.data, size)
	}
	return out, .None
}

// slice_elem_typeid maps a slice typeid to its element typeid, or {} when the
// slice type is not supported.
@(private)
slice_elem_typeid :: proc(t: typeid) -> typeid {
	switch t {
	case typeid_of([]i8):   return typeid_of(i8)
	case typeid_of([]i16):  return typeid_of(i16)
	case typeid_of([]i32):  return typeid_of(i32)
	case typeid_of([]i64):  return typeid_of(i64)
	case typeid_of([]u8):   return typeid_of(u8)
	case typeid_of([]u16):  return typeid_of(u16)
	case typeid_of([]u32):  return typeid_of(u32)
	case typeid_of([]u64):  return typeid_of(u64)
	case typeid_of([]int):  return typeid_of(int)
	case typeid_of([]uint): return typeid_of(uint)
	case typeid_of([]f32):  return typeid_of(f32)
	case typeid_of([]f64):  return typeid_of(f64)
	case typeid_of([]bool): return typeid_of(bool)
	case typeid_of([]string): return typeid_of(string)
	case:
		return {}
	}
}

// literal_slice_column builds a column from a slice literal: one row per
// element, all valid. slice_dtype is the literal's type ([]T); elem_dtype the
// element type.
@(private)
literal_slice_column :: proc(allocator: mem.Allocator, l: expr.Lit, slice_dtype: typeid, elem_dtype: typeid) -> (Column, Error) {
	switch slice_dtype {
	case typeid_of([]i8):   return literal_slice_typed(allocator, l, i8)
	case typeid_of([]i16):  return literal_slice_typed(allocator, l, i16)
	case typeid_of([]i32):  return literal_slice_typed(allocator, l, i32)
	case typeid_of([]i64):  return literal_slice_typed(allocator, l, i64)
	case typeid_of([]u8):   return literal_slice_typed(allocator, l, u8)
	case typeid_of([]u16):  return literal_slice_typed(allocator, l, u16)
	case typeid_of([]u32):  return literal_slice_typed(allocator, l, u32)
	case typeid_of([]u64):  return literal_slice_typed(allocator, l, u64)
	case typeid_of([]int):  return literal_slice_typed(allocator, l, int)
	case typeid_of([]uint): return literal_slice_typed(allocator, l, uint)
	case typeid_of([]f32):  return literal_slice_typed(allocator, l, f32)
	case typeid_of([]f64):  return literal_slice_typed(allocator, l, f64)
	case typeid_of([]bool): return literal_slice_typed(allocator, l, bool)
	case typeid_of([]string): return literal_slice_typed(allocator, l, string)
	case:
		return {}, .Unsupported_Operation
	}
}

@(private)
literal_slice_typed :: proc(allocator: mem.Allocator, l: expr.Lit, $T: typeid) -> (Column, Error) {
	local := l
	values, ok := expr.lit_as(&local, []T)
	if !ok {
		return {}, .Type_Mismatch
	}
	return column_from("", values, allocator)
}

// ptr_offset advances a raw pointer by n bytes.
@(private)
ptr_offset :: proc(p: rawptr, n: int) -> rawptr {
	return (^u8)(uintptr(p) + uintptr(n))
}

// --- binary ops --------------------------------------------------------------

// binary_eval computes lhs op rhs elementwise. One operand may be a literal
// column that gets implicitly converted to the other side's type when both
// are numeric (DESIGN.md §6.3). The result column is owned by the caller.
@(private)
binary_eval :: proc(allocator: mem.Allocator, op: expr.Binary_Op, lhs, rhs: ^Column, lhs_is_lit, rhs_is_lit: bool) -> (Column, Error) {
	// literal coercion (the only implicit conversion, constants only)
	if lhs.dtype != rhs.dtype {
		if lhs_is_lit && is_numeric_type(lhs.dtype) && is_numeric_type(rhs.dtype) {
			coerced, cerr := cast_eval(allocator, lhs, rhs.dtype)
			if cerr != .None {
				return {}, cerr
			}
			column_destroy(lhs)
			lhs^ = coerced
		} else if rhs_is_lit && is_numeric_type(rhs.dtype) && is_numeric_type(lhs.dtype) {
			coerced, cerr := cast_eval(allocator, rhs, lhs.dtype)
			if cerr != .None {
				return {}, cerr
			}
			column_destroy(rhs)
			rhs^ = coerced
		} else {
			return {}, .Type_Mismatch
		}
	}

	if err := validate_binary(op, lhs.dtype); err != .None {
		return {}, err
	}

	out_dtype := lhs.dtype
	if is_cmp_op(op) {
		out_dtype = typeid_of(bool)
	}
	size, align, ok := type_layout(out_dtype)
	if !ok {
		return {}, .Unsupported_Operation
	}

	out, err := column_alloc(allocator, "", out_dtype, size, align, lhs.count)
	if err != .None {
		return {}, err
	}

	// validity is all-true, then kernels clear rows where an operand is NULL
	// (or integer division/modulo by zero).
	may_null := lhs.valid != nil || rhs.valid != nil || ((op == .Div || op == .Mod) && is_int_type(out_dtype))
	if may_null {
		v := bm_make(out.count, true, allocator)
		if v == nil && out.count != 0 {
			column_destroy(&out)
			return {}, .Allocator_Failure
		}
		out.valid = v
	}

	b_err := binary_op(op, lhs, rhs, &out)
	if b_err != .None {
		column_destroy(&out)
		return {}, b_err
	}
	return out, .None
}

// validate_binary checks that op is supported for columns of dtype.
@(private)
validate_binary :: proc(op: expr.Binary_Op, dtype: typeid) -> Error {
	switch op {
	case .Add, .Sub, .Mul, .Div, .Mod:
		if !is_numeric_type(dtype) {
			return .Unsupported_Operation
		}
	case .Eq, .Ne:
		if is_numeric_type(dtype) || dtype == typeid_of(bool) || dtype == typeid_of(string) {
			return .None
		}
		return .Unsupported_Operation
	case .Lt, .Le, .Gt, .Ge:
		if dtype == typeid_of(bool) {
			return .Unsupported_Operation
		}
		if is_numeric_type(dtype) || dtype == typeid_of(string) {
			return .None
		}
		return .Unsupported_Operation
	case .And, .Or:
		if dtype != typeid_of(bool) {
			return .Unsupported_Operation
		}
	}
	return .None
}

@(private)
is_arith_op :: proc(op: expr.Binary_Op) -> bool {
	return op == .Add || op == .Sub || op == .Mul || op == .Div || op == .Mod
}

@(private)
is_cmp_op :: proc(op: expr.Binary_Op) -> bool {
	return op == .Eq || op == .Ne || op == .Lt || op == .Le || op == .Gt || op == .Ge
}

// binary_op dispatches to the typed kernel. lhs and rhs have equal dtype
// (enforced by binary_eval).
@(private)
binary_op :: proc(op: expr.Binary_Op, lhs, rhs, out: ^Column) -> Error {
	if is_arith_op(op) {
		switch lhs.dtype {
		case typeid_of(i8):   return binary_arith(op, lhs, rhs, out, i8)
		case typeid_of(i16):  return binary_arith(op, lhs, rhs, out, i16)
		case typeid_of(i32):  return binary_arith(op, lhs, rhs, out, i32)
		case typeid_of(i64):  return binary_arith(op, lhs, rhs, out, i64)
		case typeid_of(u8):   return binary_arith(op, lhs, rhs, out, u8)
		case typeid_of(u16):  return binary_arith(op, lhs, rhs, out, u16)
		case typeid_of(u32):  return binary_arith(op, lhs, rhs, out, u32)
		case typeid_of(u64):  return binary_arith(op, lhs, rhs, out, u64)
		case typeid_of(int):  return binary_arith(op, lhs, rhs, out, int)
		case typeid_of(uint): return binary_arith(op, lhs, rhs, out, uint)
		case typeid_of(f32):  return binary_arith(op, lhs, rhs, out, f32)
		case typeid_of(f64):  return binary_arith(op, lhs, rhs, out, f64)
		}
		return .Unsupported_Operation
	}
	if is_cmp_op(op) {
		return binary_cmp_dispatch(op, lhs, rhs, out)
	}
	if op == .And || op == .Or {
		return binary_bool(op, lhs, rhs, out)
	}
	return .Unsupported_Operation
}

// binary_cmp_dispatch routes an equality/ordering op to the kernel for the
// concrete dtype of lhs/rhs.
@(private)
binary_cmp_dispatch :: proc(op: expr.Binary_Op, lhs, rhs, out: ^Column) -> Error {
	if lhs.dtype == typeid_of(bool) {
		return binary_bool(op, lhs, rhs, out)
	}
	switch lhs.dtype {
	case typeid_of(i8):   binary_cmp(op, lhs, rhs, out, i8)
	case typeid_of(i16):  binary_cmp(op, lhs, rhs, out, i16)
	case typeid_of(i32):  binary_cmp(op, lhs, rhs, out, i32)
	case typeid_of(i64):  binary_cmp(op, lhs, rhs, out, i64)
	case typeid_of(u8):   binary_cmp(op, lhs, rhs, out, u8)
	case typeid_of(u16):  binary_cmp(op, lhs, rhs, out, u16)
	case typeid_of(u32):  binary_cmp(op, lhs, rhs, out, u32)
	case typeid_of(u64):  binary_cmp(op, lhs, rhs, out, u64)
	case typeid_of(int):  binary_cmp(op, lhs, rhs, out, int)
	case typeid_of(uint): binary_cmp(op, lhs, rhs, out, uint)
	case typeid_of(f32):  binary_cmp(op, lhs, rhs, out, f32)
	case typeid_of(f64):  binary_cmp(op, lhs, rhs, out, f64)
	case typeid_of(string): binary_cmp(op, lhs, rhs, out, string)
	}
	return .None
}

// binary_arith computes + - * / % elementwise over two same-typed numeric
// columns. out has dtype T. zero_is_null makes integer division/modulo by
// zero yield NULL instead of panicking.
@(private)
binary_arith :: proc(op: expr.Binary_Op, lhs, rhs, out: ^Column, $T: typeid) -> Error {
	n := lhs.count
	lv := column_typed_view(lhs, T)
	rv := column_typed_view(rhs, T)
	ov := column_typed_view(out, T)
	zero_is_null := is_int_type(typeid_of(T))

	// Fast path: both operands all-valid. Skips per-element row_valid
	// checks, removing branches from the inner loop and enabling compiler
	// auto-vectorization.
	if lhs.valid == nil && rhs.valid == nil {
		#partial switch op {
		case .Add:
			when T == f64 { parallel_simd_add_f64(lv, rv, ov) }
			else when T == i64 { parallel_simd_add_i64(lv, rv, ov) }
			else { simd_add(lv, rv, ov) }
		case .Sub:
			when T == f64 { parallel_simd_sub_f64(lv, rv, ov) }
			else when T == i64 { parallel_simd_sub_i64(lv, rv, ov) }
			else { simd_sub(lv, rv, ov) }
		case .Mul:
			when T == f64 { parallel_simd_mul_f64(lv, rv, ov) }
			else when T == i64 { parallel_simd_mul_i64(lv, rv, ov) }
			else { simd_mul(lv, rv, ov) }
		case .Div:
			for i in 0 ..< n {
				if zero_is_null && rv[i] == 0 {
					if out.valid != nil {
						bm_set(out.valid, i, false)
					}
					continue
				}
				ov[i] = lv[i] / rv[i]
			}
		case .Mod:
			for i in 0 ..< n {
				mod_row(op, lhs, rhs, out, i, T)
			}
		default:
			for i in 0 ..< n {
				#partial switch op {
				}
			}
		}
		return .None
	}

	for i in 0 ..< n {
		if !row_valid(lhs.valid, i) || !row_valid(rhs.valid, i) {
			if out.valid != nil {
				bm_set(out.valid, i, false)
			}
			continue
		}
		#partial switch op {
		case .Add:
			ov[i] = lv[i] + rv[i]
		case .Sub:
			ov[i] = lv[i] - rv[i]
		case .Mul:
			ov[i] = lv[i] * rv[i]
		case .Div:
			if zero_is_null && rv[i] == 0 {
				if out.valid != nil {
					bm_set(out.valid, i, false)
				}
				continue
			}
			ov[i] = lv[i] / rv[i]
		case .Mod:
			mod_row(op, lhs, rhs, out, i, T)
		}
	}
	return .None
}

// mod_row computes lv[i] % rv[i] into out[i]; float modulo uses math.mod_*,
// integer modulo uses % and yields NULL on divisor zero (the validity flags
// are set up by binary_eval, which marks integer Div/Mod as may_null).
@(private)
mod_row :: proc(op: expr.Binary_Op, lhs, rhs, out: ^Column, i: int, $T: typeid) {
	if !row_valid(lhs.valid, i) || !row_valid(rhs.valid, i) {
		if out.valid != nil {
			bm_set(out.valid, i, false)
		}
		return
	}
	lv := column_typed_view(lhs, T)
	rv := column_typed_view(rhs, T)
	ov := column_typed_view(out, T)
	if is_int_type(typeid_of(T)) && rv[i] == 0 {
		if out.valid != nil {
			bm_set(out.valid, i, false)
		}
		return
	}
	when T == f32 {
		ov[i] = math.mod_f32(lv[i], rv[i])
	} else when T == f64 {
		ov[i] = math.mod_f64(lv[i], rv[i])
	} else {
		ov[i] = lv[i] % rv[i]
	}
}

// binary_cmp computes == != < <= > >= elementwise; out has dtype bool.
@(private)
binary_cmp :: proc(op: expr.Binary_Op, lhs, rhs, out: ^Column, $T: typeid) {
	n := lhs.count
	lv := column_typed_view(lhs, T)
	rv := column_typed_view(rhs, T)
	ov := column_typed_view(out, bool)

	// Fast path: both operands all-valid.
	if lhs.valid == nil && rhs.valid == nil {
		for i in 0 ..< n {
			#partial switch op {
			case .Eq: ov[i] = lv[i] == rv[i]
			case .Ne: ov[i] = lv[i] != rv[i]
			case .Lt: ov[i] = lv[i] < rv[i]
			case .Le: ov[i] = lv[i] <= rv[i]
			case .Gt: ov[i] = lv[i] > rv[i]
			case .Ge: ov[i] = lv[i] >= rv[i]
			}
		}
		return
	}

	for i in 0 ..< n {
		if !row_valid(lhs.valid, i) || !row_valid(rhs.valid, i) {
			if out.valid != nil {
				bm_set(out.valid, i, false)
			}
			continue
		}
		#partial switch op {
		case .Eq:
			ov[i] = lv[i] == rv[i]
		case .Ne:
			ov[i] = lv[i] != rv[i]
		case .Lt:
			ov[i] = lv[i] < rv[i]
		case .Le:
			ov[i] = lv[i] <= rv[i]
		case .Gt:
			ov[i] = lv[i] > rv[i]
		case .Ge:
			ov[i] = lv[i] >= rv[i]
		}
	}
}

// binary_bool computes and/or (and equality) over two bool columns.
@(private)
binary_bool :: proc(op: expr.Binary_Op, lhs, rhs, out: ^Column) -> Error {
	n := lhs.count
	lb := column_typed_view(lhs, bool)
	rb := column_typed_view(rhs, bool)
	ov := column_typed_view(out, bool)

	// Fast path: both operands all-valid.
	if lhs.valid == nil && rhs.valid == nil {
		for i in 0 ..< n {
			#partial switch op {
			case .And: ov[i] = lb[i] && rb[i]
			case .Or:  ov[i] = lb[i] || rb[i]
			case .Eq:  ov[i] = lb[i] == rb[i]
			case .Ne:  ov[i] = lb[i] != rb[i]
			}
		}
		return .None
	}

	for i in 0 ..< n {
		if !row_valid(lhs.valid, i) || !row_valid(rhs.valid, i) {
			if out.valid != nil {
				bm_set(out.valid, i, false)
			}
			continue
		}
		#partial switch op {
		case .And:
			ov[i] = lb[i] && rb[i]
		case .Or:
			ov[i] = lb[i] || rb[i]
		case .Eq:
			ov[i] = lb[i] == rb[i]
		case .Ne:
			ov[i] = lb[i] != rb[i]
		}
	}
	return .None
}

// --- unary ops ---------------------------------------------------------------

// unary_eval computes -x (numeric) or !x (bool).
@(private)
unary_eval :: proc(allocator: mem.Allocator, op: expr.Unary_Op, child: ^Column) -> (Column, Error) {
	switch op {
	case .Neg:
		if !is_numeric_type(child.dtype) {
			return {}, .Unsupported_Operation
		}
		size, align, ok := type_layout(child.dtype)
		if !ok {
			return {}, .Unsupported_Operation
		}
		out, err := column_alloc(allocator, child.name, child.dtype, size, align, child.count)
		if err != .None {
			return {}, err
		}
		copy_validity(allocator, child, &out)
		switch child.dtype {
		case typeid_of(i8):   neg_typed(child, &out, i8)
		case typeid_of(i16):  neg_typed(child, &out, i16)
		case typeid_of(i32):  neg_typed(child, &out, i32)
		case typeid_of(i64):  neg_typed(child, &out, i64)
		case typeid_of(int):  neg_typed(child, &out, int)
		case typeid_of(f32):  neg_typed(child, &out, f32)
		case typeid_of(f64):  neg_typed(child, &out, f64)
		case:
			return {}, .Unsupported_Operation
		}
		return out, .None

	case .Not:
		if child.dtype != typeid_of(bool) {
			return {}, .Unsupported_Operation
		}
		out, err := column_alloc(allocator, child.name, bool, size_of(bool), align_of(bool), child.count)
		if err != .None {
			return {}, err
		}
		copy_validity(allocator, child, &out)
		iv := column_typed_view(child, bool)
		ov := column_typed_view(&out, bool)
		for i in 0 ..< child.count {
			if row_valid(child.valid, i) {
				ov[i] = !iv[i]
			}
		}
		return out, .None

	case:
		return {}, .Unsupported_Operation
	}
}

// neg_typed negates elementwise; out mirrors child's validity.
@(private)
neg_typed :: proc(child, out: ^Column, $T: typeid) {
	iv := column_typed_view(child, T)
	ov := column_typed_view(out, T)
	if child.valid == nil {
		when T == f64 { parallel_simd_neg_f64(iv, ov) }
		else when T == i64 { parallel_simd_neg_i64(iv, ov) }
		else { simd_neg(iv, ov) }
		return
	}
	for i in 0 ..< len(iv) {
		if row_valid(child.valid, i) {
			ov[i] = -iv[i]
		}
	}
}

// --- cast --------------------------------------------------------------------

// cast_eval converts a numeric column to another numeric type. Only explicit
// casts are supported; there is no implicit conversion (principle 6), except
// literal coercion handled in binary_eval.
@(private)
cast_eval :: proc(allocator: mem.Allocator, child: ^Column, to: typeid) -> (Column, Error) {
	if !is_numeric_type(child.dtype) || !is_numeric_type(to) {
		return {}, .Unsupported_Operation
	}
	if child.dtype == to {
		return column_copy(child, allocator)
	}
	size, align, ok := type_layout(to)
	if !ok {
		return {}, .Unsupported_Operation
	}
	out, err := column_alloc(allocator, child.name, to, size, align, child.count)
	if err != .None {
		return {}, err
	}
	copy_validity(allocator, child, &out)

	switch child.dtype {
	case typeid_of(i8):   cast_typed(child, &out, i8)
	case typeid_of(i16):  cast_typed(child, &out, i16)
	case typeid_of(i32):  cast_typed(child, &out, i32)
	case typeid_of(i64):  cast_typed(child, &out, i64)
	case typeid_of(u8):   cast_typed(child, &out, u8)
	case typeid_of(u16):  cast_typed(child, &out, u16)
	case typeid_of(u32):  cast_typed(child, &out, u32)
	case typeid_of(u64):  cast_typed(child, &out, u64)
	case typeid_of(int):  cast_typed(child, &out, int)
	case typeid_of(uint): cast_typed(child, &out, uint)
	case typeid_of(f32):  cast_typed(child, &out, f32)
	case typeid_of(f64):  cast_typed(child, &out, f64)
	case:
		column_destroy(&out)
		return {}, .Unsupported_Operation
	}
	return out, .None
}

// cast_typed casts src (type T) into out (target U), elementwise, preserving
// NULLs (out mirrors src's validity).
@(private)
cast_typed :: proc(src, out: ^Column, $T: typeid) {
	switch out.dtype {
	case typeid_of(i8):   cast_loop(src, out, T, i8)
	case typeid_of(i16):  cast_loop(src, out, T, i16)
	case typeid_of(i32):  cast_loop(src, out, T, i32)
	case typeid_of(i64):  cast_loop(src, out, T, i64)
	case typeid_of(u8):   cast_loop(src, out, T, u8)
	case typeid_of(u16):  cast_loop(src, out, T, u16)
	case typeid_of(u32):  cast_loop(src, out, T, u32)
	case typeid_of(u64):  cast_loop(src, out, T, u64)
	case typeid_of(int):  cast_loop(src, out, T, int)
	case typeid_of(uint): cast_loop(src, out, T, uint)
	case typeid_of(f32):  cast_loop(src, out, T, f32)
	case typeid_of(f64):  cast_loop(src, out, T, f64)
	}
}

// cast_loop converts src values (type T) into out (type U) elementwise.
@(private)
cast_loop :: proc(src, out: ^Column, $T: typeid, $U: typeid) {
	sv := column_typed_view(src, T)
	ov := column_typed_view(out, U)
	for i in 0 ..< len(sv) {
		if row_valid(src.valid, i) {
			ov[i] = U(sv[i])
		}
	}
}

// --- null tests --------------------------------------------------------------

// not_null_eval builds a bool column that is true where the child row is
// non-NULL. The result is always fully valid.
@(private)
not_null_eval :: proc(allocator: mem.Allocator, child: ^Column) -> (Column, Error) {
	out, err := column_alloc(allocator, child.name, bool, size_of(bool), align_of(bool), child.count)
	if err != .None {
		return {}, err
	}
	ov := column_typed_view(&out, bool)
	for i in 0 ..< child.count {
		ov[i] = row_valid(child.valid, i)
	}
	return out, .None
}

// --- S3.10 function expressions ----------------------------------------------

// eval_child evaluates a single child expression against df.
@(private)
eval_child :: proc(allocator: mem.Allocator, df: ^DataFrame, e: ^expr.Expr) -> (Column, Error) {
	out, err := expr_eval(allocator, df, e)
	if err != .None {
		return {}, err
	}
	return out, .None
}

// func_eval implements the Func node (S3.10): a single-child numeric function.
@(private)
func_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Func) -> (Column, Error) {
	child, cerr := eval_child(allocator, df, n.expr)
	if cerr != .None {
		return {}, cerr
	}
	defer column_destroy(&child)
	if !is_numeric_type(child.dtype) {
		return {}, .Unsupported_Operation
	}

	out, err := column_alloc(allocator, child.name, child.dtype, size_of_ty(child.dtype), align_of_ty(child.dtype), child.count)
	if err != .None {
		return {}, err
	}
	column_copy_validity(&out, &child)

	#partial switch n.kind {
	case .Abs:
		func_abs(child.dtype, &child, &out)
	case .Sign:
		func_sign(child.dtype, &child, &out)
	case .Round:
		func_round(child.dtype, &child, &out, n.decimals)
	case .Diff:
		func_diff(child.dtype, &child, &out, n.n)
	case .Cum_Sum:
		func_cum_sum(child.dtype, &child, &out)
	case .Pct_Change:
		column_destroy(&out)
		out, err = column_alloc(allocator, child.name, f64, size_of(f64), align_of(f64), child.count)
		if err != .None {
			return {}, err
		}
		column_copy_validity(&out, &child)
		func_pct_change(child.dtype, &child, &out, n.n)
	case:
		column_destroy(&out)
		return {}, .Invalid_Argument
	}
	return out, .None
}

// size_of_ty / align_of_ty return the size/alignment of a runtime dtype.
@(private)
size_of_ty :: proc(t: typeid) -> int {
	switch t {
	case typeid_of(i8):   return size_of(i8)
	case typeid_of(i16):  return size_of(i16)
	case typeid_of(i32):  return size_of(i32)
	case typeid_of(i64):  return size_of(i64)
	case typeid_of(u8):   return size_of(u8)
	case typeid_of(u16):  return size_of(u16)
	case typeid_of(u32):  return size_of(u32)
	case typeid_of(u64):  return size_of(u64)
	case typeid_of(int):  return size_of(int)
	case typeid_of(uint): return size_of(uint)
	case typeid_of(f32):  return size_of(f32)
	case typeid_of(f64):  return size_of(f64)
	case typeid_of(bool): return size_of(bool)
	case typeid_of(string): return size_of(string)
	case typeid_of(Date), typeid_of(Datetime), typeid_of(Time), typeid_of(Duration):
		return size_of(i64)
	case:
		return 0
	}
}

@(private)
align_of_ty :: proc(t: typeid) -> int {
	switch t {
	case typeid_of(i8):   return align_of(i8)
	case typeid_of(i16):  return align_of(i16)
	case typeid_of(i32):  return align_of(i32)
	case typeid_of(i64):  return align_of(i64)
	case typeid_of(u8):   return align_of(u8)
	case typeid_of(u16):  return align_of(u16)
	case typeid_of(u32):  return align_of(u32)
	case typeid_of(u64):  return align_of(u64)
	case typeid_of(int):  return align_of(int)
	case typeid_of(uint): return align_of(uint)
	case typeid_of(f32):  return align_of(f32)
	case typeid_of(f64):  return align_of(f64)
	case typeid_of(bool): return align_of(bool)
	case typeid_of(string): return align_of(string)
	case typeid_of(Date), typeid_of(Datetime), typeid_of(Time), typeid_of(Duration):
		return align_of(i64)
	case:
		return 0
	}
}

// slice_typeid_of maps an element typeid to the []element typeid, or {} when
// the element type is not supported.
@(private)
slice_typeid_of :: proc(t: typeid) -> typeid {
	switch t {
	case typeid_of(i8):   return typeid_of([]i8)
	case typeid_of(i16):  return typeid_of([]i16)
	case typeid_of(i32):  return typeid_of([]i32)
	case typeid_of(i64):  return typeid_of([]i64)
	case typeid_of(u8):   return typeid_of([]u8)
	case typeid_of(u16):  return typeid_of([]u16)
	case typeid_of(u32):  return typeid_of([]u32)
	case typeid_of(u64):  return typeid_of([]u64)
	case typeid_of(int):  return typeid_of([]int)
	case typeid_of(uint): return typeid_of([]uint)
	case typeid_of(f32):  return typeid_of([]f32)
	case typeid_of(f64):  return typeid_of([]f64)
	case typeid_of(bool): return typeid_of([]bool)
	case typeid_of(string): return typeid_of([]string)
	case:
		return {}
	}
}

@(private)
func_abs :: proc(t: typeid, child, out: ^Column) {
	switch t {
	case typeid_of(i8):   func_abs_typed(child, out, i8)
	case typeid_of(i16):  func_abs_typed(child, out, i16)
	case typeid_of(i32):  func_abs_typed(child, out, i32)
	case typeid_of(i64):  func_abs_typed(child, out, i64)
	case typeid_of(u8):   func_abs_typed(child, out, u8)
	case typeid_of(u16):  func_abs_typed(child, out, u16)
	case typeid_of(u32):  func_abs_typed(child, out, u32)
	case typeid_of(u64):  func_abs_typed(child, out, u64)
	case typeid_of(int):  func_abs_typed(child, out, int)
	case typeid_of(uint): func_abs_typed(child, out, uint)
	case typeid_of(f32):  func_abs_typed(child, out, f32)
	case typeid_of(f64):  func_abs_typed(child, out, f64)
	}
}

// func_abs_typed fills out[i] = abs(in[i]) preserving NULLs.
@(private)
func_abs_typed :: proc(child, out: ^Column, $T: typeid) {
	iv := column_typed_view(child, T)
	ov := column_typed_view(out, T)
	if child.valid == nil {
		when T == f64 { parallel_simd_abs_f64(iv, ov) }
		else { simd_abs(iv, ov) }
		return
	}
	for i in 0 ..< len(iv) {
		if row_valid(child.valid, i) {
			ov[i] = math.abs(iv[i])
		}
	}
}

@(private)
func_sign :: proc(t: typeid, child, out: ^Column) {
	switch t {
	case typeid_of(i8):   func_sign_typed(child, out, i8)
	case typeid_of(i16):  func_sign_typed(child, out, i16)
	case typeid_of(i32):  func_sign_typed(child, out, i32)
	case typeid_of(i64):  func_sign_typed(child, out, i64)
	case typeid_of(int):  func_sign_typed(child, out, int)
	case typeid_of(f32):  func_sign_typed(child, out, f32)
	case typeid_of(f64):  func_sign_typed(child, out, f64)
	case typeid_of(u8), typeid_of(u16), typeid_of(u32), typeid_of(u64), typeid_of(uint):
		// sign of an unsigned value is always 0 or 1.
		func_sign_unsigned(child, out, t)
	}
}

@(private)
func_sign_typed :: proc(child, out: ^Column, $T: typeid) {
	iv := column_typed_view(child, T)
	ov := column_typed_view(out, T)
	when T == f32 || T == f64 {
		if child.valid == nil {
			for i in 0 ..< len(iv) {
				ov[i] = math.sign(iv[i])
			}
			return
		}
		for i in 0 ..< len(iv) {
			if row_valid(child.valid, i) {
				ov[i] = math.sign(iv[i])
			}
		}
	} else {
		if child.valid == nil {
			for i in 0 ..< len(iv) {
				ov[i] = T((iv[i] > 0 ? 1 : 0) - (iv[i] < 0 ? 1 : 0))
			}
			return
		}
		for i in 0 ..< len(iv) {
			if row_valid(child.valid, i) {
				ov[i] = T((iv[i] > 0 ? 1 : 0) - (iv[i] < 0 ? 1 : 0))
			}
		}
	}
}

@(private)
func_sign_unsigned :: proc(child, out: ^Column, t: typeid) {
	switch t {
	case typeid_of(u8):   func_sign_unsigned_typed(child, out, u8)
	case typeid_of(u16):  func_sign_unsigned_typed(child, out, u16)
	case typeid_of(u32):  func_sign_unsigned_typed(child, out, u32)
	case typeid_of(u64):  func_sign_unsigned_typed(child, out, u64)
	case typeid_of(uint): func_sign_unsigned_typed(child, out, uint)
	}
}

@(private)
func_sign_unsigned_typed :: proc(child, out: ^Column, $T: typeid) {
	iv := column_typed_view(child, T)
	ov := column_typed_view(out, T)
	for i in 0 ..< len(iv) {
		if row_valid(child.valid, i) {
			ov[i] = T(iv[i] > 0 ? 1 : 0)
		}
	}
}

// func_round rounds float columns to `decimals` places (negative decimals round
// to tens, hundreds, ...). Integer columns round to themselves.
@(private)
func_round :: proc(t: typeid, child, out: ^Column, decimals: i32) {
	switch t {
	case typeid_of(f32):
		factor := f32(math.pow(10.0, f64(decimals)))
		func_round_typed(child, out, f32, factor)
	case typeid_of(f64):
		factor := math.pow(10.0, f64(decimals))
		func_round_typed(child, out, f64, factor)
	case:
		// ints already round to themselves; copy values.
		func_identity_typed(child, out, t)
	}
}

@(private)
func_round_typed :: proc(child, out: ^Column, $T: typeid, factor: T) {
	iv := column_typed_view(child, T)
	ov := column_typed_view(out, T)
	if child.valid == nil {
		for i in 0 ..< len(iv) {
			ov[i] = math.round(iv[i] * factor) / factor
		}
		return
	}
	for i in 0 ..< len(iv) {
		if row_valid(child.valid, i) {
			ov[i] = math.round(iv[i] * factor) / factor
		}
	}
}

// func_identity_typed copies values of a column of runtime type t.
@(private)
func_identity_typed :: proc(child, out: ^Column, t: typeid) {
	switch t {
	case typeid_of(i8):   copy_typed(child, out, i8)
	case typeid_of(i16):  copy_typed(child, out, i16)
	case typeid_of(i32):  copy_typed(child, out, i32)
	case typeid_of(i64):  copy_typed(child, out, i64)
	case typeid_of(u8):   copy_typed(child, out, u8)
	case typeid_of(u16):  copy_typed(child, out, u16)
	case typeid_of(u32):  copy_typed(child, out, u32)
	case typeid_of(u64):  copy_typed(child, out, u64)
	case typeid_of(int):  copy_typed(child, out, int)
	case typeid_of(uint): copy_typed(child, out, uint)
	case typeid_of(f32):  copy_typed(child, out, f32)
	case typeid_of(f64):  copy_typed(child, out, f64)
	}
}

@(private)
copy_typed :: proc(child, out: ^Column, $T: typeid) {
	iv := column_typed_view(child, T)
	ov := column_typed_view(out, T)
	copy_slice(ov, iv)
}

// func_diff fills out[i] = in[i] - in[i-n]; the first n rows are NULL.
@(private)
func_diff :: proc(t: typeid, child, out: ^Column, n: int) {
	switch t {
	case typeid_of(i8):   func_diff_typed(child, out, i8, n)
	case typeid_of(i16):  func_diff_typed(child, out, i16, n)
	case typeid_of(i32):  func_diff_typed(child, out, i32, n)
	case typeid_of(i64):  func_diff_typed(child, out, i64, n)
	case typeid_of(u8):   func_diff_typed(child, out, u8, n)
	case typeid_of(u16):  func_diff_typed(child, out, u16, n)
	case typeid_of(u32):  func_diff_typed(child, out, u32, n)
	case typeid_of(u64):  func_diff_typed(child, out, u64, n)
	case typeid_of(int):  func_diff_typed(child, out, int, n)
	case typeid_of(uint): func_diff_typed(child, out, uint, n)
	case typeid_of(f32):  func_diff_typed(child, out, f32, n)
	case typeid_of(f64):  func_diff_typed(child, out, f64, n)
	}
}

@(private)
func_diff_typed :: proc(child, out: ^Column, $T: typeid, n: int) {
	iv := column_typed_view(child, T)
	ov := column_typed_view(out, T)
	lag := max(n, 1)
	for i in 0 ..< len(iv) {
		if i < lag || !row_valid(child.valid, i) || !row_valid(child.valid, i - lag) {
			column_set_valid(&out^, i, false)
			continue
		}
		ov[i] = iv[i] - iv[i - lag]
	}
}

// func_pct_change fills out (f64) with (in[i] - in[i-n]) / in[i-n]; the first n
// rows are NULL, as are rows whose lag value is 0.
@(private)
func_pct_change :: proc(t: typeid, child, out: ^Column, n: int) {
	switch t {
	case typeid_of(i8):   func_pct_change_typed(child, out, i8, n)
	case typeid_of(i16):  func_pct_change_typed(child, out, i16, n)
	case typeid_of(i32):  func_pct_change_typed(child, out, i32, n)
	case typeid_of(i64):  func_pct_change_typed(child, out, i64, n)
	case typeid_of(u8):   func_pct_change_typed(child, out, u8, n)
	case typeid_of(u16):  func_pct_change_typed(child, out, u16, n)
	case typeid_of(u32):  func_pct_change_typed(child, out, u32, n)
	case typeid_of(u64):  func_pct_change_typed(child, out, u64, n)
	case typeid_of(int):  func_pct_change_typed(child, out, int, n)
	case typeid_of(uint): func_pct_change_typed(child, out, uint, n)
	case typeid_of(f32):  func_pct_change_typed(child, out, f32, n)
	case typeid_of(f64):  func_pct_change_typed(child, out, f64, n)
	}
}

@(private)
func_pct_change_typed :: proc(child, out: ^Column, $T: typeid, n: int) {
	iv := column_typed_view(child, T)
	ov := column_typed_view(out, f64)
	lag := max(n, 1)
	for i in 0 ..< len(iv) {
		if i < lag || !row_valid(child.valid, i) || !row_valid(child.valid, i - lag) || f64(iv[i - lag]) == 0 {
			column_set_valid(&out^, i, false)
			continue
		}
		ov[i] = (f64(iv[i]) - f64(iv[i - lag])) / f64(iv[i - lag])
	}
}

// func_cum_sum fills out with the running sum of in (numeric only).
@(private)
func_cum_sum :: proc(t: typeid, child, out: ^Column) {
	switch t {
	case typeid_of(i8):   func_cum_sum_typed(child, out, i8)
	case typeid_of(i16):  func_cum_sum_typed(child, out, i16)
	case typeid_of(i32):  func_cum_sum_typed(child, out, i32)
	case typeid_of(i64):  func_cum_sum_typed(child, out, i64)
	case typeid_of(u8):   func_cum_sum_typed(child, out, u8)
	case typeid_of(u16):  func_cum_sum_typed(child, out, u16)
	case typeid_of(u32):  func_cum_sum_typed(child, out, u32)
	case typeid_of(u64):  func_cum_sum_typed(child, out, u64)
	case typeid_of(int):  func_cum_sum_typed(child, out, int)
	case typeid_of(uint): func_cum_sum_typed(child, out, uint)
	case typeid_of(f32):  func_cum_sum_typed(child, out, f32)
	case typeid_of(f64):  func_cum_sum_typed(child, out, f64)
	}
}

@(private)
func_cum_sum_typed :: proc(child, out: ^Column, $T: typeid) {
	iv := column_typed_view(child, T)
	ov := column_typed_view(out, T)
	acc: T
	for i in 0 ..< len(iv) {
		if row_valid(child.valid, i) {
			acc += iv[i]
			ov[i] = acc
		}
	}
}

// is_between_eval implements Is_Between: child rows between two constant
// scalar bounds (inclusive), NULL where the child is NULL.
@(private)
is_between_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Is_Between) -> (Column, Error) {
	child, cerr := eval_child(allocator, df, n.expr)
	if cerr != .None {
		return {}, cerr
	}
	defer column_destroy(&child)
	if !is_numeric_type(child.dtype) {
		return {}, .Unsupported_Operation
	}

	lo, lerr := eval_child(allocator, df, n.lower)
	if lerr != .None {
		return {}, lerr
	}
	defer column_destroy(&lo)
	hi, herr := eval_child(allocator, df, n.upper)
	if herr != .None {
		return {}, herr
	}
	defer column_destroy(&hi)
	if is_numeric_type(lo.dtype) && lo.dtype != child.dtype {
		coerced, lo_err := cast_eval(allocator, &lo, child.dtype)
		if lo_err != .None {
			return {}, lo_err
		}
		column_destroy(&lo)
		lo = coerced
	}
	if is_numeric_type(hi.dtype) && hi.dtype != child.dtype {
		coerced, hi_err := cast_eval(allocator, &hi, child.dtype)
		if hi_err != .None {
			return {}, hi_err
		}
		column_destroy(&hi)
		hi = coerced
	}
	if lo.dtype != child.dtype || hi.dtype != child.dtype {
		return {}, .Type_Mismatch
	}

	out, err := column_alloc(allocator, child.name, bool, size_of(bool), align_of(bool), child.count)
	if err != .None {
		return {}, err
	}
	column_copy_validity(&out, &child)

	switch child.dtype {
	case typeid_of(i8):   is_between_typed(&child, &lo, &hi, &out, i8)
	case typeid_of(i16):  is_between_typed(&child, &lo, &hi, &out, i16)
	case typeid_of(i32):  is_between_typed(&child, &lo, &hi, &out, i32)
	case typeid_of(i64):  is_between_typed(&child, &lo, &hi, &out, i64)
	case typeid_of(u8):   is_between_typed(&child, &lo, &hi, &out, u8)
	case typeid_of(u16):  is_between_typed(&child, &lo, &hi, &out, u16)
	case typeid_of(u32):  is_between_typed(&child, &lo, &hi, &out, u32)
	case typeid_of(u64):  is_between_typed(&child, &lo, &hi, &out, u64)
	case typeid_of(int):  is_between_typed(&child, &lo, &hi, &out, int)
	case typeid_of(uint): is_between_typed(&child, &lo, &hi, &out, uint)
	case typeid_of(f32):  is_between_typed(&child, &lo, &hi, &out, f32)
	case typeid_of(f64):  is_between_typed(&child, &lo, &hi, &out, f64)
	case:
		return {}, .Unsupported_Operation
	}
	return out, .None
}

@(private)
is_between_typed :: proc(child, lo, hi, out: ^Column, $T: typeid) {
	iv := column_typed_view(child, T)
	lv := column_typed_view(lo, T)
	hv := column_typed_view(hi, T)
	ov := column_typed_view(out, bool)
	if len(lv) == 0 || len(hv) == 0 {
		return
	}
	for i in 0 ..< len(iv) {
		if row_valid(child.valid, i) {
			ov[i] = lv[0] <= iv[i] && iv[i] <= hv[0]
		}
	}
}

// is_in_eval implements Is_In: membership of each child row in a constant
// list literal of the same dtype.
@(private)
is_in_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Is_In) -> (Column, Error) {
	child, cerr := eval_child(allocator, df, n.expr)
	if cerr != .None {
		return {}, cerr
	}
	defer column_destroy(&child)

	lit, ok := n.values^.(expr.Lit)
	if !ok {
		return {}, .Invalid_Argument
	}
	if lit.dtype != slice_typeid_of(child.dtype) {
		return {}, .Type_Mismatch
	}

	out, err := column_alloc(allocator, child.name, bool, size_of(bool), align_of(bool), child.count)
	if err != .None {
		return {}, err
	}
	column_copy_validity(&out, &child)

	switch child.dtype {
	case typeid_of(i8):   is_in_typed(&child, &lit, &out, i8)
	case typeid_of(i16):  is_in_typed(&child, &lit, &out, i16)
	case typeid_of(i32):  is_in_typed(&child, &lit, &out, i32)
	case typeid_of(i64):  is_in_typed(&child, &lit, &out, i64)
	case typeid_of(u8):   is_in_typed(&child, &lit, &out, u8)
	case typeid_of(u16):  is_in_typed(&child, &lit, &out, u16)
	case typeid_of(u32):  is_in_typed(&child, &lit, &out, u32)
	case typeid_of(u64):  is_in_typed(&child, &lit, &out, u64)
	case typeid_of(int):  is_in_typed(&child, &lit, &out, int)
	case typeid_of(uint): is_in_typed(&child, &lit, &out, uint)
	case typeid_of(f32):  is_in_typed(&child, &lit, &out, f32)
	case typeid_of(f64):  is_in_typed(&child, &lit, &out, f64)
	case typeid_of(string):
		values, vok := expr.lit_as(&lit, []string)
		if !vok {
			column_destroy(&out)
			return {}, .Type_Mismatch
		}
		set := make(map[string]bool)
		for v in values {
			set[v] = true
		}
		is_in_string_typed(&child, &out, set)
	case:
		column_destroy(&out)
		return {}, .Unsupported_Operation
	}
	return out, .None
}

@(private)
is_in_typed :: proc(child: ^Column, lit: ^expr.Lit, out: ^Column, $T: typeid) {
	values, ok := expr.lit_as(lit, []T)
	if !ok {
		return
	}
	set := make(map[T]bool)
	defer delete(set)
	for v in values {
		set[v] = true
	}
	iv := column_typed_view(child, T)
	ov := column_typed_view(out, bool)
	for i in 0 ..< len(iv) {
		if row_valid(child.valid, i) {
			ov[i] = set[iv[i]]
		}
	}
}

@(private)
is_in_string_typed :: proc(child, out: ^Column, set: map[string]bool) {
	defer delete(set)
	iv := column_typed_view(child, string)
	ov := column_typed_view(out, bool)
	for i in 0 ..< len(iv) {
		if row_valid(child.valid, i) {
			ov[i] = set[iv[i]]
		}
	}
}

// arange_eval implements Arange: an int column start ..< end from two
// constant scalar expressions.
@(private)
arange_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Arange) -> (Column, Error) {
	start, serr := eval_child(allocator, df, n.start)
	if serr != .None {
		return {}, serr
	}
	defer column_destroy(&start)
	end, eerr := eval_child(allocator, df, n.end)
	if eerr != .None {
		return {}, eerr
	}
	defer column_destroy(&end)

	sv, svalid, sv_err := column_get(&start, 0, int)
	if sv_err != .None {
		return {}, sv_err
	}
	if !svalid {
		return {}, .Invalid_Argument
	}
	start_v := sv
	ev, evalid, ev_err := column_get(&end, 0, int)
	if ev_err != .None {
		return {}, ev_err
	}
	if !evalid {
		return {}, .Invalid_Argument
	}
	count := max(ev - start_v, 0)

	out, err := column_alloc(allocator, "", int, size_of(int), align_of(int), count)
	if err != .None {
		return {}, err
	}
	ov := column_typed_view(&out, int)
	for i in 0 ..< count {
		ov[i] = start_v + i
	}
	return out, .None
}

// arg_where_eval implements Arg_Where: the row indices where expr is true.
// The result is an int column (length == number of matches).
@(private)
arg_where_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Arg_Where) -> (Column, Error) {
	child, cerr := eval_child(allocator, df, n.expr)
	if cerr != .None {
		return {}, cerr
	}
	defer column_destroy(&child)
	if child.dtype != bool {
		return {}, .Type_Mismatch
	}

	bv := column_typed_view(&child, bool)
	count := 0
	for i in 0 ..< len(bv) {
		if row_valid(child.valid, i) && bv[i] {
			count += 1
		}
	}

	out, err := column_alloc(allocator, "", int, size_of(int), align_of(int), count)
	if err != .None {
		return {}, err
	}
	ov := column_typed_view(&out, int)
	at := 0
	for i in 0 ..< len(bv) {
		if row_valid(child.valid, i) && bv[i] {
			ov[at] = i
			at += 1
		}
	}
	return out, .None
}

// distinct_eval implements Distinct (first/last occurrence markers). NULL rows
// are marked (a NULL is considered a distinct value), matching polars.
@(private)
distinct_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Distinct) -> (Column, Error) {
	child, cerr := eval_child(allocator, df, n.expr)
	if cerr != .None {
		return {}, cerr
	}
	defer column_destroy(&child)

	out, err := column_alloc(allocator, child.name, bool, size_of(bool), align_of(bool), child.count)
	if err != .None {
		return {}, err
	}
	// The marker column is always fully valid: every row is either marked or not.
	column_clear_validity(&out, true)

	switch child.dtype {
	case typeid_of(i8):   distinct_typed(&child, &out, i8, n.kind == .First_Distinct)
	case typeid_of(i16):  distinct_typed(&child, &out, i16, n.kind == .First_Distinct)
	case typeid_of(i32):  distinct_typed(&child, &out, i32, n.kind == .First_Distinct)
	case typeid_of(i64):  distinct_typed(&child, &out, i64, n.kind == .First_Distinct)
	case typeid_of(u8):   distinct_typed(&child, &out, u8, n.kind == .First_Distinct)
	case typeid_of(u16):  distinct_typed(&child, &out, u16, n.kind == .First_Distinct)
	case typeid_of(u32):  distinct_typed(&child, &out, u32, n.kind == .First_Distinct)
	case typeid_of(u64):  distinct_typed(&child, &out, u64, n.kind == .First_Distinct)
	case typeid_of(int):  distinct_typed(&child, &out, int, n.kind == .First_Distinct)
	case typeid_of(uint): distinct_typed(&child, &out, uint, n.kind == .First_Distinct)
	case typeid_of(f32):  distinct_typed(&child, &out, f32, n.kind == .First_Distinct)
	case typeid_of(f64):  distinct_typed(&child, &out, f64, n.kind == .First_Distinct)
	case typeid_of(bool): distinct_typed(&child, &out, bool, n.kind == .First_Distinct)
	case typeid_of(string): distinct_typed(&child, &out, string, n.kind == .First_Distinct)
	case:
		column_destroy(&out)
		return {}, .Unsupported_Operation
	}
	return out, .None
}

@(private)
distinct_typed :: proc(child, out: ^Column, $T: typeid, first: bool) {
	iv := column_typed_view(child, T)
	ov := column_typed_view(out, bool)
	seen := make(map[T]bool)
	defer delete(seen)
	seen_null := false
	if first {
		for i in 0 ..< len(iv) {
			if !row_valid(child.valid, i) {
				if !seen_null {
					ov[i] = true
					seen_null = true
				}
				continue
			}
			if !seen[iv[i]] {
				ov[i] = true
				seen[iv[i]] = true
			}
		}
	} else {
		for i := len(iv) - 1; i >= 0; i -= 1 {
			if !row_valid(child.valid, i) {
				if !seen_null {
					ov[i] = true
					seen_null = true
				}
				continue
			}
			if !seen[iv[i]] {
				ov[i] = true
				seen[iv[i]] = true
			}
		}
	}
}

// dot_product_eval implements Dot_Product: sum of lhs[i] * rhs[i] over two
// equal-length numeric columns, as a single-row f64 column. NULL rows are
// skipped (treated as 0).
@(private)
dot_product_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Dot_Product) -> (Column, Error) {
	lhs, lerr := eval_child(allocator, df, n.lhs)
	if lerr != .None {
		return {}, lerr
	}
	defer column_destroy(&lhs)
	rhs, rerr := eval_child(allocator, df, n.rhs)
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
	dot_product_dispatch(&lhs, &rhs, &out, lhs.dtype, rhs.dtype)
	return out, .None
}

@(private)
dot_product_dispatch :: proc(lhs, rhs, out: ^Column, t, u: typeid) {
	switch t {
	case typeid_of(i8):   dot_product_rhs(lhs, rhs, out, i8, u)
	case typeid_of(i16):  dot_product_rhs(lhs, rhs, out, i16, u)
	case typeid_of(i32):  dot_product_rhs(lhs, rhs, out, i32, u)
	case typeid_of(i64):  dot_product_rhs(lhs, rhs, out, i64, u)
	case typeid_of(u8):   dot_product_rhs(lhs, rhs, out, u8, u)
	case typeid_of(u16):  dot_product_rhs(lhs, rhs, out, u16, u)
	case typeid_of(u32):  dot_product_rhs(lhs, rhs, out, u32, u)
	case typeid_of(u64):  dot_product_rhs(lhs, rhs, out, u64, u)
	case typeid_of(int):  dot_product_rhs(lhs, rhs, out, int, u)
	case typeid_of(uint): dot_product_rhs(lhs, rhs, out, uint, u)
	case typeid_of(f32):  dot_product_rhs(lhs, rhs, out, f32, u)
	case typeid_of(f64):  dot_product_rhs(lhs, rhs, out, f64, u)
	}
}

@(private)
dot_product_rhs :: proc(lhs, rhs, out: ^Column, $T: typeid, u: typeid) {
	switch u {
	case typeid_of(i8):   dot_product_typed(lhs, rhs, out, T, i8)
	case typeid_of(i16):  dot_product_typed(lhs, rhs, out, T, i16)
	case typeid_of(i32):  dot_product_typed(lhs, rhs, out, T, i32)
	case typeid_of(i64):  dot_product_typed(lhs, rhs, out, T, i64)
	case typeid_of(u8):   dot_product_typed(lhs, rhs, out, T, u8)
	case typeid_of(u16):  dot_product_typed(lhs, rhs, out, T, u16)
	case typeid_of(u32):  dot_product_typed(lhs, rhs, out, T, u32)
	case typeid_of(u64):  dot_product_typed(lhs, rhs, out, T, u64)
	case typeid_of(int):  dot_product_typed(lhs, rhs, out, T, int)
	case typeid_of(uint): dot_product_typed(lhs, rhs, out, T, uint)
	case typeid_of(f32):  dot_product_typed(lhs, rhs, out, T, f32)
	case typeid_of(f64):  dot_product_typed(lhs, rhs, out, T, f64)
	}
}

@(private)
dot_product_typed :: proc(lhs, rhs, out: ^Column, $T: typeid, $U: typeid) {
	lv := column_typed_view(lhs, T)
	rv := column_typed_view(rhs, U)
	ov := column_typed_view(out, f64)
	acc: f64
	for i in 0 ..< len(lv) {
		if row_valid(lhs.valid, i) && row_valid(rhs.valid, i) {
			acc += f64(lv[i]) * f64(rv[i])
		}
	}
	ov[0] = acc
}

// concat_str_eval implements Concat_Str: row-wise string join of the parts
// with a separator. NULL where any part row is NULL.
@(private)
concat_str_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Concat_Str) -> (Column, Error) {
	if len(n.exprs) == 0 {
		return {}, .Invalid_Argument
	}
	cols := make([dynamic]Column, 0, len(n.exprs), allocator)
	defer {
		for &c in cols {
			column_destroy(&c)
		}
		delete(cols)
	}
	count := -1
	for e in n.exprs {
		c, cerr := eval_child(allocator, df, e)
		if cerr != .None {
			return {}, cerr
		}
		if c.dtype != typeid_of(string) {
			column_destroy(&c)
			return {}, .Type_Mismatch
		}
		if count == -1 {
			count = c.count
		} else if c.count != count {
			column_destroy(&c)
			return {}, .Length_Mismatch
		}
		append(&cols, c)
	}

	out, err := column_alloc(allocator, "", string, size_of(string), align_of(string), count)
	if err != .None {
		return {}, err
	}
	for i in 0 ..< count {
		all_valid := true
		for &c in cols {
			if !row_valid(c.valid, i) {
				all_valid = false
				break
			}
		}
		if !all_valid {
			column_set_valid(&out, i, false)
		}
	}

	ov := column_typed_view(&out, string)
	total := 0
	lengths := make([]int, count, allocator)
	defer delete(lengths)
	for i in 0 ..< count {
		if !row_valid(out.valid, i) {
			continue
		}
		sz := 0
		for j in 0 ..< len(cols) {
			if j > 0 {
				sz += len(n.separator)
			}
			sz += len(column_typed_view(&cols[j], string)[i])
		}
		lengths[i] = sz
		total += sz
	}
	if total > 0 {
		blob, a_err := mem.alloc(total, 1, allocator)
		if a_err != .None || blob == nil {
			column_destroy(&out)
			return {}, .Allocator_Failure
		}
		out.payload = blob
		out.payload_size = total
		cursor := uintptr(blob)
		// delegate the actual join to core:strings; copy the result into the
		// owned blob and release the per-row allocation.
		scratch := make([]string, len(cols), allocator)
		defer delete(scratch)
		for i in 0 ..< count {
			if !row_valid(out.valid, i) {
				continue
			}
			start := cursor
			for j in 0 ..< len(cols) {
				scratch[j] = column_typed_view(&cols[j], string)[i]
			}
			joined, jerr := strings.join(scratch, n.separator, allocator)
			if jerr != .None {
				column_destroy(&out)
				return {}, .Allocator_Failure
			}
			if len(joined) > 0 {
				mem.copy(rawptr(cursor), raw_data(transmute([]byte)joined), len(joined))
				cursor += uintptr(len(joined))
			}
			delete(joined)
			ov[i] = transmute(string)runtime.Raw_String{data = (^u8)(start), len = lengths[i]}
		}
	}
	return out, .None
}

// search_sorted_eval implements Search_Sorted: binary-search each value of
// `values` in the sorted column `sorted`, returning insertion indices.
@(private)
search_sorted_eval :: proc(allocator: mem.Allocator, df: ^DataFrame, n: expr.Search_Sorted) -> (Column, Error) {
	sorted, serr := eval_child(allocator, df, n.sorted)
	if serr != .None {
		return {}, serr
	}
	defer column_destroy(&sorted)
	values, verr := eval_child(allocator, df, n.values)
	if verr != .None {
		return {}, verr
	}
	defer column_destroy(&values)
	if sorted.dtype != values.dtype {
		return {}, .Type_Mismatch
	}
	// the sorted column must be NULL-free to be comparable.
	if sorted.valid != nil {
		return {}, .Null_Value
	}

	out, err := column_alloc(allocator, "", int, size_of(int), align_of(int), values.count)
	if err != .None {
		return {}, err
	}
	column_copy_validity(&out, &values)

	switch sorted.dtype {
	case typeid_of(i8):   search_sorted_typed(&sorted, &values, &out, i8)
	case typeid_of(i16):  search_sorted_typed(&sorted, &values, &out, i16)
	case typeid_of(i32):  search_sorted_typed(&sorted, &values, &out, i32)
	case typeid_of(i64):  search_sorted_typed(&sorted, &values, &out, i64)
	case typeid_of(u8):   search_sorted_typed(&sorted, &values, &out, u8)
	case typeid_of(u16):  search_sorted_typed(&sorted, &values, &out, u16)
	case typeid_of(u32):  search_sorted_typed(&sorted, &values, &out, u32)
	case typeid_of(u64):  search_sorted_typed(&sorted, &values, &out, u64)
	case typeid_of(int):  search_sorted_typed(&sorted, &values, &out, int)
	case typeid_of(uint): search_sorted_typed(&sorted, &values, &out, uint)
	case typeid_of(f32):  search_sorted_typed(&sorted, &values, &out, f32)
	case typeid_of(f64):  search_sorted_typed(&sorted, &values, &out, f64)
	case typeid_of(string): search_sorted_typed(&sorted, &values, &out, string)
	case:
		column_destroy(&out)
		return {}, .Unsupported_Operation
	}
	return out, .None
}

// search_sorted_typed returns the insertion index (lower bound) of each value
// in a sorted, NULL-free column.
@(private)
search_sorted_typed :: proc(sorted, values, out: ^Column, $T: typeid) {
	sv := column_typed_view(sorted, T)
	vv := column_typed_view(values, T)
	ov := column_typed_view(out, int)
	for i in 0 ..< len(vv) {
		if !row_valid(values.valid, i) {
			continue
		}
		lo := 0
		hi := len(sv)
		for lo < hi {
			mid := (lo + hi) / 2
			if sv[mid] < vv[i] {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		ov[i] = lo
	}
}
