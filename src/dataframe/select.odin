package dataframe

// Column projections (Stage 4): select, select_by_name, with_columns, drop.
//
// Results are always new, owned DataFrames; the source is borrowed.

import "core:mem"
import "expr"

// dataframe_select evaluates each expression against df and returns a new
// DataFrame holding the results in order. Every expression's result must be
// named: a Col keeps its source name, an Alias supplies one, and any other
// expression must be wrapped in alias or select returns .Invalid_Argument
// (there is no implicit generated name). Duplicate result names are an error.
dataframe_select :: proc(df: ^DataFrame, exprs: []^expr.Expr, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	out = dataframe_create(allocator)
	oa: OpArena
	op_arena_init(&oa, allocator)
	defer op_arena_destroy(&oa)
	for e in exprs {
		col, eval_err := expr_eval(allocator, df, e, &oa)
		if eval_err != .None {
			dataframe_destroy(&out)
			return {}, eval_err
		}
		if col.name == "" {
			column_destroy(&col)
			dataframe_destroy(&out)
			return {}, .Invalid_Argument
		}
		if add_err := dataframe_add_column(&out, &col); add_err != .None {
			column_destroy(&col)
			dataframe_destroy(&out)
			return {}, add_err
		}
	}
	return out, .None
}

// dataframe_select_by_name returns a new DataFrame holding deep copies of the
// named columns in the order given. Unknown names are an error; duplicate
// names are an error.
dataframe_select_by_name :: proc(df: ^DataFrame, names: []string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	out = dataframe_create(allocator)
	for name in names {
		src, get_err := dataframe_get_column(df, name)
		if get_err != .None {
			dataframe_destroy(&out)
			return {}, get_err
		}
		cc, copy_err := column_copy(src, allocator)
		if copy_err != .None {
			dataframe_destroy(&out)
			return {}, copy_err
		}
		if add_err := dataframe_add_column(&out, &cc); add_err != .None {
			column_destroy(&cc)
			dataframe_destroy(&out)
			return {}, add_err
		}
	}
	return out, .None
}

// dataframe_with_columns returns a deep copy of df with the results of exprs
// added. A result whose name matches an existing column replaces it in place
// (order preserved); new names are appended at the end. Every result must be
// named (same rule as select). When two results share a name, the last one
// wins. On error nothing is consumed and df is unchanged.
dataframe_with_columns :: proc(df: ^DataFrame, exprs: []^expr.Expr, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	results := make([dynamic]Column, allocator)
	defer {
		for &c in results {
			column_destroy(&c)
		}
		delete(results)
	}
	oa: OpArena
	op_arena_init(&oa, allocator)
	defer op_arena_destroy(&oa)
	for e in exprs {
		c, eval_err := expr_eval(allocator, df, e, &oa)
		if eval_err != .None {
			return {}, eval_err
		}
		if c.name == "" {
			column_destroy(&c)
			return {}, .Invalid_Argument
		}
		if _, a_err := append(&results, c); a_err != .None {
			column_destroy(&c)
			return {}, .Allocator_Failure
		}
	}

	copy_out, copy_err := dataframe_copy(df, allocator)
	if copy_err != .None {
		return {}, copy_err
	}
	out = copy_out
	for &c in results {
		if i, ok := column_set_get(&out.columns, c.name); ok {
			dataframe_remove_column(&out, c.name) or_return
		}
		if add_err := dataframe_add_column(&out, &c); add_err != .None {
			dataframe_destroy(&out)
			return {}, add_err
		}
		c = {} // ownership transferred into out
	}
	return out, .None
}

// dataframe_drop returns a deep copy of df without the named columns. Unknown
// names are an error (nothing is consumed); duplicate names are tolerated.
dataframe_drop :: proc(df: ^DataFrame, names: []string, allocator := context.allocator) -> (out: DataFrame, err: Error) {
	if len(names) == 0 {
		return dataframe_copy(df, allocator)
	}
	dropped := make(map[string]bool, len(names), allocator)
	defer delete(dropped)
	for name in names {
		if !dataframe_has_column(df, name) {
			return {}, .Column_Not_Found
		}
		dropped[name] = true
	}

	out = dataframe_create(allocator)
	for i in 0 ..< df.columns.count {
		c := column_set_to_column(&df.columns, i)
		if c.name in dropped {
			continue
		}
		cc, copy_err := column_copy(&c, allocator)
		if copy_err != .None {
			dataframe_destroy(&out)
			return {}, copy_err
		}
		if add_err := dataframe_add_column(&out, &cc); add_err != .None {
			column_destroy(&cc)
			dataframe_destroy(&out)
			return {}, add_err
		}
	}
	return out, .None
}
