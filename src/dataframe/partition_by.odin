package dataframe

// Splitting into groups (Stage 4): partition_by.

import "core:mem"
import "expr"

// dataframe_partition_by splits df into one DataFrame per distinct
// combination of the evaluated key expressions (polars partition_by). Each
// partition owns its columns; the returned slice is allocated with allocator
// and must be released with dataframe_partitions_destroy. Partitions appear
// in first-appearance order and each preserves source row order.
//
// NULL keys: NULL is a per-column key value — rows that are NULL in the same
// key columns form a single partition (polars behavior).
dataframe_partition_by :: proc(df: ^DataFrame, exprs: []^expr.Expr, allocator := context.allocator) -> (partitions: []DataFrame, err: Error) {
	if len(exprs) == 0 {
		return nil, .Invalid_Argument
	}

	key_cols := make([]Column, len(exprs), allocator)
	key_ptrs := make([]^Column, len(exprs), allocator)
	defer {
		for &c in key_cols {
			column_destroy(&c)
		}
		delete(key_cols, allocator)
		delete(key_ptrs, allocator)
	}
	oa: OpArena
	op_arena_init(&oa, allocator)
	defer op_arena_destroy(&oa)
	for e, i in exprs {
		c, eval_err := expr_eval(allocator, df, e, &oa)
		if eval_err != .None {
			return nil, eval_err
		}
		key_cols[i] = c
		key_ptrs[i] = &key_cols[i]
	}

	rows := dataframe_num_rows(df)
	// groups maps an owned key string to the row indices of its partition.
	// order lists the keys in first-appearance order. The map and the list
	// both store the string header; the key bytes are owned once, via order.
	groups := make(map[string][dynamic]int, 0, allocator)
	order := make([dynamic]string, allocator)
	defer {
		for k in order {
			delete_string(k, allocator)
		}
		delete(order)
		for _, g in groups {
			delete(g)
		}
		delete(groups)
	}

	buf := make([dynamic]byte, allocator)
	defer delete(buf)
	for row in 0 ..< rows {
		if enc_err := encode_row(key_ptrs, row, &buf); enc_err != .None {
			return nil, enc_err
		}
		key := string(buf[:])
		if _, exists := groups[key]; !exists {
			owned, o_err := clone_name(allocator, key)
			if o_err != .None {
				return nil, o_err
			}
			groups[owned] = nil
			append(&order, owned)
		}
		append(&groups[key], row)
	}

	partitions = make([]DataFrame, len(order), allocator)
	if partitions == nil && len(order) != 0 {
		return nil, .Allocator_Failure
	}
	for key, i in order {
		group := groups[key]
		partition, t_err := take_columns(df, allocator, group[:])
		if t_err != .None {
			for j in 0 ..< i {
				dataframe_destroy(&partitions[j])
			}
			delete(partitions, allocator)
			return nil, t_err
		}
		partitions[i] = partition
	}
	return partitions, .None
}

// dataframe_partitions_destroy releases the partitions returned by
// dataframe_partition_by, including the slice itself.
dataframe_partitions_destroy :: proc(partitions: []DataFrame, allocator: mem.Allocator) {
	for &p in partitions {
		dataframe_destroy(&p)
	}
	delete(partitions, allocator)
}
