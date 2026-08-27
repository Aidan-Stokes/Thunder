package dataframe

// Chain-friendly short names (S10.2).
//
// Odin has no method-call sugar for package procs: `->` only calls proc-typed
// *fields* on a struct, and this Odin version has no `|>` pipe operator. The
// idiomatic way to compose steps is therefore short proc names that take the
// receiver first and return a new owned DataFrame, composed with qualified
// calls and Odin's error operators:
//
//	out, err := dataframe.sort(
//		dataframe.filter(&df, pred),
//		[]dataframe.Sort_Key{dataframe.sort_key("x")},
//	)
//	if err != .None { return err }
//	res := dataframe.head(&out, 5) or_return
//
// These aliases bind the short names so the blessed spelling is
// `dataframe.filter` (matching the DESIGN.md §15 sketch) instead of
// `dataframe.dataframe_filter`. Every alias is a plain proc alias: the
// dataframe_* procs remain the underlying implementations, ownership
// semantics are unchanged, and each call returns a new DataFrame the caller
// owns.

// filter chains dataframe_filter.
filter :: dataframe_filter

// select chains dataframe_select.
select :: dataframe_select

// sort chains dataframe_sort (keyed by []Sort_Key; see sort_key).
sort :: dataframe_sort

// sort_by chains dataframe_sort_by (keyed by expressions).
sort_by :: dataframe_sort_by

// head chains dataframe_head.
head :: dataframe_head

// tail chains dataframe_tail.
tail :: dataframe_tail

// slice chains dataframe_slice.
slice :: dataframe_slice

// take chains dataframe_take.
take :: dataframe_take

// limit chains dataframe_limit.
limit :: dataframe_limit

// with_columns chains dataframe_with_columns.
with_columns :: dataframe_with_columns

// drop chains dataframe_drop.
drop :: dataframe_drop

// unique chains dataframe_unique.
unique :: dataframe_unique

// group_by chains dataframe_group_by; pipe the result into agg.
group_by :: dataframe_group_by

// agg chains dataframe_group_by_agg on a Group_By.
agg :: dataframe_group_by_agg

// partition_by chains dataframe_partition_by.
partition_by :: dataframe_partition_by

// explode chains dataframe_explode.
explode :: dataframe_explode

// unnest chains dataframe_unnest.
unnest :: dataframe_unnest
