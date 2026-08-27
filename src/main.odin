package main

import "dataframe"
import "dataframe/expr"
import "core:fmt"

// Demo of the blessed eager API (Stage 10): column construction, expression
// predicates, short-name pipelines (S10.2), group_by/agg, the schema-shaped
// constructor (S10.4), dataframe_print (S10.3), and temporal ops (S14.8).
main :: proc() {
	if err := run(); err != .None {
		fmt.eprintln("demo failed:", err)
	}
}

run :: proc() -> dataframe.Error {
	// 1. Build a DataFrame from columns.
	age, a_err := dataframe.column_from("age", []i32{25, 30, 35, 40})
	if a_err != .None {
		return a_err
	}
	name, n_err := dataframe.column_from("name", []string{"ada", "grace", "katherine", "margaret"})
	if n_err != .None {
		return n_err
	}
	dept, d_err := dataframe.column_from("dept", []string{"eng", "sales", "eng", "eng"})
	if d_err != .None {
		return d_err
	}
	salary, s_err := dataframe.column_from_with_valid(
		"salary",
		[]f64{150000, 90000, 0.0, 210000},
		[]bool{true, true, false, true},
	)
	if s_err != .None {
		return s_err
	}

	df, df_err := dataframe.dataframe_from_columns([]^dataframe.Column{&age, &name, &dept, &salary})
	if df_err != .None {
		return df_err
	}
	defer dataframe.dataframe_destroy(&df)

	fmt.println("== dataframe_print ==")
	dataframe.dataframe_print(&df)

	// 2. Expression predicates.
	ctx := expr.context_create(context.allocator)
	defer expr.context_destroy(&ctx)

	// 3. Pipelines: short aliases compose with or_return.
	fmt.println()
	fmt.println("== pipeline: filter(age >= 30) -> sort(salary desc) -> head(2) ==")
	filtered := dataframe.filter(&df, expr.ge(&ctx, expr.col(&ctx, "age"), expr.lit(&ctx, i32(30)))) or_return
	defer dataframe.dataframe_destroy(&filtered)

	sorted := dataframe.sort(&filtered, []dataframe.Sort_Key{dataframe.sort_key("salary", .Desc)}) or_return
	defer dataframe.dataframe_destroy(&sorted)

	top := dataframe.head(&sorted, 2) or_return
	defer dataframe.dataframe_destroy(&top)
	dataframe.dataframe_print(&top)

	// 4. group_by + agg.
	fmt.println()
	fmt.println("== group_by(dept) -> agg(mean salary, count) ==")
	gb := dataframe.group_by(&df, []^expr.Expr{expr.col(&ctx, "dept")}) or_return
	defer dataframe.dataframe_group_by_destroy(&gb)

	grouped := dataframe.agg(&gb, []^expr.Expr{
		expr.alias(&ctx, expr.mean_(&ctx, expr.col(&ctx, "salary")), "mean_salary"),
		expr.alias(&ctx, expr.count_(&ctx, expr.col(&ctx, "salary")), "n"),
	}) or_return
	defer dataframe.dataframe_destroy(&grouped)
	dataframe.dataframe_print(&grouped)

	// 5. Schema-shaped constructor: start empty, fill later.
	fmt.println()
	fmt.println("== create_with_schema (empty, same shape) ==")
	schema, schema_err := dataframe.dataframe_schema(&df)
	if schema_err != .None {
		return schema_err
	}
	defer dataframe.schema_destroy(&schema)
	empty, empty_err := dataframe.dataframe_create_with_schema(schema)
	if empty_err != .None {
		return empty_err
	}
	defer dataframe.dataframe_destroy(&empty)
	dataframe.dataframe_print(&empty)

	// 6. Temporal (S14.8): date_range, dt_* accessors, truncate.
	fmt.println()
	fmt.println("== temporal: date_range + truncate to hours ==")
	start_dt, _ := dataframe.datetime_create(2024, 1, 1, 0, 0, 0)
	end_dt, _ := dataframe.datetime_create(2024, 1, 3, 0, 0, 0)
	rng, r_err := dataframe.date_range(start_dt, end_dt, dataframe.duration_from_hours(6))
	if r_err != .None {
		return r_err
	}
	defer dataframe.column_destroy(&rng)
	trunc, tr_err := dataframe.truncate(&rng, dataframe.duration_from_days(1))
	if tr_err != .None {
		return tr_err
	}
	defer dataframe.column_destroy(&trunc)
	day, day_err := dataframe.dt_day(&rng)
	if day_err != .None {
		return day_err
	}
	defer dataframe.column_destroy(&day)
	if err := dataframe.column_rename(&rng, "ts"); err != .None {
		return err
	}
	if err := dataframe.column_rename(&trunc, "day_start"); err != .None {
		return err
	}
	if err := dataframe.column_rename(&day, "day_of_month"); err != .None {
		return err
	}
	rdf, rdf_err := dataframe.dataframe_from_columns([]^dataframe.Column{&rng, &trunc, &day})
	if rdf_err != .None {
		return rdf_err
	}
	defer dataframe.dataframe_destroy(&rdf)
	dataframe.dataframe_print(&rdf)

	return .None
}
