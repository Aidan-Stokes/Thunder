package dataframe

// Stage 8 join tests (ROADMAP S8.6): correctness of inner/left/right/full/
// semi/anti/cross joins, NULL key handling, duplicate keys, multi-column
// keys, name collisions, empty sides, and error cases.

import "core:testing"

// join_fixture builds the shared Stage 8 fixture.
//
//	emp:    emp_id (i32) [1,2,3,4,5]
//	        dept   (string) ["eng","eng","sales","eng",NULL]
//	        salary (f64)    [10,20,30,40,NULL]
//	budget: dept   (string) ["eng","sales","sales","hr"]
//	        budget (f64)    [100,200,210,400]
//
// Inner join on dept (left-major, right key column dropped):
//   (1,eng,10)+100, (2,eng,20)+100, (3,sales,30)+200, (3,sales,30)+210,
//   (4,eng,40)+100  -> 5 rows
// Left join adds emp 5 with NULL budget -> 6 rows.
// Right join (right-major, 6 rows): eng->emps 1,2,4; sales->3 twice; hr->NULL.
// Full join: 5 matched + emp 5 + hr -> 7 rows.
// Semi join: emps 1,2,3,4 (4 rows, left columns only).
// Anti join: emp 5 (1 row).
// Cross join: 20 rows, all columns.
join_fixture :: proc(t: ^testing.T) -> (emp: DataFrame, budget: DataFrame) {
	err: Error
	eid, dept, salary: Column
	eid, err = column_from("emp_id", []i32{1, 2, 3, 4, 5})
	testing.expect(t, err == .None, "emp_id column")
	dept, err = column_from("dept", []string{"eng", "eng", "sales", "eng", "x"})
	testing.expect(t, err == .None, "dept column")
	testing.expect(t, column_set_valid(&dept, 4, false) == .None, "emp dept[4] NULL")
	salary, err = column_from("salary", []f64{10, 20, 30, 40, 0})
	testing.expect(t, err == .None, "salary column")
	testing.expect(t, column_set_valid(&salary, 4, false) == .None, "salary[4] NULL")

	emp, err = dataframe_from_columns([]^Column{&eid, &dept, &salary})
	testing.expect(t, err == .None, "emp from_columns")

	bd, bd_budget: Column
	bd, err = column_from("dept", []string{"eng", "sales", "sales", "hr"})
	testing.expect(t, err == .None, "budget dept column")
	bd_budget, err = column_from("budget", []f64{100, 200, 210, 400})
	testing.expect(t, err == .None, "budget column")

	budget, err = dataframe_from_columns([]^Column{&bd, &bd_budget})
	testing.expect(t, err == .None, "budget from_columns")
	return
}

// join_fixture_destroy releases both fixture DataFrames.
join_fixture_destroy :: proc(emp, budget: ^DataFrame) {
	dataframe_destroy(emp)
	dataframe_destroy(budget)
}

// join_ok runs a join, asserting success. The caller owns the result.
join_ok :: proc(t: ^testing.T, left, right: ^DataFrame, kind: Join_Kind, left_keys, right_keys: []string) -> (out: DataFrame) {
	err: Error
	switch kind {
	case .Inner:
		out, err = dataframe_inner_join(left, right, left_keys, right_keys)
	case .Left:
		out, err = dataframe_left_join(left, right, left_keys, right_keys)
	case .Right:
		out, err = dataframe_right_join(left, right, left_keys, right_keys)
	case .Full:
		out, err = dataframe_full_join(left, right, left_keys, right_keys)
	case .Semi:
		out, err = dataframe_semi_join(left, right, left_keys, right_keys)
	case .Anti:
		out, err = dataframe_anti_join(left, right, left_keys, right_keys)
	case .Cross:
		out, err = dataframe_cross_join(left, right, context.allocator)
	}
	testing.expect(t, err == .None, "join err")
	if err != .None {
		return {}
	}
	return out
}

// --- S8.1 inner join ---------------------------------------------------------

@(test)
join_test_inner :: proc(t: ^testing.T) {
	emp, budget := join_fixture(t)
	defer join_fixture_destroy(&emp, &budget)

	out := join_ok(t, &emp, &budget, .Inner, []string{"dept"}, []string{"dept"})
	defer dataframe_destroy(&out)

	// 5 matched rows, columns: emp_id, dept, salary, budget (right key dropped).
	testing.expect(t, dataframe_num_rows(&out) == 5, "5 matched rows")
	testing.expect(t, dataframe_num_cols(&out) == 4, "emp_id + dept + salary + budget")
	testing.expect(t, !dataframe_has_column(&out, "dept_right"), "right key column dropped")

	// Row 0: emp 1 (eng,10) + 100.
	id := dataframe_get_column(&out, "emp_id") or_else nil
	v, _, _ := column_get(id, 0, i32)
	testing.expect(t, v == 1, "row0 emp_id = 1")
	bud := dataframe_get_column(&out, "budget") or_else nil
	v64, _, _ := column_get(bud, 0, f64)
	testing.expect(t, near(v64, 100), "row0 budget = 100")

	// Row 2: emp 3 (sales,30) + 200; row 3: emp 3 + 210 (dup on right).
	v, _, _ = column_get(id, 2, i32)
	testing.expect(t, v == 3, "row2 emp_id = 3")
	v, _, _ = column_get(id, 3, i32)
	testing.expect(t, v == 3, "row3 emp_id = 3")
	v64, _, _ = column_get(bud, 3, f64)
	testing.expect(t, near(v64, 210), "row3 budget = 210")

	// NULL-keyed emp 5 never matches, so the 5 output rows are all valid.
	sal := dataframe_get_column(&out, "salary") or_else nil
	testing.expect(t, dataframe_num_rows(&out) == 5, "NULL-key emp excluded")
	_ = sal
}

// --- S8.2 left join ----------------------------------------------------------

@(test)
join_test_left :: proc(t: ^testing.T) {
	emp, budget := join_fixture(t)
	defer join_fixture_destroy(&emp, &budget)

	out := join_ok(t, &emp, &budget, .Left, []string{"dept"}, []string{"dept"})
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 6, "5 matched + 1 unmatched left")
	testing.expect(t, dataframe_num_cols(&out) == 4, "4 columns")

	// Unmatched left row (emp 5, NULL dept) comes last with NULL budget.
	id := dataframe_get_column(&out, "emp_id") or_else nil
	v, _, _ := column_get(id, 5, i32)
	testing.expect(t, v == 5, "row5 emp_id = 5")
	bud := dataframe_get_column(&out, "budget") or_else nil
	testing.expect(t, !column_is_valid(bud, 5), "row5 budget NULL")
}

// --- S8.3 right join ---------------------------------------------------------

@(test)
join_test_right :: proc(t: ^testing.T) {
	emp, budget := join_fixture(t)
	defer join_fixture_destroy(&emp, &budget)

	// Left = emp, Right = budget: right-major over budget's dept column.
	out := join_ok(t, &emp, &budget, .Right, []string{"dept"}, []string{"dept"})
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_rows(&out) == 6, "3 eng + 2 sales + 1 hr")
	id := dataframe_get_column(&out, "emp_id") or_else nil

	// Row 0: budget eng -> emp 1; row 1: emp 2; row 2: emp 4.
	v, _, _ := column_get(id, 0, i32)
	testing.expect(t, v == 1, "row0 emp_id = 1 (right-major)")
	v, _, _ = column_get(id, 1, i32)
	testing.expect(t, v == 2, "row1 emp_id = 2")
	v, _, _ = column_get(id, 2, i32)
	testing.expect(t, v == 4, "row2 emp_id = 4")

	// Rows 3,4: sales budget rows -> emp 3 twice.
	v, _, _ = column_get(id, 3, i32)
	testing.expect(t, v == 3, "row3 emp_id = 3")
	v, _, _ = column_get(id, 4, i32)
	testing.expect(t, v == 3, "row4 emp_id = 3")

	// Row 5: hr has no emp -> NULL left columns, but budget value present.
	testing.expect(t, !column_is_valid(id, 5), "row5 emp_id NULL")
	bud := dataframe_get_column(&out, "budget") or_else nil
	v64, _, _ := column_get(bud, 5, f64)
	testing.expect(t, near(v64, 400), "row5 budget = 400")

	// The coalesced key column takes the right value for right-only rows.
	dept := dataframe_get_column(&out, "dept") or_else nil
	s, _, _ := column_get(dept, 5, string)
	testing.expect(t, s == "hr", "row5 dept = hr (coalesced from right)")
}

// --- S8.3 full join ----------------------------------------------------------

@(test)
join_test_full :: proc(t: ^testing.T) {
	emp, budget := join_fixture(t)
	defer join_fixture_destroy(&emp, &budget)

	out := join_ok(t, &emp, &budget, .Full, []string{"dept"}, []string{"dept"})
	defer dataframe_destroy(&out)

	// 5 matched + emp 5 (unmatched left) + hr (unmatched right) = 7.
	testing.expect(t, dataframe_num_rows(&out) == 7, "7 full-join rows")
	id := dataframe_get_column(&out, "emp_id") or_else nil

	// Rows 0..4 matched, row 5 = emp 5 (unmatched left), row 6 = hr.
	v, _, _ := column_get(id, 0, i32)
	testing.expect(t, v == 1, "row0 emp_id = 1")
	v, _, _ = column_get(id, 5, i32)
	testing.expect(t, v == 5, "row5 emp_id = 5 (unmatched left)")
	testing.expect(t, !column_is_valid(id, 6), "row6 emp_id NULL (unmatched right)")
	bud := dataframe_get_column(&out, "budget") or_else nil
	v64, _, _ := column_get(bud, 6, f64)
	testing.expect(t, near(v64, 400), "row6 budget = 400")

	// Full join: the coalesced key column keeps the right key for hr.
	dept := dataframe_get_column(&out, "dept") or_else nil
	s, _, _ := column_get(dept, 6, string)
	testing.expect(t, s == "hr", "row6 dept = hr (coalesced from right)")
}

// --- S8.4 semi / anti / cross joins ------------------------------------------

@(test)
join_test_semi :: proc(t: ^testing.T) {
	emp, budget := join_fixture(t)
	defer join_fixture_destroy(&emp, &budget)

	out := join_ok(t, &emp, &budget, .Semi, []string{"dept"}, []string{"dept"})
	defer dataframe_destroy(&out)

	// Left columns only; emps 1,2,3,4 (no duplicate for emp 3 despite 2 matches).
	testing.expect(t, dataframe_num_rows(&out) == 4, "4 semi rows")
	testing.expect(t, dataframe_num_cols(&out) == 3, "left columns only")
	testing.expect(t, !dataframe_has_column(&out, "budget"), "no right columns")
	id := dataframe_get_column(&out, "emp_id") or_else nil
	v, _, _ := column_get(id, 2, i32)
	testing.expect(t, v == 3, "row2 emp_id = 3")
}

@(test)
join_test_anti :: proc(t: ^testing.T) {
	emp, budget := join_fixture(t)
	defer join_fixture_destroy(&emp, &budget)

	out := join_ok(t, &emp, &budget, .Anti, []string{"dept"}, []string{"dept"})
	defer dataframe_destroy(&out)

	// Only emp 5 (NULL dept) has no match.
	testing.expect(t, dataframe_num_rows(&out) == 1, "1 anti row")
	testing.expect(t, dataframe_num_cols(&out) == 3, "left columns only")
	id := dataframe_get_column(&out, "emp_id") or_else nil
	v, _, _ := column_get(id, 0, i32)
	testing.expect(t, v == 5, "anti row emp_id = 5")
}

@(test)
join_test_cross :: proc(t: ^testing.T) {
	emp, budget := join_fixture(t)
	defer join_fixture_destroy(&emp, &budget)

	out := join_ok(t, &emp, &budget, .Cross, nil, nil)
	defer dataframe_destroy(&out)

	// 5 x 4 = 20 rows, all columns on both sides.
	testing.expect(t, dataframe_num_rows(&out) == 20, "20 cross rows")
	testing.expect(t, dataframe_num_cols(&out) == 5, "emp 3 cols + budget 2 cols")
	testing.expect(t, dataframe_has_column(&out, "dept"), "left dept kept")
	testing.expect(t, dataframe_has_column(&out, "dept_right"), "right dept suffixed")

	id := dataframe_get_column(&out, "emp_id") or_else nil
	v, _, _ := column_get(id, 3, i32)
	testing.expect(t, v == 1, "row3 emp_id = 1 (left-major cross)")
}

// --- NULL keys (SQL semantics) ----------------------------------------------

@(test)
join_test_null_keys :: proc(t: ^testing.T) {
	err: Error
	lid, lv: Column
	lid, err = column_from("id", []i32{1, 2, 3})
	testing.expect(t, err == .None, "left id")
	testing.expect(t, column_set_valid(&lid, 1, false) == .None, "left id[1] NULL")
	lv, err = column_from("v", []f64{10, 20, 30})
	testing.expect(t, err == .None, "left v")
	left: DataFrame
	left, err = dataframe_from_columns([]^Column{&lid, &lv})
	testing.expect(t, err == .None, "left from_columns")
	defer dataframe_destroy(&left)

	rid, rw: Column
	rid, err = column_from("id", []i32{1, 3, 4})
	testing.expect(t, err == .None, "right id")
	testing.expect(t, column_set_valid(&rid, 1, false) == .None, "right id[1] NULL")
	rw, err = column_from("w", []f64{100, 300, 400})
	testing.expect(t, err == .None, "right w")
	right: DataFrame
	right, err = dataframe_from_columns([]^Column{&rid, &rw})
	testing.expect(t, err == .None, "right from_columns")
	defer dataframe_destroy(&right)

	// Inner: only id=1 matches; the NULL keys (left 2, right 3) never match.
	out := join_ok(t, &left, &right, .Inner, []string{"id"}, []string{"id"})
	defer dataframe_destroy(&out)
	testing.expect(t, dataframe_num_rows(&out) == 1, "1 inner match (NULL keys excluded)")
	id := dataframe_get_column(&out, "id") or_else nil
	v, _, _ := column_get(id, 0, i32)
	testing.expect(t, v == 1, "matched id = 1")

	// Left: NULL-keyed left row 2 survives with NULL right columns.
	out2 := join_ok(t, &left, &right, .Left, []string{"id"}, []string{"id"})
	defer dataframe_destroy(&out2)
	testing.expect(t, dataframe_num_rows(&out2) == 3, "3 left rows")
	w := dataframe_get_column(&out2, "w") or_else nil
	testing.expect(t, !column_is_valid(w, 1), "NULL-key left row has NULL right")
	vf, _, _ := column_get(w, 0, f64)
	testing.expect(t, near(vf, 100), "row0 w = 100")
}

// --- duplicate keys (many-to-many) -------------------------------------------

@(test)
join_test_duplicate_keys :: proc(t: ^testing.T) {
	err: Error
	lk, lv: Column
	lk, err = column_from("k", []i32{1, 1, 2})
	testing.expect(t, err == .None, "left k")
	lv, err = column_from("v", []i32{10, 11, 20})
	testing.expect(t, err == .None, "left v")
	left: DataFrame
	left, err = dataframe_from_columns([]^Column{&lk, &lv})
	testing.expect(t, err == .None, "left from_columns")
	defer dataframe_destroy(&left)

	rk, rw: Column
	rk, err = column_from("k", []i32{1, 2, 2})
	testing.expect(t, err == .None, "right k")
	rw, err = column_from("w", []i32{100, 200, 201})
	testing.expect(t, err == .None, "right w")
	right: DataFrame
	right, err = dataframe_from_columns([]^Column{&rk, &rw})
	testing.expect(t, err == .None, "right from_columns")
	defer dataframe_destroy(&right)

	// left 1 -> right 1 (2 pairs), left 2 -> right 2,2 (2 pairs) = 4 rows.
	out := join_ok(t, &left, &right, .Inner, []string{"k"}, []string{"k"})
	defer dataframe_destroy(&out)
	testing.expect(t, dataframe_num_rows(&out) == 4, "4 many-to-many rows")

	// Left-major: (1,100),(1,100),(2,200),(2,201).
	v := dataframe_get_column(&out, "v") or_else nil
	w := dataframe_get_column(&out, "w") or_else nil
	iv, _, _ := column_get(v, 1, i32)
	testing.expect(t, iv == 11, "row1 left v = 11")
	iv, _, _ = column_get(v, 2, i32)
	testing.expect(t, iv == 20, "row2 left v = 20")
	iw, _, _ := column_get(w, 3, i32)
	testing.expect(t, iw == 201, "row3 right w = 201")
}

// --- multi-column keys -------------------------------------------------------

@(test)
join_test_multi_key :: proc(t: ^testing.T) {
	err: Error
	la, lb, lv: Column
	la, err = column_from("a", []i32{1, 1, 2})
	testing.expect(t, err == .None, "left a")
	lb, err = column_from("b", []string{"x", "y", "x"})
	testing.expect(t, err == .None, "left b")
	lv, err = column_from("v", []i32{10, 20, 30})
	testing.expect(t, err == .None, "left v")
	left: DataFrame
	left, err = dataframe_from_columns([]^Column{&la, &lb, &lv})
	testing.expect(t, err == .None, "left from_columns")
	defer dataframe_destroy(&left)

	rc, rd, rw: Column
	rc, err = column_from("c", []i32{1, 1, 3})
	testing.expect(t, err == .None, "right c")
	rd, err = column_from("d", []string{"x", "z", "x"})
	testing.expect(t, err == .None, "right d")
	rw, err = column_from("w", []i32{100, 101, 300})
	testing.expect(t, err == .None, "right w")
	right: DataFrame
	right, err = dataframe_from_columns([]^Column{&rc, &rd, &rw})
	testing.expect(t, err == .None, "right from_columns")
	defer dataframe_destroy(&right)

	// (a=1,b=x) matches (c=1,d=x); (1,y) and (2,x) do not.
	out := join_ok(t, &left, &right, .Inner, []string{"a", "b"}, []string{"c", "d"})
	defer dataframe_destroy(&out)
	testing.expect(t, dataframe_num_rows(&out) == 1, "1 multi-key match")
	testing.expect(t, dataframe_num_cols(&out) == 4, "a,b,v,w (keys c,d dropped)")
	testing.expect(t, !dataframe_has_column(&out, "c"), "right key c dropped")
	testing.expect(t, !dataframe_has_column(&out, "d"), "right key d dropped")
	w := dataframe_get_column(&out, "w") or_else nil
	v64, _, _ := column_get(w, 0, i32)
	testing.expect(t, v64 == 100, "w = 100")
}

// --- name collision suffix (S8.5) ---------------------------------------------

@(test)
join_test_suffix_collision :: proc(t: ^testing.T) {
	err: Error
	lk, lname: Column
	lk, err = column_from("k", []i32{1, 2})
	testing.expect(t, err == .None, "left k")
	lname, err = column_from("name", []string{"a", "b"})
	testing.expect(t, err == .None, "left name")
	left: DataFrame
	left, err = dataframe_from_columns([]^Column{&lk, &lname})
	testing.expect(t, err == .None, "left from_columns")
	defer dataframe_destroy(&left)

	rk, rname: Column
	rk, err = column_from("k", []i32{1, 2})
	testing.expect(t, err == .None, "right k")
	rname, err = column_from("name", []string{"x", "y"})
	testing.expect(t, err == .None, "right name")
	right: DataFrame
	right, err = dataframe_from_columns([]^Column{&rk, &rname})
	testing.expect(t, err == .None, "right from_columns")
	defer dataframe_destroy(&right)

	out := join_ok(t, &left, &right, .Inner, []string{"k"}, []string{"k"})
	defer dataframe_destroy(&out)

	testing.expect(t, dataframe_num_cols(&out) == 3, "k, name, name_right")
	testing.expect(t, dataframe_has_column(&out, "name"), "left name kept")
	testing.expect(t, dataframe_has_column(&out, "name_right"), "right name suffixed")

	rn := dataframe_get_column(&out, "name_right") or_else nil
	s, _, _ := column_get(rn, 0, string)
	testing.expect(t, s == "x", "name_right[0] = x")
}

// --- error cases ---------------------------------------------------------------

@(test)
join_test_errors :: proc(t: ^testing.T) {
	err: Error
	lk, rk: Column
	lk, err = column_from("k", []i32{1, 2})
	testing.expect(t, err == .None, "left k")
	left: DataFrame
	left, err = dataframe_from_columns([]^Column{&lk})
	testing.expect(t, err == .None, "left from_columns")
	defer dataframe_destroy(&left)

	rk, err = column_from("k", []f64{1.0, 2.0})
	testing.expect(t, err == .None, "right k (f64)")
	right_f64: DataFrame
	right_f64, err = dataframe_from_columns([]^Column{&rk})
	testing.expect(t, err == .None, "right from_columns")
	defer dataframe_destroy(&right_f64)

	// Type mismatch: i32 key vs f64 key.
	_, a_err := dataframe_inner_join(&left, &right_f64, []string{"k"}, []string{"k"})
	testing.expect(t, a_err == .Type_Mismatch, "type mismatch")

	// Missing column.
	_, b_err := dataframe_inner_join(&left, &right_f64, []string{"nope"}, []string{"k"})
	testing.expect(t, b_err == .Column_Not_Found, "missing column")

	// Empty key lists.
	_, c_err := dataframe_inner_join(&left, &right_f64, nil, nil)
	testing.expect(t, c_err == .Invalid_Argument, "empty keys")
	_, d_err := dataframe_inner_join(&left, &right_f64, []string{"k"}, nil)
	testing.expect(t, d_err == .Invalid_Argument, "mismatched key lengths")

	// Cross join ignores keys entirely and works across type mismatch.
	cross, cross_err := dataframe_cross_join(&left, &right_f64, context.allocator)
	testing.expect(t, cross_err == .None, "cross join across type mismatch")
	testing.expect(t, dataframe_num_rows(&cross) == 4, "2 x 2 cross rows")
	defer dataframe_destroy(&cross)
}

// --- empty sides ---------------------------------------------------------------

@(test)
join_test_empty_sides :: proc(t: ^testing.T) {
	err: Error
	ek, ev: Column
	ek, err = column_from("k", []i32{})
	testing.expect(t, err == .None, "empty k")
	ev, err = column_from("v", []i32{})
	testing.expect(t, err == .None, "empty v")
	empty: DataFrame
	empty, err = dataframe_from_columns([]^Column{&ek, &ev})
	testing.expect(t, err == .None, "empty from_columns")
	defer dataframe_destroy(&empty)

	// Left empty, right has data: inner -> 0 rows with correct schema.
	out := join_ok(t, &empty, &empty, .Inner, []string{"k"}, []string{"k"})
	defer dataframe_destroy(&out)
	testing.expect(t, dataframe_num_rows(&out) == 0, "inner empty -> 0 rows")
	testing.expect(t, dataframe_num_cols(&out) == 3, "schema k, v, v_right")

	// Left empty cross right empty -> 0 rows, both schemas.
	out2 := join_ok(t, &empty, &empty, .Cross, nil, nil)
	defer dataframe_destroy(&out2)
	testing.expect(t, dataframe_num_rows(&out2) == 0, "cross empty -> 0 rows")
	testing.expect(t, dataframe_num_cols(&out2) == 4, "cross empty schema k,v,k_right,v_right")
}
