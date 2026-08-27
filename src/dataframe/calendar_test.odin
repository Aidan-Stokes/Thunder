package dataframe

// Stage 14.8 tests: temporal dtypes, calendar math, dt_* accessors,
// date_range, truncate, sort and print of temporal columns.

import "core:testing"
import "core:fmt"
import "core:strings"

// expect_i64_col checks the values and validity of an i64 column.
expect_i64_col :: proc(t: ^testing.T, what: string, col: ^Column, want: []i64, want_valid: []bool = nil) {
	testing.expect(t, col.dtype == typeid_of(i64), fmt.tprintf("%s dtype", what))
	testing.expect(t, col.count == len(want), fmt.tprintf("%s row count", what))
	if col.count != len(want) {
		return
	}
	v := column_typed_view(col, i64)
	for i in 0 ..< len(want) {
		testing.expect(t, v[i] == want[i], fmt.tprintf("%s row value", what))
	}
	if want_valid != nil {
		for i in 0 ..< len(want) {
			testing.expect(t, column_is_valid(col, i) == want_valid[i], fmt.tprintf("%s row validity", what))
		}
	}
}

@(test)
calendar_civil_roundtrip :: proc(t: ^testing.T) {
	cases := []struct {
		y, m, d: i64
	}{
		{1970, 1, 1},
		{1969, 12, 31},
		{2000, 1, 1},
		{2000, 2, 29},
		{2024, 2, 29},
		{1900, 2, 28}, // 1900 is not a leap year
		{1972, 12, 31},
		{1, 1, 1},
		{1999, 12, 31},
	}
	for c in cases {
		z := days_from_civil(c.y, c.m, c.d)
		yy, mm, dd := civil_from_days(z)
		testing.expect(t, yy == c.y && mm == c.m && dd == c.d, fmt.tprintf("roundtrip %d-%02d-%02d -> %d-%02d-%02d", c.y, c.m, c.d, yy, mm, dd))
	}
	// known epoch day counts
	testing.expect(t, days_from_civil(1970, 1, 1) == 0, "epoch is day 0")
	testing.expect(t, days_from_civil(1969, 12, 31) == -1, "pre-epoch day -1")
	testing.expect(t, days_from_civil(2000, 1, 1) == 10957, "2000-01-01 is day 10957")
}

@(test)
calendar_weekday :: proc(t: ^testing.T) {
	testing.expect(t, weekday_from_days(days_from_civil(1970, 1, 1)) == 4, "1970-01-01 is Thursday")
	testing.expect(t, weekday_from_days(days_from_civil(1969, 12, 31)) == 3, "1969-12-31 is Wednesday")
	testing.expect(t, weekday_from_days(days_from_civil(2000, 1, 1)) == 6, "2000-01-01 is Saturday")
	testing.expect(t, weekday_from_days(days_from_civil(2024, 1, 1)) == 1, "2024-01-01 is Monday")
}

@(test)
calendar_date_create :: proc(t: ^testing.T) {
	d, err := date_create(2024, 2, 29)
	testing.expect(t, err == .None, "leap day valid")
	testing.expect(t, i64(d) == days_from_civil(2024, 2, 29), "leap day value")
	_, err = date_create(2024, 2, 30)
	testing.expect(t, err == .Invalid_Argument, "Feb 30 rejected")
	_, err = date_create(2023, 2, 29)
	testing.expect(t, err == .Invalid_Argument, "non-leap Feb 29 rejected")
	_, err = date_create(2024, 0, 1)
	testing.expect(t, err == .Invalid_Argument, "month 0 rejected")
	_, err = date_create(2024, 13, 1)
	testing.expect(t, err == .Invalid_Argument, "month 13 rejected")
	_, err = date_create(2024, 4, 31)
	testing.expect(t, err == .Invalid_Argument, "April 31 rejected")
}

@(test)
calendar_datetime_fields :: proc(t: ^testing.T) {
	dt, err := datetime_create(2024, 1, 1, 12, 30, 45, 123456)
	testing.expect(t, err == .None, "datetime_create")
	testing.expect(t, datetime_year(dt) == 2024, "year")
	testing.expect(t, datetime_month(dt) == 1, "month")
	testing.expect(t, datetime_day(dt) == 1, "day")
	testing.expect(t, datetime_weekday(dt) == 1, "weekday Monday")
	testing.expect(t, datetime_hour(dt) == 12, "hour")
	testing.expect(t, datetime_minute(dt) == 30, "minute")
	testing.expect(t, datetime_second(dt) == 45, "second")
	testing.expect(t, datetime_microsecond(dt) == 123456, "microsecond")
	// the day before, pre-epoch
	dt2, err2 := datetime_create(1969, 12, 31, 23, 59, 59)
	testing.expect(t, err2 == .None, "pre-epoch datetime_create")
	testing.expect(t, datetime_year(dt2) == 1969 && datetime_day(dt2) == 31, "pre-epoch fields")
	// invalid time fields
	_, err = datetime_create(2024, 1, 1, 24, 0, 0)
	testing.expect(t, err == .Invalid_Argument, "hour 24 rejected")
	_, err = time_create(12, 60, 0)
	testing.expect(t, err == .Invalid_Argument, "minute 60 rejected")
}

@(test)
calendar_conversions :: proc(t: ^testing.T) {
	d, _ := date_create(2024, 3, 10)
	dt, _ := datetime_create(2024, 3, 10, 7, 15, 30)
	midnight := date_to_datetime(d)
	testing.expect(t, datetime_to_date(dt) == d, "datetime_to_date")
	testing.expect(t, datetime_time_of_day(dt) == Time(7 * US_PER_HOUR + 15 * US_PER_MINUTE + 30 * US_PER_SECOND), "time_of_day")
	testing.expect(t, datetime_year(midnight) == 2024 && datetime_hour(midnight) == 0, "midnight conversion")

	added := datetime_add(dt, duration_from_hours(2))
	testing.expect(t, datetime_hour(added) == 9, "add 2h")
	testing.expect(t, datetime_diff(added, dt) == duration_from_hours(2), "diff 2h")
	testing.expect(t, datetime_diff(dt, added) == duration_from_hours(-2), "negative diff")
	testing.expect(t, datetime_second(datetime_add(added, duration_from_seconds(30))) == 0, "add seconds roll over")
}

@(test)
calendar_column_accessors :: proc(t: ^testing.T) {
	dt1, _ := datetime_create(2024, 1, 1, 9, 30, 0)
	dt2, _ := datetime_create(1970, 6, 15, 23, 59, 59)
	dt3, _ := datetime_create(2024, 12, 31, 0, 0, 1)
	col, err := column_from("dt", []Datetime{dt1, dt2, dt3})
	testing.expect(t, err == .None, "datetime column")
	defer column_destroy(&col)
	testing.expect(t, column_set_valid(&col, 1, false) == .None, "row 1 NULL")

	yr, yr_err := dt_year(&col)
	defer column_destroy(&yr)
	testing.expect(t, yr_err == .None, "dt_year")
	expect_i64_col(t, "year", &yr, []i64{2024, 0, 2024}, []bool{true, false, true})

	wd, wd_err := dt_weekday(&col)
	defer column_destroy(&wd)
	testing.expect(t, wd_err == .None, "dt_weekday")
	expect_i64_col(t, "weekday", &wd, []i64{1, 0, 2}, []bool{true, false, true})

	hr, hr_err := dt_hour(&col)
	defer column_destroy(&hr)
	testing.expect(t, hr_err == .None, "dt_hour")
	expect_i64_col(t, "hour", &hr, []i64{9, 0, 0}, []bool{true, false, true})

	// a Date column drives the date accessors
	dc, dc_err := column_from("d", []Date{Date(days_from_civil(2024, 2, 29)), Date(days_from_civil(1969, 12, 31))})
	testing.expect(t, dc_err == .None, "date column")
	defer column_destroy(&dc)
	dy, dy_err := dt_year(&dc)
	defer column_destroy(&dy)
	expect_i64_col(t, "date year", &dy, []i64{2024, 1969})

	// wrong input dtypes
	_, err = dt_hour(&dc)
	testing.expect(t, err == .Type_Mismatch, "date has no hour")
	tc, terr := column_from("t", []Time{Time(3600 * US_PER_SECOND)})
	testing.expect(t, terr == .None, "time column")
	defer column_destroy(&tc)
	_, err = dt_year(&tc)
	testing.expect(t, err == .Type_Mismatch, "time has no year")
	th, th_err := dt_hour(&tc)
	defer column_destroy(&th)
	expect_i64_col(t, "time hour", &th, []i64{1})
	_, err = dt_year(&yr)
	testing.expect(t, err == .Type_Mismatch, "i64 not temporal")
}

@(test)
calendar_sort_dates :: proc(t: ^testing.T) {
	d1, _ := date_create(2024, 3, 10)
	d2, _ := date_create(2024, 1, 15)
	d3, _ := date_create(2023, 12, 31)
	d4, _ := date_create(1969, 6, 1)
	id, ierr := column_from("id", []i32{1, 2, 3, 4})
	testing.expect(t, ierr == .None, "id column")
	dcol, derr := column_from("d", []Date{d1, d2, d3, d4})
	testing.expect(t, derr == .None, "date column")
	df, ferr := dataframe_from_columns([]^Column{&id, &dcol})
	testing.expect(t, ferr == .None, "from_columns")
	defer dataframe_destroy(&df)

	out, serr := dataframe_sort(&df, []Sort_Key{sort_key("d")})
	defer dataframe_destroy(&out)
	testing.expect(t, serr == .None, "sort dates asc")
	expect_sorted_id(t, "asc", &out, []i32{4, 3, 2, 1})
}

@(test)
calendar_date_range :: proc(t: ^testing.T) {
	start, _ := datetime_create(2024, 1, 1, 0, 0, 0)
	end, _ := datetime_create(2024, 1, 3, 0, 0, 0)
	every := duration_from_days(1)

	both, err := date_range(start, end, every, .Both)
	defer column_destroy(&both)
	testing.expect(t, err == .None, "date_range both")
	testing.expect(t, both.count == 3, "both: 3 values")
	bv := column_typed_view(&both, Datetime)
	testing.expect(t, datetime_day(bv[0]) == 1 && datetime_day(bv[1]) == 2 && datetime_day(bv[2]) == 3, "both: days 1,2,3")

	left, l_err := date_range(start, end, every, .Left)
	defer column_destroy(&left)
	testing.expect(t, l_err == .None && left.count == 2, "left: 2 values")
	lv := column_typed_view(&left, Datetime)
	testing.expect(t, datetime_day(lv[0]) == 1 && datetime_day(lv[1]) == 2, "left: days 1,2")

	right, r_err := date_range(start, end, every, .Right)
	defer column_destroy(&right)
	testing.expect(t, r_err == .None && right.count == 2, "right: 2 values")
	rv := column_typed_view(&right, Datetime)
	testing.expect(t, datetime_day(rv[0]) == 2 && datetime_day(rv[1]) == 3, "right: days 2,3")

	none, n_err := date_range(start, end, every, .None)
	defer column_destroy(&none)
	testing.expect(t, n_err == .None && none.count == 1, "none: 1 value")
	nv := column_typed_view(&none, Datetime)
	testing.expect(t, datetime_day(nv[0]) == 2, "none: day 2")

	// a non-boundary end keeps the last element under .Left/.None
	mid, _ := datetime_create(2024, 1, 2, 12, 0, 0)
	left2, l2_err := date_range(start, mid, every, .Left)
	defer column_destroy(&left2)
	testing.expect(t, l2_err == .None && left2.count == 2, "non-boundary left: 2 values")

	// every must be positive and end >= start
	_, err = date_range(end, start, every)
	testing.expect(t, err == .Invalid_Argument, "start > end rejected")
	_, err = date_range(start, end, Duration(0))
	testing.expect(t, err == .Invalid_Argument, "zero every rejected")
}

@(test)
calendar_truncate :: proc(t: ^testing.T) {
	noon, _ := datetime_create(1972, 6, 15, 12, 0, 0)
	pre, _ := datetime_create(1969, 12, 31, 12, 0, 0)
	exact, _ := datetime_create(1969, 12, 31, 0, 0, 0)
	col, err := column_from("t", []Datetime{noon, pre, exact})
	testing.expect(t, err == .None, "datetime column")
	defer column_destroy(&col)

	out, terr := truncate(&col, duration_from_days(1))
	defer column_destroy(&out)
	testing.expect(t, terr == .None, "truncate")
	ov := column_typed_view(&out, Datetime)
	testing.expect(t, datetime_hour(ov[0]) == 0 && datetime_day(ov[0]) == 15, "truncate floors to midnight")
	testing.expect(t, datetime_hour(ov[1]) == 0 && datetime_year(ov[1]) == 1969, "pre-epoch truncates to 1969-12-31 (floor, not trunc)")
	testing.expect(t, ov[2] == exact, "exact boundary unchanged")

	// offset shifts the window start
	off := duration_from_hours(12)
	early, _ := datetime_create(2024, 1, 1, 11, 0, 0)
	late, _ := datetime_create(2024, 1, 1, 13, 0, 0)
	col2, c2_err := column_from("t", []Datetime{early, late})
	testing.expect(t, c2_err == .None, "offset column")
	defer column_destroy(&col2)
	out2, t2err := truncate(&col2, duration_from_days(1), off)
	defer column_destroy(&out2)
	testing.expect(t, t2err == .None, "truncate with offset")
	o2 := column_typed_view(&out2, Datetime)
	testing.expect(t, datetime_year(o2[0]) == 2023 && datetime_month(o2[0]) == 12 && datetime_day(o2[0]) == 31, "11:00 falls in the previous 12:00-window")
	testing.expect(t, datetime_day(o2[1]) == 1 && datetime_hour(o2[1]) == 12, "13:00 falls in the 12:00-window")

	// NULLs pass through
	col3, _ := column_from("t", []Datetime{noon, pre})
	defer column_destroy(&col3)
	testing.expect(t, column_set_valid(&col3, 0, false) == .None, "mark NULL")
	out3, t3err := truncate(&col3, duration_from_days(1))
	defer column_destroy(&out3)
	testing.expect(t, t3err == .None && !column_is_valid(&out3, 0) && column_is_valid(&out3, 1), "NULL preserved")

	// wrong dtype
	i64col, _ := column_from("t", []i64{1, 2})
	defer column_destroy(&i64col)
	_, terr = truncate(&i64col, duration_from_days(1))
	testing.expect(t, terr == .Type_Mismatch, "i64 rejected")
}

@(test)
calendar_print_temporal :: proc(t: ^testing.T) {
	dt, _ := datetime_create(2024, 1, 1, 12, 30, 45, 123456)
	dcol, d_err := column_from("d", []Date{Date(days_from_civil(2024, 2, 29))})
	testing.expect(t, d_err == .None, "date column")
	tcol, t_err := column_from("t", []Time{Time(7 * US_PER_HOUR + 5 * US_PER_MINUTE + 9 * US_PER_SECOND)})
	testing.expect(t, t_err == .None, "time column")
	dtcol, dt_err := column_from("dt", []Datetime{dt})
	testing.expect(t, dt_err == .None, "datetime column")
	dur, dur_err := column_from("dur", []Duration{duration_from_days(1) + duration_from_seconds(2)})
	testing.expect(t, dur_err == .None, "duration column")
	neg, neg_err := column_from("neg", []Duration{Duration(-(i64(US_PER_HOUR) + i64(US_PER_SECOND)))})
	testing.expect(t, neg_err == .None, "negative duration column")

	df, ferr := dataframe_from_columns([]^Column{&dcol, &tcol, &dtcol, &dur, &neg})
	testing.expect(t, ferr == .None, "from_columns")
	defer dataframe_destroy(&df)

	s, serr := dataframe_to_string(&df)
	defer delete(s)
	testing.expect(t, serr == .None, "to_string")
	testing.expect(t, strings.contains(s, "2024-02-29"), "date renders ISO")
	testing.expect(t, strings.contains(s, "07:05:09.000000"), "time renders ISO")
	testing.expect(t, strings.contains(s, "2024-01-01T12:30:45.123456"), "datetime renders ISO")
	testing.expect(t, strings.contains(s, "1:00:00:02.000000"), "duration renders days")
	testing.expect(t, strings.contains(s, "-01:00:01.000000"), "negative duration renders")
}

@(test)
calendar_column_rename :: proc(t: ^testing.T) {
	col, err := column_from("a", []i32{1, 2})
	defer column_destroy(&col)
	testing.expect(t, err == .None, "column")
	testing.expect(t, column_rename(&col, "b") == .None, "rename")
	testing.expect(t, col.name == "b", "name updated")
	testing.expect(t, column_rename(&col, "") == .Column_Name_Empty, "empty rejected")
	testing.expect(t, col.name == "b", "name unchanged on error")
	testing.expect(t, column_rename(&col, "c") == .None, "rename again")
}

@(test)
calendar_schema_and_copy :: proc(t: ^testing.T) {
	col, err := column_from("dt", []Datetime{})
	testing.expect(t, err == .None, "empty datetime column")
	defer column_destroy(&col)

	df, ferr := dataframe_from_columns([]^Column{&col})
	testing.expect(t, ferr == .None, "from_columns")
	defer dataframe_destroy(&df)

	schema, serr := dataframe_schema(&df)
	defer schema_destroy(&schema)
	testing.expect(t, serr == .None, "schema")
	empty, eerr := dataframe_create_with_schema(schema)
	testing.expect(t, eerr == .None, "create_with_schema accepts datetime")
	defer dataframe_destroy(&empty)
	ecol, gerr := dataframe_get_column(&empty, "dt")
	testing.expect(t, gerr == .None && ecol.dtype == typeid_of(Datetime), "schema preserves dtype")

	d1, _ := date_create(2024, 3, 10)
	src, s_err := column_from("d", []Date{d1})
	testing.expect(t, s_err == .None, "source date column")
	defer column_destroy(&src)
	cp, cerr := column_copy(&src)
	defer column_destroy(&cp)
	testing.expect(t, cerr == .None && cp.dtype == typeid_of(Date) && cp.count == 1, "copy preserves dtype")
}
