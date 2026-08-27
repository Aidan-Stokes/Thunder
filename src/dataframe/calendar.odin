package dataframe

// Temporal types (DESIGN.md §18.5, ROADMAP S14.8): four distinct i64-backed
// scalar types plus calendar math, column field accessors (dt_*), date_range,
// and truncate.
//
// Storage is i64, so encode_row (grouping/joining keys), byte-copy kernels,
// and sort comparison work unchanged; temporal arithmetic is expressed in
// these units:
//
//	Date     days since 1970-01-01 (proleptic Gregorian calendar)
//	Datetime microseconds since 1970-01-01T00:00:00
//	Time     microseconds since midnight
//	Duration signed microseconds
//
// All day arithmetic uses floor division, so pre-1970 values are exact
// (truncation would shift 1969-12-31T00:00:00.000001 to day 0). NULL rows
// stay NULL; there is no implicit conversion between temporal types and
// scalars (principle 6).

import "core:mem"
import "core:strings"

Date     :: distinct i64
Datetime :: distinct i64
Time     :: distinct i64
Duration :: distinct i64

@(private)
US_PER_SECOND :: 1_000_000
@(private)
US_PER_MINUTE :: 60_000_000
@(private)
US_PER_HOUR   :: 3_600_000_000
@(private)
US_PER_DAY    :: 86_400_000_000

// floor_div divides with the quotient rounded toward negative infinity. b must
// be positive (true for every call site here).
@(private)
floor_div :: proc(a, b: i64) -> i64 {
	q := a / b
	if r := a % b; r < 0 {
		q -= 1
	}
	return q
}

// floor_mod is the non-negative remainder: floor_mod(a, b) is in [0, b).
@(private)
floor_mod :: proc(a, b: i64) -> i64 {
	return a - floor_div(a, b) * b
}

// --- calendar math (Howard Hinnant's algorithms, public domain) ---------------

// days_from_civil converts a proleptic Gregorian date to days since
// 1970-01-01. The range y in [-32767, 32767] is exact.
@(private)
days_from_civil :: proc(y0, m0, d0: i64) -> i64 {
	y := y0
	m := m0
	d := d0
	y -= m <= 2 ? 1 : 0
	era := (y >= 0 ? y : y - 399) / 400
	yoe := y - era * 400
	doy := (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1
	doe := yoe * 365 + yoe / 4 - yoe / 100 + doy
	return era * 146097 + doe - 719468
}

// civil_from_days is the inverse of days_from_civil.
@(private)
civil_from_days :: proc(z: i64) -> (y, m, d: i64) {
	dz := z + 719468
	era := (dz >= 0 ? dz : dz - 146096) / 146097
	doe := dz - era * 146097
	yoe := (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
	yy := yoe + era * 400
	doy := doe - (365 * yoe + yoe / 4 - yoe / 100)
	mp := (5 * doy + 2) / 153
	dd := doy - (153 * mp + 2) / 5 + 1
	mm := mp + (mp < 10 ? 3 : -9)
	if mm <= 2 {
		yy += 1
	}
	return yy, mm, dd
}

// weekday_from_days maps a day count to 1=Monday .. 7=Sunday.
// 1970-01-01 (day 0) was a Thursday (4).
@(private)
weekday_from_days :: proc(z: i64) -> i64 {
	wd := floor_mod(z + 4, 7)
	if wd == 0 {
		wd = 7
	}
	return wd
}

@(private)
is_leap_year :: proc(year: int) -> bool {
	return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

// days_in_month returns the number of days in month (1..12), or 0 for an
// out-of-range month.
@(private)
days_in_month :: proc(year, month: int) -> int {
	switch month {
	case 1, 3, 5, 7, 8, 10, 12:
		return 31
	case 4, 6, 9, 11:
		return 30
	case 2:
		return is_leap_year(year) ? 29 : 28
	case:
		return 0
	}
}

// --- constructors -------------------------------------------------------------

// date_create validates and builds a Date; out-of-range fields are
// .Invalid_Argument.
date_create :: proc(year, month, day: int) -> (Date, Error) {
	if month < 1 || month > 12 || day < 1 || day > days_in_month(year, month) {
		return {}, .Invalid_Argument
	}
	return Date(days_from_civil(i64(year), i64(month), i64(day))), .None
}

// datetime_create validates and builds a Datetime.
datetime_create :: proc(year, month, day, hour, minute, second: int, microsecond := 0) -> (Datetime, Error) {
	d, err := date_create(year, month, day)
	if err != .None {
		return {}, err
	}
	if hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 59 || microsecond < 0 || microsecond > 999_999 {
		return {}, .Invalid_Argument
	}
	return Datetime(i64(d) * US_PER_DAY + i64(hour) * US_PER_HOUR + i64(minute) * US_PER_MINUTE + i64(second) * US_PER_SECOND + i64(microsecond)), .None
}

// time_create validates and builds a Time.
time_create :: proc(hour, minute, second: int, microsecond := 0) -> (Time, Error) {
	if hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 59 || microsecond < 0 || microsecond > 999_999 {
		return {}, .Invalid_Argument
	}
	return Time(i64(hour) * US_PER_HOUR + i64(minute) * US_PER_MINUTE + i64(second) * US_PER_SECOND + i64(microsecond)), .None
}

// duration_from_* build a Duration in the named unit.
duration_from_days :: proc(days: int) -> Duration {
	return Duration(i64(days) * US_PER_DAY)
}

duration_from_hours :: proc(hours: int) -> Duration {
	return Duration(i64(hours) * US_PER_HOUR)
}

duration_from_minutes :: proc(minutes: int) -> Duration {
	return Duration(i64(minutes) * US_PER_MINUTE)
}

duration_from_seconds :: proc(seconds: int) -> Duration {
	return Duration(i64(seconds) * US_PER_SECOND)
}

duration_from_microseconds :: proc(us: i64) -> Duration {
	return Duration(us)
}

// --- conversions --------------------------------------------------------------

// date_to_datetime converts a Date to midnight of the same day.
date_to_datetime :: proc(d: Date) -> Datetime {
	return Datetime(i64(d) * US_PER_DAY)
}

// datetime_to_date truncates a Datetime to its day.
datetime_to_date :: proc(dt: Datetime) -> Date {
	return Date(floor_div(i64(dt), US_PER_DAY))
}

// datetime_add shifts a Datetime by a Duration.
datetime_add :: proc(dt: Datetime, dur: Duration) -> Datetime {
	return Datetime(i64(dt) + i64(dur))
}

// datetime_diff returns a - b as a Duration.
datetime_diff :: proc(a, b: Datetime) -> Duration {
	return Duration(i64(a) - i64(b))
}

// datetime_time_of_day extracts the time elapsed since midnight.
datetime_time_of_day :: proc(dt: Datetime) -> Time {
	return Time(floor_mod(i64(dt), US_PER_DAY))
}

// --- scalar field accessors ---------------------------------------------------

date_year :: proc(d: Date) -> i64 {
	y, _, _ := civil_from_days(i64(d))
	return y
}

date_month :: proc(d: Date) -> i64 {
	_, m, _ := civil_from_days(i64(d))
	return m
}

date_day :: proc(d: Date) -> i64 {
	_, _, day := civil_from_days(i64(d))
	return day
}

date_weekday :: proc(d: Date) -> i64 {
	return weekday_from_days(i64(d))
}

datetime_year :: proc(dt: Datetime) -> i64 {
	return date_year(datetime_to_date(dt))
}

datetime_month :: proc(dt: Datetime) -> i64 {
	return date_month(datetime_to_date(dt))
}

datetime_day :: proc(dt: Datetime) -> i64 {
	return date_day(datetime_to_date(dt))
}

datetime_weekday :: proc(dt: Datetime) -> i64 {
	return weekday_from_days(floor_div(i64(dt), US_PER_DAY))
}

datetime_hour :: proc(dt: Datetime) -> i64 {
	return floor_mod(i64(dt), US_PER_DAY) / US_PER_HOUR
}

datetime_minute :: proc(dt: Datetime) -> i64 {
	return floor_mod(i64(dt), US_PER_HOUR) / US_PER_MINUTE
}

datetime_second :: proc(dt: Datetime) -> i64 {
	return floor_mod(i64(dt), US_PER_MINUTE) / US_PER_SECOND
}

datetime_microsecond :: proc(dt: Datetime) -> i64 {
	return floor_mod(i64(dt), US_PER_SECOND)
}

time_hour :: proc(t: Time) -> i64 {
	return i64(t) / US_PER_HOUR
}

time_minute :: proc(t: Time) -> i64 {
	return i64(t) / US_PER_MINUTE
}

time_second :: proc(t: Time) -> i64 {
	return i64(t) / US_PER_SECOND
}

time_microsecond :: proc(t: Time) -> i64 {
	return i64(t) % US_PER_SECOND
}

// --- column field accessors ---------------------------------------------------
//
// dt_year/month/day/weekday accept a Date or Datetime column; dt_hour/minute/
// second/microsecond accept a Datetime or Time column. The result is an i64
// column with the source name; NULL rows stay NULL; a wrong input dtype is
// .Type_Mismatch.

@(private)
Temporal_Field :: enum byte {
	Year,
	Month,
	Day,
	Weekday,
	Hour,
	Minute,
	Second,
	Microsecond,
}

@(private)
temporal_field_column :: proc(allocator: mem.Allocator, col: ^Column, f: Temporal_Field) -> (out: Column, err: Error) {
	out = column_alloc(allocator, col.name, typeid_of(i64), size_of(i64), align_of(i64), col.count) or_return
	ov := column_typed_view(&out, i64)

	switch col.dtype {
	case typeid_of(Date):
		if f >= .Hour {
			column_destroy(&out)
			return {}, .Type_Mismatch
		}
		dv := column_typed_view(col, Date)
		for i in 0 ..< col.count {
			if !column_is_valid(col, i) {
				_ = column_set_null(&out, i)
				continue
			}
			ov[i] = date_field_value(dv[i], f)
		}
	case typeid_of(Datetime):
		dv := column_typed_view(col, Datetime)
		for i in 0 ..< col.count {
			if !column_is_valid(col, i) {
				_ = column_set_null(&out, i)
				continue
			}
			ov[i] = datetime_field_value(dv[i], f)
		}
	case typeid_of(Time):
		if f < .Hour {
			column_destroy(&out)
			return {}, .Type_Mismatch
		}
		tv := column_typed_view(col, Time)
		for i in 0 ..< col.count {
			if !column_is_valid(col, i) {
				_ = column_set_null(&out, i)
				continue
			}
			ov[i] = time_field_value(tv[i], f)
		}
	case:
		column_destroy(&out)
		return {}, .Type_Mismatch
	}
	return out, .None
}

@(private)
date_field_value :: proc(d: Date, f: Temporal_Field) -> i64 {
	y, m, day := civil_from_days(i64(d))
	#partial switch f {
	case .Year:
		return y
	case .Month:
		return m
	case .Day:
		return day
	case .Weekday:
		return weekday_from_days(i64(d))
	case:
		return 0
	}
}

@(private)
datetime_field_value :: proc(dt: Datetime, f: Temporal_Field) -> i64 {
	if f < .Hour {
		return date_field_value(datetime_to_date(dt), f)
	}
	#partial switch f {
	case .Hour:
		return datetime_hour(dt)
	case .Minute:
		return datetime_minute(dt)
	case .Second:
		return datetime_second(dt)
	case .Microsecond:
		return datetime_microsecond(dt)
	case:
		return 0
	}
}

@(private)
time_field_value :: proc(t: Time, f: Temporal_Field) -> i64 {
	#partial switch f {
	case .Hour:
		return time_hour(t)
	case .Minute:
		return time_minute(t)
	case .Second:
		return time_second(t)
	case .Microsecond:
		return time_microsecond(t)
	case:
		return 0
	}
}

dt_year :: proc(col: ^Column, allocator := context.allocator) -> (Column, Error) {
	return temporal_field_column(allocator, col, .Year)
}

dt_month :: proc(col: ^Column, allocator := context.allocator) -> (Column, Error) {
	return temporal_field_column(allocator, col, .Month)
}

dt_day :: proc(col: ^Column, allocator := context.allocator) -> (Column, Error) {
	return temporal_field_column(allocator, col, .Day)
}

dt_weekday :: proc(col: ^Column, allocator := context.allocator) -> (Column, Error) {
	return temporal_field_column(allocator, col, .Weekday)
}

dt_hour :: proc(col: ^Column, allocator := context.allocator) -> (Column, Error) {
	return temporal_field_column(allocator, col, .Hour)
}

dt_minute :: proc(col: ^Column, allocator := context.allocator) -> (Column, Error) {
	return temporal_field_column(allocator, col, .Minute)
}

dt_second :: proc(col: ^Column, allocator := context.allocator) -> (Column, Error) {
	return temporal_field_column(allocator, col, .Second)
}

dt_microsecond :: proc(col: ^Column, allocator := context.allocator) -> (Column, Error) {
	return temporal_field_column(allocator, col, .Microsecond)
}

// --- date_range and truncate --------------------------------------------------
//
// date_range builds a Datetime column of window starts start + k*every within
// [start, end]. closed selects which range endpoints are kept when they land
// exactly on a boundary:
//
//	.Both  start .. end
//	.Left  start .. end-exclusive
//	.Right start-exclusive .. end
//	.None  start-exclusive .. end-exclusive
//
// truncate floors each value to its window start floor((t - offset)/every) *
// every + offset (floor division, so pre-epoch datetimes are exact).

// Closed_Interval names the endpoint convention of a window or range.
Closed_Interval :: enum byte {
	Both,
	Left,
	Right,
	None,
}

date_range :: proc(
	start, end: Datetime,
	every: Duration,
	closed: Closed_Interval = .Both,
	name: string = "date_range",
	allocator := context.allocator,
) -> (Column, Error) {
	if every <= Duration(0) {
		return {}, .Invalid_Argument
	}
	if end < start {
		return {}, .Invalid_Argument
	}
	every_us := i64(every)
	span_us := i64(end) - i64(start)
	last := floor_div(span_us, every_us)
	first := i64(0)
	end_is_boundary := i64(end) == i64(start) + last * every_us
	switch closed {
	case .Both:
		// keep first .. last
	case .Left:
		if end_is_boundary {
			last -= 1
		}
	case .Right:
		first += 1
	case .None:
		if end_is_boundary {
			last -= 1
		}
		first += 1
	}
	n := last - first + 1
	if n <= 0 {
		return {}, .Invalid_Argument
	}
	out, err := column_alloc(allocator, name, typeid_of(Datetime), size_of(Datetime), align_of(Datetime), int(n))
	if err != .None {
		return {}, err
	}
	dv := column_typed_view(&out, Datetime)
	for i in 0 ..< n {
		k := first + i
		dv[i] = Datetime(i64(start) + k * every_us)
	}
	return out, .None
}

// truncate floors a Datetime column to window starts; every must be positive.
truncate :: proc(col: ^Column, every: Duration, offset: Duration = Duration(0), allocator := context.allocator) -> (Column, Error) {
	if col.dtype != typeid_of(Datetime) {
		return {}, .Type_Mismatch
	}
	if every <= Duration(0) {
		return {}, .Invalid_Argument
	}
	out, err := column_copy(col, allocator)
	if err != .None {
		return {}, err
	}
	every_us := i64(every)
	off_us := i64(offset)
	dv := column_typed_view(&out, Datetime)
	for i in 0 ..< out.count {
		if column_is_valid(&out, i) {
			t := i64(dv[i])
			dv[i] = Datetime(floor_div(t - off_us, every_us) * every_us + off_us)
		}
	}
	return out, .None
}
