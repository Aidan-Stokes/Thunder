package dataframe

// Human-readable DataFrame rendering (Stage 10): dataframe_to_string builds
// an aligned text table and dataframe_print writes it to stdout.
//
// Layout (polars-inspired, plain text):
//
//   shape: (3, 3)
//   age  name     salary
//    25  ada      150000
//    30  grace    null
//    35  katherine 95000
//
// Each column is left-aligned to the width of its widest cell (header or
// value) and columns are separated by two spaces. NULL rows render as "null"
// so they stay distinct from the value 0 / the empty string (principle 8).
// String values containing newlines are not reflowed and will misalign.

import "core:fmt"
import "core:strings"
import "base:runtime"

// dataframe_to_string renders df as a table string owned by the caller
// (release with delete). Every supported column dtype is rendered; NULL rows
// appear as "null".
dataframe_to_string :: proc(df: ^DataFrame, allocator := context.allocator) -> (string, Error) {
	sb := strings.builder_make(allocator)
	defer strings.builder_destroy(&sb)

	n_rows := dataframe_num_rows(df)
	n_cols := dataframe_num_cols(df)
	fmt.sbprintf(&sb, "shape: (%d, %d)\n", n_rows, n_cols)

	widths := make([]int, n_cols, allocator)
	if widths == nil && n_cols != 0 {
		return {}, .Allocator_Failure
	}
	defer delete(widths, allocator)

	scratch := strings.builder_make(allocator)
	defer strings.builder_destroy(&scratch)

	cs := &df.columns
	for col_i in 0 ..< n_cols {
		if w := len(cs.names[col_i]); w > widths[col_i] {
			widths[col_i] = w
		}
		for row in 0 ..< n_rows {
			strings.builder_reset(&scratch)
			render_cell_cs(cs, col_i, row, &scratch)
			if w := strings.builder_len(scratch); w > widths[col_i] {
				widths[col_i] = w
			}
		}
	}

	write_padded_cell := proc(sb, scratch: ^strings.Builder, width: int, pad: bool) {
		cell := strings.to_string(scratch^)
		strings.write_string(sb, cell)
		if pad {
			for _ in len(cell) ..< width {
				strings.write_byte(sb, ' ')
			}
		}
	}

	for col_i in 0 ..< n_cols {
		if col_i > 0 {
			strings.write_string(&sb, "  ")
		}
		strings.builder_reset(&scratch)
		strings.write_string(&scratch, cs.names[col_i])
		write_padded_cell(&sb, &scratch, widths[col_i], col_i < n_cols - 1)
	}
	if n_cols > 0 {
		strings.write_byte(&sb, '\n')
	}

	for row in 0 ..< n_rows {
		for col_i in 0 ..< n_cols {
			if col_i > 0 {
				strings.write_string(&sb, "  ")
			}
			strings.builder_reset(&scratch)
			render_cell_cs(cs, col_i, row, &scratch)
			write_padded_cell(&sb, &scratch, widths[col_i], col_i < n_cols - 1)
		}
		if n_cols > 0 {
			strings.write_byte(&sb, '\n')
		}
	}

	// clone detaches the built string from the builder, which is destroyed by
	// the defer above (the returned string stays owned by the caller).
	s, c_err := strings.clone(strings.to_string(sb), allocator)
	if c_err != runtime.Allocator_Error.None {
		return {}, .Allocator_Failure
	}
	return s, .None
}

// dataframe_print renders df to stdout via dataframe_to_string.
dataframe_print :: proc(df: ^DataFrame) {
	s, err := dataframe_to_string(df)
	if err != .None {
		fmt.eprintln("dataframe_print failed:", err)
		return
	}
	defer delete(s)
	fmt.print(s)
}

// render_cell writes the display text of column col at row into sb. NULL rows
// render as "null"; strings render verbatim; every other supported dtype uses
// default formatting.
@(private)
render_cell :: proc(col: ^Column, row: int, sb: ^strings.Builder) {
	if !column_is_valid(col, row) {
		strings.write_string(sb, "null")
		return
	}
	if is_categorical(col) {
		if s, ok := categorical_value(col, row); ok {
			strings.write_string(sb, s)
		}
		return
	}
	switch col.dtype {
	case typeid_of(bool):   fmt.sbprint(sb, column_typed_view(col, bool)[row])
	case typeid_of(i8):     fmt.sbprint(sb, column_typed_view(col, i8)[row])
	case typeid_of(i16):    fmt.sbprint(sb, column_typed_view(col, i16)[row])
	case typeid_of(i32):    fmt.sbprint(sb, column_typed_view(col, i32)[row])
	case typeid_of(i64):    fmt.sbprint(sb, column_typed_view(col, i64)[row])
	case typeid_of(u8):     fmt.sbprint(sb, column_typed_view(col, u8)[row])
	case typeid_of(u16):    fmt.sbprint(sb, column_typed_view(col, u16)[row])
	case typeid_of(u32):    fmt.sbprint(sb, column_typed_view(col, u32)[row])
	case typeid_of(u64):    fmt.sbprint(sb, column_typed_view(col, u64)[row])
	case typeid_of(int):    fmt.sbprint(sb, column_typed_view(col, int)[row])
	case typeid_of(uint):   fmt.sbprint(sb, column_typed_view(col, uint)[row])
	case typeid_of(f32):    fmt.sbprint(sb, column_typed_view(col, f32)[row])
	case typeid_of(f64):    fmt.sbprint(sb, column_typed_view(col, f64)[row])
	case typeid_of(string): strings.write_string(sb, column_typed_view(col, string)[row])
	case typeid_of(Date):     render_date_cell(sb, column_typed_view(col, Date)[row])
	case typeid_of(Datetime): render_datetime_cell(sb, column_typed_view(col, Datetime)[row])
	case typeid_of(Time):     render_time_cell(sb, column_typed_view(col, Time)[row])
	case typeid_of(Duration): render_duration_cell(sb, column_typed_view(col, Duration)[row])
	case typeid_of(List_Ref): render_list_cell(col, row, sb)
	case:
		strings.write_string(sb, "?")
	}
}

// render_cell_cs writes the display text of ColumnSet column i at row into sb,
// reading directly from ColumnSet arrays without reconstructing a Column.
@(private)
render_cell_cs :: proc(cs: ^ColumnSet, col_i: int, row: int, sb: ^strings.Builder) {
	if !row_valid(cs.valids[col_i], row) {
		strings.write_string(sb, "null")
		return
	}
	dt := cs.dtypes[col_i]
	meta := &cs.metas[col_i]
	switch dt {
	case typeid_of(bool):   fmt.sbprint(sb, cs_typed_view(cs, col_i, bool)[row])
	case typeid_of(i8):     fmt.sbprint(sb, cs_typed_view(cs, col_i, i8)[row])
	case typeid_of(i16):    fmt.sbprint(sb, cs_typed_view(cs, col_i, i16)[row])
	case typeid_of(i32):    fmt.sbprint(sb, cs_typed_view(cs, col_i, i32)[row])
	case typeid_of(i64):    fmt.sbprint(sb, cs_typed_view(cs, col_i, i64)[row])
	case typeid_of(u8):     fmt.sbprint(sb, cs_typed_view(cs, col_i, u8)[row])
	case typeid_of(u16):    fmt.sbprint(sb, cs_typed_view(cs, col_i, u16)[row])
	case typeid_of(u32):    fmt.sbprint(sb, cs_typed_view(cs, col_i, u32)[row])
	case typeid_of(u64):    fmt.sbprint(sb, cs_typed_view(cs, col_i, u64)[row])
	case typeid_of(int):    fmt.sbprint(sb, cs_typed_view(cs, col_i, int)[row])
	case typeid_of(uint):   fmt.sbprint(sb, cs_typed_view(cs, col_i, uint)[row])
	case typeid_of(f32):    fmt.sbprint(sb, cs_typed_view(cs, col_i, f32)[row])
	case typeid_of(f64):    fmt.sbprint(sb, cs_typed_view(cs, col_i, f64)[row])
	case typeid_of(string): strings.write_string(sb, cs_typed_view(cs, col_i, string)[row])
	case typeid_of(Date):     render_date_cell(sb, cs_typed_view(cs, col_i, Date)[row])
	case typeid_of(Datetime): render_datetime_cell(sb, cs_typed_view(cs, col_i, Datetime)[row])
	case typeid_of(Time):     render_time_cell(sb, cs_typed_view(cs, col_i, Time)[row])
	case typeid_of(Duration): render_duration_cell(sb, cs_typed_view(cs, col_i, Duration)[row])
	case typeid_of(List_Ref):
		// List columns need full Column for render_list_cell.
		col := Column{
			dtype       = dt,
			data        = cs.data[col_i],
			count       = cs.rows[col_i],
			valid       = cs.valids[col_i],
			alloc       = meta.alloc,
			payload     = meta.payload,
			payload_size = meta.payload_size,
			inner_dtype = meta.inner_dtype,
			inner_valid = meta.inner_valid,
		}
		render_list_cell(&col, row, sb)
	case:
		if meta.categories != nil {
			col := Column{
				dtype           = dt,
				data            = cs.data[col_i],
				count           = cs.rows[col_i],
				valid           = cs.valids[col_i],
				alloc           = meta.alloc,
				categories      = meta.categories,
				categorical_kind = meta.categorical_kind,
			}
			if s, ok := categorical_value(&col, row); ok {
				strings.write_string(sb, s)
			}
		} else {
			strings.write_string(sb, "?")
		}
	}
}

// render_list_cell writes a list row as [e0, e1, ...], with NULL elements as
// "null" and the row's NULL handled by the caller.
@(private)
render_list_cell :: proc(col: ^Column, row: int, sb: ^strings.Builder) {
	ref := column_typed_view(col, List_Ref)[row]
	strings.write_byte(sb, '[')
	for e in ref.off ..< ref.off + ref.len {
		if e > ref.off {
			strings.write_string(sb, ", ")
		}
		if !inner_is_valid(col, e) {
			strings.write_string(sb, "null")
			continue
		}
		render_inner_cell(sb, col.inner_dtype, col, e)
	}
	strings.write_byte(sb, ']')
}

// render_inner_cell writes one inner element of a list column.
@(private)
render_inner_cell :: proc(sb: ^strings.Builder, dt: typeid, col: ^Column, e: int) {
	switch dt {
	case typeid_of(bool):   fmt.sbprint(sb, list_inner_view(col, bool)[e])
	case typeid_of(i8):     fmt.sbprint(sb, list_inner_view(col, i8)[e])
	case typeid_of(i16):    fmt.sbprint(sb, list_inner_view(col, i16)[e])
	case typeid_of(i32):    fmt.sbprint(sb, list_inner_view(col, i32)[e])
	case typeid_of(i64):    fmt.sbprint(sb, list_inner_view(col, i64)[e])
	case typeid_of(u8):     fmt.sbprint(sb, list_inner_view(col, u8)[e])
	case typeid_of(u16):    fmt.sbprint(sb, list_inner_view(col, u16)[e])
	case typeid_of(u32):    fmt.sbprint(sb, list_inner_view(col, u32)[e])
	case typeid_of(u64):    fmt.sbprint(sb, list_inner_view(col, u64)[e])
	case typeid_of(int):    fmt.sbprint(sb, list_inner_view(col, int)[e])
	case typeid_of(uint):   fmt.sbprint(sb, list_inner_view(col, uint)[e])
	case typeid_of(f32):    fmt.sbprint(sb, list_inner_view(col, f32)[e])
	case typeid_of(f64):    fmt.sbprint(sb, list_inner_view(col, f64)[e])
	case typeid_of(string): strings.write_string(sb, list_inner_view(col, string)[e])
	case typeid_of(Date):     render_date_cell(sb, list_inner_view(col, Date)[e])
	case typeid_of(Datetime): render_datetime_cell(sb, list_inner_view(col, Datetime)[e])
	case typeid_of(Time):     render_time_cell(sb, list_inner_view(col, Time)[e])
	case typeid_of(Duration): render_duration_cell(sb, list_inner_view(col, Duration)[e])
	case:
		strings.write_string(sb, "?")
	}
}

// render_date_cell writes a Date as YYYY-MM-DD.
@(private)
render_date_cell :: proc(sb: ^strings.Builder, d: Date) {
	y, m, day := civil_from_days(i64(d))
	fmt.sbprintf(sb, "%04d-%02d-%02d", y, m, day)
}

// render_datetime_cell writes a Datetime as YYYY-MM-DDTHH:MM:SS.ffffff.
@(private)
render_datetime_cell :: proc(sb: ^strings.Builder, dt: Datetime) {
	d := datetime_to_date(dt)
	y, m, day := civil_from_days(i64(d))
	fmt.sbprintf(sb, "%04d-%02d-%02dT", y, m, day)
	render_time_cell(sb, datetime_time_of_day(dt))
}

// render_time_cell writes a Time as HH:MM:SS.ffffff.
@(private)
render_time_cell :: proc(sb: ^strings.Builder, t: Time) {
	us := i64(t)
	fmt.sbprintf(
		sb,
		"%02d:%02d:%02d.%06d",
		us / US_PER_HOUR,
		us % US_PER_HOUR / US_PER_MINUTE,
		us % US_PER_MINUTE / US_PER_SECOND,
		us % US_PER_SECOND,
	)
}

// render_duration_cell writes a Duration as [-][DDD:]HH:MM:SS.ffffff, omitting
// the day field when it is zero.
@(private)
render_duration_cell :: proc(sb: ^strings.Builder, dur: Duration) {
	us := i64(dur)
	neg := us < 0
	if neg {
		us = -us
		strings.write_byte(sb, '-')
	}
	days := us / US_PER_DAY
	rem := us % US_PER_DAY
	if days != 0 {
		fmt.sbprintf(sb, "%d:", days)
	}
	fmt.sbprintf(
		sb,
		"%02d:%02d:%02d.%06d",
		rem / US_PER_HOUR,
		rem % US_PER_HOUR / US_PER_MINUTE,
		rem % US_PER_MINUTE / US_PER_SECOND,
		rem % US_PER_SECOND,
	)
}
