package dataframe

// S15.7 tests: parallel CSV chunk splitting, parallel parse correctness
// (parallel output == sequential output for all column types), quoted
// newlines across chunks, edge cases (tiny file, single chunk, all NULLs).

import "core:fmt"
import "core:os"
import "core:strings"
import "core:testing"

// --- chunk splitting tests --------------------------------------------------

@(test)
csv_find_chunks_basic :: proc(t: ^testing.T) {
	// 5 records, 2 threads -> expect chunks that cover all records.
	data := "1,2\n3,4\n5,6\n7,8\n9,10\n"
	chunks := csv_find_chunks(data, 2)
	testing.expect(t, chunks != nil, "chunks should not be nil")
	defer delete(chunks)
	testing.expect(t, len(chunks) == 2, fmt.tprintf("expected 2 chunks, got %d", len(chunks)))

	// All data should be covered.
	for c in chunks {
		testing.expect(t, c.start <= c.end, "start <= end")
		testing.expect(t, c.end <= len(data), "end <= len(data)")
	}
	// First chunk starts at 0.
	testing.expect(t, chunks[0].start == 0, "first chunk starts at 0")
	// Last chunk ends at len(data).
	testing.expect(t, chunks[len(chunks) - 1].end == len(data), fmt.tprintf("last chunk end %d != %d", chunks[len(chunks) - 1].end, len(data)))
}

@(test)
csv_find_chunks_single :: proc(t: ^testing.T) {
	data := "a,b\n1,2\n"
	chunks := csv_find_chunks(data, 1)
	testing.expect(t, chunks != nil, "chunks should not be nil")
	defer delete(chunks)
	testing.expect(t, len(chunks) == 1, "1 chunk")
	testing.expect(t, chunks[0].start == 0, "start=0")
	testing.expect(t, chunks[0].end == len(data), fmt.tprintf("end=%d full len=%d", chunks[0].end, len(data)))
}

@(test)
csv_find_chunks_more_threads_than_records :: proc(t: ^testing.T) {
	data := "a,b\n1,2\n3,4\n"
	chunks := csv_find_chunks(data, 8)
	testing.expect(t, chunks != nil, "chunks should not be nil")
	defer delete(chunks)
	testing.expect(t, len(chunks) == 8, "8 chunks requested")

	// All data still covered.
	for c in chunks {
		testing.expect(t, c.start <= c.end, "start <= end")
	}
	testing.expect(t, chunks[0].start == 0, "first starts at 0")
	testing.expect(t, chunks[len(chunks) - 1].end == len(data), "last ends at data end")
}

@(test)
csv_find_chunks_quoted_newlines :: proc(t: ^testing.T) {
	// Record 2 has a quoted newline — chunk boundary must not split inside.
	data := "a,b\n1,2\n3,\"hello\nworld\"\n7,8\n"
	chunks := csv_find_chunks(data, 2)
	testing.expect(t, chunks != nil, "chunks not nil")
	defer delete(chunks)
	testing.expect(t, len(chunks) == 2, "2 chunks")

	// Verify no chunk boundary falls inside the quoted field.
	for c in chunks {
		// Boundary must be at 0, a newline, or data end.
		if c.start > 0 {
			testing.expect(t, data[c.start - 1] == '\n' || data[c.start - 1] == '\r',
				fmt.tprintf("chunk start %d not at record boundary", c.start))
		}
	}
}

@(test)
csv_find_chunks_empty :: proc(t: ^testing.T) {
	chunks := csv_find_chunks("", 4)
	testing.expect(t, chunks == nil, "empty data -> nil chunks")
}

// --- parallel vs sequential correctness tests -------------------------------

// csv_sequential reads path with the sequential (non-parallel) path and
// returns the DataFrame.
csv_sequential :: proc(path: string) -> (DataFrame, Error) {
	data, o_err := os.read_entire_file_from_path(path, context.allocator)
	if o_err != os.ERROR_NONE {
		return {}, .CSV_Error
	}
	defer delete(data)

	kinds, names, k_err := csv_infer_types(string(data), context.allocator, CSV_Options{})
	if k_err != .None {
		return {}, k_err
	}
	defer delete(kinds)
	defer csv_delete_strings(names, context.allocator)

	return csv_build_columns(string(data), context.allocator, CSV_Options{}, names, kinds)
}

// csv_parallel reads path with the parallel path and returns the DataFrame.
csv_parallel :: proc(path: string) -> (DataFrame, Error) {
	return dataframe_read_csv(path)
}

// csv_expect_eq asserts that two DataFrames are structurally identical:
// same row count, same column count, same column names, same dtypes,
// same values, same validity.
csv_expect_eq :: proc(t: ^testing.T, a, b: ^DataFrame, label: string) {
	testing.expect(t, dataframe_num_rows(a) == dataframe_num_rows(b),
		fmt.tprintf("%s: row count %d != %d", label, dataframe_num_rows(a), dataframe_num_rows(b)))
	testing.expect(t, dataframe_num_cols(a) == dataframe_num_cols(b),
		fmt.tprintf("%s: col count %d != %d", label, dataframe_num_cols(a), dataframe_num_cols(b)))

	nr := min(dataframe_num_rows(a), dataframe_num_rows(b))
	nc := min(dataframe_num_cols(a), dataframe_num_cols(b))

	for ci in 0 ..< nc {
		ca := dataframe_column_at(a, ci) or_else nil
		cb := dataframe_column_at(b, ci) or_else nil
		if ca == nil || cb == nil {
			testing.expect(t, false, fmt.tprintf("%s: col %d is nil", label, ci))
			continue
		}
		testing.expect(t, ca.name == cb.name,
			fmt.tprintf("%s: col %d name %s != %s", label, ci, ca.name, cb.name))
		testing.expect(t, ca.dtype == cb.dtype,
			fmt.tprintf("%s: col %d dtype mismatch", label, ci))

		switch ca.dtype {
		case typeid_of(i64):
			for ri in 0 ..< nr {
				va, ea, _ := column_get(ca, ri, i64)
				vb, eb, _ := column_get(cb, ri, i64)
				testing.expect(t, ea == eb,
					fmt.tprintf("%s: row %d col %d valid mismatch", label, ri, ci))
				if ea && eb {
					testing.expect(t, va == vb,
						fmt.tprintf("%s: row %d col %d i64 %d != %d", label, ri, ci, va, vb))
				}
			}
		case typeid_of(f64):
			for ri in 0 ..< nr {
				va, ea, _ := column_get(ca, ri, f64)
				vb, eb, _ := column_get(cb, ri, f64)
				testing.expect(t, ea == eb,
					fmt.tprintf("%s: row %d col %d valid mismatch", label, ri, ci))
				if ea && eb {
					testing.expect(t, va == vb,
						fmt.tprintf("%s: row %d col %d f64 %v != %v", label, ri, ci, va, vb))
				}
			}
		case typeid_of(bool):
			for ri in 0 ..< nr {
				va, ea, _ := column_get(ca, ri, bool)
				vb, eb, _ := column_get(cb, ri, bool)
				testing.expect(t, ea == eb,
					fmt.tprintf("%s: row %d col %d valid mismatch", label, ri, ci))
				if ea && eb {
					testing.expect(t, va == vb,
						fmt.tprintf("%s: row %d col %d bool %v != %v", label, ri, ci, va, vb))
				}
			}
		case typeid_of(string):
			for ri in 0 ..< nr {
				va, ea, _ := column_get(ca, ri, string)
				vb, eb, _ := column_get(cb, ri, string)
				testing.expect(t, ea == eb,
					fmt.tprintf("%s: row %d col %d valid mismatch", label, ri, ci))
				if ea && eb {
					testing.expect(t, va == vb,
						fmt.tprintf("%s: row %d col %d string %s != %s", label, ri, ci, va, vb))
				}
			}
		}
	}
}

// csv_generate_4col writes a 4-column CSV (id i64, score f64, ok bool, name
// string) with ~1% NULLs to a scratch file.
csv_generate_4col :: proc(t: ^testing.T, n: int, name: string) -> string {
	path := csv_path(name)
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)
	strings.write_string(&sb, "id,score,ok,name\n")
	for i in 0 ..< n {
		score := f64(i % 1000) / 8.0
		ok := i % 2 == 0
		if i % 97 == 0 {
			fmt.sbprintf(&sb, "%d,,%t,name%07d\n", i, ok, i)
		} else {
			fmt.sbprintf(&sb, "%d,%.4f,%t,name%07d\n", i, score, ok, i)
		}
	}
	if err := os.write_entire_file_from_string(path, strings.to_string(sb)); err != os.ERROR_NONE {
		testing.expect(t, false, fmt.tprintf("write %s: %v", name, err))
	}
	return path
}

@(test)
csv_parallel_correctness_100 :: proc(t: ^testing.T) {
	path := csv_generate_4col(t, 100, "parallel_100")
	defer csv_cleanup(t, path)

	par, p_err := csv_parallel(path)
	testing.expect(t, p_err == .None, fmt.tprintf("parallel: %v", p_err))
	defer dataframe_destroy(&par)

	ser, s_err := csv_sequential(path)
	testing.expect(t, s_err == .None, fmt.tprintf("sequential: %v", s_err))
	defer dataframe_destroy(&ser)

	csv_expect_eq(t, &ser, &par, "100 rows")
}

@(test)
csv_parallel_correctness_10k :: proc(t: ^testing.T) {
	path := csv_generate_4col(t, 10_000, "parallel_10k")
	defer csv_cleanup(t, path)

	par, p_err := csv_parallel(path)
	testing.expect(t, p_err == .None, fmt.tprintf("parallel: %v", p_err))
	defer dataframe_destroy(&par)

	ser, s_err := csv_sequential(path)
	testing.expect(t, s_err == .None, fmt.tprintf("sequential: %v", s_err))
	defer dataframe_destroy(&ser)

	csv_expect_eq(t, &ser, &par, "10K rows")
}

@(test)
csv_parallel_correctness_100k :: proc(t: ^testing.T) {
	path := csv_generate_4col(t, 100_000, "parallel_100k")
	defer csv_cleanup(t, path)

	par, p_err := csv_parallel(path)
	testing.expect(t, p_err == .None, fmt.tprintf("parallel: %v", p_err))
	defer dataframe_destroy(&par)

	ser, s_err := csv_sequential(path)
	testing.expect(t, s_err == .None, fmt.tprintf("sequential: %v", s_err))
	defer dataframe_destroy(&ser)

	csv_expect_eq(t, &ser, &par, "100K rows")
}

@(test)
csv_parallel_correctness_1m :: proc(t: ^testing.T) {
	path := csv_generate_4col(t, 1_000_000, "parallel_1m")
	defer csv_cleanup(t, path)

	par, p_err := csv_parallel(path)
	testing.expect(t, p_err == .None, fmt.tprintf("parallel: %v", p_err))
	defer dataframe_destroy(&par)

	ser, s_err := csv_sequential(path)
	testing.expect(t, s_err == .None, fmt.tprintf("sequential: %v", s_err))
	defer dataframe_destroy(&ser)

	csv_expect_eq(t, &ser, &par, "1M rows")
}

@(test)
csv_parallel_all_strings :: proc(t: ^testing.T) {
	path := csv_path("parallel_strings")
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)
	strings.write_string(&sb, "a,b,c\n")
	for i in 0 ..< 50_000 {
		fmt.sbprintf(&sb, "foo%05d,bar%05d,baz%05d\n", i, i, i)
	}
	if err := os.write_entire_file_from_string(path, strings.to_string(sb)); err != os.ERROR_NONE {
		testing.expect(t, false, fmt.tprintf("write: %v", err))
	}
	defer csv_cleanup(t, path)

	par, p_err := csv_parallel(path)
	testing.expect(t, p_err == .None, fmt.tprintf("parallel: %v", p_err))
	defer dataframe_destroy(&par)

	ser, s_err := csv_sequential(path)
	testing.expect(t, s_err == .None, fmt.tprintf("sequential: %v", s_err))
	defer dataframe_destroy(&ser)

	csv_expect_eq(t, &ser, &par, "all strings")
}

@(test)
csv_parallel_all_nulls :: proc(t: ^testing.T) {
	path := csv_path("parallel_nulls")
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)
	strings.write_string(&sb, "a,b\n")
	for i in 0 ..< 50_000 {
		fmt.sbprintf(&sb, ",\n")
	}
	if err := os.write_entire_file_from_string(path, strings.to_string(sb)); err != os.ERROR_NONE {
		testing.expect(t, false, fmt.tprintf("write: %v", err))
	}
	defer csv_cleanup(t, path)

	par, p_err := csv_parallel(path)
	testing.expect(t, p_err == .None, fmt.tprintf("parallel: %v", p_err))
	defer dataframe_destroy(&par)

	ser, s_err := csv_sequential(path)
	testing.expect(t, s_err == .None, fmt.tprintf("sequential: %v", s_err))
	defer dataframe_destroy(&ser)

	csv_expect_eq(t, &ser, &par, "all nulls")
}

@(test)
csv_parallel_quoted_newlines :: proc(t: ^testing.T) {
	path := csv_path("parallel_quoted")
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)
	strings.write_string(&sb, "id,text\n")
	for i in 0 ..< 20_000 {
		if i % 500 == 0 {
			fmt.sbprintf(&sb, "%d,\"line1\nline2\nline3\"\n", i)
		} else {
			fmt.sbprintf(&sb, "%d,text%05d\n", i, i)
		}
	}
	if err := os.write_entire_file_from_string(path, strings.to_string(sb)); err != os.ERROR_NONE {
		testing.expect(t, false, fmt.tprintf("write: %v", err))
	}
	defer csv_cleanup(t, path)

	par, p_err := csv_parallel(path)
	testing.expect(t, p_err == .None, fmt.tprintf("parallel: %v", p_err))
	defer dataframe_destroy(&par)

	ser, s_err := csv_sequential(path)
	testing.expect(t, s_err == .None, fmt.tprintf("sequential: %v", s_err))
	defer dataframe_destroy(&ser)

	csv_expect_eq(t, &ser, &par, "quoted newlines")
}

@(test)
csv_parallel_with_schema :: proc(t: ^testing.T) {
	path := csv_generate_4col(t, 100_000, "parallel_schema")
	defer csv_cleanup(t, path)

	schema, s_err := schema_create([]Field{
			{"id", typeid_of(i64)},
			{"score", typeid_of(f64)},
			{"ok", typeid_of(bool)},
			{"name", typeid_of(string)},
		},
	)
	testing.expect(t, s_err == .None, fmt.tprintf("schema_create: %v", s_err))
	defer schema_destroy(&schema)

	par, p_err := dataframe_read_csv_with_schema(path, schema)
	testing.expect(t, p_err == .None, fmt.tprintf("parallel schema: %v", p_err))
	defer dataframe_destroy(&par)

	// Verify schema-constrained read gives same data as inferred.
	ser, s_err2 := csv_sequential(path)
	testing.expect(t, s_err2 == .None, fmt.tprintf("sequential: %v", s_err2))
	defer dataframe_destroy(&ser)

	csv_expect_eq(t, &ser, &par, "with schema")
}

@(test)
csv_parallel_with_columns :: proc(t: ^testing.T) {
	path := csv_generate_4col(t, 100_000, "parallel_keep")
	defer csv_cleanup(t, path)

	par, p_err := dataframe_read_csv_with_columns(path, []string{"id", "name"})
	testing.expect(t, p_err == .None, fmt.tprintf("parallel keep: %v", p_err))
	defer dataframe_destroy(&par)

	testing.expect(t, dataframe_num_cols(&par) == 2, "2 columns")
	testing.expect(t, dataframe_num_rows(&par) == 100_000, "100K rows")

	// Verify values are correct for the kept columns.
	id_col := dataframe_get_column(&par, "id") or_else nil
	testing.expect(t, id_col != nil, "id col exists")
	if id_col != nil {
		v, valid, err := column_get(id_col, 0, i64)
		testing.expect(t, err == .None, "get id[0]")
		testing.expect(t, valid, "id[0] valid")
		testing.expect(t, v == 0, fmt.tprintf("id[0] = %d", v))

		v, valid, err = column_get(id_col, 1, i64)
		testing.expect(t, err == .None, "get id[1]")
		testing.expect(t, valid, "id[1] valid")
		testing.expect(t, v == 1, fmt.tprintf("id[1] = %d", v))
	}
}
