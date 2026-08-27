#+build !linux
#+build !darwin
package odinarrow

// Platforms without the unix mmap path: signal the IPC reader to fall back to
// reading the whole file into memory.  The two #+build lines AND together, so
// this file is compiled only when the target is neither linux nor darwin.

_ipc_map_file :: proc(path: string) -> (data: []u8, unmap: proc(_: []u8), ok: bool) {
	return
}
