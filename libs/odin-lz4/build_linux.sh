#!/bin/bash
set -e

# Build lz4 static libraries for Linux (x86_64)
# Includes block, HC, frame, and xxhash support.

INCLUDE="-I c/include"

# lz4 (block + frame + HC + xxhash)
gcc -c c/lz4.c      $INCLUDE -o lz4.o      -O2 -m64
gcc -c c/lz4hc.c    $INCLUDE -o lz4hc.o    -O2 -m64
gcc -c c/xxhash.c    $INCLUDE -o xxhash.o   -O2 -m64
gcc -c c/lz4frame.c  $INCLUDE -o lz4frame.o -O2 -m64
ar rcs lz4/lz4_linux_x64.a lz4.o lz4hc.o xxhash.o lz4frame.o
rm lz4.o lz4hc.o xxhash.o lz4frame.o

echo "lz4_linux_x64.a built successfully"
