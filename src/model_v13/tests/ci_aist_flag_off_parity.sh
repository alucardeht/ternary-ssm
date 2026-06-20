#!/usr/bin/env bash
#
# ci_aist_flag_off_parity.sh — R355 AIST flag-OFF parity gate
#
# Confirms that compiling the files touched by R355 slice 2 with the
# V12_AIST_ENABLED flag undefined produces bit-identical object files
# to compiling them with the flag explicitly defined to 0. This is the
# default-OFF guarantee from r355-aist-handoff §7.
#
# Strategy:
#   1. Build each .c file twice into two scratch directories.
#      pass A: no flag at all (matches pre-R355 invocation)
#      pass B: -DV12_AIST_ENABLED=0 (explicit-off form)
#   2. Diff the resulting .o files byte-for-byte.
#   3. Exit 0 iff every diff is empty.
#
# Run from the model_v12 directory:
#   sh tests/ci_aist_flag_off_parity.sh
#
# Mirrors ci_f2_flag_off_parity.sh exactly.

set -e

CC="${CC:-gcc}"

CFLAGS="-O2 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -I src/ -fopenmp"

case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*) CFLAGS="$CFLAGS -DUSE_MMAP=0" ;;
esac

# File list — every .c that was touched by R355 slice 2. aist_gate.h is
# header-only, so transitive includes are covered by rebuilding optimizer.c.
FILES="
src/core/optimizer.c
"

DIR_A="build/parity-A-aist-noflag"
DIR_B="build/parity-B-aist-flag0"

rm -rf "$DIR_A" "$DIR_B"
mkdir -p "$DIR_A" "$DIR_B"

mismatches=0

for src in $FILES; do
    base=$(basename "$src" .c)
    obj_a="$DIR_A/${base}.o"
    obj_b="$DIR_B/${base}.o"

    echo "compiling $src (pass A: flag undefined)"
    $CC $CFLAGS -c "$src" -o "$obj_a"

    echo "compiling $src (pass B: -DV12_AIST_ENABLED=0)"
    $CC $CFLAGS -DV12_AIST_ENABLED=0 -c "$src" -o "$obj_b"

    if cmp -s "$obj_a" "$obj_b"; then
        echo "[AIST PARITY] [PASS] $src"
    else
        echo "[AIST PARITY] [FAIL] $src"
        mismatches=$((mismatches + 1))
    fi
done

if [ "$mismatches" -eq 0 ]; then
    echo ""
    echo "[AIST PARITY] all $(echo $FILES | wc -w) object files identical"
    exit 0
else
    echo ""
    echo "[AIST PARITY] $mismatches mismatch(es) — flag-OFF rollback is broken"
    exit 1
fi
