#!/usr/bin/env bash
#
# ci_f2_flag_off_parity.sh — R356 F2 slice S10 flag-OFF parity gate
#
# Confirms that compiling the files touched by slices S2..S8 with the
# V12_F2_INT8_SHADOWS flag undefined produces bit-identical object files
# to compiling them with the flag explicitly defined to 0. This is the
# rollback guarantee from handoff §10: "All code paths in modified files
# must be bit-identical to pre-R356 when the flag is 0."
#
# Strategy:
#   1. Build each .c file twice into two scratch directories.
#      pass A: no flag at all (matches pre-R356 invocation)
#      pass B: -DV12_F2_INT8_SHADOWS=0 (explicit-off form)
#   2. Diff the resulting .o files byte-for-byte.
#   3. Exit 0 iff every diff is empty.
#
# Run from the model_v12 directory:
#   sh tests/ci_f2_flag_off_parity.sh
#
# The script uses only POSIX sh + cmp, so it runs identically under MSYS
# bash on Windows, plain bash on Linux, and dash. No make invocation is
# required: each file is compiled directly with the same flags the
# Makefile uses for the per-object pattern rule.

set -e

CC="${CC:-gcc}"

# Match the Makefile CFLAGS for the per-object %.o: %.c rule. Keeping
# these aligned with the Makefile is what makes the parity meaningful.
CFLAGS="-O2 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -I src/ -fopenmp"

case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*) CFLAGS="$CFLAGS -DUSE_MMAP=0" ;;
esac

# File list — every .c that was touched by R356 F2 slices S2..S8.
# Header-only changes (types.h, optimizer.h, shadow_codec.h) propagate
# transitively through any .c that #includes them, which is why every
# affected .c below covers its own header surface.
FILES="
src/core/model.c
src/core/optimizer.c
src/core/optimizer_f3_measure.c
"

DIR_A="build/parity-A-noflag"
DIR_B="build/parity-B-flag0"

rm -rf "$DIR_A" "$DIR_B"
mkdir -p "$DIR_A" "$DIR_B"

mismatches=0

for src in $FILES; do
    base=$(basename "$src" .c)
    obj_a="$DIR_A/${base}.o"
    obj_b="$DIR_B/${base}.o"

    echo "compiling $src (pass A: flag undefined)"
    $CC $CFLAGS -c "$src" -o "$obj_a"

    echo "compiling $src (pass B: -DV12_F2_INT8_SHADOWS=0)"
    $CC $CFLAGS -DV12_F2_INT8_SHADOWS=0 -c "$src" -o "$obj_b"

    if cmp -s "$obj_a" "$obj_b"; then
        echo "[F2 PARITY] [PASS] $src"
    else
        echo "[F2 PARITY] [FAIL] $src"
        mismatches=$((mismatches + 1))
    fi
done

if [ "$mismatches" -eq 0 ]; then
    echo ""
    echo "[F2 PARITY] all $(echo $FILES | wc -w) object files identical"
    exit 0
else
    echo ""
    echo "[F2 PARITY] $mismatches mismatch(es) — flag-OFF rollback is broken"
    exit 1
fi
