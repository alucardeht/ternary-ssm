"""Cross-platform C compiler for WPH (Weight Pattern Hashing) ternary GEMV kernel."""

import os
import sys
import subprocess
import platform
from pathlib import Path


def get_kernel_dir() -> Path:
    return Path(__file__).parent


def get_output_filename() -> str:
    system = platform.system()
    if system == 'Windows':
        return 'wph_gemv.dll'
    elif system == 'Darwin':
        return 'wph_gemv.dylib'
    else:
        return 'wph_gemv.so'


def source_newer_than_binary(src: Path, binary: Path) -> bool:
    if not binary.exists():
        return True
    return src.stat().st_mtime > binary.stat().st_mtime


def compile_with_gcc() -> bool:
    kernel_dir = get_kernel_dir()
    src_file = kernel_dir / 'wph_gemv.c'
    output_file = kernel_dir / get_output_filename()

    if not source_newer_than_binary(src_file, output_file):
        print(f"[Compile] {output_file.name} up-to-date, skipping")
        return True

    cmd = [
        'gcc',
        '-O3',
        '-march=native',
        '-shared',
        '-fPIC',
        str(src_file),
        '-o', str(output_file)
    ]

    print(f"[Compile] GCC: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[Compile] GCC failed:\n{result.stderr}")
        return False

    print(f"[Compile] Successfully compiled {output_file.name}")
    return True


def compile_with_msvc() -> bool:
    kernel_dir = get_kernel_dir()
    src_file = kernel_dir / 'wph_gemv.c'
    output_file = kernel_dir / get_output_filename()

    if not source_newer_than_binary(src_file, output_file):
        print(f"[Compile] {output_file.name} up-to-date, skipping")
        return True

    obj_file = kernel_dir / 'wph_gemv.obj'

    cmd = [
        'cl.exe',
        '/O2',
        '/LD',
        f'/Fo{obj_file}',
        f'/Fe{output_file}',
        str(src_file)
    ]

    print(f"[Compile] MSVC: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[Compile] MSVC failed:\n{result.stderr}")
        return False

    if obj_file.exists():
        obj_file.unlink()

    print(f"[Compile] Successfully compiled {output_file.name}")
    return True


def compile():
    print("[Compile] WPH (Weight Pattern Hashing) ternary GEMV kernel")

    system = platform.system()

    if system == 'Windows':
        try:
            return compile_with_msvc()
        except FileNotFoundError:
            print("[Compile] cl.exe not found, trying MinGW gcc...")
            return compile_with_gcc()
    else:
        return compile_with_gcc()


if __name__ == '__main__':
    success = compile()
    sys.exit(0 if success else 1)
