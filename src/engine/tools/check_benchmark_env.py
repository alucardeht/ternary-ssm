#!/usr/bin/env python3

import sys
import subprocess
from pathlib import Path


def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        return True, None
    except ImportError as e:
        return False, str(e)


def check_binary(binary_path, description):
    path = Path(binary_path)
    if path.exists() and path.is_file():
        return True, str(path)
    return False, f"Not found: {binary_path}"


def check_directory(dir_path, description):
    path = Path(dir_path)
    if path.exists() and path.is_dir():
        return True, str(path)
    return False, f"Not found: {dir_path}"


def main():
    workspace = Path("C:/Users/th1rt/Documents/Workspaces")
    code_ai = workspace / "code-ai-workspace"
    llama_cpp = workspace / "llama.cpp"

    print("=" * 70)
    print("BENCHMARK ENVIRONMENT CHECKER")
    print("=" * 70)
    print()

    all_ok = True

    print("[PYTHON PACKAGES]")
    print()

    packages = [
        ("numpy", "numpy"),
        ("gguf", "gguf"),
    ]

    for package, import_name in packages:
        ok, error = check_package(package, import_name)
        status = "✓" if ok else "✗"
        print(f"  {status} {package:<20}", end="")
        if ok:
            print(" (installed)")
        else:
            print(f" (MISSING)")
            print(f"      Install: pip install {package}")
            all_ok = False
        print()

    print()
    print("[LLAMA.CPP BINARIES]")
    print()

    binaries = [
        (llama_cpp / "build/bin/llama-quantize.exe", "llama-quantize"),
        (llama_cpp / "build/bin/llama-bench.exe", "llama-bench"),
        (llama_cpp / "build/bin/llama-cli.exe", "llama-cli"),
        (llama_cpp / "build/bin/test-backend-ops.exe", "test-backend-ops"),
    ]

    for binary_path, description in binaries:
        ok, msg = check_binary(binary_path, description)
        status = "✓" if ok else "✗"
        print(f"  {status} {description:<20}", end="")
        if ok:
            print(f" ({msg})")
        else:
            print(f" (MISSING)")
            print(f"      {msg}")
            all_ok = False
        print()

    if not (llama_cpp / "build/bin/llama-quantize.exe").exists():
        print()
        print("  To build llama.cpp:")
        print("    cd C:/Users/th1rt/Documents/Workspaces/llama.cpp")
        print(
            "    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_AVX2=ON -DGGML_NATIVE=ON"
        )
        print("    cmake --build build --config Release -j")
        print()

    print()
    print("[OUR ENGINE BINARIES]")
    print()

    engine_binaries = [
        (code_ai / "src/engine/build/bench_gemv.exe", "bench_gemv"),
        (code_ai / "src/engine/build/bench_components.exe", "bench_components"),
        (code_ai / "src/engine/build/trnm_cli.exe", "trnm_cli"),
    ]

    for binary_path, description in engine_binaries:
        ok, msg = check_binary(binary_path, description)
        status = "✓" if ok else "✗"
        print(f"  {status} {description:<20}", end="")
        if ok:
            print(f" ({msg})")
        else:
            print(f" (MISSING)")
            print(f"      {msg}")
            all_ok = False
        print()

    if not (code_ai / "src/engine/build/bench_gemv.exe").exists():
        print()
        print("  To build our engine:")
        print("    cd C:/Users/th1rt/Documents/Workspaces/code-ai-workspace/src/engine")
        print("    cmake -B build -DCMAKE_BUILD_TYPE=Release")
        print("    cmake --build build --config Release")
        print()

    print()
    print("[DIRECTORIES]")
    print()

    directories = [
        (code_ai, "code-ai-workspace"),
        (llama_cpp, "llama.cpp"),
        (code_ai / "models", "models output directory"),
    ]

    for dir_path, description in directories:
        ok, msg = check_directory(dir_path, description)
        status = "✓" if ok else "✗"
        print(f"  {status} {description:<30}", end="")
        if ok:
            print(f" ({msg})")
        else:
            print(f" (MISSING)")
            if "models" in description:
                print(f"      Will be created on first run")
        print()

    print()
    print("=" * 70)

    if all_ok:
        print("✓ All dependencies ready for benchmarking")
        print()
        print("Next step:")
        print("  python src/engine/tools/create_tiny_gguf.py -o models/tiny-6m-f16.gguf")
        print()
        return 0
    else:
        print("✗ Some dependencies are missing")
        print()
        print("Install missing packages, then re-run this checker.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
