#!/usr/bin/env python3
import os
import sys
import subprocess
import struct
import argparse
from pathlib import Path


def find_glslc():
    candidates = [
        "glslc",
        "glslangValidator",
    ]

    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk:
        sdk_bin = os.path.join(vulkan_sdk, "Bin")
        if os.path.isdir(sdk_bin):
            for name in candidates:
                path = os.path.join(sdk_bin, name)
                if os.path.isfile(path):
                    return path
                if sys.platform == "win32":
                    path_exe = path + ".exe"
                    if os.path.isfile(path_exe):
                        return path_exe

    for name in candidates:
        result = subprocess.run(
            ["where" if sys.platform == "win32" else "which", name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]

    return None


def compile_shader(glslc_path, comp_file, spv_file):
    cmd = [glslc_path, str(comp_file), "-o", str(spv_file)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"Error compiling {comp_file}:")
            print(result.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"Timeout compiling {comp_file}")
        return False
    except FileNotFoundError:
        print(f"Compiler not found: {glslc_path}")
        return False


def spv_to_c_array(spv_file, var_name):
    try:
        with open(spv_file, "rb") as f:
            data = f.read()
    except IOError as e:
        print(f"Error reading {spv_file}: {e}")
        return None

    if len(data) % 4 != 0:
        print(f"SPV file size not multiple of 4: {spv_file}")
        return None

    words = struct.unpack(f"<{len(data)//4}I", data)

    c_code = f"static const uint32_t {var_name}[] = {{\n"

    for i, word in enumerate(words):
        if i % 8 == 0:
            c_code += "    "
        c_code += f"0x{word:08x}"
        if i < len(words) - 1:
            c_code += ", "
        if i % 8 == 7:
            c_code += "\n"

    if len(words) % 8 != 0:
        c_code += "\n"

    c_code += "};\n"
    c_code += f"static const size_t {var_name}_size = sizeof({var_name});\n\n"

    return c_code


def main():
    parser = argparse.ArgumentParser(
        description="Compile GLSL shaders to SPIR-V and embed in C header"
    )
    parser.add_argument(
        "--shader-dir",
        default="src/engine/shaders",
        help="Directory containing .comp files",
    )
    parser.add_argument(
        "--output", default="src/engine/include/vulkan_shaders.h", help="Output header file"
    )
    parser.add_argument(
        "--glslc", default=None, help="Path to glslc compiler (auto-detect if not specified)"
    )

    args = parser.parse_args()

    shader_dir = Path(args.shader_dir)
    if not shader_dir.is_dir():
        print(f"Shader directory not found: {shader_dir}")
        return 1

    glslc_path = args.glslc or find_glslc()
    if not glslc_path:
        print(
            "glslc not found. Please install Vulkan SDK or specify --glslc path."
        )
        print("  https://vulkan.lunarg.com/sdk/home")
        return 1

    print(f"Using compiler: {glslc_path}")

    shader_files = sorted(shader_dir.glob("*.comp"))
    if not shader_files:
        print(f"No .comp files found in {shader_dir}")
        return 1

    print(f"Found {len(shader_files)} shaders")

    compiled_shaders = {}
    tmp_spv_files = []

    for comp_file in shader_files:
        spv_file = comp_file.with_suffix(".spv")
        var_name = f"shader_{comp_file.stem}_spv"

        print(f"  {comp_file.name} -> {var_name}...", end=" ")
        sys.stdout.flush()

        if not compile_shader(glslc_path, comp_file, spv_file):
            return 1

        c_code = spv_to_c_array(spv_file, var_name)
        if not c_code:
            return 1

        compiled_shaders[var_name] = c_code
        tmp_spv_files.append(spv_file)
        print("OK")

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("#ifndef VULKAN_SHADERS_H\n")
        f.write("#define VULKAN_SHADERS_H\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <stddef.h>\n\n")

        f.write("/* Auto-generated SPIR-V shader bytecode */\n\n")

        for var_name, c_code in compiled_shaders.items():
            f.write(c_code)

        f.write("#endif /* VULKAN_SHADERS_H */\n")

    print(f"\nGenerated: {output_file}")

    for spv_file in tmp_spv_files:
        try:
            spv_file.unlink()
        except OSError:
            pass

    print(f"Compiled {len(compiled_shaders)} shaders successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
