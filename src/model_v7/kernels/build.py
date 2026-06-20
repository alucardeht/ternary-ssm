import os
import sys
import subprocess


def build_single(cu_file, arch='sm_120'):
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    dll_file = cu_file.replace('.cu', '.dll')

    cuda_path = os.environ.get('CUDA_PATH', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8')
    nvcc = os.path.join(cuda_path, 'bin', 'nvcc.exe')

    if not os.path.exists(nvcc):
        print(f"ERROR: nvcc not found at {nvcc}")
        sys.exit(1)

    cmd = [
        nvcc,
        '-O3',
        f'-arch={arch}',
        '--shared',
        '-allow-unsupported-compiler',
        '-o', dll_file,
        cu_file,
    ]

    print(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode}):")
        print(result.stderr)
        sys.exit(1)

    print(f"SUCCESS: {dll_file}")
    return dll_file


def build(cu_file=None, arch='sm_120'):
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    if cu_file is None:
        cu_file = os.path.join(kernel_dir, 'ternary_matmul.cu')

    return build_single(cu_file, arch=arch)


def build_all(arch='sm_120'):
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    cu_files = [
        os.path.join(kernel_dir, 'ternary_matmul.cu'),
        os.path.join(kernel_dir, 'fused_ssm_step.cu'),
    ]

    results = []
    for cu_file in cu_files:
        if os.path.exists(cu_file):
            print(f"\nBuilding {os.path.basename(cu_file)}...")
            results.append(build_single(cu_file, arch=arch))
        else:
            print(f"Warning: {cu_file} not found, skipping")

    return results


if __name__ == '__main__':
    arch = sys.argv[1] if len(sys.argv) > 1 else 'sm_120'
    target = sys.argv[2] if len(sys.argv) > 2 else 'ternary'

    if target == 'all':
        build_all(arch=arch)
    elif target == 'ssm':
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        cu_file = os.path.join(kernel_dir, 'fused_ssm_step.cu')
        build(cu_file, arch=arch)
    else:
        build(arch=arch)
