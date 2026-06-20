"""NVRTC-based CUDA kernel compiler. Compiles CUDA source at runtime without MSVC."""

import ctypes
import os
import glob
import torch
from pathlib import Path


class NVRTCKernel:
    """Compiles CUDA source via NVRTC and launches via CUDA Driver API."""

    def __init__(self):
        self._nvrtc = None
        self._cuda = None
        self._module = None
        self._functions = {}
        self._available = False
        self._load_libraries()

    def _find_nvrtc_dll(self) -> str:
        """Find nvrtc DLL in CUDA installation."""
        cuda_path = os.environ.get('CUDA_PATH', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8')

        bin_path = os.path.join(cuda_path, 'bin')
        if os.path.exists(bin_path):
            dlls = glob.glob(os.path.join(bin_path, 'nvrtc64_*.dll'))
            if dlls:
                return sorted(dlls)[-1]

        raise FileNotFoundError(f"NVRTC DLL not found in {bin_path}")

    def _load_libraries(self):
        """Load NVRTC and CUDA Driver API libraries."""
        try:
            nvrtc_dll = self._find_nvrtc_dll()
            self._nvrtc = ctypes.CDLL(nvrtc_dll)
            self._cuda = ctypes.CDLL('nvcuda.dll')

            self._setup_nvrtc_functions()
            self._setup_cuda_functions()

            # Initialize CUDA Driver API
            self._cuda.cuInit(0)

            self._available = True
        except Exception as e:
            print(f"Warning: Failed to load NVRTC/CUDA libraries: {e}")
            self._available = False

    def _setup_nvrtc_functions(self):
        """Setup NVRTC function signatures."""
        self._nvrtc.nvrtcCreateProgram.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
        ]
        self._nvrtc.nvrtcCreateProgram.restype = ctypes.c_int

        self._nvrtc.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self._nvrtc.nvrtcCompileProgram.restype = ctypes.c_int

        self._nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._nvrtc.nvrtcGetPTXSize.restype = ctypes.c_int

        self._nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._nvrtc.nvrtcGetPTX.restype = ctypes.c_int

        self._nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.c_void_p]
        self._nvrtc.nvrtcDestroyProgram.restype = ctypes.c_int

        self._nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int

        self._nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int

    def _setup_cuda_functions(self):
        """Setup CUDA Driver API function signatures."""
        self._cuda.cuModuleLoadData.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._cuda.cuModuleLoadData.restype = ctypes.c_int

        self._cuda.cuModuleGetFunction.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
        self._cuda.cuModuleGetFunction.restype = ctypes.c_int

        self._cuda.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        self._cuda.cuLaunchKernel.restype = ctypes.c_int

        self._cuda.cuCtxGetCurrent.argtypes = [ctypes.c_void_p]
        self._cuda.cuCtxGetCurrent.restype = ctypes.c_int

        self._cuda.cuCtxSynchronize.argtypes = []
        self._cuda.cuCtxSynchronize.restype = ctypes.c_int

        self._cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
        self._cuda.cuModuleUnload.restype = ctypes.c_int

        self._cuda.cuDeviceGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self._cuda.cuDeviceGetAttribute.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._available

    def _get_gpu_arch(self) -> str:
        """Auto-detect GPU compute capability (e.g., 'sm_89')."""
        if not self._available:
            raise RuntimeError("NVRTC/CUDA not available")

        try:
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

            major = ctypes.c_int()
            minor = ctypes.c_int()

            res_major = self._cuda.cuDeviceGetAttribute(
                ctypes.byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0
            )
            res_minor = self._cuda.cuDeviceGetAttribute(
                ctypes.byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0
            )

            if res_major == 0 and res_minor == 0:
                return f"sm_{major.value}{minor.value}"
            else:
                return "sm_89"
        except Exception as e:
            print(f"Warning: Failed to detect GPU architecture: {e}. Using sm_89.")
            return "sm_89"

    def compile(self, source: str, arch: str = None) -> bool:
        """Compile CUDA source to PTX and load as module.

        Args:
            source: CUDA C++ source code string
            arch: GPU architecture (e.g. "sm_89"). Auto-detected if None.

        Returns:
            True if compilation succeeded
        """
        if not self._available:
            print("Error: NVRTC/CUDA not available")
            return False

        if arch is None:
            arch = self._get_gpu_arch()

        if self._module is not None:
            self._cuda.cuModuleUnload(self._module)
            self._module = None
            self._functions = {}

        try:
            program = ctypes.c_void_p()
            source_bytes = source.encode('utf-8')
            program_name = b'cuda_kernel'

            res = self._nvrtc.nvrtcCreateProgram(
                ctypes.byref(program),
                source_bytes,
                program_name,
                0, None, None
            )
            if res != 0:
                raise RuntimeError(f"nvrtcCreateProgram failed: error {res}")

            compile_options = [f'--gpu-architecture={arch}'.encode('utf-8')]
            compile_options_ptrs = (ctypes.c_char_p * len(compile_options))(*compile_options)

            res = self._nvrtc.nvrtcCompileProgram(program, len(compile_options), compile_options_ptrs)

            log_size = ctypes.c_size_t()
            self._nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))

            if log_size.value > 1:
                log_buffer = ctypes.create_string_buffer(log_size.value)
                self._nvrtc.nvrtcGetProgramLog(program, log_buffer)
                log_text = log_buffer.value.decode('utf-8', errors='ignore')
                if res != 0:
                    print(f"Compilation log:\n{log_text}")

            if res != 0:
                self._nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
                raise RuntimeError(f"nvrtcCompileProgram failed: error {res}")

            ptx_size = ctypes.c_size_t()
            res = self._nvrtc.nvrtcGetPTXSize(program, ctypes.byref(ptx_size))
            if res != 0:
                self._nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
                raise RuntimeError(f"nvrtcGetPTXSize failed: error {res}")

            ptx_buffer = ctypes.create_string_buffer(ptx_size.value)
            res = self._nvrtc.nvrtcGetPTX(program, ptx_buffer)
            if res != 0:
                self._nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
                raise RuntimeError(f"nvrtcGetPTX failed: error {res}")

            # Use .value (null-terminated) not .raw (may have padding)
            ptx_data = ptx_buffer.value

            # Ensure CUDA context exists (PyTorch may not have initialized yet)
            if torch.cuda.is_available():
                torch.cuda.init()

            module = ctypes.c_void_p()
            res = self._cuda.cuModuleLoadData(ctypes.byref(module), ptx_data)
            if res != 0:
                raise RuntimeError(f"cuModuleLoadData failed: error {res}")

            self._module = module
            self._functions = {}

            self._nvrtc.nvrtcDestroyProgram(ctypes.byref(program))

            return True

        except Exception as e:
            print(f"Compilation error: {e}")
            return False

    def compile_file(self, cu_path: str, arch: str = None) -> bool:
        """Compile a .cu file."""
        try:
            with open(cu_path, 'r') as f:
                source = f.read()
            return self.compile(source, arch)
        except Exception as e:
            print(f"Error reading CUDA file: {e}")
            return False

    def get_function(self, name: str):
        """Get a CUfunction handle by name. Cached."""
        if self._module is None:
            raise RuntimeError("No CUDA module loaded. Call compile() first.")

        if name in self._functions:
            return self._functions[name]

        func = ctypes.c_void_p()
        name_bytes = name.encode('utf-8')
        res = self._cuda.cuModuleGetFunction(ctypes.byref(func), self._module, name_bytes)

        if res != 0:
            raise RuntimeError(f"cuModuleGetFunction failed for '{name}': error {res}")

        self._functions[name] = func
        return func

    def launch(self, func_name: str, grid: tuple, block: tuple,
               args: list, shared_mem: int = 0, stream: int = 0):
        """Launch a kernel.

        Args:
            func_name: Name of the __global__ function
            grid: (gridDimX, gridDimY, gridDimZ)
            block: (blockDimX, blockDimY, blockDimZ)
            args: List of kernel arguments. Each should be:
                  - torch.Tensor (passes data_ptr as void*)
                  - int (passes as c_int)
                  - float (passes as c_float)
            shared_mem: Dynamic shared memory bytes
            stream: CUDA stream (0 = default)
        """
        func = self.get_function(func_name)

        arg_ctypes = []
        arg_ptrs = []

        for arg in args:
            if isinstance(arg, torch.Tensor):
                ptr = ctypes.c_void_p(arg.data_ptr())
                arg_ctypes.append(ptr)
                arg_ptrs.append(ctypes.addressof(ptr))
            elif isinstance(arg, int):
                val = ctypes.c_int(arg)
                arg_ctypes.append(val)
                arg_ptrs.append(ctypes.addressof(val))
            elif isinstance(arg, float):
                val = ctypes.c_float(arg)
                arg_ctypes.append(val)
                arg_ptrs.append(ctypes.addressof(val))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        if arg_ptrs:
            kernel_args = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)
        else:
            kernel_args = None

        res = self._cuda.cuLaunchKernel(
            func,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem,
            ctypes.c_void_p(stream),
            kernel_args,
            None
        )

        if res != 0:
            raise RuntimeError(f"cuLaunchKernel failed: error {res}")

    def __del__(self):
        """Cleanup: unload module."""
        if self._module is not None and self._cuda is not None:
            try:
                self._cuda.cuModuleUnload(self._module)
            except Exception:
                pass
