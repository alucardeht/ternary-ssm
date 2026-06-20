import ctypes
import os
import torch


class TernaryMatmulKernel:

    def __init__(self, dll_path=None):
        if dll_path is None:
            dll_path = os.path.join(os.path.dirname(__file__), 'ternary_matmul.dll')
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"CUDA kernel DLL not found: {dll_path}")

        self.lib = ctypes.CDLL(dll_path)
        self._setup_functions()

    def _setup_functions(self):
        if hasattr(self.lib, 'launch_ternary_matmul'):
            self.lib.launch_ternary_matmul.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            self.lib.launch_ternary_matmul.restype = None
            self._has_v0 = True
        else:
            self._has_v0 = False

        if hasattr(self.lib, 'launch_ternary_matmul_int8'):
            self.lib.launch_ternary_matmul_int8.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            self.lib.launch_ternary_matmul_int8.restype = None
            self._has_v1 = True
        else:
            self._has_v1 = False

    def matmul(self, x, packed_layer):
        from .pack_weights import quantize_activations_int8

        orig_shape = x.shape[:-1]
        K = x.shape[-1]
        x_flat = x.reshape(-1, K).contiguous()
        M = x_flat.shape[0]
        N = packed_layer.out_features

        K_padded = packed_layer.w_packed.shape[1] * 16
        if K < K_padded:
            x_flat = torch.nn.functional.pad(x_flat, (0, K_padded - K))

        y = torch.zeros(M, N, dtype=torch.float32, device=x.device)

        if self._has_v1:
            x_int8, act_scale = quantize_activations_int8(x_flat)
            x_int8 = x_int8.contiguous()
            self.lib.launch_ternary_matmul_int8(
                x_int8.data_ptr(),
                packed_layer.w_packed.data_ptr(),
                packed_layer.scales.data_ptr(),
                ctypes.c_float(act_scale),
                packed_layer.bias.data_ptr() if packed_layer.bias is not None else None,
                y.data_ptr(),
                M, N, K_padded,
                packed_layer.group_size,
            )
        elif self._has_v0:
            return self._fallback_matmul(x, packed_layer)
        else:
            return self._fallback_matmul(x, packed_layer)

        torch.cuda.synchronize()
        return y.reshape(*orig_shape, N)

    def _fallback_matmul(self, x, packed_layer):
        from .pack_weights import unpack_ternary_weights

        orig_shape = x.shape[:-1]
        K = x.shape[-1]

        ternary = unpack_ternary_weights(packed_layer).to(x.device)

        N = packed_layer.out_features
        gs = packed_layer.group_size
        K_padded = ternary.shape[1]
        num_groups = K_padded // gs

        scales_expanded = packed_layer.scales.to(x.device)
        scales_full = scales_expanded.repeat_interleave(gs, dim=1)[:, :K_padded]

        w_dequantized = ternary * scales_full

        x_padded = x.reshape(-1, K)
        if K < K_padded:
            x_padded = torch.nn.functional.pad(x_padded, (0, K_padded - K))

        y = torch.nn.functional.linear(
            x_padded,
            w_dequantized,
            packed_layer.bias.to(x.device) if packed_layer.bias is not None else None
        )

        return y.reshape(*orig_shape, N)
