#include <cstdint>
#include <cuda_runtime.h>

extern "C" {

__global__ void ternary_matmul_kernel(
    const float* __restrict__ x,          // (M, K)
    const uint8_t* __restrict__ w_packed,  // (N, K_packed)
    const float* __restrict__ scales,      // (total_groups,)
    const float* __restrict__ bias,        // (N,) or nullptr
    float* __restrict__ y,                // (M, N)
    int M, int N, int K, int group_size
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || n >= N) return;

    float sum = 0.0f;
    int K_packed = (K + 3) / 4;
    int weight_row_start = n * K;

    for (int k = 0; k < K; k += 4) {
        uint8_t packed = w_packed[n * K_packed + k / 4];
        int group_idx = (weight_row_start + k) / group_size;
        float scale = scales[group_idx];

        int remaining = K - k;
        int count = remaining < 4 ? remaining : 4;

        for (int j = 0; j < count; j++) {
            uint8_t val = (packed >> (j * 2)) & 0x3;
            float x_val = x[m * K + k + j];
            if (val == 1) {
                sum += x_val * scale;
            } else if (val == 2) {
                sum -= x_val * scale;
            }
        }
    }

    if (bias != nullptr) sum += bias[n];
    y[m * N + n] = sum;
}

__declspec(dllexport) void launch_ternary_matmul(
    const float* x, const uint8_t* w_packed, const float* scales,
    const float* bias, float* y,
    int M, int N, int K, int group_size,
    int block_x, int block_y
) {
    dim3 block(block_x, block_y);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    ternary_matmul_kernel<<<grid, block>>>(
        x, w_packed, scales, bias, y, M, N, K, group_size
    );
}

#define WARPS_PER_BLOCK 8
#define WARP_SIZE 32

__global__ void ternary_matmul_int8_kernel(
    const int8_t* __restrict__ x_int8,     // (M, K) INT8 quantized activations
    const uint32_t* __restrict__ w_packed, // (N, K_packed/4) packed uint32
    const float* __restrict__ w_scales,    // (N, num_groups) per-group scales
    float act_scale,                       // activation scale (single value)
    const float* __restrict__ bias,        // (N,) or nullptr
    float* __restrict__ y,                 // (M, N) output
    int M, int N, int K, int group_size
) {
    int n = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;
    if (n >= N) return;

    int lane = threadIdx.x;
    int K_packed = (K + 15) / 16;
    int num_groups = (K + group_size - 1) / group_size;

    float float_acc = 0.0f;

    for (int g = 0; g < num_groups; g++) {
        int k_group_start = g * group_size;
        int k_group_end = min(k_group_start + group_size, K);
        int int_acc = 0;

        for (int k = k_group_start + lane * 4; k < k_group_end; k += WARP_SIZE * 4) {
            if (k + 4 > K) break;

            int a_packed = *reinterpret_cast<const int*>(&x_int8[k]);

            int packed_idx = k / 16;
            int bit_offset = (k % 16) * 2;
            uint32_t packed_word = w_packed[n * K_packed + packed_idx];

            int8_t w[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int bits = (packed_word >> (bit_offset + i * 2)) & 0x3;
                w[i] = (bits == 1) ? 1 : ((bits == 2) ? -1 : 0);
            }
            int w_packed_i8 = *reinterpret_cast<int*>(w);

            int_acc = __dp4a(a_packed, w_packed_i8, int_acc);
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            int_acc += __shfl_xor_sync(0xffffffff, int_acc, offset);
        }

        if (lane == 0) {
            float_acc += (float)int_acc * w_scales[n * num_groups + g];
        }
    }

    if (lane == 0) {
        float result = float_acc * act_scale;
        if (bias != nullptr) result += bias[n];
        y[n] = result;
    }
}

__declspec(dllexport) void launch_ternary_matmul_int8(
    const int8_t* __restrict__ x_int8,
    const uint32_t* __restrict__ w_packed,
    const float* __restrict__ w_scales,
    float act_scale,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int M, int N, int K, int group_size
) {
    dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    ternary_matmul_int8_kernel<<<grid, block>>>(
        x_int8, w_packed, w_scales, act_scale, bias, y,
        M, N, K, group_size
    );
}

}  // extern "C"
