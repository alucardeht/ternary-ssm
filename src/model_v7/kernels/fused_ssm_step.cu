/*
Device-only CUDA kernel for fused SSM recurrence step.
Compiled at runtime via NVRTC (NVIDIA Runtime Compilation).
No host compiler (MSVC) required. Uses CUDA Driver API for execution.
*/

// No includes needed — NVRTC provides built-in math (expf) and intrinsics (__shfl_xor_sync)

extern "C" {

__global__ void fused_ssm_step_kernel(
    float* __restrict__ h,          // (batch, n_heads, d_state, head_dim) - IN/OUT state
    float* __restrict__ y,          // (batch, n_heads, head_dim) - OUT output
    const float* __restrict__ dt,   // (batch, n_heads) - scalar dt after softplus
    const float* __restrict__ A,    // (n_heads,) - A matrix log-space (negative values)
    const float* __restrict__ B,    // (batch, d_state) - input projection
    const float* __restrict__ C,    // (batch, d_state) - output projection
    const float* __restrict__ x,    // (batch, n_heads, head_dim) - activation (post-conv+silu)
    const float* __restrict__ D,    // (n_heads, head_dim) - skip connection weights
    int batch, int n_heads, int d_state, int head_dim
) {
    int h_idx = blockIdx.x;  // head index
    int b_idx = blockIdx.y;  // batch index

    int tx = threadIdx.x;    // thread within block, 0-31 (warp)
    int ty = threadIdx.y;    // within head_dim, 0 to (head_dim-1)

    if (h_idx >= n_heads || b_idx >= batch || ty >= head_dim) return;

    // Load scalar values per block
    int dt_idx = b_idx * n_heads + h_idx;
    float dt_val = dt[dt_idx];
    float A_val = A[h_idx];

    // Compute decay factor: dA = exp(dt * A)
    // A is in log-space (negative), so this is exponential decay
    float dA = expf(dt_val * A_val);

    // Load x_val once (same for all d_state iterations of this thread)
    int x_idx = b_idx * n_heads * head_dim + h_idx * head_dim + ty;
    float x_val = x[x_idx];

    // Accumulator for output reduction across d_state
    float partial_y = 0.0f;

    // Process d_state dimension with stride (handle d_state > 32)
    for (int s = tx; s < d_state; s += 32) {
        // Load B and C values for this d_state position
        int bc_idx = b_idx * d_state + s;
        float B_val = B[bc_idx];
        float C_val = C[bc_idx];

        // Compute input contribution to state: dt * B * x
        float dBx = dt_val * B_val * x_val;

        // Compute flat index into h tensor
        // h: (batch, n_heads, d_state, head_dim)
        // Linear: b * (n_heads * d_state * head_dim) + h_idx * (d_state * head_dim) + s * head_dim + ty
        int h_flat_idx = ((b_idx * n_heads + h_idx) * d_state + s) * head_dim + ty;

        // Load old state
        float h_old = h[h_flat_idx];

        // Update state: h_new = dA * h_old + dBx (stays in FP32)
        float h_new = dA * h_old + dBx;

        // Write back updated state
        h[h_flat_idx] = h_new;

        // Accumulate output contribution: C * h_new
        partial_y += C_val * h_new;
    }

    // Warp reduction: sum partial_y across all threads in warp (reduce across tx)
    // This reduces the d_state dimension within the warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial_y += __shfl_xor_sync(0xFFFFFFFF, partial_y, offset);
    }

    // Thread 0 of warp (tx == 0) writes final output
    if (tx == 0) {
        int out_idx = b_idx * n_heads * head_dim + h_idx * head_dim + ty;
        int D_idx = h_idx * head_dim + ty;
        float D_val = D[D_idx];

        // y = partial_y (h @ C) + D * x (skip connection)
        y[out_idx] = partial_y + D_val * x_val;
    }
}

}  // extern "C"
