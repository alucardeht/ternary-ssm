#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "runtime/vulkan_backend.h"

int main(void) {
    printf("=== Vulkan Backend Test ===\n\n");

    /* --- Step 1: Init Vulkan --- */
    VulkanContext *ctx = NULL;
    int rc = vk_init(&ctx);
    if (rc != 0 || ctx == NULL) {
        printf("[SKIP] Vulkan not available (no GPU or no driver)\n");
        return 0;
    }
    vk_print_stats(ctx);
    printf("\n");

    /* --- Step 2: Define CSR --- */
    int rows = 4, cols = 4;

    /* row_ptr_pos: [0, 2, 4, 5, 7] */
    int rpp[] = {0, 2, 4, 5, 7};
    int pi[]  = {0, 3, 1, 2, 3, 0, 1};

    /* row_ptr_neg: [0, 1, 1, 3, 5] */
    int rpn[] = {0, 1, 1, 3, 5};
    int ni[]  = {1, 0, 2, 2, 3};

    /* Input vector */
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};

    /* Expected output (CPU reference):
     * y[0] = x[0]+x[3] - x[1] = 1+4-2 = 3.0
     * y[1] = x[1]+x[2] - (nothing) = 2+3 = 5.0
     * y[2] = x[3] - (x[0]+x[2]) = 4-(1+3) = 0.0
     * y[3] = x[0]+x[1] - (x[2]+x[3]) = 1+2-(3+4) = -4.0
     */
    float expected[] = {3.0f, 5.0f, 0.0f, -4.0f};

    /* --- Step 3: Upload to GPU --- */
    printf("[TEST] Uploading CSR and input vector...\n");

    VkBufferHandle h_rpp = vk_alloc(ctx, sizeof(rpp), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    VkBufferHandle h_pi  = vk_alloc(ctx, sizeof(pi),  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    VkBufferHandle h_rpn = vk_alloc(ctx, sizeof(rpn), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    VkBufferHandle h_ni  = vk_alloc(ctx, sizeof(ni),  VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    VkBufferHandle h_x   = vk_alloc(ctx, sizeof(x),   VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    VkBufferHandle h_y   = vk_alloc(ctx, rows * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);

    if (!VK_BUFFER_VALID(h_rpp) || !VK_BUFFER_VALID(h_pi) ||
        !VK_BUFFER_VALID(h_rpn) || !VK_BUFFER_VALID(h_ni) ||
        !VK_BUFFER_VALID(h_x)   || !VK_BUFFER_VALID(h_y)) {
        printf("[FAIL] Buffer allocation failed\n");
        vk_destroy(ctx);
        return 1;
    }
    printf("[TEST] Allocated 6 GPU buffers (%.1f KB total)\n",
           (double)vk_allocated_bytes(ctx) / 1024.0);

    vk_upload(ctx, h_rpp, rpp, sizeof(rpp));
    vk_upload(ctx, h_pi,  pi,  sizeof(pi));
    vk_upload(ctx, h_rpn, rpn, sizeof(rpn));
    vk_upload(ctx, h_ni,  ni,  sizeof(ni));
    vk_upload(ctx, h_x,   x,   sizeof(x));

    /* --- Step 4: Dispatch ternary_matvec --- */
    printf("[TEST] Dispatching ternary_matvec (4x4, batch=1)...\n");

    VkTernaryCSR csr;
    csr.row_ptr_pos = h_rpp;
    csr.pos_indices = h_pi;
    csr.row_ptr_neg = h_rpn;
    csr.neg_indices = h_ni;
    csr.rows = (uint32_t)rows;
    csr.cols = (uint32_t)cols;

    rc = vk_dispatch_ternary_matvec(ctx, &csr, h_x, h_y, 1);
    if (rc != 0) {
        printf("[FAIL] Dispatch failed with rc=%d\n", rc);
        vk_destroy(ctx);
        return 1;
    }
    vk_sync(ctx);

    /* --- Step 5: Download result --- */
    float result[4] = {0};
    rc = vk_download(ctx, result, h_y, sizeof(result));
    if (rc != 0) {
        printf("[FAIL] Download failed with rc=%d\n", rc);
        vk_destroy(ctx);
        return 1;
    }

    /* --- Step 6: Compare --- */
    printf("\n[TEST] Results:\n");
    int pass = 1;
    for (int i = 0; i < rows; i++) {
        float diff = fabsf(result[i] - expected[i]);
        const char *status = (diff < 1e-5f) ? "OK" : "MISMATCH";
        printf("  y[%d] = %8.4f  (expected %8.4f)  %s\n",
               i, result[i], expected[i], status);
        if (diff >= 1e-5f) pass = 0;
    }

    printf("\n");
    if (pass) {
        printf("[PASS] Ternary matvec on GPU matches CPU reference!\n");
    } else {
        printf("[FAIL] GPU result does not match CPU reference\n");
    }

    /* --- Step 7: Larger benchmark (production-scale) --- */
    printf("\n[TEST] Production-scale benchmark: 1024x1024, ~50%% nonzero...\n");
    int big_rows = 1024, big_cols = 1024;
    int nnz_per_row = 512;
    int total_nnz = big_rows * nnz_per_row;
    int half_nnz = total_nnz / 2;

    int *big_rpp = (int *)malloc((big_rows + 1) * sizeof(int));
    int *big_pi  = (int *)malloc(half_nnz * sizeof(int));
    int *big_rpn = (int *)malloc((big_rows + 1) * sizeof(int));
    int *big_ni  = (int *)malloc(half_nnz * sizeof(int));
    float *big_x = (float *)malloc(big_cols * sizeof(float));
    float *big_y_cpu = (float *)calloc(big_rows, sizeof(float));

    srand(42);
    for (int i = 0; i <= big_rows; i++) {
        big_rpp[i] = i * (nnz_per_row / 2);
        big_rpn[i] = i * (nnz_per_row / 2);
    }
    for (int i = 0; i < half_nnz; i++) {
        big_pi[i] = rand() % big_cols;
        big_ni[i] = rand() % big_cols;
    }
    for (int i = 0; i < big_cols; i++) {
        big_x[i] = (float)(rand() % 1000) / 1000.0f;
    }

    for (int r = 0; r < big_rows; r++) {
        float acc = 0.0f;
        for (int k = big_rpp[r]; k < big_rpp[r + 1]; k++) acc += big_x[big_pi[k]];
        for (int k = big_rpn[r]; k < big_rpn[r + 1]; k++) acc -= big_x[big_ni[k]];
        big_y_cpu[r] = acc;
    }

    VkBufferHandle bh_rpp = vk_alloc(ctx, (big_rows + 1) * sizeof(int), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    VkBufferHandle bh_pi  = vk_alloc(ctx, half_nnz * sizeof(int), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    VkBufferHandle bh_rpn = vk_alloc(ctx, (big_rows + 1) * sizeof(int), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    VkBufferHandle bh_ni  = vk_alloc(ctx, half_nnz * sizeof(int), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    VkBufferHandle bh_x   = vk_alloc(ctx, big_cols * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST);
    VkBufferHandle bh_y   = vk_alloc(ctx, big_rows * sizeof(float), VK_BUF_STORAGE | VK_BUF_TRANSFER_DST | VK_BUF_TRANSFER_SRC);

    vk_upload(ctx, bh_rpp, big_rpp, (big_rows + 1) * sizeof(int));
    vk_upload(ctx, bh_pi,  big_pi,  half_nnz * sizeof(int));
    vk_upload(ctx, bh_rpn, big_rpn, (big_rows + 1) * sizeof(int));
    vk_upload(ctx, bh_ni,  big_ni,  half_nnz * sizeof(int));
    vk_upload(ctx, bh_x,   big_x,   big_cols * sizeof(float));

    VkTernaryCSR big_csr;
    big_csr.row_ptr_pos = bh_rpp;
    big_csr.pos_indices = bh_pi;
    big_csr.row_ptr_neg = bh_rpn;
    big_csr.neg_indices = bh_ni;
    big_csr.rows = (uint32_t)big_rows;
    big_csr.cols = (uint32_t)big_cols;

    vk_dispatch_ternary_matvec(ctx, &big_csr, bh_x, bh_y, 1);
    vk_sync(ctx);

    #ifdef _WIN32
    LARGE_INTEGER freq, t1, t2;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t1);
    #endif

    int n_iters = 100;
    for (int i = 0; i < n_iters; i++) {
        vk_dispatch_ternary_matvec(ctx, &big_csr, bh_x, bh_y, 1);
    }
    vk_sync(ctx);

    #ifdef _WIN32
    QueryPerformanceCounter(&t2);
    double elapsed_ms = (double)(t2.QuadPart - t1.QuadPart) * 1000.0 / (double)freq.QuadPart;
    printf("[TEST] 100 dispatches of 1024x1024 matvec: %.2f ms (%.3f ms/dispatch)\n",
           elapsed_ms, elapsed_ms / n_iters);
    #endif

    float *big_y_gpu = (float *)malloc(big_rows * sizeof(float));
    vk_download(ctx, big_y_gpu, bh_y, big_rows * sizeof(float));

    float max_diff = 0.0f;
    int mismatches = 0;
    for (int r = 0; r < big_rows; r++) {
        float diff = fabsf(big_y_gpu[r] - big_y_cpu[r]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.01f) mismatches++;
    }
    printf("[TEST] Max abs diff: %.6f, mismatches (>0.01): %d/%d\n",
           max_diff, mismatches, big_rows);

    if (mismatches == 0 && max_diff < 0.01f) {
        printf("[PASS] Production-scale GPU matvec verified!\n");
    } else {
        printf("[FAIL] Production-scale verification failed\n");
    }

    vk_free(ctx, bh_rpp); vk_free(ctx, bh_pi);
    vk_free(ctx, bh_rpn); vk_free(ctx, bh_ni);
    vk_free(ctx, bh_x);   vk_free(ctx, bh_y);
    vk_free(ctx, h_rpp); vk_free(ctx, h_pi);
    vk_free(ctx, h_rpn); vk_free(ctx, h_ni);
    vk_free(ctx, h_x);   vk_free(ctx, h_y);

    free(big_rpp); free(big_pi); free(big_rpn); free(big_ni);
    free(big_x); free(big_y_cpu); free(big_y_gpu);

    vk_destroy(ctx);
    printf("\n[TEST] All resources cleaned up. Done.\n");
    return 0;
}
