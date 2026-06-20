#include "gpu_pipeline_internal.h"

#ifdef V12_HAS_VULKAN

int ensure_mprm_adam_pipeline(VulkanContext *ctx) {
    if (mprm_adam_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = mprm_adam_spv_len;
    smci.pCode = (const uint32_t *)mprm_adam_spv;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = vk_get_global_pipeline_layout(ctx);
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &mprm_adam_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

void vk_dispatch_mprm_adam(VulkanContext *ctx, VkBufferHandle param, VkBufferHandle grad,
                                  VkBufferHandle p_u, VkBufferHandle p_v, VkBufferHandle sig_m, VkBufferHandle sig_v,
                                  uint32_t rows, uint32_t cols, uint32_t r, float lr, float b1, float b2, float eps, float inv_bc1, float inv_bc2) {
    if (ensure_mprm_adam_pipeline(ctx) != 0) return;
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mprm_adam_pipeline);
    
    vk_bind_buffer_to_descriptor(ctx, 0, 0, param);
    vk_bind_buffer_to_descriptor(ctx, 0, 1, grad);
    vk_bind_buffer_to_descriptor(ctx, 0, 2, p_u);
    vk_bind_buffer_to_descriptor(ctx, 0, 3, p_v);
    vk_bind_buffer_to_descriptor(ctx, 0, 4, sig_m);
    vk_bind_buffer_to_descriptor(ctx, 0, 5, sig_v);
    
    struct {
        uint32_t rows;
        uint32_t cols;
        uint32_t r;
        float lr;
        float beta1;
        float beta2;
        float eps;
        float inv_bc1;
        float inv_bc2;
    } pc;
    pc.rows = rows;
    pc.cols = cols;
    pc.r = r;
    pc.lr = lr;
    pc.beta1 = b1;
    pc.beta2 = b2;
    pc.eps = eps;
    pc.inv_bc1 = inv_bc1;
    pc.inv_bc2 = inv_bc2;
    
    vkCmdPushConstants(cmd, vk_get_global_pipeline_layout(ctx), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t gx = (rows * cols + 255) / 256;
    vkCmdDispatch(cmd, gx, 1, 1);
}

int ensure_entangled_adam_pipeline(VulkanContext *ctx) {
    if (entangled_adam_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = entangled_adam_update_spv_len;
    smci.pCode    = (const uint32_t *)entangled_adam_update_spv;
    VkShaderModule comp;
    if (vkCreateShaderModule(device, &smci, NULL, &comp) != VK_SUCCESS) return -1;
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, vk_get_pipe_ssm_scan(ctx));
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = comp;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &entangled_adam_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, comp, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, comp, NULL);
    return 0;
}

void vk_dispatch_entangled_adam_update(VulkanContext *ctx, VkBufferHandle packed_state, VkBufferHandle grad, uint32_t N, float lr, float b1, float b2, float eps) {
    if (!VK_BUFFER_VALID(packed_state) || !VK_BUFFER_VALID(grad)) return;
    if (ensure_entangled_adam_pipeline(ctx) != 0) return;

    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }


    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, entangled_adam_pipeline);

    vk_bind_buffer_to_descriptor(ctx, slot, 0, packed_state);
    vk_bind_buffer_to_descriptor(ctx, slot, 1, grad);

    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);

    struct { uint32_t N; float lr; float b1; float b2; float eps; } pc;
    pc.N = N;
    pc.lr = lr;
    pc.b1 = b1;
    pc.b2 = b2;
    pc.eps = eps;

    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (N + 255) / 256, 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

int ensure_sp19_holo_adam_pipeline(VulkanContext *ctx) {
    if (sp19_holo_adam_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sp19_holo_adam_reduce_spv_len;
    smci.pCode    = (const uint32_t *)sp19_holo_adam_reduce_spv;
    VkShaderModule comp;
    if (vkCreateShaderModule(device, &smci, NULL, &comp) != VK_SUCCESS) return -1;
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, vk_get_pipe_ssm_scan(ctx));
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = comp;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &sp19_holo_adam_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, comp, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, comp, NULL);
    return 0;
}

void vk_dispatch_sp19_holo_adam_reduce(VulkanContext *ctx, 
    VkBufferHandle grad, VkBufferHandle holo_U, VkBufferHandle holo_V, uint32_t rank,
    VkBufferHandle shadow, VkBufferHandle sampled, VkBufferHandle grad_accum,
    VkBufferHandle m_row, VkBufferHandle m_col, VkBufferHandle v_row, VkBufferHandle v_col,
    VkBufferHandle g_row, VkBufferHandle g_col, VkBufferHandle gq_row, VkBufferHandle gq_col,
    VkBufferHandle cnt_row, VkBufferHandle cnt_col, VkBufferHandle mean_v_sums, VkBufferHandle row_map,
    uint32_t matrix_rows, uint32_t moment_rows, uint32_t cols, float lr, float inv_bc1, float inv_bc2,
    uint32_t row_map_mode, uint32_t evidence_gate, uint32_t evidence_mode, float evidence_boundary_eps, float tau,
    VkBufferHandle evidence_stats) {
    
    if (!VK_BUFFER_VALID(shadow)) return;
    if (ensure_sp19_holo_adam_pipeline(ctx) != 0) return;

    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }

    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp19_holo_adam_pipeline);

    vk_bind_buffer_to_descriptor(ctx, slot, 0, holo_U);
    vk_bind_buffer_to_descriptor(ctx, slot, 13, holo_V);
    vk_bind_buffer_to_descriptor(ctx, slot, 14, grad); // BufS
    vk_bind_buffer_to_descriptor(ctx, slot, 1, shadow);
    vk_bind_buffer_to_descriptor(ctx, slot, 2, sampled);
    vk_bind_buffer_to_descriptor(ctx, slot, 3, grad_accum);
    vk_bind_buffer_to_descriptor(ctx, slot, 4, g_row);
    vk_bind_buffer_to_descriptor(ctx, slot, 5, g_col);
    vk_bind_buffer_to_descriptor(ctx, slot, 6, gq_row);
    vk_bind_buffer_to_descriptor(ctx, slot, 7, gq_col);
    vk_bind_buffer_to_descriptor(ctx, slot, 8, cnt_row);
    vk_bind_buffer_to_descriptor(ctx, slot, 9, cnt_col);
    vk_bind_buffer_to_descriptor(ctx, slot, 10, row_map);
    vk_bind_buffer_to_descriptor(ctx, slot, 11, evidence_stats);

    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);

    struct {
        uint32_t N;
        uint32_t rank;
        uint32_t cols;
        float lr;
        float inv_bc1;
        float inv_bc2;
        uint32_t row_map_mode;
        float crystallization_tau;
        float boundary_eps;
    } pc;
    pc.N = matrix_rows * cols;
    pc.rank = rank;
    pc.cols = cols;
    pc.lr = lr;
    pc.inv_bc1 = inv_bc1;
    pc.inv_bc2 = inv_bc2;
    pc.row_map_mode = row_map_mode;
    pc.crystallization_tau = tau;
    pc.boundary_eps = evidence_boundary_eps;

    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (pc.N + 255) / 256, 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

void moments_to_gpu_flat(const GpKlMoments *src, float *dst, int ds) {
    for (int i = 0; i < ds; i++) dst[i]         = src->mean[i];
    for (int i = 0; i < ds; i++) dst[ds + i]     = src->M2[i];
    for (int i = 0; i < ds; i++) dst[2*ds + i]   = src->lag1_sum[i];
    for (int i = 0; i < ds; i++) dst[3*ds + i]   = src->lag1_prev[i];
    uint32_t n_bits;
    memcpy(&n_bits, &src->n, sizeof(uint32_t));
    dst[4*ds]     = *(const float *)&n_bits;
    dst[4*ds + 1] = src->stability_sum;
    for (int p = 0; p < 32; p++) dst[4*ds + 2  + p] = src->orth_sum[p];
    for (int p = 0; p < 32; p++) dst[4*ds + 34 + p] = (float)src->orth_pair_a[p];
    for (int p = 0; p < 32; p++) dst[4*ds + 66 + p] = (float)src->orth_pair_b[p];
}

void moments_from_gpu_flat(GpKlMoments *dst, const float *src, int ds) {
    for (int i = 0; i < ds; i++) dst->mean[i]      = src[i];
    for (int i = 0; i < ds; i++) dst->M2[i]        = src[ds + i];
    for (int i = 0; i < ds; i++) dst->lag1_sum[i]  = src[2*ds + i];
    for (int i = 0; i < ds; i++) dst->lag1_prev[i] = src[3*ds + i];
    uint32_t n_bits;
    memcpy(&n_bits, &src[4*ds], sizeof(uint32_t));
    memcpy(&dst->n, &n_bits, sizeof(int32_t));
    dst->stability_sum = src[4*ds + 1];
    for (int p = 0; p < 32; p++) dst->orth_sum[p] = src[4*ds + 2 + p];
}

void download_lm_hot_row_moments(VulkanContext *ctx,
                                        VkBufferHandle src_full,
                                        const int32_t *hot_to_vocab,
                                        float *dst_hot) {
    if (!ctx || !VK_BUFFER_VALID(src_full) || !hot_to_vocab || !dst_hot) return;

    float *full_rows = (float *)malloc((size_t)V12_VOCAB_SIZE * sizeof(float));
    if (!full_rows) return;

    vk_download(ctx, full_rows, src_full, (size_t)V12_VOCAB_SIZE * sizeof(float));
    vk_batch_checkpoint(ctx);
    for (int h = 0; h < V12_K_HOT; h++) {
        int32_t vocab_idx = hot_to_vocab[h];
        dst_hot[h] = (vocab_idx >= 0 && vocab_idx < V12_VOCAB_SIZE)
                   ? full_rows[vocab_idx]
                   : 0.0f;
    }
    free(full_rows);
}

void upload_lm_hot_row_moments(VulkanContext *ctx,
                                      VkBufferHandle dst_full,
                                      const int32_t *hot_to_vocab,
                                      const float *src_hot) {
    if (!ctx || !VK_BUFFER_VALID(dst_full) || !hot_to_vocab || !src_hot) return;

    float *full_rows = (float *)calloc((size_t)V12_VOCAB_SIZE, sizeof(float));
    if (!full_rows) return;

    for (int h = 0; h < V12_K_HOT; h++) {
        int32_t vocab_idx = hot_to_vocab[h];
        if (vocab_idx >= 0 && vocab_idx < V12_VOCAB_SIZE) {
            full_rows[vocab_idx] = src_hot[h];
        }
    }
    vk_upload(ctx, dst_full, full_rows, (size_t)V12_VOCAB_SIZE * sizeof(float));
    free(full_rows);
}


#else // !V12_HAS_VULKAN

void gpu_pipeline_offload_moments(GpuPipeline *p, Model *m, int k)                { (void)p; (void)m; (void)k; }

void gpu_optimizer_step_dense(GpuPipeline *p, Model *m, float lr, int step)       { (void)p; (void)m; (void)lr; (void)step; }

void gpu_zero_grad_accumulators(GpuPipeline *p)                                    { (void)p; }

void gpu_scale_gradients(GpuPipeline *p, float scale)                              { (void)p; (void)scale; }


#endif // V12_HAS_VULKAN
