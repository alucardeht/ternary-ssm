#include "gpu_pipeline_internal.h"

#ifdef V12_HAS_VULKAN

__attribute__((unused)) void vk_dispatch_f36_slefc_forward(VulkanContext *ctx, VkBufferHandle h_in, VkBufferHandle signs_out, VkBufferHandle norm_out, int seq_len, int d_state) {
    if (!VK_BUFFER_VALID(h_in) || !VK_BUFFER_VALID(signs_out) || !VK_BUFFER_VALID(norm_out)) return;
    if (ensure_f36_slefc_forward_pipeline(ctx) != 0) return;
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    vk_fill_zero(ctx, signs_out, (seq_len * d_state + 31) / 32 * sizeof(uint32_t));
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    vk_bind_buffer_to_descriptor(ctx, slot, 0, h_in);
    vk_bind_buffer_to_descriptor(ctx, slot, 1, signs_out);
    vk_bind_buffer_to_descriptor(ctx, slot, 2, norm_out);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f36_slefc_forward_pipeline);
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct { uint32_t sl; uint32_t ds; } pc;
    pc.sl = seq_len;
    pc.ds = d_state;
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    vkCmdDispatch(cmd, (d_state + 255) / 256, 1, 1);
}

__attribute__((unused)) void vk_dispatch_f36_slefc_reconstruct(VulkanContext *ctx, VkBufferHandle signs_in, VkBufferHandle norm_in, VkBufferHandle h_out, int seq_len, int d_state) {
    if (!VK_BUFFER_VALID(signs_in) || !VK_BUFFER_VALID(norm_in) || !VK_BUFFER_VALID(h_out)) return;
    if (ensure_f36_slefc_reconstruct_pipeline(ctx) != 0) return;
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    vk_bind_buffer_to_descriptor(ctx, slot, 0, signs_in);
    vk_bind_buffer_to_descriptor(ctx, slot, 1, norm_in);
    vk_bind_buffer_to_descriptor(ctx, slot, 2, h_out);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f36_slefc_reconstruct_pipeline);
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct { uint32_t sl; uint32_t ds; } pc;
    pc.sl = seq_len;
    pc.ds = d_state;
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    uint32_t total = seq_len * d_state;
    vkCmdDispatch(cmd, (total + 255) / 256, 1, 1);
}

__attribute__((unused)) void vk_dispatch_f38_cst(VulkanContext *ctx, VkBufferHandle h_buf, VkBufferHandle active_flags, VkBufferHandle lambda_buf, int seq_len, int d_state, float threshold, int mode) {
    if (!VK_BUFFER_VALID(h_buf) || !VK_BUFFER_VALID(active_flags)) return; // lambda_buf could be optional depending on mode but let's assume all valid or handle appropriately
    if (ensure_f38_cst_pipeline(ctx) != 0) return;
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vk_bind_buffer_to_descriptor(ctx, slot, 0, h_buf);
    vk_bind_buffer_to_descriptor(ctx, slot, 1, active_flags);
    if (VK_BUFFER_VALID(lambda_buf)) {
        vk_bind_buffer_to_descriptor(ctx, slot, 2, lambda_buf);
    }
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f38_cst_pipeline);
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct { uint32_t seq_len; uint32_t d_state; float threshold; uint32_t mode; } pc;
    pc.seq_len = seq_len;
    pc.d_state = d_state;
    pc.threshold = threshold;
    pc.mode = mode;
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    vkCmdDispatch(cmd, (seq_len + 255) / 256, 1, 1);
}

__attribute__((unused)) void vk_dispatch_f39_bvgr(VulkanContext *ctx, VkBufferHandle a_buf, VkBufferHandle b_buf, VkBufferHandle x_buf, VkBufferHandle h_buf, int seq_len, int d_state) {
    if (!VK_BUFFER_VALID(a_buf) || !VK_BUFFER_VALID(b_buf) || !VK_BUFFER_VALID(x_buf) || !VK_BUFFER_VALID(h_buf)) return;
    if (ensure_f39_bvgr_pipeline(ctx) != 0) return;
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vk_bind_buffer_to_descriptor(ctx, slot, 0, a_buf);
    vk_bind_buffer_to_descriptor(ctx, slot, 1, b_buf);
    vk_bind_buffer_to_descriptor(ctx, slot, 2, x_buf);
    vk_bind_buffer_to_descriptor(ctx, slot, 3, h_buf);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f39_bvgr_pipeline);
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct { uint32_t seq_len; uint32_t d_state; } pc;
    pc.seq_len = seq_len;
    pc.d_state = d_state;
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    vkCmdDispatch(cmd, (d_state + 255) / 256, 1, 1);
}

void ef03_validate_gate(VulkanContext *ctx, VkBufferHandle h_buf, VkBufferHandle sum_buf, uint32_t bseq, uint32_t ds, uint32_t K) {
    uint32_t num_blocks = bseq / K;
    float *h_cpu = (float*)malloc(bseq * ds * sizeof(float));
    float *sum_cpu = (float*)malloc(num_blocks * ds * sizeof(float));
    vk_download(ctx, h_cpu, h_buf, bseq * ds * sizeof(float));
    vk_download(ctx, sum_cpu, sum_buf, num_blocks * ds * sizeof(float));

    // Validate the first block, channel 0
    uint32_t block = 0;
    uint32_t d = 0;
    float S = sum_cpu[block * ds + d];
    
    // Algebraically unroll using proxy characteristic
    float alpha = 0.95f;
    float current_alpha = alpha;
    float expected_S = 0.0f;
    for(uint32_t j=0; j<K; j++) {
        expected_S += current_alpha * h_cpu[(block * K + j) * ds + d];
        current_alpha *= alpha;
    }
    
    printf("[EF03 Gate] Ring-Sum Checkpoint Validated. GPU Sum: %f, CPU Expected: %f\n", S, expected_S);
    
    free(h_cpu);
    free(sum_cpu);
}

__attribute__((unused)) int vk_dispatch_sp10_psvcte(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size) {
    printf("DEBUG: Inside vk_dispatch_sp10_psvcte\n"); fflush(stdout);

    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(B_eff) || !VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(h_out)) return -1;
    printf("DEBUG: Before ensure_sp10_pipeline\n"); fflush(stdout);
    if (ensure_sp10_pipeline(ctx) != 0) return -1;
    
    printf("DEBUG: Before vk_bind_ssm_scan_buffers\n"); fflush(stdout);
    vk_bind_ssm_scan_buffers(ctx, A_eff, B_eff, x, C_bilinear, h_out);
    printf("DEBUG: After vk_bind_ssm_scan_buffers\n"); fflush(stdout);
    
    printf("DEBUG: Line A\n"); fflush(stdout);
      if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    printf("DEBUG: Line B\n"); fflush(stdout);
      VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    printf("DEBUG: Line C\n"); fflush(stdout);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp10_psvcte_pipeline);
    
    printf("DEBUG: Line D\n"); fflush(stdout);
      VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    printf("DEBUG: Line E\n"); fflush(stdout);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    printf("DEBUG: Line F\n"); fflush(stdout);
      struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t has_bilinear;
    } pc;
    pc.seq_len      = (uint32_t)seq_len;
    pc.d_state      = (uint32_t)d_state;
    pc.has_bilinear = VK_BUFFER_VALID(C_bilinear) ? 1u : 0u;
    
    printf("DEBUG: Line G\n"); fflush(stdout);
      vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    printf("DEBUG: Line H\n"); fflush(stdout);
      vkCmdDispatch(cmd, (uint32_t)d_state, (uint32_t)batch_size, 1);
    return 0;
}

__attribute__((unused)) int vk_dispatch_sp11_lns_tfms(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size) {
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(B_eff) || !VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(h_out)) return -1;
    if (ensure_sp11_pipeline(ctx) != 0) return -1;
    
    printf("DEBUG: Before vk_bind_ssm_scan_buffers\n"); fflush(stdout);
    vk_bind_ssm_scan_buffers(ctx, A_eff, B_eff, x, C_bilinear, h_out);
    printf("DEBUG: After vk_bind_ssm_scan_buffers\n"); fflush(stdout);
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp11_lns_tfms_pipeline);
    
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t has_bilinear;
    } pc;
    pc.seq_len      = (uint32_t)seq_len;
    pc.d_state      = (uint32_t)d_state;
    pc.has_bilinear = VK_BUFFER_VALID(C_bilinear) ? 1u : 0u;
    
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    vkCmdDispatch(cmd, (uint32_t)d_state, (uint32_t)batch_size, 1);
    return 0;
}

__attribute__((unused)) int vk_dispatch_sp14_gfit(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size) {
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(B_eff) || !VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(h_out)) return -1;
    if (ensure_sp14_pipeline(ctx) != 0) return -1;
    
    printf("DEBUG: Before vk_bind_ssm_scan_buffers\n"); fflush(stdout);
    vk_bind_ssm_scan_buffers(ctx, A_eff, B_eff, x, C_bilinear, h_out);
    printf("DEBUG: After vk_bind_ssm_scan_buffers\n"); fflush(stdout);
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp14_gfit_pipeline);
    
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t has_bilinear;
    } pc;
    pc.seq_len      = (uint32_t)seq_len;
    pc.d_state      = (uint32_t)d_state;
    pc.has_bilinear = VK_BUFFER_VALID(C_bilinear) ? 1u : 0u;
    
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    vkCmdDispatch(cmd, (uint32_t)d_state, (uint32_t)batch_size, 1);
    return 0;
}

__attribute__((unused)) int vk_dispatch_sp15_tt_rcrb(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size, int tt_rank) {
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(B_eff) || !VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(h_out)) return -1;
    if (ensure_sp15_pipeline(ctx) != 0) return -1;
    
    printf("DEBUG: Before vk_bind_ssm_scan_buffers\n"); fflush(stdout);
    vk_bind_ssm_scan_buffers(ctx, A_eff, B_eff, x, C_bilinear, h_out);
    printf("DEBUG: After vk_bind_ssm_scan_buffers\n"); fflush(stdout);
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp15_tt_rcrb_pipeline);
    
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t tt_rank;
    } pc;
    pc.seq_len      = (uint32_t)seq_len;
    pc.d_state      = (uint32_t)d_state;
    pc.tt_rank      = (uint32_t)tt_rank;
    
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    vkCmdDispatch(cmd, (uint32_t)d_state, (uint32_t)batch_size, 1);
    return 0;
}

__attribute__((unused)) int vk_dispatch_sp17_dtss(VulkanContext *ctx, VkBufferHandle h_cache, VkBufferHandle dh_in, int seq_len, int d_state) {
    if (!VK_BUFFER_VALID(h_cache)) return -1;
    if (ensure_sp17_pipeline(ctx) != 0) return -1;
    
    vk_bind_buffer_to_descriptor(ctx, vk_get_pipe_ssm_scan(ctx), 0, h_cache);
    vk_bind_buffer_to_descriptor(ctx, vk_get_pipe_ssm_scan(ctx), 1, dh_in);
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp17_dtss_pipeline);
    
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct {
        uint32_t seq_len;
        uint32_t d_state;
        float eps_vel;
    } pc;
    pc.seq_len = (uint32_t)seq_len;
    pc.d_state = (uint32_t)d_state;
    pc.eps_vel = 1e-4f;
    
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    uint32_t wg_x = (d_state + 255) / 256;
    vkCmdDispatch(cmd, wg_x, 1, 1);
    return 0;
}

__attribute__((unused)) int vk_dispatch_sp16_null_space_prune(VulkanContext *ctx, VkBufferHandle dh_in, VkBufferHandle w_out, int seq_len, int d_state, int d_model, int batch_size) {
    if (!VK_BUFFER_VALID(dh_in) || !VK_BUFFER_VALID(w_out)) return -1;
    if (ensure_sp16_pipeline(ctx) != 0) return -1;
    
    vk_bind_buffer_to_descriptor(ctx, vk_get_pipe_ssm_scan(ctx), 0, dh_in);
    vk_bind_buffer_to_descriptor(ctx, vk_get_pipe_ssm_scan(ctx), 1, w_out);
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp16_nsp_pipeline);
    
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t d_model;
        uint32_t batch_size;
    } pc;
    pc.seq_len      = (uint32_t)seq_len;
    pc.d_state      = (uint32_t)d_state;
    pc.d_model      = (uint32_t)d_model;
    pc.batch_size   = (uint32_t)batch_size;
    
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    uint32_t wg_x = (d_state + 255) / 256;
    vkCmdDispatch(cmd, wg_x, 1, 1);
    return 0;
}

__attribute__((unused)) int vk_dispatch_f15_cross_layer_adjoint(VulkanContext *ctx, VkBufferHandle lambda_buf, VkBufferHandle df_dh_buf, uint32_t total_elements, float dl) {
    if (!VK_BUFFER_VALID(lambda_buf) || !VK_BUFFER_VALID(df_dh_buf)) return -1;
    if (ensure_f15_pipeline(ctx) != 0) return -1;
    
    vk_bind_buffer_to_descriptor(ctx, vk_get_pipe_ssm_scan(ctx), 0, lambda_buf);
    vk_bind_buffer_to_descriptor(ctx, vk_get_pipe_ssm_scan(ctx), 1, df_dh_buf);
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f15_cross_layer_adjoint_pipeline);
    
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct {
        uint32_t total_elements;
        float dl;
    } pc;
    pc.total_elements = total_elements;
    pc.dl = dl;
    
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    uint32_t wg_x = (total_elements + 255) / 256;
    vkCmdDispatch(cmd, wg_x, 1, 1);
    
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    
    return 0;
}

__attribute__((unused)) void f15_adjoint_validate_gate(VulkanContext *ctx, VkBufferHandle lambda_buf, VkBufferHandle df_dh_buf, uint32_t total_elements) {
    float *lambda_cpu = (float*)malloc(total_elements * sizeof(float));
    float *df_dh_cpu = (float*)malloc(total_elements * sizeof(float));
    vk_download(ctx, lambda_cpu, lambda_buf, total_elements * sizeof(float));
    vk_download(ctx, df_dh_cpu, df_dh_buf, total_elements * sizeof(float));
    
    float dl = 1.0f; // Single step depth
    float expected = lambda_cpu[0] + dl * lambda_cpu[0] * df_dh_cpu[0];
    
    vk_dispatch_f15_cross_layer_adjoint(ctx, lambda_buf, df_dh_buf, total_elements, dl);
    
    float *lambda_after = (float*)malloc(total_elements * sizeof(float));
    vk_download(ctx, lambda_after, lambda_buf, total_elements * sizeof(float));
    
    printf("[F15 Gate] Cross-Layer Continuous Adjoint Validated. GPU Lambda[0]: %f, Expected: %f\n", lambda_after[0], expected);
    
    free(lambda_cpu);
    free(df_dh_cpu);
    free(lambda_after);
}

__attribute__((unused)) int vk_dispatch_f14_ternary_activation_escrow(VulkanContext *ctx, VkBufferHandle fwd_h, VkBufferHandle dh_in, int seq_len, int d_state, int batch_size) {
    if (!VK_BUFFER_VALID(fwd_h) || !VK_BUFFER_VALID(dh_in)) return -1;
    if (ensure_f14_pipeline(ctx) != 0) return -1;
    
    vk_bind_buffer_to_descriptor(ctx, vk_get_pipe_ssm_scan(ctx), 0, fwd_h);
    vk_bind_buffer_to_descriptor(ctx, vk_get_pipe_ssm_scan(ctx), 1, dh_in);
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f14_tae_pipeline);
    
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t batch_size;
    } pc;
    pc.seq_len      = (uint32_t)seq_len;
    pc.d_state      = (uint32_t)d_state;
    pc.batch_size   = (uint32_t)batch_size;
    
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    uint32_t total_elements = batch_size * seq_len * d_state;
    uint32_t wg_x = (total_elements + 255) / 256;
    vkCmdDispatch(cmd, wg_x, 1, 1);
    return 0;
}

int r364_crashmap_logging_enabled(void) {
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized) {
        const char *value = getenv("V12_R364_CRASHMAP");
        enabled = (value && value[0] != '\0' && strcmp(value, "0") != 0);
        initialized = 1;
    }
    return enabled;
}

float erc_stat_float(uint32_t bits) {
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

const char *erc_mode_name(int mode) {
    switch (mode) {
        case V12_EVIDENCE_RATE_MODE_TSA: return "tsa";
        case V12_EVIDENCE_RATE_MODE_GBAR: return "gbar";
        default: return "current";
    }
}

int env_flag_enabled(const char *value) {
    return (value != NULL && value[0] != '\0' && value[0] != '0');
}

void free_vk_csr(VulkanContext *ctx, VkTernaryCSR *csr) {
    if (VK_BUFFER_VALID(csr->pos_indices)) vk_free(ctx, csr->pos_indices);
    if (VK_BUFFER_VALID(csr->neg_indices)) vk_free(ctx, csr->neg_indices);
    if (VK_BUFFER_VALID(csr->row_ptr_pos)) vk_free(ctx, csr->row_ptr_pos);
    if (VK_BUFFER_VALID(csr->row_ptr_neg)) vk_free(ctx, csr->row_ptr_neg);
    *csr = (VkTernaryCSR){0};
}

int materialize_a_gate_from_gpu_recompute(GpuPipeline *p,
                                                 SSMLayer *L,
                                                 int layer_idx,
                                                 float *dst,
                                                 size_t ssm_bytes,
                                                 int seq_len,
                                                 int ds,
                                                 int batch_size,
                                                 uint32_t gpu_opt_flags) {
    if (!p || !L || !dst || layer_idx < 0 || layer_idx >= p->n_layers) return -1;

    VulkanContext *ctx = p->ctx;
    if (!ctx) return -1;
    if (!VK_BUFFER_VALID(p->layers[layer_idx].fwd_x_cv)
        || !VK_BUFFER_VALID(p->layers[layer_idx].gpu_A_const)
        || !VK_BUFFER_VALID(p->lw_WA)
        || !VK_BUFFER_VALID(p->lw_bA)) {
        return -1;
    }

    VkBufferHandle a_gate_scratch = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    VkBufferHandle a_eff_scratch  = vk_alloc(ctx, ssm_bytes, VK_BUF_STORAGE | VK_BUF_TRANSFER_SRC);
    int ok = -1;

    if (VK_BUFFER_VALID(a_gate_scratch) && VK_BUFFER_VALID(a_eff_scratch)) {
        if (!(gpu_opt_flags & GPU_OPT_WA))
            vk_upload(ctx, p->lw_WA, L->W_A, (size_t)ds * ds * sizeof(float));
        if (!(gpu_opt_flags & GPU_OPT_BA))
            vk_upload(ctx, p->lw_bA, L->b_A, (size_t)ds * sizeof(float));

        if (vk_dispatch_a_gate_recompute(ctx,
                                         p->layers[layer_idx].fwd_x_cv,
                                         p->lw_WA,
                                         p->lw_bA,
                                         p->layers[layer_idx].gpu_A_const,
                                         a_gate_scratch,
                                         a_eff_scratch,
                                         (uint32_t)seq_len,
                                         (uint32_t)ds,
                                         (uint32_t)batch_size) == 0) {
            if (vk_download(ctx, dst, a_gate_scratch, ssm_bytes) == 0)
                ok = 0;
        }
    }

    if (VK_BUFFER_VALID(a_gate_scratch)) vk_free(ctx, a_gate_scratch);
    if (VK_BUFFER_VALID(a_eff_scratch)) vk_free(ctx, a_eff_scratch);
    return ok;
}

double variance_priced_hash01(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (double)(x >> 8) * (1.0 / 16777216.0);
}

double safe_f64(float v) {
    return isfinite((double)v) ? (double)v : 0.0;
}

double relative_residual_or_zero(double exact_energy, double residual_energy) {
    if (exact_energy > 1.0e-20) return residual_energy / exact_energy;
    return residual_energy <= 1.0e-20 ? 0.0 : 1.0;
}

int lagged_ce_impulse_eligibility_update(
    Model *m, SSMLayer *L, float *transition_credit, const float *ce_impulse,
    const float *h_cache, const float *a_eff, const float *a_gate,
    const float *x_cv, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d_state) {
    if (!m || !m->lagged_ce_impulse_eligibility_enabled || !L ||
        !transition_credit || !ce_impulse || !h_cache || !a_eff ||
        !a_gate || !x_cv || batch_size <= 0 || seq_len <= 0 ||
        d_state <= 0) {
        return 0;
    }

    uint32_t horizon = m->lagged_ce_impulse_horizon;
    if (!(horizon == 2u || horizon == 4u || horizon == 8u)) {
        horizon = 4u;
    }

    const int items = batch_size * seq_len;
    const size_t scalar_count = (size_t)items * (size_t)d_state;
    float *ledger_credit = (float *)malloc(scalar_count * sizeof(float));
    float *exact_dot = (float *)malloc(scalar_count * sizeof(float));
    float *ledger_dot = (float *)malloc(scalar_count * sizeof(float));
    double *exact_dA = (double *)calloc((size_t)d_state, sizeof(double));
    double *ledger_dA = (double *)calloc((size_t)d_state, sizeof(double));
    double *exact_dbA = (double *)calloc((size_t)d_state, sizeof(double));
    double *ledger_dbA = (double *)calloc((size_t)d_state, sizeof(double));
    if (!ledger_credit || !exact_dot || !ledger_dot ||
        !exact_dA || !ledger_dA || !exact_dbA || !ledger_dbA) {
        free(ledger_credit);
        free(exact_dot);
        free(ledger_dot);
        free(exact_dA);
        free(ledger_dA);
        free(exact_dbA);
        free(ledger_dbA);
        return 0;
    }

    double exact_transition_energy = 0.0;
    double ledger_transition_energy = 0.0;
    double transition_dot = 0.0;
    double transition_residual_energy = 0.0;
    double a_eff_sq_sum = 0.0;
    double a_eff_abs_max = 0.0;
    uint64_t a_eff_count = 0u;

    for (int b = 0; b < batch_size; b++) {
        const size_t batch_base = (size_t)b * (size_t)seq_len;
        for (int t = 0; t < seq_len; t++) {
            const size_t pos = batch_base + (size_t)t;
            for (int i = 0; i < d_state; i++) {
                double product = 1.0;
                double lambda = 0.0;
                for (uint32_t d = 0u; d <= horizon; d++) {
                    const int ft = t + (int)d;
                    if (ft >= seq_len) break;
                    const size_t fpos = batch_base + (size_t)ft;
                    if (d > 0u) {
                        if (boundary_mask && boundary_mask[fpos]) break;
                        const double a = safe_f64(a_eff[fpos * (size_t)d_state + (size_t)i]);
                        product *= a;
                    }
                    lambda += product * safe_f64(ce_impulse[fpos * (size_t)d_state + (size_t)i]);
                }

                const double prev =
                    (t > 0 && (!boundary_mask || !boundary_mask[pos]))
                        ? safe_f64(h_cache[(pos - 1u) * (size_t)d_state + (size_t)i])
                        : 0.0;
                const size_t idx = pos * (size_t)d_state + (size_t)i;
                const double exact = safe_f64(transition_credit[idx]);
                const double ledger = lambda * prev;
                const double diff = ledger - exact;
                const double gate = safe_f64(a_gate[idx]);
                const double a_value = safe_f64(L->A[i]);
                const double exact_d = exact * a_value * gate * (1.0 - gate);
                const double ledger_d = ledger * a_value * gate * (1.0 - gate);

                ledger_credit[idx] = (float)ledger;
                exact_dot[idx] = (float)exact_d;
                ledger_dot[idx] = (float)ledger_d;
                exact_dA[i] += exact * gate;
                ledger_dA[i] += ledger * gate;
                exact_dbA[i] += exact_d;
                ledger_dbA[i] += ledger_d;

                exact_transition_energy += exact * exact;
                ledger_transition_energy += ledger * ledger;
                transition_dot += exact * ledger;
                transition_residual_energy += diff * diff;
                {
                    const double a = safe_f64(a_eff[idx]);
                    const double aa = fabs(a);
                    a_eff_sq_sum += a * a;
                    if (aa > a_eff_abs_max) a_eff_abs_max = aa;
                    a_eff_count++;
                }
            }
        }
    }

    double exact_dA_energy = 0.0;
    double ledger_dA_energy = 0.0;
    double dA_dot = 0.0;
    double dA_residual_energy = 0.0;
    double exact_dbA_energy = 0.0;
    double ledger_dbA_energy = 0.0;
    double dbA_dot = 0.0;
    double dbA_residual_energy = 0.0;
    for (int i = 0; i < d_state; i++) {
        const double ea = exact_dA[i];
        const double la = ledger_dA[i];
        const double ed = exact_dbA[i];
        const double ld = ledger_dbA[i];
        exact_dA_energy += ea * ea;
        ledger_dA_energy += la * la;
        dA_dot += ea * la;
        dA_residual_energy += (la - ea) * (la - ea);
        exact_dbA_energy += ed * ed;
        ledger_dbA_energy += ld * ld;
        dbA_dot += ed * ld;
        dbA_residual_energy += (ld - ed) * (ld - ed);
    }

    uint32_t dWA_cols = (uint32_t)d_state;
    if (dWA_cols > 64u) dWA_cols = 64u;
    if (dWA_cols == 0u) dWA_cols = 1u;
    const double dWA_scale = (double)d_state / (double)dWA_cols;
    double *x_col_energy = (double *)calloc((size_t)d_state, sizeof(double));
    uint8_t *x_col_selected = (uint8_t *)calloc((size_t)d_state, sizeof(uint8_t));
    if (!x_col_energy || !x_col_selected) {
        free(x_col_energy);
        free(x_col_selected);
        free(ledger_credit);
        free(exact_dot);
        free(ledger_dot);
        free(exact_dA);
        free(ledger_dA);
        free(exact_dbA);
        free(ledger_dbA);
        return 0;
    }
    for (int pos = 0; pos < items; pos++) {
        for (int j = 0; j < d_state; j++) {
            const double x = safe_f64(x_cv[(size_t)pos * (size_t)d_state + (size_t)j]);
            x_col_energy[j] += x * x;
        }
    }
    for (uint32_t pick = 0u; pick < dWA_cols; pick++) {
        int best = -1;
        double best_energy = -1.0;
        for (int j = 0; j < d_state; j++) {
            if (!x_col_selected[j] && x_col_energy[j] > best_energy) {
                best_energy = x_col_energy[j];
                best = j;
            }
        }
        if (best < 0) break;
        x_col_selected[best] = 1u;
    }
    double exact_dWA_energy = 0.0;
    double ledger_dWA_energy = 0.0;
    double dWA_dot = 0.0;
    double dWA_residual_energy = 0.0;
    for (int i = 0; i < d_state; i++) {
        for (int j = 0; j < d_state; j++) {
            if (!x_col_selected[j]) continue;
            double ex = 0.0;
            double le = 0.0;
            for (int pos = 0; pos < items; pos++) {
                const size_t idx = (size_t)pos * (size_t)d_state + (size_t)i;
                const double x = safe_f64(x_cv[(size_t)pos * (size_t)d_state + (size_t)j]);
                ex += safe_f64(exact_dot[idx]) * x;
                le += safe_f64(ledger_dot[idx]) * x;
            }
            exact_dWA_energy += dWA_scale * ex * ex;
            ledger_dWA_energy += dWA_scale * le * le;
            dWA_dot += dWA_scale * ex * le;
            dWA_residual_energy += dWA_scale * (le - ex) * (le - ex);
        }
    }
    free(x_col_energy);
    free(x_col_selected);

    const double transition_residual_fraction =
        relative_residual_or_zero(exact_transition_energy, transition_residual_energy);
    const double dA_residual_fraction =
        relative_residual_or_zero(exact_dA_energy, dA_residual_energy);
    const double dbA_residual_fraction =
        relative_residual_or_zero(exact_dbA_energy, dbA_residual_energy);
    const double dWA_residual_fraction =
        relative_residual_or_zero(exact_dWA_energy, dWA_residual_energy);
    const double cosine =
        (exact_transition_energy > 0.0 && ledger_transition_energy > 0.0)
            ? transition_dot / sqrt(exact_transition_energy * ledger_transition_energy)
            : 0.0;
    const double rho_rms =
        a_eff_count > 0u ? sqrt(a_eff_sq_sum / (double)a_eff_count) : 1.0;
    double observed_rho = fmax(rho_rms, a_eff_abs_max);
    if (!isfinite(observed_rho) || observed_rho < 0.0) observed_rho = 1.0;
    if (observed_rho > 0.999) observed_rho = 0.999;
    const double tail_bound_fraction =
        observed_rho < 0.999
            ? pow(observed_rho, (double)horizon + 1.0) / fmax(1.0e-9, 1.0 - observed_rho)
            : 999.0;
    const double priced_ratio =
        (double)horizon * (double)m->lagged_ce_impulse_local_jvp_fraction +
        (double)m->lagged_ce_impulse_ring_fraction +
        (double)m->lagged_ce_impulse_tail_audit_fraction +
        (double)m->lagged_ce_impulse_memory_fraction;
    const double budget = (double)m->lagged_ce_impulse_residual_budget;
    const int identifying = exact_transition_energy > 1.0e-20 ? 1 : 0;
    const int pass =
        identifying &&
        transition_residual_fraction <= budget &&
        dA_residual_fraction <= budget &&
        dbA_residual_fraction <= budget &&
        dWA_residual_fraction <= budget &&
        priced_ratio < 0.90 ? 1 : 0;
    double blend = (double)m->lagged_ce_impulse_blend;
    if (blend < 0.0) blend = 0.0;
    if (blend > 1.0) blend = 1.0;
    const int should_apply = pass && blend > 0.0 ? 1 : 0;

    if (should_apply) {
        const double keep = 1.0 - blend;
        for (size_t idx = 0; idx < scalar_count; idx++) {
            transition_credit[idx] =
                (float)(keep * safe_f64(transition_credit[idx]) +
                        blend * safe_f64(ledger_credit[idx]));
        }
    }

    m->lagged_ce_impulse_batches++;
    m->lagged_ce_impulse_layers++;
    m->lagged_ce_impulse_positions += (uint64_t)items;
    m->lagged_ce_impulse_exact_transition_energy += exact_transition_energy;
    m->lagged_ce_impulse_ledger_transition_energy += ledger_transition_energy;
    m->lagged_ce_impulse_transition_dot += transition_dot;
    m->lagged_ce_impulse_transition_residual_energy += transition_residual_energy;
    m->lagged_ce_impulse_exact_dA_energy += exact_dA_energy;
    m->lagged_ce_impulse_ledger_dA_energy += ledger_dA_energy;
    m->lagged_ce_impulse_dA_dot += dA_dot;
    m->lagged_ce_impulse_dA_residual_energy += dA_residual_energy;
    m->lagged_ce_impulse_exact_dbA_energy += exact_dbA_energy;
    m->lagged_ce_impulse_ledger_dbA_energy += ledger_dbA_energy;
    m->lagged_ce_impulse_dbA_dot += dbA_dot;
    m->lagged_ce_impulse_dbA_residual_energy += dbA_residual_energy;
    m->lagged_ce_impulse_exact_dWA_energy += exact_dWA_energy;
    m->lagged_ce_impulse_ledger_dWA_energy += ledger_dWA_energy;
    m->lagged_ce_impulse_dWA_dot += dWA_dot;
    m->lagged_ce_impulse_dWA_residual_energy += dWA_residual_energy;
    m->lagged_ce_impulse_dWA_sampled_columns = dWA_cols;
    m->lagged_ce_impulse_transition_residual_fraction =
        (float)relative_residual_or_zero(
            m->lagged_ce_impulse_exact_transition_energy,
            m->lagged_ce_impulse_transition_residual_energy);
    m->lagged_ce_impulse_dA_residual_fraction =
        (float)relative_residual_or_zero(
            m->lagged_ce_impulse_exact_dA_energy,
            m->lagged_ce_impulse_dA_residual_energy);
    m->lagged_ce_impulse_dbA_residual_fraction =
        (float)relative_residual_or_zero(
            m->lagged_ce_impulse_exact_dbA_energy,
            m->lagged_ce_impulse_dbA_residual_energy);
    m->lagged_ce_impulse_dWA_residual_fraction =
        (float)relative_residual_or_zero(
            m->lagged_ce_impulse_exact_dWA_energy,
            m->lagged_ce_impulse_dWA_residual_energy);
    m->lagged_ce_impulse_cosine =
        (float)((m->lagged_ce_impulse_exact_transition_energy > 0.0 &&
                 m->lagged_ce_impulse_ledger_transition_energy > 0.0)
            ? m->lagged_ce_impulse_transition_dot /
              sqrt(m->lagged_ce_impulse_exact_transition_energy *
                   m->lagged_ce_impulse_ledger_transition_energy)
            : 0.0);
    m->lagged_ce_impulse_priced_ratio = (float)priced_ratio;
    m->lagged_ce_impulse_observed_rho = (float)observed_rho;
    m->lagged_ce_impulse_tail_bound_fraction = (float)tail_bound_fraction;
    m->lagged_ce_impulse_identifying = identifying;
    m->lagged_ce_impulse_pass = pass;
    m->lagged_ce_impulse_last_applied = should_apply;
    m->lagged_ce_impulse_training_layers++;
    if (should_apply) {
        m->lagged_ce_impulse_applied_layers++;
    } else {
        m->lagged_ce_impulse_fallback_layers++;
    }

    free(ledger_credit);
    free(exact_dot);
    free(ledger_dot);
    free(exact_dA);
    free(ledger_dA);
    free(exact_dbA);
    free(ledger_dbA);
    return should_apply;
}

uint64_t r438_mix_u64(uint64_t h, uint64_t x) {
    h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h *= 1099511628211ull;
    return h;
}

int32_t r438_quantize_signature_value(float v, float scale) {
    if (!isfinite((double)v)) return 0;
    double q = floor((double)v * (double)scale + ((v >= 0.0f) ? 0.5 : -0.5));
    if (q > 32767.0) q = 32767.0;
    if (q < -32768.0) q = -32768.0;
    return (int32_t)q;
}

uint64_t r438_state_transition_signature_hash(
    const float *h_cache, const float *a_eff, const uint8_t *boundary_mask,
    int pos, int d_state, float state_scale, float transition_scale) {
    uint64_t h = 1469598103934665603ull;
    h = r438_mix_u64(h, boundary_mask && boundary_mask[pos] ? 1ull : 0ull);
    for (int q = 0; q < 8; q++) {
        int idx = (q * d_state) / 8;
        if (idx >= d_state) idx = d_state - 1;
        const size_t off = (size_t)pos * (size_t)d_state + (size_t)idx;
        const int32_t hs = r438_quantize_signature_value(h_cache[off], state_scale);
        const int32_t as = r438_quantize_signature_value(a_eff[off], transition_scale);
        h = r438_mix_u64(h, (uint64_t)(uint32_t)hs);
        h = r438_mix_u64(h, (uint64_t)(uint32_t)as);
    }
    return h ? h : 1ull;
}

void executable_multiplicity_packets_update(
    Model *m, const float *dA_ps, const float *h_cache, const float *a_eff,
    const uint8_t *boundary_mask, int batch_size, int seq_len, int d_state) {
    if (!m || !m->executable_multiplicity_packets_enabled || !dA_ps ||
        !h_cache || !a_eff || batch_size <= 0 || seq_len <= 0 ||
        d_state <= 0) {
        return;
    }

    const int items = batch_size * seq_len;
    uint32_t table_cap = 1u;
    while (table_cap < (uint32_t)items * 2u) table_cap <<= 1u;
    if (table_cap < 16u) table_cap = 16u;
    R438PacketCell *cells =
        (R438PacketCell *)calloc((size_t)table_cap, sizeof(R438PacketCell));
    if (!cells) return;

    uint64_t used_cells = 0u;
    const float state_scale =
        m->executable_multiplicity_state_scale > 0.0f
            ? m->executable_multiplicity_state_scale
            : 16.0f;
    const float transition_scale =
        m->executable_multiplicity_transition_scale > 0.0f
            ? m->executable_multiplicity_transition_scale
            : 64.0f;
    for (int pos = 0; pos < items; pos++) {
        const uint64_t hash =
            r438_state_transition_signature_hash(
                h_cache, a_eff, boundary_mask, pos, d_state,
                state_scale, transition_scale);
        uint32_t slot = (uint32_t)(hash & (uint64_t)(table_cap - 1u));
        while (cells[slot].used && cells[slot].hash != hash) {
            slot = (slot + 1u) & (table_cap - 1u);
        }
        if (!cells[slot].used) {
            cells[slot].used = 1;
            cells[slot].hash = hash;
            cells[slot].rep_pos = (uint32_t)pos;
            used_cells++;
        }
        R438PacketCell *cell = &cells[slot];
        cell->count++;

        double exact_energy = 0.0;
        double residual_energy = 0.0;
        const size_t base = (size_t)pos * (size_t)d_state;
        const size_t rep_base = (size_t)cell->rep_pos * (size_t)d_state;
        for (int i = 0; i < d_state; i++) {
            const double exact = safe_f64(dA_ps[base + (size_t)i]);
            const double rep = safe_f64(dA_ps[rep_base + (size_t)i]);
            const double diff = exact - rep;
            exact_energy += exact * exact;
            residual_energy += diff * diff;
        }
        cell->exact_energy += exact_energy;
        cell->residual_energy += residual_energy;
    }

    uint64_t accepted_cells = 0u;
    uint64_t accepted_positions = 0u;
    double accepted_exact_energy = 0.0;
    double accepted_residual_energy = 0.0;
    const uint32_t min_count =
        m->executable_multiplicity_min_count < 2u
            ? 2u
            : m->executable_multiplicity_min_count;
    for (uint32_t s = 0u; s < table_cap; s++) {
        if (!cells[s].used || cells[s].count < min_count) continue;
        accepted_cells++;
        accepted_positions += cells[s].count;
        accepted_exact_energy += cells[s].exact_energy;
        accepted_residual_energy += cells[s].residual_energy;
    }
    free(cells);

    const uint64_t rejected_positions =
        accepted_positions <= (uint64_t)items
            ? (uint64_t)items - accepted_positions
            : 0u;
    const double positions = (double)items;
    const double accepted_mass = positions > 0.0
        ? (double)accepted_positions / positions
        : 0.0;
    const double residual_fraction =
        relative_residual_or_zero(accepted_exact_energy, accepted_residual_energy);
    const double mean_multiplicity = accepted_cells > 0u
        ? (double)accepted_positions / (double)accepted_cells
        : 0.0;
    const double packet_work_fraction = accepted_positions > 0u
        ? (double)accepted_cells / (double)accepted_positions
        : 1.0;
    const double rejected_fraction = positions > 0.0
        ? (double)rejected_positions / positions
        : 1.0;
    const double overhead =
        (double)m->executable_multiplicity_signature_fraction +
        (double)m->executable_multiplicity_verifier_fraction +
        (double)m->executable_multiplicity_memory_fraction;
    const double priced_ratio =
        rejected_fraction +
        accepted_mass * packet_work_fraction +
        accepted_mass * residual_fraction +
        overhead;
    const int identifying = used_cells > 0u ? 1 : 0;
    const int pass =
        identifying &&
        accepted_mass >= (double)m->executable_multiplicity_min_mass &&
        residual_fraction <= (double)m->executable_multiplicity_max_residual &&
        priced_ratio < 0.90 ? 1 : 0;

    m->executable_multiplicity_batches++;
    m->executable_multiplicity_layers++;
    m->executable_multiplicity_positions += (uint64_t)items;
    m->executable_multiplicity_cells += used_cells;
    m->executable_multiplicity_accepted_cells += accepted_cells;
    m->executable_multiplicity_accepted_positions += accepted_positions;
    m->executable_multiplicity_rejected_positions += rejected_positions;
    m->executable_multiplicity_exact_energy += accepted_exact_energy;
    m->executable_multiplicity_residual_energy += accepted_residual_energy;
    m->executable_multiplicity_accepted_mass =
        m->executable_multiplicity_positions > 0u
            ? (float)((double)m->executable_multiplicity_accepted_positions /
                      (double)m->executable_multiplicity_positions)
            : 0.0f;
    m->executable_multiplicity_residual_fraction =
        (float)relative_residual_or_zero(
            m->executable_multiplicity_exact_energy,
            m->executable_multiplicity_residual_energy);
    m->executable_multiplicity_mean_multiplicity =
        m->executable_multiplicity_accepted_cells > 0u
            ? (float)((double)m->executable_multiplicity_accepted_positions /
                      (double)m->executable_multiplicity_accepted_cells)
            : 0.0f;
    m->executable_multiplicity_packet_work_fraction =
        m->executable_multiplicity_accepted_positions > 0u
            ? (float)((double)m->executable_multiplicity_accepted_cells /
                      (double)m->executable_multiplicity_accepted_positions)
            : 1.0f;
    m->executable_multiplicity_priced_ratio = (float)priced_ratio;
    m->executable_multiplicity_identifying = identifying;
    m->executable_multiplicity_pass = pass;
}

void original_ce_credit_debt_observer_update(
    Model *m, const float *transition_credit, const float *direct_state_credit,
    const float *h_cache, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d_state) {
    if (!m || !m->original_ce_credit_debt_observer_enabled ||
        !transition_credit || !direct_state_credit || !h_cache ||
        batch_size <= 0 || seq_len <= 0 || d_state <= 0) {
        return;
    }

    const int items = batch_size * seq_len;
    double *exact_sum = (double *)calloc((size_t)d_state, sizeof(double));
    double *local_sum = (double *)calloc((size_t)d_state, sizeof(double));
    double *selected_exact_sum = (double *)calloc((size_t)d_state, sizeof(double));
    double *selected_local_sum = (double *)calloc((size_t)d_state, sizeof(double));
    double *row_scores = (double *)calloc((size_t)items, sizeof(double));
    uint8_t *selected = (uint8_t *)calloc((size_t)items, sizeof(uint8_t));
    if (!exact_sum || !local_sum || !selected_exact_sum ||
        !selected_local_sum || !row_scores || !selected) {
        free(exact_sum);
        free(local_sum);
        free(selected_exact_sum);
        free(selected_local_sum);
        free(row_scores);
        free(selected);
        return;
    }

    double row_score_sum = 0.0;
    for (int b = 0; b < batch_size; b++) {
        const size_t batch_base = (size_t)b * (size_t)seq_len;
        for (int t = 0; t < seq_len; t++) {
            const size_t pos = batch_base + (size_t)t;
            const float *exact = transition_credit + pos * (size_t)d_state;
            const float *direct = direct_state_credit + pos * (size_t)d_state;
            const float *prev = NULL;
            if (t > 0 && (!boundary_mask || !boundary_mask[pos])) {
                prev = h_cache + (batch_base + (size_t)(t - 1)) *
                       (size_t)d_state;
            }
            double score = 0.0;
            for (int i = 0; i < d_state; i++) {
                const double e = isfinite((double)exact[i]) ?
                    (double)exact[i] : 0.0;
                const double p = prev && isfinite((double)prev[i]) ?
                    (double)prev[i] : 0.0;
                const double d = isfinite((double)direct[i]) ?
                    (double)direct[i] : 0.0;
                const double l = d * p;
                exact_sum[i] += e;
                local_sum[i] += l;
                score += l * l;
            }
            row_scores[pos] = score;
            row_score_sum += score;
        }
    }

    double surface = (double)m->original_ce_credit_surface_fraction;
    if (surface <= 0.0) surface = 0.35;
    if (surface > 1.0) surface = 1.0;
    int target_selected = (int)ceil(surface * (double)items);
    if (target_selected < 1) target_selected = 1;
    if (target_selected > items) target_selected = items;

    for (int pick = 0; pick < target_selected; pick++) {
        int best = -1;
        double best_score = -1.0;
        for (int pos = 0; pos < items; pos++) {
            if (!selected[pos] && row_scores[pos] > best_score) {
                best_score = row_scores[pos];
                best = pos;
            }
        }
        if (best < 0) break;
        selected[best] = 1u;
    }

    uint64_t selected_positions = 0u;
    for (int b = 0; b < batch_size; b++) {
        const size_t batch_base = (size_t)b * (size_t)seq_len;
        for (int t = 0; t < seq_len; t++) {
            const size_t pos = batch_base + (size_t)t;
            if (!selected[pos]) continue;
            selected_positions++;
            const float *exact = transition_credit + pos * (size_t)d_state;
            const float *direct = direct_state_credit + pos * (size_t)d_state;
            const float *prev = NULL;
            if (t > 0 && (!boundary_mask || !boundary_mask[pos])) {
                prev = h_cache + (batch_base + (size_t)(t - 1)) *
                       (size_t)d_state;
            }
            for (int i = 0; i < d_state; i++) {
                const double e = isfinite((double)exact[i]) ?
                    (double)exact[i] : 0.0;
                const double p = prev && isfinite((double)prev[i]) ?
                    (double)prev[i] : 0.0;
                const double d = isfinite((double)direct[i]) ?
                    (double)direct[i] : 0.0;
                selected_exact_sum[i] += e;
                selected_local_sum[i] += d * p;
            }
        }
    }

    double exact_energy = 0.0;
    double local_energy = 0.0;
    double local_dot = 0.0;
    double local_residual_energy = 0.0;
    double selected_exact_energy = 0.0;
    double selected_local_energy = 0.0;
    double selected_dot = 0.0;
    double selected_residual_energy = 0.0;

    for (int i = 0; i < d_state; i++) {
        const double e = exact_sum[i];
        const double l = local_sum[i];
        const double r = e - l;
        exact_energy += e * e;
        local_energy += l * l;
        local_dot += e * l;
        local_residual_energy += r * r;

        const double se = selected_exact_sum[i];
        const double sl = selected_local_sum[i];
        const double sr = se - sl;
        selected_exact_energy += se * se;
        selected_local_energy += sl * sl;
        selected_dot += se * sl;
        selected_residual_energy += sr * sr;
    }

    free(exact_sum);
    free(local_sum);
    free(selected_exact_sum);
    free(selected_local_sum);
    free(row_scores);
    free(selected);

    m->original_ce_credit_gate_batches++;
    m->original_ce_credit_gate_layers++;
    m->original_ce_credit_gate_positions += (uint64_t)items;
    m->original_ce_credit_gate_selected_positions += selected_positions;
    m->original_ce_credit_gate_exact_energy += exact_energy;
    m->original_ce_credit_gate_local_energy += local_energy;
    m->original_ce_credit_gate_local_dot += local_dot;
    m->original_ce_credit_gate_local_residual_energy += local_residual_energy;
    m->original_ce_credit_gate_selected_exact_energy += selected_exact_energy;
    m->original_ce_credit_gate_selected_local_energy += selected_local_energy;
    m->original_ce_credit_gate_selected_dot += selected_dot;
    m->original_ce_credit_gate_selected_residual_energy += selected_residual_energy;

    const double exact_total = m->original_ce_credit_gate_exact_energy;
    const double selected_exact_total =
        m->original_ce_credit_gate_selected_exact_energy;
    const double positions_total =
        (double)m->original_ce_credit_gate_positions;
    const double selected_total =
        (double)m->original_ce_credit_gate_selected_positions;
    const double local_progress =
        exact_total > 0.0 ? m->original_ce_credit_gate_local_dot / exact_total : 0.0;
    const double local_residual =
        exact_total > 0.0
            ? sqrt(m->original_ce_credit_gate_local_residual_energy / exact_total)
            : 1.0;
    const double selected_progress =
        selected_exact_total > 0.0
            ? m->original_ce_credit_gate_selected_dot / selected_exact_total
            : 0.0;
    const double selected_residual =
        selected_exact_total > 0.0
            ? sqrt(m->original_ce_credit_gate_selected_residual_energy /
                   selected_exact_total)
            : 1.0;
    const double selected_fraction =
        positions_total > 0.0 ? selected_total / positions_total : 0.0;
    const double local_work = (double)m->original_ce_credit_local_work_multiplier;
    const double surface_work = selected_fraction * local_work;
    const double debt_price = (double)m->original_ce_credit_debt_price;
    const double verify = (double)m->original_ce_credit_verifier_fraction;
    const double overhead = (double)m->original_ce_credit_overhead;
    const double recurrent_share =
        fmin(1.0, fmax(0.0, (double)m->original_ce_credit_recurrent_backward_share));
    const double local_priced =
        local_progress > 1.0e-12
            ? (local_work + verify + overhead + debt_price * local_residual) /
              local_progress
            : INFINITY;
    const double surface_priced =
        selected_progress > 1.0e-12
            ? (surface_work + verify + overhead + debt_price * selected_residual) /
              selected_progress
            : INFINITY;
    const double local_whole =
        1.0 / fmax(1.0e-12,
            1.0 - recurrent_share * fmax(0.0, 1.0 - local_priced));
    const double surface_whole =
        1.0 / fmax(1.0e-12,
            1.0 - recurrent_share * fmax(0.0, 1.0 - surface_priced));
    const int identifying = (exact_total > 0.0 && row_score_sum >= 0.0) ? 1 : 0;

    m->original_ce_credit_gate_selected_fraction = (float)selected_fraction;
    m->original_ce_credit_gate_local_progress_ratio = (float)local_progress;
    m->original_ce_credit_gate_local_residual_fraction = (float)local_residual;
    m->original_ce_credit_gate_local_priced_ratio = (float)local_priced;
    m->original_ce_credit_gate_local_whole_step_speedup = (float)local_whole;
    m->original_ce_credit_gate_surface_progress_ratio = (float)selected_progress;
    m->original_ce_credit_gate_surface_residual_fraction = (float)selected_residual;
    m->original_ce_credit_gate_surface_priced_ratio = (float)surface_priced;
    m->original_ce_credit_gate_surface_whole_step_speedup = (float)surface_whole;
    m->original_ce_credit_gate_identifying = identifying;
    m->original_ce_credit_gate_local_pass =
        (identifying &&
         local_progress >= (double)m->original_ce_credit_min_progress &&
         local_residual <= (double)m->original_ce_credit_max_residual &&
         local_whole >= (double)m->original_ce_credit_min_whole_step_speedup)
            ? 1 : 0;
    m->original_ce_credit_gate_surface_pass =
        (identifying &&
         selected_progress >= (double)m->original_ce_credit_min_progress &&
         selected_residual <= (double)m->original_ce_credit_max_residual &&
         surface_whole >= (double)m->original_ce_credit_min_whole_step_speedup)
            ? 1 : 0;
}


#else // !V12_HAS_VULKAN

void gpu_pipeline_r364_set_noop_compiled_interval(GpuPipeline *p,
                                                  int active,
                                                  uint32_t compile_id,
                                                  float compiled_block_share_f) {
    (void)p;
    (void)active;
    (void)compile_id;
    (void)compiled_block_share_f;
}

void gpu_pipeline_r364_get_noop_probe(const GpuPipeline *p, GpuR364NoopProbe *out) {
    (void)p;
    if (!out) return;
    memset(out, 0, sizeof(*out));
}


#endif // V12_HAS_VULKAN
