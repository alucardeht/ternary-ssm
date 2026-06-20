#include "gpu_pipeline_internal.h"
#ifdef V12_HAS_VULKAN
VkPipeline mprm_adam_pipeline = VK_NULL_HANDLE;
VkPipeline f35_idre_pipeline = VK_NULL_HANDLE;
VkPipeline f36_slefc_forward_pipeline = VK_NULL_HANDLE;
VkPipeline f36_slefc_reconstruct_pipeline = VK_NULL_HANDLE;
VkPipeline f34_srat_forward_pipeline = VK_NULL_HANDLE;
VkPipeline f34_srat_reconstruct_pipeline = VK_NULL_HANDLE;
VkPipeline f01_rrd_involution_pipeline = VK_NULL_HANDLE;
VkPipeline e32_aoi_ssm_pipeline = VK_NULL_HANDLE;
VkPipeline ef03_pipeline = VK_NULL_HANDLE;
VkPipeline entangled_adam_pipeline = VK_NULL_HANDLE;
VkPipeline sp19_holo_adam_pipeline = VK_NULL_HANDLE;
VkPipeline f37_taqg_pipeline = VK_NULL_HANDLE;
VkPipeline f38_cst_pipeline = VK_NULL_HANDLE;
VkPipeline e13_lrms_pipeline = VK_NULL_HANDLE;
VkPipeline f39_bvgr_pipeline = VK_NULL_HANDLE;
VkPipeline sp10_psvcte_pipeline = VK_NULL_HANDLE;
VkPipeline sp11_lns_tfms_pipeline = VK_NULL_HANDLE;
VkPipeline sp12_lcmbi_pipeline = VK_NULL_HANDLE;
VkPipeline sp13_sde_bvdj_pipeline = VK_NULL_HANDLE;
VkPipeline sp14_gfit_pipeline = VK_NULL_HANDLE;
VkPipeline sp15_tt_rcrb_pipeline = VK_NULL_HANDLE;
VkPipeline f12_fourier_pipeline = VK_NULL_HANDLE;
VkPipeline e13_parametric_cotangent_pipeline = VK_NULL_HANDLE;
VkPipeline sp16_nsp_pipeline = VK_NULL_HANDLE;
VkPipeline f14_tae_pipeline = VK_NULL_HANDLE;
VkPipeline sp17_dtss_pipeline = VK_NULL_HANDLE;
VkPipeline f15_cross_layer_adjoint_pipeline = VK_NULL_HANDLE;
VkPipeline btrp_token_gather_pipeline = VK_NULL_HANDLE;




int ensure_f35_idre_pipeline(VulkanContext *ctx) {
    if (f35_idre_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = f35_idre_spv_len;
    smci.pCode = (const uint32_t *)f35_idre_spv;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &f35_idre_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

void vk_dispatch_f35_idre(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle v, VkBufferHandle dh_out, int seq_len, int d_state, int batch_size) {
    printf("DEBUG: Inside vk_dispatch_sp12_lcmbi\n"); fflush(stdout);
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(v) || !VK_BUFFER_VALID(dh_out)) return;
    if (ensure_f35_idre_pipeline(ctx) != 0) return;
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vk_bind_buffer_to_descriptor(ctx, slot, 0, A_eff);
    vk_bind_buffer_to_descriptor(ctx, slot, 1, v);
    vk_bind_buffer_to_descriptor(ctx, slot, 2, dh_out);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f35_idre_pipeline);
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct { uint32_t sl; uint32_t ds; uint32_t horiz; } pc;
    pc.sl = seq_len;
    pc.ds = d_state;
    pc.horiz = 0;
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    uint32_t total = seq_len * d_state;
    printf("DEBUG: Before vkCmdDispatch in sp12\n"); fflush(stdout);
    vkCmdDispatch(cmd, (pc.ds + 255) / 256, pc.sl, batch_size);
}

int ensure_f36_slefc_forward_pipeline(VulkanContext *ctx) {
    if (f36_slefc_forward_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = f36_slefc_forward_spv_len;
    smci.pCode = (const uint32_t *)f36_slefc_forward_spv;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &f36_slefc_forward_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_f36_slefc_reconstruct_pipeline(VulkanContext *ctx) {
    if (f36_slefc_reconstruct_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = f36_slefc_reconstruct_spv_len;
    smci.pCode = (const uint32_t *)f36_slefc_reconstruct_spv;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &f36_slefc_reconstruct_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_f34_srat_forward_pipeline(VulkanContext *ctx) {
    if (f34_srat_forward_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = f34_srat_forward_spv_len;
    smci.pCode = (const uint32_t *)f34_srat_forward_spv;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &f34_srat_forward_pipeline) != VK_SUCCESS) return -1;
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_f34_srat_reconstruct_pipeline(VulkanContext *ctx) {
    if (f34_srat_reconstruct_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = f34_srat_reconstruct_spv_len;
    smci.pCode = (const uint32_t *)f34_srat_reconstruct_spv;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &f34_srat_reconstruct_pipeline) != VK_SUCCESS) return -1;
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

void vk_dispatch_f34_srat_forward(VulkanContext *ctx, VkBufferHandle h_in, VkBufferHandle p_r, VkBufferHandle z_out, int seq_len, int d_state, int r) {
    if (!VK_BUFFER_VALID(h_in) || !VK_BUFFER_VALID(p_r) || !VK_BUFFER_VALID(z_out)) return;
    if (ensure_f34_srat_forward_pipeline(ctx) != 0) return;
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    // Bind buffers manually
    vk_bind_buffer_to_descriptor(ctx, slot, 0, h_in);
    vk_bind_buffer_to_descriptor(ctx, slot, 1, p_r);
    vk_bind_buffer_to_descriptor(ctx, slot, 2, z_out);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f34_srat_forward_pipeline);
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct { uint32_t sl; uint32_t ds; uint32_t rad; } pc;
    pc.sl = seq_len;
    pc.ds = d_state;
    pc.rad = r;
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t total = seq_len * r; // batch=1 assumed in loop
    printf("DEBUG: Before vkCmdDispatch in sp12\n"); fflush(stdout);
    vkCmdDispatch(cmd, (total + 255) / 256, 1, 1);
}

void vk_dispatch_f34_srat_reconstruct(VulkanContext *ctx, VkBufferHandle z_in, VkBufferHandle p_r, VkBufferHandle h_out, int seq_len, int d_state, int r) {
    if (!VK_BUFFER_VALID(z_in) || !VK_BUFFER_VALID(p_r) || !VK_BUFFER_VALID(h_out)) return;
    if (ensure_f34_srat_reconstruct_pipeline(ctx) != 0) return;
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vk_bind_buffer_to_descriptor(ctx, slot, 0, z_in);
    vk_bind_buffer_to_descriptor(ctx, slot, 1, p_r);
    vk_bind_buffer_to_descriptor(ctx, slot, 2, h_out);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f34_srat_reconstruct_pipeline);
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct { uint32_t sl; uint32_t ds; uint32_t rad; } pc;
    pc.sl = seq_len;
    pc.ds = d_state;
    pc.rad = r;
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t total = seq_len * d_state;
    printf("DEBUG: Before vkCmdDispatch in sp12\n"); fflush(stdout);
    vkCmdDispatch(cmd, (total + 255) / 256, 1, 1);
}

int ensure_e32_aoi_ssm_pipeline(VulkanContext *ctx) {
    if (e32_aoi_ssm_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = e32_aoi_ssm_spv_len;
    smci.pCode    = (const uint32_t *)e32_aoi_ssm_spv;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;

    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);

    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &e32_aoi_ssm_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_f01_rrd_involution_pipeline(VulkanContext *ctx) {
    if (f01_rrd_involution_pipeline != VK_NULL_HANDLE) return 0;

    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = f01_rrd_involution_spv_len;
    smci.pCode    = (const uint32_t *)f01_rrd_involution_spv;

    VkShaderModule sm;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;

    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);

    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &f01_rrd_involution_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int vk_dispatch_e32_aoi_ssm(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size) {
    printf("DEBUG: Inside vk_dispatch_sp12_lcmbi\n"); fflush(stdout);
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(B_eff) || !VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(h_out)) return -1;
    if (ensure_e32_aoi_ssm_pipeline(ctx) != 0) return -1;

    vk_bind_ssm_scan_buffers(ctx, A_eff, B_eff, x, C_bilinear, h_out);

    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
    }

    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, e32_aoi_ssm_pipeline);

    struct { uint32_t sl; uint32_t ds; uint32_t hb; } pc_data;
    pc_data.sl = seq_len;
    pc_data.ds = d_state;
    pc_data.hb = VK_BUFFER_VALID(C_bilinear) ? 1 : 0;
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_data), &pc_data);

    printf("DEBUG: Before vkCmdDispatch in sp12\n"); fflush(stdout);
    vkCmdDispatch(cmd, (d_state + 255) / 256, batch_size, 1);

    return 0;
}

int vk_dispatch_f01_rrd_involution(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size) {
    printf("DEBUG: Inside vk_dispatch_sp12_lcmbi\n"); fflush(stdout);
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(B_eff) || !VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(h_out)) return -1;
    if (ensure_f01_rrd_involution_pipeline(ctx) != 0) return -1;

    vk_bind_ssm_scan_buffers(ctx, A_eff, B_eff, x, C_bilinear, h_out);

    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }

    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f01_rrd_involution_pipeline);

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

    printf("DEBUG: Before vkCmdDispatch in sp12\n"); fflush(stdout);
    vkCmdDispatch(cmd, (uint32_t)d_state, (uint32_t)batch_size, 1);
    return 0;
}

int ensure_ef03_pipeline(VulkanContext *ctx) {
    if (ef03_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = ef03_ring_checkpoint_spv_len;
    smci.pCode    = (const uint32_t *)ef03_ring_checkpoint_spv;
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
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &ef03_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, comp, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, comp, NULL);
    return 0;
}

int ensure_f38_cst_pipeline(VulkanContext *ctx) {
    if (f38_cst_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = f38_cst_spv_len;
    smci.pCode = (const uint32_t *)f38_cst_spv;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &f38_cst_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_e13_lrms_pipeline(VulkanContext *ctx) {
    if (e13_lrms_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = e13_lrms_spv_len;
    smci.pCode = (const uint32_t *)e13_lrms_spv;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &e13_lrms_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

void vk_dispatch_e13_lrms(VulkanContext *ctx, VkBufferHandle nodes, uint32_t total_elements, float lrms_multiplier) {
    if (!VK_BUFFER_VALID(nodes)) return;
    if (ensure_e13_lrms_pipeline(ctx) != 0) return;
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vk_bind_buffer_to_descriptor(ctx, slot, 0, nodes);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, e13_lrms_pipeline);
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);
    
    struct { uint32_t total_elements; float lrms_multiplier; } pc;
    pc.total_elements = total_elements;
    pc.lrms_multiplier = lrms_multiplier;
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    printf("DEBUG: Before vkCmdDispatch in sp12\n"); fflush(stdout);
    vkCmdDispatch(cmd, (total_elements + 255) / 256, 1, 1);
}

int ensure_f39_bvgr_pipeline(VulkanContext *ctx) {
    if (f39_bvgr_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = f39_bvgr_spv_len;
    smci.pCode = (const uint32_t *)f39_bvgr_spv;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &f39_bvgr_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_f37_taqg_pipeline(VulkanContext *ctx) {
    if (f37_taqg_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = f37_taqg_spv_len;
    smci.pCode    = (const uint32_t *)f37_taqg_spv;
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
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &f37_taqg_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, comp, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, comp, NULL);
    return 0;
}

void vk_dispatch_f37_taqg(VulkanContext *ctx, VkBufferHandle shadow, VkBufferHandle grad, uint32_t N, float lr) {
// //     if (!VK_BUFFER_VALID(shadow) || !VK_BUFFER_VALID(grad)) return;
// //     if (ensure_f37_taqg_pipeline(ctx) != 0) return;
// // 
// //     if (!vk_is_recording(ctx)) {
// //         VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
// //         vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
// //         vk_set_recording(ctx, 1);
// //     }
// // 
// //     vkCmdBindPipeline(vk_get_command_buffer(ctx), VK_PIPELINE_BIND_POINT_COMPUTE, f37_taqg_pipeline);
// //     
// //     VkDescriptorSet ds = vk_allocate_descriptor_set(ctx);
// //     VkDescriptorBufferInfo binfo[2] = {0};
// //     
// //     binfo[0].buffer = shadow; binfo[0].range = VK_WHOLE_SIZE;
// //     binfo[1].buffer = grad;   binfo[1].range = VK_WHOLE_SIZE;
// // 
// //     VkWriteDescriptorSet writes[2] = {0};
// //     for(int i=0; i<2; i++) {
// //         writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
// //         writes[i].dstSet = ds;
// //         writes[i].dstBinding = i;
// //         writes[i].descriptorCount = 1;
// //         writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
// //         writes[i].pBufferInfo = &binfo[i];
// //     }
// //     vkUpdateDescriptorSets(vk_get_device(ctx), 2, writes, 0, NULL);
// //     
// //     vkCmdBindDescriptorSets(vk_get_command_buffer(ctx), VK_PIPELINE_BIND_POINT_COMPUTE,
// //                             vk_get_pipeline_layout(ctx, vk_get_pipe_ssm_scan(ctx)),
// //                             0, 1, &ds, 0, NULL);
// //                             
// //     struct { uint32_t n; float lr; } pc = {N, lr};
// //     vkCmdPushConstants(vk_get_command_buffer(ctx), vk_get_pipeline_layout(ctx, vk_get_pipe_ssm_scan(ctx)),
// //                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
// //                        
// //     vkCmdDispatch(vk_get_command_buffer(ctx), (N + 255) / 256, 1, 1);
}

void vk_dispatch_ef03_ring_checkpoint(VulkanContext *ctx, VkBufferHandle h, VkBufferHandle sum, uint32_t bseq, uint32_t ds, uint32_t K) {
    if (!VK_BUFFER_VALID(h) || !VK_BUFFER_VALID(sum)) return;
    if (ensure_ef03_pipeline(ctx) != 0) return;

    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }

    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ef03_pipeline);

    vk_bind_buffer_to_descriptor(ctx, slot, 0, h);
    vk_bind_buffer_to_descriptor(ctx, slot, 1, sum);

    VkDescriptorSet dset = vk_get_descriptor_set(ctx, slot);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, slot), 0, 1, &dset, 0, NULL);

    struct { uint32_t bseq; uint32_t ds; uint32_t K; } pc;
    pc.bseq = bseq;
    pc.ds = ds;
    pc.K = K;

    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, slot), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);

    uint32_t num_blocks = bseq / K;
    uint32_t total_invocations = num_blocks * ds;
    uint32_t groups = (total_invocations + 255) / 256;
    printf("DEBUG: Before vkCmdDispatch in sp12\n"); fflush(stdout);
    vkCmdDispatch(cmd, groups, 1, 1);
}

int ensure_sp10_pipeline(VulkanContext *ctx) {
    printf("DEBUG: Inside ensure_sp10_pipeline\n"); fflush(stdout);
    if (sp10_psvcte_pipeline != VK_NULL_HANDLE) return 0;
    
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sp10_psvcte_spv_len;
    smci.pCode    = (const uint32_t *)sp10_psvcte_spv;
    
    VkShaderModule sm;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &sp10_psvcte_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_sp11_pipeline(VulkanContext *ctx) {
    if (sp11_lns_tfms_pipeline != VK_NULL_HANDLE) return 0;
    
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sp11_lns_tfms_spv_len;
    smci.pCode    = (const uint32_t *)sp11_lns_tfms_spv;
    
    VkShaderModule sm;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &sp11_lns_tfms_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_sp12_pipeline(VulkanContext *ctx) {
    if (sp12_lcmbi_pipeline != VK_NULL_HANDLE) return 0;
    
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sp12_lcmbi_spv_len;
    smci.pCode    = (const uint32_t *)sp12_lcmbi_spv;
    
    VkShaderModule sm;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &sp12_lcmbi_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int vk_dispatch_sp12_lcmbi(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size) {
    printf("DEBUG: Inside vk_dispatch_sp12_lcmbi\n"); fflush(stdout);
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(B_eff) || !VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(h_out)) return -1;
    if (ensure_sp12_pipeline(ctx) != 0) return -1;
    
    vk_bind_ssm_scan_buffers(ctx, A_eff, B_eff, x, C_bilinear, h_out);
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp12_lcmbi_pipeline);
    
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
    
    printf("DEBUG: Before vkCmdDispatch in sp12\n"); fflush(stdout);
    vkCmdDispatch(cmd, (uint32_t)d_state, (uint32_t)batch_size, 1);
    return 0;
}

int ensure_sp13_pipeline(VulkanContext *ctx) {
    if (sp13_sde_bvdj_pipeline != VK_NULL_HANDLE) return 0;
    
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sp13_sde_bvdj_spv_len;
    smci.pCode    = (const uint32_t *)sp13_sde_bvdj_spv;
    
    VkShaderModule sm;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &sp13_sde_bvdj_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int vk_dispatch_sp13_sde_bvdj(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size) {
    printf("DEBUG: Inside vk_dispatch_sp12_lcmbi\n"); fflush(stdout);
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(B_eff) || !VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(h_out)) return -1;
    if (ensure_sp13_pipeline(ctx) != 0) return -1;
    
    vk_bind_ssm_scan_buffers(ctx, A_eff, B_eff, x, C_bilinear, h_out);
    
    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    int slot = vk_get_pipe_ssm_scan(ctx);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp13_sde_bvdj_pipeline);
    
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
    
    printf("DEBUG: Before vkCmdDispatch in sp12\n"); fflush(stdout);
    vkCmdDispatch(cmd, (uint32_t)d_state, (uint32_t)batch_size, 1);
    return 0;
}

int ensure_sp14_pipeline(VulkanContext *ctx) {
    if (sp14_gfit_pipeline != VK_NULL_HANDLE) return 0;
    
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sp14_gfit_spv_len;
    smci.pCode    = (const uint32_t *)sp14_gfit_spv;
    
    VkShaderModule sm;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &sp14_gfit_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_sp15_pipeline(VulkanContext *ctx) {
    if (sp15_tt_rcrb_pipeline != VK_NULL_HANDLE) return 0;
    
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sp15_tt_rcrb_spv_len;
    smci.pCode    = (const uint32_t *)sp15_tt_rcrb_spv;
    
    VkShaderModule sm;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &sp15_tt_rcrb_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_f12_fourier_pipeline(VulkanContext *ctx) {
    if (f12_fourier_pipeline != VK_NULL_HANDLE) return 0;
    
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = f12_fourier_ssm_spv_len;
    smci.pCode    = (const uint32_t *)f12_fourier_ssm_spv;
    
    VkShaderModule sm;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    // We need a layout with 2 buffers. sp11 uses 3 buffers. ssm_scan uses many.
    // Let's reuse sp11 layout or any generic layout.
    // Actually, vk_get_pipe_ssm_scan(ctx) provides a layout with up to 22 buffers.
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &f12_fourier_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int vk_dispatch_f12_fourier_ssm(VulkanContext *ctx, VkBufferHandle h_cache, int seq_len, int d_state, int batch_size) {
    if (ensure_f12_fourier_pipeline(ctx) != 0) return -1;
    if (!VK_BUFFER_VALID(h_cache)) return -1;
    
    if (!VK_BUFFER_VALID(f12_aux_buffer)) {
        f12_aux_buffer = vk_alloc(ctx, (size_t)batch_size * seq_len * d_state * 2 * sizeof(float), VK_BUF_STORAGE);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f12_fourier_pipeline);
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    vk_bind_buffer_to_descriptor(ctx, pipe_idx, 0, h_cache);
    vk_bind_buffer_to_descriptor(ctx, pipe_idx, 1, f12_aux_buffer);
    
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, pipe_idx);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            vk_get_pipeline_layout(ctx, pipe_idx),
                            0, 1, &dset, 0, NULL);
                            
    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t batch_size;
    } pc;
    pc.seq_len = (uint32_t)seq_len;
    pc.d_state = (uint32_t)d_state;
    pc.batch_size = (uint32_t)batch_size;
    
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, vk_get_pipe_ssm_scan(ctx)), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    uint32_t threads = (uint32_t)(batch_size * d_state);
    uint32_t groups = (threads + 63) / 64;
    printf("DEBUG: Before vkCmdDispatch in sp12\n"); fflush(stdout);
    vkCmdDispatch(cmd, groups, 1, 1);
    return 0;
}

int ensure_e13_parametric_cotangent_pipeline(VulkanContext *ctx) {

    if (e13_parametric_cotangent_pipeline != VK_NULL_HANDLE) return 0;
    
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = e13_parametric_cotangent_spv_len;
    smci.pCode    = (const uint32_t *)e13_parametric_cotangent_spv;
    
    VkShaderModule sm;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &e13_parametric_cotangent_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int vk_dispatch_e13_parametric_cotangent(VulkanContext *ctx, VkBufferHandle x_proj, VkBufferHandle w_pred, VkBufferHandle lambda_out, int seq_len, int d_state, int batch_size) {
    if (ensure_e13_parametric_cotangent_pipeline(ctx) != 0) return -1;
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, e13_parametric_cotangent_pipeline);
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    vk_bind_buffer_to_descriptor(ctx, pipe_idx, 0, x_proj);
    vk_bind_buffer_to_descriptor(ctx, pipe_idx, 1, w_pred);
    vk_bind_buffer_to_descriptor(ctx, pipe_idx, 2, lambda_out);
    
    VkDescriptorSet dset = vk_get_descriptor_set(ctx, pipe_idx);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            vk_get_pipeline_layout(ctx, pipe_idx),
                            0, 1, &dset, 0, NULL);
                            
    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t batch_size;
    } pc;
    pc.seq_len = (uint32_t)seq_len;
    pc.d_state = (uint32_t)d_state;
    pc.batch_size = (uint32_t)batch_size;
    
    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, pipe_idx), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vk_profile_mark_dispatch(ctx);
    
    uint32_t total_items = batch_size * seq_len * d_state;
    uint32_t groups = (total_items + 63) / 64;
    printf("DEBUG: Before vkCmdDispatch in sp12\n"); fflush(stdout);
    vkCmdDispatch(cmd, groups, 1, 1);
    return 0;
}

int ensure_sp16_pipeline(VulkanContext *ctx) {
    if (sp16_nsp_pipeline != VK_NULL_HANDLE) return 0;
    
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sp16_null_space_prune_spv_len;
    smci.pCode    = (const uint32_t *)sp16_null_space_prune_spv;
    
    VkShaderModule sm;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx); // Reuse any compatible pipeline layout (4 buffers, 1 pc)
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &sp16_nsp_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_f14_pipeline(VulkanContext *ctx) {
    if (f14_tae_pipeline != VK_NULL_HANDLE) return 0;
    
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = f14_ternary_activation_escrow_spv_len;
    smci.pCode    = (const uint32_t *)f14_ternary_activation_escrow_spv;
    
    VkShaderModule sm;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx); 
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &f14_tae_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_sp17_pipeline(VulkanContext *ctx) {
    if (sp17_dtss_pipeline != VK_NULL_HANDLE) return 0;
    
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sp17_dtss_spv_len;
    smci.pCode    = (const uint32_t *)sp17_dtss_spv;
    
    VkShaderModule sm;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &sp17_dtss_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_f15_pipeline(VulkanContext *ctx) {
    if (f15_cross_layer_adjoint_pipeline != VK_NULL_HANDLE) return 0;
    
    VkDevice device = vk_get_device(ctx);
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = f15_cross_layer_adjoint_spv_len;
    smci.pCode    = (const uint32_t *)f15_cross_layer_adjoint_spv;
    
    VkShaderModule sm;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    int pipe_idx = vk_get_pipe_ssm_scan(ctx);
    VkPipelineLayout layout = vk_get_pipeline_layout(ctx, pipe_idx);
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &f15_cross_layer_adjoint_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

int ensure_btrp_token_gather_pipeline(VulkanContext *ctx) {
    if (btrp_token_gather_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = btrp_token_gather_spv_len;
    smci.pCode = (const uint32_t *)btrp_token_gather_spv;
    if (vkCreateShaderModule(device, &smci, NULL, &sm) != VK_SUCCESS) return -1;
    
    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = vk_get_global_pipeline_layout(ctx);
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &btrp_token_gather_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

void vk_dispatch_btrp_token_gather(VulkanContext *ctx, VkBufferHandle embed_E, VkBufferHandle tokens, VkBufferHandle out_rows, uint32_t seq_len, uint32_t d_embed, uint32_t batch_size, uint32_t vocab_size, uint32_t r) {
    if (ensure_btrp_token_gather_pipeline(ctx) != 0) return;
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, btrp_token_gather_pipeline);
    vk_bind_buffer_to_descriptor(ctx, 0, 0, embed_E);
    vk_bind_buffer_to_descriptor(ctx, 0, 1, tokens);
    vk_bind_buffer_to_descriptor(ctx, 0, 2, out_rows);
    
    VkPipelineLayout layout = vk_get_global_pipeline_layout(ctx);
    VkDescriptorSet set = vk_get_descriptor_set(ctx, 0);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    
    uint32_t pc[4] = { seq_len, d_embed, vocab_size, r };
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    printf("DEBUG: Before vkCmdDispatch in sp12\n"); fflush(stdout);
    vkCmdDispatch(cmd, (seq_len + 127) / 128, batch_size, 1);
}

#endif // V12_HAS_VULKAN

