static VkPipeline f14_sose_pipeline = VK_NULL_HANDLE;
static VkPipeline f14_sose_backward_pipeline = VK_NULL_HANDLE;

static int ensure_f14_sose_pipeline(VulkanContext *ctx) {
    if (f14_sose_pipeline != VK_NULL_HANDLE) return 0;
    VkDevice device = vk_get_device(ctx);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = ssm_scan_spv_len;
    smci.pCode = (const uint32_t *)ssm_scan_spv;
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
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &f14_sose_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, sm, NULL);
        return -1;
    }
    vkDestroyShaderModule(device, sm, NULL);
    return 0;
}

static void vk_dispatch_f14_sose(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bl, VkBufferHandle h_out, int seq_len, int d_state, int batch_size) {
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(B_eff) || !VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(h_out)) return;
    if (ensure_f14_sose_pipeline(ctx) != 0) return;

    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f14_sose_pipeline);

    VkDescriptorSet set = vk_get_descriptor_set(ctx, vk_get_pipe_ssm_scan(ctx));
    vk_update_descriptor_set_buffer(ctx, set, 0, A_eff);
    vk_update_descriptor_set_buffer(ctx, set, 1, B_eff);
    vk_update_descriptor_set_buffer(ctx, set, 2, x);
    vk_update_descriptor_set_buffer(ctx, set, 3, C_bl);
    vk_update_descriptor_set_buffer(ctx, set, 4, h_out);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_get_pipeline_layout(ctx, vk_get_pipe_ssm_scan(ctx)), 0, 1, &set, 0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t has_bilinear;
    } pc = {(uint32_t)seq_len, (uint32_t)d_state, C_bl != VK_NULL_HANDLE ? 1 : 0};

    vkCmdPushConstants(cmd, vk_get_pipeline_layout(ctx, vk_get_pipe_ssm_scan(ctx)), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vk_profile_mark_dispatch(ctx);
    vkCmdDispatch(cmd, (d_state + 255) / 256, batch_size, 1);
}
