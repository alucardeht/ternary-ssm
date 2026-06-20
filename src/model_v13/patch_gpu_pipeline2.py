import os

path = "src/gpu/gpu_pipeline.c"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Define the C code for pipeline and dispatch
sp11_code = """
static VkPipeline sp11_lns_tfms_pipeline = VK_NULL_HANDLE;

static int ensure_sp11_pipeline(VulkanContext *ctx) {
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

static __attribute__((unused)) int vk_dispatch_sp11_lns_tfms(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out, int seq_len, int d_state, int batch_size) {
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(B_eff) || !VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(h_out)) return -1;
    if (ensure_sp11_pipeline(ctx) != 0) return -1;
    
    vk_bind_ssm_scan_buffers(ctx, A_eff, B_eff, x, C_bilinear, h_out);
    
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
"""

if "sp11_lns_tfms_pipeline" not in content:
    content = content.replace("static VkPipeline sp12_lcmbi_pipeline", sp11_code + "\nstatic VkPipeline sp12_lcmbi_pipeline")

dispatch_block = """#elif R504_S11_LNS_TFMS
            vk_dispatch_sp11_lns_tfms(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);"""

if "R504_S11_LNS_TFMS" not in content:
    content = content.replace("#elif R504_S14_GFIT", dispatch_block + "\n#elif R504_S14_GFIT")

# Define R504_S11_LNS_TFMS
if "#define R504_S11_LNS_TFMS" not in content:
    content = content.replace("#define R503_PHASE_1_EFFICIENCY 1", "#define R503_PHASE_1_EFFICIENCY 1\n#define R504_S11_LNS_TFMS 1")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
