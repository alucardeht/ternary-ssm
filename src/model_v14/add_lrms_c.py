import re

with open('src/gpu/gpu_pipeline.c', 'r') as f:
    content = f.read()

# 1. Include header
content = content.replace(
    '#include "../runtime/shaders/e13_parametric_cotangent_spv.h"',
    '#include "../runtime/shaders/e13_parametric_cotangent_spv.h"\n#include "../runtime/shaders/e13_lrms_spv.h"'
)

# 2. Add pipeline functions
pipeline_code = """
static VkPipeline e13_lrms_pipeline = VK_NULL_HANDLE;

static int ensure_e13_lrms_pipeline(VulkanContext *ctx) {
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

static void vk_dispatch_e13_lrms(VulkanContext *ctx, VkBufferHandle nodes, uint32_t total_elements, float lrms_multiplier) {
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
    
    vkCmdDispatch(cmd, (total_elements + 255) / 256, 1, 1);
}
"""

content = content.replace(
    'static VkPipeline f39_bvgr_pipeline = VK_NULL_HANDLE;',
    pipeline_code + '\nstatic VkPipeline f39_bvgr_pipeline = VK_NULL_HANDLE;'
)

# 3. Add to gpu_train_step
# We'll calculate the lrms_multiplier and apply the dispatch to residual[V12_N_LAYERS], and add to loss.
# We will insert it right before Phase 4: Download final residual.
patch_lrms = """
    /* 13. Lagrangian-Relaxed Memory Substrate (LRMS) */
    float lrms_penalty = 0.0f;
    {
        size_t mem_used = vk_allocated_bytes(ctx);
        float limit = 10.8f * 1024.0f * 1024.0f * 1024.0f;
        float overage = (float)mem_used - limit;
        if (overage > 0.0f) {
            float lambda_mem = 1.0e-9f; // soft-routing scale
            float multiplier = lambda_mem * overage;
            if (multiplier > 1.0f) multiplier = 1.0f;
            
            // Shift full memory nodes to sparse residual settlement approximations
            vk_dispatch_e13_lrms(ctx, p->residual[V12_N_LAYERS], (uint32_t)(bseq * d), multiplier);
            
            lrms_penalty = multiplier;
        }
    }

    /* Phase 4: Download final residual"""

content = content.replace(
    '    /* Phase 4: Download final residual',
    patch_lrms
)

# 4. Add the penalty to the returned loss.
# Replace `return loss;` with `return loss + lrms_penalty;`
# But we need `lrms_penalty` in scope. We can just compute it again or make it available.
# Wait, let's just make it a variable at the top of gpu_train_step.

content = content.replace(
    'float gpu_train_step(GpuPipeline *p, Model *m, Cache *c,\n                     const uint16_t *tokens, const uint8_t *boundary_mask,\n                     int seq_len, int seq_idx, int batch_size,\n                     const TrainState *state) {',
    'float gpu_train_step(GpuPipeline *p, Model *m, Cache *c,\n                     const uint16_t *tokens, const uint8_t *boundary_mask,\n                     int seq_len, int seq_idx, int batch_size,\n                     const TrainState *state) {\n    float lrms_penalty_global = 0.0f;'
)

content = content.replace(
    'lrms_penalty = multiplier;',
    'lrms_penalty = multiplier;\n            lrms_penalty_global = lrms_penalty;'
)

content = content.replace(
    '    return loss;',
    '    return loss + lrms_penalty_global;'
)

with open('src/gpu/gpu_pipeline.c', 'w') as f:
    f.write(content)
