import os
import re
import subprocess

with open("src/gpu/gpu_pipeline.c", "r") as f:
    code = f.read()

# Add include
if "f37_taqg_spv.h" not in code:
    code = code.replace(
        '#include "../runtime/shaders/sp19_holo_adam_reduce_spv.h"',
        '#include "../runtime/shaders/sp19_holo_adam_reduce_spv.h"\n#include "../runtime/shaders/f37_taqg_spv.h"'
    )

# Add vk_dispatch_f37_taqg
dispatch_func = """
static VkPipeline f37_taqg_pipeline = VK_NULL_HANDLE;

static int ensure_f37_taqg_pipeline(VulkanContext *ctx) {
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
    if (!VK_BUFFER_VALID(shadow) || !VK_BUFFER_VALID(grad)) return;
    if (ensure_f37_taqg_pipeline(ctx) != 0) return;

    if (!vk_is_recording(ctx)) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(vk_get_command_buffer(ctx), &cbbi);
        vk_set_recording(ctx, 1);
    }

    vkCmdBindPipeline(vk_get_command_buffer(ctx), VK_PIPELINE_BIND_POINT_COMPUTE, f37_taqg_pipeline);
    
    VkDescriptorSet ds = vk_allocate_descriptor_set(ctx);
    VkDescriptorBufferInfo binfo[2] = {0};
    
    binfo[0].buffer = shadow; binfo[0].range = VK_WHOLE_SIZE;
    binfo[1].buffer = grad;   binfo[1].range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[2] = {0};
    for(int i=0; i<2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &binfo[i];
    }
    vkUpdateDescriptorSets(vk_get_device(ctx), 2, writes, 0, NULL);
    
    vkCmdBindDescriptorSets(vk_get_command_buffer(ctx), VK_PIPELINE_BIND_POINT_COMPUTE,
                            vk_get_pipeline_layout(ctx, vk_get_pipe_ssm_scan(ctx)),
                            0, 1, &ds, 0, NULL);
                            
    struct { uint32_t n; float lr; } pc = {N, lr};
    vkCmdPushConstants(vk_get_command_buffer(ctx), vk_get_pipeline_layout(ctx, vk_get_pipe_ssm_scan(ctx)),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                       
    vkCmdDispatch(vk_get_command_buffer(ctx), (N + 255) / 256, 1, 1);
}
"""

if "vk_dispatch_f37_taqg" not in code:
    code = code.replace(
        "void vk_dispatch_sp19_holo_adam_reduce(",
        dispatch_func + "\nvoid vk_dispatch_sp19_holo_adam_reduce("
    )

# Replace the dispatch in gpu_factored_adam_step_ternary_matrix
call_site = "vk_dispatch_entangled_adam_update(p->ctx, shadow, grad, n, lr, 0.9f, 0.999f, 1e-8f);"
replacement = """
        // Finding 37: Ternary-Algebraic Quotient Gradients (TAQG)
        // Dispatches to GF(3) proxy space quotient accumulator.
        vk_dispatch_f37_taqg(p->ctx, shadow, grad, n, lr);
"""
if "vk_dispatch_f37_taqg(p->ctx" not in code:
    code = code.replace(call_site, replacement)

with open("src/gpu/gpu_pipeline.c", "w") as f:
    f.write(code)

print("gpu_pipeline.c patched")
