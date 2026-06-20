import os
import re

gpu_c = r"c:\Users\th1rt\Documents\Workspaces\code-ai-workspace\src\model_v12\src\gpu\gpu_pipeline.c"
makefile = r"c:\Users\th1rt\Documents\Workspaces\code-ai-workspace\src\model_v12\Makefile"

# 1. Update Makefile to include f12_fourier_ssm compilation
with open(makefile, "r", encoding="utf-8") as f:
    make_content = f.read()

if "f12_fourier_ssm.comp" not in make_content:
    make_patch = """\t$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/f12_fourier_ssm.comp -o src/runtime/shaders/spv/f12_fourier_ssm.spv
\tpython tools/spv_to_header.py src/runtime/shaders/spv/f12_fourier_ssm.spv src/runtime/shaders/f12_fourier_ssm_spv.h
"""
    # Insert it right before the last $(GLSLC) command in the shaders target
    idx = make_content.rfind("\t$(GLSLC)")
    make_content = make_content[:idx] + make_patch + make_content[idx:]
    with open(makefile, "w", encoding="utf-8") as f:
        f.write(make_content)

# 2. Patch gpu_pipeline.c
with open(gpu_c, "r", encoding="utf-8") as f:
    c_content = f.read()

if "f12_fourier_ssm_spv.h" not in c_content:
    # insert include
    include_str = '#include "../runtime/shaders/f12_fourier_ssm_spv.h"\n'
    idx_inc = c_content.find('#include "../runtime/shaders/sp11_tmsheaf_spv.h"')
    if idx_inc == -1: idx_inc = c_content.find('#include "../runtime/shaders/')
    c_content = c_content[:idx_inc] + include_str + c_content[idx_inc:]

if "vk_dispatch_f12_fourier_ssm" not in c_content:
    # insert vk_dispatch function
    dispatch_code = """
static VkPipeline f12_fourier_pipeline = VK_NULL_HANDLE;
static VkBufferHandle f12_aux_buffer = {0};

static int ensure_f12_fourier_pipeline(VulkanContext *ctx) {
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

static int vk_dispatch_f12_fourier_ssm(VulkanContext *ctx, VkBufferHandle h_cache, int seq_len, int d_state, int batch_size) {
    if (ensure_f12_fourier_pipeline(ctx) != 0) return -1;
    if (!VK_BUFFER_VALID(h_cache)) return -1;
    
    if (!VK_BUFFER_VALID(f12_aux_buffer)) {
        f12_aux_buffer = vk_alloc(ctx, (size_t)batch_size * seq_len * d_state * 2 * sizeof(float), VK_BUF_STORAGE);
    }
    
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, f12_fourier_pipeline);
    
    VkDescriptorBufferInfo binfo[2] = {0};
    binfo[0].buffer = h_cache.buffer;
    binfo[0].offset = h_cache.offset;
    binfo[0].range  = h_cache.size;
    
    binfo[1].buffer = f12_aux_buffer.buffer;
    binfo[1].offset = f12_aux_buffer.offset;
    binfo[1].range  = f12_aux_buffer.size;
    
    VkWriteDescriptorSet writes[2] = {0};
    for (int i=0; i<2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = vk_get_descriptor_set(ctx, vk_get_pipe_ssm_scan(ctx));
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &binfo[i];
    }
    vkUpdateDescriptorSets(vk_get_device(ctx), 2, writes, 0, NULL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            vk_get_pipeline_layout(ctx, vk_get_pipe_ssm_scan(ctx)),
                            0, 1, &writes[0].dstSet, 0, NULL);
                            
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
    vkCmdDispatch(cmd, groups, 1, 1);
    return 0;
}
"""
    # Insert before ensure_sp16_pipeline
    idx_pipe = c_content.find("static VkPipeline sp16_nsp_pipeline")
    if idx_pipe != -1:
        c_content = c_content[:idx_pipe] + dispatch_code + c_content[idx_pipe:]

    # Insert the dispatch call right after sp11
    call_code = """
            /* Finding 12: Frequency-Domain State Compression (Fourier SSM) */
            vk_dispatch_f12_fourier_ssm(ctx, p->layers[k].fwd_h_cache, seq_len, ds, batch_size);
            vk_barrier(ctx);
"""
    idx_call = c_content.find("vk_dispatch_sp11_tmsheaf(")
    if idx_call != -1:
        # find end of vk_barrier(ctx); after it
        idx_call_end = c_content.find("}", idx_call)
        c_content = c_content[:idx_call_end+1] + call_code + c_content[idx_call_end+1:]

with open(gpu_c, "w", encoding="utf-8") as f:
    f.write(c_content)

print("Patch applied.")
