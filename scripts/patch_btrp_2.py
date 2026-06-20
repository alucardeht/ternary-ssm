import os
import re
import subprocess

# 1. Compile the BTRP shader we already created
print("Compiling BTRP shader...")
subprocess.run(["C:\\VulkanSDK\\1.4.341.1\\Bin\\glslc.exe", "-O", 
                "src/model_v12/src/runtime/shaders/btrp_token_gather.comp", 
                "-o", "src/model_v12/src/runtime/shaders/spv/btrp_token_gather.spv"], check=True)
subprocess.run(["python", "src/model_v12/tools/spv_to_header.py", 
                "src/model_v12/src/runtime/shaders/spv/btrp_token_gather.spv", 
                "src/model_v12/src/runtime/shaders/btrp_token_gather_spv.h"], check=True)

# 2. Patch gpu_pipeline.c
with open("src/model_v12/src/gpu/gpu_pipeline.c", "r", encoding="utf-8") as f:
    code = f.read()

# Add include and pipeline code
pipeline_code = """
#include "../runtime/shaders/btrp_token_gather_spv.h"

static VkPipeline btrp_token_gather_pipeline = VK_NULL_HANDLE;

static int ensure_btrp_token_gather_pipeline(VulkanContext *ctx) {
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

static void vk_dispatch_btrp_token_gather(VulkanContext *ctx, VkBufferHandle embed_E, VkBufferHandle tokens, VkBufferHandle out_rows, uint32_t seq_len, uint32_t d_embed, uint32_t batch_size, uint32_t vocab_size, uint32_t r) {
    if (ensure_btrp_token_gather_pipeline(ctx) != 0) return;
    VkCommandBuffer cmd = vk_get_command_buffer(ctx);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, btrp_token_gather_pipeline);
    vk_bind_buffer_to_descriptor(ctx, 0, 0, embed_E);
    vk_bind_buffer_to_descriptor(ctx, 0, 1, tokens);
    vk_bind_buffer_to_descriptor(ctx, 0, 2, out_rows);
    
    uint32_t pc[4] = { seq_len, d_embed, vocab_size, r };
    vkCmdPushConstants(cmd, vk_get_global_pipeline_layout(ctx), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cmd, (seq_len + 127) / 128, batch_size, 1);
}
"""

if "vk_dispatch_btrp_token_gather" not in code:
    # Inject right before gpu_pipeline_create
    code = code.replace("GpuPipeline *gpu_pipeline_create(", pipeline_code + "\nGpuPipeline *gpu_pipeline_create(")

# Replace the original token_gather call with BTRP dispatch if R504_E11_BTRP is defined
orig_dispatch = r"""vk_dispatch_token_gather\(ctx,
                                 p->embed_E_gpu, p->embed_tokens, p->embed_gathered,
                                 \(uint32_t\)seq_len, \(uint32_t\)V12_D_EMBED, batch_size\);"""
new_dispatch = """#ifdef R504_E11_BTRP
        vk_dispatch_btrp_token_gather(ctx, p->embed_E_gpu, p->embed_tokens, p->embed_gathered, (uint32_t)seq_len, (uint32_t)V12_D_EMBED, batch_size, m->vocab_size, BTRP_R);
#else
        vk_dispatch_token_gather(ctx,
                                 p->embed_E_gpu, p->embed_tokens, p->embed_gathered,
                                 (uint32_t)seq_len, (uint32_t)V12_D_EMBED, batch_size);
#endif"""
code = re.sub(orig_dispatch, new_dispatch, code, count=1)

with open("src/model_v12/src/gpu/gpu_pipeline.c", "w", encoding="utf-8") as f:
    f.write(code)
print("gpu_pipeline.c modified successfully.")
