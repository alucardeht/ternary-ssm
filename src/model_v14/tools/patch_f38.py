import re
import sys
import os

MAKEFILE_PATH = "Makefile"
GPU_PIPELINE_PATH = "src/gpu/gpu_pipeline.c"

def patch_makefile():
    with open(MAKEFILE_PATH, "r") as f:
        content = f.read()

    if "f38_cst.comp" in content:
        print("Makefile already patched.")
        return

    # Add compilation rule for f38_cst.comp
    patch_str = """\t$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/f38_cst.comp -o src/runtime/shaders/spv/f38_cst.spv
\tpython tools/spv_to_header.py src/runtime/shaders/spv/f38_cst.spv src/runtime/shaders/f38_cst_spv.h
endif"""
    
    new_content = content.replace("endif\n\n\nall: train", patch_str + "\n\n\nall: train")
    if new_content == content:
        # Fallback
        new_content = content.replace("endif\n\nall: train", patch_str + "\n\nall: train")
        
    with open(MAKEFILE_PATH, "w") as f:
        f.write(new_content)
    print("Makefile patched.")

def patch_gpu_pipeline():
    with open(GPU_PIPELINE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    if "f38_cst_spv.h" in content:
        print("gpu_pipeline.c already patched.")
        return

    # 1. Include header
    include_str = '#include "../runtime/shaders/f38_cst_spv.h"\n'
    if '#include "../runtime/shaders/f37_taqg_spv.h"' in content:
        content = content.replace(
            '#include "../runtime/shaders/f37_taqg_spv.h"\n',
            '#include "../runtime/shaders/f37_taqg_spv.h"\n' + include_str
        )
    else:
        # Just find the last #include "../runtime/shaders/..." and add it
        includes = re.findall(r'#include "\.\./runtime/shaders/.*_spv\.h"\n', content)
        if includes:
            content = content.replace(includes[-1], includes[-1] + include_str)

    # 2. Add pipeline variable and functions
    dispatch_code = """
static VkPipeline f38_cst_pipeline = VK_NULL_HANDLE;

static int ensure_f38_cst_pipeline(VulkanContext *ctx) {
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

static __attribute__((unused)) void vk_dispatch_f38_cst(VulkanContext *ctx, VkBufferHandle h_buf, VkBufferHandle active_flags, VkBufferHandle lambda_buf, int seq_len, int d_state, float threshold, int mode) {
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
"""

    decl_match = re.search(r'static VkPipeline f37_taqg_pipeline = VK_NULL_HANDLE;\n', content)
    if decl_match:
        content = content.replace(decl_match.group(0), decl_match.group(0) + dispatch_code)
    else:
        # Fallback
        pipe_decls = re.findall(r'static VkPipeline [a-zA-Z0-9_]+_pipeline = VK_NULL_HANDLE;\n', content)
        if pipe_decls:
             content = content.replace(pipe_decls[-1], pipe_decls[-1] + dispatch_code)

    with open(GPU_PIPELINE_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("gpu_pipeline.c patched.")

if __name__ == "__main__":
    patch_makefile()
    patch_gpu_pipeline()
