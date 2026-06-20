import sys

def patch_header():
    with open("src/runtime/vulkan_backend.h", "r") as f:
        content = f.read()
    
    if "vk_dispatch_vsa_fwht_forward" not in content:
        insert_idx = content.find("VkPipelineLayout vk_get_global_pipeline_layout(VulkanContext *ctx);")
        
        decls = """
void vk_dispatch_vsa_fwht_forward(VulkanContext *ctx, VkBufferHandle state_in, VkBufferHandle state_out, uint32_t num_tokens, uint32_t ds);
void vk_dispatch_vsa_bent_recurrence(VulkanContext *ctx, VkBufferHandle state_in, VkBufferHandle key_in, VkBufferHandle state_out, uint32_t num_tokens, uint32_t ds);
void vk_dispatch_vsa_fractional_laplacian(VulkanContext *ctx, VkBufferHandle in_data, VkBufferHandle alpha_in, VkBufferHandle out_data, uint32_t num_tokens, uint32_t ds);
"""
        content = content[:insert_idx] + decls + content[insert_idx:]
        with open("src/runtime/vulkan_backend.h", "w") as f:
            f.write(content)
        print("Patched vulkan_backend.h")

def patch_impl():
    with open("src/runtime/vulkan_backend_impl.c", "r") as f:
        content = f.read()

    if "vk_dispatch_vsa_fwht_forward" not in content:
        insert_idx = content.find("VkCommandBuffer vk_get_command_buffer(VulkanContext *ctx)")
        
        impl = """
#include "shaders/vsa_fwht_forward_spv.h"
#include "shaders/vsa_bent_recurrence_spv.h"
#include "shaders/vsa_fractional_laplacian_spv.h"

// ----------------------------------------------------
// VSA Fractional Engine Dispatchers
// ----------------------------------------------------

// 1. FWHT
static int vk_create_vsa_fwht_forward_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count++;
    ctx->pipe_vsa_fwht_forward = slot;

    VkDescriptorSetLayoutBinding bindings[2] = {0};
    bindings[0].binding = 0; bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bindings[0].descriptorCount = 1; bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1; bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bindings[1].descriptorCount = 1; bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layout_info.bindingCount = 2; layout_info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &ctx->desc_layouts[slot]) != VK_SUCCESS) return -1;

    VkPushConstantRange pc_range = {0}; pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pc_range.offset = 0; pc_range.size = 2 * sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pipeline_layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_info.setLayoutCount = 1; pipeline_layout_info.pSetLayouts = &ctx->desc_layouts[slot];
    pipeline_layout_info.pushConstantRangeCount = 1; pipeline_layout_info.pPushConstantRanges = &pc_range;
    if (vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL, &ctx->pipeline_layouts[slot]) != VK_SUCCESS) return -1;

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = vsa_fwht_forward_spv_len; smci.pCode = (const uint32_t *)vsa_fwht_forward_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL, &ctx->shader_modules[slot]) != VK_SUCCESS) return -1;

    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot]; cpci.stage.pName = "main"; cpci.layout = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, &ctx->pipelines[slot]) != VK_SUCCESS) return -1;

    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
    VkDescriptorPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pool_info.poolSizeCount = 1; pool_info.pPoolSizes = &pool_size; pool_info.maxSets = 1;
    if (vkCreateDescriptorPool(ctx->device, &pool_info, NULL, &ctx->desc_pools[slot]) != VK_SUCCESS) return -1;

    VkDescriptorSetAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    alloc_info.descriptorPool = ctx->desc_pools[slot]; alloc_info.descriptorSetCount = 1; alloc_info.pSetLayouts = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &alloc_info, &ctx->desc_sets[slot]) != VK_SUCCESS) return -1;

    return 0;
}

void vk_dispatch_vsa_fwht_forward(VulkanContext *ctx, VkBufferHandle state_in, VkBufferHandle state_out, uint32_t num_tokens, uint32_t ds) {
    if (!ctx || !ctx->recording) return;
    uint32_t slot = ctx->pipe_vsa_fwht_forward;

    VkDescriptorBufferInfo b_info[2] = {
        {ctx->buffers[state_in.index], 0, ctx->buf_sizes[state_in.index]},
        {ctx->buffers[state_out.index], 0, ctx->buf_sizes[state_out.index]}
    };
    VkWriteDescriptorSet writes[2] = {0};
    for(int i=0; i<2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = ctx->desc_sets[slot];
        writes[i].dstBinding = i; writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &b_info[i];
    }
    vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);
    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);
    uint32_t pc[2] = {num_tokens, ds};
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(ctx->cmd_buf, num_tokens, 1, 1);
}

// 2. Bent Recurrence
static int vk_create_vsa_bent_recurrence_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count++; ctx->pipe_vsa_bent_recurrence = slot;
    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for(int i=0;i<3;i++) { bindings[i].binding = i; bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bindings[i].descriptorCount = 1; bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layout_info.bindingCount = 3; layout_info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &ctx->desc_layouts[slot]) != VK_SUCCESS) return -1;
    VkPushConstantRange pc_range = {0}; pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pc_range.offset = 0; pc_range.size = 2 * sizeof(uint32_t);
    VkPipelineLayoutCreateInfo pipeline_layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_info.setLayoutCount = 1; pipeline_layout_info.pSetLayouts = &ctx->desc_layouts[slot];
    pipeline_layout_info.pushConstantRangeCount = 1; pipeline_layout_info.pPushConstantRanges = &pc_range;
    if (vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL, &ctx->pipeline_layouts[slot]) != VK_SUCCESS) return -1;
    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = vsa_bent_recurrence_spv_len; smci.pCode = (const uint32_t *)vsa_bent_recurrence_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL, &ctx->shader_modules[slot]) != VK_SUCCESS) return -1;
    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot]; cpci.stage.pName = "main"; cpci.layout = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, &ctx->pipelines[slot]) != VK_SUCCESS) return -1;
    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pool_info.poolSizeCount = 1; pool_info.pPoolSizes = &pool_size; pool_info.maxSets = 1;
    if (vkCreateDescriptorPool(ctx->device, &pool_info, NULL, &ctx->desc_pools[slot]) != VK_SUCCESS) return -1;
    VkDescriptorSetAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    alloc_info.descriptorPool = ctx->desc_pools[slot]; alloc_info.descriptorSetCount = 1; alloc_info.pSetLayouts = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &alloc_info, &ctx->desc_sets[slot]) != VK_SUCCESS) return -1;
    return 0;
}

void vk_dispatch_vsa_bent_recurrence(VulkanContext *ctx, VkBufferHandle state_in, VkBufferHandle key_in, VkBufferHandle state_out, uint32_t num_tokens, uint32_t ds) {
    if (!ctx || !ctx->recording) return;
    uint32_t slot = ctx->pipe_vsa_bent_recurrence;
    VkDescriptorBufferInfo b_info[3] = {
        {ctx->buffers[state_in.index], 0, ctx->buf_sizes[state_in.index]},
        {ctx->buffers[key_in.index], 0, ctx->buf_sizes[key_in.index]},
        {ctx->buffers[state_out.index], 0, ctx->buf_sizes[state_out.index]}
    };
    VkWriteDescriptorSet writes[3] = {0};
    for(int i=0; i<3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = ctx->desc_sets[slot];
        writes[i].dstBinding = i; writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &b_info[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);
    uint32_t pc[2] = {num_tokens, ds};
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(ctx->cmd_buf, num_tokens, 1, 1);
}

// 3. Fractional Laplacian
static int vk_create_vsa_fractional_laplacian_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count++; ctx->pipe_vsa_fractional_laplacian = slot;
    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for(int i=0;i<3;i++) { bindings[i].binding = i; bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bindings[i].descriptorCount = 1; bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layout_info.bindingCount = 3; layout_info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &ctx->desc_layouts[slot]) != VK_SUCCESS) return -1;
    VkPushConstantRange pc_range = {0}; pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pc_range.offset = 0; pc_range.size = 2 * sizeof(uint32_t);
    VkPipelineLayoutCreateInfo pipeline_layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_info.setLayoutCount = 1; pipeline_layout_info.pSetLayouts = &ctx->desc_layouts[slot];
    pipeline_layout_info.pushConstantRangeCount = 1; pipeline_layout_info.pPushConstantRanges = &pc_range;
    if (vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL, &ctx->pipeline_layouts[slot]) != VK_SUCCESS) return -1;
    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = vsa_fractional_laplacian_spv_len; smci.pCode = (const uint32_t *)vsa_fractional_laplacian_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL, &ctx->shader_modules[slot]) != VK_SUCCESS) return -1;
    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot]; cpci.stage.pName = "main"; cpci.layout = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, &ctx->pipelines[slot]) != VK_SUCCESS) return -1;
    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pool_info.poolSizeCount = 1; pool_info.pPoolSizes = &pool_size; pool_info.maxSets = 1;
    if (vkCreateDescriptorPool(ctx->device, &pool_info, NULL, &ctx->desc_pools[slot]) != VK_SUCCESS) return -1;
    VkDescriptorSetAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    alloc_info.descriptorPool = ctx->desc_pools[slot]; alloc_info.descriptorSetCount = 1; alloc_info.pSetLayouts = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &alloc_info, &ctx->desc_sets[slot]) != VK_SUCCESS) return -1;
    return 0;
}

void vk_dispatch_vsa_fractional_laplacian(VulkanContext *ctx, VkBufferHandle in_data, VkBufferHandle alpha_in, VkBufferHandle out_data, uint32_t num_tokens, uint32_t ds) {
    if (!ctx || !ctx->recording) return;
    uint32_t slot = ctx->pipe_vsa_fractional_laplacian;
    VkDescriptorBufferInfo b_info[3] = {
        {ctx->buffers[in_data.index], 0, ctx->buf_sizes[in_data.index]},
        {ctx->buffers[alpha_in.index], 0, ctx->buf_sizes[alpha_in.index]},
        {ctx->buffers[out_data.index], 0, ctx->buf_sizes[out_data.index]}
    };
    VkWriteDescriptorSet writes[3] = {0};
    for(int i=0; i<3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = ctx->desc_sets[slot];
        writes[i].dstBinding = i; writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &b_info[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);
    uint32_t pc[2] = {num_tokens, ds};
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(ctx->cmd_buf, num_tokens, 1, 1);
}

"""
        content = content[:insert_idx] + impl + content[insert_idx:]
        with open("src/runtime/vulkan_backend_impl.c", "w") as f:
            f.write(content)
        print("Patched vulkan_backend_impl.c")

def patch_vulkan_backend_init():
    with open("src/runtime/vulkan_backend.c", "r") as f:
        content = f.read()

    if "vk_create_vsa_fwht_forward_pipeline" not in content:
        insert_idx = content.find("if (vk_create_sp07_spbsc_pipeline")
        
        init_impl = """
    if (vk_create_vsa_fwht_forward_pipeline(ctx) != 0) { fprintf(stderr, "Failed vsa_fwht_forward\\n"); return -1; }
    if (vk_create_vsa_bent_recurrence_pipeline(ctx) != 0) { fprintf(stderr, "Failed vsa_bent\\n"); return -1; }
    if (vk_create_vsa_fractional_laplacian_pipeline(ctx) != 0) { fprintf(stderr, "Failed vsa_laplacian\\n"); return -1; }
"""
        content = content[:insert_idx] + init_impl + content[insert_idx:]
        with open("src/runtime/vulkan_backend.c", "w") as f:
            f.write(content)
        print("Patched vulkan_backend.c init")

def add_ctx_fields():
    with open("src/runtime/vulkan_backend.h", "r") as f:
        content = f.read()
    if "uint32_t pipe_vsa_fwht_forward;" not in content:
        insert_idx = content.find("uint32_t pipe_sp11_tmsheaf;")
        fields = """
    uint32_t pipe_vsa_fwht_forward;
    uint32_t pipe_vsa_bent_recurrence;
    uint32_t pipe_vsa_fractional_laplacian;
"""
        content = content[:insert_idx] + fields + content[insert_idx:]
        with open("src/runtime/vulkan_backend.h", "w") as f:
            f.write(content)
        print("Patched vulkan_backend.h ctx fields")

if __name__ == "__main__":
    patch_header()
    patch_impl()
    add_ctx_fields()
    patch_vulkan_backend_init()
