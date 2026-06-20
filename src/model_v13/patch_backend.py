import sys

file_path = r'C:\Users\th1rt\Documents\Workspaces\code-ai-workspace\src\model_v12\src\runtime\vulkan_backend_impl.c'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add pipe_sp11_tmsheaf
content = content.replace('uint32_t            pipe_sp09_htr1p;', 'uint32_t            pipe_sp09_htr1p;\n    uint32_t            pipe_sp11_tmsheaf;')

# Add pipeline creation call
content = content.replace('if (vk_create_sp09_htr1p_pipeline(ctx) != 0) {', 'if (vk_create_sp11_tmsheaf_pipeline(ctx) != 0) {\n        fprintf(stderr, "[VK] failed to create sp11_tmsheaf pipeline\\n");\n        return -1;\n    }\n\n    if (vk_create_sp09_htr1p_pipeline(ctx) != 0) {')

# Add the implementation
impl = '''
#include "shaders/sp11_tmsheaf_spv.h"

static int vk_create_sp11_tmsheaf_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count++;
    ctx->pipe_sp11_tmsheaf = slot;

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layout_info.bindingCount = 3;
    layout_info.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &ctx->desc_set_layouts[slot]) != VK_SUCCESS) {
        return -1;
    }

    VkPushConstantRange push_constant;
    push_constant.offset = 0;
    push_constant.size = 8;
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo pipeline_layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &ctx->desc_set_layouts[slot];
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant;

    if (vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL, &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        return -1;
    }

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = sp11_tmsheaf_spv_len;
    smci.pCode    = (const uint32_t *)sp11_tmsheaf_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL, &ctx->shader_modules[slot]) != VK_SUCCESS) {
        return -1;
    }

    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, &ctx->pipelines[slot]) != VK_SUCCESS) {
        return -1;
    }

    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;
    if (vkCreateDescriptorPool(ctx->device, &pool_info, NULL, &ctx->desc_pools[slot]) != VK_SUCCESS) {
        return -1;
    }

    VkDescriptorSetAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    alloc_info.descriptorPool = ctx->desc_pools[slot];
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &ctx->desc_set_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &alloc_info, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        return -1;
    }

    return 0;
}

int vk_dispatch_sp11_tmsheaf(VulkanContext *ctx, VkBufferHandle in_A, VkBufferHandle in_B, VkBufferHandle out_sim, uint32_t num_tokens, uint32_t ds) {
    if (!ctx || !ctx->recording) return -1;
    uint32_t slot = ctx->pipe_sp11_tmsheaf;

    VkDescriptorBufferInfo b_info[3] = {
        {ctx->buffers[in_A.index], 0, ctx->buf_sizes[in_A.index]},
        {ctx->buffers[in_B.index], 0, ctx->buf_sizes[in_B.index]},
        {ctx->buffers[out_sim.index], 0, ctx->buf_sizes[out_sim.index]}
    };

    VkWriteDescriptorSet writes[3] = {0};
    for(int i=0; i<3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ctx->desc_sets[slot];
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &b_info[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);

    uint32_t pc[2] = {num_tokens, ds};
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot], VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);

    vkCmdDispatch(ctx->cmd_buf, num_tokens, 1, 1);
    return 0;
}
'''

content = content.replace('int vk_dispatch_sp09_htr1p(', impl + '\nint vk_dispatch_sp09_htr1p(')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('vulkan_backend_impl.c updated')

h_file = r'C:\Users\th1rt\Documents\Workspaces\code-ai-workspace\src\model_v12\src\runtime\vulkan_backend.h'
with open(h_file, 'r', encoding='utf-8') as f:
    h_content = f.read()

h_impl = 'int vk_dispatch_sp11_tmsheaf(VulkanContext *ctx, VkBufferHandle in_A, VkBufferHandle in_B, VkBufferHandle out_sim, uint32_t num_tokens, uint32_t ds);\n'
h_content = h_content.replace('int vk_dispatch_sp09_htr1p(', h_impl + 'int vk_dispatch_sp09_htr1p(')

with open(h_file, 'w', encoding='utf-8') as f:
    f.write(h_content)

print('vulkan_backend.h updated')
