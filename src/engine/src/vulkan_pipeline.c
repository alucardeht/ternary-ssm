#include "vulkan_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ENGINE_VULKAN
#include <vulkan/vulkan.h>

enum {
    VK_PIPE_TERNARY_GEMV = 0,
    VK_PIPE_FLOAT_GEMV = 1,
    VK_PIPE_RMSNORM = 2,
    VK_PIPE_ELEMENTWISE = 3,
    VK_PIPE_CONV1D_STEP = 4,
    VK_PIPE_ROPE = 5,
    VK_PIPE_ATTENTION = 6,
    VK_PIPE_COUNT = 7
};

typedef struct {
    const uint32_t* data;
    size_t size;
} shader_spirv_t;

static VkShaderModule create_shader_module(VkDevice device, const uint32_t* code, size_t code_size) {
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code_size,
        .pCode = code,
    };

    VkShaderModule module;
    VkResult res = vkCreateShaderModule(device, &create_info, NULL, &module);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to create shader module (code %d)\n", res);
        return VK_NULL_HANDLE;
    }

    return module;
}

static VkDescriptorSetLayout create_descriptor_layout(VkDevice device) {
    VkDescriptorSetLayoutBinding bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };

    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };

    VkDescriptorSetLayout layout;
    VkResult res = vkCreateDescriptorSetLayout(device, &layout_info, NULL, &layout);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to create descriptor set layout (code %d)\n", res);
        return VK_NULL_HANDLE;
    }

    return layout;
}

static VkDescriptorPool create_descriptor_pool(VkDevice device) {
    VkDescriptorPoolSize pool_sizes[1] = {
        {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3,
        },
    };

    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = pool_sizes,
    };

    VkDescriptorPool pool;
    VkResult res = vkCreateDescriptorPool(device, &pool_info, NULL, &pool);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to create descriptor pool (code %d)\n", res);
        return VK_NULL_HANDLE;
    }

    return pool;
}

static VkDescriptorSet allocate_descriptor_set(VkDevice device, VkDescriptorPool pool,
                                               VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };

    VkDescriptorSet set;
    VkResult res = vkAllocateDescriptorSets(device, &alloc_info, &set);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to allocate descriptor set (code %d)\n", res);
        return VK_NULL_HANDLE;
    }

    return set;
}

static void update_descriptor_set(VkDevice device, VkDescriptorSet set,
                                  VkBuffer weight_buf, VkBuffer scratch_buf,
                                  VkBuffer staging_buf,
                                  uint32_t weight_size, uint32_t scratch_size,
                                  uint32_t staging_size) {
    VkDescriptorBufferInfo buffer_infos[3] = {
        {
            .buffer = weight_buf,
            .offset = 0,
            .range = weight_size,
        },
        {
            .buffer = scratch_buf,
            .offset = 0,
            .range = scratch_size,
        },
        {
            .buffer = staging_buf,
            .offset = 0,
            .range = staging_size,
        },
    };

    VkWriteDescriptorSet writes[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[2],
        },
    };

    vkUpdateDescriptorSets(device, 3, writes, 0, NULL);
}

static VkPipelineLayout create_pipeline_layout(VkDevice device,
                                               VkDescriptorSetLayout desc_layout) {
    VkPushConstantRange push_const_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 128,
    };

    VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_const_range,
    };

    VkPipelineLayout layout;
    VkResult res = vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to create pipeline layout (code %d)\n", res);
        return VK_NULL_HANDLE;
    }

    return layout;
}

static VkPipeline create_compute_pipeline(VkDevice device, VkPipelineLayout layout,
                                         VkShaderModule shader) {
    VkPipelineShaderStageCreateInfo shader_stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader,
        .pName = "main",
    };

    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = shader_stage,
        .layout = layout,
    };

    VkPipeline pipeline;
    VkResult res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to create compute pipeline (code %d)\n", res);
        return VK_NULL_HANDLE;
    }

    return pipeline;
}

bool vk_create_pipelines(vk_context_t* ctx) {
    if (!ctx) {
        return false;
    }

    ctx->desc_layout = create_descriptor_layout(ctx->device);
    if (ctx->desc_layout == VK_NULL_HANDLE) {
        return false;
    }

    ctx->desc_pool = create_descriptor_pool(ctx->device);
    if (ctx->desc_pool == VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layout, NULL);
        ctx->desc_layout = VK_NULL_HANDLE;
        return false;
    }

    ctx->desc_set = allocate_descriptor_set(ctx->device, ctx->desc_pool, ctx->desc_layout);
    if (ctx->desc_set == VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layout, NULL);
        ctx->desc_pool = VK_NULL_HANDLE;
        ctx->desc_layout = VK_NULL_HANDLE;
        return false;
    }

    update_descriptor_set(ctx->device, ctx->desc_set,
                         ctx->weight_buf, ctx->scratch_buf, ctx->staging_buf,
                         ctx->weight_size, ctx->scratch_size, ctx->staging_size);

    VkPipelineLayout shared_layout = create_pipeline_layout(ctx->device, ctx->desc_layout);
    if (shared_layout == VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layout, NULL);
        ctx->desc_pool = VK_NULL_HANDLE;
        ctx->desc_layout = VK_NULL_HANDLE;
        return false;
    }

    for (int i = 0; i < VK_PIPE_COUNT; i++) {
        ctx->pipeline_layouts[i] = shared_layout;
    }

    shader_spirv_t shaders[VK_PIPE_COUNT];
    memset(shaders, 0, sizeof(shaders));

#ifdef VULKAN_SHADERS_H
#include "vulkan_shaders.h"
#endif

    for (int i = 0; i < VK_PIPE_COUNT; i++) {
        if (shaders[i].data == NULL || shaders[i].size == 0) {
            fprintf(stderr, "Vulkan: shader %d SPIR-V not available (vulkan_shaders.h not generated)\n", i);
            for (int j = 0; j < i; j++) {
                if (ctx->shader_modules[j] != VK_NULL_HANDLE) {
                    vkDestroyShaderModule(ctx->device, ctx->shader_modules[j], NULL);
                }
                if (ctx->pipelines[j] != VK_NULL_HANDLE) {
                    vkDestroyPipeline(ctx->device, ctx->pipelines[j], NULL);
                }
            }
            for (int j = 0; j < VK_PIPE_COUNT; j++) {
                vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[j], NULL);
            }
            vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
            vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layout, NULL);
            ctx->desc_pool = VK_NULL_HANDLE;
            ctx->desc_layout = VK_NULL_HANDLE;
            return false;
        }

        ctx->shader_modules[i] = create_shader_module(ctx->device, shaders[i].data, shaders[i].size);
        if (ctx->shader_modules[i] == VK_NULL_HANDLE) {
            for (int j = 0; j < i; j++) {
                vkDestroyShaderModule(ctx->device, ctx->shader_modules[j], NULL);
                vkDestroyPipeline(ctx->device, ctx->pipelines[j], NULL);
            }
            for (int j = 0; j < VK_PIPE_COUNT; j++) {
                vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[j], NULL);
            }
            vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
            vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layout, NULL);
            ctx->desc_pool = VK_NULL_HANDLE;
            ctx->desc_layout = VK_NULL_HANDLE;
            return false;
        }

        ctx->pipelines[i] = create_compute_pipeline(ctx->device, ctx->pipeline_layouts[i],
                                                    ctx->shader_modules[i]);
        if (ctx->pipelines[i] == VK_NULL_HANDLE) {
            for (int j = 0; j <= i; j++) {
                vkDestroyShaderModule(ctx->device, ctx->shader_modules[j], NULL);
                if (j < i) {
                    vkDestroyPipeline(ctx->device, ctx->pipelines[j], NULL);
                }
            }
            for (int j = 0; j < VK_PIPE_COUNT; j++) {
                vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[j], NULL);
            }
            vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
            vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layout, NULL);
            ctx->desc_pool = VK_NULL_HANDLE;
            ctx->desc_layout = VK_NULL_HANDLE;
            return false;
        }
    }

    return true;
}

void vk_destroy_pipelines(vk_context_t* ctx) {
    if (!ctx || ctx->device == VK_NULL_HANDLE) {
        return;
    }

    for (int i = 0; i < VK_PIPE_COUNT; i++) {
        if (ctx->pipelines[i] != VK_NULL_HANDLE) {
            vkDestroyPipeline(ctx->device, ctx->pipelines[i], NULL);
            ctx->pipelines[i] = VK_NULL_HANDLE;
        }
        if (ctx->shader_modules[i] != VK_NULL_HANDLE) {
            vkDestroyShaderModule(ctx->device, ctx->shader_modules[i], NULL);
            ctx->shader_modules[i] = VK_NULL_HANDLE;
        }
    }

    for (int i = 0; i < VK_PIPE_COUNT; i++) {
        if (ctx->pipeline_layouts[i] != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[i], NULL);
            ctx->pipeline_layouts[i] = VK_NULL_HANDLE;
        }
    }

    if (ctx->desc_set != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &ctx->desc_set);
        ctx->desc_set = VK_NULL_HANDLE;
    }

    if (ctx->desc_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
        ctx->desc_pool = VK_NULL_HANDLE;
    }

    if (ctx->desc_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layout, NULL);
        ctx->desc_layout = VK_NULL_HANDLE;
    }
}

void vk_bind_pipeline(vk_context_t* ctx, int pipeline_idx) {
    if (!ctx || pipeline_idx < 0 || pipeline_idx >= VK_PIPE_COUNT) {
        return;
    }

    if (!ctx->recording) {
        return;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[pipeline_idx]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                           ctx->pipeline_layouts[pipeline_idx], 0, 1, &ctx->desc_set, 0, NULL);
}

#else

bool vk_create_pipelines(vk_context_t* ctx) {
    (void)ctx;
    return false;
}

void vk_destroy_pipelines(vk_context_t* ctx) {
    (void)ctx;
}

void vk_bind_pipeline(vk_context_t* ctx, int pipeline_idx) {
    (void)ctx;
    (void)pipeline_idx;
}

#endif
