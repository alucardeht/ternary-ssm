#include "vulkan_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ENGINE_VULKAN
#include <vulkan/vulkan.h>

extern struct vk_context_s;

typedef struct {
    uint32_t embedding_offset;
    uint32_t final_norm_offset;
    uint32_t lm_head_offset;
    uint32_t lm_head_size;
    uint32_t* layer_offsets;
    uint32_t* layer_sizes;
    uint32_t total_size;
} vk_weight_map_t;

static uint32_t align64(uint32_t size) {
    return (size + 63) & ~63U;
}

static bool find_memory_type(VkPhysicalDevice phys_device, uint32_t type_filter,
                            VkMemoryPropertyFlags props, uint32_t* out_type) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys_device, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props) {
            *out_type = i;
            return true;
        }
    }
    return false;
}

static VkBuffer create_buffer(VkDevice device, VkPhysicalDevice phys_device,
                             size_t size, VkBufferUsageFlags usage,
                             VkMemoryPropertyFlags mem_props, VkDeviceMemory* out_memory) {
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VkBuffer buffer;
    VkResult res = vkCreateBuffer(device, &buffer_info, NULL, &buffer);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to create buffer (size %zu, code %d)\n", size, res);
        return VK_NULL_HANDLE;
    }

    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(device, buffer, &mem_req);

    uint32_t memory_type;
    if (!find_memory_type(phys_device, mem_req.memoryTypeBits, mem_props, &memory_type)) {
        fprintf(stderr, "Vulkan: no suitable memory type found for buffer\n");
        vkDestroyBuffer(device, buffer, NULL);
        return VK_NULL_HANDLE;
    }

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = memory_type,
    };

    VkDeviceMemory memory;
    res = vkAllocateMemory(device, &alloc_info, NULL, &memory);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to allocate buffer memory (code %d)\n", res);
        vkDestroyBuffer(device, buffer, NULL);
        return VK_NULL_HANDLE;
    }

    vkBindBufferMemory(device, buffer, memory, 0);
    *out_memory = memory;
    return buffer;
}

static uint32_t calculate_weight_buffer_size(const trnm_model_t* model) {
    uint32_t total = 0;

    total = align64(total + model->vocab_size * model->d_model * sizeof(float));

    for (int32_t i = 0; i < model->n_layers; i++) {
        uint32_t ternary_data_size;
        uint32_t float_size;

        if (model->layer_types[i] == 0) {
            ternary_data_size = (model->d_model * model->d_inner + 3) / 4;
            ternary_data_size += (model->d_inner * model->d_model + 3) / 4;
            ternary_data_size += (model->d_model * model->d_model + 3) / 4;

            float_size = model->d_inner * sizeof(float);
            float_size += model->d_inner * sizeof(float);
            float_size += model->d_conv * model->d_inner * sizeof(float);
            float_size += model->d_inner * sizeof(float);
            float_size += model->d_inner * sizeof(float);
            float_size += model->d_model * sizeof(float);
            float_size += model->d_model * sizeof(float);
            float_size += model->d_model * sizeof(float);
        } else {
            ternary_data_size = (model->d_model * model->d_model + 3) / 4 * 3;
            ternary_data_size += (model->d_inner * model->d_model + 3) / 4 * 2;

            float_size = model->d_model * sizeof(float) * 3;
            float_size += model->d_inner * sizeof(float) * 2;
            float_size += model->d_model * sizeof(float) * 2;
            float_size += 256 / 32 * sizeof(float);
        }

        total = align64(total + ternary_data_size);
        total = align64(total + float_size);
    }

    total = align64(total + model->d_model * sizeof(float));

    if (model->lm_head_ternary) {
        total = align64(total + (model->vocab_size * model->d_model + 3) / 4);
        total = align64(total + model->vocab_size * sizeof(float));
    } else {
        total = align64(total + model->vocab_size * model->d_model * sizeof(float));
    }

    return total;
}

bool vk_create_buffers(vk_context_t* ctx) {
    if (!ctx || !ctx->model) {
        fprintf(stderr, "Vulkan: vk_create_buffers called without model\n");
        return false;
    }

    ctx->weight_size = calculate_weight_buffer_size(ctx->model);

    uint32_t d_model = ctx->model->d_model;
    uint32_t d_inner = ctx->model->d_inner;
    uint32_t vocab_size = ctx->model->vocab_size;
    uint32_t d_conv = ctx->model->d_conv;

    uint32_t scratch_regions = align64(d_model * sizeof(float)) * 3;
    scratch_regions += align64(vocab_size * sizeof(float));

    uint32_t layer_scratch = align64((d_inner > d_model ? d_inner : d_model) * sizeof(float) * 8);
    scratch_regions += layer_scratch;

    ctx->scratch_size = scratch_regions;

    ctx->staging_size = (1 << 20);
    if (vocab_size * sizeof(float) > ctx->staging_size) {
        ctx->staging_size = vocab_size * sizeof(float);
    }
    if (d_model * d_inner * sizeof(float) > ctx->staging_size) {
        ctx->staging_size = d_model * d_inner * sizeof(float);
    }

    ctx->weight_buf = create_buffer(ctx->device, ctx->phys_device,
                                     ctx->weight_size,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                     &ctx->weight_mem);
    if (ctx->weight_buf == VK_NULL_HANDLE) {
        return false;
    }

    ctx->scratch_buf = create_buffer(ctx->device, ctx->phys_device,
                                      ctx->scratch_size,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                      &ctx->scratch_mem);
    if (ctx->scratch_buf == VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx->device, ctx->weight_buf, NULL);
        vkFreeMemory(ctx->device, ctx->weight_mem, NULL);
        return false;
    }

    ctx->staging_buf = create_buffer(ctx->device, ctx->phys_device,
                                      ctx->staging_size,
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      &ctx->staging_mem);
    if (ctx->staging_buf == VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx->device, ctx->weight_buf, NULL);
        vkFreeMemory(ctx->device, ctx->weight_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->scratch_buf, NULL);
        vkFreeMemory(ctx->device, ctx->scratch_mem, NULL);
        return false;
    }

    VkResult res = vkMapMemory(ctx->device, ctx->staging_mem, 0, ctx->staging_size, 0,
                               &ctx->staging_mapped);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to map staging buffer memory (code %d)\n", res);
        vkDestroyBuffer(ctx->device, ctx->weight_buf, NULL);
        vkFreeMemory(ctx->device, ctx->weight_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->scratch_buf, NULL);
        vkFreeMemory(ctx->device, ctx->scratch_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        return false;
    }

    return true;
}

void vk_destroy_buffers(vk_context_t* ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->staging_mapped && ctx->device != VK_NULL_HANDLE) {
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        ctx->staging_mapped = NULL;
    }

    if (ctx->device != VK_NULL_HANDLE) {
        if (ctx->weight_buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(ctx->device, ctx->weight_buf, NULL);
            ctx->weight_buf = VK_NULL_HANDLE;
        }
        if (ctx->weight_mem != VK_NULL_HANDLE) {
            vkFreeMemory(ctx->device, ctx->weight_mem, NULL);
            ctx->weight_mem = VK_NULL_HANDLE;
        }

        if (ctx->scratch_buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(ctx->device, ctx->scratch_buf, NULL);
            ctx->scratch_buf = VK_NULL_HANDLE;
        }
        if (ctx->scratch_mem != VK_NULL_HANDLE) {
            vkFreeMemory(ctx->device, ctx->scratch_mem, NULL);
            ctx->scratch_mem = VK_NULL_HANDLE;
        }

        if (ctx->staging_buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
            ctx->staging_buf = VK_NULL_HANDLE;
        }
        if (ctx->staging_mem != VK_NULL_HANDLE) {
            vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
            ctx->staging_mem = VK_NULL_HANDLE;
        }
    }
}

void vk_upload_to_scratch(vk_context_t* ctx, uint32_t off, const void* data, size_t bytes) {
    if (!ctx || !ctx->staging_mapped || off + bytes > ctx->staging_size) {
        return;
    }

    memcpy((uint8_t*)ctx->staging_mapped + off, data, bytes);
}

void vk_download_from_scratch(vk_context_t* ctx, uint32_t off, void* data, size_t bytes) {
    if (!ctx || !ctx->staging_mapped || off + bytes > ctx->staging_size) {
        return;
    }

    memcpy(data, (uint8_t*)ctx->staging_mapped + off, bytes);
}

static void copy_buffer_region(vk_context_t* ctx, VkBuffer src, VkBuffer dst,
                               uint32_t src_off, uint32_t dst_off, size_t size) {
    if (!ctx->recording) {
        VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(ctx->cmd_buf, &begin_info);
        ctx->recording = true;
    }

    VkBufferCopy copy_region = {
        .srcOffset = src_off,
        .dstOffset = dst_off,
        .size = size,
    };

    vkCmdCopyBuffer(ctx->cmd_buf, src, dst, 1, &copy_region);

    VkBufferMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = dst,
        .offset = dst_off,
        .size = size,
    };

    vkCmdPipelineBarrier(ctx->cmd_buf,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 0, NULL, 1, &barrier, 0, NULL);
}

int vk_upload_weights(vk_context_t* ctx, const trnm_model_t* model) {
    if (!ctx || !model) {
        fprintf(stderr, "Vulkan: vk_upload_weights called with NULL context or model\n");
        return -1;
    }

    ctx->model = model;

    if (!vk_create_buffers(ctx)) {
        fprintf(stderr, "Vulkan: failed to create buffers\n");
        return -1;
    }

    return 0;
}

#else

bool vk_create_buffers(vk_context_t* ctx) {
    (void)ctx;
    return false;
}

void vk_destroy_buffers(vk_context_t* ctx) {
    (void)ctx;
}

void vk_upload_to_scratch(vk_context_t* ctx, uint32_t off, const void* data, size_t bytes) {
    (void)ctx;
    (void)off;
    (void)data;
    (void)bytes;
}

void vk_download_from_scratch(vk_context_t* ctx, uint32_t off, void* data, size_t bytes) {
    (void)ctx;
    (void)off;
    (void)data;
    (void)bytes;
}

#endif
