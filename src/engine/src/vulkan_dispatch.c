#include "vulkan_backend.h"
#include <stdio.h>
#include <string.h>

#ifdef ENGINE_VULKAN
#include <vulkan/vulkan.h>

typedef struct {
    uint32_t y_off;
    uint32_t W_off;
    uint32_t scale_off;
    uint32_t x_off;
    uint32_t bias_off;
    int32_t rows;
    int32_t cols;
} pc_ternary_gemv_t;

typedef struct {
    uint32_t y_off;
    uint32_t W_off;
    uint32_t x_off;
    uint32_t bias_off;
    int32_t rows;
    int32_t cols;
} pc_float_gemv_t;

typedef struct {
    uint32_t y_off;
    uint32_t x_off;
    uint32_t w_off;
    int32_t n;
    uint32_t eps_bits;
} pc_rmsnorm_t;

typedef struct {
    uint32_t y_off;
    uint32_t a_off;
    uint32_t b_off;
    int32_t n;
    int32_t mode;
} pc_elementwise_t;

typedef struct {
    uint32_t q_off;
    uint32_t k_off;
    uint32_t freq_off;
    int32_t n_heads;
    int32_t head_dim;
    int32_t pos;
} pc_rope_t;

typedef struct {
    uint32_t out_off;
    uint32_t q_off;
    uint32_t k_cache_off;
    uint32_t v_cache_off;
    uint32_t scores_off;
    int32_t n_heads;
    int32_t head_dim;
    int32_t cache_len;
    int32_t cache_start;
    int32_t window_size;
    uint32_t scale_bits;
} pc_attention_t;

typedef struct {
    uint32_t y_off;
    uint32_t conv_buf_off;
    uint32_t x_off;
    uint32_t w_off;
    uint32_t bias_off;
    int32_t d_inner;
    int32_t d_conv;
} pc_conv1d_step_t;

typedef struct vk_context_s vk_context_t;

extern struct vk_context_s {
    VkInstance instance;
    VkPhysicalDevice phys_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t queue_family;

    VkCommandPool cmd_pool;
    VkCommandBuffer cmd_buf;
    VkFence fence;

    VkDescriptorSetLayout desc_layout;
    VkDescriptorPool desc_pool;
    VkDescriptorSet desc_set;

    VkBuffer weight_buf;
    VkBuffer scratch_buf;
    VkBuffer staging_buf;

    VkDeviceMemory weight_mem;
    VkDeviceMemory scratch_mem;
    VkDeviceMemory staging_mem;

    void* staging_mapped;

    uint32_t weight_size;
    uint32_t scratch_size;
    uint32_t staging_size;

    VkShaderModule shader_modules[7];
    VkPipelineLayout pipeline_layouts[7];
    VkPipeline pipelines[7];

    const void* model;
    bool recording;
    int dispatch_count;

    char device_name[256];
    uint64_t device_vram;
};

static bool vk_begin_recording(vk_context_t* ctx) {
    if (ctx->recording) {
        return true;
    }

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    VkResult res = vkBeginCommandBuffer(ctx->cmd_buf, &begin_info);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to begin command buffer (code %d)\n", res);
        return false;
    }

    ctx->recording = true;
    ctx->dispatch_count = 0;
    return true;
}

static void vk_memory_barrier(vk_context_t* ctx) {
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };

    vkCmdPipelineBarrier(
        ctx->cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &barrier,
        0, NULL,
        0, NULL
    );
}

void vk_ternary_gemv(vk_context_t* ctx,
    uint32_t y_off, uint32_t W_off, uint32_t scale_off,
    uint32_t x_off, uint32_t bias_off, int32_t rows, int32_t cols) {
    if (!ctx || !ctx->device) return;

    if (!vk_begin_recording(ctx)) {
        return;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[0]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx->pipeline_layouts[0], 0, 1, &ctx->desc_set, 0, NULL);

    pc_ternary_gemv_t pc = {
        .y_off = y_off,
        .W_off = W_off,
        .scale_off = scale_off,
        .x_off = x_off,
        .bias_off = bias_off,
        .rows = rows,
        .cols = cols,
    };

    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[0],
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, (uint32_t)rows, 1, 1);

    if (ctx->dispatch_count > 0) {
        vk_memory_barrier(ctx);
    }

    ctx->dispatch_count++;
}

void vk_float_gemv(vk_context_t* ctx,
    uint32_t y_off, uint32_t W_off, uint32_t x_off,
    uint32_t bias_off, int32_t rows, int32_t cols) {
    if (!ctx || !ctx->device) return;

    if (!vk_begin_recording(ctx)) {
        return;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[1]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx->pipeline_layouts[1], 0, 1, &ctx->desc_set, 0, NULL);

    pc_float_gemv_t pc = {
        .y_off = y_off,
        .W_off = W_off,
        .x_off = x_off,
        .bias_off = bias_off,
        .rows = rows,
        .cols = cols,
    };

    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[1],
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, (uint32_t)rows, 1, 1);

    vk_memory_barrier(ctx);
    ctx->dispatch_count++;
}

void vk_rmsnorm(vk_context_t* ctx,
    uint32_t y_off, uint32_t x_off, uint32_t w_off, int32_t n, float eps) {
    if (!ctx || !ctx->device) return;

    if (!vk_begin_recording(ctx)) {
        return;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[2]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx->pipeline_layouts[2], 0, 1, &ctx->desc_set, 0, NULL);

    uint32_t eps_bits;
    memcpy(&eps_bits, &eps, sizeof(float));

    pc_rmsnorm_t pc = {
        .y_off = y_off,
        .x_off = x_off,
        .w_off = w_off,
        .n = n,
        .eps_bits = eps_bits,
    };

    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[2],
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, 1, 1, 1);

    vk_memory_barrier(ctx);
    ctx->dispatch_count++;
}

void vk_elementwise(vk_context_t* ctx,
    uint32_t y_off, uint32_t a_off, uint32_t b_off, int32_t n, int32_t mode) {
    if (!ctx || !ctx->device) return;

    if (!vk_begin_recording(ctx)) {
        return;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[3]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx->pipeline_layouts[3], 0, 1, &ctx->desc_set, 0, NULL);

    pc_elementwise_t pc = {
        .y_off = y_off,
        .a_off = a_off,
        .b_off = b_off,
        .n = n,
        .mode = mode,
    };

    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[3],
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t num_workgroups = (uint32_t)((n + 255) / 256);
    vkCmdDispatch(ctx->cmd_buf, num_workgroups, 1, 1);

    vk_memory_barrier(ctx);
    ctx->dispatch_count++;
}

void vk_rope(vk_context_t* ctx,
    uint32_t q_off, uint32_t k_off, uint32_t freq_off,
    int32_t n_heads, int32_t head_dim, int32_t pos) {
    if (!ctx || !ctx->device) return;

    if (!vk_begin_recording(ctx)) {
        return;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[5]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx->pipeline_layouts[5], 0, 1, &ctx->desc_set, 0, NULL);

    pc_rope_t pc = {
        .q_off = q_off,
        .k_off = k_off,
        .freq_off = freq_off,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .pos = pos,
    };

    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[5],
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, (uint32_t)n_heads, 1, 1);

    vk_memory_barrier(ctx);
    ctx->dispatch_count++;
}

void vk_attention(vk_context_t* ctx,
    uint32_t out_off, uint32_t q_off, uint32_t k_cache_off,
    uint32_t v_cache_off, uint32_t scores_off,
    int32_t n_heads, int32_t head_dim, int32_t cache_len,
    int32_t cache_start, int32_t window_size, float scale) {
    if (!ctx || !ctx->device) return;

    if (!vk_begin_recording(ctx)) {
        return;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[6]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx->pipeline_layouts[6], 0, 1, &ctx->desc_set, 0, NULL);

    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, sizeof(float));

    pc_attention_t pc = {
        .out_off = out_off,
        .q_off = q_off,
        .k_cache_off = k_cache_off,
        .v_cache_off = v_cache_off,
        .scores_off = scores_off,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .cache_len = cache_len,
        .cache_start = cache_start,
        .window_size = window_size,
        .scale_bits = scale_bits,
    };

    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[6],
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, (uint32_t)n_heads, 1, 1);

    vk_memory_barrier(ctx);
    ctx->dispatch_count++;
}

void vk_conv1d_step(vk_context_t* ctx,
    uint32_t y_off, uint32_t conv_buf_off, uint32_t x_off,
    uint32_t w_off, uint32_t bias_off, int32_t d_inner, int32_t d_conv) {
    if (!ctx || !ctx->device) return;

    if (!vk_begin_recording(ctx)) {
        return;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[4]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
        ctx->pipeline_layouts[4], 0, 1, &ctx->desc_set, 0, NULL);

    pc_conv1d_step_t pc = {
        .y_off = y_off,
        .conv_buf_off = conv_buf_off,
        .x_off = x_off,
        .w_off = w_off,
        .bias_off = bias_off,
        .d_inner = d_inner,
        .d_conv = d_conv,
    };

    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[4],
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t num_workgroups = (uint32_t)((d_inner + 255) / 256);
    vkCmdDispatch(ctx->cmd_buf, num_workgroups, 1, 1);

    vk_memory_barrier(ctx);
    ctx->dispatch_count++;
}

void vk_submit_and_wait(vk_context_t* ctx) {
    if (!ctx || !ctx->device) return;

    if (!ctx->recording) {
        return;
    }

    VkResult res = vkEndCommandBuffer(ctx->cmd_buf);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to end command buffer (code %d)\n", res);
        return;
    }

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cmd_buf,
    };

    res = vkQueueSubmit(ctx->compute_queue, 1, &submit_info, ctx->fence);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to submit command buffer (code %d)\n", res);
        ctx->recording = false;
        return;
    }

    res = vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: fence wait failed (code %d)\n", res);
        ctx->recording = false;
        return;
    }

    res = vkResetFences(ctx->device, 1, &ctx->fence);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to reset fence (code %d)\n", res);
        ctx->recording = false;
        return;
    }

    res = vkResetCommandBuffer(ctx->cmd_buf, 0);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to reset command buffer (code %d)\n", res);
        ctx->recording = false;
        return;
    }

    ctx->recording = false;
    ctx->dispatch_count = 0;
}

#endif
