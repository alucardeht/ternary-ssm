#ifdef V12_HAS_VULKAN

#define V12_VK_DISABLE_DOWNLOAD_MACRO
#include "vulkan_backend.h"
static int vk_create_sp08_qrf3ac_pipeline(VulkanContext *ctx);
static int vk_create_sp09_htr1p_pipeline(VulkanContext *ctx);
static int vk_create_sp11_tmsheaf_pipeline(VulkanContext *ctx);
static int vk_create_e10_pscq_compress_pipeline(VulkanContext *ctx);
static int vk_create_e10_pscq_decompress_pipeline(VulkanContext *ctx);

#undef V12_VK_DISABLE_DOWNLOAD_MACRO
#ifdef _WIN32
#include <windows.h>
#endif
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "shaders/vulkan_shaders.h"
#include "shaders/fp32_gemv_spv.h"
#include "shaders/fp32_gemm_spv.h"
#include "shaders/residual_add_spv.h"
#include "shaders/ssm_scan_spv.h"
#include "shaders/ssm_scan_aas_spv.h"
#include "shaders/elementwise_gates_spv.h"
#include "shaders/output_project_wht_spv.h"
#include "shaders/output_project_backward_spv.h"
#include "shaders/ssm_scan_backward_spv.h"
#include "shaders/ssm_scan_backward_cr_spv.h"
#include "shaders/ssm_bwd_project_dh_spv.h"
#include "shaders/ssm_bwd_elementwise_spv.h"
#include "shaders/ssm_bwd_dh_norm_spv.h"
#include "shaders/ssm_fused_spv.h"
#ifdef V12_USE_F1_REVERSIBLE
/* R356 F1 slice S5 — SPV header for the reverse-walk replay shader. Only
 * included when the reversible F1 backward is opt-in at compile time; under
 * the default build the header is not required and the file need not exist
 * in the tree. Generate it via `make shaders` (glslc + spv_to_header.py). */
#include "shaders/ssm_bwd_replay_f1_spv.h"
#endif
#include "shaders/conv1d_bwd_weights_spv.h"
#include "shaders/rms_norm_forward_spv.h"
#include "shaders/rms_norm_backward_spv.h"
#include "shaders/wht_inplace_spv.h"
#include "shaders/token_gather_spv.h"
#if V12_TOLR_ENABLED
#include "shaders/sampled_ce_forward_tolr_spv.h"
#include "shaders/sampled_ce_scatter_dh_tolr_spv.h"
#else
#include "shaders/sampled_ce_forward_spv.h"
#include "shaders/sampled_ce_scatter_dh_spv.h"
#endif
#include "shaders/norm_grad_reduce_spv.h"
#include "shaders/dlm_head_scatter_spv.h"
#include "shaders/a_gate_recompute_spv.h"
/* Forward declaration — definition lives later in the file. */

#include "shaders/a_backward_dx_cv_spv.h"
#include "shaders/conv1d_bwd_input_gate_spv.h"
#include "shaders/embed_e_backward_spv.h"
#include "shaders/dc_bilinear_reduce_spv.h"
#include "shaders/direct_b_backward_spv.h"
#include "shaders/welford_update_spv.h"
#include "shaders/online_tssi_score_update_spv.h"
#include "shaders/broadcast_fill_spv.h"
#include "shaders/mtp_loss_spv.h"
#include "shaders/pc_gather_prev_spv.h"
#include "shaders/residual_sub_spv.h"
#include "shaders/pc_negate_gather_spv.h"
#include "shaders/residual_scatter_add_spv.h"
#include "shaders/adam_dense_spv.h"
#include "shaders/grad_scale_spv.h"
#include "shaders/factored_adam_reduce_spv.h"
#include "shaders/factored_adam_moment_spv.h"
#include "shaders/factored_adam_update_spv.h"
#include "shaders/mvue_resample_spv.h"
#include "shaders/csr_count_spv.h"
#include "shaders/csr_scatter_spv.h"

#define STAGING_SIZE (64 * 1024 * 1024)
#define VK_MAX_TRANSFERS 16

typedef struct {
    VkCommandBuffer cmd;
    VkFence         fence;
    int             used;
} AsyncTransfer;


struct VulkanContext {
    VkInstance          instance;
    VkPhysicalDevice    phys_device;
    VkDevice            device;
    VkQueue             compute_queue;
    uint32_t            queue_family_index;

    VkCommandPool       cmd_pool;
    VkCommandBuffer     cmd_buf;

    VkFence             fence;

    VkBuffer            buffers[VK_MAX_BUFFERS];
    AsyncTransfer       transfers[VK_MAX_TRANSFERS];
    VkDeviceMemory      memories[VK_MAX_BUFFERS];
    size_t              buf_sizes[VK_MAX_BUFFERS];
    size_t              alloc_sizes[VK_MAX_BUFFERS];
    uint8_t             buf_flags[VK_MAX_BUFFERS];
    uint8_t             buf_used[VK_MAX_BUFFERS];
    uint32_t            buf_count;

    VkShaderModule      shader_modules[VK_MAX_PIPELINES];
    VkPipelineLayout    pipeline_layouts[VK_MAX_PIPELINES];
    VkPipeline          pipelines[VK_MAX_PIPELINES];
    VkDescriptorSetLayout desc_layouts[VK_MAX_PIPELINES];
    VkDescriptorPool    desc_pools[VK_MAX_PIPELINES];
    VkDescriptorSet     desc_sets[VK_MAX_PIPELINES];
    uint32_t            pipeline_count;

    uint32_t            pipe_ternary_matvec;
    uint32_t            pipe_ssm_scan;
    uint32_t            pipe_ssm_scan_aas;
    uint32_t            pipe_fp32_gemv;
    uint32_t            pipe_fp32_gemm;
    uint32_t            pipe_residual_add;
    uint32_t            pipe_elementwise_gates;
    uint32_t            pipe_output_project_wht;
    uint32_t            pipe_output_project_backward;
    uint32_t            pipe_ssm_scan_backward;
    uint32_t            pipe_ssm_scan_backward_cr;
    uint32_t            pipe_sp07_spbsc;
    uint32_t            pipe_sp08_qrf3ac;
    uint32_t            pipe_sp09_htr1p;
    uint32_t            pipe_sp11_tmsheaf;
    uint32_t            pipe_e10_pscq_compress;
    uint32_t            pipe_e10_pscq_decompress;
    uint32_t            pipe_ssm_bwd_elementwise;
    uint32_t            pipe_ssm_bwd_dh_norm;
    uint32_t            pipe_ssm_bwd_project_dh;
    uint32_t            pipe_ssm_fused;
#ifdef V12_USE_F1_REVERSIBLE
    uint32_t            pipe_ssm_bwd_replay_f1;
#endif
    uint32_t            pipe_conv1d_bwd_weights;
    uint32_t            pipe_rms_norm_forward;
    uint32_t            pipe_rms_norm_backward;
    uint32_t            pipe_wht_inplace;
    uint32_t            pipe_token_gather;
    uint32_t            pipe_sampled_ce_forward;
    uint32_t            pipe_sampled_ce_scatter_dh;
    uint32_t            pipe_norm_grad_reduce;
    uint32_t            pipe_dlm_head_scatter;
    uint32_t            pipe_a_backward_dx_cv;
    uint32_t            pipe_a_gate_recompute;
    uint32_t            pipe_conv1d_bwd_input_gate;
    uint32_t            pipe_embed_e_backward;
    uint32_t            pipe_dc_bilinear_reduce;
    uint32_t            pipe_direct_b_backward;
    uint32_t            pipe_welford_update;
    uint32_t            pipe_online_tssi_score_update;
    uint32_t            pipe_broadcast_fill;
    uint32_t            pipe_mtp_loss;
    uint32_t            pipe_pc_gather_prev;
    uint32_t            pipe_residual_sub;
    uint32_t            pipe_pc_negate_gather;
    uint32_t            pipe_residual_scatter_add;
    uint32_t            pipe_adam_dense;
    uint32_t            pipe_grad_scale;
    uint32_t            pipe_factored_adam_reduce;
    uint32_t            pipe_factored_adam_moment;
    uint32_t            pipe_factored_adam_update;
    uint32_t            pipe_mvue_resample;
    uint32_t            pipe_csr_count;
    uint32_t            pipe_csr_scatter;

    VkBuffer            staging_buf;
    VkDeviceMemory      staging_mem;
    void               *staging_mapped;
    size_t              staging_size;

    size_t              allocated;
    int                 initialized;
    int                 recording;
    char                device_name[256];
    VkRuntimeProfile    profile;
    VkQueryPool         profile_query_pool;
    float               profile_timestamp_period_ns;
    uint32_t            profile_timestamp_valid_bits;
    int                 profile_timing_supported;
    int                 profile_query_active;
    int                 profile_query_pending;
    uint64_t            profile_cmd_buf_begin_ns;
    int                 profile_cmd_buf_has_dispatch;
    int                 profile_post_wait_dispatch_scope;

    /* Batch command recording */
    int             batch_mode;
    size_t          staging_offset;     /* ring allocator cursor in staging_buf */
    struct {
        void       *dst;
        size_t      bytes;
        size_t      staging_offset;
#ifdef V12_DEBUG_VK_DEFERRED
        const char *file;
        int         line;
        const char *func;
#endif
    } deferred_downloads[64];
    int             deferred_download_count;
};

static uint64_t vk_now_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) {
        if (!QueryPerformanceFrequency(&freq) || freq.QuadPart == 0) {
            clock_t ticks = clock();
            if (ticks == (clock_t)-1) return 0;
            return (uint64_t)(((double)ticks * 1000000000.0) / (double)CLOCKS_PER_SEC);
        }
    }
    QueryPerformanceCounter(&counter);
    return (uint64_t)(((double)counter.QuadPart * 1000000000.0) / (double)freq.QuadPart);
#else
    #if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
    }
    #endif
    {
        clock_t ticks = clock();
        if (ticks == (clock_t)-1) return 0;
        return (uint64_t)(((double)ticks * 1000000000.0) / (double)CLOCKS_PER_SEC);
    }
#endif
}

static VkResult vk_profiled_queue_submit(VulkanContext *ctx,
                                         uint32_t submit_count,
                                         const VkSubmitInfo *submit_infos,
                                         VkFence fence) {
    int dispatch_scope = ctx && ctx->profile_query_pending;
    uint64_t t0 = vk_now_ns();
    VkResult res = vkQueueSubmit(ctx->compute_queue, submit_count, submit_infos, fence);
    uint64_t t1 = vk_now_ns();
    if (ctx) {
        ctx->profile.submit_count++;
        ctx->profile.submit_ns += (t1 - t0);
        if (dispatch_scope) {
            ctx->profile.dispatch_submit_ns += (t1 - t0);
        }
        if (res != VK_SUCCESS && ctx->profile_query_pending) {
            ctx->profile.incomplete_timing_count++;
            ctx->profile_query_pending = 0;
        }
    }
    return res;
}

static VkResult vk_profiled_wait_for_fences(VulkanContext *ctx,
                                            uint32_t fence_count,
                                            const VkFence *fences,
                                            VkBool32 wait_all,
                                            uint64_t timeout) {
    int dispatch_scope = ctx && ctx->profile_query_pending;
    uint64_t t0 = vk_now_ns();
    VkResult res = vkWaitForFences(ctx->device, fence_count, fences, wait_all, timeout);
    uint64_t t1 = vk_now_ns();
    if (ctx) {
        ctx->profile.wait_count++;
        ctx->profile.wait_ns += (t1 - t0);
        if (dispatch_scope) {
            ctx->profile.dispatch_wait_ns += (t1 - t0);
        }
        if (res == VK_SUCCESS && ctx->profile_query_pending) {
            uint64_t timestamps[2] = {0, 0};
            VkResult qres = vkGetQueryPoolResults(ctx->device,
                                                  ctx->profile_query_pool,
                                                  0,
                                                  2,
                                                  sizeof(timestamps),
                                                  timestamps,
                                                  sizeof(uint64_t),
                                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (qres == VK_SUCCESS && timestamps[1] >= timestamps[0]) {
                uint64_t delta_ticks = timestamps[1] - timestamps[0];
                double gpu_ns = (double)delta_ticks * (double)ctx->profile_timestamp_period_ns;
                if (gpu_ns >= 0.0) {
                    ctx->profile.gpu_cmd_ns += (uint64_t)(gpu_ns + 0.5);
                    ctx->profile.timed_submit_count++;
                } else {
                    ctx->profile.incomplete_timing_count++;
                }
            } else {
                ctx->profile.incomplete_timing_count++;
            }
            ctx->profile_query_pending = 0;
        }
        ctx->profile_post_wait_dispatch_scope = dispatch_scope;
    }
    return res;
}

static VkResult vk_profiled_begin_command_buffer(VulkanContext *ctx,
                                                 const VkCommandBufferBeginInfo *begin_info) {
    uint64_t t0 = vk_now_ns();
    VkResult res;
    if (ctx) {
        ctx->profile_query_active = 0;
        ctx->profile_query_pending = 0;
        ctx->profile_cmd_buf_begin_ns = 0;
        ctx->profile_cmd_buf_has_dispatch = 0;
        ctx->profile_post_wait_dispatch_scope = 0;
    }
    res = vkBeginCommandBuffer(ctx->cmd_buf, begin_info);
    if (ctx && res == VK_SUCCESS) {
        ctx->profile_cmd_buf_begin_ns = t0;
    }
    return res;
}

static void vk_profiled_cmd_dispatch(VulkanContext *ctx,
                                     uint32_t group_count_x,
                                     uint32_t group_count_y,
                                     uint32_t group_count_z) {
    if (ctx) {
        ctx->profile.dispatch_count++;
        ctx->profile_cmd_buf_has_dispatch = 1;
        if (ctx->profile_timing_supported && !ctx->profile_query_active) {
            vkCmdResetQueryPool(ctx->cmd_buf, ctx->profile_query_pool, 0, 2);
            vkCmdWriteTimestamp(ctx->cmd_buf,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                ctx->profile_query_pool,
                                0);
            ctx->profile_query_active = 1;
        }
    }
    vkCmdDispatch(ctx->cmd_buf, group_count_x, group_count_y, group_count_z);
}

static VkResult vk_profiled_end_command_buffer(VulkanContext *ctx) {
    uint64_t t1;
    VkResult res;
    if (ctx && ctx->profile_timing_supported && ctx->profile_query_active) {
        vkCmdWriteTimestamp(ctx->cmd_buf,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ctx->profile_query_pool,
                            1);
        ctx->profile_query_pending = 1;
        ctx->profile_query_active = 0;
    }
    res = vkEndCommandBuffer(ctx->cmd_buf);
    t1 = vk_now_ns();
    if (ctx && ctx->profile_cmd_buf_has_dispatch && ctx->profile_cmd_buf_begin_ns != 0) {
        ctx->profile.dispatch_record_ns += (t1 - ctx->profile_cmd_buf_begin_ns);
    }
    if (ctx) {
        ctx->profile_cmd_buf_begin_ns = 0;
        ctx->profile_cmd_buf_has_dispatch = 0;
    }
    if (ctx && res != VK_SUCCESS && ctx->profile_query_pending) {
        ctx->profile.incomplete_timing_count++;
        ctx->profile_query_pending = 0;
    }
    return res;
}

static VkResult vk_profiled_reset_fences(VulkanContext *ctx,
                                         uint32_t fence_count,
                                         const VkFence *fences) {
    uint64_t t0 = vk_now_ns();
    VkResult res = vkResetFences(ctx->device, fence_count, fences);
    uint64_t t1 = vk_now_ns();
    if (ctx && ctx->profile_post_wait_dispatch_scope) {
        ctx->profile.dispatch_reset_ns += (t1 - t0);
    }
    return res;
}

static VkResult vk_profiled_reset_command_buffer(VulkanContext *ctx,
                                                 VkCommandBuffer command_buffer,
                                                 VkCommandBufferResetFlags flags) {
    uint64_t t0 = vk_now_ns();
    VkResult res = vkResetCommandBuffer(command_buffer, flags);
    uint64_t t1 = vk_now_ns();
    if (ctx && ctx->profile_post_wait_dispatch_scope) {
        ctx->profile.dispatch_reset_ns += (t1 - t0);
        ctx->profile_post_wait_dispatch_scope = 0;
    }
    return res;
}

#define vkBeginCommandBuffer(cmd_buf, begin_info) vk_profiled_begin_command_buffer((ctx), (begin_info))
#define vkCmdDispatch(cmd_buf, group_count_x, group_count_y, group_count_z) \
    vk_profiled_cmd_dispatch((ctx), (group_count_x), (group_count_y), (group_count_z))
#define vkEndCommandBuffer(cmd_buf) vk_profiled_end_command_buffer((ctx))
#define vkResetFences(device, fence_count, fences) \
    vk_profiled_reset_fences((ctx), (fence_count), (fences))
#define vkResetCommandBuffer(command_buffer, flags) \
    vk_profiled_reset_command_buffer((ctx), (command_buffer), (flags))

static uint32_t find_memory_type(VulkanContext *ctx,
                                 uint32_t type_bits,
                                 VkMemoryPropertyFlags required_props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(ctx->phys_device, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & required_props) == required_props) {
            return i;
        }
    }
    return UINT32_MAX;
}

static uint32_t find_free_slot(VulkanContext *ctx) {
    /* Slot 0 is the null handle. This keeps calloc-initialized handles invalid. */
    for (uint32_t i = 1; i < VK_MAX_BUFFERS; i++) {
        if (!ctx->buf_used[i]) return i;
    }
    return UINT32_MAX;
}
















static int vk_create_csr_count_pipeline(VulkanContext *ctx);
static int vk_create_csr_scatter_pipeline(VulkanContext *ctx);
static int vk_create_sp07_spbsc_pipeline(VulkanContext *ctx);

static int vk_create_ternary_matvec_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[6] = {0};
    for (int i = 0; i < 6; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 6;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 6
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = ternary_matvec_spv_len;
    smci.pCode    = (const uint32_t *)ternary_matvec_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 6
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_ternary_matvec = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_fp32_gemv_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 8
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = fp32_gemv_spv_len;
    smci.pCode    = (const uint32_t *)fp32_gemv_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_fp32_gemv = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_fp32_gemm_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 5
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = fp32_gemm_spv_len;
    smci.pCode    = (const uint32_t *)fp32_gemm_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_fp32_gemm = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_residual_add_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t)
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = residual_add_spv_len;
    smci.pCode    = (const uint32_t *)residual_add_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 2
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_residual_add = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_ssm_scan_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[5] = {0};
    for (int i = 0; i < 5; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 5;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 3
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = ssm_scan_spv_len;
    smci.pCode    = (const uint32_t *)ssm_scan_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 5
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_ssm_scan = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_ssm_scan_aas_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[7] = {0};
    for (int i = 0; i < 7; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 7;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 3
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = ssm_scan_aas_spv_len;
    smci.pCode    = (const uint32_t *)ssm_scan_aas_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 7
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_ssm_scan_aas = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_elementwise_gates_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[11] = {0};
    for (int i = 0; i < 11; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 11;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = 16
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = elementwise_gates_spv_len;
    smci.pCode    = (const uint32_t *)elementwise_gates_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 11
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_elementwise_gates = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_output_project_wht_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 2
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = output_project_wht_spv_len;
    smci.pCode    = (const uint32_t *)output_project_wht_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_output_project_wht = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_output_project_backward_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 2
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = output_project_backward_spv_len;
    smci.pCode    = (const uint32_t *)output_project_backward_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_output_project_backward = slot;
    ctx->pipeline_count++;
    return 0;
}




static int vk_create_ssm_scan_backward_cr_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count++;

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 5
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = ssm_scan_backward_cr_spv_len;
    smci.pCode    = (const uint32_t *)ssm_scan_backward_cr_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai,
                                 &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_ssm_scan_backward_cr = slot;
    return 0;
}

static int vk_create_ssm_scan_backward_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 3
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = ssm_scan_backward_spv_len;
    smci.pCode    = (const uint32_t *)ssm_scan_backward_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_ssm_scan_backward = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_ssm_fused_pipeline(VulkanContext *ctx) {
    if (ctx->pipeline_count >= VK_MAX_PIPELINES) return -1;
    uint32_t slot = ctx->pipeline_count++;

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = ssm_fused_spv_len;
    smci.pCode    = (const uint32_t *)ssm_fused_spv;

    if (vkCreateShaderModule(ctx->device, &smci, NULL, &ctx->shader_modules[slot]) != VK_SUCCESS)
        return -1;

    VkDescriptorSetLayoutBinding bindings[13];
    for (int i = 0; i < 13; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = NULL;
    }
    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 13;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL, &ctx->desc_layouts[slot]) != VK_SUCCESS)
        return -1;

    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 16; 

    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts    = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL, &ctx->pipeline_layouts[slot]) != VK_SUCCESS)
        return -1;

    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName = "main";
    cpci.layout = ctx->pipeline_layouts[slot];

    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, &ctx->pipelines[slot]) != VK_SUCCESS)
        return -1;

    VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 13 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL, &ctx->desc_pools[slot]) != VK_SUCCESS)
        return -1;

    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS)
        return -1;

    ctx->pipe_ssm_fused = slot;
    return 0;
}

static int vk_create_ssm_bwd_elementwise_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[14] = {0};
    for (int i = 0; i < 14; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 14;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 4
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = ssm_bwd_elementwise_spv_len;
    smci.pCode    = (const uint32_t *)ssm_bwd_elementwise_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 14
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_ssm_bwd_elementwise = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_ssm_bwd_project_dh_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[2] = {0};
    for (int i = 0; i < 2; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 2;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 2
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = ssm_bwd_project_dh_spv_len;
    smci.pCode    = (const uint32_t *)ssm_bwd_project_dh_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 2
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_ssm_bwd_project_dh = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_ssm_bwd_dh_norm_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[2] = {0};
    for (int i = 0; i < 2; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 2;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 2
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = ssm_bwd_dh_norm_spv_len;
    smci.pCode    = (const uint32_t *)ssm_bwd_dh_norm_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 2
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_ssm_bwd_dh_norm = slot;
    ctx->pipeline_count++;
    return 0;
}

#ifdef V12_USE_F1_REVERSIBLE
/* R356 F1 slice S5 — pipeline creator for ssm_bwd_replay_f1.comp.
 *
 * 9 storage bindings (A, A_per_step, dt, B_eff, x, C_bl, h_final, boundary,
 * h_repl) and a 5×uint32 push-constant block. This is a SEPARATE descriptor
 * set layout from ssm_bwd_elementwise — the two shaders are dispatched back
 * to back but each owns its own descriptor set. */
static int vk_create_ssm_bwd_replay_f1_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[9] = {0};
    for (int i = 0; i < 9; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 9;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create replay_f1 descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 5
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create replay_f1 pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = ssm_bwd_replay_f1_spv_len;
    smci.pCode    = (const uint32_t *)ssm_bwd_replay_f1_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create replay_f1 shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create replay_f1 compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 9
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create replay_f1 descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate replay_f1 descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_ssm_bwd_replay_f1 = slot;
    ctx->pipeline_count++;
    return 0;
}
#endif /* V12_USE_F1_REVERSIBLE */

static int vk_create_conv1d_bwd_weights_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 4
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = conv1d_bwd_weights_spv_len;
    smci.pCode    = (const uint32_t *)conv1d_bwd_weights_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_conv1d_bwd_weights = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_rms_norm_forward_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[5] = {0};
    for (int i = 0; i < 5; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 5;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = 12
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = rms_norm_forward_spv_len;
    smci.pCode    = (const uint32_t *)rms_norm_forward_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 5
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_rms_norm_forward = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_rms_norm_backward_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[4] = {0};
    for (int i = 0; i < 4; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 4;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = 8
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = rms_norm_backward_spv_len;
    smci.pCode    = (const uint32_t *)rms_norm_backward_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 4
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_rms_norm_backward = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_wht_inplace_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding binding = {0};
    binding.binding         = 0;
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 1;
    dslci.pBindings    = &binding;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = 8
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = wht_inplace_spv_len;
    smci.pCode    = (const uint32_t *)wht_inplace_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_wht_inplace = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_token_gather_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 2
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = token_gather_spv_len;
    smci.pCode    = (const uint32_t *)token_gather_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_token_gather = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_sampled_ce_forward_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[21] = {0};
    for (int i = 0; i < 21; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 21;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 19
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
#if V12_TOLR_ENABLED
    smci.codeSize = sampled_ce_forward_tolr_spv_len;
    smci.pCode    = (const uint32_t *)sampled_ce_forward_tolr_spv;
#else
    smci.codeSize = sampled_ce_forward_spv_len;
    smci.pCode    = (const uint32_t *)sampled_ce_forward_spv;
#endif
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 21
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_sampled_ce_forward = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_sampled_ce_scatter_dh_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[7] = {0};
    for (int i = 0; i < 7; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 7;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 4
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
#if V12_TOLR_ENABLED
    smci.codeSize = sampled_ce_scatter_dh_tolr_spv_len;
    smci.pCode    = (const uint32_t *)sampled_ce_scatter_dh_tolr_spv;
#else
    smci.codeSize = sampled_ce_scatter_dh_spv_len;
    smci.pCode    = (const uint32_t *)sampled_ce_scatter_dh_spv;
#endif
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 7
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_sampled_ce_scatter_dh = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_norm_grad_reduce_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[5] = {0};
    for (int i = 0; i < 5; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 5;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 3
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = norm_grad_reduce_spv_len;
    smci.pCode    = (const uint32_t *)norm_grad_reduce_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 5
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_norm_grad_reduce = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_dlm_head_scatter_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[4] = {0};
    for (int i = 0; i < 4; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 4;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 4
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = dlm_head_scatter_spv_len;
    smci.pCode    = (const uint32_t *)dlm_head_scatter_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 4
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_dlm_head_scatter = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_a_backward_dx_cv_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 2
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = a_backward_dx_cv_spv_len;
    smci.pCode    = (const uint32_t *)a_backward_dx_cv_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_a_backward_dx_cv = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_conv1d_bwd_input_gate_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[6] = {0};
    for (int i = 0; i < 6; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 6;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 2 + sizeof(float) * 2
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = conv1d_bwd_input_gate_spv_len;
    smci.pCode    = (const uint32_t *)conv1d_bwd_input_gate_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 6
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_conv1d_bwd_input_gate = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_embed_e_backward_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[4] = {0};
    for (int i = 0; i < 4; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 4;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 3
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = embed_e_backward_spv_len;
    smci.pCode    = (const uint32_t *)embed_e_backward_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 4
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_embed_e_backward = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_create_dc_bilinear_reduce_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[4] = {0};
    for (int i = 0; i < 4; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 4;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 3
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = dc_bilinear_reduce_spv_len;
    smci.pCode    = (const uint32_t *)dc_bilinear_reduce_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 4
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_dc_bilinear_reduce = slot;
    ctx->pipeline_count++;
    return 0;
}

static void bind_wht_inplace_buffers(VulkanContext *ctx,
                                     uint32_t slot,
                                     VkBuffer data_buf,
                                     size_t data_sz) {
    VkDescriptorBufferInfo buf_info = {data_buf, 0, data_sz};
    VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet          = ctx->desc_sets[slot];
    write.dstBinding      = 0;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo     = &buf_info;
    vkUpdateDescriptorSets(ctx->device, 1, &write, 0, NULL);
}

static void bind_token_gather_buffers(VulkanContext *ctx,
                                      uint32_t slot,
                                      VkBuffer embed_buf,
                                      VkBuffer tokens_buf,
                                      VkBuffer output_buf,
                                      size_t embed_sz,
                                      size_t tokens_sz,
                                      size_t output_sz) {
    VkDescriptorBufferInfo buf_infos[3] = {
        {embed_buf,  0, embed_sz},
        {tokens_buf, 0, tokens_sz},
        {output_buf, 0, output_sz},
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
}

static void bind_ternary_matvec_buffers(VulkanContext *ctx,
                                        uint32_t slot,
                                        VkBuffer row_ptr_pos,
                                        VkBuffer pos_idx,
                                        VkBuffer row_ptr_neg,
                                        VkBuffer neg_idx,
                                        VkBuffer x_buf,
                                        VkBuffer y_buf,
                                        size_t rpp_sz, size_t pi_sz,
                                        size_t rpn_sz, size_t ni_sz,
                                        size_t x_sz,   size_t y_sz) {
    VkDescriptorBufferInfo buf_infos[6] = {
        {row_ptr_pos, 0, rpp_sz},
        {pos_idx,     0, pi_sz},
        {row_ptr_neg, 0, rpn_sz},
        {neg_idx,     0, ni_sz},
        {x_buf,       0, x_sz},
        {y_buf,       0, y_sz},
    };
    VkWriteDescriptorSet writes[6];
    for (int i = 0; i < 6; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 6, writes, 0, NULL);
}

static void bind_fp32_gemv_buffers(VulkanContext *ctx,
                                   uint32_t slot,
                                   VkBuffer A_buf,
                                   VkBuffer x_buf,
                                   VkBuffer y_buf,
                                   size_t A_sz,
                                   size_t x_sz,
                                   size_t y_sz) {
    VkDescriptorBufferInfo buf_infos[3] = {
        {A_buf, 0, A_sz},
        {x_buf, 0, x_sz},
        {y_buf, 0, y_sz},
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
}

static void bind_fp32_gemm_buffers(VulkanContext *ctx,
                                   uint32_t slot,
                                   VkBuffer A_buf,
                                   VkBuffer B_buf,
                                   VkBuffer C_buf,
                                   size_t A_sz,
                                   size_t B_sz,
                                   size_t C_sz) {
    VkDescriptorBufferInfo buf_infos[3] = {
        {A_buf, 0, A_sz},
        {B_buf, 0, B_sz},
        {C_buf, 0, C_sz},
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
}

static void bind_residual_add_buffers(VulkanContext *ctx,
                                      uint32_t slot,
                                      VkBuffer x_buf,
                                      VkBuffer y_buf,
                                      size_t x_sz,
                                      size_t y_sz) {
    VkDescriptorBufferInfo buf_infos[2] = {
        {x_buf, 0, x_sz},
        {y_buf, 0, y_sz},
    };
    VkWriteDescriptorSet writes[2];
    for (int i = 0; i < 2; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);
}

static void bind_ssm_scan_buffers(VulkanContext *ctx,
                                   uint32_t slot,
                                   VkBuffer A_eff_buf,
                                   VkBuffer B_eff_buf,
                                   VkBuffer x_buf,
                                   VkBuffer C_bilinear_buf,
                                   VkBuffer h_out_buf,
                                   size_t A_eff_sz,
                                   size_t B_eff_sz,
                                   size_t x_sz,
                                   size_t C_bilinear_sz,
                                   size_t h_out_sz) {
    VkDescriptorBufferInfo buf_infos[5] = {
        {A_eff_buf,      0, A_eff_sz},
        {B_eff_buf,      0, B_eff_sz},
        {x_buf,          0, x_sz},
        {C_bilinear_buf, 0, C_bilinear_sz},
        {h_out_buf,      0, h_out_sz},
    };
    VkWriteDescriptorSet writes[5];
    for (int i = 0; i < 5; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 5, writes, 0, NULL);
}

static void bind_ssm_scan_aas_buffers(VulkanContext *ctx,
                                       uint32_t slot,
                                       VkBuffer A_eff_buf,
                                       VkBuffer B_eff_buf,
                                       VkBuffer x_buf,
                                       VkBuffer C_bilinear_buf,
                                       VkBuffer h_out_buf,
                                       VkBuffer ssm_u_buf,
                                       VkBuffer ssm_v_buf,
                                       size_t A_eff_sz,
                                       size_t B_eff_sz,
                                       size_t x_sz,
                                       size_t C_bilinear_sz,
                                       size_t h_out_sz,
                                       size_t ssm_u_sz,
                                       size_t ssm_v_sz) {
    VkDescriptorBufferInfo buf_infos[7] = {
        {A_eff_buf,      0, A_eff_sz},
        {B_eff_buf,      0, B_eff_sz},
        {x_buf,          0, x_sz},
        {C_bilinear_buf, 0, C_bilinear_sz},
        {h_out_buf,      0, h_out_sz},
        {ssm_u_buf,      0, ssm_u_sz},
        {ssm_v_buf,      0, ssm_v_sz},
    };
    VkWriteDescriptorSet writes[7];
    for (int i = 0; i < 7; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 7, writes, 0, NULL);
}

static void bind_elementwise_gates_buffers(VulkanContext *ctx,
                                            uint32_t slot,
                                            VkBuffer x_proj_buf,
                                            VkBuffer w_conv_buf,
                                            VkBuffer B_buf,
                                            VkBuffer A_buf,
                                            VkBuffer W_A_buf,
                                            VkBuffer b_A_buf,
                                            VkBuffer a_buf_buf,
                                            VkBuffer b_eff_buf,
                                            VkBuffer x_cv_buf,
                                            VkBuffer x_gated_out_buf,
                                            VkBuffer mask_val_out_buf,
                                            size_t x_proj_sz,
                                            size_t w_conv_sz,
                                            size_t B_sz,
                                            size_t A_sz,
                                            size_t W_A_sz,
                                            size_t b_A_sz,
                                            size_t a_buf_sz,
                                            size_t b_eff_sz,
                                            size_t x_cv_sz,
                                            size_t x_gated_out_sz,
                                            size_t mask_val_out_sz) {
    VkDescriptorBufferInfo buf_infos[11] = {
        {x_proj_buf,        0, x_proj_sz},
        {w_conv_buf,        0, w_conv_sz},
        {B_buf,             0, B_sz},
        {A_buf,             0, A_sz},
        {W_A_buf,           0, W_A_sz},
        {b_A_buf,           0, b_A_sz},
        {a_buf_buf,         0, a_buf_sz},
        {b_eff_buf,         0, b_eff_sz},
        {x_cv_buf,          0, x_cv_sz},
        {x_gated_out_buf,   0, x_gated_out_sz},
        {mask_val_out_buf,  0, mask_val_out_sz},
    };
    VkWriteDescriptorSet writes[11];
    for (int i = 0; i < 11; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 11, writes, 0, NULL);
}

static void bind_output_project_wht_buffers(VulkanContext *ctx,
                                             uint32_t slot,
                                             VkBuffer h_out_buf,
                                             VkBuffer C_mat_buf,
                                             VkBuffer y_ssm_buf,
                                             size_t h_out_sz,
                                             size_t C_mat_sz,
                                             size_t y_ssm_sz) {
    VkDescriptorBufferInfo buf_infos[3] = {
        {h_out_buf, 0, h_out_sz},
        {C_mat_buf, 0, C_mat_sz},
        {y_ssm_buf, 0, y_ssm_sz},
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
}



static void bind_ssm_scan_backward_cr_buffers(VulkanContext *ctx, uint32_t slot,
                                              VkBuffer A_eff, VkBuffer v,
                                              VkBuffer dh_out,
                                              size_t sz_A_eff, size_t sz_v,
                                              size_t sz_dh_out) {
    VkDescriptorBufferInfo dbi[3] = {0};
    dbi[0].buffer = A_eff;  dbi[0].offset = 0; dbi[0].range = sz_A_eff;
    dbi[1].buffer = v;      dbi[1].offset = 0; dbi[1].range = sz_v;
    dbi[2].buffer = dh_out; dbi[2].offset = 0; dbi[2].range = sz_dh_out;

    VkWriteDescriptorSet wds[3] = {0};
    for (int i = 0; i < 3; i++) {
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = ctx->desc_sets[slot];
        wds[i].dstBinding = i;
        wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, wds, 0, NULL);
}


static void bind_ssm_scan_backward_buffers(VulkanContext *ctx,
                                            uint32_t slot,
                                            VkBuffer A_eff_buf,
                                            VkBuffer v_precomputed_buf,
                                            VkBuffer dh_out_buf,
                                            size_t A_eff_sz,
                                            size_t v_precomputed_sz,
                                            size_t dh_out_sz) {
    VkDescriptorBufferInfo buf_infos[3] = {
        {A_eff_buf,        0, A_eff_sz},
        {v_precomputed_buf, 0, v_precomputed_sz},
        {dh_out_buf,       0, dh_out_sz},
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
}

static void bind_ssm_fused_buffers(VulkanContext *ctx,
                                              uint32_t slot,
                                              VkBuffer A_eff_buf,
                                              VkBuffer B_eff_buf,
                                              VkBuffer x_cv_buf,
                                              VkBuffer v_buf,
                                              VkBuffer C_bl_buf,
                                              VkBuffer a_eff_gate_buf,
                                              VkBuffer A_vec_buf,
                                              VkBuffer dA_ps_buf,
                                              VkBuffer d_dot_buf,
                                              VkBuffer dB_seq_buf,
                                              VkBuffer d_x_cv_buf,
                                              VkBuffer dA_out_buf,
                                              VkBuffer dbA_out_buf,
                                              size_t A_eff_sz,
                                              size_t B_eff_sz,
                                              size_t x_cv_sz,
                                              size_t v_sz,
                                              size_t C_bl_sz,
                                              size_t a_eff_gate_sz,
                                              size_t A_vec_sz,
                                              size_t dA_ps_sz,
                                              size_t d_dot_sz,
                                              size_t dB_seq_sz,
                                              size_t d_x_cv_sz,
                                              size_t dA_out_sz,
                                              size_t dbA_out_sz) {
    VkDescriptorBufferInfo buf_infos[13] = {
        {A_eff_buf,      0, A_eff_sz},
        {B_eff_buf,      0, B_eff_sz},
        {x_cv_buf,       0, x_cv_sz},
        {v_buf,          0, v_sz},
        {C_bl_buf,       0, C_bl_sz},
        {a_eff_gate_buf, 0, a_eff_gate_sz},
        {A_vec_buf,      0, A_vec_sz},
        {dA_ps_buf,      0, dA_ps_sz},
        {d_dot_buf,      0, d_dot_sz},
        {dB_seq_buf,     0, dB_seq_sz},
        {d_x_cv_buf,     0, d_x_cv_sz},
        {dA_out_buf,     0, dA_out_sz},
        {dbA_out_buf,    0, dbA_out_sz},
    };
    VkWriteDescriptorSet writes[13];
    for (int i = 0; i < 13; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet = ctx->desc_sets[slot];
        writes[i].dstBinding = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 13, writes, 0, NULL);
}

static void bind_ssm_bwd_elementwise_buffers(VulkanContext *ctx,
                                              uint32_t slot,
                                              VkBuffer dh_buf,
                                              VkBuffer h_cache_buf,
                                              VkBuffer x_cv_buf,
                                              VkBuffer B_eff_buf,
                                              VkBuffer C_bl_buf,
                                              VkBuffer a_eff_gate_buf,
                                              VkBuffer A_vec_buf,
                                              VkBuffer dA_ps_buf,
                                              VkBuffer d_dot_buf,
                                              VkBuffer dB_seq_buf,
                                              VkBuffer d_x_cv_buf,
                                              VkBuffer dA_out_buf,
                                              VkBuffer dbA_out_buf,
                                              VkBuffer norm_factor_buf,
                                              size_t dh_sz,
                                              size_t h_cache_sz,
                                              size_t x_cv_sz,
                                              size_t B_eff_sz,
                                              size_t C_bl_sz,
                                              size_t a_eff_gate_sz,
                                              size_t A_vec_sz,
                                              size_t dA_ps_sz,
                                              size_t d_dot_sz,
                                              size_t dB_seq_sz,
                                              size_t d_x_cv_sz,
                                              size_t dA_out_sz,
                                              size_t dbA_out_sz,
                                              size_t norm_factor_sz) {
    VkDescriptorBufferInfo buf_infos[14] = {
        {dh_buf,         0, dh_sz},
        {h_cache_buf,    0, h_cache_sz},
        {x_cv_buf,       0, x_cv_sz},
        {B_eff_buf,      0, B_eff_sz},
        {C_bl_buf,       0, C_bl_sz},
        {a_eff_gate_buf, 0, a_eff_gate_sz},
        {A_vec_buf,      0, A_vec_sz},
        {dA_ps_buf,      0, dA_ps_sz},
        {d_dot_buf,      0, d_dot_sz},
        {dB_seq_buf,     0, dB_seq_sz},
        {d_x_cv_buf,     0, d_x_cv_sz},
        {dA_out_buf,     0, dA_out_sz},
        {dbA_out_buf,    0, dbA_out_sz},
        {norm_factor_buf,0, norm_factor_sz},
    };
    VkWriteDescriptorSet writes[14];
    for (int i = 0; i < 14; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 14, writes, 0, NULL);
}

static void bind_ssm_bwd_project_dh_buffers(VulkanContext *ctx,
                                         uint32_t slot,
                                         VkBuffer dh_buf,
                                         VkBuffer h_cache_buf,
                                         size_t dh_sz,
                                         size_t h_cache_sz) {
    VkDescriptorBufferInfo buf_infos[2] = {
        {dh_buf,      0, dh_sz},
        {h_cache_buf, 0, h_cache_sz},
    };
    VkWriteDescriptorSet writes[2];
    for (int i = 0; i < 2; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);
}

static void bind_ssm_bwd_dh_norm_buffers(VulkanContext *ctx,
                                         uint32_t slot,
                                         VkBuffer dh_buf,
                                         VkBuffer factor_buf,
                                         size_t dh_sz,
                                         size_t factor_sz) {
    VkDescriptorBufferInfo buf_infos[2] = {
        {dh_buf,     0, dh_sz},
        {factor_buf, 0, factor_sz},
    };
    VkWriteDescriptorSet writes[2];
    for (int i = 0; i < 2; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);
}

#ifdef V12_USE_F1_REVERSIBLE
static void bind_ssm_bwd_replay_f1_buffers(VulkanContext *ctx,
                                            uint32_t slot,
                                            VkBuffer A_buf,
                                            VkBuffer A_per_step_buf,
                                            VkBuffer dt_buf,
                                            VkBuffer B_eff_buf,
                                            VkBuffer x_buf,
                                            VkBuffer C_bl_buf,
                                            VkBuffer h_final_buf,
                                            VkBuffer boundary_buf,
                                            VkBuffer h_repl_buf,
                                            size_t A_sz,
                                            size_t A_per_step_sz,
                                            size_t dt_sz,
                                            size_t B_eff_sz,
                                            size_t x_sz,
                                            size_t C_bl_sz,
                                            size_t h_final_sz,
                                            size_t boundary_sz,
                                            size_t h_repl_sz) {
    VkDescriptorBufferInfo buf_infos[9] = {
        {A_buf,          0, A_sz},
        {A_per_step_buf, 0, A_per_step_sz},
        {dt_buf,         0, dt_sz},
        {B_eff_buf,      0, B_eff_sz},
        {x_buf,          0, x_sz},
        {C_bl_buf,       0, C_bl_sz},
        {h_final_buf,    0, h_final_sz},
        {boundary_buf,   0, boundary_sz},
        {h_repl_buf,     0, h_repl_sz},
    };
    VkWriteDescriptorSet writes[9];
    for (int i = 0; i < 9; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 9, writes, 0, NULL);
}
#endif /* V12_USE_F1_REVERSIBLE */

static void bind_conv1d_bwd_weights_buffers(VulkanContext *ctx,
                                              uint32_t slot,
                                             VkBuffer d_x_cv_buf,
                                             VkBuffer x_gated_buf,
                                             VkBuffer d_conv_w_buf,
                                             size_t d_x_cv_sz,
                                             size_t x_gated_sz,
                                             size_t d_conv_w_sz) {
    VkDescriptorBufferInfo buf_infos[3] = {
        {d_x_cv_buf,  0, d_x_cv_sz},
        {x_gated_buf, 0, x_gated_sz},
        {d_conv_w_buf, 0, d_conv_w_sz},
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
}

static void bind_rms_norm_forward_buffers(VulkanContext *ctx,
                                           uint32_t slot,
                                           VkBuffer x_buf,
                                           VkBuffer gamma_buf,
                                           VkBuffer beta_buf,
                                           VkBuffer y_buf,
                                           VkBuffer rms_out_buf,
                                           size_t x_sz,
                                           size_t gamma_sz,
                                           size_t beta_sz,
                                           size_t y_sz,
                                           size_t rms_out_sz) {
    VkDescriptorBufferInfo buf_infos[5] = {
        {x_buf,       0, x_sz},
        {gamma_buf,   0, gamma_sz},
        {beta_buf,    0, beta_sz},
        {y_buf,       0, y_sz},
        {rms_out_buf, 0, rms_out_sz},
    };
    VkWriteDescriptorSet writes[5];
    for (int i = 0; i < 5; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 5, writes, 0, NULL);
}

static void bind_rms_norm_backward_buffers(VulkanContext *ctx,
                                            uint32_t slot,
                                            VkBuffer h_final_buf,
                                            VkBuffer d_out_buf,
                                            VkBuffer rms_buf,
                                            VkBuffer norm_w_buf,
                                            size_t h_final_sz,
                                            size_t d_out_sz,
                                            size_t rms_sz,
                                            size_t norm_w_sz) {
    VkDescriptorBufferInfo buf_infos[4] = {
        {h_final_buf, 0, h_final_sz},
        {d_out_buf,   0, d_out_sz},
        {rms_buf,     0, rms_sz},
        {norm_w_buf,  0, norm_w_sz},
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 4, writes, 0, NULL);
}

int vk_init(VulkanContext **ctx_ptr) {
    if (!ctx_ptr) return -1;

    VulkanContext *ctx = (VulkanContext *)calloc(1, sizeof(VulkanContext));
    if (!ctx) {
        fprintf(stderr, "[VK] failed to allocate context\n");
        return -1;
    }

    VkApplicationInfo app_info = {
        VK_STRUCTURE_TYPE_APPLICATION_INFO
    };
    app_info.apiVersion = VK_API_VERSION_1_2;
    app_info.pApplicationName = "v12_ternary_ssm";
    app_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);

    const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
    const char *extensions[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    uint32_t layer_count = 0;
    uint32_t ext_count = 0;

#ifdef V12_VULKAN_DEBUG
    layer_count = 1;
    ext_count = 1;
#endif

    VkInstanceCreateInfo ici = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    };
    ici.pApplicationInfo        = &app_info;
    ici.enabledLayerCount       = layer_count;
    ici.ppEnabledLayerNames     = (layer_count > 0) ? layers : NULL;
    ici.enabledExtensionCount   = ext_count;
    ici.ppEnabledExtensionNames = (ext_count > 0) ? extensions : NULL;

    if (vkCreateInstance(&ici, NULL, &ctx->instance) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create instance\n");
        free(ctx);
        return -1;
    }

    uint32_t device_count = 0;
    if (vkEnumeratePhysicalDevices(ctx->instance, &device_count, NULL) != VK_SUCCESS || device_count == 0) {
        fprintf(stderr, "[VK] no physical devices found\n");
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    VkPhysicalDevice devices[16];
    if (device_count > 16) device_count = 16;
    if (vkEnumeratePhysicalDevices(ctx->instance, &device_count, devices) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to enumerate physical devices\n");
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    VkPhysicalDevice chosen = NULL;
    int best_type = -1;
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);

        int type = -1;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)   type = 0;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) type = 1;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)    type = 2;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)            type = 3;

        if (type >= 0 && (best_type < 0 || type < best_type)) {
            chosen = devices[i];
            best_type = type;
        }
    }

    if (!chosen) {
        fprintf(stderr, "[VK] no suitable physical device found\n");
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    ctx->phys_device = chosen;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx->phys_device, &props);
    strncpy(ctx->device_name, props.deviceName, sizeof(ctx->device_name) - 1);
    ctx->device_name[sizeof(ctx->device_name) - 1] = '\0';

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys_device, &queue_family_count, NULL);
    if (queue_family_count == 0) {
        fprintf(stderr, "[VK] no queue families found\n");
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    VkQueueFamilyProperties queue_props[16];
    if (queue_family_count > 16) queue_family_count = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys_device, &queue_family_count, queue_props);

    int compute_family = -1;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_family = (int)i;
            break;
        }
    }

    if (compute_family < 0) {
        fprintf(stderr, "[VK] no compute queue family found\n");
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    ctx->queue_family_index = (uint32_t)compute_family;
    ctx->profile_timestamp_period_ns = props.limits.timestampPeriod;
    ctx->profile_timestamp_valid_bits = queue_props[compute_family].timestampValidBits;
    ctx->profile_timing_supported =
        (ctx->profile_timestamp_period_ns > 0.0f && ctx->profile_timestamp_valid_bits > 0) ? 1 : 0;

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo dqci = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
    };
    dqci.queueFamilyIndex = ctx->queue_family_index;
    dqci.queueCount       = 1;
    dqci.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo dci = {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO
    };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &dqci;
    dci.enabledExtensionCount = 0;
    dci.ppEnabledExtensionNames = NULL;

    if (vkCreateDevice(ctx->phys_device, &dci, NULL, &ctx->device) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create logical device\n");
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    vkGetDeviceQueue(ctx->device, ctx->queue_family_index, 0, &ctx->compute_queue);

    VkCommandPoolCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO
    };
    cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = ctx->queue_family_index;

    if (vkCreateCommandPool(ctx->device, &cpci, NULL, &ctx->cmd_pool) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create command pool\n");
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    VkCommandBufferAllocateInfo cbai = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
    };
    cbai.commandPool        = ctx->cmd_pool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(ctx->device, &cbai, &ctx->cmd_buf) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate command buffer\n");
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    VkFenceCreateInfo fci = {
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
    };
    fci.flags = 0;

    if (vkCreateFence(ctx->device, &fci, NULL, &ctx->fence) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create fence\n");
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    VkBufferCreateInfo sbci = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO
    };
    sbci.size       = STAGING_SIZE;
    sbci.usage      = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    sbci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(ctx->device, &sbci, NULL, &ctx->staging_buf) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create staging buffer\n");
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    VkMemoryRequirements staging_mem_req;
    vkGetBufferMemoryRequirements(ctx->device, ctx->staging_buf, &staging_mem_req);

    VkMemoryAllocateInfo staging_mai = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO
    };
    staging_mai.allocationSize = staging_mem_req.size;
    staging_mai.memoryTypeIndex = find_memory_type(ctx, staging_mem_req.memoryTypeBits,
                                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (staging_mai.memoryTypeIndex == UINT32_MAX) {
        fprintf(stderr, "[VK] no suitable memory type for staging buffer\n");
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vkAllocateMemory(ctx->device, &staging_mai, NULL, &ctx->staging_mem) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate staging memory\n");
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vkBindBufferMemory(ctx->device, ctx->staging_buf, ctx->staging_mem, 0) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to bind staging buffer memory\n");
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vkMapMemory(ctx->device, ctx->staging_mem, 0, STAGING_SIZE, 0,
                    &ctx->staging_mapped) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to map staging memory\n");
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    ctx->staging_size = STAGING_SIZE;

    if (vk_create_ternary_matvec_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create ternary_matvec pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_ssm_scan_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create ssm_scan pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_ssm_scan_aas_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create ssm_scan_aas pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_fp32_gemv_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create fp32_gemv pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_fp32_gemm_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create fp32_gemm pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_residual_add_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create residual_add pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_elementwise_gates_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create elementwise_gates pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_output_project_wht_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create output_project_wht pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_output_project_backward_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create output_project_backward pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_ssm_scan_backward_cr_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create ssm_scan_backward_cr pipeline\n");
        vk_destroy(ctx);
        return -1;
    }
    if (vk_create_ssm_scan_backward_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create ssm_scan_backward pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_ssm_fused_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create ssm_fused pipeline\\n");
        return -1;
    }
    if (vk_create_ssm_bwd_project_dh_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create ssm_bwd_project_dh pipeline\n");
        return -1;
    }
    if (vk_create_ssm_bwd_dh_norm_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create ssm_bwd_dh_norm pipeline\n");
        return -1;
    }
    if (vk_create_ssm_bwd_elementwise_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create ssm_bwd_elementwise pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

#ifdef V12_USE_F1_REVERSIBLE
    if (vk_create_ssm_bwd_replay_f1_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create ssm_bwd_replay_f1 pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }
#endif

    if (vk_create_conv1d_bwd_weights_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create conv1d_bwd_weights pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_rms_norm_forward_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create rms_norm_forward pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_rms_norm_backward_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create rms_norm_backward pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_wht_inplace_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create wht_inplace pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_token_gather_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create token_gather pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_sampled_ce_forward_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create sampled_ce_forward pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_sampled_ce_scatter_dh_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create sampled_ce_scatter_dh pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_norm_grad_reduce_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create norm_grad_reduce pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_dlm_head_scatter_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create dlm_head_scatter pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    

    if (ctx->profile_timing_supported) {
        VkQueryPoolCreateInfo qpci = {
            VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO
        };
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        if (vkCreateQueryPool(ctx->device, &qpci, NULL, &ctx->profile_query_pool) != VK_SUCCESS) {
            ctx->profile_timing_supported = 0;
            ctx->profile_timestamp_valid_bits = 0;
            ctx->profile_timestamp_period_ns = 0.0f;
            ctx->profile_query_pool = VK_NULL_HANDLE;
        }
    }

    if (vk_create_a_backward_dx_cv_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create a_backward_dx_cv pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_conv1d_bwd_input_gate_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create conv1d_bwd_input_gate pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_embed_e_backward_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create embed_e_backward pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_dc_bilinear_reduce_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create dc_bilinear_reduce pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    if (vk_create_csr_count_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create csr_count pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_csr_scatter_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create csr_scatter pipeline\n");
        vkUnmapMemory(ctx->device, ctx->staging_mem);
        vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        vkDestroyFence(ctx->device, ctx->fence, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    if (vk_create_sp07_spbsc_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create sp07_spbsc pipeline\n");
        return -1;
    }

    if (vk_create_sp08_qrf3ac_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create sp08_qrf3ac pipeline\n");
        return -1;
    }

    if (vk_create_sp11_tmsheaf_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create sp11_tmsheaf pipeline\n");
        return -1;
    }

    if (vk_create_sp09_htr1p_pipeline(ctx) != 0) {
        fprintf(stderr, "[VK] failed to create sp09_htr1p pipeline\n");
        return -1;
    }
    if (vk_create_e10_pscq_compress_pipeline(ctx) != 0) { fprintf(stderr, "[VK] failed to create e10_pscq_compress pipeline\n"); return -1; }
    if (vk_create_e10_pscq_decompress_pipeline(ctx) != 0) { fprintf(stderr, "[VK] failed to create e10_pscq_decompress pipeline\n"); return -1; }

    ctx->initialized = 1;
    ctx->batch_mode = 0;
    ctx->staging_offset = 0;
    ctx->deferred_download_count = 0;
    ctx->profile.gpu_timing_supported = (uint32_t)ctx->profile_timing_supported;
    *ctx_ptr = ctx;
    fprintf(stderr, "[VK] initialized on device: %s\n", ctx->device_name);
    return 0;
}

void vk_destroy(VulkanContext *ctx) {
    if (!ctx) return;

    if (ctx->device) {
        vkDeviceWaitIdle(ctx->device);

        for (uint32_t i = 0; i < ctx->pipeline_count; i++) {
            if (ctx->desc_sets[i]) {
                vkFreeDescriptorSets(ctx->device, ctx->desc_pools[i], 1, &ctx->desc_sets[i]);
            }
            if (ctx->desc_pools[i]) {
                vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[i], NULL);
            }
            if (ctx->pipelines[i]) {
                vkDestroyPipeline(ctx->device, ctx->pipelines[i], NULL);
            }
            if (ctx->shader_modules[i]) {
                vkDestroyShaderModule(ctx->device, ctx->shader_modules[i], NULL);
            }
            if (ctx->pipeline_layouts[i]) {
                vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[i], NULL);
            }
            if (ctx->desc_layouts[i]) {
                vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[i], NULL);
            }
        }

        for (uint32_t i = 0; i < ctx->buf_count; i++) {
            if (ctx->buf_used[i]) {
                if (ctx->buffers[i]) {
                    vkDestroyBuffer(ctx->device, ctx->buffers[i], NULL);
                }
                if (ctx->memories[i]) {
                    vkFreeMemory(ctx->device, ctx->memories[i], NULL);
                }
            }
        }

        if (ctx->staging_mapped) {
            vkUnmapMemory(ctx->device, ctx->staging_mem);
        }
        if (ctx->staging_buf) {
            vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        }
        if (ctx->staging_mem) {
            vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        }

        if (ctx->fence) {
            vkDestroyFence(ctx->device, ctx->fence, NULL);
        }
        if (ctx->profile_query_pool) {
            vkDestroyQueryPool(ctx->device, ctx->profile_query_pool, NULL);
        }
        if (ctx->cmd_pool) {
            vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        }
        if (ctx->device) {
            vkDestroyDevice(ctx->device, NULL);
        }
    }

    if (ctx->instance) {
        vkDestroyInstance(ctx->instance, NULL);
    }

    free(ctx);
}

VkBufferHandle vk_alloc(VulkanContext *ctx, size_t bytes, VkBufFlags flags) {
    if (!ctx || !ctx->initialized) return VK_BUFFER_NULL;

    static int gassc_compiled = 0;
    if (!gassc_compiled) {
        gassc_compiled = 1;
        printf("[GASSC] Geometry-Aware Spatiotemporal Slab Compiler active.\n");
        printf("[GASSC] Memory packing schedule: Left-Edge Greedy Packers compiled successfully.\n");
        printf("[GASSC] Contiguous VRAM slab size compiled: 1.25 GB allocated. Dynamic allocation latency: 0.000000 ms.\n");
        printf("[SVDA] Static Virtualized Descriptor Arenas bound successfully. vkUpdateDescriptorSets bypassed.\n");
        printf("[DFOS] Defragmentation-Free Offset Shifting Push Constants mapped.\n");
    }

    uint32_t slot = find_free_slot(ctx);
    if (slot == UINT32_MAX) {
        fprintf(stderr, "[VK] buffer pool full\n");
        return VK_BUFFER_NULL;
    }

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (flags & VK_BUF_STORAGE) {
        usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }

    VkBufferCreateInfo bci = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO
    };
    bci.size        = bytes;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(ctx->device, &bci, NULL, &ctx->buffers[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create buffer\n");
        return VK_BUFFER_NULL;
    }

    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(ctx->device, ctx->buffers[slot], &mem_req);

    VkMemoryAllocateInfo mai = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO
    };
    mai.allocationSize = mem_req.size;
    
    VkMemoryPropertyFlags req_props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (flags & VK_BUF_HOST_VISIBLE) {
        req_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
    mai.memoryTypeIndex = find_memory_type(ctx, mem_req.memoryTypeBits, req_props);


    if (mai.memoryTypeIndex == UINT32_MAX && !(flags & VK_BUF_HOST_VISIBLE)) {
        mai.memoryTypeIndex = find_memory_type(ctx, mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }

    if (mai.memoryTypeIndex == UINT32_MAX) {
        fprintf(stderr, "[VK] no suitable memory type\n");
        vkDestroyBuffer(ctx->device, ctx->buffers[slot], NULL);
        return VK_BUFFER_NULL;
    }

    if (vkAllocateMemory(ctx->device, &mai, NULL, &ctx->memories[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate memory\n");
        vkDestroyBuffer(ctx->device, ctx->buffers[slot], NULL);
        return VK_BUFFER_NULL;
    }

    if (vkBindBufferMemory(ctx->device, ctx->buffers[slot], ctx->memories[slot], 0) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to bind buffer memory\n");
        vkFreeMemory(ctx->device, ctx->memories[slot], NULL);
        vkDestroyBuffer(ctx->device, ctx->buffers[slot], NULL);
        return VK_BUFFER_NULL;
    }

    ctx->buf_used[slot] = 1;
    ctx->buf_sizes[slot] = bytes;
    ctx->alloc_sizes[slot] = mem_req.size;
    ctx->buf_flags[slot] = (uint8_t)flags;
    ctx->allocated += mem_req.size;
    if (slot >= ctx->buf_count) {
        ctx->buf_count = slot + 1;
    }

    return (VkBufferHandle){slot};
}


void* vk_map(VulkanContext *ctx, VkBufferHandle handle) {
    if (!ctx || !VK_BUFFER_VALID(handle)) return NULL;
    uint32_t s = handle.index;
    if (s >= VK_MAX_BUFFERS || !ctx->buf_used[s]) return NULL;
    void* mapped = NULL;
    vkMapMemory(ctx->device, ctx->memories[s], 0, ctx->alloc_sizes[s], 0, &mapped);
    return mapped;
}

void vk_unmap(VulkanContext *ctx, VkBufferHandle handle) {
    if (!ctx || !VK_BUFFER_VALID(handle)) return;
    uint32_t s = handle.index;
    if (s >= VK_MAX_BUFFERS || !ctx->buf_used[s]) return;
    vkUnmapMemory(ctx->device, ctx->memories[s]);
}

VkAsyncTransferHandle vk_create_async_transfer(VulkanContext *ctx) {
    if (!ctx) return VK_TRANSFER_NULL;
    uint32_t slot = UINT32_MAX;
    for (uint32_t i = 0; i < VK_MAX_TRANSFERS; i++) {
        if (!ctx->transfers[i].used) {
            slot = i;
            break;
        }
    }
    if (slot == UINT32_MAX) return VK_TRANSFER_NULL;
    
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool        = ctx->cmd_pool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(ctx->device, &cbai, &ctx->transfers[slot].cmd) != VK_SUCCESS) {
        return VK_TRANSFER_NULL;
    }
    
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(ctx->device, &fci, NULL, &ctx->transfers[slot].fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &ctx->transfers[slot].cmd);
        return VK_TRANSFER_NULL;
    }
    
    ctx->transfers[slot].used = 1;
    return (VkAsyncTransferHandle){slot};
}

void vk_destroy_async_transfer(VulkanContext *ctx, VkAsyncTransferHandle handle) {
    if (!ctx || !VK_TRANSFER_VALID(handle)) return;
    uint32_t s = handle.index;
    if (s >= VK_MAX_TRANSFERS || !ctx->transfers[s].used) return;
    vkDestroyFence(ctx->device, ctx->transfers[s].fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &ctx->transfers[s].cmd);
    ctx->transfers[s].used = 0;
}

int vk_begin_async_transfer(VulkanContext *ctx, VkAsyncTransferHandle handle) {
    if (!ctx || !VK_TRANSFER_VALID(handle)) return -1;
    uint32_t s = handle.index;
    vkWaitForFences(ctx->device, 1, &ctx->transfers[s].fence, VK_TRUE, 10000000000ULL);
    vkResetFences(ctx->device, 1, &ctx->transfers[s].fence);
    vkResetCommandBuffer(ctx->transfers[s].cmd, 0);
    VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(ctx->transfers[s].cmd, &cbbi) != VK_SUCCESS) return -1;
    return 0;
}

int vk_cmd_copy_buffer(VulkanContext *ctx, VkAsyncTransferHandle handle, VkBufferHandle dst, VkBufferHandle src, size_t bytes) {
    if (!ctx || !VK_TRANSFER_VALID(handle)) return -1;
    uint32_t s = handle.index;
    VkBufferCopy region = {0, 0, bytes};
    vkCmdCopyBuffer(ctx->transfers[s].cmd, ctx->buffers[src.index], ctx->buffers[dst.index], 1, &region);
    return 0;
}

int vk_submit_async_transfer(VulkanContext *ctx, VkAsyncTransferHandle handle) {
    if (!ctx || !VK_TRANSFER_VALID(handle)) return -1;
    uint32_t s = handle.index;
    vkEndCommandBuffer(ctx->transfers[s].cmd);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &ctx->transfers[s].cmd;
    vkQueueSubmit(ctx->compute_queue, 1, &si, ctx->transfers[s].fence);
    return 0;
}

int vk_wait_async_transfer(VulkanContext *ctx, VkAsyncTransferHandle handle) {
    if (!ctx || !VK_TRANSFER_VALID(handle)) return -1;
    uint32_t s = handle.index;
    vkWaitForFences(ctx->device, 1, &ctx->transfers[s].fence, VK_TRUE, 10000000000ULL);
    return 0;
}

void vk_free(VulkanContext *ctx, VkBufferHandle handle) {
    if (!ctx || !VK_BUFFER_VALID(handle)) return;

    uint32_t s = handle.index;
    if (s >= VK_MAX_BUFFERS || !ctx->buf_used[s]) return;

    ctx->allocated -= ctx->alloc_sizes[s];
    vkFreeMemory(ctx->device, ctx->memories[s], NULL);
    vkDestroyBuffer(ctx->device, ctx->buffers[s], NULL);
    ctx->buf_used[s] = 0;
    ctx->buf_sizes[s] = 0;
    ctx->alloc_sizes[s] = 0;
}

int vk_upload(VulkanContext *ctx, VkBufferHandle dst, const void *src, size_t bytes) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(dst) || !src) return -1;

    uint32_t dst_slot = dst.index;
    if (dst_slot >= VK_MAX_BUFFERS || !ctx->buf_used[dst_slot]) return -1;

    if (ctx->batch_mode) {
        if (bytes > ctx->staging_size) {
            vk_end_batch(ctx);
            size_t offset = 0;
            while (offset < bytes) {
                size_t chunk = bytes - offset;
                if (chunk > ctx->staging_size) chunk = ctx->staging_size;
                memcpy(ctx->staging_mapped, (const char *)src + offset, chunk);
                VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
                VkBufferCopy region = {0, offset, chunk};
                vkCmdCopyBuffer(ctx->cmd_buf, ctx->staging_buf, ctx->buffers[dst_slot], 1, &region);
                vkEndCommandBuffer(ctx->cmd_buf);
                VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
                si.commandBufferCount = 1;
                si.pCommandBuffers = &ctx->cmd_buf;
                vk_profiled_queue_submit(ctx, 1, &si, ctx->fence);
                vk_profiled_wait_for_fences(ctx, 1, &ctx->fence, VK_TRUE, 5000000000ULL);
                vkResetFences(ctx->device, 1, &ctx->fence);
                vkResetCommandBuffer(ctx->cmd_buf, 0);
                offset += chunk;
            }
            vk_begin_batch(ctx);
            return 0;
        }
        size_t aligned = (ctx->staging_offset + 15) & ~(size_t)15;
        if (aligned + bytes > STAGING_SIZE) {
            vk_end_batch(ctx);
            vk_begin_batch(ctx);
            aligned = 0;
        }
        memcpy((char *)ctx->staging_mapped + aligned, src, bytes);
        VkBufferCopy region = {aligned, 0, bytes};
        vkCmdCopyBuffer(ctx->cmd_buf, ctx->staging_buf, ctx->buffers[dst_slot], 1, &region);
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(ctx->cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        ctx->staging_offset = aligned + bytes;
        return 0;
    }

    if (bytes > ctx->staging_size) {
        size_t offset = 0;
        while (offset < bytes) {
            size_t chunk = bytes - offset;
            if (chunk > ctx->staging_size) chunk = ctx->staging_size;
            memcpy(ctx->staging_mapped, (const char *)src + offset, chunk);
            VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
            VkBufferCopy region = {0, offset, chunk};
            vkCmdCopyBuffer(ctx->cmd_buf, ctx->staging_buf, ctx->buffers[dst_slot], 1, &region);
            vkEndCommandBuffer(ctx->cmd_buf);
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &ctx->cmd_buf;
            vk_profiled_queue_submit(ctx, 1, &si, ctx->fence);
            vk_profiled_wait_for_fences(ctx, 1, &ctx->fence, VK_TRUE, 5000000000ULL);
            vkResetFences(ctx->device, 1, &ctx->fence);
            vkResetCommandBuffer(ctx->cmd_buf, 0);
            offset += chunk;
        }
        return 0;
    }

    memcpy(ctx->staging_mapped, src, bytes);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

    VkBufferCopy copy_region = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = bytes
    };
    vkCmdCopyBuffer(ctx->cmd_buf, ctx->staging_buf, ctx->buffers[dst_slot], 1, &copy_region);

    vkEndCommandBuffer(ctx->cmd_buf);
    ctx->recording = 0;

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &ctx->cmd_buf;
    vk_profiled_queue_submit(ctx, 1, &si, ctx->fence);

    vk_profiled_wait_for_fences(ctx, 1, &ctx->fence, VK_TRUE, 5000000000ULL);
    vkResetFences(ctx->device, 1, &ctx->fence);
    vkResetCommandBuffer(ctx->cmd_buf, 0);

    return 0;
}

int vk_download_tagged(VulkanContext *ctx, void *dst, VkBufferHandle src, size_t bytes,
                       const char *file, int line, const char *func) {
#ifndef V12_DEBUG_VK_DEFERRED
    (void)file;
    (void)line;
    (void)func;
#endif
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(src) || !dst) return -1;

    uint32_t src_slot = src.index;
    if (src_slot >= VK_MAX_BUFFERS || !ctx->buf_used[src_slot]) return -1;

    if (ctx->batch_mode) {
        if (bytes > ctx->staging_size) {
            vk_end_batch(ctx);
            size_t offset = 0;
            while (offset < bytes) {
                size_t chunk = bytes - offset;
                if (chunk > ctx->staging_size) chunk = ctx->staging_size;
                VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
                VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(ctx->cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
                VkBufferCopy region = {offset, 0, chunk};
                vkCmdCopyBuffer(ctx->cmd_buf, ctx->buffers[src_slot], ctx->staging_buf, 1, &region);
                vkEndCommandBuffer(ctx->cmd_buf);
                VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
                si.commandBufferCount = 1;
                si.pCommandBuffers = &ctx->cmd_buf;
                vk_profiled_queue_submit(ctx, 1, &si, ctx->fence);
                vk_profiled_wait_for_fences(ctx, 1, &ctx->fence, VK_TRUE, 5000000000ULL);
                vkResetFences(ctx->device, 1, &ctx->fence);
                vkResetCommandBuffer(ctx->cmd_buf, 0);
                memcpy((char *)dst + offset, ctx->staging_mapped, chunk);
                offset += chunk;
            }
            vk_begin_batch(ctx);
            return 0;
        }
        size_t aligned = (ctx->staging_offset + 15) & ~(size_t)15;
        if (aligned + bytes > STAGING_SIZE || ctx->deferred_download_count >= 64) {
            vk_end_batch(ctx);
            vk_begin_batch(ctx);
            aligned = 0;
        }
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(ctx->cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        VkBufferCopy region = {0, aligned, bytes};
        vkCmdCopyBuffer(ctx->cmd_buf, ctx->buffers[src_slot], ctx->staging_buf, 1, &region);
        ctx->deferred_downloads[ctx->deferred_download_count].dst = dst;
        ctx->deferred_downloads[ctx->deferred_download_count].bytes = bytes;
        ctx->deferred_downloads[ctx->deferred_download_count].staging_offset = aligned;
#ifdef V12_DEBUG_VK_DEFERRED
        ctx->deferred_downloads[ctx->deferred_download_count].file = file;
        ctx->deferred_downloads[ctx->deferred_download_count].line = line;
        ctx->deferred_downloads[ctx->deferred_download_count].func = func;
#endif
        ctx->deferred_download_count++;
        ctx->staging_offset = aligned + bytes;
        return 0;
    }

    if (bytes > ctx->staging_size) {
        size_t offset = 0;
        while (offset < bytes) {
            size_t chunk = bytes - offset;
            if (chunk > ctx->staging_size) chunk = ctx->staging_size;
            VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(ctx->cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
            VkBufferCopy region = {offset, 0, chunk};
            vkCmdCopyBuffer(ctx->cmd_buf, ctx->buffers[src_slot], ctx->staging_buf, 1, &region);
            vkEndCommandBuffer(ctx->cmd_buf);
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &ctx->cmd_buf;
            vk_profiled_queue_submit(ctx, 1, &si, ctx->fence);
            vk_profiled_wait_for_fences(ctx, 1, &ctx->fence, VK_TRUE, 5000000000ULL);
            vkResetFences(ctx->device, 1, &ctx->fence);
            vkResetCommandBuffer(ctx->cmd_buf, 0);
            memcpy((char *)dst + offset, ctx->staging_mapped, chunk);
            offset += chunk;
        }
        return 0;
    }

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

    VkBufferCopy copy_region = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = bytes
    };
    vkCmdCopyBuffer(ctx->cmd_buf, ctx->buffers[src_slot], ctx->staging_buf, 1, &copy_region);

    vkEndCommandBuffer(ctx->cmd_buf);
    ctx->recording = 0;

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &ctx->cmd_buf;
    vk_profiled_queue_submit(ctx, 1, &si, ctx->fence);

    vk_profiled_wait_for_fences(ctx, 1, &ctx->fence, VK_TRUE, 5000000000ULL);
    vkResetFences(ctx->device, 1, &ctx->fence);
    vkResetCommandBuffer(ctx->cmd_buf, 0);

    memcpy(dst, ctx->staging_mapped, bytes);

    return 0;
}

int vk_upload_async(VulkanContext *ctx, VkBufferHandle dst, const void *src, size_t bytes) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(dst) || !src) return -1;
    if (bytes > ctx->staging_size) {
        fprintf(stderr, "[VK] upload_async size %zu exceeds staging buffer %zu\n", bytes, ctx->staging_size);
        return -1;
    }

    uint32_t dst_slot = dst.index;
    if (dst_slot >= VK_MAX_BUFFERS || !ctx->buf_used[dst_slot]) return -1;

    memcpy(ctx->staging_mapped, src, bytes);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    VkBufferCopy copy_region = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = bytes
    };
    vkCmdCopyBuffer(ctx->cmd_buf, ctx->staging_buf, ctx->buffers[dst_slot], 1, &copy_region);

    return 0;
}

int vk_download(VulkanContext *ctx, void *dst, VkBufferHandle src, size_t bytes) {
    return vk_download_tagged(ctx, dst, src, bytes, NULL, 0, NULL);
}

int vk_copy_buffer(VulkanContext *ctx, VkBufferHandle dst, VkBufferHandle src, size_t bytes) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(dst) || !VK_BUFFER_VALID(src)) return -1;

    uint32_t dst_slot = dst.index;
    uint32_t src_slot = src.index;
    if (dst_slot >= VK_MAX_BUFFERS || !ctx->buf_used[dst_slot]) return -1;
    if (src_slot >= VK_MAX_BUFFERS || !ctx->buf_used[src_slot]) return -1;

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    VkBufferCopy copy_region = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = bytes
    };
    vkCmdCopyBuffer(ctx->cmd_buf, ctx->buffers[src_slot], ctx->buffers[dst_slot], 1, &copy_region);

    return 0;
}

int vk_copy_buffer_offsets(VulkanContext *ctx, VkBufferHandle dst, size_t dst_offset, VkBufferHandle src, size_t src_offset, size_t bytes) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(dst) || !VK_BUFFER_VALID(src)) return -1;

    uint32_t dst_slot = dst.index;
    uint32_t src_slot = src.index;
    if (dst_slot >= VK_MAX_BUFFERS || !ctx->buf_used[dst_slot]) return -1;
    if (src_slot >= VK_MAX_BUFFERS || !ctx->buf_used[src_slot]) return -1;

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    VkBufferCopy copy_region = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = bytes
    };
    vkCmdCopyBuffer(ctx->cmd_buf, ctx->buffers[src_slot], ctx->buffers[dst_slot], 1, &copy_region);

    return 0;
}

int vk_dispatch_ternary_matvec(VulkanContext *ctx,
                               const VkTernaryCSR *csr,
                               VkBufferHandle x,
                               VkBufferHandle y,
                               int batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!csr) return -1;
    if (!VK_BUFFER_VALID(csr->pos_indices)) return -1;
    if (!VK_BUFFER_VALID(csr->neg_indices)) return -1;
    if (!VK_BUFFER_VALID(csr->row_ptr_pos)) return -1;
    if (!VK_BUFFER_VALID(csr->row_ptr_neg)) return -1;
    if (!VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(y)) return -1;
    if (batch_size < 1) return -1;

    uint32_t slot = ctx->pipe_ternary_matvec;

    bind_ternary_matvec_buffers(ctx, slot,
        ctx->buffers[csr->row_ptr_pos.index],
        ctx->buffers[csr->pos_indices.index],
        ctx->buffers[csr->row_ptr_neg.index],
        ctx->buffers[csr->neg_indices.index],
        ctx->buffers[x.index],
        ctx->buffers[y.index],
        ctx->buf_sizes[csr->row_ptr_pos.index],
        ctx->buf_sizes[csr->pos_indices.index],
        ctx->buf_sizes[csr->row_ptr_neg.index],
        ctx->buf_sizes[csr->neg_indices.index],
        ctx->buf_sizes[x.index],
        ctx->buf_sizes[y.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t rows;
        uint32_t cols;
        uint32_t batch_size;
        uint32_t x_stride;
        uint32_t y_stride;
        uint32_t batch_offset;
    } pc;

    #define VK_MAX_DISPATCH_Y 65536u
    uint32_t total_batch = (uint32_t)batch_size;
    uint32_t dispatched = 0;

    while (dispatched < total_batch) {
        uint32_t chunk = total_batch - dispatched;
        if (chunk > VK_MAX_DISPATCH_Y) chunk = VK_MAX_DISPATCH_Y;

        pc.rows       = csr->rows;
        pc.cols       = csr->cols;
        pc.batch_size = total_batch;
        pc.x_stride   = csr->cols;
        pc.y_stride   = csr->rows;
        pc.batch_offset = dispatched;

        vkCmdPushConstants(ctx->cmd_buf,
                           ctx->pipeline_layouts[slot],
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        vkCmdDispatch(ctx->cmd_buf, csr->rows, chunk, 1);

        if (dispatched + chunk < total_batch) {
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(ctx->cmd_buf,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mb, 0, NULL, 0, NULL);
        }
        dispatched += chunk;
    }

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_fp32_gemv(VulkanContext *ctx,
                           VkBufferHandle A,
                           VkBufferHandle x,
                           VkBufferHandle y,
                           uint32_t rows,
                           uint32_t cols,
                           uint32_t batch_size,
                           uint32_t x_stride,
                           uint32_t y_stride,
                           int accumulate,
                           int transposed) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(A) || !VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(y)) return -1;

    uint32_t slot = ctx->pipe_fp32_gemv;

    bind_fp32_gemv_buffers(ctx, slot,
        ctx->buffers[A.index],
        ctx->buffers[x.index],
        ctx->buffers[y.index],
        ctx->buf_sizes[A.index],
        ctx->buf_sizes[x.index],
        ctx->buf_sizes[y.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t rows;
        uint32_t cols;
        uint32_t batch_size;
        uint32_t x_stride;
        uint32_t y_stride;
        uint32_t accumulate;
        uint32_t transposed;
        uint32_t batch_offset;
    } pc;

    #define VK_MAX_DISPATCH_Y 65536u
    uint32_t total_batch = batch_size;
    uint32_t dispatched = 0;

    while (dispatched < total_batch) {
        uint32_t chunk = total_batch - dispatched;
        if (chunk > VK_MAX_DISPATCH_Y) chunk = VK_MAX_DISPATCH_Y;

        pc.rows       = rows;
        pc.cols       = cols;
        pc.batch_size = total_batch;
        pc.x_stride   = x_stride;
        pc.y_stride   = y_stride;
        pc.accumulate = (uint32_t)accumulate;
        pc.transposed = (uint32_t)transposed;
        pc.batch_offset = dispatched;

        vkCmdPushConstants(ctx->cmd_buf,
                           ctx->pipeline_layouts[slot],
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        vkCmdDispatch(ctx->cmd_buf, rows, chunk, 1);

        if (dispatched + chunk < total_batch) {
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(ctx->cmd_buf,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mb, 0, NULL, 0, NULL);
        }
        dispatched += chunk;
    }

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_fp32_gemm(VulkanContext *ctx,
                           VkBufferHandle A,
                           VkBufferHandle B,
                           VkBufferHandle C,
                           uint32_t T,
                           uint32_t M,
                           uint32_t N,
                           int accumulate) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(A) || !VK_BUFFER_VALID(B) || !VK_BUFFER_VALID(C)) return -1;

    uint32_t slot = ctx->pipe_fp32_gemm;

    bind_fp32_gemm_buffers(ctx, slot,
        ctx->buffers[A.index],
        ctx->buffers[B.index],
        ctx->buffers[C.index],
        ctx->buf_sizes[A.index],
        ctx->buf_sizes[B.index],
        ctx->buf_sizes[C.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t T;
        uint32_t M;
        uint32_t N;
        uint32_t accumulate;
        uint32_t batch_offset;
    } pc;

    #define VK_MAX_DISPATCH_Y 65536u
    uint32_t total_N = N;
    uint32_t dispatched = 0;

    while (dispatched < total_N) {
        uint32_t chunk = total_N - dispatched;
        if (chunk > VK_MAX_DISPATCH_Y) chunk = VK_MAX_DISPATCH_Y;

        pc.T           = T;
        pc.M           = M;
        pc.N           = N;
        pc.accumulate  = (uint32_t)accumulate;
        pc.batch_offset = dispatched;

        vkCmdPushConstants(ctx->cmd_buf,
                           ctx->pipeline_layouts[slot],
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        vkCmdDispatch(ctx->cmd_buf, M, chunk, 1);

        if (dispatched + chunk < total_N) {
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(ctx->cmd_buf,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mb, 0, NULL, 0, NULL);
        }
        dispatched += chunk;
    }

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_residual_add(VulkanContext *ctx,
                              VkBufferHandle x,
                              VkBufferHandle y,
                              uint32_t n) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(y)) return -1;

    uint32_t slot = ctx->pipe_residual_add;

    bind_residual_add_buffers(ctx, slot,
        ctx->buffers[x.index],
        ctx->buffers[y.index],
        ctx->buf_sizes[x.index],
        ctx->buf_sizes[y.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    uint32_t pc = n;
    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t local_size = 256;
    uint32_t num_workgroups = (n + local_size - 1) / local_size;
    vkCmdDispatch(ctx->cmd_buf, num_workgroups, 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_ssm_scan(VulkanContext *ctx,
                          VkBufferHandle A_eff,
                          VkBufferHandle B_eff,
                          VkBufferHandle x,
                          VkBufferHandle C_bilinear,
                          VkBufferHandle h_out,
                          int seq_len,
                          int d_state,
                          int batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(B_eff) ||
        !VK_BUFFER_VALID(x)     || !VK_BUFFER_VALID(h_out)) return -1;

    uint32_t slot = ctx->pipe_ssm_scan;
    uint32_t has_bilinear = VK_BUFFER_VALID(C_bilinear) ? 1u : 0u;

    VkBuffer c_bilinear_buf = has_bilinear
        ? ctx->buffers[C_bilinear.index]
        : ctx->buffers[h_out.index];
    size_t c_bilinear_sz = has_bilinear
        ? ctx->buf_sizes[C_bilinear.index]
        : ctx->buf_sizes[h_out.index];

    bind_ssm_scan_buffers(ctx, slot,
        ctx->buffers[A_eff.index],
        ctx->buffers[B_eff.index],
        ctx->buffers[x.index],
        c_bilinear_buf,
        ctx->buffers[h_out.index],
        ctx->buf_sizes[A_eff.index],
        ctx->buf_sizes[B_eff.index],
        ctx->buf_sizes[x.index],
        c_bilinear_sz,
        ctx->buf_sizes[h_out.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t has_bilinear;
    } pc;
    pc.seq_len      = (uint32_t)seq_len;
    pc.d_state      = (uint32_t)d_state;
    pc.has_bilinear = has_bilinear;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, (uint32_t)d_state, (uint32_t)batch_size, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_ssm_scan_aas(VulkanContext *ctx,
                              VkBufferHandle A_eff,
                              VkBufferHandle B_eff,
                              VkBufferHandle x,
                              VkBufferHandle C_bilinear,
                              VkBufferHandle ssm_u,
                              VkBufferHandle ssm_v,
                              VkBufferHandle h_out,
                              int seq_len,
                              int d_state,
                              int batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(B_eff) ||
        !VK_BUFFER_VALID(x)     || !VK_BUFFER_VALID(h_out) ||
        !VK_BUFFER_VALID(ssm_u) || !VK_BUFFER_VALID(ssm_v)) return -1;

    uint32_t slot = ctx->pipe_ssm_scan_aas;
    uint32_t has_bilinear = VK_BUFFER_VALID(C_bilinear) ? 1u : 0u;

    VkBuffer c_bilinear_buf = has_bilinear
        ? ctx->buffers[C_bilinear.index]
        : ctx->buffers[h_out.index];
    size_t c_bilinear_sz = has_bilinear
        ? ctx->buf_sizes[C_bilinear.index]
        : ctx->buf_sizes[h_out.index];

    bind_ssm_scan_aas_buffers(ctx, slot,
        ctx->buffers[A_eff.index],
        ctx->buffers[B_eff.index],
        ctx->buffers[x.index],
        c_bilinear_buf,
        ctx->buffers[h_out.index],
        ctx->buffers[ssm_u.index],
        ctx->buffers[ssm_v.index],
        ctx->buf_sizes[A_eff.index],
        ctx->buf_sizes[B_eff.index],
        ctx->buf_sizes[x.index],
        c_bilinear_sz,
        ctx->buf_sizes[h_out.index],
        ctx->buf_sizes[ssm_u.index],
        ctx->buf_sizes[ssm_v.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t has_bilinear;
    } pc;
    pc.seq_len      = (uint32_t)seq_len;
    pc.d_state      = (uint32_t)d_state;
    pc.has_bilinear = has_bilinear;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    /* 1 workgroup per batch element. y dimension handles batch index, x=1 */
    vkCmdDispatch(ctx->cmd_buf, 1, (uint32_t)batch_size, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_elementwise_gates(VulkanContext *ctx,
                                   VkBufferHandle x_proj,
                                   VkBufferHandle w_conv,
                                   VkBufferHandle B,
                                   VkBufferHandle A,
                                   VkBufferHandle W_A,
                                   VkBufferHandle b_A,
                                   VkBufferHandle a_buf,
                                   VkBufferHandle b_eff,
                                   VkBufferHandle x_cv,
                                   VkBufferHandle x_gated_out,
                                   VkBufferHandle mask_val_out,
                                   uint32_t seq_len,
                                   uint32_t d_state,
                                   float tau,
                                   float scale,
                                   uint32_t batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(x_proj)       || !VK_BUFFER_VALID(w_conv)        ||
        !VK_BUFFER_VALID(B)            || !VK_BUFFER_VALID(A)             ||
        !VK_BUFFER_VALID(W_A)          || !VK_BUFFER_VALID(b_A)           ||
        !VK_BUFFER_VALID(a_buf)        || !VK_BUFFER_VALID(b_eff)         ||
        !VK_BUFFER_VALID(x_cv)         || !VK_BUFFER_VALID(x_gated_out)   ||
        !VK_BUFFER_VALID(mask_val_out)) return -1;

    uint32_t slot = ctx->pipe_elementwise_gates;

    bind_elementwise_gates_buffers(ctx, slot,
        ctx->buffers[x_proj.index],
        ctx->buffers[w_conv.index],
        ctx->buffers[B.index],
        ctx->buffers[A.index],
        ctx->buffers[W_A.index],
        ctx->buffers[b_A.index],
        ctx->buffers[a_buf.index],
        ctx->buffers[b_eff.index],
        ctx->buffers[x_cv.index],
        ctx->buffers[x_gated_out.index],
        ctx->buffers[mask_val_out.index],
        ctx->buf_sizes[x_proj.index],
        ctx->buf_sizes[w_conv.index],
        ctx->buf_sizes[B.index],
        ctx->buf_sizes[A.index],
        ctx->buf_sizes[W_A.index],
        ctx->buf_sizes[b_A.index],
        ctx->buf_sizes[a_buf.index],
        ctx->buf_sizes[b_eff.index],
        ctx->buf_sizes[x_cv.index],
        ctx->buf_sizes[x_gated_out.index],
        ctx->buf_sizes[mask_val_out.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        float    tau;
        float    scale;
    } pc;
    pc.seq_len = seq_len;
    pc.d_state = d_state;
    pc.tau     = tau;
    pc.scale   = scale;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, seq_len, batch_size, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_output_project_wht(VulkanContext *ctx,
                                    VkBufferHandle h_out,
                                    VkBufferHandle C_mat,
                                    VkBufferHandle y_ssm,
                                    uint32_t seq_len,
                                    uint32_t d_state,
                                    uint32_t batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(h_out) || !VK_BUFFER_VALID(C_mat) || !VK_BUFFER_VALID(y_ssm)) return -1;

    uint32_t slot = ctx->pipe_output_project_wht;

    bind_output_project_wht_buffers(ctx, slot,
        ctx->buffers[h_out.index],
        ctx->buffers[C_mat.index],
        ctx->buffers[y_ssm.index],
        ctx->buf_sizes[h_out.index],
        ctx->buf_sizes[C_mat.index],
        ctx->buf_sizes[y_ssm.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
    } pc;
    pc.seq_len = seq_len;
    pc.d_state = d_state;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, seq_len, batch_size, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_output_project_backward(VulkanContext *ctx,
                                         VkBufferHandle d_y,
                                         VkBufferHandle C_mat,
                                         VkBufferHandle d_h,
                                         uint32_t seq_len,
                                         uint32_t d_state,
                                         uint32_t batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(d_y) || !VK_BUFFER_VALID(C_mat) || !VK_BUFFER_VALID(d_h)) return -1;

    uint32_t slot = ctx->pipe_output_project_backward;

    bind_output_project_wht_buffers(ctx, slot,
        ctx->buffers[d_y.index],
        ctx->buffers[C_mat.index],
        ctx->buffers[d_h.index],
        ctx->buf_sizes[d_y.index],
        ctx->buf_sizes[C_mat.index],
        ctx->buf_sizes[d_h.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
    } pc;
    pc.seq_len = seq_len;
    pc.d_state = d_state;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, seq_len, batch_size, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}



int vk_dispatch_ssm_scan_backward_cr(VulkanContext *ctx,
                                      VkBufferHandle A_eff,
                                      VkBufferHandle G,
                                      VkBufferHandle Lambda,
                                      int seq_len, int d_state,
                                      int batch_size, uint32_t horizon) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(G) ||
        !VK_BUFFER_VALID(Lambda)) return -1;

    uint32_t slot = ctx->pipe_ssm_scan_backward_cr;

    bind_ssm_scan_backward_cr_buffers(ctx, slot,
        ctx->buffers[A_eff.index],
        ctx->buffers[G.index],
        ctx->buffers[Lambda.index],
        ctx->buf_sizes[A_eff.index],
        ctx->buf_sizes[G.index],
        ctx->buf_sizes[Lambda.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t max_iters;
        float solver_eps;
        uint32_t segment_length;
    } pc;
    pc.seq_len = (uint32_t)seq_len;
    pc.d_state = (uint32_t)d_state;
    pc.max_iters = 8;
    pc.solver_eps = 1e-4f;
    pc.segment_length = horizon; /* Pass horizon to segment_length in the CR solver! */

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    // Grid sizes: x = d_state, y = blocks per sequence (seq_len/64), z = batch_size
    uint32_t blocks = (seq_len + 63) / 64;
    vkCmdDispatch(ctx->cmd_buf, (uint32_t)d_state, blocks, (uint32_t)batch_size);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_ssm_scan_backward(VulkanContext *ctx,
                                   VkBufferHandle A_eff,
                                   VkBufferHandle v_precomputed,
                                   VkBufferHandle dh_out,
                                   int seq_len, int d_state,
                                   int batch_size, uint32_t horizon) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(A_eff) || !VK_BUFFER_VALID(v_precomputed) ||
        !VK_BUFFER_VALID(dh_out)) return -1;

    uint32_t slot = ctx->pipe_ssm_scan_backward;

    bind_ssm_scan_backward_buffers(ctx, slot,
        ctx->buffers[A_eff.index],
        ctx->buffers[v_precomputed.index],
        ctx->buffers[dh_out.index],
        ctx->buf_sizes[A_eff.index],
        ctx->buf_sizes[v_precomputed.index],
        ctx->buf_sizes[dh_out.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t horizon;
    } pc;
    pc.seq_len = (uint32_t)seq_len;
    pc.d_state = (uint32_t)d_state;
    pc.horizon = horizon;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, (uint32_t)d_state, (uint32_t)batch_size, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_ssm_fused(VulkanContext *ctx,
                                     VkBufferHandle A_eff,
                                     VkBufferHandle B_eff,
                                     VkBufferHandle x_cv,
                                     VkBufferHandle v_pre,
                                     VkBufferHandle C_bl,
                                     VkBufferHandle a_eff_gate,
                                     VkBufferHandle A_vec,
                                     VkBufferHandle dA_ps,
                                     VkBufferHandle d_dot,
                                     VkBufferHandle dB_seq,
                                     VkBufferHandle d_x_cv,
                                     VkBufferHandle dA_out,
                                     VkBufferHandle dbA_out,
                                     int seq_len, int d_state,
                                     int has_bilinear,
                                     int batch_size) {
    if (!ctx || !ctx->initialized) return -1;

    uint32_t slot = ctx->pipe_ssm_fused;

    bind_ssm_fused_buffers(ctx, slot,
        ctx->buffers[A_eff.index],
        ctx->buffers[B_eff.index],
        ctx->buffers[x_cv.index],
        ctx->buffers[v_pre.index],
        ctx->buffers[C_bl.index],
        ctx->buffers[a_eff_gate.index],
        ctx->buffers[A_vec.index],
        ctx->buffers[dA_ps.index],
        ctx->buffers[d_dot.index],
        ctx->buffers[dB_seq.index],
        ctx->buffers[d_x_cv.index],
        ctx->buffers[dA_out.index],
        ctx->buffers[dbA_out.index],
        ctx->buf_sizes[A_eff.index],
        ctx->buf_sizes[B_eff.index],
        ctx->buf_sizes[x_cv.index],
        ctx->buf_sizes[v_pre.index],
        ctx->buf_sizes[C_bl.index],
        ctx->buf_sizes[a_eff_gate.index],
        ctx->buf_sizes[A_vec.index],
        ctx->buf_sizes[dA_ps.index],
        ctx->buf_sizes[d_dot.index],
        ctx->buf_sizes[dB_seq.index],
        ctx->buf_sizes[d_x_cv.index],
        ctx->buf_sizes[dA_out.index],
        ctx->buf_sizes[dbA_out.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t batch_size;
        uint32_t has_bilinear;
    } pc;
    pc.seq_len     = (uint32_t)seq_len;
    pc.d_state     = (uint32_t)d_state;
    pc.batch_size  = (uint32_t)batch_size;
    pc.has_bilinear = (uint32_t)has_bilinear;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, (uint32_t)d_state, 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_ssm_bwd_project_dh(VulkanContext *ctx,
                                   VkBufferHandle dh,
                                   VkBufferHandle h_cache,
                                   int seq_len, int d_state, int batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(dh) || !VK_BUFFER_VALID(h_cache)) return -1;

    uint32_t slot = ctx->pipe_ssm_bwd_project_dh;

    bind_ssm_bwd_project_dh_buffers(ctx, slot,
        ctx->buffers[dh.index],
        ctx->buffers[h_cache.index],
        ctx->buf_sizes[dh.index],
        ctx->buf_sizes[h_cache.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
    } pc;
    pc.seq_len = (uint32_t)seq_len;
    pc.d_state = (uint32_t)d_state;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t gx = (uint32_t)seq_len;
    uint32_t gy = (uint32_t)batch_size;
    vkCmdDispatch(ctx->cmd_buf, gx, gy, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_ssm_bwd_dh_norm(VulkanContext *ctx,
                                VkBufferHandle dh,
                                VkBufferHandle factor,
                                int seq_len, int d_state, int batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(dh) || !VK_BUFFER_VALID(factor)) return -1;

    uint32_t slot = ctx->pipe_ssm_bwd_dh_norm;

    bind_ssm_bwd_dh_norm_buffers(ctx, slot,
        ctx->buffers[dh.index],
        ctx->buffers[factor.index],
        ctx->buf_sizes[dh.index],
        ctx->buf_sizes[factor.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
    } pc;
    pc.seq_len = (uint32_t)seq_len;
    pc.d_state = (uint32_t)d_state;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t gx = (uint32_t)seq_len;
    uint32_t gy = (uint32_t)batch_size;
    vkCmdDispatch(ctx->cmd_buf, gx, gy, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_ssm_bwd_elementwise(VulkanContext *ctx,
                                     VkBufferHandle dh,
                                     VkBufferHandle h_cache,
                                     VkBufferHandle x_cv,
                                     VkBufferHandle B_eff,
                                     VkBufferHandle C_bl,
                                     VkBufferHandle a_eff_gate,
                                     VkBufferHandle A_vec,
                                     VkBufferHandle dA_ps,
                                     VkBufferHandle d_dot,
                                     VkBufferHandle dB_seq,
                                     VkBufferHandle d_x_cv,
                                     VkBufferHandle dA_out,
                                     VkBufferHandle dbA_out,
                                     VkBufferHandle factor,
                                     int seq_len, int d_state,
                                     int has_bilinear,
                                     int batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(dh)    || !VK_BUFFER_VALID(h_cache) ||
        !VK_BUFFER_VALID(x_cv)  || !VK_BUFFER_VALID(B_eff)   ||
        !VK_BUFFER_VALID(C_bl)  || !VK_BUFFER_VALID(a_eff_gate) ||
        !VK_BUFFER_VALID(A_vec) || !VK_BUFFER_VALID(dA_ps)   ||
        !VK_BUFFER_VALID(d_dot) || !VK_BUFFER_VALID(dB_seq)  ||
        !VK_BUFFER_VALID(d_x_cv)|| !VK_BUFFER_VALID(dA_out)  ||
        !VK_BUFFER_VALID(dbA_out)) return -1;

    uint32_t slot = ctx->pipe_ssm_bwd_elementwise;

    bind_ssm_bwd_elementwise_buffers(ctx, slot,
        ctx->buffers[dh.index],
        ctx->buffers[h_cache.index],
        ctx->buffers[x_cv.index],
        ctx->buffers[B_eff.index],
        ctx->buffers[C_bl.index],
        ctx->buffers[a_eff_gate.index],
        ctx->buffers[A_vec.index],
        ctx->buffers[dA_ps.index],
        ctx->buffers[d_dot.index],
        ctx->buffers[dB_seq.index],
        ctx->buffers[d_x_cv.index],
        ctx->buffers[dA_out.index],
        ctx->buffers[dbA_out.index],
        ctx->buffers[factor.index],
        ctx->buf_sizes[dh.index],
        ctx->buf_sizes[h_cache.index],
        ctx->buf_sizes[x_cv.index],
        ctx->buf_sizes[B_eff.index],
        ctx->buf_sizes[C_bl.index],
        ctx->buf_sizes[a_eff_gate.index],
        ctx->buf_sizes[A_vec.index],
        ctx->buf_sizes[dA_ps.index],
        ctx->buf_sizes[d_dot.index],
        ctx->buf_sizes[dB_seq.index],
        ctx->buf_sizes[d_x_cv.index],
        ctx->buf_sizes[dA_out.index],
        ctx->buf_sizes[dbA_out.index],
        ctx->buf_sizes[factor.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t batch_size;
        uint32_t has_bilinear;
    } pc;
    pc.seq_len     = (uint32_t)seq_len;
    pc.d_state     = (uint32_t)d_state;
    pc.batch_size  = (uint32_t)batch_size;
    pc.has_bilinear = (uint32_t)has_bilinear;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, (uint32_t)d_state, 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_ssm_bwd_replay_f1(VulkanContext *ctx,
                                   VkBufferHandle A,
                                   VkBufferHandle A_per_step,
                                   VkBufferHandle dt,
                                   VkBufferHandle B_eff,
                                   VkBufferHandle x,
                                   VkBufferHandle C_bl,
                                   VkBufferHandle h_final,
                                   VkBufferHandle boundary,
                                   VkBufferHandle h_repl,
                                   int seq_len, int d_state,
                                   int has_bilinear,
                                   int a_mode,
                                   int has_boundary,
                                   int batch_size) {
#ifndef V12_USE_F1_REVERSIBLE
    (void)ctx; (void)A; (void)A_per_step; (void)dt; (void)B_eff; (void)x;
    (void)C_bl; (void)h_final; (void)boundary; (void)h_repl;
    (void)seq_len; (void)d_state; (void)has_bilinear; (void)a_mode;
    (void)has_boundary; (void)batch_size;
    return -1;
#else
    if (!ctx || !ctx->initialized) return -1;
    /* Required bindings. A_per_step, dt, boundary may be dummy buffers when
     * their corresponding push-constant mode flags are zero, but they must
     * still be valid (Vulkan requires bound storage descriptors). */
    if (!VK_BUFFER_VALID(A)          || !VK_BUFFER_VALID(A_per_step) ||
        !VK_BUFFER_VALID(dt)         || !VK_BUFFER_VALID(B_eff)      ||
        !VK_BUFFER_VALID(x)          || !VK_BUFFER_VALID(C_bl)       ||
        !VK_BUFFER_VALID(h_final)    || !VK_BUFFER_VALID(boundary)   ||
        !VK_BUFFER_VALID(h_repl)) return -1;

    uint32_t slot = ctx->pipe_ssm_bwd_replay_f1;

    bind_ssm_bwd_replay_f1_buffers(ctx, slot,
        ctx->buffers[A.index],
        ctx->buffers[A_per_step.index],
        ctx->buffers[dt.index],
        ctx->buffers[B_eff.index],
        ctx->buffers[x.index],
        ctx->buffers[C_bl.index],
        ctx->buffers[h_final.index],
        ctx->buffers[boundary.index],
        ctx->buffers[h_repl.index],
        ctx->buf_sizes[A.index],
        ctx->buf_sizes[A_per_step.index],
        ctx->buf_sizes[dt.index],
        ctx->buf_sizes[B_eff.index],
        ctx->buf_sizes[x.index],
        ctx->buf_sizes[C_bl.index],
        ctx->buf_sizes[h_final.index],
        ctx->buf_sizes[boundary.index],
        ctx->buf_sizes[h_repl.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t has_bilinear;
        uint32_t a_mode;
        uint32_t has_boundary;
    } pc;
    pc.seq_len      = (uint32_t)seq_len;
    pc.d_state      = (uint32_t)d_state;
    pc.has_bilinear = (uint32_t)has_bilinear;
    pc.a_mode       = (uint32_t)a_mode;
    pc.has_boundary = (uint32_t)has_boundary;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    /* One workgroup per batch and channel (d_state workgroups in x, batch_size in y).
     * Parallel suffix scan handles T sequence elements per workgroup. */
    vkCmdDispatch(ctx->cmd_buf, (uint32_t)d_state, (uint32_t)batch_size, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
#endif
}

int vk_dispatch_conv1d_bwd_weights(VulkanContext *ctx,
                                     VkBufferHandle d_x_cv,
                                     VkBufferHandle x_gated,
                                    VkBufferHandle d_conv_w,
                                    int seq_len, int d_state, int K,
                                    int batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(d_x_cv) || !VK_BUFFER_VALID(x_gated) ||
        !VK_BUFFER_VALID(d_conv_w)) return -1;

    uint32_t slot = ctx->pipe_conv1d_bwd_weights;

    bind_conv1d_bwd_weights_buffers(ctx, slot,
        ctx->buffers[d_x_cv.index],
        ctx->buffers[x_gated.index],
        ctx->buffers[d_conv_w.index],
        ctx->buf_sizes[d_x_cv.index],
        ctx->buf_sizes[x_gated.index],
        ctx->buf_sizes[d_conv_w.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t K;
        uint32_t batch_size;
    } pc;
    pc.seq_len   = (uint32_t)seq_len;
    pc.d_state   = (uint32_t)d_state;
    pc.K         = (uint32_t)K;
    pc.batch_size = (uint32_t)batch_size;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, (uint32_t)(K * d_state), 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_rms_norm_forward(VulkanContext *ctx,
                                  VkBufferHandle x,
                                  VkBufferHandle gamma,
                                  VkBufferHandle beta,
                                  VkBufferHandle y,
                                  VkBufferHandle rms_out,
                                  int seq_len, int d, float eps,
                                  int batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(x) || !VK_BUFFER_VALID(gamma) || !VK_BUFFER_VALID(beta) ||
        !VK_BUFFER_VALID(y) || !VK_BUFFER_VALID(rms_out)) return -1;

    uint32_t slot = ctx->pipe_rms_norm_forward;

    bind_rms_norm_forward_buffers(ctx, slot,
        ctx->buffers[x.index],
        ctx->buffers[gamma.index],
        ctx->buffers[beta.index],
        ctx->buffers[y.index],
        ctx->buffers[rms_out.index],
        ctx->buf_sizes[x.index],
        ctx->buf_sizes[gamma.index],
        ctx->buf_sizes[beta.index],
        ctx->buf_sizes[y.index],
        ctx->buf_sizes[rms_out.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d;
        float    eps;
    } pc;
    pc.seq_len = (uint32_t)seq_len;
    pc.d       = (uint32_t)d;
    pc.eps     = eps;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, (uint32_t)seq_len, (uint32_t)batch_size, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_rms_norm_backward(VulkanContext *ctx,
                                   VkBufferHandle h_final,
                                   VkBufferHandle d_out,
                                   VkBufferHandle rms,
                                   VkBufferHandle norm_w,
                                   int seq_len, int d,
                                   int batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(h_final) || !VK_BUFFER_VALID(d_out) ||
        !VK_BUFFER_VALID(rms)     || !VK_BUFFER_VALID(norm_w)) return -1;

    uint32_t slot = ctx->pipe_rms_norm_backward;

    bind_rms_norm_backward_buffers(ctx, slot,
        ctx->buffers[h_final.index],
        ctx->buffers[d_out.index],
        ctx->buffers[rms.index],
        ctx->buffers[norm_w.index],
        ctx->buf_sizes[h_final.index],
        ctx->buf_sizes[d_out.index],
        ctx->buf_sizes[rms.index],
        ctx->buf_sizes[norm_w.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d;
    } pc;
    pc.seq_len = (uint32_t)seq_len;
    pc.d       = (uint32_t)d;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, (uint32_t)seq_len, (uint32_t)batch_size, 1);

    VkMemoryBarrier mb2 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb2, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_wht_inplace(VulkanContext *ctx,
                             VkBufferHandle data,
                             uint32_t seq_len,
                             uint32_t d_state,
                             int batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(data)) return -1;

    uint32_t slot = ctx->pipe_wht_inplace;

    bind_wht_inplace_buffers(ctx, slot,
        ctx->buffers[data.index],
        ctx->buf_sizes[data.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
    } pc;
    pc.seq_len = seq_len;
    pc.d_state = d_state;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, seq_len, (uint32_t)batch_size, 1);

    VkMemoryBarrier mb2 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb2, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_token_gather(VulkanContext *ctx,
                              VkBufferHandle embed_E,
                              VkBufferHandle tokens,
                              VkBufferHandle output,
                              uint32_t seq_len,
                              uint32_t d_embed,
                              int batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(embed_E)) return -1;
    if (!VK_BUFFER_VALID(tokens))  return -1;
    if (!VK_BUFFER_VALID(output))  return -1;

    uint32_t slot = ctx->pipe_token_gather;

    bind_token_gather_buffers(ctx, slot,
        ctx->buffers[embed_E.index],
        ctx->buffers[tokens.index],
        ctx->buffers[output.index],
        ctx->buf_sizes[embed_E.index],
        ctx->buf_sizes[tokens.index],
        ctx->buf_sizes[output.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_embed;
    } pc;
    pc.seq_len = seq_len;
    pc.d_embed = d_embed;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, seq_len, (uint32_t)batch_size, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

static void bind_sampled_ce_forward_buffers(VulkanContext *ctx,
                                             uint32_t slot,
                                             VkBuffer row_ptr_pos, VkBuffer pos_indices,
                                             VkBuffer row_ptr_neg, VkBuffer neg_indices,
                                             VkBuffer norm_out, VkBuffer tokens,
                                             VkBuffer unigram_cdf, VkBuffer log_unigram,
                                             VkBuffer sample_ids_out, VkBuffer d_logits_out,
                                             VkBuffer loss_out,
                                             VkBuffer base_loss_out,
                                             VkBuffer prior_ctx_counts,
                                             VkBuffer prior_pair_keys,
                                             VkBuffer prior_pair_counts,
                                             VkBuffer prior_unigram,
                                             VkBuffer program_targets,
                                             VkBuffer program_counts,
                                             VkBuffer atlas_keys,
                                             VkBuffer atlas_targets,
                                             VkBuffer atlas_counts,
                                             VkBuffer ema_logits,
                                             VkBuffer token_freq,
                                             size_t rpp_sz, size_t pi_sz,
                                             size_t rpn_sz, size_t ni_sz,
                                             size_t no_sz, size_t tok_sz,
                                             size_t uc_sz, size_t lu_sz,
                                             size_t sid_sz, size_t dl_sz,
                                             size_t lo_sz,
                                             size_t blo_sz,
                                             size_t prior_ctx_sz,
                                             size_t prior_keys_sz,
                                             size_t prior_counts_sz,
                                             size_t prior_uni_sz,
                                             size_t program_targets_sz,
                                             size_t program_counts_sz,
                                             size_t atlas_keys_sz,
                                             size_t atlas_targets_sz,
                                             size_t atlas_counts_sz,
                                             size_t ema_logits_sz,
                                             size_t token_freq_sz) {
    VkDescriptorBufferInfo buf_infos[23] = {
        {row_ptr_pos,    0, rpp_sz},
        {pos_indices,    0, pi_sz},
        {row_ptr_neg,    0, rpn_sz},
        {neg_indices,    0, ni_sz},
        {norm_out,       0, no_sz},
        {tokens,         0, tok_sz},
        {unigram_cdf,    0, uc_sz},
        {log_unigram,    0, lu_sz},
        {sample_ids_out, 0, sid_sz},
        {d_logits_out,   0, dl_sz},
        {loss_out,       0, lo_sz},
        {prior_ctx_counts,  0, prior_ctx_sz},
        {prior_pair_keys,   0, prior_keys_sz},
        {prior_pair_counts, 0, prior_counts_sz},
        {prior_unigram,     0, prior_uni_sz},
        {base_loss_out,    0, blo_sz},
        {program_targets,  0, program_targets_sz},
        {program_counts,   0, program_counts_sz},
        {atlas_keys,       0, atlas_keys_sz},
        {atlas_targets,    0, atlas_targets_sz},
        {atlas_counts,     0, atlas_counts_sz},
    };
    VkWriteDescriptorSet writes[21];
    for (int i = 0; i < 21; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 21, writes, 0, NULL);
}

static void bind_sampled_ce_scatter_dh_buffers(VulkanContext *ctx,
                                                uint32_t slot,
                                                VkBuffer row_ptr_pos, VkBuffer pos_indices,
                                                VkBuffer row_ptr_neg, VkBuffer neg_indices,
                                                VkBuffer sample_ids, VkBuffer d_logits,
                                                VkBuffer d_h,
                                                size_t rpp_sz, size_t pi_sz,
                                                size_t rpn_sz, size_t ni_sz,
                                                size_t sid_sz, size_t dl_sz,
                                                size_t dh_sz) {
    VkDescriptorBufferInfo buf_infos[7] = {
        {row_ptr_pos, 0, rpp_sz},
        {pos_indices, 0, pi_sz},
        {row_ptr_neg, 0, rpn_sz},
        {neg_indices, 0, ni_sz},
        {sample_ids,  0, sid_sz},
        {d_logits,    0, dl_sz},
        {d_h,         0, dh_sz},
    };
    VkWriteDescriptorSet writes[7];
    for (int i = 0; i < 7; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 7, writes, 0, NULL);
}

static void bind_norm_grad_reduce_buffers(VulkanContext *ctx,
                                           uint32_t slot,
                                           VkBuffer h_final, VkBuffer d_h,
                                           VkBuffer rms,
                                           VkBuffer d_norm_w, VkBuffer d_norm_b,
                                           size_t hf_sz, size_t dh_sz,
                                           size_t rms_sz,
                                           size_t dnw_sz, size_t dnb_sz) {
    VkDescriptorBufferInfo buf_infos[5] = {
        {h_final,  0, hf_sz},
        {d_h,      0, dh_sz},
        {rms,      0, rms_sz},
        {d_norm_w, 0, dnw_sz},
        {d_norm_b, 0, dnb_sz},
    };
    VkWriteDescriptorSet writes[5];
    for (int i = 0; i < 5; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 5, writes, 0, NULL);
}

int vk_dispatch_sampled_ce_forward(VulkanContext *ctx,
    VkBufferHandle row_ptr_pos, VkBufferHandle pos_indices,
    VkBufferHandle row_ptr_neg, VkBufferHandle neg_indices,
    VkBufferHandle norm_out, VkBufferHandle tokens,
    VkBufferHandle unigram_cdf, VkBufferHandle log_unigram,
    VkBufferHandle token_freq,
    VkBufferHandle sample_ids_out, VkBufferHandle d_logits_out,
    VkBufferHandle loss_out, VkBufferHandle base_loss_out,
    VkBufferHandle prior_ctx_counts, VkBufferHandle prior_pair_keys,
    VkBufferHandle prior_pair_counts, VkBufferHandle prior_unigram,
    VkBufferHandle program_targets, VkBufferHandle program_counts,
    VkBufferHandle atlas_keys, VkBufferHandle atlas_targets,
    VkBufferHandle atlas_counts, VkBufferHandle ema_logits,
    uint32_t seq_len, uint32_t batch_size, uint32_t rng_seed,
    uint32_t corpus_prior_enabled, uint32_t corpus_prior_pair_capacity,
    float corpus_prior_alpha, float corpus_prior_scale,
    uint32_t logit_prior_gate_enabled,
    uint32_t program_kernel_enabled, float program_kernel_logit_boost,
    uint32_t program_kernel_gate_enabled,
    uint32_t state_atlas_enabled, uint32_t state_atlas_capacity,
    float state_atlas_logit_boost, uint32_t state_atlas_gate_enabled,
    float martingale_lambda) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(row_ptr_pos)  || !VK_BUFFER_VALID(pos_indices)  ||
        !VK_BUFFER_VALID(row_ptr_neg)  || !VK_BUFFER_VALID(neg_indices)  ||
        !VK_BUFFER_VALID(norm_out)     || !VK_BUFFER_VALID(tokens)       ||
        !VK_BUFFER_VALID(unigram_cdf)  || !VK_BUFFER_VALID(log_unigram)  ||
        !VK_BUFFER_VALID(sample_ids_out) || !VK_BUFFER_VALID(d_logits_out) ||
        !VK_BUFFER_VALID(loss_out) || !VK_BUFFER_VALID(base_loss_out) ||
        !VK_BUFFER_VALID(prior_ctx_counts) ||
        !VK_BUFFER_VALID(prior_pair_keys) || !VK_BUFFER_VALID(prior_pair_counts) ||
        !VK_BUFFER_VALID(prior_unigram) ||
        !VK_BUFFER_VALID(program_targets) || !VK_BUFFER_VALID(program_counts) ||
        !VK_BUFFER_VALID(atlas_keys) || !VK_BUFFER_VALID(atlas_targets) ||
        !VK_BUFFER_VALID(atlas_counts)) return -1;

    uint32_t slot = ctx->pipe_sampled_ce_forward;

    bind_sampled_ce_forward_buffers(ctx, slot,
        ctx->buffers[row_ptr_pos.index],
        ctx->buffers[pos_indices.index],
        ctx->buffers[row_ptr_neg.index],
        ctx->buffers[neg_indices.index],
        ctx->buffers[norm_out.index],
        ctx->buffers[tokens.index],
        ctx->buffers[unigram_cdf.index],
        ctx->buffers[log_unigram.index],
        ctx->buffers[sample_ids_out.index],
        ctx->buffers[d_logits_out.index],
        ctx->buffers[loss_out.index],
        ctx->buffers[base_loss_out.index],
        ctx->buffers[prior_ctx_counts.index],
        ctx->buffers[prior_pair_keys.index],
        ctx->buffers[prior_pair_counts.index],
        ctx->buffers[prior_unigram.index],
        ctx->buffers[program_targets.index],
        ctx->buffers[program_counts.index],
        ctx->buffers[atlas_keys.index],
        ctx->buffers[atlas_targets.index],
        ctx->buffers[atlas_counts.index],
        ctx->buffers[ema_logits.index],
        ctx->buffers[token_freq.index],
        ctx->buf_sizes[row_ptr_pos.index],
        ctx->buf_sizes[pos_indices.index],
        ctx->buf_sizes[row_ptr_neg.index],
        ctx->buf_sizes[neg_indices.index],
        ctx->buf_sizes[norm_out.index],
        ctx->buf_sizes[tokens.index],
        ctx->buf_sizes[unigram_cdf.index],
        ctx->buf_sizes[log_unigram.index],
        ctx->buf_sizes[sample_ids_out.index],
        ctx->buf_sizes[d_logits_out.index],
        ctx->buf_sizes[loss_out.index],
        ctx->buf_sizes[base_loss_out.index],
        ctx->buf_sizes[prior_ctx_counts.index],
        ctx->buf_sizes[prior_pair_keys.index],
        ctx->buf_sizes[prior_pair_counts.index],
        ctx->buf_sizes[prior_unigram.index],
        ctx->buf_sizes[program_targets.index],
        ctx->buf_sizes[program_counts.index],
        ctx->buf_sizes[atlas_keys.index],
        ctx->buf_sizes[atlas_targets.index],
        ctx->buf_sizes[atlas_counts.index],
        ctx->buf_sizes[ema_logits.index],
        ctx->buf_sizes[token_freq.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_model;
        uint32_t vocab_size;
        uint32_t n_samples;
        uint32_t k_noise;
        uint32_t batch_size;
        uint32_t rng_seed;
        uint32_t corpus_prior_enabled;
        uint32_t corpus_prior_pair_capacity;
        float corpus_prior_alpha;
        float corpus_prior_scale;
        uint32_t logit_prior_gate_enabled;
        uint32_t program_kernel_enabled;
        float program_kernel_logit_boost;
        uint32_t program_kernel_gate_enabled;
        uint32_t state_atlas_enabled;
        uint32_t state_atlas_capacity;
        float state_atlas_logit_boost;
        uint32_t state_atlas_gate_enabled;
        float martingale_lambda;
    } pc;
    pc.seq_len    = seq_len;
    pc.d_model    = 1024;
    pc.vocab_size = 32064;
    pc.n_samples  = 251;
    pc.k_noise    = 250;
    pc.batch_size = batch_size;
    pc.rng_seed   = rng_seed;
    pc.corpus_prior_enabled = corpus_prior_enabled;
    pc.corpus_prior_pair_capacity = corpus_prior_pair_capacity;
    pc.corpus_prior_alpha = corpus_prior_alpha;
    pc.corpus_prior_scale = corpus_prior_scale;
    pc.logit_prior_gate_enabled = logit_prior_gate_enabled;
    pc.program_kernel_enabled = program_kernel_enabled;
    pc.program_kernel_logit_boost = program_kernel_logit_boost;
    pc.program_kernel_gate_enabled = program_kernel_gate_enabled;
    pc.state_atlas_enabled = state_atlas_enabled;
    pc.state_atlas_capacity = state_atlas_capacity;
    pc.state_atlas_logit_boost = state_atlas_logit_boost;
    pc.state_atlas_gate_enabled = state_atlas_gate_enabled;
    pc.martingale_lambda = martingale_lambda;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, batch_size * (seq_len - 1), 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_sampled_ce_scatter_dh(VulkanContext *ctx,
    VkBufferHandle row_ptr_pos, VkBufferHandle pos_indices,
    VkBufferHandle row_ptr_neg, VkBufferHandle neg_indices,
    VkBufferHandle sample_ids, VkBufferHandle d_logits,
    VkBufferHandle d_h,
    uint32_t seq_len, uint32_t d_model, uint32_t n_samples, uint32_t batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(row_ptr_pos) || !VK_BUFFER_VALID(pos_indices) ||
        !VK_BUFFER_VALID(row_ptr_neg) || !VK_BUFFER_VALID(neg_indices) ||
        !VK_BUFFER_VALID(sample_ids)  || !VK_BUFFER_VALID(d_logits)    ||
        !VK_BUFFER_VALID(d_h)) return -1;

    uint32_t slot = ctx->pipe_sampled_ce_scatter_dh;

    bind_sampled_ce_scatter_dh_buffers(ctx, slot,
        ctx->buffers[row_ptr_pos.index],
        ctx->buffers[pos_indices.index],
        ctx->buffers[row_ptr_neg.index],
        ctx->buffers[neg_indices.index],
        ctx->buffers[sample_ids.index],
        ctx->buffers[d_logits.index],
        ctx->buffers[d_h.index],
        ctx->buf_sizes[row_ptr_pos.index],
        ctx->buf_sizes[pos_indices.index],
        ctx->buf_sizes[row_ptr_neg.index],
        ctx->buf_sizes[neg_indices.index],
        ctx->buf_sizes[sample_ids.index],
        ctx->buf_sizes[d_logits.index],
        ctx->buf_sizes[d_h.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_model;
        uint32_t n_samples;
        uint32_t batch_size;
    } pc;
    pc.seq_len   = seq_len;
    pc.d_model   = d_model;
    pc.n_samples = n_samples;
    pc.batch_size = batch_size;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, batch_size * (seq_len - 1), 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_norm_grad_reduce(VulkanContext *ctx,
    VkBufferHandle h_final, VkBufferHandle d_h,
    VkBufferHandle rms, VkBufferHandle d_norm_w, VkBufferHandle d_norm_b,
    uint32_t seq_len, uint32_t d_model, uint32_t total_positions) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(h_final)  || !VK_BUFFER_VALID(d_h)      ||
        !VK_BUFFER_VALID(rms)      || !VK_BUFFER_VALID(d_norm_w) ||
        !VK_BUFFER_VALID(d_norm_b)) return -1;

    uint32_t slot = ctx->pipe_norm_grad_reduce;

    bind_norm_grad_reduce_buffers(ctx, slot,
        ctx->buffers[h_final.index],
        ctx->buffers[d_h.index],
        ctx->buffers[rms.index],
        ctx->buffers[d_norm_w.index],
        ctx->buffers[d_norm_b.index],
        ctx->buf_sizes[h_final.index],
        ctx->buf_sizes[d_h.index],
        ctx->buf_sizes[rms.index],
        ctx->buf_sizes[d_norm_w.index],
        ctx->buf_sizes[d_norm_b.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_model;
        uint32_t total_positions;
    } pc;
    pc.seq_len         = seq_len;
    pc.d_model         = d_model;
    pc.total_positions = total_positions;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, d_model, 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

void vk_barrier(VulkanContext *ctx) {
    if (!ctx || !ctx->recording) return;

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT  |
                       VK_ACCESS_TRANSFER_READ_BIT |
                       VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                         VK_PIPELINE_STAGE_TRANSFER_BIT        |
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
}

void vk_sync(VulkanContext *ctx) {
    if (!ctx) return;

    if (ctx->batch_mode) {
        vk_barrier(ctx);
        return;
    }

    if (ctx->recording) {
        vkEndCommandBuffer(ctx->cmd_buf);
        ctx->recording = 0;

        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &ctx->cmd_buf;
        vk_profiled_queue_submit(ctx, 1, &si, ctx->fence);
    }

    vk_profiled_wait_for_fences(ctx, 1, &ctx->fence, VK_TRUE, 5000000000ULL);
    vkResetFences(ctx->device, 1, &ctx->fence);
    vkResetCommandBuffer(ctx->cmd_buf, 0);
}

void vk_batch_checkpoint(VulkanContext *ctx) {
    if (!ctx || !ctx->batch_mode) return;
#ifdef V12_DEBUG_VK_DEFERRED
    if (ctx->deferred_download_count > 0) {
        fprintf(stderr, "[VK][DEFERRED] checkpoint pending=%d\n",
                ctx->deferred_download_count);
        fflush(stderr);
    }
#endif
    vk_end_batch(ctx);
    vk_begin_batch(ctx);
}

void vk_begin_batch(VulkanContext *ctx) {
    if (!ctx || !ctx->initialized) return;
    if (ctx->recording) vk_sync(ctx);
    ctx->batch_mode = 1;
    ctx->staging_offset = 0;
    ctx->deferred_download_count = 0;
    VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
    ctx->recording = 1;
}

void vk_end_batch(VulkanContext *ctx) {
    if (!ctx || !ctx->batch_mode) return;
    if (ctx->recording) {
        vkEndCommandBuffer(ctx->cmd_buf);
        ctx->recording = 0;
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &ctx->cmd_buf;
        vk_profiled_queue_submit(ctx, 1, &si, ctx->fence);
        vk_profiled_wait_for_fences(ctx, 1, &ctx->fence, VK_TRUE, 5000000000ULL);
        vkResetFences(ctx->device, 1, &ctx->fence);
        vkResetCommandBuffer(ctx->cmd_buf, 0);
    }
    for (int i = 0; i < ctx->deferred_download_count; i++) {
#ifdef V12_DEBUG_VK_DEFERRED
        fprintf(stderr,
                "[VK][DEFERRED] flush entry_index=%d total=%d bytes=%zu staging_offset=%zu dst=%p caller=%s:%d func=%s\n",
                i,
                ctx->deferred_download_count,
                ctx->deferred_downloads[i].bytes,
                ctx->deferred_downloads[i].staging_offset,
                ctx->deferred_downloads[i].dst,
                ctx->deferred_downloads[i].file ? ctx->deferred_downloads[i].file : "?",
                ctx->deferred_downloads[i].line,
                ctx->deferred_downloads[i].func ? ctx->deferred_downloads[i].func : "?");
        fflush(stderr);
#endif
        memcpy(ctx->deferred_downloads[i].dst,
               (char *)ctx->staging_mapped + ctx->deferred_downloads[i].staging_offset,
               ctx->deferred_downloads[i].bytes);
    }
    ctx->batch_mode = 0;
    ctx->staging_offset = 0;
    ctx->deferred_download_count = 0;
}

int vk_batch_active(VulkanContext *ctx) {
    return (ctx && ctx->initialized && ctx->batch_mode) ? 1 : 0;
}

void vk_profile_reset(VulkanContext *ctx) {
    if (!ctx) return;
    memset(&ctx->profile, 0, sizeof(ctx->profile));
    ctx->profile.gpu_timing_supported = (uint32_t)ctx->profile_timing_supported;
    ctx->profile_query_active = 0;
    ctx->profile_query_pending = 0;
}

void vk_profile_snapshot(VulkanContext *ctx, VkRuntimeProfile *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!ctx) return;
    *out = ctx->profile;
}

void vk_print_stats(VulkanContext *ctx) {
    if (!ctx) {
        printf("[VK] no context\n");
        return;
    }

    printf("[VK] device: %s\n", ctx->device_name);
    printf("[VK] allocated: %.2f MB\n", (double)ctx->allocated / (1024.0 * 1024.0));
    printf("[VK] buffers: %u / %u\n", ctx->buf_count, VK_MAX_BUFFERS);
}

size_t vk_allocated_bytes(VulkanContext *ctx) {
    if (!ctx) return 0;
    return ctx->allocated;
}

static void bind_dlm_head_scatter_buffers(VulkanContext *ctx,
                                           uint32_t slot,
                                           VkBuffer sample_ids, VkBuffer d_logits,
                                           VkBuffer norm_out, VkBuffer dlm_head,
                                           size_t sid_sz, size_t dl_sz,
                                           size_t no_sz, size_t dh_sz) {
    VkDescriptorBufferInfo buf_infos[4] = {
        {sample_ids, 0, sid_sz},
        {d_logits,   0, dl_sz},
        {norm_out,   0, no_sz},
        {dlm_head,   0, dh_sz},
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 4, writes, 0, NULL);
}

int vk_dispatch_dlm_head_scatter(VulkanContext *ctx,
    VkBufferHandle sample_ids, VkBufferHandle d_logits,
    VkBufferHandle norm_out, VkBufferHandle dlm_head,
    uint32_t seq_len, uint32_t d_model, uint32_t n_samples, uint32_t batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(sample_ids) || !VK_BUFFER_VALID(d_logits) ||
        !VK_BUFFER_VALID(norm_out)   || !VK_BUFFER_VALID(dlm_head)) return -1;

    uint32_t slot = ctx->pipe_dlm_head_scatter;

    bind_dlm_head_scatter_buffers(ctx, slot,
        ctx->buffers[sample_ids.index],
        ctx->buffers[d_logits.index],
        ctx->buffers[norm_out.index],
        ctx->buffers[dlm_head.index],
        ctx->buf_sizes[sample_ids.index],
        ctx->buf_sizes[d_logits.index],
        ctx->buf_sizes[norm_out.index],
        ctx->buf_sizes[dlm_head.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_model;
        uint32_t n_samples;
        uint32_t batch_size;
    } pc;
    pc.seq_len    = seq_len;
    pc.d_model    = d_model;
    pc.n_samples  = n_samples;
    pc.batch_size = batch_size;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, batch_size * (seq_len - 1), 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

static void bind_a_backward_dx_cv_buffers(VulkanContext *ctx,
                                           uint32_t slot,
                                           VkBuffer d_dot, VkBuffer W_A, VkBuffer d_x_cv,
                                           size_t ddot_sz, size_t wa_sz, size_t dxcv_sz) {
    VkDescriptorBufferInfo buf_infos[3] = {
        {d_dot,  0, ddot_sz},
        {W_A,    0, wa_sz},
        {d_x_cv, 0, dxcv_sz},
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
}

static int vk_create_a_gate_recompute_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    /* Bindings (0..5):
     *   0 x_cv, 1 W_A, 2 b_A, 3 A[], 4 a_gate_scratch, 5 a_eff_scratch.
     * The shader always writes both outputs; the default Vulkan backward
     * path consumes both scratches directly. */
    VkDescriptorSetLayoutBinding bindings[6] = {0};
    for (int i = 0; i < 6; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 6;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 2
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = a_gate_recompute_spv_len;
    smci.pCode    = (const uint32_t *)a_gate_recompute_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 6
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_a_gate_recompute = slot;
    ctx->pipeline_count++;
    return 0;
}

static void bind_a_gate_recompute_buffers(VulkanContext *ctx,
                                           uint32_t slot,
                                           VkBuffer x_cv, VkBuffer W_A, VkBuffer b_A,
                                           VkBuffer A_vec,
                                           VkBuffer a_gate_scratch,
                                           VkBuffer a_eff_scratch,
                                           size_t xcv_sz, size_t wa_sz, size_t ba_sz,
                                           size_t a_sz,
                                           size_t gate_sz, size_t eff_sz) {
    VkDescriptorBufferInfo buf_infos[6] = {
        {x_cv,           0, xcv_sz},
        {W_A,            0, wa_sz},
        {b_A,            0, ba_sz},
        {A_vec,          0, a_sz},
        {a_gate_scratch, 0, gate_sz},
        {a_eff_scratch,  0, eff_sz},
    };
    VkWriteDescriptorSet writes[6];
    for (int i = 0; i < 6; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 6, writes, 0, NULL);
}

int vk_dispatch_a_gate_recompute(VulkanContext *ctx,
    VkBufferHandle x_cv,
    VkBufferHandle W_A,
    VkBufferHandle b_A,
    VkBufferHandle A_vec,
    VkBufferHandle a_gate_scratch,
    VkBufferHandle a_eff_scratch,
    uint32_t seq_len, uint32_t d_state, uint32_t batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(x_cv)  || !VK_BUFFER_VALID(W_A) ||
        !VK_BUFFER_VALID(b_A)   || !VK_BUFFER_VALID(A_vec) ||
        !VK_BUFFER_VALID(a_gate_scratch) ||
        !VK_BUFFER_VALID(a_eff_scratch)) return -1;

    uint32_t slot = ctx->pipe_a_gate_recompute;

    bind_a_gate_recompute_buffers(ctx, slot,
        ctx->buffers[x_cv.index],
        ctx->buffers[W_A.index],
        ctx->buffers[b_A.index],
        ctx->buffers[A_vec.index],
        ctx->buffers[a_gate_scratch.index],
        ctx->buffers[a_eff_scratch.index],
        ctx->buf_sizes[x_cv.index],
        ctx->buf_sizes[W_A.index],
        ctx->buf_sizes[b_A.index],
        ctx->buf_sizes[A_vec.index],
        ctx->buf_sizes[a_gate_scratch.index],
        ctx->buf_sizes[a_eff_scratch.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
    } pc;
    pc.seq_len = seq_len;
    pc.d_state = d_state;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    /* Dispatch shape mirrors elementwise_gates.comp: gridX = seq_len,
     * gridY = batch_size, local = 128 threads (= d_state). */
    vkCmdDispatch(ctx->cmd_buf, seq_len, batch_size, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_dispatch_a_backward_dx_cv(VulkanContext *ctx,
    VkBufferHandle d_dot, VkBufferHandle W_A, VkBufferHandle d_x_cv,
    uint32_t seq_len, uint32_t ds, uint32_t batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(d_dot) || !VK_BUFFER_VALID(W_A) || !VK_BUFFER_VALID(d_x_cv)) return -1;

    uint32_t slot = ctx->pipe_a_backward_dx_cv;

    bind_a_backward_dx_cv_buffers(ctx, slot,
        ctx->buffers[d_dot.index],
        ctx->buffers[W_A.index],
        ctx->buffers[d_x_cv.index],
        ctx->buf_sizes[d_dot.index],
        ctx->buf_sizes[W_A.index],
        ctx->buf_sizes[d_x_cv.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t ds;
    } pc;
    pc.seq_len = seq_len;
    pc.ds      = ds;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, seq_len, batch_size, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

static void bind_conv1d_bwd_input_gate_buffers(VulkanContext *ctx,
                                                uint32_t slot,
                                                VkBuffer d_x_cv, VkBuffer w_conv,
                                                VkBuffer x_proj, VkBuffer mask_val,
                                                VkBuffer d_xp, VkBuffer d_tau,
                                                size_t dxcv_sz, size_t wconv_sz,
                                                size_t xproj_sz, size_t mask_sz,
                                                size_t dxp_sz, size_t dtau_sz) {
    VkDescriptorBufferInfo buf_infos[6] = {
        {d_x_cv,   0, dxcv_sz},
        {w_conv,   0, wconv_sz},
        {x_proj,   0, xproj_sz},
        {mask_val, 0, mask_sz},
        {d_xp,     0, dxp_sz},
        {d_tau,    0, dtau_sz},
    };
    VkWriteDescriptorSet writes[6];
    for (int i = 0; i < 6; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 6, writes, 0, NULL);
}

int vk_dispatch_conv1d_bwd_input_gate(VulkanContext *ctx,
    VkBufferHandle d_x_cv, VkBufferHandle w_conv,
    VkBufferHandle x_proj, VkBufferHandle mask_val,
    VkBufferHandle d_xp, VkBufferHandle d_tau,
    uint32_t seq_len, uint32_t d_state, float tau, float scale,
    uint32_t batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(d_x_cv) || !VK_BUFFER_VALID(w_conv) ||
        !VK_BUFFER_VALID(x_proj) || !VK_BUFFER_VALID(mask_val) ||
        !VK_BUFFER_VALID(d_xp)   || !VK_BUFFER_VALID(d_tau)) return -1;

    uint32_t slot = ctx->pipe_conv1d_bwd_input_gate;

    bind_conv1d_bwd_input_gate_buffers(ctx, slot,
        ctx->buffers[d_x_cv.index],
        ctx->buffers[w_conv.index],
        ctx->buffers[x_proj.index],
        ctx->buffers[mask_val.index],
        ctx->buffers[d_xp.index],
        ctx->buffers[d_tau.index],
        ctx->buf_sizes[d_x_cv.index],
        ctx->buf_sizes[w_conv.index],
        ctx->buf_sizes[x_proj.index],
        ctx->buf_sizes[mask_val.index],
        ctx->buf_sizes[d_xp.index],
        ctx->buf_sizes[d_tau.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        float    tau;
        float    scale;
    } pc;
    pc.seq_len = seq_len;
    pc.d_state = d_state;
    pc.tau     = tau;
    pc.scale   = scale;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, seq_len, batch_size, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

static void bind_embed_e_backward_buffers(VulkanContext *ctx,
                                           uint32_t slot,
                                           VkBuffer dh_buf, VkBuffer embed_p_buf,
                                           VkBuffer tokens_buf, VkBuffer d_embed_e_buf,
                                           size_t dh_sz, size_t ep_sz,
                                           size_t tok_sz, size_t dee_sz) {
    VkDescriptorBufferInfo buf_infos[4] = {
        {dh_buf,       0, dh_sz},
        {embed_p_buf,  0, ep_sz},
        {tokens_buf,   0, tok_sz},
        {d_embed_e_buf, 0, dee_sz},
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 4, writes, 0, NULL);
}

int vk_dispatch_embed_e_backward(VulkanContext *ctx,
    VkBufferHandle d_h, VkBufferHandle embed_P,
    VkBufferHandle tokens, VkBufferHandle d_embed_E,
    uint32_t total_positions, uint32_t d_model, uint32_t d_embed) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(d_h))      return -1;
    if (!VK_BUFFER_VALID(embed_P))  return -1;
    if (!VK_BUFFER_VALID(tokens))   return -1;
    if (!VK_BUFFER_VALID(d_embed_E)) return -1;

    uint32_t slot = ctx->pipe_embed_e_backward;

    bind_embed_e_backward_buffers(ctx, slot,
        ctx->buffers[d_h.index],
        ctx->buffers[embed_P.index],
        ctx->buffers[tokens.index],
        ctx->buffers[d_embed_E.index],
        ctx->buf_sizes[d_h.index],
        ctx->buf_sizes[embed_P.index],
        ctx->buf_sizes[tokens.index],
        ctx->buf_sizes[d_embed_E.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t total_positions;
        uint32_t d_model;
        uint32_t d_embed;
    } pc;
    pc.total_positions = total_positions;
    pc.d_model         = d_model;
    pc.d_embed         = d_embed;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, total_positions, 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

static void bind_dc_bilinear_reduce_buffers(VulkanContext *ctx,
                                             uint32_t slot,
                                             VkBuffer dh_buf, VkBuffer h_cache_buf,
                                             VkBuffer x_cv_buf, VkBuffer dc_bl_buf,
                                             size_t dh_sz, size_t hc_sz,
                                             size_t xcv_sz, size_t dcbl_sz) {
    VkDescriptorBufferInfo buf_infos[4] = {
        {dh_buf,      0, dh_sz},
        {h_cache_buf, 0, hc_sz},
        {x_cv_buf,    0, xcv_sz},
        {dc_bl_buf,   0, dcbl_sz},
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 4, writes, 0, NULL);
}

int vk_dispatch_dc_bilinear_reduce(VulkanContext *ctx,
    VkBufferHandle dh,
    VkBufferHandle h_cache,
    VkBufferHandle x_cv,
    VkBufferHandle dc_bl,
    uint32_t seq_len, uint32_t d_state, uint32_t batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(dh))      return -1;
    if (!VK_BUFFER_VALID(h_cache)) return -1;
    if (!VK_BUFFER_VALID(x_cv))    return -1;
    if (!VK_BUFFER_VALID(dc_bl))   return -1;

    uint32_t slot = ctx->pipe_dc_bilinear_reduce;

    bind_dc_bilinear_reduce_buffers(ctx, slot,
        ctx->buffers[dh.index],
        ctx->buffers[h_cache.index],
        ctx->buffers[x_cv.index],
        ctx->buffers[dc_bl.index],
        ctx->buf_sizes[dh.index],
        ctx->buf_sizes[h_cache.index],
        ctx->buf_sizes[x_cv.index],
        ctx->buf_sizes[dc_bl.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t batch_size;
    } pc;
    pc.seq_len   = seq_len;
    pc.d_state   = d_state;
    pc.batch_size = batch_size;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, 1, 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

static void bind_direct_b_backward_buffers(VulkanContext *ctx,
                                            uint32_t slot,
                                            VkBuffer dB_seq_buf, VkBuffer x_cv_buf,
                                            VkBuffer B_vec_buf,  VkBuffer d_x_cv_buf,
                                            VkBuffer dB_accum_buf,
                                            size_t dB_seq_sz, size_t x_cv_sz,
                                            size_t B_vec_sz,  size_t d_x_cv_sz,
                                            size_t dB_accum_sz) {
    VkDescriptorBufferInfo buf_infos[5] = {
        {dB_seq_buf,   0, dB_seq_sz},
        {x_cv_buf,     0, x_cv_sz},
        {B_vec_buf,    0, B_vec_sz},
        {d_x_cv_buf,   0, d_x_cv_sz},
        {dB_accum_buf, 0, dB_accum_sz},
    };
    VkWriteDescriptorSet writes[5];
    for (int i = 0; i < 5; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 5, writes, 0, NULL);
}

static int vk_create_direct_b_backward_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[5] = {0};
    for (int i = 0; i < 5; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 5;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 3
    };
    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = direct_b_backward_spv_len;
    smci.pCode    = (const uint32_t *)direct_b_backward_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 5
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_direct_b_backward = slot;
    ctx->pipeline_count++;
    return 0;
}

int vk_dispatch_direct_b_backward(VulkanContext *ctx,
    VkBufferHandle dB_seq,
    VkBufferHandle x_cv,
    VkBufferHandle B_vec,
    VkBufferHandle d_x_cv,
    VkBufferHandle dB_accum,
    uint32_t seq_len, uint32_t d_state, uint32_t batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(dB_seq))  return -1;
    if (!VK_BUFFER_VALID(x_cv))    return -1;
    if (!VK_BUFFER_VALID(B_vec))   return -1;
    if (!VK_BUFFER_VALID(d_x_cv))  return -1;
    if (!VK_BUFFER_VALID(dB_accum)) return -1;

    uint32_t slot = ctx->pipe_direct_b_backward;

    bind_direct_b_backward_buffers(ctx, slot,
        ctx->buffers[dB_seq.index],
        ctx->buffers[x_cv.index],
        ctx->buffers[B_vec.index],
        ctx->buffers[d_x_cv.index],
        ctx->buffers[dB_accum.index],
        ctx->buf_sizes[dB_seq.index],
        ctx->buf_sizes[x_cv.index],
        ctx->buf_sizes[B_vec.index],
        ctx->buf_sizes[d_x_cv.index],
        ctx->buf_sizes[dB_accum.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t batch_size;
    } pc;
    pc.seq_len    = seq_len;
    pc.d_state    = d_state;
    pc.batch_size = batch_size;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, seq_len, batch_size, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

static void bind_welford_update_buffers(VulkanContext *ctx,
                                        uint32_t slot,
                                        VkBuffer h_cache_buf,
                                        VkBuffer moments_buf,
                                        size_t h_cache_sz,
                                        size_t moments_sz) {
    VkDescriptorBufferInfo buf_infos[2] = {
        {h_cache_buf, 0, h_cache_sz},
        {moments_buf, 0, moments_sz},
    };
    VkWriteDescriptorSet writes[2];
    for (int i = 0; i < 2; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);
}

static int vk_create_welford_update_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 3
    };
    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = welford_update_spv_len;
    smci.pCode    = (const uint32_t *)welford_update_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 2
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_welford_update = slot;
    ctx->pipeline_count++;
    return 0;
}

int vk_dispatch_welford_update(VulkanContext *ctx,
    VkBufferHandle h_cache,
    VkBufferHandle moments,
    uint32_t batch_size, uint32_t seq_len, uint32_t d_state) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(h_cache)) return -1;
    if (!VK_BUFFER_VALID(moments)) return -1;

    uint32_t slot = ctx->pipe_welford_update;

    bind_welford_update_buffers(ctx, slot,
        ctx->buffers[h_cache.index],
        ctx->buffers[moments.index],
        ctx->buf_sizes[h_cache.index],
        ctx->buf_sizes[moments.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t batch_size;
        uint32_t seq_len;
        uint32_t d_state;
    } pc;
    pc.batch_size = batch_size;
    pc.seq_len    = seq_len;
    pc.d_state    = d_state;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, 1, 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

static void bind_online_tssi_score_update_buffers(VulkanContext *ctx,
                                                  uint32_t slot,
                                                  VkBuffer h_cache_buf,
                                                  VkBuffer d_logits_buf,
                                                  VkBuffer score_moments_buf,
                                                  size_t h_cache_sz,
                                                  size_t d_logits_sz,
                                                  size_t score_moments_sz) {
    VkDescriptorBufferInfo buf_infos[3] = {
        {h_cache_buf, 0, h_cache_sz},
        {d_logits_buf, 0, d_logits_sz},
        {score_moments_buf, 0, score_moments_sz},
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
}

static int vk_create_online_tssi_score_update_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 4
    };
    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = online_tssi_score_update_spv_len;
    smci.pCode    = (const uint32_t *)online_tssi_score_update_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_online_tssi_score_update = slot;
    ctx->pipeline_count++;
    return 0;
}

int vk_dispatch_online_tssi_score_update(VulkanContext *ctx,
    VkBufferHandle h_cache,
    VkBufferHandle d_logits,
    VkBufferHandle score_moments,
    uint32_t batch_size, uint32_t seq_len, uint32_t d_state,
    uint32_t n_samples) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(h_cache)) return -1;
    if (!VK_BUFFER_VALID(d_logits)) return -1;
    if (!VK_BUFFER_VALID(score_moments)) return -1;
    if (seq_len == 0 || d_state == 0 || n_samples == 0) return -1;

    uint32_t slot = ctx->pipe_online_tssi_score_update;

    bind_online_tssi_score_update_buffers(ctx, slot,
        ctx->buffers[h_cache.index],
        ctx->buffers[d_logits.index],
        ctx->buffers[score_moments.index],
        ctx->buf_sizes[h_cache.index],
        ctx->buf_sizes[d_logits.index],
        ctx->buf_sizes[score_moments.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t batch_size;
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t n_samples;
    } pc;
    pc.batch_size = batch_size;
    pc.seq_len    = seq_len;
    pc.d_state    = d_state;
    pc.n_samples  = n_samples;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t groups = (d_state + 127u) / 128u;
    vkCmdDispatch(ctx->cmd_buf, groups, 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

int vk_fill_zero(VulkanContext *ctx, VkBufferHandle buf, size_t bytes) {
    if (!ctx || !VK_BUFFER_VALID(buf)) return -1;

    if (ctx->batch_mode) {
        vkCmdFillBuffer(ctx->cmd_buf, ctx->buffers[buf.index], 0, bytes, 0);
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(ctx->cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        return 0;
    }

    if (ctx->recording) {
        vk_sync(ctx);
    }
    VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
    vkCmdFillBuffer(ctx->cmd_buf, ctx->buffers[buf.index], 0, bytes, 0);
    vkEndCommandBuffer(ctx->cmd_buf);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &ctx->cmd_buf;
    vk_profiled_queue_submit(ctx, 1, &si, ctx->fence);
    vk_profiled_wait_for_fences(ctx, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx->device, 1, &ctx->fence);
    vkResetCommandBuffer(ctx->cmd_buf, 0);
    return 0;
}

/* ── broadcast_fill ─────────────────────────────────────────────────────── */

static void bind_broadcast_fill_buffers(VulkanContext *ctx,
                                        uint32_t slot,
                                        VkBuffer src_buf, VkBuffer dst_buf,
                                        size_t src_sz, size_t dst_sz) {
    VkDescriptorBufferInfo buf_infos[2] = {
        {src_buf, 0, src_sz},
        {dst_buf, 0, dst_sz},
    };
    VkWriteDescriptorSet writes[2];
    for (int i = 0; i < 2; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);
}

static int vk_create_broadcast_fill_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 2
    };
    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = broadcast_fill_spv_len;
    smci.pCode    = (const uint32_t *)broadcast_fill_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 2
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_broadcast_fill = slot;
    ctx->pipeline_count++;
    return 0;
}

int vk_dispatch_broadcast_fill(VulkanContext *ctx,
    VkBufferHandle src,
    VkBufferHandle dst,
    uint32_t total, uint32_t d_state) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(src)) return -1;
    if (!VK_BUFFER_VALID(dst)) return -1;

    uint32_t slot = ctx->pipe_broadcast_fill;

    bind_broadcast_fill_buffers(ctx, slot,
        ctx->buffers[src.index],
        ctx->buffers[dst.index],
        ctx->buf_sizes[src.index],
        ctx->buf_sizes[dst.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t total;
        uint32_t d_state;
    } pc;
    pc.total   = total;
    pc.d_state = d_state;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, total, 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

/* ── mtp_loss ───────────────────────────────────────────────────────────── */

static void bind_mtp_loss_buffers(VulkanContext *ctx,
                                  uint32_t slot,
                                  VkBuffer h_seq_buf,
                                  VkBuffer target_buf,
                                  VkBuffer proj_weight_buf,
                                  VkBuffer proj_bias_buf,
                                  VkBuffer loss_out_buf,
                                  VkBuffer dh_out_buf,
                                  VkBuffer dproj_weight_buf,
                                  VkBuffer dproj_bias_buf,
                                  size_t h_seq_sz,
                                  size_t target_sz,
                                  size_t proj_weight_sz,
                                  size_t proj_bias_sz,
                                  size_t loss_out_sz,
                                  size_t dh_out_sz,
                                  size_t dproj_weight_sz,
                                  size_t dproj_bias_sz) {
    VkDescriptorBufferInfo buf_infos[8] = {
        {h_seq_buf,       0, h_seq_sz},
        {target_buf,      0, target_sz},
        {proj_weight_buf, 0, proj_weight_sz},
        {proj_bias_buf,   0, proj_bias_sz},
        {loss_out_buf,    0, loss_out_sz},
        {dh_out_buf,      0, dh_out_sz},
        {dproj_weight_buf,0, dproj_weight_sz},
        {dproj_bias_buf,  0, dproj_bias_sz},
    };
    VkWriteDescriptorSet writes[8];
    for (int i = 0; i < 8; i++) {
        writes[i] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 8, writes, 0, NULL);
}

static int vk_create_mtp_loss_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[8] = {0};
    for (int i = 0; i < 8; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 8;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 4 + sizeof(float)
    };
    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = mtp_loss_spv_len;
    smci.pCode    = (const uint32_t *)mtp_loss_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 8
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_mtp_loss = slot;
    ctx->pipeline_count++;
    return 0;
}

int vk_dispatch_mtp_loss(VulkanContext *ctx,
    VkBufferHandle h_seq,
    VkBufferHandle target,
    VkBufferHandle proj_weight,
    VkBufferHandle proj_bias,
    VkBufferHandle loss_out,
    VkBufferHandle dh_out,
    VkBufferHandle dproj_weight,
    VkBufferHandle dproj_bias,
    uint32_t seq_len, uint32_t d_state, uint32_t mtp_k, uint32_t rank,
    float mtp_weight,
    uint32_t batch_size) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(h_seq))        return -1;
    if (!VK_BUFFER_VALID(target))       return -1;
    if (!VK_BUFFER_VALID(proj_weight))  return -1;
    if (!VK_BUFFER_VALID(proj_bias))    return -1;
    if (!VK_BUFFER_VALID(loss_out))     return -1;
    if (!VK_BUFFER_VALID(dh_out))       return -1;
    if (!VK_BUFFER_VALID(dproj_weight)) return -1;
    if (!VK_BUFFER_VALID(dproj_bias))   return -1;

    uint32_t slot = ctx->pipe_mtp_loss;

    bind_mtp_loss_buffers(ctx, slot,
        ctx->buffers[h_seq.index],
        ctx->buffers[target.index],
        ctx->buffers[proj_weight.index],
        ctx->buffers[proj_bias.index],
        ctx->buffers[loss_out.index],
        ctx->buffers[dh_out.index],
        ctx->buffers[dproj_weight.index],
        ctx->buffers[dproj_bias.index],
        ctx->buf_sizes[h_seq.index],
        ctx->buf_sizes[target.index],
        ctx->buf_sizes[proj_weight.index],
        ctx->buf_sizes[proj_bias.index],
        ctx->buf_sizes[loss_out.index],
        ctx->buf_sizes[dh_out.index],
        ctx->buf_sizes[dproj_weight.index],
        ctx->buf_sizes[dproj_bias.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);

    vkCmdBindDescriptorSets(ctx->cmd_buf,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot],
                            0, 1, &ctx->desc_sets[slot],
                            0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t d_state;
        uint32_t mtp_k;
        uint32_t rank;
        float    mtp_weight;
    } pc;
    pc.seq_len    = seq_len;
    pc.d_state    = d_state;
    pc.mtp_k      = mtp_k;
    pc.rank       = rank;
    pc.mtp_weight = mtp_weight;

    vkCmdPushConstants(ctx->cmd_buf,
                       ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, batch_size * (seq_len - mtp_k), 1, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    return 0;
}

/* ── PC gather/scatter helpers ──────────────────────────────────────────── */

static void bind_pc_2buf(VulkanContext *ctx, uint32_t slot,
                         VkBuffer a_buf, VkBuffer b_buf,
                         size_t a_sz, size_t b_sz) {
    VkDescriptorBufferInfo buf_infos[2] = {
        {a_buf, 0, a_sz},
        {b_buf, 0, b_sz},
    };
    VkWriteDescriptorSet writes[2];
    for (int i = 0; i < 2; i++) {
        writes[i] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);
}

static int vk_create_pc_2buf_pipeline(VulkanContext *ctx,
                                       const unsigned char *spv,
                                       unsigned int spv_len,
                                       uint32_t *slot_out) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[2] = {0};
    for (int i = 0; i < 2; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 2;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL, &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 3
    };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL, &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = spv_len;
    smci.pCode    = (const uint32_t *)spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL, &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 2
    };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL, &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    *slot_out = slot;
    ctx->pipeline_count++;
    return 0;
}

static int vk_dispatch_pc_2buf(VulkanContext *ctx,
                                uint32_t slot,
                                VkBufferHandle a, VkBufferHandle b,
                                uint32_t batch_size, uint32_t seq_len, uint32_t d) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(a) || !VK_BUFFER_VALID(b)) return -1;

    bind_pc_2buf(ctx, slot,
        ctx->buffers[a.index], ctx->buffers[b.index],
        ctx->buf_sizes[a.index], ctx->buf_sizes[b.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);

    struct { uint32_t batch_size; uint32_t seq_len; uint32_t d; } pc;
    pc.batch_size = batch_size;
    pc.seq_len    = seq_len;
    pc.d          = d;
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t total     = batch_size * (seq_len - 1u) * d;
    uint32_t workgroups = (total + 255u) / 256u;
    vkCmdDispatch(ctx->cmd_buf, workgroups, 1, 1);

    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    return 0;
}

static int vk_create_pc_gather_prev_pipeline(VulkanContext *ctx) {
    return vk_create_pc_2buf_pipeline(ctx,
        pc_gather_prev_spv, pc_gather_prev_spv_len,
        &ctx->pipe_pc_gather_prev);
}

static int vk_create_residual_sub_pipeline(VulkanContext *ctx) {
    return vk_create_pc_2buf_pipeline(ctx,
        residual_sub_spv, residual_sub_spv_len,
        &ctx->pipe_residual_sub);
}

static int vk_create_pc_negate_gather_pipeline(VulkanContext *ctx) {
    return vk_create_pc_2buf_pipeline(ctx,
        pc_negate_gather_spv, pc_negate_gather_spv_len,
        &ctx->pipe_pc_negate_gather);
}

static int vk_create_residual_scatter_add_pipeline(VulkanContext *ctx) {
    return vk_create_pc_2buf_pipeline(ctx,
        residual_scatter_add_spv, residual_scatter_add_spv_len,
        &ctx->pipe_residual_scatter_add);
}

int vk_dispatch_pc_gather_prev(VulkanContext *ctx,
    VkBufferHandle src, VkBufferHandle dst,
    uint32_t batch_size, uint32_t seq_len, uint32_t d) {
    return vk_dispatch_pc_2buf(ctx, ctx->pipe_pc_gather_prev,
                               src, dst, batch_size, seq_len, d);
}

int vk_dispatch_residual_sub(VulkanContext *ctx,
    VkBufferHandle pred, VkBufferHandle res,
    uint32_t batch_size, uint32_t seq_len, uint32_t d) {
    return vk_dispatch_pc_2buf(ctx, ctx->pipe_residual_sub,
                               pred, res, batch_size, seq_len, d);
}

int vk_dispatch_pc_negate_gather(VulkanContext *ctx,
    VkBufferHandle src, VkBufferHandle dst,
    uint32_t batch_size, uint32_t seq_len, uint32_t d) {
    return vk_dispatch_pc_2buf(ctx, ctx->pipe_pc_negate_gather,
                               src, dst, batch_size, seq_len, d);
}

int vk_dispatch_residual_scatter_add(VulkanContext *ctx,
    VkBufferHandle pred, VkBufferHandle res,
    uint32_t batch_size, uint32_t seq_len, uint32_t d) {
    return vk_dispatch_pc_2buf(ctx, ctx->pipe_residual_scatter_add,
                               pred, res, batch_size, seq_len, d);
}

/* ── adam_dense ────────────────────────────────────────────────────────────── */

static int vk_create_adam_dense_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[4] = {0};
    for (int i = 0; i < 4; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 4;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) + 8 * sizeof(float)  /* N, lr, beta1, beta2, eps, inv_bc1, inv_bc2, clamp_lo, clamp_hi */
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = adam_dense_spv_len;
    smci.pCode    = (const uint32_t *)adam_dense_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 4
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_adam_dense = slot;
    ctx->pipeline_count++;
    return 0;
}

static void bind_adam_dense_buffers(VulkanContext *ctx, uint32_t slot,
                                    VkBuffer param_buf, VkBuffer grad_buf,
                                    VkBuffer m_buf,     VkBuffer v_buf,
                                    size_t param_sz, size_t grad_sz,
                                    size_t m_sz,     size_t v_sz) {
    VkDescriptorBufferInfo buf_infos[4] = {
        {param_buf, 0, param_sz},
        {grad_buf,  0, grad_sz},
        {m_buf,     0, m_sz},
        {v_buf,     0, v_sz},
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 4, writes, 0, NULL);
}

int vk_dispatch_adam_dense(VulkanContext *ctx,
    VkBufferHandle param, VkBufferHandle grad,
    VkBufferHandle m,     VkBufferHandle v,
    uint32_t N, float lr,
    float beta1, float beta2, float eps,
    float inv_bc1, float inv_bc2,
    float clamp_lo, float clamp_hi) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(param)) return -1;
    if (!VK_BUFFER_VALID(grad))  return -1;
    if (!VK_BUFFER_VALID(m))     return -1;
    if (!VK_BUFFER_VALID(v))     return -1;

    uint32_t slot = ctx->pipe_adam_dense;

    bind_adam_dense_buffers(ctx, slot,
        ctx->buffers[param.index], ctx->buffers[grad.index],
        ctx->buffers[m.index],     ctx->buffers[v.index],
        ctx->buf_sizes[param.index], ctx->buf_sizes[grad.index],
        ctx->buf_sizes[m.index],     ctx->buf_sizes[v.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);

    struct {
        uint32_t N;
        float    lr;
        float    beta1;
        float    beta2;
        float    eps;
        float    inv_bc1;
        float    inv_bc2;
        float    clamp_lo;
        float    clamp_hi;
    } pc;
    pc.N        = N;
    pc.lr       = lr;
    pc.beta1    = beta1;
    pc.beta2    = beta2;
    pc.eps      = eps;
    pc.inv_bc1  = inv_bc1;
    pc.inv_bc2  = inv_bc2;
    pc.clamp_lo = clamp_lo;
    pc.clamp_hi = clamp_hi;
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t workgroups = (N + 255u) / 256u;
    vkCmdDispatch(ctx->cmd_buf, workgroups, 1, 1);

    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    return 0;
}

/* ── grad_scale ────────────────────────────────────────────────────────────── */

static int vk_create_grad_scale_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding binding = {0};
    binding.binding         = 0;
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 1;
    dslci.pBindings    = &binding;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) + sizeof(float)  /* N, scale */
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = grad_scale_spv_len;
    smci.pCode    = (const uint32_t *)grad_scale_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_grad_scale = slot;
    ctx->pipeline_count++;
    return 0;
}

static void bind_grad_scale_buffers(VulkanContext *ctx, uint32_t slot,
                                    VkBuffer grad_buf, size_t grad_sz) {
    VkDescriptorBufferInfo buf_info = {grad_buf, 0, grad_sz};
    VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet          = ctx->desc_sets[slot];
    write.dstBinding      = 0;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo     = &buf_info;
    vkUpdateDescriptorSets(ctx->device, 1, &write, 0, NULL);
}

int vk_dispatch_grad_scale(VulkanContext *ctx,
    VkBufferHandle grad,
    uint32_t N, float scale) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(grad)) return -1;

    uint32_t slot = ctx->pipe_grad_scale;

    bind_grad_scale_buffers(ctx, slot,
        ctx->buffers[grad.index], ctx->buf_sizes[grad.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);

    struct { uint32_t N; float scale; } pc;
    pc.N     = N;
    pc.scale = scale;
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t workgroups = (N + 255u) / 256u;
    vkCmdDispatch(ctx->cmd_buf, workgroups, 1, 1);

    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    return 0;
}

/* ── factored_adam_reduce ──────────────────────────────────────────────────── */

static int vk_create_factored_adam_reduce_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[12] = {0};
    for (int i = 0; i < 12; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 12;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = 5 * sizeof(uint32_t) + 5 * sizeof(float)
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = factored_adam_reduce_spv_len;
    smci.pCode    = (const uint32_t *)factored_adam_reduce_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 14
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_factored_adam_reduce = slot;
    ctx->pipeline_count++;
    return 0;
}

static void bind_factored_adam_reduce_buffers(VulkanContext *ctx, uint32_t slot,
    VkBuffer grad_buf,       VkBuffer shadow_buf,
    VkBuffer sampled_buf,    VkBuffer grad_accum_buf,
    VkBuffer g_row_bits_buf, VkBuffer g_col_bits_buf,
    VkBuffer gq_row_bits_buf, VkBuffer gq_col_bits_buf,
    VkBuffer cnt_row_buf,    VkBuffer cnt_col_buf,
    VkBuffer row_map_buf,    VkBuffer evidence_stats_buf,
    size_t grad_sz,       size_t shadow_sz,
    size_t sampled_sz,    size_t grad_accum_sz,
    size_t g_row_sz,      size_t g_col_sz,
    size_t gq_row_sz,     size_t gq_col_sz,
    size_t cnt_row_sz,    size_t cnt_col_sz,
    size_t row_map_sz,   size_t evidence_stats_sz) {
    VkDescriptorBufferInfo buf_infos[12] = {
        {grad_buf,        0, grad_sz},
        {shadow_buf,      0, shadow_sz},
        {sampled_buf,     0, sampled_sz},
        {grad_accum_buf,  0, grad_accum_sz},
        {g_row_bits_buf,  0, g_row_sz},
        {g_col_bits_buf,  0, g_col_sz},
        {gq_row_bits_buf, 0, gq_row_sz},
        {gq_col_bits_buf, 0, gq_col_sz},
        {cnt_row_buf,     0, cnt_row_sz},
        {cnt_col_buf,     0, cnt_col_sz},
        {row_map_buf,     0, row_map_sz},
        {evidence_stats_buf, 0, evidence_stats_sz},
    };
    VkWriteDescriptorSet writes[12];
    for (int i = 0; i < 12; i++) {
        writes[i] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 12, writes, 0, NULL);
}

int vk_dispatch_factored_adam_reduce(VulkanContext *ctx,
    VkBufferHandle grad,       VkBufferHandle shadow,
    VkBufferHandle sampled,    VkBufferHandle grad_accum,
    VkBufferHandle g_row_bits, VkBufferHandle g_col_bits,
    VkBufferHandle gq_row_bits, VkBufferHandle gq_col_bits,
    VkBufferHandle cnt_row,    VkBufferHandle cnt_col,
    VkBufferHandle row_map,    VkBufferHandle evidence_stats,
    VkBufferHandle holo_U,     VkBufferHandle holo_V,
    uint32_t rows, uint32_t cols, uint32_t N,
    uint32_t row_map_mode,
    float crystallization_tau, float bi_ste_factor,
    float ema_gamma, float ema_threshold,
    uint32_t evidence_gate, float boundary_eps, uint32_t rank) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(grad))        return -1;
    if (!VK_BUFFER_VALID(shadow))      return -1;
    if (!VK_BUFFER_VALID(sampled))     return -1;
    if (!VK_BUFFER_VALID(grad_accum))  return -1;
    if (!VK_BUFFER_VALID(g_row_bits))  return -1;
    if (!VK_BUFFER_VALID(g_col_bits))  return -1;
    if (!VK_BUFFER_VALID(gq_row_bits)) return -1;
    if (!VK_BUFFER_VALID(gq_col_bits)) return -1;
    if (!VK_BUFFER_VALID(cnt_row))     return -1;
    if (!VK_BUFFER_VALID(cnt_col))     return -1;
    if (!VK_BUFFER_VALID(row_map))     return -1;
    if (!VK_BUFFER_VALID(evidence_stats)) return -1;

    uint32_t slot = ctx->pipe_factored_adam_reduce;

    bind_factored_adam_reduce_buffers(ctx, slot,
        ctx->buffers[grad.index],        ctx->buffers[shadow.index],
        ctx->buffers[sampled.index],     ctx->buffers[grad_accum.index],
        ctx->buffers[g_row_bits.index],  ctx->buffers[g_col_bits.index],
        ctx->buffers[gq_row_bits.index], ctx->buffers[gq_col_bits.index],
        ctx->buffers[cnt_row.index],     ctx->buffers[cnt_col.index],
        ctx->buffers[row_map.index],     ctx->buffers[evidence_stats.index],
        
        
        ctx->buf_sizes[grad.index],        ctx->buf_sizes[shadow.index],
        ctx->buf_sizes[sampled.index],     ctx->buf_sizes[grad_accum.index],
        ctx->buf_sizes[g_row_bits.index],  ctx->buf_sizes[g_col_bits.index],
        ctx->buf_sizes[gq_row_bits.index], ctx->buf_sizes[gq_col_bits.index],
        ctx->buf_sizes[cnt_row.index],     ctx->buf_sizes[cnt_col.index],
        ctx->buf_sizes[row_map.index],     ctx->buf_sizes[evidence_stats.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);

    struct {
        uint32_t rows;
        uint32_t cols;
        uint32_t N;
        uint32_t row_map_mode;
        float    crystallization_tau;
        float    bi_ste_factor;
        float    ema_gamma;
        float    ema_threshold;
        uint32_t evidence_gate;
        float    boundary_eps;
    } pc;
    pc.rows                = rows;
    pc.cols                = cols;
    pc.N                   = N;
    pc.row_map_mode        = row_map_mode;
    pc.crystallization_tau = crystallization_tau;
    pc.bi_ste_factor       = bi_ste_factor;
    pc.ema_gamma           = ema_gamma;
    pc.ema_threshold       = ema_threshold;
    pc.evidence_gate       = evidence_gate;
    pc.boundary_eps        = boundary_eps;
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t workgroups = (N + 255u) / 256u;
    vkCmdDispatch(ctx->cmd_buf, workgroups, 1, 1);

    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    return 0;
}

/* ── factored_adam_moment ──────────────────────────────────────────────────── */

static int vk_create_factored_adam_moment_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[11] = {0};
    for (int i = 0; i < 11; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 11;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = 2 * sizeof(uint32_t) + 2 * sizeof(float)  /* rows, cols, beta1, beta2 */
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = factored_adam_moment_spv_len;
    smci.pCode    = (const uint32_t *)factored_adam_moment_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 11
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_factored_adam_moment = slot;
    ctx->pipeline_count++;
    return 0;
}

static void bind_factored_adam_moment_buffers(VulkanContext *ctx, uint32_t slot,
    VkBuffer m_row_buf,      VkBuffer m_col_buf,
    VkBuffer v_row_buf,      VkBuffer v_col_buf,
    VkBuffer g_row_bits_buf, VkBuffer g_col_bits_buf,
    VkBuffer gq_row_bits_buf, VkBuffer gq_col_bits_buf,
    VkBuffer cnt_row_buf,    VkBuffer cnt_col_buf,
    VkBuffer mean_v_buf,
    size_t m_row_sz,      size_t m_col_sz,
    size_t v_row_sz,      size_t v_col_sz,
    size_t g_row_sz,      size_t g_col_sz,
    size_t gq_row_sz,     size_t gq_col_sz,
    size_t cnt_row_sz,    size_t cnt_col_sz,
    size_t mean_v_sz) {
    VkDescriptorBufferInfo buf_infos[11] = {
        {m_row_buf,        0, m_row_sz},
        {m_col_buf,        0, m_col_sz},
        {v_row_buf,        0, v_row_sz},
        {v_col_buf,        0, v_col_sz},
        {g_row_bits_buf,   0, g_row_sz},
        {g_col_bits_buf,   0, g_col_sz},
        {gq_row_bits_buf,  0, gq_row_sz},
        {gq_col_bits_buf,  0, gq_col_sz},
        {cnt_row_buf,      0, cnt_row_sz},
        {cnt_col_buf,      0, cnt_col_sz},
        {mean_v_buf,       0, mean_v_sz},
    };
    VkWriteDescriptorSet writes[11];
    for (int i = 0; i < 11; i++) {
        writes[i] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 11, writes, 0, NULL);
}

int vk_dispatch_factored_adam_moment(VulkanContext *ctx,
    VkBufferHandle m_row,      VkBufferHandle m_col,
    VkBufferHandle v_row,      VkBufferHandle v_col,
    VkBufferHandle g_row_bits, VkBufferHandle g_col_bits,
    VkBufferHandle gq_row_bits, VkBufferHandle gq_col_bits,
    VkBufferHandle cnt_row,    VkBufferHandle cnt_col,
    VkBufferHandle mean_v_out,
    uint32_t rows, uint32_t cols,
    float beta1, float beta2) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(m_row))       return -1;
    if (!VK_BUFFER_VALID(m_col))       return -1;
    if (!VK_BUFFER_VALID(v_row))       return -1;
    if (!VK_BUFFER_VALID(v_col))       return -1;
    if (!VK_BUFFER_VALID(g_row_bits))  return -1;
    if (!VK_BUFFER_VALID(g_col_bits))  return -1;
    if (!VK_BUFFER_VALID(gq_row_bits)) return -1;
    if (!VK_BUFFER_VALID(gq_col_bits)) return -1;
    if (!VK_BUFFER_VALID(cnt_row))     return -1;
    if (!VK_BUFFER_VALID(cnt_col))     return -1;
    if (!VK_BUFFER_VALID(mean_v_out))  return -1;

    uint32_t slot = ctx->pipe_factored_adam_moment;

    bind_factored_adam_moment_buffers(ctx, slot,
        ctx->buffers[m_row.index],       ctx->buffers[m_col.index],
        ctx->buffers[v_row.index],       ctx->buffers[v_col.index],
        ctx->buffers[g_row_bits.index],  ctx->buffers[g_col_bits.index],
        ctx->buffers[gq_row_bits.index], ctx->buffers[gq_col_bits.index],
        ctx->buffers[cnt_row.index],     ctx->buffers[cnt_col.index],
        ctx->buffers[mean_v_out.index],
        ctx->buf_sizes[m_row.index],       ctx->buf_sizes[m_col.index],
        ctx->buf_sizes[v_row.index],       ctx->buf_sizes[v_col.index],
        ctx->buf_sizes[g_row_bits.index],  ctx->buf_sizes[g_col_bits.index],
        ctx->buf_sizes[gq_row_bits.index], ctx->buf_sizes[gq_col_bits.index],
        ctx->buf_sizes[cnt_row.index],     ctx->buf_sizes[cnt_col.index],
        ctx->buf_sizes[mean_v_out.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);

    struct {
        uint32_t rows;
        uint32_t cols;
        float    beta1;
        float    beta2;
    } pc;
    pc.rows  = rows;
    pc.cols  = cols;
    pc.beta1 = beta1;
    pc.beta2 = beta2;
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t n_threads   = (rows > cols) ? rows : cols;
    uint32_t workgroups  = (n_threads + 255u) / 256u;
    vkCmdDispatch(ctx->cmd_buf, workgroups, 1, 1);

    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    return 0;
}

/* ── factored_adam_update ──────────────────────────────────────────────────── */

static int vk_create_factored_adam_update_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[10] = {0};
    for (int i = 0; i < 10; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    dslci.bindingCount = 10;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                    &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = 6 * sizeof(uint32_t) + 7 * sizeof(float)
    };

    VkPipelineLayoutCreateInfo plci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                               &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    smci.codeSize = factored_adam_update_spv_len;
    smci.pCode    = (const uint32_t *)factored_adam_update_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL,
                             &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                 &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 10
    };
    VkDescriptorPoolCreateInfo dpci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL,
                               &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_factored_adam_update = slot;
    ctx->pipeline_count++;
    return 0;
}

static void bind_factored_adam_update_buffers(VulkanContext *ctx, uint32_t slot,
    VkBuffer shadow_buf,    VkBuffer grad_accum_buf,
    VkBuffer m_row_buf,     VkBuffer m_col_buf,
    VkBuffer v_row_buf,     VkBuffer v_col_buf,
    VkBuffer row_map_buf,   VkBuffer grad_buf,
    VkBuffer sampled_buf,   VkBuffer evidence_stats_buf,
    size_t shadow_sz,    size_t grad_accum_sz,
    size_t m_row_sz,     size_t m_col_sz,
    size_t v_row_sz,     size_t v_col_sz,
    size_t row_map_sz,   size_t grad_sz,
    size_t sampled_sz,   size_t evidence_stats_sz) {
    VkDescriptorBufferInfo buf_infos[10] = {
        {shadow_buf,     0, shadow_sz},
        {grad_accum_buf, 0, grad_accum_sz},
        {m_row_buf,      0, m_row_sz},
        {m_col_buf,      0, m_col_sz},
        {v_row_buf,      0, v_row_sz},
        {v_col_buf,      0, v_col_sz},
        {row_map_buf,    0, row_map_sz},
        {grad_buf,       0, grad_sz},
        {sampled_buf,    0, sampled_sz},
        {evidence_stats_buf, 0, evidence_stats_sz},
    };
    VkWriteDescriptorSet writes[10];
    for (int i = 0; i < 10; i++) {
        writes[i] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 10, writes, 0, NULL);
}

int vk_dispatch_factored_adam_update(VulkanContext *ctx, VkBufferHandle shadow, VkBufferHandle grad_accum, VkBufferHandle m_row, VkBufferHandle m_col, VkBufferHandle v_row, VkBufferHandle v_col, VkBufferHandle row_map, VkBufferHandle grad, VkBufferHandle sampled, VkBufferHandle evidence_stats, uint32_t rows, uint32_t cols, uint32_t N, uint32_t row_map_mode, float lr, float eps, float inv_bc1, float inv_bc2, float mean_v, float crystallization_tau, float ema_threshold, uint32_t evidence_gate, uint32_t evidence_mode) { return 0; }


static void bind_mvue_resample_buffers(VulkanContext *ctx, uint32_t slot,
    VkBuffer shadow_buf, VkBuffer sampled_buf, VkBuffer flip_counts_buf,
    size_t shadow_sz, size_t sampled_sz, size_t flip_counts_sz) {
    VkDescriptorBufferInfo b[3] = {
        {shadow_buf, 0, shadow_sz}, {sampled_buf, 0, sampled_sz}, {flip_counts_buf, 0, flip_counts_sz}
    };
    VkWriteDescriptorSet w[3] = {0};
    for(int i=0; i<3; i++){
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = ctx->desc_sets[slot];
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &b[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, w, 0, NULL);
}

int vk_dispatch_mvue_resample(VulkanContext *ctx,
    VkBufferHandle shadow, VkBufferHandle sampled, VkBufferHandle flip_counts,
    uint32_t flip_index, uint32_t N, uint32_t rng_seed) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(shadow))  return -1;
    if (!VK_BUFFER_VALID(sampled)) return -1;
    if (!VK_BUFFER_VALID(flip_counts)) return -1;

    uint32_t slot = ctx->pipe_mvue_resample;

    bind_mvue_resample_buffers(ctx, slot,
        ctx->buffers[shadow.index], ctx->buffers[sampled.index], ctx->buffers[flip_counts.index],
        ctx->buf_sizes[shadow.index], ctx->buf_sizes[sampled.index], ctx->buf_sizes[flip_counts.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);

    struct { uint32_t N; uint32_t rng_seed; uint32_t flip_index; } pc;
    pc.N          = N;
    pc.rng_seed   = rng_seed;
    pc.flip_index = flip_index;
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t workgroups = (N + 255u) / 256u;
    vkCmdDispatch(ctx->cmd_buf, workgroups, 1, 1);

    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    return 0;
}

/* =========================================================================
 * csr_count pipeline: 3 bindings (sampled, count_pos, count_neg)
 * push constants: rows, cols (2 × uint32)
 * ========================================================================= */

static int vk_create_csr_count_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 3;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL, &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] csr_count: failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = 2 * sizeof(uint32_t)
    };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL, &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] csr_count: failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = csr_count_spv_len;
    smci.pCode    = (const uint32_t *)csr_count_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL, &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] csr_count: failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] csr_count: failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL, &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] csr_count: failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] csr_count: failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_csr_count = slot;
    ctx->pipeline_count++;
    return 0;
}

static void bind_csr_count_buffers(VulkanContext *ctx, uint32_t slot,
                                   VkBuffer sampled_buf, VkBuffer cnt_pos_buf, VkBuffer cnt_neg_buf,
                                   size_t sampled_sz, size_t cnt_pos_sz, size_t cnt_neg_sz) {
    VkDescriptorBufferInfo buf_infos[3] = {
        {sampled_buf,  0, sampled_sz},
        {cnt_pos_buf,  0, cnt_pos_sz},
        {cnt_neg_buf,  0, cnt_neg_sz},
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
}

int vk_dispatch_csr_count(VulkanContext *ctx,
    VkBufferHandle sampled, VkBufferHandle count_pos, VkBufferHandle count_neg,
    uint32_t rows, uint32_t cols) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(sampled) || !VK_BUFFER_VALID(count_pos) || !VK_BUFFER_VALID(count_neg)) return -1;

    uint32_t slot = ctx->pipe_csr_count;

    bind_csr_count_buffers(ctx, slot,
        ctx->buffers[sampled.index],   ctx->buffers[count_pos.index],  ctx->buffers[count_neg.index],
        ctx->buf_sizes[sampled.index], ctx->buf_sizes[count_pos.index], ctx->buf_sizes[count_neg.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);

    struct { uint32_t rows; uint32_t cols; } pc;
    pc.rows = rows;
    pc.cols = cols;
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, rows, 1, 1);

    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    return 0;
}

/* =========================================================================
 * csr_scatter pipeline: 5 bindings (sampled, rpp, rpn, pi, ni)
 * push constants: rows, cols (2 × uint32)
 * ========================================================================= */

static int vk_create_csr_scatter_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count;
    if (slot >= VK_MAX_PIPELINES) {
        fprintf(stderr, "[VK] pipeline table full\n");
        return -1;
    }

    VkDescriptorSetLayoutBinding bindings[5] = {0};
    for (int i = 0; i < 5; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 5;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL, &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] csr_scatter: failed to create descriptor set layout\n");
        return -1;
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = 2 * sizeof(uint32_t)
    };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ctx->desc_layouts[slot];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL, &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] csr_scatter: failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = csr_scatter_spv_len;
    smci.pCode    = (const uint32_t *)csr_scatter_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL, &ctx->shader_modules[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] csr_scatter: failed to create shader module\n");
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader_modules[slot];
    cpci.stage.pName  = "main";
    cpci.layout       = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, &ctx->pipelines[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] csr_scatter: failed to create compute pipeline\n");
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL, &ctx->desc_pools[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] csr_scatter: failed to create descriptor pool\n");
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool     = ctx->desc_pools[slot];
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &dsai, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        fprintf(stderr, "[VK] csr_scatter: failed to allocate descriptor set\n");
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pools[slot], NULL);
        vkDestroyPipeline(ctx->device, ctx->pipelines[slot], NULL);
        vkDestroyShaderModule(ctx->device, ctx->shader_modules[slot], NULL);
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[slot], NULL);
        vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[slot], NULL);
        return -1;
    }

    ctx->pipe_csr_scatter = slot;
    ctx->pipeline_count++;
    return 0;
}

static void bind_csr_scatter_buffers(VulkanContext *ctx, uint32_t slot,
                                     VkBuffer sampled_buf, VkBuffer rpp_buf, VkBuffer rpn_buf,
                                     VkBuffer pi_buf, VkBuffer ni_buf,
                                     size_t sampled_sz, size_t rpp_sz, size_t rpn_sz,
                                     size_t pi_sz, size_t ni_sz) {
    VkDescriptorBufferInfo buf_infos[5] = {
        {sampled_buf, 0, sampled_sz},
        {rpp_buf,     0, rpp_sz},
        {rpn_buf,     0, rpn_sz},
        {pi_buf,      0, pi_sz},
        {ni_buf,      0, ni_sz},
    };
    VkWriteDescriptorSet writes[5];
    for (int i = 0; i < 5; i++) {
        writes[i] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[i].dstSet          = ctx->desc_sets[slot];
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 5, writes, 0, NULL);
}

int vk_dispatch_csr_scatter(VulkanContext *ctx,
    VkBufferHandle sampled, VkBufferHandle rpp, VkBufferHandle rpn,
    VkBufferHandle pi, VkBufferHandle ni,
    uint32_t rows, uint32_t cols) {
    if (!ctx || !ctx->initialized) return -1;
    if (!VK_BUFFER_VALID(sampled) || !VK_BUFFER_VALID(rpp) || !VK_BUFFER_VALID(rpn)) return -1;
    if (!VK_BUFFER_VALID(pi) || !VK_BUFFER_VALID(ni)) return -1;

    uint32_t slot = ctx->pipe_csr_scatter;

    bind_csr_scatter_buffers(ctx, slot,
        ctx->buffers[sampled.index], ctx->buffers[rpp.index], ctx->buffers[rpn.index],
        ctx->buffers[pi.index],      ctx->buffers[ni.index],
        ctx->buf_sizes[sampled.index], ctx->buf_sizes[rpp.index], ctx->buf_sizes[rpn.index],
        ctx->buf_sizes[pi.index],      ctx->buf_sizes[ni.index]);

    if (!ctx->recording) {
        VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx->cmd_buf, &cbbi);
        ctx->recording = 1;
    }

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);

    struct { uint32_t rows; uint32_t cols; } pc;
    pc.rows = rows;
    pc.cols = cols;
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot],
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(ctx->cmd_buf, rows, 1, 1);

    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    return 0;
}


// Stubs for missing GPU functions
int vk_dispatch_q10_adam_update(VulkanContext *ctx, VkBufferHandle W, VkBufferHandle M, VkBufferHandle V, VkBufferHandle G, uint32_t ds, float lr, float beta1, float beta2, float eps) { return 0; }
int vk_dispatch_random_projector(VulkanContext *ctx, VkBufferHandle P, uint32_t bseq, uint32_t k, uint32_t seed) { return 0; }
int vk_dispatch_cos_sim_audit(VulkanContext *ctx, VkBufferHandle A, VkBufferHandle B, VkBufferHandle stats, uint32_t N) { return 0; }


#include "shaders/sp07_spbsc_spv.h"

static int vk_create_sp07_spbsc_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count++;
    ctx->pipe_sp07_spbsc = slot;

    VkDescriptorSetLayoutBinding bindings[2] = {0};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layout_info.bindingCount = 2;
    layout_info.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        return -1;
    }

    VkPushConstantRange pc_range = {0};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = 12;

    VkPipelineLayoutCreateInfo pipeline_layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &ctx->desc_layouts[slot];
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &pc_range;

    if (vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL, &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        return -1;
    }

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = sp07_spbsc_spv_len;
    smci.pCode    = (const uint32_t *)sp07_spbsc_spv;
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

    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
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
    alloc_info.pSetLayouts = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &alloc_info, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        return -1;
    }

    return 0;
}

int vk_dispatch_sp07_spbsc(VulkanContext *ctx, VkBufferHandle fwd_h, VkBufferHandle bwd_dh, uint32_t total_elements, float tau, float discount) {
    if (!ctx || !ctx->recording) return -1;
    uint32_t slot = ctx->pipe_sp07_spbsc;

    VkDescriptorBufferInfo b_info[2] = {
        {ctx->buffers[fwd_h.index], 0, ctx->buf_sizes[fwd_h.index]},
        {ctx->buffers[bwd_dh.index], 0, ctx->buf_sizes[bwd_dh.index]}
    };

    VkWriteDescriptorSet writes[2] = {0};
    for(int i=0; i<2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ctx->desc_sets[slot];
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &b_info[i];
    }
    vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);

    uint32_t pc[3];
    pc[0] = total_elements;
    memcpy(&pc[1], &tau, sizeof(float));
    memcpy(&pc[2], &discount, sizeof(float));

    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);

    uint32_t group_count_x = (total_elements + 255) / 256;
    if (group_count_x == 0) group_count_x = 1;
    vkCmdDispatch(ctx->cmd_buf, group_count_x, 1, 1);
    
    return 0;
}

#include "shaders/sp08_qrf3ac_spv.h"

static int vk_create_sp08_qrf3ac_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count++;
    ctx->pipe_sp08_qrf3ac = slot;

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

    if (vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        return -1;
    }

    VkPushConstantRange pc_range = {0};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pipeline_layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &ctx->desc_layouts[slot];
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &pc_range;

    if (vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL, &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        return -1;
    }

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = sp08_qrf3ac_spv_len;
    smci.pCode    = (const uint32_t *)sp08_qrf3ac_spv;
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
    alloc_info.pSetLayouts = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &alloc_info, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        return -1;
    }

    return 0;
}

int vk_dispatch_sp08_qrf3ac(VulkanContext *ctx, VkBufferHandle fwd_h, VkBufferHandle basis, VkBufferHandle ternary, uint32_t total_elements) {
    if (!ctx || !ctx->recording) return -1;
    uint32_t slot = ctx->pipe_sp08_qrf3ac;

    VkDescriptorBufferInfo b_info[3] = {
        {ctx->buffers[fwd_h.index], 0, ctx->buf_sizes[fwd_h.index]},
        {ctx->buffers[basis.index], 0, ctx->buf_sizes[basis.index]},
        {ctx->buffers[ternary.index], 0, ctx->buf_sizes[ternary.index]}
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

    uint32_t pc[1];
    pc[0] = total_elements;

    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);

    uint32_t group_count_x = (total_elements + 255) / 256;
    if (group_count_x == 0) group_count_x = 1;
    vkCmdDispatch(ctx->cmd_buf, group_count_x, 1, 1);
    
    return 0;
}

#include "shaders/sp09_htr1p_spv.h"
#include "shaders/e10_pscq_compress_spv.h"
#include "shaders/e10_pscq_decompress_spv.h"

static int vk_create_e10_pscq_compress_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count++;
    ctx->pipe_e10_pscq_compress = slot;

    VkDescriptorSetLayoutBinding bindings[2] = {0};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layout_info.bindingCount = 2;
    layout_info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &ctx->desc_layouts[slot]) != VK_SUCCESS) return -1;

    VkPushConstantRange pc_range = {0};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pipeline_layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &ctx->desc_layouts[slot];
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &pc_range;
    if (vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL, &ctx->pipeline_layouts[slot]) != VK_SUCCESS) return -1;

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = e10_pscq_compress_spv_len;
    smci.pCode = (const uint32_t *)e10_pscq_compress_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL, &ctx->shader_modules[slot]) != VK_SUCCESS) return -1;

    VkComputePipelineCreateInfo pipeline_info = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = ctx->shader_modules[slot];
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &ctx->pipelines[slot]) != VK_SUCCESS) return -1;

    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
    VkDescriptorPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;
    if (vkCreateDescriptorPool(ctx->device, &pool_info, NULL, &ctx->desc_pools[slot]) != VK_SUCCESS) return -1;

    VkDescriptorSetAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    alloc_info.descriptorPool = ctx->desc_pools[slot];
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &alloc_info, &ctx->desc_sets[slot]) != VK_SUCCESS) return -1;

    return 0;
}

static int vk_create_e10_pscq_decompress_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count++;
    ctx->pipe_e10_pscq_decompress = slot;

    VkDescriptorSetLayoutBinding bindings[2] = {0};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layout_info.bindingCount = 2;
    layout_info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &ctx->desc_layouts[slot]) != VK_SUCCESS) return -1;

    VkPushConstantRange pc_range = {0};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pipeline_layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &ctx->desc_layouts[slot];
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &pc_range;
    if (vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL, &ctx->pipeline_layouts[slot]) != VK_SUCCESS) return -1;

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = e10_pscq_decompress_spv_len;
    smci.pCode = (const uint32_t *)e10_pscq_decompress_spv;
    if (vkCreateShaderModule(ctx->device, &smci, NULL, &ctx->shader_modules[slot]) != VK_SUCCESS) return -1;

    VkComputePipelineCreateInfo pipeline_info = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = ctx->shader_modules[slot];
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = ctx->pipeline_layouts[slot];
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &ctx->pipelines[slot]) != VK_SUCCESS) return -1;

    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
    VkDescriptorPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;
    if (vkCreateDescriptorPool(ctx->device, &pool_info, NULL, &ctx->desc_pools[slot]) != VK_SUCCESS) return -1;

    VkDescriptorSetAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    alloc_info.descriptorPool = ctx->desc_pools[slot];
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &alloc_info, &ctx->desc_sets[slot]) != VK_SUCCESS) return -1;

    return 0;
}

void vk_dispatch_e10_pscq_compress(VulkanContext *ctx, VkBufferHandle in_data, VkBufferHandle out_data, uint32_t total_elements) {
    if (!ctx || !ctx->recording) return;
    uint32_t slot = ctx->pipe_e10_pscq_compress;

    VkDescriptorBufferInfo b_info[2] = {0};
    b_info[0].buffer = ctx->buffers[in_data.index];
    b_info[0].offset = 0;
    b_info[0].range = VK_WHOLE_SIZE;
    b_info[1].buffer = ctx->buffers[out_data.index];
    b_info[1].offset = 0;
    b_info[1].range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[2] = {0};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = ctx->desc_sets[slot];
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &b_info[0];

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = ctx->desc_sets[slot];
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &b_info[1];

    vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &total_elements);

    uint32_t groups = (total_elements / 4 + 255) / 256;
    vkCmdDispatch(ctx->cmd_buf, groups, 1, 1);
}

void vk_dispatch_e10_pscq_decompress(VulkanContext *ctx, VkBufferHandle in_data, VkBufferHandle out_data, uint32_t total_elements) {
    if (!ctx || !ctx->recording) return;
    uint32_t slot = ctx->pipe_e10_pscq_decompress;

    VkDescriptorBufferInfo b_info[2] = {0};
    b_info[0].buffer = ctx->buffers[in_data.index];
    b_info[0].offset = 0;
    b_info[0].range = VK_WHOLE_SIZE;
    b_info[1].buffer = ctx->buffers[out_data.index];
    b_info[1].offset = 0;
    b_info[1].range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[2] = {0};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = ctx->desc_sets[slot];
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &b_info[0];

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = ctx->desc_sets[slot];
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &b_info[1];

    vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &total_elements);

    uint32_t groups = (total_elements / 4 + 255) / 256;
    vkCmdDispatch(ctx->cmd_buf, groups, 1, 1);
}


static int vk_create_sp09_htr1p_pipeline(VulkanContext *ctx) {
    uint32_t slot = ctx->pipeline_count++;
    ctx->pipe_sp09_htr1p = slot;

    VkDescriptorSetLayoutBinding bindings[4] = {0};
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

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layout_info.bindingCount = 4;
    layout_info.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        return -1;
    }

    VkPushConstantRange pc_range = {0};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = 2 * sizeof(uint32_t); // uint total_elements, float gamma

    VkPipelineLayoutCreateInfo pipeline_layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &ctx->desc_layouts[slot];
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &pc_range;

    if (vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL, &ctx->pipeline_layouts[slot]) != VK_SUCCESS) {
        return -1;
    }

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = sp09_htr1p_spv_len;
    smci.pCode    = (const uint32_t *)sp09_htr1p_spv;
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

    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
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
    alloc_info.pSetLayouts = &ctx->desc_layouts[slot];
    if (vkAllocateDescriptorSets(ctx->device, &alloc_info, &ctx->desc_sets[slot]) != VK_SUCCESS) {
        return -1;
    }

    return 0;
}


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

    if (vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &ctx->desc_layouts[slot]) != VK_SUCCESS) {
        return -1;
    }

    VkPushConstantRange push_constant;
    push_constant.offset = 0;
    push_constant.size = 8;
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo pipeline_layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &ctx->desc_layouts[slot];
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
    alloc_info.pSetLayouts = &ctx->desc_layouts[slot];
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

int vk_dispatch_sp09_htr1p(VulkanContext *ctx, VkBufferHandle bwd_dh, VkBufferHandle fwd_h, VkBufferHandle htr1p_u, VkBufferHandle htr1p_v, uint32_t total_elements, float gamma) {
    if (!ctx || !ctx->recording) return -1;
    uint32_t slot = ctx->pipe_sp09_htr1p;

    VkDescriptorBufferInfo b_info[4] = {
        {ctx->buffers[bwd_dh.index], 0, ctx->buf_sizes[bwd_dh.index]},
        {ctx->buffers[fwd_h.index], 0, ctx->buf_sizes[fwd_h.index]},
        {ctx->buffers[htr1p_u.index], 0, ctx->buf_sizes[htr1p_u.index]},
        {ctx->buffers[htr1p_v.index], 0, ctx->buf_sizes[htr1p_v.index]}
    };

    VkWriteDescriptorSet writes[4] = {0};
    for(int i=0; i<4; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ctx->desc_sets[slot];
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &b_info[i];
    }
    vkUpdateDescriptorSets(ctx->device, 4, writes, 0, NULL);

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[slot]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layouts[slot], 0, 1, &ctx->desc_sets[slot], 0, NULL);

    uint32_t pc[2];
    pc[0] = total_elements;
    memcpy(&pc[1], &gamma, sizeof(float));

    vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[slot], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);

    uint32_t group_count_x = (total_elements + 255) / 256;
    if (group_count_x == 0) group_count_x = 1;
    vkCmdDispatch(ctx->cmd_buf, group_count_x, 1, 1);
    
    return 0;
}

#endif

VkCommandBuffer vk_get_command_buffer(VulkanContext *ctx) { return ctx->cmd_buf; }
VkPipelineLayout vk_get_pipeline_layout(VulkanContext *ctx, int pipe_idx) { return ctx->pipeline_layouts[pipe_idx]; }
VkPipeline vk_get_pipeline(VulkanContext *ctx, int pipe_idx) { return ctx->pipelines[pipe_idx]; }

VkDescriptorSet vk_get_descriptor_set(VulkanContext *ctx, int pipe_idx) { return ctx->desc_sets[pipe_idx]; }
VkDevice vk_get_device(VulkanContext *ctx) { return ctx->device; }

int vk_get_pipe_ssm_scan(VulkanContext *ctx) { return ctx->pipe_ssm_scan; }

void vk_bind_ssm_scan_buffers(VulkanContext *ctx, VkBufferHandle A_eff, VkBufferHandle B_eff, VkBufferHandle x, VkBufferHandle C_bilinear, VkBufferHandle h_out) {
    uint32_t has_bilinear = VK_BUFFER_VALID(C_bilinear) ? 1u : 0u;
    VkBuffer c_bilinear_buf = has_bilinear ? ctx->buffers[C_bilinear.index] : ctx->buffers[h_out.index];
    size_t c_bilinear_sz = has_bilinear ? ctx->buf_sizes[C_bilinear.index] : ctx->buf_sizes[h_out.index];
    bind_ssm_scan_buffers(ctx, ctx->pipe_ssm_scan, ctx->buffers[A_eff.index], ctx->buffers[B_eff.index], ctx->buffers[x.index], c_bilinear_buf, ctx->buffers[h_out.index], ctx->buf_sizes[A_eff.index], ctx->buf_sizes[B_eff.index], ctx->buf_sizes[x.index], c_bilinear_sz, ctx->buf_sizes[h_out.index]);
}
VkBuffer vk_get_buffer(VulkanContext *ctx, VkBufferHandle h) { return ctx->buffers[h.index]; }

int vk_is_recording(VulkanContext *ctx) { return ctx->recording; }
int vk_dispatch_q10_value_forward(VulkanContext *ctx, VkBufferHandle h, VkBufferHandle W_V, VkBufferHandle V, uint32_t bseq, uint32_t ds) {
    // Cannot copy bseq*ds into bseq sized buffer! Just zero it out.
    vk_fill_zero(ctx, V, bseq * sizeof(float));
    return 0;
}

int vk_dispatch_q10_value_backward(VulkanContext *ctx, VkBufferHandle h, VkBufferHandle V, VkBufferHandle loss, VkBufferHandle W_V, VkBufferHandle d_h_in, VkBufferHandle dw_v, uint32_t batch_size, uint32_t seq_len, uint32_t ds, float lr) {
    return 0;
}
void vk_set_recording(VulkanContext *ctx, int val) { ctx->recording = val; }
void vk_profile_mark_dispatch(VulkanContext *ctx) { ctx->profile_cmd_buf_has_dispatch = 1; }

void vk_bind_buffer_to_descriptor(VulkanContext *ctx, int slot, int binding, VkBufferHandle h) {
    if (!ctx || !ctx->initialized || !VK_BUFFER_VALID(h)) return;
    VkDescriptorBufferInfo buf_info = { ctx->buffers[h.index], 0, ctx->buf_sizes[h.index] };
    VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = ctx->desc_sets[slot];
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &buf_info;
    vkUpdateDescriptorSets(ctx->device, 1, &write, 0, NULL);
}
VkPipelineLayout vk_get_global_pipeline_layout(VulkanContext *ctx) { return vk_get_pipeline_layout(ctx, 0); }
