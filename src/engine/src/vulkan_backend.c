#include "vulkan_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ENGINE_VULKAN
#include <vulkan/vulkan.h>

typedef struct {
    uint32_t family;
    uint32_t count;
} vk_queue_family_t;

struct vk_context_s {
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

    const trnm_model_t* model;
    bool recording;
    int dispatch_count;

    char device_name[256];
    uint64_t device_vram;
};

static bool vk_check_compute_queue(VkPhysicalDevice phys_device, uint32_t* family_idx) {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_family_count, NULL);

    if (queue_family_count == 0) {
        return false;
    }

    VkQueueFamilyProperties* queue_families = malloc(queue_family_count * sizeof(VkQueueFamilyProperties));
    if (!queue_families) {
        return false;
    }

    vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_family_count, queue_families);

    bool found = false;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            *family_idx = i;
            found = true;
            break;
        }
    }

    free(queue_families);
    return found;
}

static bool vk_select_physical_device(VkInstance instance, VkPhysicalDevice* out_phys_device, uint32_t* out_queue_family) {
    uint32_t device_count = 0;
    VkResult res = vkEnumeratePhysicalDevices(instance, &device_count, NULL);
    if (res != VK_SUCCESS || device_count == 0) {
        return false;
    }

    VkPhysicalDevice* devices = malloc(device_count * sizeof(VkPhysicalDevice));
    if (!devices) {
        return false;
    }

    res = vkEnumeratePhysicalDevices(instance, &device_count, devices);
    if (res != VK_SUCCESS) {
        free(devices);
        return false;
    }

    VkPhysicalDevice discrete_gpu = VK_NULL_HANDLE;
    VkPhysicalDevice integrated_gpu = VK_NULL_HANDLE;
    uint32_t discrete_queue = UINT32_MAX;
    uint32_t integrated_queue = UINT32_MAX;

    for (uint32_t i = 0; i < device_count; i++) {
        uint32_t queue_family = UINT32_MAX;
        if (!vk_check_compute_queue(devices[i], &queue_family)) {
            continue;
        }

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);

        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            discrete_gpu = devices[i];
            discrete_queue = queue_family;
        } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && integrated_gpu == VK_NULL_HANDLE) {
            integrated_gpu = devices[i];
            integrated_queue = queue_family;
        }
    }

    bool found = false;
    if (discrete_gpu != VK_NULL_HANDLE) {
        *out_phys_device = discrete_gpu;
        *out_queue_family = discrete_queue;
        found = true;
    } else if (integrated_gpu != VK_NULL_HANDLE) {
        *out_phys_device = integrated_gpu;
        *out_queue_family = integrated_queue;
        found = true;
    }

    free(devices);
    return found;
}

bool vk_available(void) {
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "TernaryEngine",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "TernaryEngine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0,
    };

    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = 0,
        .enabledExtensionCount = 0,
    };

    VkInstance instance;
    VkResult res = vkCreateInstance(&create_info, NULL, &instance);
    if (res != VK_SUCCESS) {
        return false;
    }

    VkPhysicalDevice phys_device;
    uint32_t queue_family;
    bool has_compute = vk_select_physical_device(instance, &phys_device, &queue_family);

    vkDestroyInstance(instance, NULL);

    return has_compute;
}

vk_context_t* vk_init(void) {
    vk_context_t* ctx = calloc(1, sizeof(vk_context_t));
    if (!ctx) {
        fprintf(stderr, "Vulkan: failed to allocate context\n");
        return NULL;
    }

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "TernaryEngine",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "TernaryEngine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0,
    };

    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = 0,
        .enabledExtensionCount = 0,
    };

    VkResult res = vkCreateInstance(&instance_info, NULL, &ctx->instance);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to create instance (code %d)\n", res);
        free(ctx);
        return NULL;
    }

    if (!vk_select_physical_device(ctx->instance, &ctx->phys_device, &ctx->queue_family)) {
        fprintf(stderr, "Vulkan: no compute-capable GPU found\n");
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return NULL;
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx->phys_device, &props);
    strncpy(ctx->device_name, (const char*)props.deviceName, sizeof(ctx->device_name) - 1);

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(ctx->phys_device, &mem_props);
    ctx->device_vram = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            ctx->device_vram = mem_props.memoryHeaps[i].size;
            break;
        }
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = 0,
        .enabledLayerCount = 0,
    };

    res = vkCreateDevice(ctx->phys_device, &device_info, NULL, &ctx->device);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to create device (code %d)\n", res);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return NULL;
    }

    vkGetDeviceQueue(ctx->device, ctx->queue_family, 0, &ctx->compute_queue);

    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx->queue_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };

    res = vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->cmd_pool);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to create command pool (code %d)\n", res);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return NULL;
    }

    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    res = vkAllocateCommandBuffers(ctx->device, &alloc_info, &ctx->cmd_buf);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to allocate command buffer (code %d)\n", res);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return NULL;
    }

    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    res = vkCreateFence(ctx->device, &fence_info, NULL, &ctx->fence);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to create fence (code %d)\n", res);
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &ctx->cmd_buf);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return NULL;
    }

    ctx->recording = false;
    ctx->dispatch_count = 0;
    ctx->model = NULL;

    return ctx;
}

void vk_shutdown(vk_context_t* ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(ctx->device);

        if (ctx->fence != VK_NULL_HANDLE) {
            vkDestroyFence(ctx->device, ctx->fence, NULL);
        }

        if (ctx->cmd_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        }

        if (ctx->desc_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
        }

        if (ctx->desc_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layout, NULL);
        }

        for (int i = 0; i < 7; i++) {
            if (ctx->pipelines[i] != VK_NULL_HANDLE) {
                vkDestroyPipeline(ctx->device, ctx->pipelines[i], NULL);
            }
            if (ctx->pipeline_layouts[i] != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[i], NULL);
            }
            if (ctx->shader_modules[i] != VK_NULL_HANDLE) {
                vkDestroyShaderModule(ctx->device, ctx->shader_modules[i], NULL);
            }
        }

        if (ctx->weight_buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(ctx->device, ctx->weight_buf, NULL);
        }
        if (ctx->weight_mem != VK_NULL_HANDLE) {
            vkFreeMemory(ctx->device, ctx->weight_mem, NULL);
        }

        if (ctx->scratch_buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(ctx->device, ctx->scratch_buf, NULL);
        }
        if (ctx->scratch_mem != VK_NULL_HANDLE) {
            vkFreeMemory(ctx->device, ctx->scratch_mem, NULL);
        }

        if (ctx->staging_buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        }
        if (ctx->staging_mem != VK_NULL_HANDLE) {
            vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        }

        vkDestroyDevice(ctx->device, NULL);
    }

    if (ctx->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(ctx->instance, NULL);
    }

    free(ctx);
}

int vk_upload_weights(vk_context_t* ctx, const trnm_model_t* model) {
    if (!ctx || !model) {
        fprintf(stderr, "Vulkan: vk_upload_weights called with NULL context or model\n");
        return -1;
    }

    ctx->model = model;
    return 0;
}

void vk_ternary_gemv(vk_context_t* ctx,
    uint32_t y_off, uint32_t W_off, uint32_t scale_off,
    uint32_t x_off, uint32_t bias_off, int32_t rows, int32_t cols) {
    if (!ctx) return;
}

void vk_float_gemv(vk_context_t* ctx,
    uint32_t y_off, uint32_t W_off, uint32_t x_off,
    uint32_t bias_off, int32_t rows, int32_t cols) {
    if (!ctx) return;
}

void vk_rmsnorm(vk_context_t* ctx,
    uint32_t y_off, uint32_t x_off, uint32_t w_off, int32_t n, float eps) {
    if (!ctx) return;
}

void vk_elementwise(vk_context_t* ctx,
    uint32_t y_off, uint32_t a_off, uint32_t b_off, int32_t n, int32_t mode) {
    if (!ctx) return;
}

void vk_rope(vk_context_t* ctx,
    uint32_t q_off, uint32_t k_off, uint32_t freq_off,
    int32_t n_heads, int32_t head_dim, int32_t pos) {
    if (!ctx) return;
}

void vk_attention(vk_context_t* ctx,
    uint32_t out_off, uint32_t q_off, uint32_t k_cache_off,
    uint32_t v_cache_off, uint32_t scores_off,
    int32_t n_heads, int32_t head_dim, int32_t cache_len,
    int32_t cache_start, int32_t window_size, float scale) {
    if (!ctx) return;
}

void vk_conv1d_step(vk_context_t* ctx,
    uint32_t y_off, uint32_t conv_buf_off, uint32_t x_off,
    uint32_t w_off, uint32_t bias_off, int32_t d_inner, int32_t d_conv) {
    if (!ctx) return;
}

void vk_upload_to_scratch(vk_context_t* ctx, uint32_t off, const void* data, size_t bytes) {
    if (!ctx) return;
}

void vk_download_from_scratch(vk_context_t* ctx, uint32_t off, void* data, size_t bytes) {
    if (!ctx) return;
}

void vk_submit_and_wait(vk_context_t* ctx) {
    if (!ctx) return;
}

const char* vk_device_name(const vk_context_t* ctx) {
    if (!ctx) return "unknown";
    return ctx->device_name;
}

uint64_t vk_device_memory(const vk_context_t* ctx) {
    if (!ctx) return 0;
    return ctx->device_vram;
}

#else

bool vk_available(void) {
    return false;
}

vk_context_t* vk_init(void) {
    fprintf(stderr, "Vulkan: not compiled in (ENGINE_VULKAN not defined)\n");
    return NULL;
}

void vk_shutdown(vk_context_t* ctx) {
    (void)ctx;
}

int vk_upload_weights(vk_context_t* ctx, const trnm_model_t* model) {
    (void)ctx;
    (void)model;
    return -1;
}

void vk_ternary_gemv(vk_context_t* ctx,
    uint32_t y_off, uint32_t W_off, uint32_t scale_off,
    uint32_t x_off, uint32_t bias_off, int32_t rows, int32_t cols) {
    (void)ctx;
    (void)y_off;
    (void)W_off;
    (void)scale_off;
    (void)x_off;
    (void)bias_off;
    (void)rows;
    (void)cols;
}

void vk_float_gemv(vk_context_t* ctx,
    uint32_t y_off, uint32_t W_off, uint32_t x_off,
    uint32_t bias_off, int32_t rows, int32_t cols) {
    (void)ctx;
    (void)y_off;
    (void)W_off;
    (void)x_off;
    (void)bias_off;
    (void)rows;
    (void)cols;
}

void vk_rmsnorm(vk_context_t* ctx,
    uint32_t y_off, uint32_t x_off, uint32_t w_off, int32_t n, float eps) {
    (void)ctx;
    (void)y_off;
    (void)x_off;
    (void)w_off;
    (void)n;
    (void)eps;
}

void vk_elementwise(vk_context_t* ctx,
    uint32_t y_off, uint32_t a_off, uint32_t b_off, int32_t n, int32_t mode) {
    (void)ctx;
    (void)y_off;
    (void)a_off;
    (void)b_off;
    (void)n;
    (void)mode;
}

void vk_rope(vk_context_t* ctx,
    uint32_t q_off, uint32_t k_off, uint32_t freq_off,
    int32_t n_heads, int32_t head_dim, int32_t pos) {
    (void)ctx;
    (void)q_off;
    (void)k_off;
    (void)freq_off;
    (void)n_heads;
    (void)head_dim;
    (void)pos;
}

void vk_attention(vk_context_t* ctx,
    uint32_t out_off, uint32_t q_off, uint32_t k_cache_off,
    uint32_t v_cache_off, uint32_t scores_off,
    int32_t n_heads, int32_t head_dim, int32_t cache_len,
    int32_t cache_start, int32_t window_size, float scale) {
    (void)ctx;
    (void)out_off;
    (void)q_off;
    (void)k_cache_off;
    (void)v_cache_off;
    (void)scores_off;
    (void)n_heads;
    (void)head_dim;
    (void)cache_len;
    (void)cache_start;
    (void)window_size;
    (void)scale;
}

void vk_conv1d_step(vk_context_t* ctx,
    uint32_t y_off, uint32_t conv_buf_off, uint32_t x_off,
    uint32_t w_off, uint32_t bias_off, int32_t d_inner, int32_t d_conv) {
    (void)ctx;
    (void)y_off;
    (void)conv_buf_off;
    (void)x_off;
    (void)w_off;
    (void)bias_off;
    (void)d_inner;
    (void)d_conv;
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

void vk_submit_and_wait(vk_context_t* ctx) {
    (void)ctx;
}

const char* vk_device_name(const vk_context_t* ctx) {
    (void)ctx;
    return "unavailable";
}

uint64_t vk_device_memory(const vk_context_t* ctx) {
    (void)ctx;
    return 0;
}

#endif
