import sys

def patch_init():
    with open("src/runtime/vulkan_backend_impl.c", "r") as f:
        content = f.read()

    # Add forward decls
    if "static int vk_create_vsa_fwht_forward_pipeline(VulkanContext *ctx);" not in content:
        decl_insert = content.find("static int vk_create_sp07_spbsc_pipeline(VulkanContext *ctx);")
        decls = """static int vk_create_vsa_fwht_forward_pipeline(VulkanContext *ctx);
static int vk_create_vsa_bent_recurrence_pipeline(VulkanContext *ctx);
static int vk_create_vsa_fractional_laplacian_pipeline(VulkanContext *ctx);
"""
        content = content[:decl_insert] + decls + content[decl_insert:]

    # Add init calls
    init_call_str = "if (vk_create_sp07_spbsc_pipeline(ctx) != 0) {"
    if init_call_str in content and "vk_create_vsa_fwht_forward_pipeline" not in content[content.find(init_call_str):]:
        init_insert = content.find(init_call_str)
        init_calls = """    if (vk_create_vsa_fwht_forward_pipeline(ctx) != 0) {
        fprintf(stderr, "Failed to create VSA FWHT pipeline\\n");
        return -1;
    }
    if (vk_create_vsa_bent_recurrence_pipeline(ctx) != 0) {
        fprintf(stderr, "Failed to create VSA Bent Recurrence pipeline\\n");
        return -1;
    }
    if (vk_create_vsa_fractional_laplacian_pipeline(ctx) != 0) {
        fprintf(stderr, "Failed to create VSA Fractional Laplacian pipeline\\n");
        return -1;
    }
"""
        content = content[:init_insert] + init_calls + content[init_insert:]

    with open("src/runtime/vulkan_backend_impl.c", "w") as f:
        f.write(content)
    print("Patched vulkan_backend_impl.c init calls")

if __name__ == "__main__":
    patch_init()
