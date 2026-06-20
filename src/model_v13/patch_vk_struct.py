import sys

def patch_struct():
    with open("src/runtime/vulkan_backend_impl.c", "r") as f:
        content = f.read()

    target = "    uint32_t            pipe_rms_norm_forward;"
    replacement = target + """
    uint32_t            pipe_vsa_fwht_forward;
    uint32_t            pipe_vsa_bent_recurrence;
    uint32_t            pipe_vsa_fractional_laplacian;
"""
    if target in content and "pipe_vsa_fwht_forward" not in content[:content.find(target) + 1000]:
        content = content.replace(target, replacement)
        with open("src/runtime/vulkan_backend_impl.c", "w") as f:
            f.write(content)
        print("Patched struct in vulkan_backend_impl.c")
    else:
        print("Struct already patched or target not found")

if __name__ == "__main__":
    patch_struct()
