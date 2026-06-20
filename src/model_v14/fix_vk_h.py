import sys

def patch_h():
    with open("src/runtime/vulkan_backend.h", "r") as f:
        content = f.read()

    target = """    uint32_t pipe_vsa_fwht_forward;
    uint32_t pipe_vsa_bent_recurrence;
    uint32_t pipe_vsa_fractional_laplacian;
"""
    if target in content:
        content = content.replace(target, "")
        with open("src/runtime/vulkan_backend.h", "w") as f:
            f.write(content)
        print("Patched vulkan_backend.h to remove global vars")

if __name__ == "__main__":
    patch_h()
