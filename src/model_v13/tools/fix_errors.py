import sys

GPU_PIPELINE_PATH = "src/gpu/gpu_pipeline.c"

def fix_errors():
    with open(GPU_PIPELINE_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 1. Comment out body of vk_dispatch_f37_taqg
    in_f37 = False
    for i in range(len(lines)):
        if "void vk_dispatch_f37_taqg(" in lines[i]:
            in_f37 = True
            continue
        if in_f37:
            if lines[i] == "}\n":
                in_f37 = False
                continue
            lines[i] = "// " + lines[i]

    # 2. Comment out specific bad blocks for sp19
    for i in range(9255, 9270):
        lines[i] = "// " + lines[i]
        
    for i in range(10850, 10865):
        lines[i] = "// " + lines[i]

    with open(GPU_PIPELINE_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Patched bad lines in gpu_pipeline.c")

if __name__ == "__main__":
    fix_errors()
