import sys

GPU_PIPELINE_PATH = "src/gpu/gpu_pipeline.c"

def fix_corruption():
    with open(GPU_PIPELINE_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    holo_start = 514
    holo_end = 678
    entangled_start = 679
    entangled_end = 702

    # Verify we are on the right lines
    if "sp19_holo_adam_pipeline" in "".join(lines[holo_start:holo_start+5]) and "VkCommandBuffer cmd" in lines[entangled_start]:
        new_lines = lines[:holo_start] + lines[entangled_start:entangled_end+1] + ["\n"] + lines[holo_start:holo_end+1] + lines[entangled_end+1:]
        with open(GPU_PIPELINE_PATH, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print("Fixed corrupted gpu_pipeline.c successfully.")
    else:
        print("Lines did not match expectations.")
        for i in range(holo_start, holo_start+5):
            print(f"{i}: {lines[i].strip()}")
        print(f"{entangled_start}: {lines[entangled_start].strip()}")

if __name__ == "__main__":
    fix_corruption()
