import re
import os

print("Applying BTRP patch...")

with open("src/model_v12/src/gpu/gpu_pipeline.c", "r", encoding="utf-8") as f:
    code = f.read()

if "#define R504_E11_BTRP 1" not in code:
    code = code.replace("#define R504_E10_PSCQ 1", "#define R504_E10_PSCQ 1\n#define R504_E11_BTRP 1\n#define BTRP_R 16\n")

    # Replace opt_dlm_head size logic
    p_lm_head = r"size_t lm_head_bytes = \(size_t\)m->vocab_size \* d \* sizeof\(float\);"
    r_lm_head = """#ifdef R504_E11_BTRP
    size_t lm_head_bytes = ((size_t)m->vocab_size * BTRP_R + BTRP_R * BTRP_R + BTRP_R * d) * sizeof(float);
#else
    size_t lm_head_bytes = (size_t)m->vocab_size * d * sizeof(float);
#endif"""
    code = re.sub(p_lm_head, r_lm_head, code, count=1)

    p_embed_E = r"size_t embed_E_bytes = \(size_t\)vocab \* d_embed \* sizeof\(float\);"
    r_embed_E = """#ifdef R504_E11_BTRP
    size_t embed_E_bytes = ((size_t)vocab * BTRP_R + BTRP_R * BTRP_R + BTRP_R * d_embed) * sizeof(float);
#else
    size_t embed_E_bytes = (size_t)vocab * d_embed * sizeof(float);
#endif"""
    code = re.sub(p_embed_E, r_embed_E, code, count=1)

    with open("src/model_v12/src/gpu/gpu_pipeline.c", "w", encoding="utf-8") as f:
        f.write(code)
    print("Patched gpu_pipeline.c allocations")
else:
    print("Already patched")
