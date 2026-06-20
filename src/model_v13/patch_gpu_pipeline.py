import os
path = "src/gpu/gpu_pipeline.c"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
if "#include \"../runtime/shaders/sp11_lns_tfms_spv.h\"" not in content:
    content = content.replace("#include \"../runtime/shaders/sp12_lcmbi_spv.h\"",
                              "#include \"../runtime/shaders/sp12_lcmbi_spv.h\"\n#include \"../runtime/shaders/sp11_lns_tfms_spv.h\"")
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
