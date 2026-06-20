import os

with open("Makefile", "r") as f:
    makefile = f.read()

shaders = [
    "vsa_fwht_forward",
    "vsa_bent_recurrence",
    "vsa_fractional_laplacian",
    "vsa_dyadic_translation",
    "vsa_dyadic_ternary_convolution"
]

rules = ""
for s in shaders:
    if f"{s}.comp" not in makefile:
        rules += f"\t$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/{s}.comp -o src/runtime/shaders/spv/{s}.spv\n"
        rules += f"\tpython tools/spv_to_header.py src/runtime/shaders/spv/{s}.spv src/runtime/shaders/{s}_spv.h\n"

if rules:
    makefile = makefile.replace("endif\n\n\nall: train", rules + "endif\n\n\nall: train")
    with open("Makefile", "w") as f:
        f.write(makefile)
    print("Makefile updated with VSA shaders.")
else:
    print("VSA shaders already in Makefile.")
