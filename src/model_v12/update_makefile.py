import os

with open("Makefile", "r") as f:
    makefile = f.read()

# Add to SHADERS logic
if "f37_taqg.comp" not in makefile:
    shader_rule = """	$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/f37_taqg.comp -o src/runtime/shaders/spv/f37_taqg.spv
	python tools/spv_to_header.py src/runtime/shaders/spv/f37_taqg.spv src/runtime/shaders/f37_taqg_spv.h
"""
    makefile = makefile.replace("endif\n\n\nall: train", shader_rule + "endif\n\n\nall: train")
    
with open("Makefile", "w") as f:
    f.write(makefile)

print("Makefile updated")
