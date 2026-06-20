import sys

with open('Makefile', 'r') as f:
    content = f.read()

target_str = "python tools/spv_to_header.py src/runtime/shaders/spv/f39_bvgr.spv src/runtime/shaders/f39_bvgr_spv.h\n"
insert_str = "\t$(GLSLC) -O --target-env=vulkan1.2 src/runtime/shaders/e13_lrms.comp -o src/runtime/shaders/spv/e13_lrms.spv\n\tpython tools/spv_to_header.py src/runtime/shaders/spv/e13_lrms.spv src/runtime/shaders/e13_lrms_spv.h\n"

if 'e13_lrms.spv' not in content:
    content = content.replace(target_str, target_str + insert_str)

with open('Makefile', 'w') as f:
    f.write(content)
