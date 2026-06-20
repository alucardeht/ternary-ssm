import sys

content = open('gpu_pipeline.c', 'r', encoding='utf-8').read()

old_sig = '''    uint32_t evidence_mode,
    float evidence_boundary_eps) {'''

new_sig = '''    uint32_t evidence_mode,
    float evidence_boundary_eps,
    float tau,
    float *out_mean_v) {'''

if old_sig in content:
    content = content.replace(old_sig, new_sig)
else:
    print("old_sig not found")
    sys.exit(1)

old_body1 = '''            matrix_rows, cols, n, row_map_mode,
            V12_CRYSTALLIZATION_TAU, V12_BI_STE_FACTOR,'''

new_body1 = '''            matrix_rows, cols, n, row_map_mode,
            tau, V12_BI_STE_FACTOR,'''

content = content.replace(old_body1, new_body1)

old_body2 = '''            lr, 1.0e-8f, inv_bc1, inv_bc2, mean_v,
            V12_CRYSTALLIZATION_TAU, V12_EMA_SPRT_THRESHOLD,'''

new_body2 = '''            lr, 1.0e-8f, inv_bc1, inv_bc2, mean_v,
            tau, V12_EMA_SPRT_THRESHOLD,'''

content = content.replace(old_body2, new_body2)

old_body3 = '''                         (mean_sums[1] / (float)cols);'''

new_body3 = '''                         (mean_sums[1] / (float)cols);
    if (out_mean_v) *out_mean_v = mean_v;'''

content = content.replace(old_body3, new_body3)

open('gpu_pipeline.c', 'w', encoding='utf-8', newline='\n').write(content)
print("gpu_pipeline.c patched")
