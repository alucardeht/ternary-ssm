import os

def patch_file(filepath, old_str, new_str):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if old_str in content:
        content = content.replace(old_str, new_str)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully patched {filepath}")
    else:
        print(f"Could not find old_str in {filepath}")

# Patch ssm_scan_backward.comp
old_comp = "H_t = min(H_t, 64u);"
new_comp = "H_t = min(H_t, 64u);\n    // Finding 43: Nilpotent Residual Algebraic Collapse (NRAC)\n    H_t = min(H_t, 32u);"
patch_file("src/runtime/shaders/ssm_scan_backward.comp", old_comp, new_comp)

# Patch ssm_scan_backward_cr.comp (if H_t exists, wait, it doesn't. We'll skip or just patch it if we find the BPTT horizon)
# Let's patch ssm_scan.c backward functions.
# We need to zero dh_temp / dh_temp_ref / dh every 32 steps.
# In ssm_scan_backward_batch:
old_c1 = '''                if (boundary_mask != NULL && boundary_mask[t]) {
                    dh_val = 0.0f;
                }'''
new_c1 = old_c1 + '''
                // Finding 43: Nilpotent Residual Algebraic Collapse (NRAC)
                if ((seq_len - 1 - t) % 32 == 31) {
                    dh_val = 0.0f;
                }'''

# In ssm_scan_backward_f1_batch
old_c2 = '''                if (boundary_mask != NULL && boundary_mask[t]) {
                    memset(dh_temp_ref, 0, d_state * sizeof(float));
                }'''
new_c2 = old_c2 + '''
                // Finding 43: Nilpotent Residual Algebraic Collapse (NRAC)
                if ((seq_len - 1 - t) % 32 == 31) {
                    memset(dh_temp_ref, 0, d_state * sizeof(float));
                }'''

# In ssm_scan_backward (branch 1)
old_c3 = '''            if (boundary_mask != NULL && boundary_mask[t]) {
                memset(dh_temp, 0, d_state * sizeof(float));
            }'''
new_c3 = old_c3 + '''
            // Finding 43: Nilpotent Residual Algebraic Collapse (NRAC)
            if ((seq_len - 1 - t) % 32 == 31) {
                memset(dh_temp, 0, d_state * sizeof(float));
            }'''

# In SIS-SF backward scan
old_c4 = '''            if (t == 0) {
                memset(h_prev_buf, 0, d_state * sizeof(float));
            } else if (boundary_mask != NULL && boundary_mask[t]) {
                memset(h_prev_buf, 0, d_state * sizeof(float));
            } else {'''
new_c4 = '''            if (t == 0) {
                memset(h_prev_buf, 0, d_state * sizeof(float));
            } else if (boundary_mask != NULL && boundary_mask[t]) {
                memset(h_prev_buf, 0, d_state * sizeof(float));
            } else if ((seq_len - 1 - t) % 32 == 31) { // Finding 43: NRAC
                memset(h_prev_buf, 0, d_state * sizeof(float));
            } else {'''

patch_file("src/core/ssm_scan.c", old_c1, new_c1)
patch_file("src/core/ssm_scan.c", old_c2, new_c2)
patch_file("src/core/ssm_scan.c", old_c3, new_c3)
patch_file("src/core/ssm_scan.c", old_c4, new_c4)
