import sys

def patch_train():
    with open("src/gpu/gpu_pipeline_train.c", "r") as f:
        content = f.read()

    target = """#if R504_S10_PSVCTE
#pragma message("Compiling R504_S10_PSVCTE")
            vk_dispatch_sp10_psvcte(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);
#elif R504_S11_LNS_TFMS
#pragma message("Compiling R504_S11_LNS_TFMS")
            vk_dispatch_sp11_lns_tfms(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);
#elif R504_S14_GFIT
#pragma message("Compiling R504_S14_GFIT")
            vk_dispatch_sp14_gfit(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);
#else
#pragma message("Compiling MTP else branch")
            vk_dispatch_sp03_almm(ctx, p->ssm_a, p->ssm_b, p->ssm_x,
                                 p->ssm_cbl, p->ssm_h, seq_len, ds, batch_size);
#endif"""

    replacement = """            // 3. VSA Engine (Replaces SSM Scan entirely)
            // Use ssm_cbl as a dummy alpha array (ensure bounds are safe)
            vk_dispatch_vsa_fractional_laplacian(ctx, p->ssm_x, p->ssm_cbl, p->buf_y, seq_len * batch_size, ds);
            vk_barrier(ctx);

            vk_dispatch_vsa_fwht_forward(ctx, p->buf_y, p->ssm_a, seq_len * batch_size, ds);
            vk_barrier(ctx);

            // Use ssm_v as the Bent-Sequence key
            vk_dispatch_vsa_bent_recurrence(ctx, p->ssm_a, p->ssm_v, p->ssm_b, seq_len * batch_size, ds);
            vk_barrier(ctx);

            vk_dispatch_vsa_fwht_forward(ctx, p->ssm_b, p->ssm_h, seq_len * batch_size, ds);
            // FWHT is unscaled. We could scale it here but we'll let output projection handle it.
"""
    if target in content:
        content = content.replace(target, replacement)
        with open("src/gpu/gpu_pipeline_train.c", "w") as f:
            f.write(content)
        print("Patched gpu_pipeline_train.c")
    else:
        print("Target not found in gpu_pipeline_train.c")

if __name__ == "__main__":
    patch_train()
