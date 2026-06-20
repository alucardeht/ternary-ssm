import os
import re

def patch_types_h():
    path = "src/core/types.h"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "lccap_gate_enabled" not in content:
        insert_text = """
    /* R501-S61 Lookahead-Certified CE Anchor Pruning (LCCAP) */
    int32_t lccap_gate_enabled;
    int32_t lccap_segment_length;
    float   lccap_target_loss;
    float   lccap_epsilon_safe;
    uint64_t lccap_segments_total;
    uint64_t lccap_segments_pruned;
"""
        content = re.sub(r'(int32_t use_residual_shell;\s*/\*[^\n]*\*/)', r'\1' + insert_text, content)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Patched types.h")

def patch_gpu_pipeline_train():
    path = "src/gpu/gpu_pipeline_train.c"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "m->lccap_gate_enabled" not in content:
        content = content.replace(
            "const int ce_mask_on = rs_on || eqp_on;",
            "const int ce_mask_on = rs_on || eqp_on || m->lccap_gate_enabled;"
        )
        content = content.replace(
            "const int ce_mask_on_cpu = m->use_residual_shell || eqp_on_cpu || m->mbts_enabled;",
            "const int ce_mask_on_cpu = m->use_residual_shell || eqp_on_cpu || m->mbts_enabled || m->lccap_gate_enabled;"
        )
        content = content.replace(
            "const int ce_mask_on_cpu = m->use_residual_shell || eqp_on_cpu;",
            "const int ce_mask_on_cpu = m->use_residual_shell || eqp_on_cpu || m->lccap_gate_enabled;"
        )
        
        gpu_hook = """                if (!rs_mask[i]) {
                    for (int s = 0; s < N_SAMPLES; s++) row[s] = 0.0f;
                } else if (scale != 1.0f) {"""
                
        lccap_gpu = """
                /* R501-S61: Lookahead-Certified CE Anchor Pruning */
                if (m->lccap_gate_enabled) {
                    int seg_len = m->lccap_segment_length > 0 ? m->lccap_segment_length : 8;
                    float threshold = m->lccap_target_loss + m->lccap_epsilon_safe;
                    for (int b = 0; b < batch_size; b++) {
                        for (int t = 0; t < seq_len - 1; t += seg_len) {
                            int chunk_end = t + seg_len;
                            if (chunk_end > seq_len - 1) chunk_end = seq_len - 1;
                            int actual_len = chunk_end - t;
                            if (actual_len == 0) continue;
                            
                            m->lccap_segments_total++;
                            int lookahead_idx = b * (seq_len - 1) + (chunk_end - 1);
                            float lookahead_loss = loss_tmp[lookahead_idx];
                            
                            if (lookahead_loss <= threshold) {
                                int certified = 1;
                                for (int i = t; i < chunk_end; i++) {
                                    int idx = b * (seq_len - 1) + i;
                                    if (loss_tmp[idx] > threshold) {
                                        certified = 0;
                                        break;
                                    }
                                }
                                if (certified) {
                                    m->lccap_segments_pruned++;
                                    for (int i = t; i < chunk_end; i++) {
                                        int idx = b * (seq_len - 1) + i;
                                        rs_mask[idx] = 0;
                                    }
                                }
                            }
                        }
                    }
                    if (m->step > 0 && m->step % 100 == 0 && m->lccap_segments_total > 0) {
                        float pruned_frac = (float)((double)m->lccap_segments_pruned / (double)m->lccap_segments_total);
                        printf("[LCCAP] step=%u pruned_segments=%.6f\\n", m->step, pruned_frac);
                    }
                }
"""
        content = content.replace("                for (int i = 0; i < n_valid; i++) {", lccap_gpu + "\n                for (int i = 0; i < n_valid; i++) {")

        cpu_hook_find = """                for (int i = 0; i < n_valid_cpu; i++) rs_mask[i] = 1;"""
        content = content.replace(cpu_hook_find, cpu_hook_find + "\n" + lccap_gpu.replace("loss_tmp", "loss_tmp_cpu"))

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Patched gpu_pipeline_train.c")

def patch_train_c():
    path = "src/train.c"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "--lccap" not in content:
        insert_text = """
    } else if (strcmp(argv[i], "--lccap") == 0) {
        m->lccap_gate_enabled = 1;
        m->lccap_segment_length = 8;
        m->lccap_target_loss = 0.18f;
        m->lccap_epsilon_safe = 0.025f;
        printf("[LCCAP] enabled\\n");
"""
        content = re.sub(r'(} else if \(strcmp\(argv\[i\], "--use-residual-shell"\) == 0\) \{)', insert_text + r'\1', content)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Patched train.c")

if __name__ == "__main__":
    patch_types_h()
    patch_gpu_pipeline_train()
    patch_train_c()
