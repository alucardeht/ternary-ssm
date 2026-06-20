import sys

content = open('gpu_pipeline.c', 'r', encoding='utf-8').read()

old_1 = '''                gpu_update_groups += (uint32_t)gpu_factored_adam_step_ternary_matrix(
                    p,
                    p->opt_layers[k].gpu_dw_in,'''
new_1 = '''                float tau = V12_CRYSTALLIZATION_TAU;
                if (m->continuous_ternary_margin_enabled) {
                    tau = 1.0f + m->continuous_ternary_margin_c / (sqrtf(m->last_mean_v_w_in[k]) + 1e-8f);
                }
                gpu_update_groups += (uint32_t)gpu_factored_adam_step_ternary_matrix(
                    p,
                    p->opt_layers[k].gpu_dw_in,'''

content = content.replace(old_1, new_1)

old_2 = '''                    evidence_gate, evidence_mode, m->evidence_rate_boundary_eps);'''
new_2 = '''                    evidence_gate, evidence_mode, m->evidence_rate_boundary_eps,
                    tau, &((Model *)m)->last_mean_v_w_in[k]);'''
content = content.replace(old_2, new_2, 1)

old_3 = '''                gpu_update_groups += (uint32_t)gpu_factored_adam_step_ternary_matrix(
                    p,
                    p->opt_layers[k].gpu_dw_out,'''
new_3 = '''                float tau = V12_CRYSTALLIZATION_TAU;
                if (m->continuous_ternary_margin_enabled) {
                    tau = 1.0f + m->continuous_ternary_margin_c / (sqrtf(m->last_mean_v_w_out[k]) + 1e-8f);
                }
                gpu_update_groups += (uint32_t)gpu_factored_adam_step_ternary_matrix(
                    p,
                    p->opt_layers[k].gpu_dw_out,'''
content = content.replace(old_3, new_3)

old_4 = '''                    evidence_gate, evidence_mode, m->evidence_rate_boundary_eps);'''
new_4 = '''                    evidence_gate, evidence_mode, m->evidence_rate_boundary_eps,
                    tau, &((Model *)m)->last_mean_v_w_out[k]);'''
content = content.replace(old_4, new_4, 1)

old_5 = '''        gpu_update_groups += (uint32_t)gpu_factored_adam_step_ternary_matrix(
            p,
            p->ce_dlm_head,'''
new_5 = '''        float tau = V12_CRYSTALLIZATION_TAU;
        if (m->continuous_ternary_margin_enabled) {
            tau = 1.0f + m->continuous_ternary_margin_c / (sqrtf(m->last_mean_v_lm_head) + 1e-8f);
        }
        gpu_update_groups += (uint32_t)gpu_factored_adam_step_ternary_matrix(
            p,
            p->ce_dlm_head,'''
content = content.replace(old_5, new_5)

old_6 = '''            evidence_gate, evidence_mode, m->evidence_rate_boundary_eps);'''
new_6 = '''            evidence_gate, evidence_mode, m->evidence_rate_boundary_eps,
            tau, &((Model *)m)->last_mean_v_lm_head);'''
content = content.replace(old_6, new_6, 1)

old_7 = '''        gpu_update_groups += (uint32_t)gpu_factored_adam_step_ternary_matrix(
            p,
            p->opt_dw_pred,'''
new_7 = '''        float tau = V12_CRYSTALLIZATION_TAU;
        if (m->continuous_ternary_margin_enabled) {
            tau = 1.0f + m->continuous_ternary_margin_c / (sqrtf(m->last_mean_v_w_pred) + 1e-8f);
        }
        gpu_update_groups += (uint32_t)gpu_factored_adam_step_ternary_matrix(
            p,
            p->opt_dw_pred,'''
content = content.replace(old_7, new_7)

old_8 = '''            evidence_gate, evidence_mode, m->evidence_rate_boundary_eps);'''
new_8 = '''            evidence_gate, evidence_mode, m->evidence_rate_boundary_eps,
            tau, &((Model *)m)->last_mean_v_w_pred);'''
content = content.replace(old_8, new_8, 1)

open('gpu_pipeline.c', 'w', encoding='utf-8', newline='\n').write(content)
print("gpu_pipeline.c patched")
