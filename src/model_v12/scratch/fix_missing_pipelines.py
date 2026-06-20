import sys

file_path = "src/runtime/vulkan_backend_impl.c"
with open(file_path, "r") as f:
    content = f.read()

missing_pipelines = [
    "vk_create_factored_adam_update_pipeline",
    "vk_create_factored_adam_moment_pipeline",
    "vk_create_factored_adam_reduce_pipeline",
    "vk_create_grad_scale_pipeline",
    "vk_create_adam_dense_pipeline",
    "vk_create_residual_scatter_add_pipeline",
    "vk_create_pc_negate_gather_pipeline",
    "vk_create_residual_sub_pipeline",
    "vk_create_pc_gather_prev_pipeline",
    "vk_create_mtp_loss_pipeline",
    "vk_create_broadcast_fill_pipeline",
    "vk_create_online_tssi_score_update_pipeline",
    "vk_create_welford_update_pipeline",
    "vk_create_direct_b_backward_pipeline",
    "vk_create_a_gate_recompute_pipeline"
]

injection = ""
for func in missing_pipelines:
    pipe_name = func.replace("vk_create_", "").replace("_pipeline", "")
    injection += f"    if ({func}(ctx) != 0) {{ fprintf(stderr, \"[VK] failed to create {pipe_name} pipeline\\n\"); return -1; }}\n"

if "vk_create_a_gate_recompute_pipeline" not in content.split("ctx->initialized = 1;")[0]:
    content = content.replace("    ctx->initialized = 1;", injection + "\n    ctx->initialized = 1;")

with open(file_path, "w") as f:
    f.write(content)
