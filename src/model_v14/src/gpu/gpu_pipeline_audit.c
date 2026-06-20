#include "gpu_pipeline_internal.h"

#ifdef V12_HAS_VULKAN

int r400_obs_trace_enabled(void) {
    if (!g_r400_obs_trace_checked) {
        const char *flag = getenv(R400_OBS_TRACE_ENV);
        g_r400_obs_trace_enabled = env_flag_enabled(flag);
        g_r400_obs_trace_checked = 1;
    }
    return g_r400_obs_trace_enabled;
}

int r400_obs_trace_mkdir_single(const char *path) {
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST) return 0;
#else
    if (mkdir(path, 0777) == 0 || errno == EEXIST) return 0;
#endif
    return -1;
}

int r400_obs_trace_ensure_parent_dirs(const char *path) {
    char scratch[1024];
    size_t len;
    size_t i;

    if (!path || !path[0]) return -1;
    len = strlen(path);
    if (len >= sizeof(scratch)) return -1;
    memcpy(scratch, path, len + 1);
    for (i = 0; i < len; i++) {
        if (scratch[i] != '/' && scratch[i] != '\\') continue;
        if (i == 0) continue;
        if (i == 2 && scratch[1] == ':') continue;
        scratch[i] = '\0';
        if (r400_obs_trace_mkdir_single(scratch) != 0) return -1;
        scratch[i] = path[i];
    }
    return 0;
}

int r400_obs_trace_truncate_file(FILE *fp, long offset) {
    if (!fp || offset < 0) return -1;
    fflush(fp);
#ifdef _WIN32
    return (_chsize_s(_fileno(fp), (size_t)offset) == 0) ? 0 : -1;
#else
    return (ftruncate(fileno(fp), (off_t)offset) == 0) ? 0 : -1;
#endif
}

void r400_obs_trace_emit_non_identifying(R400ObsTraceConfig *cfg,
                                                const char *reason,
                                                int step,
                                                int layer,
                                                const char *missing_object) {
    (void)cfg;
    fprintf(stderr,
            "[R400_OBS_TRACE] non_identifying reason=%s step=%d layer=%d missing_object=%s\n",
            reason ? reason : "unknown",
            step,
            layer,
            missing_object ? missing_object : "none");
    fflush(stderr);
}

void r400_obs_trace_init_config(R400ObsTraceConfig *cfg,
                                       int batch_size,
                                       int n_layers) {
    const char *path;
    const char *max_steps_env;
    const char *max_records_env;
    const char *layer_env;
    const char *sample_env;
    const char *horizon_env;

    if (!cfg || cfg->initialized) return;

    memset(cfg, 0, sizeof(*cfg));
    cfg->enabled = r400_obs_trace_enabled();
    cfg->initialized = 1;
    cfg->max_steps = 2;
    cfg->max_records = 64ULL;
    cfg->layer_filter = -1;
    cfg->sample_count = 128;
    cfg->horizon = 4;
    cfg->batch_size = batch_size;
    cfg->n_layers = n_layers;
    cfg->summary_offset = -1L;
    snprintf(cfg->run_id, sizeof(cfg->run_id), "r400-obs-%p", (void *)cfg);

    path = getenv(R400_OBS_TRACE_PATH_ENV);
    max_steps_env = getenv(R400_OBS_TRACE_MAX_STEPS_ENV);
    max_records_env = getenv(R400_OBS_TRACE_MAX_RECORDS_ENV);
    layer_env = getenv(R400_OBS_TRACE_LAYER_ENV);
    sample_env = getenv(R400_OBS_TRACE_SAMPLE_ENV);
    horizon_env = getenv(R400_OBS_TRACE_HORIZON_ENV);

    if (max_steps_env && max_steps_env[0] != '\0') {
        cfg->max_steps = atoi(max_steps_env);
        if (cfg->max_steps <= 0) cfg->max_steps = 2;
    }
    if (max_records_env && max_records_env[0] != '\0') {
        unsigned long long parsed = strtoull(max_records_env, NULL, 10);
        if (parsed > 0ULL) cfg->max_records = (uint64_t)parsed;
    }
    if (layer_env && layer_env[0] != '\0') {
        cfg->layer_filter = atoi(layer_env);
    }
    if (sample_env && sample_env[0] != '\0') {
        cfg->sample_count = atoi(sample_env);
        if (cfg->sample_count <= 0) cfg->sample_count = 128;
    }
    if (horizon_env && horizon_env[0] != '\0') {
        cfg->horizon = atoi(horizon_env);
        if (cfg->horizon <= 0) cfg->horizon = 4;
    }
    if (path && path[0] != '\0') {
        snprintf(cfg->path, sizeof(cfg->path), "%s", path);
    } else {
        snprintf(cfg->path, sizeof(cfg->path), "%s", R400_OBS_TRACE_DEFAULT_PATH);
    }
}

void r400_obs_trace_log_activation_once(R400ObsTraceConfig *cfg) {
    char layer_buf[32];

    if (!cfg || !cfg->enabled || cfg->activation_line_emitted) return;
    if (cfg->layer_filter < 0) {
        snprintf(layer_buf, sizeof(layer_buf), "%s", "all");
    } else {
        snprintf(layer_buf, sizeof(layer_buf), "%d", cfg->layer_filter);
    }
    fprintf(stderr,
            "[R400_OBS_TRACE] active=1 mode=diagnostic-only source=real-gpu-training-observability "
            "path=%s max_steps=%d max_records=%llu layer=%s sample=%d horizon=%d "
            "no_training_behavior_change=1 schema=%s\n",
            cfg->path[0] ? cfg->path : R400_OBS_TRACE_DEFAULT_PATH,
            cfg->max_steps,
            (unsigned long long)cfg->max_records,
            layer_buf,
            cfg->sample_count,
            cfg->horizon,
            R400_OBS_TRACE_SCHEMA);
    fflush(stderr);
    cfg->activation_line_emitted = 1;
}

int r400_obs_trace_should_capture(R400ObsTraceConfig *cfg,
                                         int step,
                                         int layer) {
    if (!cfg || !cfg->enabled) return 0;
    if (step < 0 || step >= cfg->max_steps) return 0;
    if (cfg->records_written >= cfg->max_records) return 0;
    if (cfg->layer_filter >= cfg->n_layers && !cfg->config_error_emitted) {
        r400_obs_trace_emit_non_identifying(cfg, "layer_out_of_range", step, cfg->layer_filter, "none");
        cfg->config_error_emitted = 1;
    }
    if (cfg->layer_filter >= cfg->n_layers) return 0;
    if (cfg->layer_filter >= 0 && layer >= 0 && layer != cfg->layer_filter) return 0;
    return 1;
}

FILE *r400_obs_trace_open_file(R400ObsTraceConfig *cfg) {
    FILE *fp;

    if (!cfg || !cfg->enabled) return NULL;
    if (r400_obs_trace_ensure_parent_dirs(cfg->path) != 0) {
        if (!cfg->write_error_emitted) {
            r400_obs_trace_emit_non_identifying(cfg, "path_create_failed", -1, -1, "none");
            cfg->write_error_emitted = 1;
        }
        return NULL;
    }

    fp = fopen(cfg->path, cfg->meta_written ? "r+b" : "w");
    if (!fp) {
        if (!cfg->write_error_emitted) {
            r400_obs_trace_emit_non_identifying(cfg, "path_open_failed", -1, -1, "none");
            cfg->write_error_emitted = 1;
        }
        return NULL;
    }

    if (cfg->meta_written) {
        if (cfg->summary_written && cfg->summary_offset >= 0) {
                if (r400_obs_trace_truncate_file(fp, cfg->summary_offset) != 0) {
                fclose(fp);
                if (!cfg->write_error_emitted) {
                    r400_obs_trace_emit_non_identifying(cfg, "summary_truncate_failed", -1, -1, "none");
                    cfg->write_error_emitted = 1;
                }
                return NULL;
            }
        }
        if (fseek(fp, 0L, SEEK_END) != 0) {
            fclose(fp);
            if (!cfg->write_error_emitted) {
                r400_obs_trace_emit_non_identifying(cfg, "path_seek_failed", -1, -1, "none");
                cfg->write_error_emitted = 1;
            }
            return NULL;
        }
    }

    return fp;
}

int r400_obs_trace_write_meta(FILE *fp, const R400ObsTraceConfig *cfg) {
    if (!fp || !cfg) return -1;
    if (fprintf(fp,
                "{\"kind\":\"meta\",\"schema\":\"%s\",\"mode\":\"diagnostic-only\","
                "\"source\":\"real-gpu-training-observability\",\"no_training_behavior_change\":true,"
                "\"activation_line_emitted\":true,\"max_steps\":%d,\"max_records\":%llu,"
                "\"sample\":%d,\"horizon_k\":%d,\"batch_size\":%d}\n",
                R400_OBS_TRACE_SCHEMA,
                cfg->max_steps,
                (unsigned long long)cfg->max_records,
                cfg->sample_count,
                cfg->horizon,
                cfg->batch_size) < 0) {
        return -1;
    }
    return 0;
}

int r400_obs_trace_group_index(R400ObsTraceConfig *cfg,
                                      const char *alignment_key) {
    if (!cfg || !alignment_key || !alignment_key[0]) return -1;
    for (int i = 0; i < cfg->group_count; i++) {
        if (strcmp(cfg->groups[i].alignment_key, alignment_key) == 0) return i;
    }
    if (cfg->group_count >= R400_OBS_TRACE_GROUP_CAP) return -1;
    snprintf(cfg->groups[cfg->group_count].alignment_key,
             sizeof(cfg->groups[cfg->group_count].alignment_key),
             "%s",
             alignment_key);
    cfg->groups[cfg->group_count].flags = 0u;
    cfg->group_count++;
    return cfg->group_count - 1;
}

void r400_obs_trace_mark_group(R400ObsTraceConfig *cfg,
                                      const char *alignment_key,
                                      const char *object_kind,
                                      int source_valid,
                                      double l2,
                                      uint64_t nonzero_count,
                                      const char *zero_reason) {
    unsigned flags = 0u;
    int group_index = r400_obs_trace_group_index(cfg, alignment_key);
    int live = source_valid && isfinite(l2) && l2 > 0.0 && nonzero_count > 0ULL;

    if (group_index < 0 || !object_kind) return;

    if (strcmp(object_kind, "state_current") == 0 && live) {
        flags |= R400_OBS_FLAG_STATE_LIVE;
    } else if (strcmp(object_kind, "transition_A") == 0 && live) {
        flags |= R400_OBS_FLAG_TRANSITION_LIVE;
    } else if ((strcmp(object_kind, "credit_dA_ps") == 0
                || strcmp(object_kind, "lm_transition_grad_sketch") == 0
                || strcmp(object_kind, "bootstrap_transition_grad_sketch") == 0)) {
        if (live) {
            flags |= R400_OBS_FLAG_OBJECTIVE_LIVE;
        } else if (!source_valid
                   || (zero_reason
                       && (strcmp(zero_reason, "objective_missing") == 0
                           || strcmp(zero_reason, "source_missing") == 0
                           || strcmp(zero_reason, "not_produced") == 0
                           || strcmp(zero_reason, "not_aligned") == 0))) {
            flags |= R400_OBS_FLAG_OBJECTIVE_ABSENT;
        }
    } else if (strcmp(object_kind, "boundary_shadow_pre") == 0) {
        flags |= source_valid ? R400_OBS_FLAG_BOUNDARY_PRE : R400_OBS_FLAG_BOUNDARY_ABSENT;
    } else if (strcmp(object_kind, "boundary_shadow_post") == 0) {
        flags |= source_valid ? R400_OBS_FLAG_BOUNDARY_POST : R400_OBS_FLAG_BOUNDARY_ABSENT;
    } else if (strcmp(object_kind, "ternary_sample_mask") == 0) {
        flags |= source_valid ? R400_OBS_FLAG_BOUNDARY_SAMPLE : R400_OBS_FLAG_BOUNDARY_ABSENT;
    } else if (strcmp(object_kind, "ternary_q_mask") == 0) {
        flags |= source_valid ? R400_OBS_FLAG_BOUNDARY_Q : R400_OBS_FLAG_BOUNDARY_ABSENT;
    }

    cfg->groups[group_index].flags |= flags;
}

void r400_obs_trace_compute_summary(const R400ObsTraceConfig *cfg,
                                           int *aligned_groups,
                                           int *live_state_groups,
                                           int *live_transition_groups,
                                           int *live_boundary_groups,
                                           int *live_objective_grad_groups,
                                           const char **decision,
                                           const char **reason) {
    int aligned = cfg ? cfg->group_count : 0;
    int states = 0;
    int transitions = 0;
    int boundaries = 0;
    int objectives = 0;
    int identifying_state_transition = 0;
    int identifying_boundary = 0;
    int identifying_objective = 0;

    if (aligned_groups) *aligned_groups = 0;
    if (live_state_groups) *live_state_groups = 0;
    if (live_transition_groups) *live_transition_groups = 0;
    if (live_boundary_groups) *live_boundary_groups = 0;
    if (live_objective_grad_groups) *live_objective_grad_groups = 0;
    if (decision) *decision = "NON_IDENTIFYING_REAL_PATH";
    if (reason) *reason = "no_aligned_groups";
    if (!cfg) return;

    for (int i = 0; i < cfg->group_count; i++) {
        const unsigned flags = cfg->groups[i].flags;
        if (flags & R400_OBS_FLAG_STATE_LIVE) states++;
        if (flags & R400_OBS_FLAG_TRANSITION_LIVE) transitions++;
        if (flags & (R400_OBS_FLAG_BOUNDARY_PRE
                   | R400_OBS_FLAG_BOUNDARY_POST
                   | R400_OBS_FLAG_BOUNDARY_SAMPLE
                   | R400_OBS_FLAG_BOUNDARY_Q)) {
            boundaries++;
        }
        if (flags & R400_OBS_FLAG_OBJECTIVE_LIVE) objectives++;
        if ((flags & R400_OBS_FLAG_STATE_LIVE) && (flags & R400_OBS_FLAG_TRANSITION_LIVE)) {
            identifying_state_transition = 1;
        }
        if ((flags & R400_OBS_FLAG_OBJECTIVE_LIVE) || (flags & R400_OBS_FLAG_OBJECTIVE_ABSENT)) {
            identifying_objective = 1;
        }
        if (((flags & R400_OBS_FLAG_BOUNDARY_PRE)
             && (flags & R400_OBS_FLAG_BOUNDARY_POST)
             && ((flags & R400_OBS_FLAG_BOUNDARY_SAMPLE) || (flags & R400_OBS_FLAG_BOUNDARY_Q)))
            || (flags & R400_OBS_FLAG_BOUNDARY_ABSENT)) {
            identifying_boundary = 1;
        }
    }

    if (aligned_groups) *aligned_groups = aligned;
    if (live_state_groups) *live_state_groups = states;
    if (live_transition_groups) *live_transition_groups = transitions;
    if (live_boundary_groups) *live_boundary_groups = boundaries;
    if (live_objective_grad_groups) *live_objective_grad_groups = objectives;

    if (aligned == 0) {
        if (reason) *reason = "no_aligned_groups";
        return;
    }
    if (!identifying_state_transition) {
        if (reason) *reason = "missing_live_state_or_transition";
        return;
    }
    if (!identifying_objective) {
        if (reason) *reason = "missing_credit_or_objective_grad";
        return;
    }
    if (!identifying_boundary) {
        if (reason) *reason = "missing_boundary_alignment";
        return;
    }

    if (decision) *decision = "PASS_IDENTIFYING_INFRASTRUCTURE";
    if (reason) *reason = "aligned_real_path_observability";
}

int r400_obs_trace_write_summary(FILE *fp, R400ObsTraceConfig *cfg) {
    int aligned_groups;
    int live_state_groups;
    int live_transition_groups;
    int live_boundary_groups;
    int live_objective_grad_groups;
    const char *decision;
    const char *reason;

    if (!fp || !cfg) return -1;

    r400_obs_trace_compute_summary(cfg,
                                   &aligned_groups,
                                   &live_state_groups,
                                   &live_transition_groups,
                                   &live_boundary_groups,
                                   &live_objective_grad_groups,
                                   &decision,
                                   &reason);
    fprintf(stderr,
            "[R400_OBS_TRACE] summary records=%llu aligned_groups=%d live_state_groups=%d "
            "live_transition_groups=%d live_boundary_groups=%d live_objective_grad_groups=%d "
            "decision=%s reason=%s path=%s\n",
            (unsigned long long)cfg->records_written,
            aligned_groups,
            live_state_groups,
            live_transition_groups,
            live_boundary_groups,
            live_objective_grad_groups,
            decision,
            reason,
            cfg->path[0] ? cfg->path : R400_OBS_TRACE_DEFAULT_PATH);
    fflush(stderr);

    if (fprintf(fp,
                "{\"kind\":\"summary\",\"schema\":\"%s\",\"records\":%llu,"
                "\"aligned_groups\":%d,\"live_state_groups\":%d,\"live_transition_groups\":%d,"
                "\"live_boundary_groups\":%d,\"live_objective_grad_groups\":%d,"
                "\"decision\":\"%s\",\"reason\":\"%s\"}\n",
                R400_OBS_TRACE_SCHEMA,
                (unsigned long long)cfg->records_written,
                aligned_groups,
                live_state_groups,
                live_transition_groups,
                live_boundary_groups,
                live_objective_grad_groups,
                decision,
                reason) < 0) {
        return -1;
    }
    cfg->summary_written = 1;
    return 0;
}

int r400_obs_trace_append_summary(FILE *fp, R400ObsTraceConfig *cfg) {
    if (!fp || !cfg) return -1;
    cfg->summary_offset = ftell(fp);
    if (cfg->summary_offset < 0) return -1;
    if (r400_obs_trace_write_summary(fp, cfg) != 0) return -1;
    fflush(fp);
    return 0;
}

void r400_obs_trace_summarize_float_values(const float *values,
                                                  int count,
                                                  double *l2,
                                                  uint64_t *nonzero_count,
                                                  uint64_t *finite_count,
                                                  const char **zero_reason) {
    double sum_sq = 0.0;
    uint64_t nonzero = 0ULL;
    uint64_t finite = 0ULL;

    for (int i = 0; i < count; i++) {
        const float value = values ? values[i] : 0.0f;
        if (!isfinite(value)) continue;
        sum_sq += (double)value * (double)value;
        finite++;
        if (fabsf(value) > 0.0f) nonzero++;
    }

    if (l2) *l2 = sum_sq;
    if (nonzero_count) *nonzero_count = nonzero;
    if (finite_count) *finite_count = finite;
    if (zero_reason) {
        if (count <= 0) *zero_reason = "source_missing";
        else if (finite == 0ULL) *zero_reason = "source_missing";
        else if (nonzero == 0ULL) *zero_reason = "all_zero";
        else *zero_reason = "not_zero";
    }
}

void r400_obs_trace_summarize_double_values(const double *values,
                                                   int count,
                                                   double *l2,
                                                   uint64_t *nonzero_count,
                                                   uint64_t *finite_count,
                                                   const char **zero_reason) {
    double sum_sq = 0.0;
    uint64_t nonzero = 0ULL;
    uint64_t finite = 0ULL;

    for (int i = 0; i < count; i++) {
        const double value = values ? values[i] : 0.0;
        if (!isfinite(value)) continue;
        sum_sq += value * value;
        finite++;
        if (fabs(value) > 0.0) nonzero++;
    }

    if (l2) *l2 = sum_sq;
    if (nonzero_count) *nonzero_count = nonzero;
    if (finite_count) *finite_count = finite;
    if (zero_reason) {
        if (count <= 0) *zero_reason = "source_missing";
        else if (finite == 0ULL) *zero_reason = "source_missing";
        else if (nonzero == 0ULL) *zero_reason = "all_zero";
        else *zero_reason = "not_zero";
    }
}

int r400_obs_trace_write_record(FILE *fp,
                                       R400ObsTraceConfig *cfg,
                                       int step,
                                       int microbatch,
                                       int layer,
                                       int seq_index,
                                       int horizon_k,
                                       const char *matrix_family,
                                       const char *objective_kind,
                                       int coord_index,
                                       const char *alignment_key,
                                       const char *object_kind,
                                       const char *producer_stage,
                                       const char *consumer_stage,
                                       const char *source_name,
                                       int source_valid,
                                       const char *shape,
                                       int sample_count,
                                       double l2,
                                       uint64_t nonzero_count,
                                       uint64_t finite_count,
                                       const char *zero_reason) {
    if (!fp || !cfg || !alignment_key || !object_kind) return -1;
    if (cfg->records_written >= cfg->max_records) return 0;
    if (!isfinite(l2)) return -1;
    if (fprintf(fp,
                "{\"kind\":\"record\",\"schema\":\"%s\",\"run_id\":\"%s\",\"step\":%d,"
                "\"microbatch\":%d,\"layer\":%d,\"seq_index\":%d,\"horizon_k\":%d,"
                "\"matrix_family\":\"%s\",\"objective_kind\":\"%s\",\"coord_index\":%d,"
                "\"alignment_key\":\"%s\",\"object_kind\":\"%s\",\"producer_stage\":\"%s\","
                "\"consumer_stage\":\"%s\",\"source_name\":\"%s\",\"source_valid\":%s,"
                "\"shape\":\"%s\",\"sample_count\":%d,\"l2\":%.9g,\"nonzero_count\":%llu,"
                "\"finite_count\":%llu,\"zero_reason\":\"%s\"}\n",
                R400_OBS_TRACE_SCHEMA,
                cfg->run_id,
                step,
                microbatch,
                layer,
                seq_index,
                horizon_k,
                matrix_family ? matrix_family : "none",
                objective_kind ? objective_kind : "none",
                coord_index,
                alignment_key,
                object_kind,
                producer_stage ? producer_stage : "unknown",
                consumer_stage ? consumer_stage : "r400_obs_trace",
                source_name ? source_name : "unknown",
                source_valid ? "true" : "false",
                shape ? shape : "unknown",
                sample_count,
                l2,
                (unsigned long long)nonzero_count,
                (unsigned long long)finite_count,
                zero_reason ? zero_reason : "unknown") < 0) {
        return -1;
    }

    cfg->records_written++;
    r400_obs_trace_mark_group(cfg,
                              alignment_key,
                              object_kind,
                              source_valid,
                              l2,
                              nonzero_count,
                              zero_reason);
    fprintf(stderr,
            "[R400_OBS_TRACE] captured step=%d layer=%d object=%s source=%s l2=%.9g "
            "nonzero=%llu alignment_key=%s path=%s\n",
            step,
            layer,
            object_kind,
            source_name ? source_name : "unknown",
            l2,
            (unsigned long long)nonzero_count,
            alignment_key,
            cfg->path[0] ? cfg->path : R400_OBS_TRACE_DEFAULT_PATH);
    fflush(stderr);
    return 0;
}

int r400_obs_trace_write_absence_record(FILE *fp,
                                               R400ObsTraceConfig *cfg,
                                               int step,
                                               int microbatch,
                                               int layer,
                                               int seq_index,
                                               int horizon_k,
                                               const char *matrix_family,
                                               const char *objective_kind,
                                               int coord_index,
                                               const char *alignment_key,
                                               const char *object_kind,
                                               const char *producer_stage,
                                               const char *source_name,
                                               const char *shape,
                                               const char *zero_reason) {
    return r400_obs_trace_write_record(fp,
                                       cfg,
                                       step,
                                       microbatch,
                                       layer,
                                       seq_index,
                                       horizon_k,
                                       matrix_family,
                                       objective_kind,
                                       coord_index,
                                       alignment_key,
                                       object_kind,
                                       producer_stage,
                                       "r400_obs_trace",
                                       source_name,
                                       0,
                                       shape,
                                       0,
                                       0.0,
                                       0ULL,
                                       0ULL,
                                       zero_reason);
}

int r400_obs_trace_capture_training(GpuPipeline *p,
                                           VulkanContext *ctx,
                                           R400ObsTraceConfig *cfg,
                                           const SSMLayer *layer_weights,
                                           int step,
                                           int layer,
                                           int microbatch,
                                           const float *dA_ps,
                                           int seq_len,
                                           int ds,
                                           int batch_size);

static int r400_obs_trace_capture_boundary_pair(FILE *fp,
                                                R400ObsTraceConfig *cfg,
                                                int step,
                                                int layer,
                                                const char *matrix_family,
                                                int coord_index,
                                                float grad_before,
                                                float shadow_before,
                                                float sampled_before,
                                                float shadow_after,
                                                float sampled_after,
                                                float ema_threshold) {
    char alignment_key[160];
    double pre_values[3];
    double post_values[1];
    double sample_values[2];
    double l2 = 0.0;
    uint64_t nonzero_count = 0ULL;
    uint64_t finite_count = 0ULL;
    const char *zero_reason = "source_missing";

    if (!fp || !cfg) return -1;

    snprintf(alignment_key,
             sizeof(alignment_key),
             "step=%d|mb=0|layer=%d|seq=0|h=%d|family=%s|obj=boundary|coord=%d",
             step,
             layer,
             cfg->horizon,
             matrix_family ? matrix_family : "A",
             coord_index);

    pre_values[0] = (double)grad_before;
    pre_values[1] = (double)ema_threshold;
    pre_values[2] = (double)shadow_before;
    r400_obs_trace_summarize_double_values(pre_values, 3, &l2, &nonzero_count, &finite_count, &zero_reason);
    if (r400_obs_trace_write_record(fp,
                                    cfg,
                                    step,
                                    0,
                                    layer,
                                    0,
                                    cfg->horizon,
                                    matrix_family,
                                    "boundary",
                                    coord_index,
                                    alignment_key,
                                    "boundary_shadow_pre",
                                    "gpu_optimizer_pre_update",
                                    "r400_obs_trace",
                                    "opt_layers.shadow_w_in+grad_accum_in",
                                    1,
                                    "grad_accum_before,ema_threshold,shadow_before",
                                    3,
                                    l2,
                                    nonzero_count,
                                    finite_count,
                                    zero_reason) != 0) {
        return -1;
    }

    post_values[0] = (double)shadow_after;
    r400_obs_trace_summarize_double_values(post_values, 1, &l2, &nonzero_count, &finite_count, &zero_reason);
    if (r400_obs_trace_write_record(fp,
                                    cfg,
                                    step,
                                    0,
                                    layer,
                                    0,
                                    cfg->horizon,
                                    matrix_family,
                                    "boundary",
                                    coord_index,
                                    alignment_key,
                                    "boundary_shadow_post",
                                    "gpu_optimizer_post_update",
                                    "r400_obs_trace",
                                    "opt_layers.shadow_w_in",
                                    1,
                                    "shadow_after",
                                    1,
                                    l2,
                                    nonzero_count,
                                    finite_count,
                                    zero_reason) != 0) {
        return -1;
    }

    sample_values[0] = (double)sampled_before;
    sample_values[1] = (double)sampled_after;
    r400_obs_trace_summarize_double_values(sample_values, 2, &l2, &nonzero_count, &finite_count, &zero_reason);
    if (r400_obs_trace_write_record(fp,
                                    cfg,
                                    step,
                                    0,
                                    layer,
                                    0,
                                    cfg->horizon,
                                    matrix_family,
                                    "boundary",
                                    coord_index,
                                    alignment_key,
                                    "ternary_sample_mask",
                                    "gpu_optimizer_boundary",
                                    "r400_obs_trace",
                                    "opt_layers.sampled_w_in",
                                    1,
                                    "sampled_before,sampled_after",
                                    2,
                                    l2,
                                    nonzero_count,
                                    finite_count,
                                    zero_reason) != 0) {
        return -1;
    }

    if (r400_obs_trace_write_absence_record(fp,
                                            cfg,
                                            step,
                                            0,
                                            layer,
                                            0,
                                            cfg->horizon,
                                            matrix_family,
                                            "boundary",
                                            coord_index,
                                            alignment_key,
                                            "ternary_q_mask",
                                            "optimizer_boundary_source",
                                            "optimizer.q_mask_unavailable",
                                            "none",
                                            "source_missing") != 0) {
        return -1;
    }

    return 0;
}

int r400_obs_trace_capture_training(GpuPipeline *p,
                                           VulkanContext *ctx,
                                           R400ObsTraceConfig *cfg,
                                           const SSMLayer *layer_weights,
                                           int step,
                                           int layer,
                                           int microbatch,
                                           const float *dA_ps,
                                           int seq_len,
                                           int ds,
                                           int batch_size) {
    const size_t state_count = (size_t)batch_size * (size_t)seq_len * (size_t)ds;
    const size_t state_bytes = state_count * sizeof(float);
    int sample_count = 0;
    float *state = NULL;
    float *transition = NULL;
    float *adjoint = NULL;
    float *credit = NULL;
    float *a_gate = NULL;
    float *lm_grad = NULL;
    float *sample_buffer = NULL;
    FILE *fp = NULL;
    char alignment_key[160];
    double l2 = 0.0;
    uint64_t nonzero_count = 0ULL;
    uint64_t finite_count = 0ULL;
    const char *zero_reason = "source_missing";
    int status = -1;

    if (!p || !ctx || !cfg) return -1;
    if (!r400_obs_trace_should_capture(cfg, step, layer)) return 0;

    fp = r400_obs_trace_open_file(cfg);
    if (!fp) return -1;
    if (!cfg->meta_written) {
        if (r400_obs_trace_write_meta(fp, cfg) != 0) {
            r400_obs_trace_emit_non_identifying(cfg, "meta_write_failed", step, layer, "none");
            goto cleanup;
        }
        cfg->meta_written = 1;
    }

    sample_count = cfg->sample_count;
    if (sample_count > ds) sample_count = ds;
    if (sample_count <= 0) sample_count = ds;
    sample_buffer = (float *)malloc((size_t)sample_count * sizeof(float));
    if (!sample_buffer) {
        r400_obs_trace_emit_non_identifying(cfg, "alloc_failed", step, layer, "state_current");
        goto cleanup;
    }

    snprintf(alignment_key,
             sizeof(alignment_key),
             "step=%d|mb=%d|layer=%d|seq=0|h=%d|family=A|obj=none|coord=-1",
             step,
             microbatch,
             layer,
             cfg->horizon);

    if (layer < 0 || layer >= p->n_layers || !VK_BUFFER_VALID(p->layers[layer].fwd_h_cache)) {
        r400_obs_trace_emit_non_identifying(cfg, "state_source_unavailable", step, layer, "state_current");
        if (r400_obs_trace_write_absence_record(fp, cfg, step, microbatch, layer, 0, cfg->horizon,
                                                "A", "none", -1, alignment_key, "state_current",
                                                "forward_ssm", "layer.fwd_h_cache", "batch,seq,d_state",
                                                "source_missing") != 0) {
            goto cleanup;
        }
        if (r400_obs_trace_append_summary(fp, cfg) != 0) goto cleanup;
        status = 0;
        goto cleanup;
    }

    state = (float *)malloc(state_bytes);
    transition = VK_BUFFER_VALID(p->ssm_a) ? (float *)malloc(state_bytes) : NULL;
    adjoint = VK_BUFFER_VALID(p->bwd_d_y_ssm) ? (float *)malloc(state_bytes) : NULL;
    credit = dA_ps ? (float *)malloc(state_bytes) : NULL;
    a_gate = VK_BUFFER_VALID(p->layers[layer].fwd_a_buf) ? (float *)malloc(state_bytes) : NULL;
    lm_grad = (float *)calloc((size_t)ds, sizeof(float));
    if (!state || !lm_grad || (VK_BUFFER_VALID(p->ssm_a) && !transition)
        || (VK_BUFFER_VALID(p->bwd_d_y_ssm) && !adjoint)
        || (dA_ps && !credit)
        || (VK_BUFFER_VALID(p->layers[layer].fwd_a_buf) && !a_gate)) {
        r400_obs_trace_emit_non_identifying(cfg, "alloc_failed", step, layer, "state_current");
        goto cleanup;
    }

    vk_download(ctx, state, p->layers[layer].fwd_h_cache, state_bytes);
    if (transition) vk_download(ctx, transition, p->ssm_a, state_bytes);
    if (adjoint) vk_download(ctx, adjoint, p->bwd_d_y_ssm, state_bytes);
    if (a_gate) vk_download(ctx, a_gate, p->layers[layer].fwd_a_buf, state_bytes);
    if (credit && dA_ps) memcpy(credit, dA_ps, state_bytes);

    for (int i = 0; i < sample_count; i++) {
        int dim = (sample_count >= ds) ? i : (int)(((size_t)i * (size_t)ds) / (size_t)sample_count);
        if (dim >= ds) dim = ds - 1;
        sample_buffer[i] = state[dim];
    }
    r400_obs_trace_summarize_float_values(sample_buffer, sample_count, &l2, &nonzero_count, &finite_count, &zero_reason);
    if (r400_obs_trace_write_record(fp, cfg, step, microbatch, layer, 0, cfg->horizon,
                                    "A", "none", -1, alignment_key, "state_current",
                                    "forward_ssm", "r400_obs_trace", "layer.fwd_h_cache", 1,
                                    "batch,seq,d_state", sample_count, l2, nonzero_count,
                                    finite_count, zero_reason) != 0) {
        goto cleanup;
    }

    if (cfg->horizon < seq_len) {
        for (int i = 0; i < sample_count; i++) {
            int dim = (sample_count >= ds) ? i : (int)(((size_t)i * (size_t)ds) / (size_t)sample_count);
            if (dim >= ds) dim = ds - 1;
            sample_buffer[i] = state[(size_t)cfg->horizon * (size_t)ds + (size_t)dim];
        }
        r400_obs_trace_summarize_float_values(sample_buffer, sample_count, &l2, &nonzero_count, &finite_count, &zero_reason);
        if (r400_obs_trace_write_record(fp, cfg, step, microbatch, layer, cfg->horizon, cfg->horizon,
                                        "A", "none", -1, alignment_key, "state_future",
                                        "forward_ssm", "r400_obs_trace", "layer.fwd_h_cache", 1,
                                        "batch,seq,d_state", sample_count, l2, nonzero_count,
                                        finite_count, zero_reason) != 0) {
            goto cleanup;
        }
    } else if (r400_obs_trace_write_absence_record(fp, cfg, step, microbatch, layer, -1, cfg->horizon,
                                                   "A", "none", -1, alignment_key, "state_future",
                                                   "forward_ssm", "layer.fwd_h_cache", "batch,seq,d_state",
                                                   "not_produced") != 0) {
        goto cleanup;
    }

    if (transition) {
        for (int i = 0; i < sample_count; i++) {
            int dim = (sample_count >= ds) ? i : (int)(((size_t)i * (size_t)ds) / (size_t)sample_count);
            if (dim >= ds) dim = ds - 1;
            sample_buffer[i] = transition[dim];
        }
        r400_obs_trace_summarize_float_values(sample_buffer, sample_count, &l2, &nonzero_count, &finite_count, &zero_reason);
        if (r400_obs_trace_write_record(fp, cfg, step, microbatch, layer, 0, cfg->horizon,
                                        "A", "none", -1, alignment_key, "transition_A",
                                        "forward_ssm", "r400_obs_trace", "ssm_a", 1,
                                        "batch,seq,d_state", sample_count, l2, nonzero_count,
                                        finite_count, zero_reason) != 0) {
            goto cleanup;
        }
    } else if (r400_obs_trace_write_absence_record(fp, cfg, step, microbatch, layer, 0, cfg->horizon,
                                                   "A", "none", -1, alignment_key, "transition_A",
                                                   "forward_ssm", "ssm_a", "batch,seq,d_state",
                                                   "source_missing") != 0) {
        goto cleanup;
    }

    if (adjoint) {
        for (int i = 0; i < sample_count; i++) {
            int dim = (sample_count >= ds) ? i : (int)(((size_t)i * (size_t)ds) / (size_t)sample_count);
            if (dim >= ds) dim = ds - 1;
            sample_buffer[i] = adjoint[dim];
        }
        r400_obs_trace_summarize_float_values(sample_buffer, sample_count, &l2, &nonzero_count, &finite_count, &zero_reason);
        if (r400_obs_trace_write_record(fp, cfg, step, microbatch, layer, 0, cfg->horizon,
                                        "A", "none", -1, alignment_key, "adjoint",
                                        "backward_ssm", "r400_obs_trace", "bwd_d_y_ssm", 1,
                                        "batch,seq,d_state", sample_count, l2, nonzero_count,
                                        finite_count, zero_reason) != 0) {
            goto cleanup;
        }
    } else if (r400_obs_trace_write_absence_record(fp, cfg, step, microbatch, layer, 0, cfg->horizon,
                                                   "A", "none", -1, alignment_key, "adjoint",
                                                   "backward_ssm", "bwd_d_y_ssm", "batch,seq,d_state",
                                                   "not_produced") != 0) {
        goto cleanup;
    }

    if (credit) {
        for (int i = 0; i < sample_count; i++) {
            int dim = (sample_count >= ds) ? i : (int)(((size_t)i * (size_t)ds) / (size_t)sample_count);
            if (dim >= ds) dim = ds - 1;
            sample_buffer[i] = credit[dim];
        }
        r400_obs_trace_summarize_float_values(sample_buffer, sample_count, &l2, &nonzero_count, &finite_count, &zero_reason);
        if (r400_obs_trace_write_record(fp, cfg, step, microbatch, layer, 0, cfg->horizon,
                                        "A", "none", -1, alignment_key, "credit_dA_ps",
                                        "backward_ssm", "r400_obs_trace", "dA_ps", 1,
                                        "batch,seq,d_state", sample_count, l2, nonzero_count,
                                        finite_count, zero_reason) != 0) {
            goto cleanup;
        }
    } else if (r400_obs_trace_write_absence_record(fp, cfg, step, microbatch, layer, 0, cfg->horizon,
                                                   "A", "none", -1, alignment_key, "credit_dA_ps",
                                                   "backward_ssm", "dA_ps", "batch,seq,d_state",
                                                   "not_produced") != 0) {
        goto cleanup;
    }

    if (credit && a_gate && layer_weights && layer_weights->A) {
        for (int dim = 0; dim < ds; dim++) {
            double accum = 0.0;
            const double a_value = isfinite(layer_weights->A[dim]) ? (double)layer_weights->A[dim] : 0.0;
            if (fabs(a_value) <= R400_OBS_TRACE_EPS) continue;
            for (int t = 0; t < seq_len; t++) {
                const size_t index = (size_t)t * (size_t)ds + (size_t)dim;
                const double gate_ti = (double)a_gate[index] / a_value;
                const double da_eff = (double)credit[index];
                if (!isfinite(gate_ti) || !isfinite(da_eff)) continue;
                accum += da_eff * gate_ti;
            }
            lm_grad[dim] = (float)accum;
        }
        for (int i = 0; i < sample_count; i++) {
            int dim = (sample_count >= ds) ? i : (int)(((size_t)i * (size_t)ds) / (size_t)sample_count);
            if (dim >= ds) dim = ds - 1;
            sample_buffer[i] = lm_grad[dim];
        }
        r400_obs_trace_summarize_float_values(sample_buffer, sample_count, &l2, &nonzero_count, &finite_count, &zero_reason);
        if (r400_obs_trace_write_record(fp, cfg, step, microbatch, layer, 0, cfg->horizon,
                                        "A", "lm", -1, alignment_key, "lm_transition_grad_sketch",
                                        "backward_ssm", "r400_obs_trace", "dA_ps+a_gate/A", 1,
                                        "d_state", sample_count, l2, nonzero_count,
                                        finite_count, zero_reason) != 0) {
            goto cleanup;
        }
    } else if (r400_obs_trace_write_absence_record(fp, cfg, step, microbatch, layer, 0, cfg->horizon,
                                                   "A", "lm", -1, alignment_key, "lm_transition_grad_sketch",
                                                   "backward_ssm", "dA_ps+a_gate/A", "d_state",
                                                   "not_produced") != 0) {
        goto cleanup;
    }

    if (r400_obs_trace_write_absence_record(fp, cfg, step, microbatch, layer, 0, cfg->horizon,
                                            "A", "bootstrap", -1, alignment_key, "bootstrap_transition_grad_sketch",
                                            "bootstrap_objective", "bootstrap_transition_grad", "d_state",
                                            "objective_missing") != 0) {
        goto cleanup;
    }

    if (r400_obs_trace_append_summary(fp, cfg) != 0) goto cleanup;
    status = 0;

cleanup:
    if (status != 0 && cfg && !cfg->write_error_emitted) {
        r400_obs_trace_emit_non_identifying(cfg, "record_write_failed", step, layer, "none");
        cfg->write_error_emitted = 1;
    }
    if (fp) fclose(fp);
    free(state);
    free(transition);
    free(adjoint);
    free(credit);
    free(a_gate);
    free(lm_grad);
    free(sample_buffer);
    return status;
}

void certified_silent_span_gate_update(
    Model *m, const float *losses, int batch_size, int seq_len) {
    if (!m || !m->certified_silent_span_gate_enabled || !losses ||
        batch_size <= 0 || seq_len <= 1) {
        return;
    }

    const int n_valid = batch_size * (seq_len - 1);
    const uint32_t min_len = m->certified_silent_span_min_len > 0u
        ? m->certified_silent_span_min_len
        : 1u;
    const float max_loss = m->certified_silent_span_max_loss > 0.0f
        ? m->certified_silent_span_max_loss
        : 0.10f;
    uint64_t certified = 0;
    uint64_t spans = 0;
    double grad_bound_sum = 0.0;

    for (int b = 0; b < batch_size; b++) {
        const float *row = losses + (size_t)b * (size_t)(seq_len - 1);
        int run_len = 0;
        double run_grad_bound = 0.0;

        for (int t = 0; t < seq_len - 1; t++) {
            const float L = row[t];
            const int finite = isfinite(L) && L >= 0.0f;
            const int silent = finite && L <= max_loss;
            if (silent) {
                run_len++;
                const double py = exp(-(double)L);
                const double bound = 1.4142135623730951 * (1.0 - py);
                run_grad_bound += bound > 0.0 ? bound : 0.0;
            } else {
                if (run_len >= (int)min_len) {
                    certified += (uint64_t)run_len;
                    spans++;
                    grad_bound_sum += run_grad_bound;
                }
                run_len = 0;
                run_grad_bound = 0.0;
            }
        }
        if (run_len >= (int)min_len) {
            certified += (uint64_t)run_len;
            spans++;
            grad_bound_sum += run_grad_bound;
        }
    }

    const double total = (double)n_valid;
    const double fraction = total > 0.0 ? (double)certified / total : 0.0;
    const double mean_grad_bound = certified > 0
        ? grad_bound_sum / (double)certified
        : 0.0;
    const double priced =
        1.0 - fraction +
        (double)m->certified_silent_span_cert_cost +
        (double)m->certified_silent_span_repair_cost;

    m->certified_silent_span_gate_batches++;
    m->certified_silent_span_gate_positions += (uint64_t)n_valid;
    m->certified_silent_span_gate_tokens += certified;
    m->certified_silent_span_gate_spans += spans;
    m->certified_silent_span_gate_fraction = (float)fraction;
    m->certified_silent_span_gate_grad_bound = (float)mean_grad_bound;
    m->certified_silent_span_gate_priced_ratio = (float)priced;
}

void certified_artifact_boundary_gate_update(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len) {
    if (!m || !m->certified_artifact_boundary_gate_enabled || !tokens ||
        !m->certified_artifact_boundary_certified ||
        batch_size <= 0 || seq_len <= 1) {
        return;
    }

    uint64_t positions = 0;
    uint64_t routed = 0;
    uint64_t correct = 0;
    uint64_t incorrect = 0;
    uint64_t abstain = 0;

    for (int b = 0; b < batch_size; b++) {
        const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
        for (int t = 0; t + 1 < seq_len; t++) {
            positions++;
            const uint32_t predicted =
                certified_artifact_boundary_predict(m, tok_b[t]);
            if (predicted == UINT32_MAX) {
                abstain++;
                continue;
            }
            routed++;
            if (predicted == (uint32_t)tok_b[t + 1]) {
                correct++;
            } else {
                incorrect++;
            }
        }
    }

    m->certified_artifact_boundary_gate_batches++;
    m->certified_artifact_boundary_gate_positions += positions;
    m->certified_artifact_boundary_gate_routed += routed;
    m->certified_artifact_boundary_gate_correct += correct;
    m->certified_artifact_boundary_gate_incorrect += incorrect;
    m->certified_artifact_boundary_gate_abstain += abstain;

    const double total = (double)m->certified_artifact_boundary_gate_positions;
    const double routed_total =
        (double)m->certified_artifact_boundary_gate_routed;
    const double incorrect_total =
        (double)m->certified_artifact_boundary_gate_incorrect;
    const double b_k = total > 0.0 ? routed_total / total : 0.0;
    const double false_rate = routed_total > 0.0 ? incorrect_total / routed_total : 0.0;
    const double epsilon = (m->certified_artifact_boundary_epsilon > 0.0f &&
                            m->certified_artifact_boundary_epsilon < 1.0f)
        ? (double)m->certified_artifact_boundary_epsilon
        : 0.001;
    const double routed_ce = -log(1.0 - epsilon);
    const double ce_residual = b_k * routed_ce;
    const double priced =
        (1.0 - b_k) * (double)m->certified_artifact_boundary_kappa +
        (double)m->certified_artifact_boundary_overhead;

    m->certified_artifact_boundary_gate_b_k = (float)b_k;
    m->certified_artifact_boundary_gate_false_route_rate = (float)false_rate;
    m->certified_artifact_boundary_gate_ce_residual = (float)ce_residual;
    m->certified_artifact_boundary_gate_priced_ratio = (float)priced;
    if (m->certified_artifact_boundary_gate_routed +
        m->certified_artifact_boundary_gate_abstain !=
        m->certified_artifact_boundary_gate_positions) {
        m->certified_artifact_boundary_obligation_audit = 0;
    }
}

void residual_active_sampling_gate_update(
    Model *m, const float *losses, int batch_size, int seq_len) {
    if (!m || !m->residual_active_sampling_gate_enabled || !losses ||
        batch_size <= 0 || seq_len <= 1) {
        return;
    }

    const int n_valid = batch_size * (seq_len - 1);
    const double floor_score = m->residual_active_sampling_floor > 0.0f
        ? (double)m->residual_active_sampling_floor
        : 1.0e-6;
    const double target_fraction =
        (m->residual_active_sampling_target_fraction > 0.0f &&
         m->residual_active_sampling_target_fraction <= 1.0f)
            ? (double)m->residual_active_sampling_target_fraction
            : 0.50;

    double loss_sum = 0.0;
    double score_sum = 0.0;
    double score_sq_sum = 0.0;
    double min_score = 1.0e300;

    for (int i = 0; i < n_valid; i++) {
        const double L = (double)losses[i];
        const double finite_loss = (isfinite(L) && L >= 0.0) ? L : 0.0;
        const double score = finite_loss > floor_score ? finite_loss : floor_score;
        loss_sum += finite_loss;
        score_sum += score;
        score_sq_sum += score * score;
        if (score < min_score) min_score = score;
    }

    const double n = (double)n_valid;
    const double mean_loss = n > 0.0 ? loss_sum / n : 0.0;
    const double mean_score = n > 0.0 ? score_sum / n : 0.0;
    const double var_score = n > 0.0 ? score_sq_sum / n - mean_score * mean_score : 0.0;
    const double score_cv = mean_score > 0.0 && var_score > 0.0
        ? sqrt(var_score) / mean_score
        : 0.0;
    const double ess_fraction =
        (n > 0.0 && score_sq_sum > 0.0)
            ? (score_sum * score_sum) / (n * score_sq_sum)
            : 0.0;
    const double max_weight =
        (n > 0.0 && min_score > 0.0)
            ? score_sum / (n * min_score)
            : 0.0;
    const double variance_proxy =
        (target_fraction > 0.0 && ess_fraction > 0.0)
            ? 1.0 / (target_fraction * ess_fraction)
            : 0.0;

    m->residual_active_sampling_gate_batches++;
    m->residual_active_sampling_gate_positions += (uint64_t)n_valid;
    m->residual_active_sampling_gate_mean_loss = (float)mean_loss;
    m->residual_active_sampling_gate_score_cv = (float)score_cv;
    m->residual_active_sampling_gate_ess_fraction = (float)ess_fraction;
    m->residual_active_sampling_gate_max_weight = (float)max_weight;
    m->residual_active_sampling_gate_variance_proxy = (float)variance_proxy;
}

uint64_t exact_quotient_padding_mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

int exact_quotient_padding_choose_active_len(
    const Model *m, int seq_len, uint32_t step, uint64_t microbatch, int batch) {
    if (!m || seq_len <= 2) return seq_len;
    uint32_t min_len = m->exact_quotient_padding_min_active_len;
    if (min_len < 2u) min_len = 2u;
    if (min_len > (uint32_t)seq_len) min_len = (uint32_t)seq_len;

    const uint32_t span = (uint32_t)seq_len - min_len + 1u;
    if (span <= 1u) return seq_len;

    uint64_t x = m->exact_quotient_padding_seed;
    x ^= ((uint64_t)step + 0x9e3779b97f4a7c15ull);
    x ^= (microbatch + 0xbf58476d1ce4e5b9ull) << 1;
    x ^= ((uint64_t)(uint32_t)batch + 0x94d049bb133111ebull) << 7;
    x = exact_quotient_padding_mix64(x);
    return (int)(min_len + (uint32_t)(x % span));
}

int *exact_quotient_padding_build_active_lens(
    Model *m, int batch_size, int seq_len, const TrainState *state,
    int *active_targets_out, int *inactive_targets_out) {
    if (active_targets_out) *active_targets_out = batch_size * (seq_len - 1);
    if (inactive_targets_out) *inactive_targets_out = 0;
    if (!m || !m->exact_quotient_padding_enabled ||
        batch_size <= 0 || seq_len <= 1) {
        return NULL;
    }

    int *active_lens = (int *)malloc((size_t)batch_size * sizeof(int));
    if (!active_lens) {
        fprintf(stderr,
                "[EXACT_QUOTIENT_PADDING] allocation_failed=1 scope=active_lens fallback=full_ce\n");
        return NULL;
    }

    const uint32_t step = state ? state->step : (uint32_t)m->step;
    const uint64_t microbatch = m->exact_quotient_padding_batches;
    int active_targets = 0;
    for (int b = 0; b < batch_size; b++) {
        int active_len = exact_quotient_padding_choose_active_len(
            m, seq_len, step, microbatch, b);
        if (active_len < 2) active_len = 2;
        if (active_len > seq_len) active_len = seq_len;
        active_lens[b] = active_len;
        active_targets += active_len - 1;
    }

    const int total_targets = batch_size * (seq_len - 1);
    const int inactive_targets = total_targets - active_targets;
    if (active_targets_out) *active_targets_out = active_targets;
    if (inactive_targets_out) *inactive_targets_out = inactive_targets;
    return active_lens;
}

void exact_quotient_padding_record(
    Model *m, int total_targets, int active_targets) {
    if (!m || !m->exact_quotient_padding_enabled || total_targets <= 0) return;
    if (active_targets < 0) active_targets = 0;
    if (active_targets > total_targets) active_targets = total_targets;
    const int inactive_targets = total_targets - active_targets;

    m->exact_quotient_padding_batches++;
    m->exact_quotient_padding_total_targets += (uint64_t)total_targets;
    m->exact_quotient_padding_active_targets += (uint64_t)active_targets;
    m->exact_quotient_padding_inactive_targets += (uint64_t)inactive_targets;
    if (m->exact_quotient_padding_total_targets > 0u) {
        m->exact_quotient_padding_active_fraction =
            (float)((double)m->exact_quotient_padding_active_targets /
                    (double)m->exact_quotient_padding_total_targets);
    }
}

void prefix_frontier_gate_update(
    Model *m, const uint16_t *tokens, const float *d_logits,
    int batch_size, int seq_len, int n_samples) {
    if (!m || !m->prefix_frontier_gate_enabled || !tokens || !d_logits ||
        batch_size <= 0 || seq_len <= 2 || n_samples <= 0) {
        return;
    }

    uint64_t positions = 0;
    uint64_t supported = 0;
    uint64_t saved_terms = 0;
    uint64_t multiplicity_sum = 0;
    double total_energy = 0.0;
    double supported_energy = 0.0;
    const uint32_t min_count = m->prefix_frontier_min_count > 0u
        ? m->prefix_frontier_min_count
        : 2u;

    for (int b = 0; b < batch_size; b++) {
        const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
        for (int t = 1; t + 1 < seq_len; t++) {
            const int row = b * (seq_len - 1) + t;
            const float *row_logits = d_logits + (size_t)row * (size_t)n_samples;
            double energy = 0.0;
            for (int s = 0; s < n_samples; s++) {
                const double g = (double)row_logits[s];
                if (isfinite(g)) energy += g * g;
            }
            const uint32_t count = prefix_frontier_count(m, tok_b[t - 1], tok_b[t]);
            positions++;
            total_energy += energy;
            if (count >= min_count) {
                supported++;
                saved_terms += (uint64_t)(count - 1u);
                multiplicity_sum += (uint64_t)count;
                supported_energy += energy;
            }
        }
    }

    m->prefix_frontier_gate_batches++;
    m->prefix_frontier_gate_positions += positions;
    m->prefix_frontier_gate_supported += supported;
    m->prefix_frontier_gate_saved_terms += saved_terms;
    m->prefix_frontier_gate_multiplicity_sum += multiplicity_sum;
    m->prefix_frontier_gate_total_energy += total_energy;
    m->prefix_frontier_gate_supported_energy += supported_energy;

    const double total_positions = (double)m->prefix_frontier_gate_positions;
    const double supported_total = (double)m->prefix_frontier_gate_supported;
    const double saved_total = (double)m->prefix_frontier_gate_saved_terms;
    const double multiplicity_total = (double)m->prefix_frontier_gate_multiplicity_sum;
    const double reuse_fraction =
        total_positions > 0.0 ? fmin(1.0, saved_total / total_positions) : 0.0;
    const double energy_fraction =
        m->prefix_frontier_gate_total_energy > 0.0
            ? m->prefix_frontier_gate_supported_energy /
              m->prefix_frontier_gate_total_energy
            : 0.0;
    const double mean_multiplicity =
        supported_total > 0.0 ? multiplicity_total / supported_total : 0.0;
    double priced = 1.0 - reuse_fraction * energy_fraction
                  + (double)m->prefix_frontier_overhead
                  + (double)m->prefix_frontier_repair_cost;
    if (priced < 0.0) priced = 0.0;

    m->prefix_frontier_gate_reuse_fraction = (float)reuse_fraction;
    m->prefix_frontier_gate_energy_fraction = (float)energy_fraction;
    m->prefix_frontier_gate_mean_multiplicity = (float)mean_multiplicity;
    m->prefix_frontier_gate_priced_ratio = (float)priced;
}

void qgmf_gate_update(Model *m, const uint16_t *tokens, int batch_size, int seq_len) {
    if (!m || !m->qgmf_gate_enabled || !tokens || batch_size <= 1 || seq_len <= 1) return;
    
    uint64_t collisions = 0;
    for (int t = 1; t < seq_len; t++) {
        for (int b1 = 0; b1 < batch_size; b1++) {
            for (int b2 = b1 + 1; b2 < batch_size; b2++) {
                int match = 1;
                for (int i = 0; i <= t; i++) {
                    if (tokens[b1 * seq_len + i] != tokens[b2 * seq_len + i]) {
                        match = 0;
                        break;
                    }
                }
                if (match) {
                    collisions++;
                }
            }
        }
    }
    
    m->qgmf_gate_batches++;
    m->qgmf_gate_collisions += collisions;
    
    if (collisions > 0 || (m->qgmf_gate_batches % 100 == 0)) {
        printf("[F08 QGMF Gate] Quotient Group Memory Factorization - Prefix collisions: %llu (Batches: %llu)\n", 
            (unsigned long long)collisions, (unsigned long long)m->qgmf_gate_batches);
    }
}

void exact_window_multiset_gate_update(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len) {
    if (!m || !m->exact_window_multiset_gate_enabled || !tokens ||
        batch_size <= 0 || seq_len <= 1) {
        return;
    }

    uint64_t positions = 0;
    uint64_t supported = 0;
    uint64_t saved_terms = 0;
    uint64_t multiplicity_sum = 0;
    const uint32_t min_count = m->exact_window_multiset_min_count > 0u
        ? m->exact_window_multiset_min_count
        : 1u;

    for (int b = 0; b < batch_size; b++) {
        const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
        for (int t = 0; t + 1 < seq_len; t++) {
            const uint32_t count =
                exact_window_multiset_count_live(m, tok_b, seq_len, t);
            positions++;
            if (count >= min_count) {
                supported++;
                multiplicity_sum += (uint64_t)count;
                saved_terms += (uint64_t)(count - 1u);
            }
        }
    }

    m->exact_window_multiset_gate_batches++;
    m->exact_window_multiset_gate_positions += positions;
    m->exact_window_multiset_gate_supported += supported;
    m->exact_window_multiset_gate_saved_terms += saved_terms;
    m->exact_window_multiset_gate_multiplicity_sum += multiplicity_sum;

    const double total = (double)m->exact_window_multiset_gate_positions;
    const double supported_total = (double)m->exact_window_multiset_gate_supported;
    const double saved_total = (double)m->exact_window_multiset_gate_saved_terms;
    const double multiplicity_total = (double)m->exact_window_multiset_gate_multiplicity_sum;
    m->exact_window_multiset_gate_live_coverage =
        total > 0.0 ? (float)(supported_total / total) : 0.0f;
    m->exact_window_multiset_gate_saved_fraction =
        multiplicity_total > 0.0 ? (float)(saved_total / multiplicity_total) : 0.0f;
    m->exact_window_multiset_gate_mean_multiplicity =
        supported_total > 0.0 ? (float)(multiplicity_total / supported_total) : 0.0f;
    m->exact_window_multiset_gate_priced_ratio =
        m->exact_window_multiset_overhead +
        m->exact_window_multiset_gate_q_ratio *
            (1.0f + m->exact_window_multiset_noise_penalty);
}

void active_state_subspace_gate_update(
    Model *m, const float *grad, int batch_size, int seq_len, int d) {
    if (!m || !m->active_state_subspace_gate_enabled || !grad ||
        batch_size <= 0 || seq_len <= 0 || d <= 0) {
        return;
    }

    double dim_energy[V12_D_MODEL];
    for (int j = 0; j < d; j++) dim_energy[j] = 0.0;

    const int positions = batch_size * seq_len;
    double total_energy = 0.0;
    for (int i = 0; i < positions; i++) {
        const float *row = grad + (size_t)i * (size_t)d;
        for (int j = 0; j < d; j++) {
            const double g = (double)row[j];
            if (!isfinite(g)) continue;
            const double e = g * g;
            dim_energy[j] += e;
            total_energy += e;
        }
    }

    uint32_t k = m->active_state_subspace_k > 0u
        ? m->active_state_subspace_k
        : (uint32_t)d;
    if (k > (uint32_t)d) k = (uint32_t)d;

    double topk_energy = 0.0;
    for (uint32_t pick = 0; pick < k; pick++) {
        int best = -1;
        double best_energy = -1.0;
        for (int j = 0; j < d; j++) {
            if (dim_energy[j] > best_energy) {
                best_energy = dim_energy[j];
                best = j;
            }
        }
        if (best < 0 || best_energy <= 0.0) break;
        topk_energy += best_energy;
        dim_energy[best] = -1.0;
    }

    m->active_state_subspace_gate_batches++;
    m->active_state_subspace_gate_positions += (uint64_t)positions;
    m->active_state_subspace_gate_total_energy += total_energy;
    m->active_state_subspace_gate_topk_energy += topk_energy;

    const double k_over_d = d > 0 ? (double)k / (double)d : 1.0;
    const double energy_fraction =
        m->active_state_subspace_gate_total_energy > 0.0
            ? m->active_state_subspace_gate_topk_energy /
              m->active_state_subspace_gate_total_energy
            : 0.0;
    const double tail_regret = fmax(0.0, 1.0 - energy_fraction);
    double priced = 0.0;
    if (energy_fraction > 1.0e-30) {
        priced = (k_over_d + (double)m->active_state_subspace_overhead + tail_regret)
               / energy_fraction;
    }

    m->active_state_subspace_gate_k_over_d = (float)k_over_d;
    m->active_state_subspace_gate_energy_fraction = (float)energy_fraction;
    m->active_state_subspace_gate_tail_regret = (float)tail_regret;
    m->active_state_subspace_gate_priced_ratio = (float)priced;
}

void boundary_liability_ledger_gate_update(
    Model *m, const float *grad, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d) {
    if (!m || !m->boundary_liability_ledger_gate_enabled || !grad ||
        !boundary_mask || batch_size <= 0 || seq_len <= 0 || d <= 0) {
        return;
    }

    const int positions = batch_size * seq_len;
    double total_energy = 0.0;
    double boundary_energy = 0.0;
    uint64_t boundary_positions = 0;

    for (int i = 0; i < positions; i++) {
        const float *row = grad + (size_t)i * (size_t)d;
        double row_energy = 0.0;
        for (int j = 0; j < d; j++) {
            const double g = (double)row[j];
            if (isfinite(g)) row_energy += g * g;
        }
        total_energy += row_energy;
        if (boundary_mask[i]) {
            boundary_positions++;
            boundary_energy += row_energy;
        }
    }

    const double mean_energy =
        positions > 0 ? total_energy / (double)positions : 0.0;
    const double eps = (m->boundary_liability_null_epsilon > 0.0f)
        ? (double)m->boundary_liability_null_epsilon
        : 1.0e-6;
    const double null_threshold = mean_energy * eps;
    uint64_t null_positions = 0;
    uint64_t boundary_null_positions = 0;
    double null_energy = 0.0;
    double boundary_null_energy = 0.0;

    for (int i = 0; i < positions; i++) {
        const float *row = grad + (size_t)i * (size_t)d;
        double row_energy = 0.0;
        for (int j = 0; j < d; j++) {
            const double g = (double)row[j];
            if (isfinite(g)) row_energy += g * g;
        }
        if (row_energy <= null_threshold) {
            null_positions++;
            null_energy += row_energy;
            if (boundary_mask[i]) {
                boundary_null_positions++;
                boundary_null_energy += row_energy;
            }
        }
    }

    m->boundary_liability_gate_batches++;
    m->boundary_liability_gate_positions += (uint64_t)positions;
    m->boundary_liability_gate_boundary_positions += boundary_positions;
    m->boundary_liability_gate_null_positions += null_positions;
    m->boundary_liability_gate_boundary_null_positions += boundary_null_positions;
    m->boundary_liability_gate_total_energy += total_energy;
    m->boundary_liability_gate_boundary_energy += boundary_energy;
    m->boundary_liability_gate_null_energy += null_energy;
    m->boundary_liability_gate_boundary_null_energy += boundary_null_energy;

    const double total_positions = (double)m->boundary_liability_gate_positions;
    const double boundary_total =
        (double)m->boundary_liability_gate_boundary_positions;
    const double null_total = (double)m->boundary_liability_gate_null_positions;
    const double boundary_null_total =
        (double)m->boundary_liability_gate_boundary_null_positions;
    const double energy_total = m->boundary_liability_gate_total_energy;
    const double boundary_null_fraction =
        total_positions > 0.0 ? boundary_null_total / total_positions : 0.0;
    const double false_discharge_energy =
        energy_total > 0.0
            ? m->boundary_liability_gate_boundary_null_energy / energy_total
            : 0.0;
    const double priced_gain =
        boundary_null_fraction - false_discharge_energy
        - (double)m->boundary_liability_overhead;

    m->boundary_liability_gate_boundary_fraction =
        total_positions > 0.0 ? (float)(boundary_total / total_positions) : 0.0f;
    m->boundary_liability_gate_boundary_energy_fraction =
        energy_total > 0.0
            ? (float)(m->boundary_liability_gate_boundary_energy / energy_total)
            : 0.0f;
    m->boundary_liability_gate_null_fraction =
        total_positions > 0.0 ? (float)(null_total / total_positions) : 0.0f;
    m->boundary_liability_gate_boundary_null_fraction =
        (float)boundary_null_fraction;
    m->boundary_liability_gate_false_discharge_energy =
        (float)false_discharge_energy;
    m->boundary_liability_gate_priced_gain = (float)priced_gain;
    m->boundary_liability_gate_pass = (priced_gain > 0.10) ? 1 : 0;
}

void boundary_flux_sheaf_gate_update(
    Model *m, const float *transition_credit, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d_state) {
    if (!m || !m->boundary_flux_sheaf_gate_enabled || !transition_credit ||
        !boundary_mask || batch_size <= 0 || seq_len <= 0 || d_state <= 0) {
        return;
    }

    const int positions = batch_size * seq_len;
    double total_energy = 0.0;
    double boundary_energy = 0.0;
    double pre_boundary_energy = 0.0;
    double endpoint_energy = 0.0;
    uint64_t boundary_positions = 0;
    uint64_t pre_boundary_positions = 0;
    uint64_t endpoint_positions = 0;

    for (int b = 0; b < batch_size; b++) {
        const size_t batch_base = (size_t)b * (size_t)seq_len;
        for (int t = 0; t < seq_len; t++) {
            const size_t pos = batch_base + (size_t)t;
            const float *row = transition_credit + pos * (size_t)d_state;
            double row_energy = 0.0;
            for (int i = 0; i < d_state; i++) {
                const double g = (double)row[i];
                if (isfinite(g)) row_energy += g * g;
            }

            const int is_boundary = boundary_mask[pos] ? 1 : 0;
            const int is_pre_boundary =
                (t + 1 < seq_len && boundary_mask[pos + 1u]) ? 1 : 0;
            const int is_endpoint = is_boundary || is_pre_boundary;

            total_energy += row_energy;
            if (is_boundary) {
                boundary_positions++;
                boundary_energy += row_energy;
            }
            if (is_pre_boundary) {
                pre_boundary_positions++;
                pre_boundary_energy += row_energy;
            }
            if (is_endpoint) {
                endpoint_positions++;
                endpoint_energy += row_energy;
            }
        }
    }

    m->boundary_flux_gate_batches++;
    m->boundary_flux_gate_layers++;
    m->boundary_flux_gate_positions += (uint64_t)positions;
    m->boundary_flux_gate_boundary_positions += boundary_positions;
    m->boundary_flux_gate_pre_boundary_positions += pre_boundary_positions;
    m->boundary_flux_gate_endpoint_positions += endpoint_positions;
    m->boundary_flux_gate_total_energy += total_energy;
    m->boundary_flux_gate_boundary_energy += boundary_energy;
    m->boundary_flux_gate_pre_boundary_energy += pre_boundary_energy;
    m->boundary_flux_gate_endpoint_energy += endpoint_energy;

    const double total_positions = (double)m->boundary_flux_gate_positions;
    const double energy_total = m->boundary_flux_gate_total_energy;
    const double endpoint_fraction =
        total_positions > 0.0
            ? (double)m->boundary_flux_gate_endpoint_positions / total_positions
            : 0.0;
    const double energy_fraction =
        energy_total > 0.0
            ? m->boundary_flux_gate_endpoint_energy / energy_total
            : 0.0;
    const int identifying = (energy_total > 0.0) ? 1 : 0;
    const double kappa_est = fmax(0.0, 1.0 - energy_fraction);
    const double priced_gain =
        energy_fraction * (1.0 - kappa_est) -
        (double)m->boundary_flux_sheaf_overhead;

    m->boundary_flux_gate_boundary_fraction =
        total_positions > 0.0
            ? (float)((double)m->boundary_flux_gate_boundary_positions /
                      total_positions)
            : 0.0f;
    m->boundary_flux_gate_pre_boundary_fraction =
        total_positions > 0.0
            ? (float)((double)m->boundary_flux_gate_pre_boundary_positions /
                      total_positions)
            : 0.0f;
    m->boundary_flux_gate_endpoint_fraction = (float)endpoint_fraction;
    m->boundary_flux_gate_energy_fraction = (float)energy_fraction;
    m->boundary_flux_gate_kappa_est = (float)kappa_est;
    m->boundary_flux_gate_priced_gain = (float)priced_gain;
    m->boundary_flux_gate_identifying = identifying;
    m->boundary_flux_gate_pass =
        (identifying && priced_gain > 0.10) ? 1 : 0;
}

void boundary_cotangent_credit_gate_update(
    Model *m, const float *transition_credit, const uint8_t *boundary_mask,
    int batch_size, int seq_len, int d_state) {
    if (!m || !m->boundary_cotangent_credit_gate_enabled ||
        !transition_credit || !boundary_mask ||
        batch_size <= 0 || seq_len <= 0 || d_state <= 0) {
        return;
    }

    int *anchors = (int *)malloc((size_t)seq_len * sizeof(int));
    if (!anchors) return;

    const int positions = batch_size * seq_len;
    double exact_energy = 0.0;
    double recon_energy = 0.0;
    double dot = 0.0;
    double residual_energy = 0.0;
    uint64_t anchor_positions = 0;

    for (int b = 0; b < batch_size; b++) {
        const size_t batch_base = (size_t)b * (size_t)seq_len;
        int anchor_count = 0;

        for (int t = 0; t < seq_len; t++) {
            const size_t pos = batch_base + (size_t)t;
            const int is_boundary = boundary_mask[pos] ? 1 : 0;
            const int is_pre_boundary =
                (t + 1 < seq_len && boundary_mask[pos + 1u]) ? 1 : 0;
            if (t == 0 || t + 1 == seq_len || is_boundary || is_pre_boundary) {
                anchors[anchor_count++] = t;
            }
        }

        if (anchor_count <= 0) {
            anchors[anchor_count++] = 0;
        }
        anchor_positions += (uint64_t)anchor_count;

        if (anchor_count == 1) {
            const float *anchor_row =
                transition_credit + (batch_base + (size_t)anchors[0]) *
                (size_t)d_state;
            for (int t = 0; t < seq_len; t++) {
                const float *row =
                    transition_credit + (batch_base + (size_t)t) *
                    (size_t)d_state;
                for (int i = 0; i < d_state; i++) {
                    const double g = (double)row[i];
                    if (!isfinite(g)) continue;
                    const double r =
                        isfinite((double)anchor_row[i]) ?
                        (double)anchor_row[i] : 0.0;
                    const double e = g - r;
                    exact_energy += g * g;
                    recon_energy += r * r;
                    dot += g * r;
                    residual_energy += e * e;
                }
            }
            continue;
        }

        for (int a_idx = 0; a_idx + 1 < anchor_count; a_idx++) {
            const int left_t = anchors[a_idx];
            const int right_t = anchors[a_idx + 1];
            const int width = right_t - left_t;
            const int start_t = (a_idx == 0) ? left_t : left_t + 1;
            const float *left_row =
                transition_credit + (batch_base + (size_t)left_t) *
                (size_t)d_state;
            const float *right_row =
                transition_credit + (batch_base + (size_t)right_t) *
                (size_t)d_state;

            for (int t = start_t; t <= right_t; t++) {
                const double alpha =
                    width > 0 ? (double)(t - left_t) / (double)width : 0.0;
                const float *row =
                    transition_credit + (batch_base + (size_t)t) *
                    (size_t)d_state;
                for (int i = 0; i < d_state; i++) {
                    const double g = (double)row[i];
                    if (!isfinite(g)) continue;
                    const double l =
                        isfinite((double)left_row[i]) ?
                        (double)left_row[i] : 0.0;
                    const double r_anchor =
                        isfinite((double)right_row[i]) ?
                        (double)right_row[i] : 0.0;
                    const double r = l + alpha * (r_anchor - l);
                    const double e = g - r;
                    exact_energy += g * g;
                    recon_energy += r * r;
                    dot += g * r;
                    residual_energy += e * e;
                }
            }
        }
    }

    free(anchors);

    m->boundary_cotangent_gate_batches++;
    m->boundary_cotangent_gate_layers++;
    m->boundary_cotangent_gate_positions += (uint64_t)positions;
    m->boundary_cotangent_gate_anchor_positions += anchor_positions;
    m->boundary_cotangent_gate_exact_energy += exact_energy;
    m->boundary_cotangent_gate_recon_energy += recon_energy;
    m->boundary_cotangent_gate_dot += dot;
    m->boundary_cotangent_gate_residual_energy += residual_energy;

    const double total_positions =
        (double)m->boundary_cotangent_gate_positions;
    const double anchors_total =
        (double)m->boundary_cotangent_gate_anchor_positions;
    const double exact_total = m->boundary_cotangent_gate_exact_energy;
    const double recon_total = m->boundary_cotangent_gate_recon_energy;
    const double dot_total = m->boundary_cotangent_gate_dot;
    const double residual_total = m->boundary_cotangent_gate_residual_energy;
    const double anchor_fraction =
        total_positions > 0.0 ? anchors_total / total_positions : 1.0;
    const double residual_fraction =
        exact_total > 0.0 ? residual_total / exact_total : 1.0;
    const double energy_covered = fmax(0.0, 1.0 - residual_fraction);
    const double cosine_den = sqrt(exact_total * recon_total);
    const double cosine = cosine_den > 0.0 ? dot_total / cosine_den : 0.0;
    const double verify_fraction =
        m->boundary_cotangent_verify_fraction >= 0.0f
            ? (double)m->boundary_cotangent_verify_fraction
            : 1.0;
    const double overhead = (double)m->boundary_cotangent_overhead;
    const double priced_ratio =
        anchor_fraction + verify_fraction + overhead + residual_fraction;
    const double unverified_priced_ratio =
        anchor_fraction + overhead + residual_fraction;
    const double descent_per_paid_work =
        priced_ratio > 0.0 ? energy_covered / priced_ratio : 0.0;
    const double unverified_descent_per_paid_work =
        unverified_priced_ratio > 0.0
            ? energy_covered / unverified_priced_ratio
            : 0.0;
    const int identifying =
        (m->boundary_cotangent_gate_positions > 0u && exact_total > 0.0) ? 1 : 0;

    m->boundary_cotangent_gate_anchor_fraction = (float)anchor_fraction;
    m->boundary_cotangent_gate_saved_recurrent_fraction =
        (float)fmax(0.0, 1.0 - anchor_fraction);
    m->boundary_cotangent_gate_cosine = (float)cosine;
    m->boundary_cotangent_gate_residual_fraction = (float)residual_fraction;
    m->boundary_cotangent_gate_energy_covered = (float)energy_covered;
    m->boundary_cotangent_gate_basis_bytes_per_token =
        total_positions > 0.0
            ? (float)((anchors_total * (double)d_state * sizeof(float)) /
                      total_positions)
            : 0.0f;
    m->boundary_cotangent_gate_verification_fraction = (float)verify_fraction;
    m->boundary_cotangent_gate_priced_ratio = (float)priced_ratio;
    m->boundary_cotangent_gate_unverified_priced_ratio =
        (float)unverified_priced_ratio;
    m->boundary_cotangent_gate_descent_per_paid_work =
        (float)descent_per_paid_work;
    m->boundary_cotangent_gate_unverified_descent_per_paid_work =
        (float)unverified_descent_per_paid_work;
    m->boundary_cotangent_gate_identifying = identifying;
    m->boundary_cotangent_gate_pass =
        (identifying &&
         residual_fraction <= (double)m->boundary_cotangent_residual_budget &&
         priced_ratio < 0.90 &&
         descent_per_paid_work > 1.0) ? 1 : 0;
}

int variance_priced_credit_gate_update(
    Model *m, float *transition_credit, int batch_size,
    int seq_len, int d_state) {
    if (!m || !m->variance_priced_credit_gate_enabled ||
        !transition_credit || batch_size <= 0 || seq_len <= 0 ||
        d_state <= 0) {
        return 0;
    }

    const int items = batch_size * seq_len;
    float *norms = (float *)malloc((size_t)items * sizeof(float));
    double *probs = (double *)calloc((size_t)items, sizeof(double));
    uint8_t *sampled_mask = (uint8_t *)calloc((size_t)items, sizeof(uint8_t));
    double *exact_sum = (double *)calloc((size_t)d_state, sizeof(double));
    double *sampled_sum = (double *)calloc((size_t)d_state, sizeof(double));
    if (!norms || !probs || !sampled_mask || !exact_sum || !sampled_sum) {
        free(norms);
        free(probs);
        free(sampled_mask);
        free(exact_sum);
        free(sampled_sum);
        return 0;
    }

    double norm_sum = 0.0;
    double norm_sq_sum = 0.0;
    double max_norm = 0.0;
    int nonzero_items = 0;

    for (int pos = 0; pos < items; pos++) {
        const float *row = transition_credit + (size_t)pos * (size_t)d_state;
        double e = 0.0;
        for (int i = 0; i < d_state; i++) {
            const double g = (double)row[i];
            if (isfinite(g)) e += g * g;
        }
        const double n = sqrt(e);
        norms[pos] = (float)n;
        norm_sum += n;
        norm_sq_sum += n * n;
        if (n > max_norm) max_norm = n;
        if (n > 0.0) nonzero_items++;
    }

    double sample_fraction = (double)m->variance_priced_sample_fraction;
    if (sample_fraction < 0.0) sample_fraction = 0.0;
    if (sample_fraction > 1.0) sample_fraction = 1.0;
    const double target_samples = sample_fraction * (double)items;

    double lambda = 0.0;
    if (target_samples >= (double)nonzero_items) {
        lambda = INFINITY;
    } else if (target_samples > 0.0 && norm_sum > 0.0) {
        double lo = 0.0;
        double hi = 1.0;
        for (int grow = 0; grow < 32; grow++) {
            double s = 0.0;
            for (int pos = 0; pos < items; pos++) {
                const double n = (double)norms[pos];
                if (n > 0.0) s += fmin(1.0, hi * n);
            }
            if (s >= target_samples) break;
            hi *= 2.0;
        }
        for (int it = 0; it < 40; it++) {
            const double mid = 0.5 * (lo + hi);
            double s = 0.0;
            for (int pos = 0; pos < items; pos++) {
                const double n = (double)norms[pos];
                if (n > 0.0) s += fmin(1.0, mid * n);
            }
            if (s < target_samples) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lambda = hi;
    }

    uint64_t sampled_items = 0;
    const uint32_t gate_salt =
        (uint32_t)(m->variance_priced_gate_layers * 747796405u) ^
        (uint32_t)(m->variance_priced_gate_batches * 277803737u);

    for (int pos = 0; pos < items; pos++) {
        const float *row = transition_credit + (size_t)pos * (size_t)d_state;
        const double n = (double)norms[pos];
        double p_i = 0.0;
        if (n > 0.0) {
            p_i = isinf(lambda) ? 1.0 : fmin(1.0, lambda * n);
        }
        const uint32_t h =
            gate_salt ^ (uint32_t)(pos * 2891336453u + 0x9e3779b9u);
        const int sampled = (p_i >= 1.0 || variance_priced_hash01(h) < p_i);
        probs[pos] = p_i;
        sampled_mask[pos] = sampled ? 1u : 0u;
        if (sampled) sampled_items++;

        for (int i = 0; i < d_state; i++) {
            const double g = (double)row[i];
            if (!isfinite(g)) continue;
            exact_sum[i] += g;
            if (sampled && p_i > 0.0) {
                sampled_sum[i] += g / p_i;
            }
        }
    }

    double exact_energy = 0.0;
    double sampled_energy = 0.0;
    double dot = 0.0;
    double residual_energy = 0.0;
    for (int i = 0; i < d_state; i++) {
        const double g = exact_sum[i];
        const double s = sampled_sum[i];
        const double r = s - g;
        exact_energy += g * g;
        sampled_energy += s * s;
        dot += g * s;
        residual_energy += r * r;
    }

    const double local_sampled_fraction =
        items > 0 ? (double)sampled_items / (double)items : 0.0;
    const double local_alignment =
        exact_energy > 0.0 ? dot / exact_energy : 0.0;
    const double local_variance_proxy =
        exact_energy > 0.0 ? residual_energy / exact_energy : 1.0;
    const double priced_work =
        local_sampled_fraction +
        (double)m->variance_priced_score_fraction +
        (double)m->variance_priced_verify_fraction +
        (double)m->variance_priced_overhead;
    const double local_descent_proxy =
        fmax(0.0, local_alignment - local_variance_proxy);
    const double local_descent_per_paid_work =
        priced_work > 0.0 ? local_descent_proxy / priced_work : 0.0;
    const int local_identifying = (items > 0 && exact_energy > 0.0) ? 1 : 0;
    const double min_descent_per_paid =
        m->amortized_ce_martingale_min_descent_per_paid_work > 0.0f
            ? (double)m->amortized_ce_martingale_min_descent_per_paid_work
            : 1.111;
    const int should_apply =
        (m->amortized_ce_martingale_training_enabled &&
         local_identifying &&
         local_descent_per_paid_work >= min_descent_per_paid) ? 1 : 0;

    m->amortized_ce_martingale_last_descent_per_paid_work =
        (float)local_descent_per_paid_work;
    m->amortized_ce_martingale_last_sampled_fraction =
        (float)local_sampled_fraction;
    m->amortized_ce_martingale_last_applied = should_apply;
    if (m->amortized_ce_martingale_training_enabled) {
        m->amortized_ce_martingale_training_layers++;
        if (should_apply) {
            m->amortized_ce_martingale_applied_layers++;
        } else {
            m->amortized_ce_martingale_fallback_layers++;
        }
    }

    if (should_apply) {
        for (int pos = 0; pos < items; pos++) {
            float *row = transition_credit + (size_t)pos * (size_t)d_state;
            const double p_i = probs[pos];
            const int sampled = sampled_mask[pos] != 0u;
            const float scale = (sampled && p_i > 0.0)
                ? (float)(1.0 / p_i)
                : 0.0f;
            for (int i = 0; i < d_state; i++) {
                row[i] = isfinite((double)row[i]) ? row[i] * scale : 0.0f;
            }
        }
    }

    m->variance_priced_gate_batches++;
    m->variance_priced_gate_layers++;
    m->variance_priced_gate_items += (uint64_t)items;
    m->variance_priced_gate_sampled_items += sampled_items;
    m->variance_priced_gate_norm_sum += norm_sum;
    m->variance_priced_gate_norm_sq_sum += norm_sq_sum;
    m->variance_priced_gate_exact_energy += exact_energy;
    m->variance_priced_gate_sampled_energy += sampled_energy;
    m->variance_priced_gate_dot += dot;
    m->variance_priced_gate_residual_energy += residual_energy;
    if (max_norm > (double)m->variance_priced_gate_max_norm) {
        m->variance_priced_gate_max_norm = (float)max_norm;
    }

    const double total_items = (double)m->variance_priced_gate_items;
    const double sampled_total = (double)m->variance_priced_gate_sampled_items;
    const double norm_total = m->variance_priced_gate_norm_sum;
    const double norm_sq_total = m->variance_priced_gate_norm_sq_sum;
    const double exact_total = m->variance_priced_gate_exact_energy;
    const double dot_total = m->variance_priced_gate_dot;
    const double residual_total = m->variance_priced_gate_residual_energy;
    const double sampled_fraction =
        total_items > 0.0 ? sampled_total / total_items : 0.0;
    const double norm_ess_fraction =
        (total_items > 0.0 && norm_sq_total > 0.0)
            ? (norm_total * norm_total) / (total_items * norm_sq_total)
            : 0.0;
    const double alignment =
        exact_total > 0.0 ? dot_total / exact_total : 0.0;
    const double variance_proxy =
        exact_total > 0.0 ? residual_total / exact_total : 1.0;
    const double aggregate_priced_work =
        sampled_fraction +
        (double)m->variance_priced_score_fraction +
        (double)m->variance_priced_verify_fraction +
        (double)m->variance_priced_overhead;
    const double descent_proxy = fmax(0.0, alignment - variance_proxy);
    const double descent_per_paid_work =
        aggregate_priced_work > 0.0 ? descent_proxy / aggregate_priced_work : 0.0;
    const int identifying =
        (m->variance_priced_gate_items > 0u && exact_total > 0.0) ? 1 : 0;

    m->variance_priced_gate_sampled_fraction = (float)sampled_fraction;
    m->variance_priced_gate_norm_ess_fraction = (float)norm_ess_fraction;
    m->variance_priced_gate_mean_norm =
        total_items > 0.0 ? (float)(norm_total / total_items) : 0.0f;
    m->variance_priced_gate_alignment = (float)alignment;
    m->variance_priced_gate_variance_proxy = (float)variance_proxy;
    m->variance_priced_gate_priced_work = (float)aggregate_priced_work;
    m->variance_priced_gate_descent_proxy = (float)descent_proxy;
    m->variance_priced_gate_descent_per_paid_work =
        (float)descent_per_paid_work;
    m->variance_priced_gate_identifying = identifying;
    m->variance_priced_gate_pass =
        (identifying && descent_per_paid_work > 1.111) ? 1 : 0;

    free(norms);
    free(probs);
    free(sampled_mask);
    free(exact_sum);
    free(sampled_sum);
    return should_apply;
}

void fp_interface_gap_probe_capture_layer(
    GpuPipeline *p, Model *m, int step, int layer,
    int batch_size, int seq_len, int d) {
    if (!p || !m || !m->fp_interface_gap_probe_enabled ||
        layer < 0 || layer >= V12_N_LAYERS ||
        batch_size <= 0 || seq_len <= 0 || d <= 0 ||
        !VK_BUFFER_VALID(p->residual[layer]) ||
        !VK_BUFFER_VALID(p->residual[layer + 1]) ||
        !VK_BUFFER_VALID(p->bwd_d_h_in)) {
        return;
    }

    const size_t n = (size_t)batch_size * (size_t)seq_len * (size_t)d;
    const size_t bytes = n * sizeof(float);
    const int ds = p->d_state;
    const size_t y_n = (size_t)batch_size * (size_t)seq_len * (size_t)ds;
    const size_t y_bytes = y_n * sizeof(float);
    float *r_in = (float *)malloc(bytes);
    float *r_out = (float *)malloc(bytes);
    float *grad = (float *)malloc(bytes);
    float *x_proj = VK_BUFFER_VALID(p->layers[layer].fwd_x_proj)
        ? (float *)malloc(y_bytes)
        : NULL;
    float *x_cv = VK_BUFFER_VALID(p->layers[layer].fwd_x_cv)
        ? (float *)malloc(y_bytes)
        : NULL;
    float *b_eff = VK_BUFFER_VALID(p->layers[layer].fwd_b_eff)
        ? (float *)malloc(y_bytes)
        : NULL;
    float *h_cache = VK_BUFFER_VALID(p->layers[layer].fwd_h_cache)
        ? (float *)malloc(y_bytes)
        : NULL;
    float *y_ssm = VK_BUFFER_VALID(p->layers[layer].fwd_y_ssm)
        ? (float *)malloc(y_bytes)
        : NULL;
    if (!r_in || !r_out || !grad) {
        free(r_in);
        free(r_out);
        free(grad);
        free(x_proj);
        free(x_cv);
        free(b_eff);
        free(h_cache);
        free(y_ssm);
        return;
    }

    vk_download(p->ctx, r_in, p->residual[layer], bytes);
    vk_download(p->ctx, r_out, p->residual[layer + 1], bytes);
    vk_download(p->ctx, grad, p->bwd_d_h_in, bytes);
    if (x_proj) {
        vk_download(p->ctx, x_proj, p->layers[layer].fwd_x_proj, y_bytes);
    }
    if (x_cv) {
        vk_download(p->ctx, x_cv, p->layers[layer].fwd_x_cv, y_bytes);
    }
    if (b_eff) {
        vk_download(p->ctx, b_eff, p->layers[layer].fwd_b_eff, y_bytes);
    }
    if (h_cache) {
        vk_download(p->ctx, h_cache, p->layers[layer].fwd_h_cache, y_bytes);
    }
    if (y_ssm) {
        vk_download(p->ctx, y_ssm, p->layers[layer].fwd_y_ssm, y_bytes);
    }

    double input_norm2 = 0.0;
    double delta_norm2 = 0.0;
    double grad_norm2 = 0.0;
    double dot = 0.0;
    uint64_t finite_count = 0;
    for (size_t i = 0; i < n; i++) {
        const double in_v = (double)r_in[i];
        const double delta = (double)r_out[i] - in_v;
        const double g = (double)grad[i];
        if (!isfinite(in_v) || !isfinite(delta) || !isfinite(g)) continue;
        input_norm2 += in_v * in_v;
        delta_norm2 += delta * delta;
        grad_norm2 += g * g;
        dot += delta * g;
        finite_count++;
    }
    double y_norm2 = 0.0;
    double x_proj_norm2 = 0.0;
    double x_cv_norm2 = 0.0;
    double b_eff_norm2 = 0.0;
    double h_norm2 = 0.0;
    uint64_t y_finite_count = 0;
    if (y_ssm) {
        for (size_t i = 0; i < y_n; i++) {
            const double y = (double)y_ssm[i];
            if (!isfinite(y)) continue;
            y_norm2 += y * y;
            y_finite_count++;
        }
    }
    uint64_t x_proj_finite_count = 0;
    if (x_proj) {
        for (size_t i = 0; i < y_n; i++) {
            const double v = (double)x_proj[i];
            if (!isfinite(v)) continue;
            x_proj_norm2 += v * v;
            x_proj_finite_count++;
        }
    }
    uint64_t x_cv_finite_count = 0;
    if (x_cv) {
        for (size_t i = 0; i < y_n; i++) {
            const double v = (double)x_cv[i];
            if (!isfinite(v)) continue;
            x_cv_norm2 += v * v;
            x_cv_finite_count++;
        }
    }
    uint64_t b_eff_finite_count = 0;
    if (b_eff) {
        for (size_t i = 0; i < y_n; i++) {
            const double v = (double)b_eff[i];
            if (!isfinite(v)) continue;
            b_eff_norm2 += v * v;
            b_eff_finite_count++;
        }
    }
    uint64_t h_finite_count = 0;
    if (h_cache) {
        for (size_t i = 0; i < y_n; i++) {
            const double v = (double)h_cache[i];
            if (!isfinite(v)) continue;
            h_norm2 += v * v;
            h_finite_count++;
        }
    }

    const double denom = (double)(finite_count > 0 ? finite_count : 1);
    const double y_denom = (double)(y_finite_count > 0 ? y_finite_count : 1);
    const double input_rms = sqrt(input_norm2 / denom);
    const double delta_rms = sqrt(delta_norm2 / denom);
    const double y_ssm_rms = sqrt(y_norm2 / y_denom);
    const double x_proj_rms =
        sqrt(x_proj_norm2 / (double)(x_proj_finite_count > 0 ? x_proj_finite_count : 1));
    const double x_cv_rms =
        sqrt(x_cv_norm2 / (double)(x_cv_finite_count > 0 ? x_cv_finite_count : 1));
    const double b_eff_rms =
        sqrt(b_eff_norm2 / (double)(b_eff_finite_count > 0 ? b_eff_finite_count : 1));
    const double h_rms =
        sqrt(h_norm2 / (double)(h_finite_count > 0 ? h_finite_count : 1));
    const double grad_rms = sqrt(grad_norm2 / denom);
    const double effect_ratio =
        input_norm2 > 1.0e-30 ? sqrt(delta_norm2 / input_norm2) : 0.0;
    const double cosine =
        (delta_norm2 > 1.0e-30 && grad_norm2 > 1.0e-30)
            ? dot / sqrt(delta_norm2 * grad_norm2)
            : 0.0;
    const double abs_projection = fabs(cosine);
    const double descent_alignment = -cosine;
    const double effect_eps = m->fp_interface_gap_effect_epsilon > 0.0f
        ? (double)m->fp_interface_gap_effect_epsilon
        : 0.05;
    const int identifying = (finite_count > 0 && input_norm2 > 1.0e-30 &&
                             grad_norm2 > 1.0e-30) ? 1 : 0;
    const int near_identity =
        (identifying && effect_ratio < effect_eps && grad_rms > 1.0e-12) ? 1 : 0;

    m->fp_interface_gap_effect_ratio[layer] = (float)effect_ratio;
    m->fp_interface_gap_input_rms[layer] = (float)input_rms;
    m->fp_interface_gap_delta_rms[layer] = (float)delta_rms;
    m->fp_interface_gap_y_ssm_rms[layer] = (float)y_ssm_rms;
    m->fp_interface_gap_grad_rms[layer] = (float)grad_rms;
    m->fp_interface_gap_abs_projection[layer] = (float)abs_projection;
    m->fp_interface_gap_descent_alignment[layer] = (float)descent_alignment;
    m->fp_interface_gap_wout_nnz[layer] =
        (uint64_t)m->layers[layer].csr_out.total_pos +
        (uint64_t)m->layers[layer].csr_out.total_neg;
    m->fp_interface_gap_near_identity[layer] = near_identity;

    m->fp_interface_gap_probe_layers++;
    if (near_identity) m->fp_interface_gap_near_identity_layers++;
    m->fp_interface_gap_probe_identifying =
        (m->fp_interface_gap_probe_identifying || identifying) ? 1 : 0;

    double sum_effect = 0.0;
    double sum_projection = 0.0;
    double min_effect = INFINITY;
    double max_grad = 0.0;
    uint64_t seen = 0;
    for (int k = 0; k < V12_N_LAYERS; k++) {
        const double er = (double)m->fp_interface_gap_effect_ratio[k];
        const double ap = (double)m->fp_interface_gap_abs_projection[k];
        const double gr = (double)m->fp_interface_gap_grad_rms[k];
        if (er <= 0.0 && ap <= 0.0 && gr <= 0.0) continue;
        sum_effect += er;
        sum_projection += ap;
        if (er < min_effect) min_effect = er;
        if (gr > max_grad) max_grad = gr;
        seen++;
    }
    m->fp_interface_gap_mean_effect_ratio =
        seen > 0 ? (float)(sum_effect / (double)seen) : 0.0f;
    m->fp_interface_gap_mean_abs_projection =
        seen > 0 ? (float)(sum_projection / (double)seen) : 0.0f;
    m->fp_interface_gap_min_effect_ratio =
        isfinite(min_effect) ? (float)min_effect : 0.0f;
    m->fp_interface_gap_max_grad_rms = (float)max_grad;

    printf("[FP_INTERFACE_GAP_PROBE] seq_idx=%d layer=%d input_rms=%.9g x_proj_rms=%.9g x_cv_rms=%.9g b_eff_rms=%.9g h_rms=%.9g delta_rms=%.9g y_ssm_rms=%.9g wout_nnz=%llu effect_ratio=%.9g grad_rms=%.9g cosine=%.9g abs_projection=%.9g descent_alignment=%.9g near_identity=%d identifying=%d finite=%llu mode=current-v12-vulkan\n",
           step,
           layer,
           input_rms,
           x_proj_rms,
           x_cv_rms,
           b_eff_rms,
           h_rms,
           delta_rms,
           y_ssm_rms,
           (unsigned long long)m->fp_interface_gap_wout_nnz[layer],
           effect_ratio,
           grad_rms,
           cosine,
           abs_projection,
           descent_alignment,
           near_identity,
           identifying,
           (unsigned long long)finite_count);

    free(r_in);
    free(r_out);
    free(grad);
    free(x_proj);
    free(x_cv);
    free(b_eff);
    free(h_cache);
    free(y_ssm);
}

void macro_transcript_gate_update(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len) {
    if (!m || !m->macro_transcript_gate_enabled || !tokens ||
        batch_size <= 0 || seq_len <= 0) {
        return;
    }

    uint64_t positions = 0;
    uint64_t ops = 0;
    uint64_t pair_macros = 0;
    uint64_t singletons = 0;
    const uint32_t min_count = m->macro_transcript_min_count
        ? m->macro_transcript_min_count
        : 2u;

    for (int b = 0; b < batch_size; b++) {
        const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
        int t = 0;
        positions += (uint64_t)seq_len;
        while (t < seq_len) {
            if (t + 1 < seq_len &&
                macro_transcript_pair_count(m, tok_b[t], tok_b[t + 1]) >= min_count) {
                pair_macros++;
                ops++;
                t += 2;
            } else {
                singletons++;
                ops++;
                t++;
            }
        }
    }

    m->macro_transcript_gate_batches++;
    m->macro_transcript_gate_positions += positions;
    m->macro_transcript_gate_ops += ops;
    m->macro_transcript_gate_pair_macros += pair_macros;
    m->macro_transcript_gate_singletons += singletons;

    const double total_ops = (double)m->macro_transcript_gate_ops;
    const double total_positions = (double)m->macro_transcript_gate_positions;
    const double length_ratio = total_positions > 0.0 ? total_ops / total_positions : 1.0;
    const double pair_fraction = total_ops > 0.0
        ? (double)m->macro_transcript_gate_pair_macros / total_ops
        : 0.0;
    const double priced =
        length_ratio * (double)m->macro_transcript_hardness +
        (double)m->macro_transcript_codec_cost +
        (double)m->macro_transcript_decode_cost;

    m->macro_transcript_gate_length_ratio = (float)length_ratio;
    m->macro_transcript_gate_pair_fraction = (float)pair_fraction;
    m->macro_transcript_gate_priced_ratio = (float)priced;
}

void reversible_deterministic_expansion_gate_update(
    Model *m, const uint16_t *tokens, int batch_size, int seq_len) {
    if (!m || !m->reversible_deterministic_expansion_gate_enabled || !tokens ||
        batch_size <= 0 || seq_len <= 0) {
        return;
    }

    uint64_t positions = 0;
    uint64_t covered = 0;
    uint64_t spans = 0;
    uint64_t coord_units = 0;
    uint64_t inverse_failures = 0;
    const uint32_t min_run = m->reversible_deterministic_expansion_min_run > 1u
        ? m->reversible_deterministic_expansion_min_run
        : 3u;

    for (int b = 0; b < batch_size; b++) {
        const uint16_t *tok_b = tokens + (size_t)b * (size_t)seq_len;
        int t = 0;
        while (t < seq_len) {
            const uint16_t tok = tok_b[t];
            int run = 1;
            while (t + run < seq_len && tok_b[t + run] == tok) run++;
            positions += (uint64_t)run;
            if (run >= (int)min_run && reversible_deterministic_expansion_token(tok)) {
                int inverse_ok = 1;
                for (int j = 0; j < run; j++) {
                    if (tok_b[t + j] != tok) {
                        inverse_ok = 0;
                        break;
                    }
                }
                if (inverse_ok) {
                    covered += (uint64_t)(run - 1);
                    spans++;
                    coord_units++;
                } else {
                    inverse_failures++;
                }
            }
            t += run;
        }
    }

    m->reversible_deterministic_expansion_gate_batches++;
    m->reversible_deterministic_expansion_gate_positions += positions;
    m->reversible_deterministic_expansion_gate_covered += covered;
    m->reversible_deterministic_expansion_gate_spans += spans;
    m->reversible_deterministic_expansion_gate_coord_units += coord_units;
    m->reversible_deterministic_expansion_inverse_failures += inverse_failures;

    const double total = (double)m->reversible_deterministic_expansion_gate_positions;
    const double lambda = total > 0.0
        ? (double)m->reversible_deterministic_expansion_gate_covered / total
        : 0.0;
    const double eta = total > 0.0
        ? (double)m->reversible_deterministic_expansion_gate_coord_units / total
        : 0.0;
    const double priced = (double)m->reversible_deterministic_expansion_overhead +
        (1.0 - lambda + eta) *
            (1.0 + (double)m->reversible_deterministic_expansion_noise_penalty);
    m->reversible_deterministic_expansion_gate_lambda = (float)lambda;
    m->reversible_deterministic_expansion_gate_eta = (float)eta;
    m->reversible_deterministic_expansion_gate_priced_ratio = (float)priced;
    m->reversible_deterministic_expansion_inverse_audit =
        (m->reversible_deterministic_expansion_inverse_failures == 0u) ? 1 : 0;
}

double paper7_dfga_rel_l2(double diff_sq, double ref_sq) {
    if (ref_sq <= 1.0e-30) {
        return diff_sq <= 1.0e-30 ? 0.0 : INFINITY;
    }
    return sqrt(diff_sq / ref_sq);
}

double paper7_dfga_cosine(double dot, double a_sq, double b_sq) {
    if (a_sq <= 1.0e-30 || b_sq <= 1.0e-30) return NAN;
    return dot / sqrt(a_sq * b_sq);
}

void paper7_dfga_real_batch_audit(
    Model *m, const float *d_logits_buf, const float *norm_out_buf,
    const int32_t *sample_ids_buf, int batch_size, int seq_len, int d,
    int n_samples, const TrainState *state) {
    if (!m || !m->paper7_dfga_real_batch_audit_enabled ||
        !d_logits_buf || !norm_out_buf || !sample_ids_buf ||
        !m->vocab_to_hot || !m->sampled_lm_head || !m->shadow_lm_head) {
        return;
    }
    const uint32_t max_batches =
        m->paper7_dfga_real_batch_audit_max_batches > 0u
            ? m->paper7_dfga_real_batch_audit_max_batches
            : 1u;
    if (m->paper7_dfga_real_batch_audit_batches >= (uint64_t)max_batches) {
        return;
    }

    const size_t rows = (size_t)V12_K_HOT;
    const size_t cols = (size_t)d;
    double *dense_g = (double *)calloc(rows * cols, sizeof(double));
    double *row_first_direct = (double *)calloc(rows, sizeof(double));
    double *row_first_dense = (double *)calloc(rows, sizeof(double));
    double *row_sq_diag = (double *)calloc(rows, sizeof(double));
    double *row_sq_exact = (double *)calloc(rows, sizeof(double));
    double *col_first_direct = (double *)calloc(cols, sizeof(double));
    double *col_first_dense = (double *)calloc(cols, sizeof(double));
    double *col_sq_diag = (double *)calloc(cols, sizeof(double));
    double *col_sq_exact = (double *)calloc(cols, sizeof(double));
    if (!dense_g || !row_first_direct || !row_first_dense || !row_sq_diag ||
        !row_sq_exact || !col_first_direct || !col_first_dense ||
        !col_sq_diag || !col_sq_exact) {
        printf("[PAPER7_DFGA_REAL_BATCH_AUDIT] active=0 reason=alloc_failed support_rows=%u d_model=%d\n",
               (unsigned)V12_K_HOT, d);
        free(dense_g);
        free(row_first_direct);
        free(row_first_dense);
        free(row_sq_diag);
        free(row_sq_exact);
        free(col_first_direct);
        free(col_first_dense);
        free(col_sq_diag);
        free(col_sq_exact);
        return;
    }

    uint64_t sample_events = 0;
    uint64_t hot_events = 0;
    uint64_t active_cells = 0;
    uint64_t invalid_samples = 0;
    uint64_t nonfinite_values = 0;
    uint64_t crystallized_cells = 0;
    uint64_t nonzero_dg_events = 0;
    uint64_t h_nonzero_cells = 0;
    double dg_abs_sum = 0.0;
    double h_abs_sum = 0.0;
    double dg_max_abs = 0.0;
    double h_max_abs = 0.0;

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            const int32_t *sids =
                sample_ids_buf + ((size_t)b * seq_len + t) * n_samples;
            const float *d_slogs =
                d_logits_buf + ((size_t)b * seq_len + t) * n_samples;
            const float *h_t = norm_out_buf + ((size_t)b * seq_len + t) * d;
            for (int s = 0; s < n_samples; s++) {
                const int32_t v_s = sids[s];
                if (v_s < 0 || v_s >= V12_VOCAB_SIZE) {
                    invalid_samples++;
                    continue;
                }
                sample_events++;
                const int32_t hot_idx = m->vocab_to_hot[v_s];
                if (hot_idx < 0 || hot_idx >= V12_K_HOT) continue;
                hot_events++;

                const float dg_f = d_slogs[s];
                if (!isfinite((double)dg_f) || fabsf(dg_f) < 1.0e-12f) {
                    if (!isfinite((double)dg_f)) nonfinite_values++;
                    continue;
                }
                const double dg_abs = fabs((double)dg_f);
                nonzero_dg_events++;
                dg_abs_sum += dg_abs;
                if (dg_abs > dg_max_abs) dg_max_abs = dg_abs;
                for (int j = 0; j < d; j++) {
                    const size_t lm_idx = (size_t)v_s * (size_t)d + (size_t)j;
                    if (fabsf(SHADOW_DECODE(m->shadow_lm_head[lm_idx])) >
                        V12_CRYSTALLIZATION_TAU) {
                        crystallized_cells++;
                        continue;
                    }
                    const float h_f = h_t[j];
                    if (!isfinite((double)h_f)) {
                        nonfinite_values++;
                        continue;
                    }
                    const double h_abs = fabs((double)h_f);
                    h_abs_sum += h_abs;
                    if (h_abs > 1.0e-12) h_nonzero_cells++;
                    if (h_abs > h_max_abs) h_max_abs = h_abs;
                    const float eff_dw_f =
                        (m->sampled_lm_head[lm_idx] == 0.0f)
                            ? (dg_f * V12_BI_STE_FACTOR)
                            : dg_f;
                    const double g = (double)eff_dw_f * (double)h_f;
                    if (!isfinite(g) || g == 0.0) {
                        if (!isfinite(g)) nonfinite_values++;
                        continue;
                    }
                    const size_t off = (size_t)hot_idx * cols + (size_t)j;
                    dense_g[off] += g;
                    row_first_direct[hot_idx] += g;
                    col_first_direct[j] += g;
                    row_sq_diag[hot_idx] += g * g;
                    col_sq_diag[j] += g * g;
                    active_cells++;
                }
            }
        }
    }

    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            const double g = dense_g[r * cols + c];
            row_first_dense[r] += g;
            row_sq_exact[r] += g * g;
            col_first_dense[c] += g;
            col_sq_exact[c] += g * g;
        }
    }

    double row_first_diff_sq = 0.0, row_first_ref_sq = 0.0;
    double col_first_diff_sq = 0.0, col_first_ref_sq = 0.0;
    double row_sq_diff_sq = 0.0, row_sq_ref_sq = 0.0;
    double col_sq_diff_sq = 0.0, col_sq_ref_sq = 0.0;
    double row_sq_dot = 0.0, row_sq_diag_norm = 0.0;
    double col_sq_dot = 0.0, col_sq_diag_norm = 0.0;
    double row_sq_diag_sum = 0.0, row_sq_exact_sum = 0.0;
    double col_sq_diag_sum = 0.0, col_sq_exact_sum = 0.0;
    double row_first_max_abs = 0.0, col_first_max_abs = 0.0;

    for (size_t r = 0; r < rows; r++) {
        const double first_diff = row_first_direct[r] - row_first_dense[r];
        row_first_diff_sq += first_diff * first_diff;
        row_first_ref_sq += row_first_dense[r] * row_first_dense[r];
        if (fabs(first_diff) > row_first_max_abs) row_first_max_abs = fabs(first_diff);

        const double sq_diff = row_sq_diag[r] - row_sq_exact[r];
        row_sq_diff_sq += sq_diff * sq_diff;
        row_sq_ref_sq += row_sq_exact[r] * row_sq_exact[r];
        row_sq_dot += row_sq_diag[r] * row_sq_exact[r];
        row_sq_diag_norm += row_sq_diag[r] * row_sq_diag[r];
        row_sq_diag_sum += row_sq_diag[r];
        row_sq_exact_sum += row_sq_exact[r];
    }
    for (size_t c = 0; c < cols; c++) {
        const double first_diff = col_first_direct[c] - col_first_dense[c];
        col_first_diff_sq += first_diff * first_diff;
        col_first_ref_sq += col_first_dense[c] * col_first_dense[c];
        if (fabs(first_diff) > col_first_max_abs) col_first_max_abs = fabs(first_diff);

        const double sq_diff = col_sq_diag[c] - col_sq_exact[c];
        col_sq_diff_sq += sq_diff * sq_diff;
        col_sq_ref_sq += col_sq_exact[c] * col_sq_exact[c];
        col_sq_dot += col_sq_diag[c] * col_sq_exact[c];
        col_sq_diag_norm += col_sq_diag[c] * col_sq_diag[c];
        col_sq_diag_sum += col_sq_diag[c];
        col_sq_exact_sum += col_sq_exact[c];
    }

    const double row_first_rel_l2 =
        paper7_dfga_rel_l2(row_first_diff_sq, row_first_ref_sq);
    const double col_first_rel_l2 =
        paper7_dfga_rel_l2(col_first_diff_sq, col_first_ref_sq);
    const double row_sq_rel_l2 = paper7_dfga_rel_l2(row_sq_diff_sq, row_sq_ref_sq);
    const double col_sq_rel_l2 = paper7_dfga_rel_l2(col_sq_diff_sq, col_sq_ref_sq);
    const double row_sq_cos =
        paper7_dfga_cosine(row_sq_dot, row_sq_diag_norm, row_sq_ref_sq);
    const double col_sq_cos =
        paper7_dfga_cosine(col_sq_dot, col_sq_diag_norm, col_sq_ref_sq);
    const double row_sq_ratio =
        row_sq_exact_sum > 1.0e-30 ? row_sq_diag_sum / row_sq_exact_sum : NAN;
    const double col_sq_ratio =
        col_sq_exact_sum > 1.0e-30 ? col_sq_diag_sum / col_sq_exact_sum : NAN;
    const double budget =
        m->paper7_dfga_real_batch_audit_residual_budget >= 0.0f
            ? (double)m->paper7_dfga_real_batch_audit_residual_budget
            : 0.35;
    const int pass =
        active_cells > 0 &&
        nonfinite_values == 0 &&
        row_first_rel_l2 <= 1.0e-6 &&
        col_first_rel_l2 <= 1.0e-6 &&
        row_sq_rel_l2 <= budget &&
        col_sq_rel_l2 <= budget;
    const uint64_t support_dense_bytes =
        (uint64_t)rows * (uint64_t)cols * (uint64_t)sizeof(float);
    const uint64_t full_dense_bytes =
        (uint64_t)V12_VOCAB_SIZE * (uint64_t)cols * (uint64_t)sizeof(float);
    const uint64_t factor_stat_bytes =
        (uint64_t)2u * ((uint64_t)rows + (uint64_t)cols) *
        (uint64_t)sizeof(float);
    const unsigned step = state ? state->step : 0u;
    m->paper7_dfga_real_batch_audit_batches++;

    printf("[PAPER7_DFGA_REAL_BATCH_AUDIT] active=1 step=%u batch=%llu support_rows=%u d_model=%d n_samples=%d sample_events=%llu hot_events=%llu nonzero_dg_events=%llu h_nonzero_cells=%llu active_cells=%llu invalid_samples=%llu nonfinite_values=%llu crystallized_cells=%llu dg_abs_mean=%.9g dg_max_abs=%.9g h_abs_mean=%.9g h_max_abs=%.9g first_row_rel_l2=%.9g first_col_rel_l2=%.9g first_row_max_abs=%.9g first_col_max_abs=%.9g row_sq_diag_rel_l2=%.9g col_sq_diag_rel_l2=%.9g row_sq_diag_cosine=%.9g col_sq_diag_cosine=%.9g row_sq_diag_to_exact_sum=%.9g col_sq_diag_to_exact_sum=%.9g residual_budget=%.9g support_dense_bytes=%llu full_dense_bytes=%llu factor_stat_bytes=%llu pass=%s\n",
           step,
           (unsigned long long)m->paper7_dfga_real_batch_audit_batches,
           (unsigned)V12_K_HOT,
           d,
           n_samples,
           (unsigned long long)sample_events,
           (unsigned long long)hot_events,
           (unsigned long long)nonzero_dg_events,
           (unsigned long long)h_nonzero_cells,
           (unsigned long long)active_cells,
           (unsigned long long)invalid_samples,
           (unsigned long long)nonfinite_values,
           (unsigned long long)crystallized_cells,
           nonzero_dg_events > 0 ? dg_abs_sum / (double)nonzero_dg_events : 0.0,
           dg_max_abs,
           h_nonzero_cells > 0 ? h_abs_sum / (double)h_nonzero_cells : 0.0,
           h_max_abs,
           row_first_rel_l2,
           col_first_rel_l2,
           row_first_max_abs,
           col_first_max_abs,
           row_sq_rel_l2,
           col_sq_rel_l2,
           row_sq_cos,
           col_sq_cos,
           row_sq_ratio,
           col_sq_ratio,
           budget,
           (unsigned long long)support_dense_bytes,
           (unsigned long long)full_dense_bytes,
           (unsigned long long)factor_stat_bytes,
           pass ? "yes" : "no");

    free(dense_g);
    free(row_first_direct);
    free(row_first_dense);
    free(row_sq_diag);
    free(row_sq_exact);
    free(col_first_direct);
    free(col_first_dense);
    free(col_sq_diag);
    free(col_sq_exact);
}


#else // !V12_HAS_VULKAN

void gpu_gradient_null_witness_gate_update(GpuPipeline *p, Model *m)               { (void)p; (void)m; }

void gpu_online_tssi_live_gate_update(GpuPipeline *p, Model *m)                    { (void)p; (void)m; }

void gpu_r423_native_growth_receipt_challenge_gate_update(GpuPipeline *p, int step) { (void)p; (void)step; }

void gpu_r423_native_growth_interaction_release_safety_gate_update(GpuPipeline *p, int step) { (void)p; (void)step; }

void gpu_r423_native_growth_interaction_safe_update(GpuPipeline *p, int step) { (void)p; (void)step; }

void gpu_r423_native_growth_genotype_exception_gate_update(GpuPipeline *p, int step) { (void)p; (void)step; }

void gpu_r423_native_growth_genotype_compact_update(GpuPipeline *p, int step) { (void)p; (void)step; }

int gpu_pipeline_write_online_tssi_selector(GpuPipeline *p, const char *path, float active_fraction, const char *round_label) {
    (void)p;
    (void)path;
    (void)active_fraction;
    (void)round_label;
    return -1;
}

int gpu_pipeline_set_online_tssi_train_selector(GpuPipeline *p,
                                                const uint8_t *state_mask,
                                                uint32_t mask_len,
                                                const char *control_name,
                                                uint32_t active_dims) {
    (void)p;
    (void)state_mask;
    (void)mask_len;
    (void)control_name;
    (void)active_dims;
    return -1;
}


#endif // V12_HAS_VULKAN
