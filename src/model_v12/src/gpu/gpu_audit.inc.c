static int r400_obs_trace_enabled(void) {
    if (!g_r400_obs_trace_checked) {
        const char *flag = getenv(R400_OBS_TRACE_ENV);
        g_r400_obs_trace_enabled = env_flag_enabled(flag);
        g_r400_obs_trace_checked = 1;
    }
    return g_r400_obs_trace_enabled;
}

static int r400_obs_trace_mkdir_single(const char *path) {
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST) return 0;
#else
    if (mkdir(path, 0777) == 0 || errno == EEXIST) return 0;
#endif
    return -1;
}

static int r400_obs_trace_ensure_parent_dirs(const char *path) {
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

static int r400_obs_trace_truncate_file(FILE *fp, long offset) {
    if (!fp || offset < 0) return -1;
    fflush(fp);
#ifdef _WIN32
    return (_chsize_s(_fileno(fp), (size_t)offset) == 0) ? 0 : -1;
#else
    return (ftruncate(fileno(fp), (off_t)offset) == 0) ? 0 : -1;
#endif
}

#define R400_OBS_FLAG_STATE_LIVE        0x0001u
#define R400_OBS_FLAG_TRANSITION_LIVE   0x0002u
#define R400_OBS_FLAG_OBJECTIVE_LIVE    0x0004u
#define R400_OBS_FLAG_OBJECTIVE_ABSENT  0x0008u
#define R400_OBS_FLAG_BOUNDARY_PRE      0x0010u
#define R400_OBS_FLAG_BOUNDARY_POST     0x0020u
#define R400_OBS_FLAG_BOUNDARY_SAMPLE   0x0040u
#define R400_OBS_FLAG_BOUNDARY_Q        0x0080u
#define R400_OBS_FLAG_BOUNDARY_ABSENT   0x0100u

static void r400_obs_trace_emit_non_identifying(R400ObsTraceConfig *cfg,
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

static void r400_obs_trace_init_config(R400ObsTraceConfig *cfg,
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

static void r400_obs_trace_log_activation_once(R400ObsTraceConfig *cfg) {
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

static int r400_obs_trace_should_capture(R400ObsTraceConfig *cfg,
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

static FILE *r400_obs_trace_open_file(R400ObsTraceConfig *cfg) {
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

static int r400_obs_trace_write_meta(FILE *fp, const R400ObsTraceConfig *cfg) {
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

static int r400_obs_trace_group_index(R400ObsTraceConfig *cfg,
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

static void r400_obs_trace_mark_group(R400ObsTraceConfig *cfg,
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

static void r400_obs_trace_compute_summary(const R400ObsTraceConfig *cfg,
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

static int r400_obs_trace_write_summary(FILE *fp, R400ObsTraceConfig *cfg) {
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

static int r400_obs_trace_append_summary(FILE *fp, R400ObsTraceConfig *cfg) {
    if (!fp || !cfg) return -1;
    cfg->summary_offset = ftell(fp);
    if (cfg->summary_offset < 0) return -1;
    if (r400_obs_trace_write_summary(fp, cfg) != 0) return -1;
    fflush(fp);
    return 0;
}

static void r400_obs_trace_summarize_float_values(const float *values,
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

static void r400_obs_trace_summarize_double_values(const double *values,
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

static int r400_obs_trace_write_record(FILE *fp,
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

static int r400_obs_trace_write_absence_record(FILE *fp,
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

static int r400_obs_trace_capture_training(GpuPipeline *p,
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

struct GpuPipeline {
    VulkanContext *ctx;
    int seq_len;
    int batch_size;
    int d_model;
    int d_state;
    int n_layers;

    /* Embedding (GPU-resident, uploaded once, re-uploaded after optimizer step) */
    VkBufferHandle embed_E_gpu;     /* [vocab × d_embed]       */
    VkBufferHandle embed_P_gpu;     /* [d_embed × d_model]     */
    VkBufferHandle embed_gathered;  /* [seq_len × d_embed]     scratch */
    VkBufferHandle embed_tokens;    /* [seq_len] uint32        scratch */

    /* Shared scratch (reused across layers) */
    VkBufferHandle buf_x;    /* [seq_len × d_model] */
    VkBufferHandle buf_y;    /* [seq_len × d_model] */

    /* Residual stream (GPU-resident across layers) */
    VkBufferHandle residual[V12_N_LAYERS + 1];

    /* Predictive coding */
    VkBufferHandle pred_A;          /* sampled_w_pred [d×d] */
    VkBufferHandle shadow_pred_A;   /* shadow_w_pred [d×d] */
    VkBufferHandle pred_x;          /* [T × d] batch input */
    VkBufferHandle pred_y;          /* [T × d] batch output */

    /* GEMM scratch */
    VkBufferHandle gemm_A;          /* [seq×max(d,ds)] */
    VkBufferHandle gemm_B;          /* [seq×max(d,ds)] */
    VkBufferHandle gemm_C;          /* [max(d,ds)×max(d,ds)] */
    uint32_t q09_k;
    VkBufferHandle q09_P;
    VkBufferHandle q09_P_left;
    VkBufferHandle q09_P_right;
    VkBufferHandle q09_stats;
    VkBufferHandle q09_dw_approx;

    // Q-10 additions
    VkBufferHandle q10_w_v;         /* [ds] */
    VkBufferHandle q10_dw_v;        /* [ds] */
    VkBufferHandle q10_v_head;      /* [batch * seq] */
    VkBufferHandle q10_w_v_m;       /* [ds] */
    VkBufferHandle q10_w_v_v;       /* [ds] */

    /* SSM scan transients (reused per-layer) */
    VkBufferHandle ssm_a;           /* A_eff [seq×ds] */
    VkBufferHandle ssm_b;           /* B_eff [seq×ds] */
    VkBufferHandle ssm_x;           /* x_cv [seq×ds] */
    VkBufferHandle ssm_h;           /* h_out [seq×ds] */
    VkBufferHandle ssm_cbl;         /* C_bilinear [ds] */
    VkBufferHandle ssm_u;           /* sampled_u [ds] */
    VkBufferHandle ef03_ring_sums;
    VkBufferHandle ssm_v;           /* sampled_v [ds] */

    /* Layer weights (uploaded per-layer, reused) */
    VkBufferHandle lw_conv;         /* [K × ds] */
    VkBufferHandle lw_B;            /* [ds] */
    VkBufferHandle lw_A;            /* [ds] */
    VkBufferHandle lw_WA;           /* [ds × ds] */
    VkBufferHandle lw_bA;           /* [ds] */
    VkBufferHandle lw_C;            /* [ds × ds] */

    /* Per-layer state */
    struct {
        VkTernaryCSR gpu_csr_in;
        VkTernaryCSR gpu_csr_out;
        VkTernaryCSR gpu_csc_in;    /* W_in^T for backward */
        VkTernaryCSR gpu_csc_out;   /* W_out^T for backward */
        int uploaded;
        int csc_uploaded;

        /* Forward activations (GPU-resident for backward) */
        VkBufferHandle fwd_x_proj;
        VkBufferHandle fwd_x_gated;
        VkBufferHandle fwd_mask_val;
        VkBufferHandle fwd_x_cv;
        VkBufferHandle fwd_b_eff;
        VkBufferHandle fwd_a_buf;
        VkBufferHandle fwd_h_cache;
        VkBufferHandle f34_P_r;
        VkBufferHandle f34_z_cache;
#ifdef R504_E10_PSCQ
    VkBufferHandle fwd_h_cache_compressed;
#endif
        VkBufferHandle qrf3ac_basis;
        VkBufferHandle qrf3ac_ternary;
        VkBufferHandle htr1p_u;
        VkBufferHandle htr1p_v;
        VkBufferHandle fwd_y_ssm;   /* per-layer y_ssm result [bseq x ds] */
        VkBufferHandle gpu_moments; /* GP-KL Welford accumulators on GPU   */
        VkBufferHandle tssi_score_moments; /* R420 compact target-conditioned state scores */
        int fwd_allocated;

        /* MTP buffers — resident weights (uploaded once, reused every backward) */
        VkBufferHandle gpu_mtp_U;   /* [d_state * mtp_rank] float — NULL if layer has no MTP */
        VkBufferHandle gpu_mtp_V;   /* [d_state * mtp_rank] float */
        VkBufferHandle gpu_A_const; /* [d_state] float — A vector broadcast source */

        /* MTP scratch — zeroed before each backward MTP dispatch */
        VkBufferHandle gpu_d_mtp_U; /* [d_state * mtp_rank] uint — CAS atomic accumulator */
        VkBufferHandle gpu_d_mtp_V; /* [d_state * mtp_rank] uint — CAS atomic accumulator */
        VkBufferHandle gpu_mtp_h;   /* [bseq * d_state] float — h_mtp from MTP forward scan */
        VkBufferHandle gpu_mtp_loss;/* [1] uint — CAS atomic loss accumulator */
        int mtp_allocated;
    } layers[V12_N_LAYERS];

    /* RMS norm weights + outputs */
    VkBufferHandle norm_w;          /* [d_model] gamma */
    VkBufferHandle norm_b;          /* [d_model] beta */
    VkBufferHandle norm_out;        /* [seq_len × d_model] normed output */
    VkBufferHandle norm_rms;        /* [seq_len] per-timestep RMS values */

    /* Backward scratch */
    VkBufferHandle bwd_d_y_ssm;
    VkBufferHandle bwd_d_xp;
    VkBufferHandle bwd_d_x_cv;
    VkBufferHandle bwd_dA_ps;
    VkBufferHandle bwd_dB_seq;
    VkBufferHandle bwd_v_pre;
VkBufferHandle bwd_d_h_in;
    VkBufferHandle bwd_d_h_in_prev_layer; // Q-16
    VkBufferHandle q16_sums_buf; // Q-16      /* GPU-resident d_h */
    VkBufferHandle bwd_h_final;     /* [batch_size × d_state] recurrent backward F1 seed state */
    VkBufferHandle bwd_d_h_prev;
    VkBufferHandle bwd_d_embed_E;   /* [vocab × d_embed] float — GPU accumulator for d_embed_E grad */
#ifdef R503_PHASE_1_EFFICIENCY
    VkBufferHandle bwd_d_embed_E_sparse; /* E-10 Sparse VRAM */
    uint32_t fati_chunk_size;            /* E-13 FATI */
#endif
    VkBufferHandle bwd_d_conv_w;
    VkBufferHandle bwd_d_tau;       /* [1] float — per-layer tau gradient accumulator */
    VkBufferHandle bwd_d_dot;       /* [bseq × ds] intermediate for A-backward */
    VkBufferHandle bwd_dA_reduce;   /* [ds] reduced dA per-state */
    VkBufferHandle bwd_dbA_reduce;  /* [ds] reduced db_A per-state */
    VkBufferHandle bwd_dC_bl_accum; /* [ds] dC_bilinear GPU accumulator */
    VkBufferHandle bwd_dB_accum;    /* [ds] uint — CAS atomic accumulator for Direct-B dB gradient */
    VkBufferHandle bwd_B_vec;       /* [ds] float — current layer B vector for Direct-B shader */

    /* =========================================================================
     * GPU Optimizer State
     *
     * Mirrors every Adam / factored-Adam / EMA-SPRT buffer in SSMLayer and
     * Model so that optimizer steps can run entirely on GPU without round-
     * tripping gradients or moments through CPU.
     *
     * Naming mirrors types.h exactly:
     *   opt_layers[k].<field>  ↔  m->layers[k].<field>
     *   opt_<global>           ↔  m-><global>
     *
     * Buffer slot 0 is reserved as VK_BUFFER_NULL, so calloc leaves handles invalid.
     * Allocation happens in gpu_optimizer_alloc() (not here).
     * ========================================================================= */

    struct {
        /* --- Ternary W_in [d_model × d_state] -------------------------------- */
        VkBufferHandle shadow_w_in;      /* shadow logits        [d_model × d_state] */
        VkBufferHandle m_w_in_row;       /* factored m row       [d_model]           */
        VkBufferHandle v_w_in_row;       /* factored v row       [d_model]           */
        VkBufferHandle m_w_in_col;       /* factored m col       [d_state]           */
        VkBufferHandle v_w_in_col;       /* factored v col       [d_state]           */
        VkBufferHandle grad_accum_in;    /* EMA-SPRT accum       [d_model × d_state] */
        VkBufferHandle crystal_mask_in;  /* crystallization mask [d_model × d_state] uint8 */
        VkBufferHandle sampled_w_in;     /* current ternary samples [d_model × d_state] float */
        VkBufferHandle g_row_in;         /* gradient row stats   [d_model] float             */
        VkBufferHandle g_col_in;         /* gradient col stats   [d_state] float             */
        VkBufferHandle gq_row_in;        /* 2nd-moment row stats [d_model] float             */
        VkBufferHandle gq_col_in;        /* 2nd-moment col stats [d_state] float             */
        VkBufferHandle cnt_row_in;       /* ternary count/row    [d_model] uint32            */
        VkBufferHandle cnt_col_in;       /* ternary count/col    [d_state] uint32            */
        VkBufferHandle mean_v_in;
        VkBufferHandle sign_ema_in; // Q-13        /* scalar mean-v output [1] float                   */

        /* --- Ternary W_out [d_state × d_model] ------------------------------- */
        VkBufferHandle shadow_w_out;
        VkBufferHandle m_w_out_row;      /* factored m row       [d_state]           */
        VkBufferHandle v_w_out_row;      /* factored v row       [d_state]           */
        VkBufferHandle m_w_out_col;      /* factored m col       [d_model]           */
        VkBufferHandle v_w_out_col;      /* factored v col       [d_model]           */
        VkBufferHandle grad_accum_out;   /* EMA-SPRT accum       [d_state × d_model] */
        VkBufferHandle crystal_mask_out; /* crystallization mask [d_state × d_model] uint8 */
        VkBufferHandle sampled_w_out;    /* current ternary samples [d_state × d_model] float */
        VkBufferHandle g_row_out;        /* gradient row stats   [d_state] float             */
        VkBufferHandle g_col_out;        /* gradient col stats   [d_model] float             */
        VkBufferHandle gq_row_out;       /* 2nd-moment row stats [d_state] float             */
        VkBufferHandle gq_col_out;       /* 2nd-moment col stats [d_model] float             */
        VkBufferHandle mprm_Sigma_M;
        VkBufferHandle mprm_Sigma_V;
        VkBufferHandle cnt_row_out;      /* ternary count/row    [d_state] uint32            */
        VkBufferHandle cnt_col_out;      /* ternary count/col    [d_model] uint32            */
        VkBufferHandle mean_v_out;
        VkBufferHandle sign_ema_out; // Q-13       /* scalar mean-v output [1] float                   */

        /* --- Dense: A [d_state] ---------------------------------------------- */
        VkBufferHandle m_A;              /* Adam m               [d_state]           */
        VkBufferHandle v_A;              /* Adam v               [d_state]           */
        VkBufferHandle dA;               /* gradient accum       [d_state]           */

        /* --- Dense: B [d_state] ---------------------------------------------- */
        VkBufferHandle m_B;
        VkBufferHandle v_B;
        VkBufferHandle dB;               /* gradient accum       [d_state]           */

        /* --- Dense: C [d_state × d_state] ------------------------------------ */
        VkBufferHandle m_C;              /* Adam m               [d_state × d_state] */
        VkBufferHandle v_C;              /* Adam v               [d_state × d_state] */
        VkBufferHandle dC;               /* gradient accum       [d_state × d_state] */

        /* --- Dense: sigma [d_state] ------------------------------------------ */
        VkBufferHandle m_sigma;
        VkBufferHandle v_sigma;
        VkBufferHandle d_sigma;          /* gradient accum       [d_state]           */

        /* --- Ternary conv1d [conv_kernel × d_state] -------------------------- */
        VkBufferHandle shadow_w_conv;    /* shadow logits        [K × d_state]       */
        VkBufferHandle m_conv;           /* Adam m               [K × d_state]       */
        VkBufferHandle v_conv;           /* Adam v               [K × d_state]       */
        VkBufferHandle d_conv_w;         /* gradient accum       [K × d_state]       */

        /* --- Dense: W_A [d_state × d_state], b_A [d_state] ------------------ */
        VkBufferHandle shadow_w_A;       /* shadow logits        [d_state × d_state] */
        VkBufferHandle m_W_A;            /* Adam m               [d_state × d_state] */
        VkBufferHandle v_W_A;            /* Adam v               [d_state × d_state] */
        VkBufferHandle dW_A;             /* gradient accum       [d_state × d_state] */
        VkBufferHandle m_b_A;            /* Adam m               [d_state]           */
        VkBufferHandle v_b_A;            /* Adam v               [d_state]           */
        VkBufferHandle db_A;             /* gradient accum       [d_state]           */

        /* --- Saved gradient buffers (written during backward, read by optimizer) */
        VkBufferHandle gpu_dw_out;       /* [d_model × d_state] saved dw_out gradient  */
        VkBufferHandle gpu_dw_in;
        VkBufferHandle holo_U_in;
        VkBufferHandle holo_V_in;
        VkBufferHandle gpu_dW_A;         /* [d_state × d_state] saved dW_A gradient    */
        VkBufferHandle gpu_dA;           /* [d_state] accumulated dA gradient           */
        VkBufferHandle gpu_dbA;          /* [d_state] accumulated db_A gradient         */
        VkBufferHandle gpu_dconv;        /* [K × d_state] accumulated d_conv_w gradient */

        /* --- Dense: W_A Adam moments [d_state × d_state] -------------------- */
        VkBufferHandle m_WA;             /* Adam m               [d_state × d_state]   */
        VkBufferHandle v_WA;             /* Adam v               [d_state × d_state]   */

        /* --- Dense: MTP U,V [d_state × mtp_rank] (NULL on odd layers) ------- */
        VkBufferHandle m_mtp_U;
        VkBufferHandle v_mtp_U;
        VkBufferHandle d_mtp_U;          /* gradient accum       [d_state × mtp_rank] */
        VkBufferHandle m_mtp_V;
        VkBufferHandle v_mtp_V;
        VkBufferHandle d_mtp_V;          /* gradient accum       [d_state × mtp_rank] */

        /* --- Ternary C_bilinear [d_state] ------------------------------------ */
        VkBufferHandle shadow_C_bilinear;
        VkBufferHandle m_C_bilinear;
        VkBufferHandle v_C_bilinear;
        VkBufferHandle dC_bilinear;      /* gradient accum       [d_state]           */

        int opt_allocated;               /* 1 after gpu_optimizer_alloc fills this slot */
    } opt_layers[V12_N_LAYERS];
    VkBufferHandle tmsheaf_sim[V12_N_LAYERS];

    /* -------------------------------------------------------------------------
     * Global (model-level) optimizer buffers
     * ------------------------------------------------------------------------- */

    /* lm_head [vocab_size × d_model] — factored Adam */
    VkBufferHandle opt_shadow_lm_head;   /* shadow logits        [vocab_size × d_model] */
    VkBufferHandle opt_m_lm_head_row;    /* factored m row       [vocab_size]            */
    VkBufferHandle opt_v_lm_head_row;    /* factored v row       [vocab_size]            */
    VkBufferHandle opt_m_lm_head_col;    /* factored m col       [d_model]               */
    VkBufferHandle opt_v_lm_head_col;    /* factored v col       [d_model]               */
    VkBufferHandle opt_dlm_head;         /* gradient accum       [vocab_size × d_model]  */
    VkBufferHandle opt_crystal_lm_head;  /* hot_mask / crystal   [vocab_size] uint8      */
    /* factored-Adam scratch for lm_head */
    VkBufferHandle opt_sampled_lm_head;  /* sampled ternary      [vocab_size × d_model] */
    VkBufferHandle opt_g_row_lm;         /* grad row stats       [vocab_size]            */
    VkBufferHandle opt_g_col_lm;         /* grad col stats       [d_model]               */
    VkBufferHandle opt_gq_row_lm;        /* quant grad row       [vocab_size]            */
    VkBufferHandle opt_gq_col_lm;        /* quant grad col       [d_model]               */
    VkBufferHandle opt_cnt_row_lm;       /* nonzero count row    [vocab_size]            */
    VkBufferHandle opt_cnt_col_lm;       /* nonzero count col    [d_model]               */
    VkBufferHandle opt_mean_v_lm;
    VkBufferHandle opt_sign_ema_lm; // Q-13        /* sum(v_row), sum(v_col) scratch [2]           */
    VkBufferHandle opt_vocab_to_hot_lm;  /* vocab row -> hot row map [vocab_size]        */

    /* w_pred [d_model × d_model] — factored Adam */
    VkBufferHandle opt_shadow_w_pred;    /* shadow logits        [d_model × d_model]     */
    VkBufferHandle opt_m_w_pred_row;     /* factored m row       [d_model]               */
    VkBufferHandle opt_v_w_pred_row;     /* factored v row       [d_model]               */
    VkBufferHandle opt_m_w_pred_col;     /* factored m col       [d_model]               */
    VkBufferHandle opt_v_w_pred_col;     /* factored v col       [d_model]               */
    VkBufferHandle opt_dw_pred;          /* per-step gradient    [d_model × d_model]     */
    VkBufferHandle opt_dlm_w_pred;       /* EMA-SPRT grad accum  [d_model × d_model]     */
    VkBufferHandle opt_sampled_w_pred;   /* sampled ternary      [d_model × d_model]     */
    VkBufferHandle opt_g_row_wp;         /* grad row stats       [d_model]               */
    VkBufferHandle opt_g_col_wp;         /* grad col stats       [d_model]               */
    VkBufferHandle opt_gq_row_wp;        /* quant grad row       [d_model]               */
    VkBufferHandle opt_gq_col_wp;        /* quant grad col       [d_model]               */
    VkBufferHandle opt_cnt_row_wp;       /* nonzero count row    [d_model]               */
    VkBufferHandle opt_cnt_col_wp;       /* nonzero count col    [d_model]               */
    VkBufferHandle opt_mean_v_wp;
    VkBufferHandle opt_sign_ema_wp; // Q-13        /* sum(v_row), sum(v_col) scratch [2]           */

    /* norm [d_model] — plain Adam */
    VkBufferHandle opt_m_norm_w;         /* Adam m               [d_model]               */
    VkBufferHandle opt_v_norm_w;         /* Adam v               [d_model]               */
    VkBufferHandle opt_m_norm_b;         /* Adam m               [d_model]               */
    VkBufferHandle opt_v_norm_b;         /* Adam v               [d_model]               */
    VkBufferHandle opt_d_norm_w;         /* gradient accum       [d_model]               */
    VkBufferHandle opt_d_norm_b;         /* gradient accum       [d_model]               */

    /* embed_E [vocab_size × d_embed] — dense Adam */
    VkBufferHandle opt_m_embed_E;        /* Adam m               [vocab_size × d_embed]  */
    VkBufferHandle opt_v_embed_E;        /* Adam v               [vocab_size × d_embed]  */
    /* factored Adam stubs (unused, reserved for future use) */
    VkBufferHandle opt_m_embed_E_row;    /* factored m row       [vocab_size]            */
    VkBufferHandle opt_v_embed_E_row;    /* factored v row       [vocab_size]            */
    VkBufferHandle opt_m_embed_E_col;    /* factored m col       [d_embed]               */
    VkBufferHandle opt_v_embed_E_col;    /* factored v col       [d_embed]               */
    VkBufferHandle opt_d_embed_E;        /* gradient accum       [vocab_size × d_embed]  */

    /* embed_P [d_embed × d_model] — plain Adam */
    VkBufferHandle opt_m_embed_P;        /* Adam m               [d_embed × d_model]     */
    VkBufferHandle opt_v_embed_P;        /* Adam v               [d_embed × d_model]     */
    
    VkBufferHandle mprm_P_U;
    VkBufferHandle mprm_P_V;

    VkBufferHandle opt_d_embed_P;        /* gradient accum       [d_embed × d_model]     */

    int opt_global_allocated;            /* 1 after gpu_optimizer_alloc fills global bufs */

    /* CPU staging for gradient download */
    float *staging_dw;              /* max(d*ds, d*d) floats */

    /* Persistent CPU scratch for backward */
    float *pc_neg_buf;              /* [T × d] */
    float *pc_res_buf;              /* [T × d] */
    float *pc_dw_buf;               /* [d × d] */

    /* Sampled CE loss GPU buffers */
    VkBufferHandle ce_row_ptr_pos;  /* lm_head CSR row_ptr_pos [vocab+1] */
    VkBufferHandle ce_pos_indices;  /* lm_head CSR pos_indices */
    VkBufferHandle ce_row_ptr_neg;  /* lm_head CSR row_ptr_neg [vocab+1] */
    VkBufferHandle ce_neg_indices;  /* lm_head CSR neg_indices */
    VkBufferHandle ce_unigram_cdf;  /* [vocab] float */
    VkBufferHandle ce_log_unigram;  /* [vocab] float */
    VkBufferHandle ce_token_freq; /* Q-06 */
    VkBufferHandle ce_sample_ids;   /* [B*(T-1)*N_SAMPLES] int32 scratch */
    VkBufferHandle ce_d_logits;     /* [B*(T-1)*N_SAMPLES] float scratch */
    VkBufferHandle ce_ema_logits;   /* [B*(T-1)*N_SAMPLES] float EMA */
    VkBufferHandle ce_loss;         /* [B*(T-1)] float per-position loss */
    VkBufferHandle ce_base_loss;    /* [B*(T-1)] baseline sampled CE */
    VkBufferHandle ce_d_norm_w;     /* [d_model] float */
    VkBufferHandle ce_d_norm_b;     /* [d_model] float */
    VkBufferHandle ce_dlm_head;     /* [vocab × d_model] float — GPU accumulator for dlm_head grad */

    VkBufferHandle prior_ctx_counts;
    VkBufferHandle prior_pair_keys;
    VkBufferHandle prior_pair_counts;
    VkBufferHandle prior_unigram;
    uint32_t prior_pair_capacity;
    int corpus_prior_enabled;
    VkBufferHandle program_kernel_targets;
    VkBufferHandle program_kernel_counts;
    int program_kernel_enabled;
    VkBufferHandle state_atlas_keys;
    VkBufferHandle state_atlas_targets;
    VkBufferHandle state_atlas_counts;
    uint32_t state_atlas_capacity;
    int state_atlas_enabled;

    /* Temporary count buffers for GPU CSR rebuild (max V12_N_LAYERS * max(d,ds) rows) */
    VkBufferHandle csr_count_pos;  /* [32064] uint32 — per-row positive count scratch */
    VkBufferHandle csr_count_neg;  /* [32064] uint32 — per-row negative count scratch */
    VkBufferHandle flip_counts;    /* [GPU_FLIP_COUNT_SLOTS] uint32 — per-matrix flip counters */
    VkBufferHandle evidence_rate_stats; /* [V12_EVIDENCE_RATE_STATS_SLOTS] uint32/float-bits */

    /* R364 slice A no-op interval probes. */
    uint32_t r364_noop_compile_id;
    uint32_t r364_noop_train_step_calls;
    uint32_t r364_noop_optimizer_step_calls;
    int r364_noop_interval_active;
    float r364_noop_compiled_block_share_f;

    uint8_t *online_tssi_train_state_mask;
    uint32_t online_tssi_train_mask_len;
    uint32_t online_tssi_train_active_dims;
    uint64_t online_tssi_train_applications;
    uint64_t online_tssi_train_masked_w_in_values;
    uint64_t online_tssi_train_masked_w_out_values;
    char online_tssi_train_control_name[32];

    /* R494 CCHS + SPCS skips */
    float prev_H;
    int has_prev_H;
    int skip_bptt_current_step;

    /* PRBLT (Pipelined Ring-Buffered Layer Transfer) */
    struct {
        VkAsyncTransferHandle transfer_handle;
        VkBufferHandle staging_buf;
        char* staging_mapped;

        VkBufferHandle shadow_w_in;
        VkBufferHandle shadow_w_out;

        VkTernaryCSR gpu_csr_in;
        VkTernaryCSR gpu_csr_out;
        VkTernaryCSR gpu_csc_in;
        VkTernaryCSR gpu_csc_out;
        int active_layer_k;
    } prblt_ring[2];
};

