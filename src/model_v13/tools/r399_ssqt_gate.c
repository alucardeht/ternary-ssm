#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#define R399_DEFAULT_OUTPUT ".work/jobs/r399-ssqt-gate-01/artifacts/ssqt_trace.jsonl"
#define R399_DEFAULT_SCENARIO "pass_fixture"
#define R399_SYNTHETIC_BLOCKER \
    "synthetic_fixture_only:no_standalone_runtime_api_exposes_ssqt_obligation_gradients_without_training_integration"

typedef struct {
    const char *output_path;
    const char *scenario;
    int steps;
    int batch_size;
    int seq_len;
    int trace_sample;
    int frontier_cells;
} GateConfig;

static int parse_int_arg(const char *text, int minimum) {
    char *end = NULL;
    long value;
    if (!text || !*text) {
        return -1;
    }
    value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < minimum || value > 1L << 20) {
        return -1;
    }
    return (int)value;
}

static int mkdir_single(const char *path) {
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST) {
        return 0;
    }
#else
    if (mkdir(path, 0777) == 0 || errno == EEXIST) {
        return 0;
    }
#endif
    return -1;
}

static int ensure_parent_dirs(const char *path) {
    char scratch[1024];
    size_t i;
    size_t len;

    if (!path) {
        return -1;
    }
    len = strlen(path);
    if (len == 0 || len >= sizeof(scratch)) {
        return -1;
    }
    memcpy(scratch, path, len + 1);
    for (i = 0; i < len; ++i) {
        if (scratch[i] != '/' && scratch[i] != '\\') {
            continue;
        }
        if (i == 0) {
            continue;
        }
        if (i == 2 && scratch[1] == ':') {
            continue;
        }
        scratch[i] = '\0';
        if (mkdir_single(scratch) != 0) {
            return -1;
        }
        scratch[i] = path[i];
    }
    return 0;
}

static float sign_preserving_base(int group_index) {
    float base = ((float)((group_index % 7) - 3) * 0.45f) +
                 ((float)(((group_index / 7) % 5) - 2) * 0.08f);
    if (fabsf(base) < 0.12f) {
        base += (group_index % 2 == 0) ? 0.28f : -0.28f;
    }
    return base;
}

static float small_noise(int index, int split) {
    return 0.035f * sinf(0.37f * (float)(index + 1)) +
           0.018f * cosf(0.19f * (float)(index + 3 * split + 1));
}

static float large_noise(int index, int split) {
    return 0.82f * sinf(0.61f * (float)(index + 1)) +
           0.67f * cosf(0.43f * (float)(index + 2 * split + 5));
}

static int scenario_group_count(const char *scenario, int total_records) {
    if (strcmp(scenario, "fail_redundancy") == 0) {
        return (total_records > 1) ? ((total_records + 1) / 2) : 1;
    }
    return (total_records >= 3) ? (total_records / 3) : 1;
}

static void print_usage(const char *program) {
    fprintf(stderr,
            "usage: %s [--output PATH] [--steps N] [--batch-size N] [--seq-len N] "
            "[--trace-sample N] [--frontier-cells N] "
            "[--scenario pass_fixture|fail_redundancy|fail_variance|fail_frontier|invalid_nodata|invalid_zero_gradient]\n",
            program);
}

static int parse_args(int argc, char **argv, GateConfig *cfg) {
    int i;
    cfg->output_path = R399_DEFAULT_OUTPUT;
    cfg->scenario = R399_DEFAULT_SCENARIO;
    cfg->steps = 2;
    cfg->batch_size = 2;
    cfg->seq_len = 8;
    cfg->trace_sample = 4;
    cfg->frontier_cells = 2;

    for (i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--output") == 0 && i + 1 < argc) {
            cfg->output_path = argv[++i];
        } else if (strcmp(arg, "--scenario") == 0 && i + 1 < argc) {
            cfg->scenario = argv[++i];
        } else if (strcmp(arg, "--steps") == 0 && i + 1 < argc) {
            cfg->steps = parse_int_arg(argv[++i], 1);
        } else if (strcmp(arg, "--batch-size") == 0 && i + 1 < argc) {
            cfg->batch_size = parse_int_arg(argv[++i], 1);
        } else if (strcmp(arg, "--seq-len") == 0 && i + 1 < argc) {
            cfg->seq_len = parse_int_arg(argv[++i], 1);
        } else if (strcmp(arg, "--trace-sample") == 0 && i + 1 < argc) {
            cfg->trace_sample = parse_int_arg(argv[++i], 1);
        } else if (strcmp(arg, "--frontier-cells") == 0 && i + 1 < argc) {
            cfg->frontier_cells = parse_int_arg(argv[++i], 1);
        } else {
            return -1;
        }
    }

    if (cfg->steps < 0 || cfg->batch_size < 0 || cfg->seq_len < 0 || cfg->trace_sample < 0 ||
        cfg->frontier_cells < 0) {
        return -1;
    }
    return 0;
}

static int write_trace(FILE *handle, const GateConfig *cfg) {
    const int sampled_tokens = (cfg->trace_sample < cfg->seq_len) ? cfg->trace_sample : cfg->seq_len;
    const int total_records = cfg->steps * cfg->batch_size * sampled_tokens * cfg->frontier_cells;
    const int group_count = scenario_group_count(cfg->scenario, total_records);
    int index = 0;

    fprintf(handle,
            "{\"kind\":\"meta\",\"schema\":\"r399.ssqt.trace.v1\",\"mode\":\"diagnostic-only\","
            "\"scenario\":\"%s\",\"activation_line_emitted\":true,\"steps\":%d,\"batch_size\":%d,"
            "\"seq_len\":%d,\"trace_sample\":%d,\"frontier_cells\":%d,"
            "\"no_training_integration\":1,\"synthetic_fixture\":true,"
            "\"blocker\":\"%s\"}\n",
            cfg->scenario,
            cfg->steps,
            cfg->batch_size,
            cfg->seq_len,
            sampled_tokens,
            cfg->frontier_cells,
            R399_SYNTHETIC_BLOCKER);

    if (strcmp(cfg->scenario, "invalid_nodata") == 0) {
        return 0;
    }

    for (int step = 0; step < cfg->steps; ++step) {
        for (int batch = 0; batch < cfg->batch_size; ++batch) {
            for (int token = 0; token < sampled_tokens; ++token) {
                for (int frontier_cell = 0; frontier_cell < cfg->frontier_cells; ++frontier_cell, ++index) {
                    const int group_index = (group_count > 0) ? (index % group_count) : 0;
                    const int duplicate_ordinal = (group_count > 0) ? (index / group_count) : index;
                    const int split = duplicate_ordinal & 1;
                    const float base = sign_preserving_base(group_index);
                    const float raw_noise =
                        (strcmp(cfg->scenario, "fail_variance") == 0) ? large_noise(index, split) : small_noise(index, split);
                    float raw_gradient = base + raw_noise;
                    float quotient_gradient = base + 0.12f * raw_noise;
                    float raw_frontier_pressure = base + 0.20f * raw_noise;
                    float quotient_frontier_pressure = base;

                    if (strcmp(cfg->scenario, "invalid_zero_gradient") == 0) {
                        raw_gradient = 0.0f;
                        quotient_gradient = 0.0f;
                        raw_frontier_pressure = 0.0f;
                        quotient_frontier_pressure = 0.0f;
                    } else if (strcmp(cfg->scenario, "fail_frontier") == 0 && (index % 4 == 0 || index % 7 == 0)) {
                        quotient_frontier_pressure = -base;
                    }

                    fprintf(handle,
                            "{\"kind\":\"record\",\"schema\":\"r399.ssqt.trace.v1\",\"exposure_id\":%d,"
                            "\"step\":%d,\"batch_index\":%d,\"token_index\":%d,\"layer_bucket\":%d,"
                            "\"frontier_cell\":%d,\"quotient_key\":\"z-%03d\",\"split_key\":%d,"
                            "\"active_frontier\":1,\"raw_gradient\":%.9g,\"quotient_gradient\":%.9g,"
                            "\"raw_frontier_pressure\":%.9g,\"quotient_frontier_pressure\":%.9g}\n",
                            index,
                            step,
                            batch,
                            token,
                            (step + batch + token + frontier_cell) % 3,
                            frontier_cell,
                            group_index,
                            split,
                            raw_gradient,
                            quotient_gradient,
                            raw_frontier_pressure,
                            quotient_frontier_pressure);
                }
            }
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    GateConfig cfg;
    FILE *handle = NULL;

    if (parse_args(argc, argv, &cfg) != 0) {
        print_usage(argv[0]);
        return 2;
    }

    if (ensure_parent_dirs(cfg.output_path) != 0) {
        fprintf(stderr, "[R399 SSQT GATE] failed output_path=%s errno=%d\n", cfg.output_path, errno);
        return 1;
    }

    printf("[R399 SSQT GATE] active=1 mode=diagnostic-only steps=%d batch_size=%d seq_len=%d "
           "trace_sample=%d frontier_cells=%d output_path=%s no_training_integration=1\n",
           cfg.steps,
           cfg.batch_size,
           cfg.seq_len,
           (cfg.trace_sample < cfg.seq_len) ? cfg.trace_sample : cfg.seq_len,
           cfg.frontier_cells,
           cfg.output_path);

    handle = fopen(cfg.output_path, "wb");
    if (!handle) {
        fprintf(stderr, "[R399 SSQT GATE] failed output_path=%s errno=%d\n", cfg.output_path, errno);
        return 1;
    }

    if (write_trace(handle, &cfg) != 0) {
        fclose(handle);
        fprintf(stderr, "[R399 SSQT GATE] failed output_path=%s errno=%d\n", cfg.output_path, errno);
        return 1;
    }

    fclose(handle);
    return 0;
}
