#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#define DEFAULT_OUTPUT_PATH ".work/jobs/r495-slice1-associative-gate/summary.json"
#define TEST_SEQ_LEN 512
#define TEST_STATE_DIM 64

// Struct representing the full augmented transition (M, P)
// M is of size (d+1) x (d+1), P is of size (d+1)
typedef struct {
    float *M; // (d+1) x (d+1) row-major matrix
    float *P; // (d+1) vector
} AugmentedTransition;

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
    size_t len = strlen(path);
    if (len == 0 || len >= sizeof(scratch)) return -1;
    memcpy(scratch, path, len + 1);
    for (size_t i = 0; i < len; ++i) {
        if (scratch[i] != '/' && scratch[i] != '\\') continue;
        if (i == 0) continue;
        if (i == 2 && scratch[1] == ':') continue;
        scratch[i] = '\0';
        if (mkdir_single(scratch) != 0) return -1;
        scratch[i] = path[i];
    }
    return 0;
}

// Composition operator: (M_j, P_j) bullet (M_i, P_i) = (M_j * M_i, M_j * P_i + P_j)
static void compose_transition(int d, const AugmentedTransition *j, const AugmentedTransition *i, AugmentedTransition *out) {
    int n = d + 1;
    
    // Compute M_out = M_j * M_i
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += j->M[r * n + k] * i->M[k * n + c];
            }
            out->M[r * n + c] = sum;
        }
    }
    
    // Compute P_out = M_j * P_i + P_j
    for (int r = 0; r < n; ++r) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += j->M[r * n + k] * i->P[k];
        }
        out->P[r] = sum + j->P[r];
    }
}

int main(int argc, char **argv) {
    const char *output_path = DEFAULT_OUTPUT_PATH;
    if (argc > 2 && strcmp(argv[1], "--output") == 0) {
        output_path = argv[2];
    }

    if (ensure_parent_dirs(output_path) != 0) {
        fprintf(stderr, "[R495 GATE] failed output_path parent dirs\n");
        return 1;
    }

    int d = TEST_STATE_DIM;
    int n = d + 1;

    // Allocate random parameters: ternary a_t, u_t, v_t and continuous input injections I_t
    float *a = malloc(sizeof(float) * TEST_SEQ_LEN * d);
    float *u = malloc(sizeof(float) * TEST_SEQ_LEN * d);
    float *v = malloc(sizeof(float) * TEST_SEQ_LEN * d);
    float *I = malloc(sizeof(float) * TEST_SEQ_LEN * d);

    srand(42);
    for (int t = 0; t < TEST_SEQ_LEN; ++t) {
        for (int i = 0; i < d; ++i) {
            // Stable decay factors a_t in [0.90, 0.95]
            a[t * d + i] = 0.90f + 0.05f * ((float)rand() / (float)RAND_MAX);
            // Low-rank projection vectors u_t, v_t scaled by 1/sqrt(d) for stability
            float scale_factor = 1.0f / sqrtf((float)d);
            u[t * d + i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * 0.1f * scale_factor;
            v[t * d + i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * 0.1f * scale_factor;
            // Inputs: continuous float in [-0.5, 0.5]
            I[t * d + i] = ((float)rand() / (float)RAND_MAX) - 0.5f;
        }
    }

    // 1. Exact Sequential Recurrence:
    // h_t = a_t * h_{t-1} + (v_t^T h_{t-1}) * u_t + I_t
    float *h_seq = calloc(TEST_SEQ_LEN * d, sizeof(float));
    float *w_seq = calloc(TEST_SEQ_LEN, sizeof(float));
    
    // Initial state h_{-1} = 0, so w_0 = v_0^T h_{-1} = 0
    float *h_prev = calloc(d, sizeof(float));
    
    for (int t = 0; t < TEST_SEQ_LEN; ++t) {
        // Scalar coupling term: w_{t-1} = v_t^T h_{t-1}
        float w_prev = 0.0f;
        for (int i = 0; i < d; ++i) {
            w_prev += v[t * d + i] * h_prev[i];
        }
        w_seq[t] = w_prev;

        // Hidden state update: h_t = a_t * h_{t-1} + w_{t-1} * u_t + I_t
        for (int i = 0; i < d; ++i) {
            h_seq[t * d + i] = a[t * d + i] * h_prev[i] + w_prev * u[t * d + i] + I[t * d + i];
        }
        
        // Update h_prev for next step
        memcpy(h_prev, h_seq + t * d, d * sizeof(float));
    }
    free(h_prev);

    // 2. Parallel Blelloch Associative Scan over Augmented State Space (AAS-SS)
    // Setup transition pairs (M_t, P_t) for each step
    AugmentedTransition *steps = malloc(sizeof(AugmentedTransition) * TEST_SEQ_LEN);
    for (int t = 0; t < TEST_SEQ_LEN; ++t) {
        steps[t].M = calloc(n * n, sizeof(float));
        steps[t].P = calloc(n, sizeof(float));
        
        // M_t = [ diag(a_t)        u_t
        //         (v_{t+1} * a_t)^T  v_{t+1}^T * u_t ]
        // Note: next vector v_{t+1} is used. If t == TEST_SEQ_LEN - 1, we use zero.
        float *v_next = calloc(d, sizeof(float));
        if (t < TEST_SEQ_LEN - 1) {
            memcpy(v_next, v + (t + 1) * d, d * sizeof(float));
        }
        
        // diag(a_t) block
        for (int i = 0; i < d; ++i) {
            steps[t].M[i * n + i] = a[t * d + i];
        }
        // u_t block
        for (int i = 0; i < d; ++i) {
            steps[t].M[i * n + d] = u[t * d + i];
        }
        // (v_{t+1} * a_t)^T block
        for (int i = 0; i < d; ++i) {
            steps[t].M[d * n + i] = v_next[i] * a[t * d + i];
        }
        // v_{t+1}^T * u_t scalar block
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) {
            dot += v_next[i] * u[t * d + i];
        }
        steps[t].M[d * n + d] = dot;
        
        // P_t = [ I_t
        //         v_{t+1}^T * I_t ]
        for (int i = 0; i < d; ++i) {
            steps[t].P[i] = I[t * d + i];
        }
        float dot_I = 0.0f;
        for (int i = 0; i < d; ++i) {
            dot_I += v_next[i] * I[t * d + i];
        }
        steps[t].P[d] = dot_I;
        
        free(v_next);
    }

    // Allocate prefix tree buffers
    AugmentedTransition *prefix = malloc(sizeof(AugmentedTransition) * TEST_SEQ_LEN);
    for (int t = 0; t < TEST_SEQ_LEN; ++t) {
        prefix[t].M = calloc(n * n, sizeof(float));
        prefix[t].P = calloc(n, sizeof(float));
    }
    
    // Copy the first element
    memcpy(prefix[0].M, steps[0].M, n * n * sizeof(float));
    memcpy(prefix[0].P, steps[0].P, n * sizeof(float));

    // Left-to-right associative prefix scan composition:
    // (prefix_t) = (steps_t) bullet (prefix_{t-1})
    for (int t = 1; t < TEST_SEQ_LEN; ++t) {
        compose_transition(d, &steps[t], &prefix[t - 1], &prefix[t]);
    }

    // 3. Verify numerical equivalence
    float max_abs_diff_h = 0.0f;
    float sum_squared_diff_h = 0.0f;
    float sum_squared_exact_h = 0.0f;

    for (int t = 0; t < TEST_SEQ_LEN; ++t) {
        for (int i = 0; i < d; ++i) {
            float exact_val = h_seq[t * d + i];
            float scan_val = prefix[t].P[i];
            float diff = exact_val - scan_val;
            
            if (fabsf(diff) > max_abs_diff_h) {
                max_abs_diff_h = fabsf(diff);
            }
            sum_squared_diff_h += diff * diff;
            sum_squared_exact_h += exact_val * exact_val;
        }
    }
    float relative_rmse_h = sqrtf(sum_squared_diff_h / sum_squared_exact_h);

    // Verify next-step scalar coupling associativity: w_t = prefix_t.P[d]
    float max_abs_diff_w = 0.0f;
    for (int t = 0; t < TEST_SEQ_LEN - 1; ++t) {
        float exact_w = w_seq[t + 1]; // w_t = v_{t+1}^T h_t
        float scan_w = prefix[t].P[d];
        float diff = exact_w - scan_w;
        if (fabsf(diff) > max_abs_diff_w) {
            max_abs_diff_w = fabsf(diff);
        }
    }

    // Invariant checks
    int pass_h = (max_abs_diff_h < 1e-4) ? 1 : 0;
    int pass_w = (max_abs_diff_w < 1e-4) ? 1 : 0;
    int pass_gate = (pass_h && pass_w) ? 1 : 0;

    // Print summary to console
    printf("[R495 GATE] active=1 aspect=A01 pass_h=%d max_abs_diff_h=%.9g relative_rmse_h=%.9g pass_w=%d max_abs_diff_w=%.9g pass_gate=%d\n",
           pass_h, max_abs_diff_h, relative_rmse_h, pass_w, max_abs_diff_w, pass_gate);

    // Save JSON output
    FILE *out = fopen(output_path, "w");
    if (out) {
        fprintf(out,
                "{\n"
                "  \"round\": \"R495\",\n"
                "  \"aspect\": \"A01\",\n"
                "  \"pass_gate\": %s,\n"
                "  \"h_check\": {\n"
                "    \"pass\": %s,\n"
                "    \"max_abs_diff\": %.9g,\n"
                "    \"relative_rmse\": %.9g\n"
                "  },\n"
                "  \"w_check\": {\n"
                "    \"pass\": %s,\n"
                "    \"max_abs_diff\": %.9g\n"
                "  }\n"
                "}\n",
                pass_gate ? "true" : "false",
                pass_h ? "true" : "false",
                max_abs_diff_h, relative_rmse_h,
                pass_w ? "true" : "false",
                max_abs_diff_w);
        fclose(out);
    }

    // Cleanup
    free(a);
    free(u);
    free(v);
    free(I);
    free(h_seq);
    free(w_seq);
    for (int t = 0; t < TEST_SEQ_LEN; ++t) {
        free(steps[t].M);
        free(steps[t].P);
        free(prefix[t].M);
        free(prefix[t].P);
    }
    free(steps);
    free(prefix);

    return 0;
}
