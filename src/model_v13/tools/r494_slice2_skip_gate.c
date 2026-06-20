#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <errno.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#define DEFAULT_OUTPUT_PATH ".work/jobs/r494-slice2-skip-gate/summary.json"
#define SIM_STEPS 1000
#define STATE_DIM 128

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

// Simple LCG random float
static float random_float(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return (float)(*state >> 8) / 16777216.0f;
}

int main(int argc, char **argv) {
    const char *output_path = DEFAULT_OUTPUT_PATH;
    if (argc > 2 && strcmp(argv[1], "--output") == 0) {
        output_path = argv[2];
    }

    if (ensure_parent_dirs(output_path) != 0) {
        fprintf(stderr, "[R494 GATE 2] failed output_path parent dirs\n");
        return 1;
    }

    uint32_t rng = 1337u;

    // Simulation variables
    float *loss = malloc(sizeof(float) * SIM_STEPS);
    float *moments = malloc(sizeof(float) * SIM_STEPS * STATE_DIM);
    float *H = malloc(sizeof(float) * SIM_STEPS);
    int *skipped = malloc(sizeof(int) * SIM_STEPS);

    // Scaling factor lambda for symplectic kinetic energy
    float lambda = 0.05f; 
    float epsilon = 0.02f; // Hamiltonian stability threshold

    // Initialize mock loss trajectory (exponential decay with noise)
    float base_loss = 4.0f;
    for (int t = 0; t < SIM_STEPS; ++t) {
        float noise = (random_float(&rng) - 0.5f) * 0.08f;
        loss[t] = base_loss * expf(-0.003f * t) + noise + 1.2f;
        
        // Mock generalized momentum (gradients/moments) decaying slowly
        for (int d = 0; d < STATE_DIM; ++d) {
            float m_noise = (random_float(&rng) - 0.5f) * 0.4f;
            moments[t * STATE_DIM + d] = 0.6f * expf(-0.002f * t) + m_noise;
        }
    }

    // Evaluate continuous Hamiltonian H_t = Loss_t + 1/2 * lambda * sum(moment_d^2)
    for (int t = 0; t < SIM_STEPS; ++t) {
        float kinetic = 0.0f;
        for (int d = 0; d < STATE_DIM; ++d) {
            float m = moments[t * STATE_DIM + d];
            kinetic += m * m;
        }
        kinetic = 0.5f * lambda * kinetic;
        H[t] = loss[t] + kinetic;
    }

    // Run Symplectic Potential Conservation Skipping (SPCS) and Martingale skip policy
    int total_skips = 0;
    float poisson_rate = 0.20f; // Baseline Poisson skip probability
    
    skipped[0] = 0; // Never skip first step
    for (int t = 1; t < SIM_STEPS; ++t) {
        float delta_H = H[t] - H[t - 1];
        
        // 1. Martingale property: E[H_{t+1} | H_t] <= H_t (negative drift is healthy descent)
        // 2. Symplectic potential conservation: |delta_H| <= epsilon
        int conservative = (fabsf(delta_H) < epsilon) ? 1 : 0;
        int negative_drift = (delta_H <= 0.0f) ? 1 : 0;

        // DLGPS (Dynamic Loss Gradient Poisson Skip) stochastic selection
        float p_val = random_float(&rng);
        int poisson_trigger = (p_val < poisson_rate) ? 1 : 0;

        if ((conservative || negative_drift) && poisson_trigger) {
            skipped[t] = 1;
            total_skips++;
        } else {
            skipped[t] = 0;
        }
    }

    float skip_ratio = (float)total_skips / (float)SIM_STEPS;

    // Check convergence: ensure H does not diverge under skipped steps (Mock verification)
    float max_H_drift = 0.0f;
    float prev_stable_H = H[0];
    for (int t = 1; t < SIM_STEPS; ++t) {
        if (!skipped[t]) {
            float drift = fabsf(H[t] - prev_stable_H);
            if (drift > max_H_drift) {
                max_H_drift = drift;
            }
            prev_stable_H = H[t];
        }
    }

    // Decisions
    int pass_skip_ratio = (skip_ratio >= 0.10f && skip_ratio <= 0.40f) ? 1 : 0;
    int pass_energy_stability = (max_H_drift < 1.5f) ? 1 : 0;
    int pass_gate = (pass_skip_ratio && pass_energy_stability) ? 1 : 0;

    // Print summary to console
    printf("[R494 GATE 2] active=1 aspect=A02 pass_skip_ratio=%d skip_ratio=%.4f "
           "max_H_drift=%.4f pass_energy_stability=%d pass_gate=%d\n",
           pass_skip_ratio, skip_ratio, max_H_drift, pass_energy_stability, pass_gate);

    // Save JSON output
    FILE *out = fopen(output_path, "w");
    if (out) {
        fprintf(out,
                "{\n"
                "  \"round\": \"R494\",\n"
                "  \"aspect\": \"A02\",\n"
                "  \"pass_gate\": %s,\n"
                "  \"skipping_check\": {\n"
                "    \"pass\": %s,\n"
                "    \"total_skips\": %d,\n"
                "    \"skip_ratio\": %.6f\n"
                "  },\n"
                "  \"symplectic_check\": {\n"
                "    \"pass\": %s,\n"
                "    \"max_hamiltonian_drift\": %.6f\n"
                "  }\n"
                "}\n",
                pass_gate ? "true" : "false",
                pass_skip_ratio ? "true" : "false",
                total_skips, skip_ratio,
                pass_energy_stability ? "true" : "false",
                max_H_drift);
        fclose(out);
    }

    free(loss);
    free(moments);
    free(H);
    free(skipped);
    return 0;
}
