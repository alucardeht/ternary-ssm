import re
with open('src/model_v12/src/core/ssm_scan.c', 'r') as f:
    text = f.read()

# Pattern 1 (SISSF loop)
p1 = r"""            // Output projection Y = C @ H
            for \(int j = 0; j < d_inner; j\+\+\) \{
                float acc = 0\.0f;
                const float \*C_row = C \+ j \* d_state;
                for \(int i = 0; i < d_state; i\+\+\) acc \+= C_row\[i\] \* h\[i\];
                y_t\[j\] = acc;
            \}"""

r1 = """            // Q-05: Recurrent State Singular Value Soft-Bounding
            float h_q05[V12_MAX_D_STATE];
            float p4_sum = 0.0f;
            for (int i = 0; i < d_state; i++) {
                float h2 = h[i] * h[i];
                p4_sum += h2 * h2;
                h_q05[i] = h[i];
            }
            float norm_p4 = powf(p4_sum, 0.25f);
            float q05_scale = 1.0f / fmaxf(1.0f, norm_p4 / 10.0f);
            for (int i = 0; i < d_state; i++) h_q05[i] *= q05_scale;

            // Output projection Y = C @ H
            for (int j = 0; j < d_inner; j++) {
                float acc = 0.0f;
                const float *C_row = C + j * d_state;
                for (int i = 0; i < d_state; i++) acc += C_row[i] * h_q05[i];
                y_t[j] = acc;
            }"""

# Pattern 2 (Sequential fallback loop)
p2 = r"""            for \(int j = 0; j < d_inner; j\+\+\) \{
                float acc = 0\.0f;
                const float \*C_row = C \+ j \* d_state;
                for \(int i = 0; i < d_state; i\+\+\) acc \+= C_row\[i\] \* h\[i\];
                y_t\[j\] = acc;
            \}
        \}
        if \(h_cache != NULL"""

r2 = """            // Q-05: Recurrent State Singular Value Soft-Bounding
            float h_q05[V12_MAX_D_STATE];
            float p4_sum = 0.0f;
            for (int i = 0; i < d_state; i++) {
                float h2 = h[i] * h[i];
                p4_sum += h2 * h2;
                h_q05[i] = h[i];
            }
            float norm_p4 = powf(p4_sum, 0.25f);
            float q05_scale = 1.0f / fmaxf(1.0f, norm_p4 / 10.0f);
            for (int i = 0; i < d_state; i++) h_q05[i] *= q05_scale;

            for (int j = 0; j < d_inner; j++) {
                float acc = 0.0f;
                const float *C_row = C + j * d_state;
                for (int i = 0; i < d_state; i++) acc += C_row[i] * h_q05[i];
                y_t[j] = acc;
            }
        }
        if (h_cache != NULL"""

# Pattern 3 (Parallel L loop)
p3 = r"""            float \*y_t = y \+ t \* d_inner;
#ifdef SSM_SIMD_AVX2
            for \(int j = 0; j < d_inner; j \+= 4\) \{
                __m256 acc0 = _mm256_setzero_ps\(\);
                __m256 acc1 = _mm256_setzero_ps\(\);
                __m256 acc2 = _mm256_setzero_ps\(\);
                __m256 acc3 = _mm256_setzero_ps\(\);
                for \(int i = 0; i < d_state; i \+= 8\) \{
                    __m256 vh = _mm256_loadu_ps\(h_local \+ i\);"""

r3 = """            float *y_t = y + t * d_inner;
            // Q-05
            float h_q05_local[V12_MAX_D_STATE];
            float p4_sum = 0.0f;
            for (int i = 0; i < d_state; i++) {
                float h2 = h_local[i] * h_local[i];
                p4_sum += h2 * h2;
                h_q05_local[i] = h_local[i];
            }
            float norm_p4 = powf(p4_sum, 0.25f);
            float q05_scale = 1.0f / fmaxf(1.0f, norm_p4 / 10.0f);
            for (int i = 0; i < d_state; i++) h_q05_local[i] *= q05_scale;

#ifdef SSM_SIMD_AVX2
            for (int j = 0; j < d_inner; j += 4) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                for (int i = 0; i < d_state; i += 8) {
                    __m256 vh = _mm256_loadu_ps(h_q05_local + i);"""

# Pattern 4 (Non-SIMD branch of L loop)
p4 = r"""#else
            for \(int j = 0; j < d_inner; j\+\+\) \{
                float acc = 0\.0f;
                const float \*C_row = C \+ j \* d_state;
                for \(int i = 0; i < d_state; i\+\+\) acc \+= C_row\[i\] \* h_local\[i\];
                y_t\[j\] = acc;
            \}
#endif"""

r4 = """#else
            for (int j = 0; j < d_inner; j++) {
                float acc = 0.0f;
                const float *C_row = C + j * d_state;
                for (int i = 0; i < d_state; i++) acc += C_row[i] * h_q05_local[i];
                y_t[j] = acc;
            }
#endif"""

text = re.sub(p1, r1, text, count=1)
text = re.sub(p2, r2, text, count=1)
text = re.sub(p3, r3, text, count=1)
text = re.sub(p4, r4, text, count=1)

with open('src/model_v12/src/core/ssm_scan.c', 'w') as f:
    f.write(text)
print("patched ssm_scan.c")
