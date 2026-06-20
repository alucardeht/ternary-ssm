/* R356 F3 — Gradient covariance rank measurement (read-only diagnostic).
 *
 * Measures sigma1^2 / ||G||_F^2 and (sigma1^2 + sigma2^2) / ||G||_F^2 on the
 * clipped FP32 gradient of model->layers[0].dw_in via deterministic power
 * iteration. Gated by both the compile flag V12_F3_MEASURE and the runtime
 * env var V12_F3_MEASURE (either missing => pure no-op).
 *
 * This header has zero side effects when V12_F3_MEASURE is undefined.
 */
#ifndef V12_OPTIMIZER_F3_MEASURE_H
#define V12_OPTIMIZER_F3_MEASURE_H

#include "model.h"

#ifdef V12_F3_MEASURE

void optimizer_f3_measure_step(const Model *model, int step);
void optimizer_f3_measure_summary(void);

#else  /* V12_F3_MEASURE undefined: hook is compiled out entirely */

static inline void optimizer_f3_measure_step(const Model *model, int step) {
    (void)model; (void)step;
}
static inline void optimizer_f3_measure_summary(void) { }

#endif /* V12_F3_MEASURE */

#endif /* V12_OPTIMIZER_F3_MEASURE_H */
