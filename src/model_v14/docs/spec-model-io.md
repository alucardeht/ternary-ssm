# Specification: src/model_v12/src/io/checkpoint.c

## Component Scope

Handles all persistence for v12: native binary checkpoints, v11 PyTorch `.pt` import,
and a portable export format. This is a pure C (C99) module with no runtime dependencies
outside the C standard library and optionally `zlib` for optional compression.

---

## Architecture Context

Before reading this spec, the key constraints from `architecture.md` are:

- Weights are ternary {-1, 0, +1} packed as INT2 (4 weights per byte)
- Shadow weights are fp32 (training only, not needed for inference checkpoints)
- Crystallization masks: per-weight uint8 flag, 1 = frozen (|logit| > 3), 0 = active
- Step counts per layer (used for per-layer LR scheduling, innovation #10 and #15)
- Model dimensions: 6 layers, d_model=1024, d_state=128, vocab=32064
- Total weight bytes (INT2): ~16.6 MB inference, ~16.6 MB + 25 MB shadow = ~41.6 MB training
- No PyTorch dependency at runtime. v11 import is a one-time migration path.

---

## File Structures

### TrainState (defined in training module, referenced here)

```c
typedef struct {
    uint32_t step;           // global step counter
    float    lr_base;        // base learning rate (default 0.1 for MVUE)
    float    lr_per_layer[MAX_LAYERS]; // per-layer LR after eigenspectrum scheduling
    float    loss_history[LOSS_HISTORY_LEN]; // ring buffer of recent losses
    uint32_t loss_history_head;              // ring buffer write pointer
    float    eval_gap;       // last measured eval-train gap (red flag if > 0.05)
    uint32_t fp_stage;       // current FP training stage (0..n_layers-1)
    uint8_t  fp_completed;   // 1 = FP pass done, ready for BCD
} TrainState;

#define MAX_LAYERS        32
#define LOSS_HISTORY_LEN  1024
```

### Model (defined in model module, referenced here)

```c
typedef struct SSMLayer {
    // INT2 packed ternary weights (4 weights per byte, row-major)
    uint8_t *w_in_int2;    // [d_model * d_state / 4] bytes
    uint8_t *w_out_int2;   // [d_state * d_model / 4] bytes
    uint8_t *w_conv_int2;  // [conv_kernel * d_state / 4] bytes

    // FP32 SSM dynamics (not ternary — diagonal A must stay fp32)
    float   *A;            // [d_state] diagonal values
    float   *B;            // [d_state] input projection
    float   *C;            // [d_model] output projection

    // MTP matrices (FP32, innovation #4)
    float   *mtp_U;        // [d_state * MTP_RANK]
    float   *mtp_V;        // [d_state * MTP_RANK]

    // Shadow weights (training only, NULL in inference mode)
    float   *shadow_w_in;  // [d_model * d_state] fp32 logits
    float   *shadow_w_out; // [d_state * d_model] fp32 logits
    float   *shadow_w_conv;// [conv_kernel * d_state] fp32 logits

    // Crystallization masks (training only, NULL in inference mode)
    uint8_t *crystal_mask_in;  // [d_model * d_state] 1=frozen
    uint8_t *crystal_mask_out; // [d_state * d_model] 1=frozen
} SSMLayer;

typedef struct {
    uint8_t   *embed_int2;   // [vocab * d_model / 4] bytes
    uint8_t   *lm_head_int2; // [d_model * vocab / 4] bytes
    float     *final_norm;   // [d_model] fp32 scale
    SSMLayer  *layers;       // [n_layers]
    uint32_t   n_layers;
    uint32_t   d_model;
    uint32_t   d_state;
    uint32_t   vocab_size;
    uint32_t   conv_kernel;
    uint32_t   mtp_rank;
} Model;
```

---

## Native File Format

### Magic and Version

```
MAGIC:   0x56 0x31 0x32 0x43  ("V12C" — v12 C format)
VERSION: uint32_t, currently 1
```

Version 1 is the only version. Future breaking changes increment this field.
Reader must reject files with unknown version rather than silently misbehave.

### Section Layout

The file is a sequence of tagged sections. Each section has a fixed header:

```c
typedef struct {
    uint32_t section_id;   // see section IDs below
    uint32_t flags;        // bitfield: SECTION_FLAG_COMPRESSED = 0x01
    uint64_t byte_length;  // length of section body in bytes (after decompression if applicable)
    uint64_t stored_length;// length of section body as stored on disk (same as byte_length if not compressed)
} SectionHeader;
```

All multi-byte integers are **little-endian**. This is non-negotiable: v12 targets x86
and ARM consumer hardware, both little-endian. No byte-swapping logic needed.

Section IDs:

```c
#define SECTION_FILE_HEADER  0x0001  // must be first section
#define SECTION_MODEL_CONFIG 0x0002
#define SECTION_WEIGHTS_INT2 0x0010  // ternary packed weights (inference + training)
#define SECTION_WEIGHTS_SSM  0x0011  // fp32 A, B, C, MTP matrices
#define SECTION_SHADOW       0x0020  // fp32 shadow logits (training only)
#define SECTION_CRYSTAL      0x0021  // crystallization masks (training only)
#define SECTION_TRAIN_STATE  0x0030  // TrainState struct (training only)
#define SECTION_METADATA     0xFF00  // JSON string (null-terminated, UTF-8)
#define SECTION_EOF          0xFFFF  // sentinel, no body
```

### Section: FILE_HEADER

```c
typedef struct {
    uint8_t  magic[4];      // "V12C"
    uint32_t version;       // 1
    uint32_t n_layers;
    uint32_t d_model;
    uint32_t d_state;
    uint32_t vocab_size;
    uint32_t conv_kernel;
    uint32_t mtp_rank;
    uint32_t checkpoint_type; // 0=inference, 1=training
    uint64_t created_at;    // unix timestamp (seconds)
    uint64_t step;          // training step at save time (0 for inference-only)
    uint8_t  reserved[32];  // zero-padded, reserved for future fields
} FileHeaderBody;
```

`checkpoint_type = 0` (inference): only INT2 + SSM_FP32 + METADATA sections present.
`checkpoint_type = 1` (training): all sections present including SHADOW, CRYSTAL, TRAIN_STATE.

### Section: MODEL_CONFIG

JSON string encoding all hyperparameters that affect weight layout:

```json
{
  "n_layers": 6,
  "d_model": 1024,
  "d_state": 128,
  "vocab_size": 32064,
  "conv_kernel": 4,
  "mtp_rank": 20,
  "d_model_version": "v12.1"
}
```

The reader must verify these values match the Model struct being loaded into.
A mismatch is a hard error, not a warning.

### Section: WEIGHTS_INT2

One contiguous blob. Layout is strictly ordered and deterministic:

```
[embed_int2]         vocab * d_model / 4 bytes
[lm_head_int2]       d_model * vocab / 4 bytes
[layer_0.w_in_int2]  d_model * d_state / 4 bytes
[layer_0.w_out_int2] d_state * d_model / 4 bytes
[layer_0.w_conv_int2] conv_kernel * d_state / 4 bytes
[layer_1.w_in_int2]  ...
...
[layer_N-1.w_conv_int2]
```

Bit encoding per byte (matches v11 `pack_ternary_bitmask` semantics but uses INT2, not bitmask):

```
bits [1:0] = weight 0 encoded as: 0b00 = 0, 0b01 = +1, 0b11 = -1
bits [3:2] = weight 1
bits [5:4] = weight 2
bits [7:6] = weight 3
```

This is the canonical v12 encoding. The v11 bitmask format (mask + sign bits) is different
and must be converted during `checkpoint_import_v11`.

### Section: WEIGHTS_SSM

FP32 values for SSM dynamics. Layout:

```
For k = 0..n_layers-1:
  [layer_k.A]     d_state floats   (diagonal SSM decay)
  [layer_k.B]     d_state floats   (input projection)
  [layer_k.C]     d_model floats   (output projection)
  [layer_k.mtp_U] d_state * mtp_rank floats
  [layer_k.mtp_V] d_state * mtp_rank floats
[final_norm]      d_model floats
```

These are always fp32. No quantization here — A must stay fp32 for stability (lessons-learned §8).

### Section: SHADOW

FP32 shadow logits for ternary weights. Only in training checkpoints.

```
For k = 0..n_layers-1:
  [layer_k.shadow_w_in]   d_model * d_state floats
  [layer_k.shadow_w_out]  d_state * d_model floats
  [layer_k.shadow_w_conv] conv_kernel * d_state floats
```

### Section: CRYSTAL

Crystallization masks. One byte per weight: 0 = active, 1 = crystallized (|logit| > 3).

```
For k = 0..n_layers-1:
  [layer_k.crystal_mask_in]  d_model * d_state bytes
  [layer_k.crystal_mask_out] d_state * d_model bytes
```

Note: conv weights intentionally excluded from crystallization (too small, ~3KB/layer total).

### Section: TRAIN_STATE

Serialized `TrainState` struct, written as a fixed-layout C struct (no padding issues — all
fields are naturally aligned). Total size: 4 + 4 + MAX_LAYERS*4 + LOSS_HISTORY_LEN*4 + 4 + 4 + 4 + 1
= 4237 bytes for MAX_LAYERS=32, LOSS_HISTORY_LEN=1024.

### Section: METADATA

Null-terminated UTF-8 JSON string. Contains human-readable info (not used by loader):

```json
{
  "created_by": "v12/checkpoint.c",
  "arch_hash": "<sha256 of architecture.md at build time>",
  "git_hash": "<optional>",
  "notes": ""
}
```

---

## Function Specifications

### `checkpoint_save`

```c
int checkpoint_save(const char *path, const Model *model, const TrainState *state);
```

**Purpose:** Write a complete native checkpoint. If `state` is NULL, writes an
inference-only checkpoint (no SHADOW, CRYSTAL, TRAIN_STATE sections).

**Behavior:**
1. Open path for writing (create or truncate). Fail if directory does not exist.
2. Write sections in this exact order: FILE_HEADER, MODEL_CONFIG, WEIGHTS_INT2, WEIGHTS_SSM,
   and if `state != NULL`: SHADOW, CRYSTAL, TRAIN_STATE. Then METADATA. Then EOF sentinel.
3. Flush and fsync before closing. A checkpoint is not considered written until fsync returns.
4. Write to a temp file (`path + ".tmp"`) then rename atomically. This prevents a corrupt
   checkpoint from a crash mid-write.

**Error handling:** Return 0 on success. On error, return a negative errno-compatible value,
clean up the temp file, and do NOT overwrite the existing checkpoint at `path`.

**Compression:** SECTION_FLAG_COMPRESSED is supported but disabled by default for INT2 and
SSM sections (they are already compact). Shadow weights may benefit from compression but
this is left to a future version. The flag field is reserved for forward compatibility.

**Return values:**
```c
#define CKPT_OK            0
#define CKPT_ERR_IO       -1   // file open/write/fsync failure
#define CKPT_ERR_NOMEM    -2   // malloc failure
#define CKPT_ERR_NULL     -3   // required pointer argument is NULL
```

### `checkpoint_load`

```c
int checkpoint_load(const char *path, Model *model, TrainState *state);
```

**Purpose:** Load a native checkpoint into an already-allocated Model and optionally
TrainState. The Model struct must already have its dimension fields set correctly (n_layers,
d_model, etc.) and all weight buffers allocated.

**Behavior:**
1. Open file, read and validate FILE_HEADER: check magic, version, and that dimensions
   match `model->n_layers`, `model->d_model`, etc.
2. Read sections in sequence. Unknown section IDs are skipped (forward compatibility).
3. If `state` is NULL, skip SHADOW, CRYSTAL, and TRAIN_STATE sections.
4. If `checkpoint_type = 0` (inference) but `state != NULL`, return CKPT_ERR_TYPE.
5. After loading WEIGHTS_INT2, recompute the CSR index arrays for ternary matmul.
   These are not stored in the checkpoint because they are derived from INT2 data.

**Validation after load:**
- Verify all A values are in (0, 1) — enforce diagonal SSM stability invariant.
- Verify `eval_gap` in TrainState is not NaN. If it is, zero it out with a warning log.
- Do NOT verify shadow weight ranges or crystallization counts. These are trusted as written.

**Return values:**
```c
#define CKPT_ERR_MAGIC    -4   // wrong magic bytes
#define CKPT_ERR_VERSION  -5   // unsupported version
#define CKPT_ERR_DIM      -6   // dimension mismatch between file and model
#define CKPT_ERR_CORRUPT  -7   // section length inconsistency or truncated file
#define CKPT_ERR_TYPE     -8   // tried to load train state from inference checkpoint
```

### `checkpoint_import_v11`

```c
int checkpoint_import_v11(const char *pt_path, Model *model);
```

**Purpose:** Read a PyTorch `.pt` file saved by v11 and populate the Model's INT2 weight
buffers and SSM fp32 buffers. Shadow weights and training state are not imported (v11
training state is incompatible with v12's MVUE + crystallization model).

**The .pt format challenge:** PyTorch `.pt` files are Python pickle format. A minimal
C pickle reader is required — only the subset of opcodes used by `torch.save` with
`weights_only=True` and `map_location='cpu'`.

Required pickle opcode support:
```
PROTO       (opcode 0x80) — version handshake
FRAME       (opcode 0x95) — framing
MARK        (opcode 0x28) — mark stack
DICT        (opcode 0x64) — build dict
SETITEMS    (opcode 0x75) — dict populate
SHORT_BINUNICODE (0x8C)   — string key
BININT1     (0x4B)        — small int
GLOBAL      (opcode 0x63) — class reference (used for tensor type)
REDUCE      (opcode 0x52) — call constructor
TUPLE       (opcode 0x74) — tuple from mark
BINPUT/GET  (0x71/0x68)   — memo
STOP        (opcode 0x2E) — end
```

Tensor data is stored as raw binary in the ZIP archive within the `.pt` file.
PyTorch `.pt` files are ZIP archives; use miniz (single-file MIT library) for extraction.

**Key mapping** (v11 state_dict key → v12 field):

```
layers.{k}.in_proj.weight        → model->layers[k].w_in_int2
layers.{k}.out_proj.weight       → model->layers[k].w_out_int2
layers.{k}.conv1d.weight         → model->layers[k].w_conv_int2
layers.{k}.ssm.A_log             → model->layers[k].A  (apply exp() then clamp to (0,1))
layers.{k}.ssm.B                 → model->layers[k].B
layers.{k}.ssm.C                 → model->layers[k].C
layers.{k}.mtp.U                 → model->layers[k].mtp_U
layers.{k}.mtp.V                 → model->layers[k].mtp_V
embed_tokens.weight              → model->embed_int2
lm_head.weight                   → model->lm_head_int2
norm.weight                      → model->final_norm
```

**Ternary conversion:** v11 stores shadow weights (fp32 logits), not packed INT2.
The import function must apply the ternary rounding:

```c
static int8_t round_ternary(float logit) {
    if (logit > 0.5f) return  1;
    if (logit < -0.5f) return -1;
    return 0;
}
```

Then pack 4 int8 values into 1 uint8 using the v12 INT2 encoding described above.

**A_log conversion:** v11 stores log(A) for numerical stability. v12 stores A directly.

```c
float a = expf(a_log_value);
a = fmaxf(0.001f, fminf(0.999f, a));  // clamp to open interval (0, 1)
```

**Return values:** Returns CKPT_OK or CKPT_ERR_IO / CKPT_ERR_CORRUPT.
Returns `CKPT_ERR_DIM` if the model dimensions in the `.pt` file do not match
the Model struct.

**Note on v11 key name variations:** Different v11 experiments used slightly different
key names (some used `in_proj`, others `W_in`). The import function must handle both.
Maintain a static key alias table:

```c
static const char *KEY_ALIASES[][2] = {
    {"W_in",  "in_proj.weight"},
    {"W_out", "out_proj.weight"},
    {NULL, NULL}
};
```

### `checkpoint_export`

```c
int checkpoint_export(const char *path, const Model *model, const char *format);
```

**Purpose:** Export the model for use outside v12. Supported formats in v1:

- `"gguf"` — GGUF format (used by llama.cpp). Only inference weights (INT2 converted to
  Q2_K approximation). This enables running v12 models in llama.cpp without modification.
- `"raw"` — flat binary dump, one file per weight matrix, named by tensor key. Useful for
  debugging and for importing into NumPy/PyTorch for validation.
- `"onnx"` — not supported in v1, returns `CKPT_ERR_UNSUPPORTED`.

```c
#define CKPT_ERR_UNSUPPORTED -9  // requested format not implemented
```

**GGUF output:** GGUF is a well-defined binary format (specification at
github.com/ggerganov/ggml/blob/master/docs/gguf.md). Implement the header and tensor
section writer. The INT2 ternary packing does not match GGUF's Q2_K exactly — use Q8_0
(8-bit quantized) as the fallback for correctness, with a note in the metadata that weights
are ternary and can be re-quantized.

**Raw output:** Create a directory at `path`. Write one file per tensor:
```
embed.bin       — vocab * d_model / 4 bytes (INT2)
lm_head.bin     — d_model * vocab / 4 bytes (INT2)
layer_0_win.bin — ...
layer_0_wout.bin
layer_0_A.bin   — d_state * 4 bytes (fp32)
...
manifest.json   — lists all files and their shapes/dtypes
```

---

## Internal Helpers (not in public header)

```c
// Write a section to an open file descriptor
static int write_section(int fd, uint32_t id, const void *data, uint64_t len);

// Read and validate the next section header from fd
static int read_section_header(int fd, SectionHeader *out_hdr);

// Skip over a section body without reading it
static int skip_section(int fd, const SectionHeader *hdr);

// Pack 4 ternary int8 values into 1 uint8 using v12 INT2 encoding
static uint8_t pack_int2_byte(int8_t w0, int8_t w1, int8_t w2, int8_t w3);

// Unpack 1 uint8 into 4 ternary int8 values
static void unpack_int2_byte(uint8_t b, int8_t *w0, int8_t *w1, int8_t *w2, int8_t *w3);

// Convert v11 bitmask+sign format to v12 INT2 packed format
static int convert_bitmask_to_int2(
    const uint8_t *mask_bits, const uint8_t *sign_bits,
    uint32_t n_weights,
    uint8_t *out_int2);

// Compute byte offset of section by ID in an open file
static int64_t find_section_offset(int fd, uint32_t section_id);

// Minimal pickle reader: extract tensor data from .pt ZIP archive
// Returns allocated buffer (caller frees) and its byte length
static void *pickle_extract_tensor(
    const char *zip_path, const char *tensor_key, uint64_t *out_bytes);
```

---

## Memory Policy

- `checkpoint_save` allocates no heap memory beyond a small stack buffer for section
  headers. Weight data is written directly from model buffers.
- `checkpoint_load` writes directly into the model's pre-allocated buffers. It does NOT
  allocate new weight buffers. The caller is responsible for allocation.
- `checkpoint_import_v11` allocates one temporary buffer per tensor (for format conversion)
  then frees it immediately after packing into the Model.
- `checkpoint_export("gguf", ...)` allocates a single staging buffer sized to the largest
  tensor. Peak additional memory: `max(vocab * d_model / 4, d_model * vocab / 4)` bytes
  = 8.2 MB for v12 default dimensions.

---

## Error Handling Rules

No silent errors. Every path that can fail must return a non-zero error code.

Log format (write to stderr, not stdout):
```
[CKPT] ERROR: <function_name>: <description> (errno=<value if applicable>)
[CKPT] WARN:  <function_name>: <description>
[CKPT] INFO:  <function_name>: <description>
```

INFO logging is active only when compiled with `#define CKPT_VERBOSE 1`.

Critical invariant: after a failed `checkpoint_load`, the Model state is undefined.
The caller must treat a partially-loaded model as uninitialized and free/re-allocate.
Document this in the header comment.

---

## Thread Safety

None. All four public functions are not thread-safe. The caller is responsible for
serializing checkpoint operations. This is intentional — v12 trains on a single thread
(or single GPU with single host thread). No mutexes, no atomics.

---

## Header File: `src/model_v12/src/io/checkpoint.h`

```c
#ifndef V12_CHECKPOINT_H
#define V12_CHECKPOINT_H

#include <stdint.h>
#include "../core/model.h"
#include "../training/train_state.h"

// Return codes
#define CKPT_OK              0
#define CKPT_ERR_IO         -1
#define CKPT_ERR_NOMEM      -2
#define CKPT_ERR_NULL       -3
#define CKPT_ERR_MAGIC      -4
#define CKPT_ERR_VERSION    -5
#define CKPT_ERR_DIM        -6
#define CKPT_ERR_CORRUPT    -7
#define CKPT_ERR_TYPE       -8
#define CKPT_ERR_UNSUPPORTED -9

// Current native format version
#define CKPT_FORMAT_VERSION 1

/*
 * Save model to native v12 binary format.
 * If state is NULL, writes inference-only checkpoint (no shadow weights).
 * Uses atomic write (temp file + rename). fsync guaranteed before return.
 * Returns CKPT_OK or negative error code.
 */
int checkpoint_save(const char *path, const Model *model, const TrainState *state);

/*
 * Load native v12 checkpoint into pre-allocated model and optional state.
 * Model dimensions must be set before calling. Weight buffers must be allocated.
 * WARNING: on failure, model state is undefined. Re-allocate before retrying.
 * Returns CKPT_OK or negative error code.
 */
int checkpoint_load(const char *path, Model *model, TrainState *state);

/*
 * Import from v11 PyTorch .pt checkpoint (one-time migration).
 * Populates INT2 weights and SSM fp32 buffers only.
 * Shadow weights and training state are not imported.
 * Returns CKPT_OK or negative error code.
 */
int checkpoint_import_v11(const char *pt_path, Model *model);

/*
 * Export model to external format.
 * format: "gguf" | "raw"
 * "onnx" returns CKPT_ERR_UNSUPPORTED.
 * Returns CKPT_OK or negative error code.
 */
int checkpoint_export(const char *path, const Model *model, const char *format);

#endif /* V12_CHECKPOINT_H */
```

---

## Dependencies

| Dependency | Reason | Source |
|-----------|--------|--------|
| `<stdio.h>` | fopen, fwrite, fread | C99 stdlib |
| `<string.h>` | memcpy, memset | C99 stdlib |
| `<errno.h>` | errno values for IO errors | C99 stdlib |
| `<math.h>` | expf for A_log conversion | C99 stdlib |
| `miniz.h` | ZIP extraction for .pt import | single-file, MIT license |
| `<unistd.h>` | fsync, rename (POSIX) | POSIX.1 |

No dependency on PyTorch, CUDA, or any ML framework.
`miniz.h` is included as a vendored single-header file at `src/model_v12/vendor/miniz.h`.

---

## Implementation Order

1. `pack_int2_byte` + `unpack_int2_byte` — trivial, needed by everything else
2. `write_section` + `read_section_header` + `skip_section` — core I/O primitives
3. `checkpoint_save` (inference mode, state=NULL) — test first with raw output
4. `checkpoint_load` (inference mode) — round-trip test against step 3
5. `checkpoint_save` + `checkpoint_load` (training mode, with shadow + crystal)
6. `convert_bitmask_to_int2` — needed for v11 import
7. `pickle_extract_tensor` via miniz — hardest part, test with known .pt files
8. `checkpoint_import_v11` — depends on 6 and 7
9. `checkpoint_export("raw")` — straightforward once save works
10. `checkpoint_export("gguf")` — last, most format-specific

---

## Test Cases (for code-writer to implement alongside)

```c
// test_checkpoint.c

// T1: save + load round-trip (inference)
// Create model with known weights, save, load into fresh model, verify byte-exact match.

// T2: save + load round-trip (training)
// Same but include shadow weights, crystal masks, and train state.

// T3: dimension mismatch rejection
// Modify n_layers in header after save, attempt load, expect CKPT_ERR_DIM.

// T4: corrupt file detection
// Truncate file mid-section, attempt load, expect CKPT_ERR_CORRUPT.

// T5: atomic write safety
// Simulate crash mid-write (kill after temp file created), verify original checkpoint intact.

// T6: INT2 pack/unpack identity
// For all 81 combinations of 4 ternary values {-1,0,+1}^4, verify pack→unpack = identity.

// T7: v11 import (if .pt file available)
// Load known r248 checkpoint, verify A values in (0,1), spot-check a weight row.

// T8: export "raw" then re-import
// Export raw, read manifest.json, load each .bin, verify shapes and spot values.
```

---

## Known Constraints from v11 Lessons

**Lesson 13 (Never-again):** No `use_ternary=False` style flags. INT2 packing is always
active in `checkpoint_save`. If a caller wants fp32 weights, use `checkpoint_export("raw")`
which writes fp32 via shadow weights. The save path does not have a mode to skip packing.

**Lesson 16 (.pyc cache):** Not applicable to C, but the equivalent risk is stale object
files. The build system must use `-MMD -MP` dependency tracking to force recompilation when
this header changes.

**Crash safety:** The atomic write (temp + rename) directly addresses the scenario from
lesson 16 — a corrupt checkpoint from an interrupted write would have forced a complete
retraining run.

**A_log conversion:** When importing from v11, always clamp A to (0.001, 0.999).
A value of 0 or 1 causes SSM scan to either zero out state immediately or accumulate
without decay — both are catastrophic. The v11 experiments never hit this because PyTorch
used A_log (log-space parameterization prevents boundary crossing), but in C we store
A directly so the clamp is mandatory.
