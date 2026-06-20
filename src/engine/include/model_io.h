#ifndef MODEL_IO_H
#define MODEL_IO_H

#include "ternary_types.h"
#include "ssm_scan.h"
#include "attention.h"
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>

#define TRNM_MAGIC          0x4D4E5254
#define TRNM_VERSION        1
#define TRNM_HEADER_SIZE    256
#define TRNM_ALIGN          64
#define TRNM_MAX_SWA_LAYERS 32
#define TRNM_ALIGN_UP(x) (((x) + (TRNM_ALIGN - 1)) & ~((uint64_t)(TRNM_ALIGN - 1)))

#define TRNM_LAYER_SSM      0
#define TRNM_LAYER_SWA      1

#define TRNM_FLAG_TERNARY       (1u << 0)
#define TRNM_FLAG_HAS_LM_HEAD   (1u << 1)
#define TRNM_FLAG_LM_TERNARY    (1u << 2)
#define TRNM_FLAG_TSCS          (1u << 3)
#define TRNM_FLAG_SSD           (1u << 4)
#define TRNM_FLAG_RELU          (1u << 5)
#define TRNM_FLAG_BITSHIFT_NORM (1u << 6)

#define TRNM_LFLAG_HAS_ROUTER  (1u << 0)
#define TRNM_LFLAG_HAS_MOE     (1u << 1)
#define TRNM_LFLAG_IS_LTI      (1u << 2)

#define TRNM_OK                 0
#define TRNM_ERR_IO            -1
#define TRNM_ERR_MAGIC         -2
#define TRNM_ERR_VERSION       -3
#define TRNM_ERR_HEADER_CRC    -4
#define TRNM_ERR_LAYER_CRC     -5
#define TRNM_ERR_FILE_SIZE     -6
#define TRNM_ERR_ALLOC         -7
#define TRNM_ERR_TRUNCATED     -8
#define TRNM_ERR_INVALID       -9

#define TRNM_STATIC_ASSERT(cond, msg) typedef char static_assertion_##msg[(cond)?1:-1]

typedef struct {
    uint8_t     magic[4];
    uint32_t    version;
    uint32_t    flags;
    uint32_t    n_layers;
    uint32_t    vocab_size;
    uint32_t    d_model;
    uint32_t    d_state;
    uint32_t    d_conv;
    uint32_t    expand;
    uint32_t    dt_rank;
    uint32_t    n_heads;
    uint32_t    ssd_n_heads;
    uint32_t    window_size;
    uint32_t    max_seq_len;
    uint32_t    ternary_group_size;
    uint32_t    n_swa_layers;
    uint32_t    d_inner;
    uint32_t    mlp_hidden;
    uint32_t    head_dim;
    uint32_t    rope_dim;
    uint64_t    file_size;
    uint64_t    layer_table_offset;
    uint64_t    embedding_offset;
    uint64_t    final_norm_offset;
    uint64_t    lm_head_offset;
    uint32_t    swa_indices[TRNM_MAX_SWA_LAYERS];
    uint32_t    reserved;
    uint32_t    header_crc32;
} trnm_header_t;

TRNM_STATIC_ASSERT(sizeof(trnm_header_t) == 256, header_size_256);

typedef struct {
    uint8_t     layer_type;
    uint8_t     layer_flags;
    uint16_t    padding;
    uint32_t    data_size;
    uint64_t    data_offset;
    uint32_t    data_crc32;
    uint32_t    reserved;
} trnm_layer_entry_t;

TRNM_STATIC_ASSERT(sizeof(trnm_layer_entry_t) == 24, layer_entry_24);

typedef struct {
    uint32_t    rows;
    uint32_t    cols;
    uint32_t    packed_size;
    uint32_t    reserved;
} trnm_ternary_hdr_t;

typedef struct {
    trnm_header_t           header;
    trnm_layer_entry_t*     layer_table;
    float*                  embedding;
    uint8_t*                layer_types;
    ssm_layer_t**           ssm_layers;
    attn_layer_t**          attn_layers;
    float**                 pre_norms;
    float*                  final_norm;
    ternary_matrix_t*       lm_head_ternary;
    float*                  lm_head_float;
} trnm_model_t;

int trnm_read_header(const char* path, trnm_header_t* header);
int trnm_validate_header(const trnm_header_t* header);
trnm_model_t* trnm_load(const char* path);
void trnm_free(trnm_model_t* model);
int trnm_save(const char* path, const trnm_model_t* model);
int trnm_load_layer(const char* path, const trnm_header_t* header,
                    const trnm_layer_entry_t* entry, void** out_layer, float** out_norm);
int trnm_load_layer_table(const char* path, const trnm_header_t* header,
                          trnm_layer_entry_t** out_table);
uint32_t trnm_crc32(const void* data, size_t len);
const char* trnm_strerror(int err);

#endif
