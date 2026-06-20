#include "model_io.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc_64(size) _aligned_malloc(size, 64)
#define aligned_free_64(ptr) _aligned_free(ptr)
#else
#define aligned_alloc_64(size) aligned_alloc(64, TRNM_ALIGN_UP(size))
#define aligned_free_64(ptr) free(ptr)
#endif

static uint32_t crc32c_table[256];
static int crc32c_table_ready = 0;

static void crc32c_init_table(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) crc = (crc >> 1) ^ 0x82F63B78;
            else crc >>= 1;
        }
        crc32c_table[i] = crc;
    }
    crc32c_table_ready = 1;
}

uint32_t trnm_crc32(const void* data, size_t len) {
    if (!crc32c_table_ready) crc32c_init_table();
    const uint8_t* p = (const uint8_t*)data;
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ crc32c_table[(crc ^ p[i]) & 0xFF];
    }
    return crc ^ 0xFFFFFFFF;
}

const char* trnm_strerror(int err) {
    switch (err) {
        case TRNM_OK:           return "OK";
        case TRNM_ERR_IO:       return "I/O error";
        case TRNM_ERR_MAGIC:    return "bad magic bytes";
        case TRNM_ERR_VERSION:  return "unsupported version";
        case TRNM_ERR_HEADER_CRC: return "header CRC32 mismatch";
        case TRNM_ERR_LAYER_CRC:  return "layer data CRC32 mismatch";
        case TRNM_ERR_FILE_SIZE:  return "file size mismatch";
        case TRNM_ERR_ALLOC:      return "memory allocation failure";
        case TRNM_ERR_TRUNCATED:  return "unexpected end of file";
        case TRNM_ERR_INVALID:    return "invalid field value";
        default: return "unknown error";
    }
}

static int write_padding(FILE* f, long current_pos) {
    long aligned = (long)TRNM_ALIGN_UP((uint64_t)current_pos);
    long pad = aligned - current_pos;
    if (pad > 0) {
        uint8_t zeros[TRNM_ALIGN] = {0};
        if (fwrite(zeros, 1, (size_t)pad, f) != (size_t)pad) return TRNM_ERR_IO;
    }
    return TRNM_OK;
}

static int skip_to_aligned(FILE* f) {
    long pos = ftell(f);
    long aligned = (long)TRNM_ALIGN_UP((uint64_t)pos);
    if (aligned != pos) {
        if (fseek(f, aligned, SEEK_SET) != 0) return TRNM_ERR_IO;
    }
    return TRNM_OK;
}

int trnm_validate_header(const trnm_header_t* h) {
    if (memcmp(h->magic, "TRNM", 4) != 0) return TRNM_ERR_MAGIC;
    if (h->version != TRNM_VERSION) return TRNM_ERR_VERSION;
    if (h->d_model == 0 || h->n_layers == 0 || h->vocab_size == 0) return TRNM_ERR_INVALID;
    if (h->n_swa_layers > TRNM_MAX_SWA_LAYERS) return TRNM_ERR_INVALID;
    if (h->n_swa_layers > h->n_layers) return TRNM_ERR_INVALID;

    for (uint32_t i = 0; i < h->n_swa_layers; i++) {
        if (h->swa_indices[i] >= h->n_layers) return TRNM_ERR_INVALID;
    }

    if (h->n_heads > 0 && h->d_model % h->n_heads != 0) return TRNM_ERR_INVALID;

    return TRNM_OK;
}

int trnm_read_header(const char* path, trnm_header_t* header) {
    FILE* f = fopen(path, "rb");
    if (!f) return TRNM_ERR_IO;

    if (fread(header, TRNM_HEADER_SIZE, 1, f) != 1) {
        fclose(f);
        return TRNM_ERR_TRUNCATED;
    }
    fclose(f);

    uint32_t computed_crc = trnm_crc32(header, TRNM_HEADER_SIZE - 4);
    if (computed_crc != header->header_crc32) return TRNM_ERR_HEADER_CRC;

    return trnm_validate_header(header);
}

static int read_float_array(FILE* f, float** out, size_t count) {
    *out = (float*)aligned_alloc_64(count * sizeof(float));
    if (!*out) return TRNM_ERR_ALLOC;
    if (fread(*out, sizeof(float), count, f) != count) return TRNM_ERR_TRUNCATED;
    return skip_to_aligned(f);
}

static int read_ternary_block(FILE* f, ternary_matrix_t** out) {
    trnm_ternary_hdr_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) return TRNM_ERR_TRUNCATED;

    *out = ternary_matrix_alloc(hdr.rows, hdr.cols);
    if (!*out) return TRNM_ERR_ALLOC;

    if (fread((*out)->data, 1, hdr.packed_size, f) != hdr.packed_size) return TRNM_ERR_TRUNCATED;
    int rc = skip_to_aligned(f);
    if (rc != TRNM_OK) return rc;

    if (fread((*out)->scales, sizeof(float), hdr.rows, f) != hdr.rows) return TRNM_ERR_TRUNCATED;
    return skip_to_aligned(f);
}

static int write_float_array(FILE* f, const float* data, size_t count) {
    if (fwrite(data, sizeof(float), count, f) != count) return TRNM_ERR_IO;
    return write_padding(f, ftell(f));
}

static int write_ternary_block(FILE* f, const ternary_matrix_t* mat) {
    trnm_ternary_hdr_t hdr;
    hdr.rows = mat->rows;
    hdr.cols = mat->cols;
    hdr.packed_size = (uint32_t)((mat->rows * mat->cols + 3) / 4);
    hdr.reserved = 0;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) return TRNM_ERR_IO;
    if (fwrite(mat->data, 1, hdr.packed_size, f) != hdr.packed_size) return TRNM_ERR_IO;
    int rc = write_padding(f, ftell(f));
    if (rc != TRNM_OK) return rc;

    if (fwrite(mat->scales, sizeof(float), (size_t)mat->rows, f) != (size_t)mat->rows) return TRNM_ERR_IO;
    return write_padding(f, ftell(f));
}

int trnm_load_layer_table(const char* path, const trnm_header_t* header,
                          trnm_layer_entry_t** out_table) {
    FILE* f = fopen(path, "rb");
    if (!f) return TRNM_ERR_IO;

    if (fseek(f, (long)header->layer_table_offset, SEEK_SET) != 0) {
        fclose(f);
        return TRNM_ERR_IO;
    }

    size_t n = header->n_layers;
    *out_table = (trnm_layer_entry_t*)calloc(n, sizeof(trnm_layer_entry_t));
    if (!*out_table) {
        fclose(f);
        return TRNM_ERR_ALLOC;
    }

    if (fread(*out_table, sizeof(trnm_layer_entry_t), n, f) != n) {
        free(*out_table);
        *out_table = NULL;
        fclose(f);
        return TRNM_ERR_TRUNCATED;
    }

    fclose(f);
    return TRNM_OK;
}

static int read_ssm_layer(FILE* f, const trnm_header_t* header,
                          uint8_t layer_flags,
                          ssm_layer_t** out_ssm, float** out_norm) {
    int rc;
    uint32_t d = header->d_model;
    uint32_t d_inner = header->d_inner;
    uint32_t d_state = header->d_state;
    uint32_t d_conv = header->d_conv;
    uint32_t dt_rank = header->dt_rank;
    uint32_t ssd_heads = header->ssd_n_heads;

    rc = read_float_array(f, out_norm, d);
    if (rc != TRNM_OK) return rc;

    ssm_layer_t* ssm = (ssm_layer_t*)calloc(1, sizeof(ssm_layer_t));
    if (!ssm) return TRNM_ERR_ALLOC;
    ssm->d_model = d;
    ssm->d_inner = d_inner;
    ssm->d_state = d_state;
    ssm->d_conv = d_conv;
    ssm->n_heads = ssd_heads;
    ssm->head_dim = d_inner / ssd_heads;
    ssm->dt_rank = dt_rank;
    ssm->use_tscs = (header->flags & TRNM_FLAG_TSCS) ? 1 : 0;

    rc = read_ternary_block(f, &ssm->in_proj);
    if (rc != TRNM_OK) {
        free(ssm);
        return rc;
    }

    rc = read_float_array(f, &ssm->conv_weight, d_inner * d_conv);
    if (rc != TRNM_OK) {
        ternary_matrix_free(ssm->in_proj);
        free(ssm);
        return rc;
    }

    rc = read_float_array(f, &ssm->conv_bias, d_inner);
    if (rc != TRNM_OK) {
        aligned_free_64(ssm->conv_weight);
        ternary_matrix_free(ssm->in_proj);
        free(ssm);
        return rc;
    }

    int is_lti = (layer_flags & TRNM_LFLAG_IS_LTI) != 0;

    if (!is_lti) {
        rc = read_ternary_block(f, &ssm->x_proj);
        if (rc != TRNM_OK) {
            aligned_free_64(ssm->conv_bias);
            aligned_free_64(ssm->conv_weight);
            ternary_matrix_free(ssm->in_proj);
            free(ssm);
            return rc;
        }

        rc = read_float_array(f, &ssm->dt_weight, d_inner * dt_rank);
        if (rc != TRNM_OK) {
            ternary_matrix_free(ssm->x_proj);
            aligned_free_64(ssm->conv_bias);
            aligned_free_64(ssm->conv_weight);
            ternary_matrix_free(ssm->in_proj);
            free(ssm);
            return rc;
        }

        rc = read_float_array(f, &ssm->dt_bias, d_inner);
        if (rc != TRNM_OK) {
            aligned_free_64(ssm->dt_weight);
            ternary_matrix_free(ssm->x_proj);
            aligned_free_64(ssm->conv_bias);
            aligned_free_64(ssm->conv_weight);
            ternary_matrix_free(ssm->in_proj);
            free(ssm);
            return rc;
        }
    }

    if (header->flags & TRNM_FLAG_TSCS) {
        ssm->A_ternary = (int8_t*)calloc(ssd_heads, sizeof(int8_t));
        if (!ssm->A_ternary) return TRNM_ERR_ALLOC;
        if (fread(ssm->A_ternary, sizeof(int8_t), ssd_heads, f) != ssd_heads) return TRNM_ERR_TRUNCATED;
        skip_to_aligned(f);
    } else {
        float* a_log;
        rc = read_float_array(f, &a_log, ssd_heads);
        if (rc != TRNM_OK) return rc;
        ssm->A = (float*)aligned_alloc_64(ssd_heads * sizeof(float));
        if (!ssm->A) {
            aligned_free_64(a_log);
            return TRNM_ERR_ALLOC;
        }
        for (uint32_t h = 0; h < ssd_heads; h++) {
            ssm->A[h] = -expf(a_log[h]);
        }
        aligned_free_64(a_log);
    }

    rc = read_float_array(f, &ssm->D, d_inner);
    if (rc != TRNM_OK) return rc;

    rc = read_ternary_block(f, &ssm->out_proj);
    if (rc != TRNM_OK) return rc;

    *out_ssm = ssm;
    return TRNM_OK;
}

static int read_swa_layer(FILE* f, const trnm_header_t* header,
                          uint8_t layer_flags,
                          attn_layer_t** out_attn, float** out_norm) {
    (void)layer_flags;
    int rc;
    uint32_t d = header->d_model;
    uint32_t mlp_h = header->mlp_hidden;
    uint32_t rope_dim = header->rope_dim;

    rc = read_float_array(f, out_norm, d);
    if (rc != TRNM_OK) return rc;

    attn_layer_t* attn = (attn_layer_t*)calloc(1, sizeof(attn_layer_t));
    if (!attn) return TRNM_ERR_ALLOC;
    attn->d_model = d;
    attn->n_heads = header->n_heads;
    attn->head_dim = header->head_dim;
    attn->window_size = header->window_size;
    attn->mlp_hidden = mlp_h;

    rc = read_float_array(f, &attn->attn_norm_weight, d);
    if (rc != TRNM_OK) {
        free(attn);
        return rc;
    }

    rc = read_ternary_block(f, &attn->q_proj);
    if (rc != TRNM_OK) {
        aligned_free_64(attn->attn_norm_weight);
        free(attn);
        return rc;
    }
    rc = read_ternary_block(f, &attn->k_proj);
    if (rc != TRNM_OK) {
        ternary_matrix_free(attn->q_proj);
        aligned_free_64(attn->attn_norm_weight);
        free(attn);
        return rc;
    }
    rc = read_ternary_block(f, &attn->v_proj);
    if (rc != TRNM_OK) {
        ternary_matrix_free(attn->k_proj);
        ternary_matrix_free(attn->q_proj);
        aligned_free_64(attn->attn_norm_weight);
        free(attn);
        return rc;
    }
    rc = read_ternary_block(f, &attn->out_proj);
    if (rc != TRNM_OK) {
        ternary_matrix_free(attn->v_proj);
        ternary_matrix_free(attn->k_proj);
        ternary_matrix_free(attn->q_proj);
        aligned_free_64(attn->attn_norm_weight);
        free(attn);
        return rc;
    }

    rc = read_float_array(f, &attn->mlp_norm_weight, d);
    if (rc != TRNM_OK) {
        ternary_matrix_free(attn->out_proj);
        ternary_matrix_free(attn->v_proj);
        ternary_matrix_free(attn->k_proj);
        ternary_matrix_free(attn->q_proj);
        aligned_free_64(attn->attn_norm_weight);
        free(attn);
        return rc;
    }

    rc = read_ternary_block(f, &attn->mlp_gate);
    if (rc != TRNM_OK) {
        aligned_free_64(attn->mlp_norm_weight);
        ternary_matrix_free(attn->out_proj);
        ternary_matrix_free(attn->v_proj);
        ternary_matrix_free(attn->k_proj);
        ternary_matrix_free(attn->q_proj);
        aligned_free_64(attn->attn_norm_weight);
        free(attn);
        return rc;
    }
    rc = read_ternary_block(f, &attn->mlp_up);
    if (rc != TRNM_OK) {
        ternary_matrix_free(attn->mlp_gate);
        aligned_free_64(attn->mlp_norm_weight);
        ternary_matrix_free(attn->out_proj);
        ternary_matrix_free(attn->v_proj);
        ternary_matrix_free(attn->k_proj);
        ternary_matrix_free(attn->q_proj);
        aligned_free_64(attn->attn_norm_weight);
        free(attn);
        return rc;
    }
    rc = read_ternary_block(f, &attn->mlp_down);
    if (rc != TRNM_OK) {
        ternary_matrix_free(attn->mlp_up);
        ternary_matrix_free(attn->mlp_gate);
        aligned_free_64(attn->mlp_norm_weight);
        ternary_matrix_free(attn->out_proj);
        ternary_matrix_free(attn->v_proj);
        ternary_matrix_free(attn->k_proj);
        ternary_matrix_free(attn->q_proj);
        aligned_free_64(attn->attn_norm_weight);
        free(attn);
        return rc;
    }

    rc = read_float_array(f, &attn->rope_inv_freq, rope_dim);
    if (rc != TRNM_OK) {
        ternary_matrix_free(attn->mlp_down);
        ternary_matrix_free(attn->mlp_up);
        ternary_matrix_free(attn->mlp_gate);
        aligned_free_64(attn->mlp_norm_weight);
        ternary_matrix_free(attn->out_proj);
        ternary_matrix_free(attn->v_proj);
        ternary_matrix_free(attn->k_proj);
        ternary_matrix_free(attn->q_proj);
        aligned_free_64(attn->attn_norm_weight);
        free(attn);
        return rc;
    }

    *out_attn = attn;
    return TRNM_OK;
}

trnm_model_t* trnm_load(const char* path) {
    trnm_header_t header;
    int rc = trnm_read_header(path, &header);
    if (rc != TRNM_OK) {
        fprintf(stderr, "TRNM load error: %s\n", trnm_strerror(rc));
        return NULL;
    }

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    trnm_model_t* model = (trnm_model_t*)calloc(1, sizeof(trnm_model_t));
    if (!model) {
        fclose(f);
        return NULL;
    }
    model->header = header;

    uint32_t n = header.n_layers;
    uint32_t d = header.d_model;

    fseek(f, (long)header.layer_table_offset, SEEK_SET);
    model->layer_table = (trnm_layer_entry_t*)calloc(n, sizeof(trnm_layer_entry_t));
    if (!model->layer_table) {
        free(model);
        fclose(f);
        return NULL;
    }
    if (fread(model->layer_table, sizeof(trnm_layer_entry_t), n, f) != n) {
        free(model->layer_table);
        free(model);
        fclose(f);
        return NULL;
    }

    model->layer_types = (uint8_t*)calloc(n, sizeof(uint8_t));
    model->ssm_layers = (ssm_layer_t**)calloc(n, sizeof(ssm_layer_t*));
    model->attn_layers = (attn_layer_t**)calloc(n, sizeof(attn_layer_t*));
    model->pre_norms = (float**)calloc(n, sizeof(float*));

    if (!model->layer_types || !model->ssm_layers || !model->attn_layers || !model->pre_norms) {
        free(model->layer_types);
        free(model->ssm_layers);
        free(model->attn_layers);
        free(model->pre_norms);
        free(model->layer_table);
        free(model);
        fclose(f);
        return NULL;
    }

    fseek(f, (long)header.embedding_offset, SEEK_SET);
    rc = read_float_array(f, &model->embedding, (size_t)header.vocab_size * d);
    if (rc != TRNM_OK) {
        fprintf(stderr, "TRNM: embedding load failed: %s\n", trnm_strerror(rc));
        goto fail;
    }

    for (uint32_t i = 0; i < n; i++) {
        trnm_layer_entry_t* entry = &model->layer_table[i];
        model->layer_types[i] = entry->layer_type;

        fseek(f, (long)entry->data_offset, SEEK_SET);

        if (entry->layer_type == TRNM_LAYER_SSM) {
            rc = read_ssm_layer(f, &header, entry->layer_flags,
                               &model->ssm_layers[i], &model->pre_norms[i]);
        } else {
            rc = read_swa_layer(f, &header, entry->layer_flags,
                               &model->attn_layers[i], &model->pre_norms[i]);
        }
        if (rc != TRNM_OK) {
            fprintf(stderr, "TRNM: layer %u load failed: %s\n", i, trnm_strerror(rc));
            goto fail;
        }
    }

    fseek(f, (long)header.final_norm_offset, SEEK_SET);
    rc = read_float_array(f, &model->final_norm, d);
    if (rc != TRNM_OK) {
        fprintf(stderr, "TRNM: final norm load failed: %s\n", trnm_strerror(rc));
        goto fail;
    }

    if (header.flags & TRNM_FLAG_HAS_LM_HEAD) {
        fseek(f, (long)header.lm_head_offset, SEEK_SET);
        if (header.flags & TRNM_FLAG_LM_TERNARY) {
            rc = read_ternary_block(f, &model->lm_head_ternary);
        } else {
            rc = read_float_array(f, &model->lm_head_float, (size_t)header.vocab_size * d);
        }
        if (rc != TRNM_OK) {
            fprintf(stderr, "TRNM: LM head load failed: %s\n", trnm_strerror(rc));
            goto fail;
        }
    }

    fclose(f);
    return model;

fail:
    fclose(f);
    trnm_free(model);
    return NULL;
}

void trnm_free(trnm_model_t* model) {
    if (!model) return;

    uint32_t n = model->header.n_layers;

    if (model->embedding) aligned_free_64(model->embedding);
    if (model->final_norm) aligned_free_64(model->final_norm);

    if (model->lm_head_ternary) ternary_matrix_free(model->lm_head_ternary);
    if (model->lm_head_float) aligned_free_64(model->lm_head_float);

    for (uint32_t i = 0; i < n; i++) {
        if (model->pre_norms && model->pre_norms[i]) aligned_free_64(model->pre_norms[i]);

        if (model->ssm_layers && model->ssm_layers[i]) {
            ssm_layer_t* s = model->ssm_layers[i];
            if (s->in_proj) ternary_matrix_free(s->in_proj);
            if (s->x_proj) ternary_matrix_free(s->x_proj);
            if (s->out_proj) ternary_matrix_free(s->out_proj);
            if (s->conv_weight) aligned_free_64(s->conv_weight);
            if (s->conv_bias) aligned_free_64(s->conv_bias);
            if (s->dt_weight) aligned_free_64(s->dt_weight);
            if (s->dt_bias) aligned_free_64(s->dt_bias);
            if (s->A) aligned_free_64(s->A);
            if (s->A_ternary) free(s->A_ternary);
            if (s->D) aligned_free_64(s->D);
            free(s);
        }

        if (model->attn_layers && model->attn_layers[i]) {
            attn_layer_t* a = model->attn_layers[i];
            if (a->q_proj) ternary_matrix_free(a->q_proj);
            if (a->k_proj) ternary_matrix_free(a->k_proj);
            if (a->v_proj) ternary_matrix_free(a->v_proj);
            if (a->out_proj) ternary_matrix_free(a->out_proj);
            if (a->mlp_gate) ternary_matrix_free(a->mlp_gate);
            if (a->mlp_up) ternary_matrix_free(a->mlp_up);
            if (a->mlp_down) ternary_matrix_free(a->mlp_down);
            if (a->attn_norm_weight) aligned_free_64(a->attn_norm_weight);
            if (a->mlp_norm_weight) aligned_free_64(a->mlp_norm_weight);
            if (a->rope_inv_freq) aligned_free_64(a->rope_inv_freq);
            free(a);
        }
    }

    free(model->layer_table);
    free(model->layer_types);
    free(model->ssm_layers);
    free(model->attn_layers);
    free(model->pre_norms);
    free(model);
}

int trnm_save(const char* path, const trnm_model_t* model) {
    FILE* f = fopen(path, "w+b");
    if (!f) return TRNM_ERR_IO;

    uint32_t n = model->header.n_layers;
    uint32_t d = model->header.d_model;

    uint8_t placeholder_header[TRNM_HEADER_SIZE] = {0};
    if (fwrite(placeholder_header, TRNM_HEADER_SIZE, 1, f) != 1) {
        fclose(f);
        return TRNM_ERR_IO;
    }

    uint64_t layer_table_offset = ftell(f);
    if (fwrite(model->layer_table, sizeof(trnm_layer_entry_t), n, f) != n) {
        fclose(f);
        return TRNM_ERR_IO;
    }
    int rc = write_padding(f, ftell(f));
    if (rc != TRNM_OK) {
        fclose(f);
        return rc;
    }

    uint64_t embedding_offset = ftell(f);
    rc = write_float_array(f, model->embedding, (size_t)d * model->header.vocab_size);
    if (rc != TRNM_OK) {
        fclose(f);
        return rc;
    }

    for (uint32_t i = 0; i < n; i++) {
        trnm_layer_entry_t* entry = &model->layer_table[i];
        entry->data_offset = ftell(f);

        long layer_start = ftell(f);

        if (entry->layer_type == TRNM_LAYER_SSM) {
            ssm_layer_t* ssm = model->ssm_layers[i];

            rc = write_float_array(f, model->pre_norms[i], d);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }

            rc = write_ternary_block(f, ssm->in_proj);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }

            rc = write_float_array(f, ssm->conv_weight, ssm->d_inner * ssm->d_conv);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }

            rc = write_float_array(f, ssm->conv_bias, ssm->d_inner);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }

            int is_lti = (model->layer_table[i].layer_flags & TRNM_LFLAG_IS_LTI) != 0;
            if (!is_lti) {
                rc = write_ternary_block(f, ssm->x_proj);
                if (rc != TRNM_OK) {
                    fclose(f);
                    return rc;
                }

                rc = write_float_array(f, ssm->dt_weight, ssm->d_inner * ssm->dt_rank);
                if (rc != TRNM_OK) {
                    fclose(f);
                    return rc;
                }

                rc = write_float_array(f, ssm->dt_bias, ssm->d_inner);
                if (rc != TRNM_OK) {
                    fclose(f);
                    return rc;
                }
            }

            if (model->header.flags & TRNM_FLAG_TSCS) {
                if (fwrite(ssm->A_ternary, sizeof(int8_t), ssm->n_heads, f) != ssm->n_heads) {
                    fclose(f);
                    return TRNM_ERR_IO;
                }
                rc = write_padding(f, ftell(f));
                if (rc != TRNM_OK) {
                    fclose(f);
                    return rc;
                }
            } else {
                float* a_log_buf = (float*)malloc(ssm->n_heads * sizeof(float));
                if (!a_log_buf) {
                    fclose(f);
                    return TRNM_ERR_ALLOC;
                }
                for (uint32_t h = 0; h < ssm->n_heads; h++) {
                    a_log_buf[h] = logf(-ssm->A[h]);
                }
                if (fwrite(a_log_buf, sizeof(float), ssm->n_heads, f) != ssm->n_heads) {
                    free(a_log_buf);
                    fclose(f);
                    return TRNM_ERR_IO;
                }
                free(a_log_buf);
                rc = write_padding(f, ftell(f));
                if (rc != TRNM_OK) {
                    fclose(f);
                    return rc;
                }
            }

            rc = write_float_array(f, ssm->D, ssm->d_inner);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }

            rc = write_ternary_block(f, ssm->out_proj);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }
        } else {
            attn_layer_t* attn = model->attn_layers[i];

            rc = write_float_array(f, model->pre_norms[i], d);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }

            rc = write_float_array(f, attn->attn_norm_weight, d);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }

            rc = write_ternary_block(f, attn->q_proj);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }
            rc = write_ternary_block(f, attn->k_proj);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }
            rc = write_ternary_block(f, attn->v_proj);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }
            rc = write_ternary_block(f, attn->out_proj);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }

            rc = write_float_array(f, attn->mlp_norm_weight, d);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }

            rc = write_ternary_block(f, attn->mlp_gate);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }
            rc = write_ternary_block(f, attn->mlp_up);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }
            rc = write_ternary_block(f, attn->mlp_down);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }

            rc = write_float_array(f, attn->rope_inv_freq, model->header.rope_dim);
            if (rc != TRNM_OK) {
                fclose(f);
                return rc;
            }
        }

        long layer_end = ftell(f);
        entry->data_size = (uint32_t)(layer_end - layer_start);

        long temp_pos = ftell(f);
        fseek(f, layer_start, SEEK_SET);
        uint8_t* layer_data = (uint8_t*)malloc((size_t)entry->data_size);
        if (fread(layer_data, 1, (size_t)entry->data_size, f) != (size_t)entry->data_size) {
            free(layer_data);
            fclose(f);
            return TRNM_ERR_IO;
        }
        entry->data_crc32 = trnm_crc32(layer_data, (size_t)entry->data_size);
        free(layer_data);
        fseek(f, temp_pos, SEEK_SET);
    }

    uint64_t final_norm_offset = ftell(f);
    rc = write_float_array(f, model->final_norm, d);
    if (rc != TRNM_OK) {
        fclose(f);
        return rc;
    }

    uint64_t lm_head_offset = 0;
    if (model->header.flags & TRNM_FLAG_HAS_LM_HEAD) {
        lm_head_offset = ftell(f);
        if (model->header.flags & TRNM_FLAG_LM_TERNARY) {
            rc = write_ternary_block(f, model->lm_head_ternary);
        } else {
            rc = write_float_array(f, model->lm_head_float, (size_t)d * model->header.vocab_size);
        }
        if (rc != TRNM_OK) {
            fclose(f);
            return rc;
        }
    }

    uint64_t file_size = ftell(f);

    trnm_header_t header_copy = model->header;
    header_copy.layer_table_offset = layer_table_offset;
    header_copy.embedding_offset = embedding_offset;
    header_copy.final_norm_offset = final_norm_offset;
    header_copy.lm_head_offset = lm_head_offset;
    header_copy.file_size = file_size;
    header_copy.header_crc32 = 0;
    header_copy.header_crc32 = trnm_crc32(&header_copy, TRNM_HEADER_SIZE - 4);

    fseek(f, 0, SEEK_SET);
    if (fwrite(&header_copy, TRNM_HEADER_SIZE, 1, f) != 1) {
        fclose(f);
        return TRNM_ERR_IO;
    }

    fseek(f, (long)layer_table_offset, SEEK_SET);
    if (fwrite(model->layer_table, sizeof(trnm_layer_entry_t), n, f) != n) {
        fclose(f);
        return TRNM_ERR_IO;
    }

    fclose(f);
    return TRNM_OK;
}

int trnm_load_layer(const char* path, const trnm_header_t* header,
                    const trnm_layer_entry_t* entry, void** out_layer, float** out_norm) {
    FILE* f = fopen(path, "rb");
    if (!f) return TRNM_ERR_IO;

    if (fseek(f, (long)entry->data_offset, SEEK_SET) != 0) {
        fclose(f);
        return TRNM_ERR_IO;
    }

    int rc;
    if (entry->layer_type == TRNM_LAYER_SSM) {
        rc = read_ssm_layer(f, header, entry->layer_flags,
                           (ssm_layer_t**)out_layer, out_norm);
    } else {
        rc = read_swa_layer(f, header, entry->layer_flags,
                           (attn_layer_t**)out_layer, out_norm);
    }

    uint32_t computed_crc = 0;
    if (rc == TRNM_OK) {
        fseek(f, (long)entry->data_offset, SEEK_SET);
        uint8_t* layer_data = (uint8_t*)malloc((size_t)entry->data_size);
        if (!layer_data) {
            fclose(f);
            return TRNM_ERR_ALLOC;
        }
        if (fread(layer_data, 1, (size_t)entry->data_size, f) != (size_t)entry->data_size) {
            free(layer_data);
            fclose(f);
            return TRNM_ERR_TRUNCATED;
        }
        computed_crc = trnm_crc32(layer_data, (size_t)entry->data_size);
        free(layer_data);

        if (computed_crc != entry->data_crc32) {
            rc = TRNM_ERR_LAYER_CRC;
        }
    }

    fclose(f);
    return rc;
}
