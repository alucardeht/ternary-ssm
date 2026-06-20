#include "dataset.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#if !(defined(USE_MMAP) && USE_MMAP == 0)
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <sys/mman.h>
  #include <unistd.h>
#endif

static uint32_t read_le_u32(const uint8_t *p) {
    return (uint32_t)p[0]
         | (uint32_t)p[1] << 8
         | (uint32_t)p[2] << 16
         | (uint32_t)p[3] << 24;
}

static uint64_t read_le_u64(const uint8_t *p) {
    return (uint64_t)read_le_u32(p) | (uint64_t)read_le_u32(p + 4) << 32;
}

static uint64_t rotl64(uint64_t x, unsigned int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(uint64_t s[4]) {
    uint64_t result = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);

    return result;
}

static void rng_seed(uint64_t s[4], uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s[i] = z ^ (z >> 31);
    }
}

Dataset *dataset_load(const char *path) {
    Dataset *ds = malloc(sizeof(Dataset));
    if (ds == NULL) {
        fprintf(stderr, "[dataset] OOM allocating Dataset struct\n");
        return NULL;
    }

    #if defined(USE_MMAP) && USE_MMAP == 0
    int use_mmap = 0;
    #else
    int use_mmap = 1;
    #endif

    if (use_mmap) {
        #if !(defined(USE_MMAP) && USE_MMAP == 0)
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "[dataset] cannot open file: %s\n", path);
            free(ds);
            return NULL;
        }

        struct stat st;
        if (fstat(fd, &st) < 0) {
            fprintf(stderr, "[dataset] cannot stat file: %s\n", path);
            close(fd);
            free(ds);
            return NULL;
        }
        off_t file_size = st.st_size;

        if (file_size < DATASET_HEADER_BYTES) {
            fprintf(stderr, "[dataset] file too small for header check\n");
            close(fd);
            free(ds);
            return NULL;
        }

        uint8_t header_buf[DATASET_HEADER_BYTES];
        ssize_t n_read = pread(fd, header_buf, DATASET_HEADER_BYTES, 0);
        if (n_read != DATASET_HEADER_BYTES) {
            fprintf(stderr, "[dataset] cannot read header\n");
            close(fd);
            free(ds);
            return NULL;
        }

        uint32_t magic = read_le_u32(header_buf + 0);
        uint32_t version = read_le_u32(header_buf + 4);
        uint64_t n_tok = read_le_u64(header_buf + 8);

        size_t data_offset = 0;
        if (magic == DATASET_MAGIC) {
            if (version != DATASET_VERSION) {
                fprintf(stderr, "[dataset] unsupported version %u\n", version);
                close(fd);
                free(ds);
                return NULL;
            }
            ds->n_tokens = n_tok;
            data_offset = DATASET_HEADER_BYTES;
        } else {
            ds->n_tokens = (size_t)file_size / sizeof(uint16_t);
            data_offset = 0;
        }

        if (ds->n_tokens == 0) {
            fprintf(stderr, "[dataset] dataset empty\n");
            close(fd);
            free(ds);
            return NULL;
        }

        void *ptr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr != MAP_FAILED) {
            ds->data = (uint16_t*)((uint8_t*)ptr + data_offset);
            ds->is_mmap = 1;
            ds->fd = fd;
            ds->mmap_size = file_size;
            uint64_t seed = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)ds;
            rng_seed(ds->rng, seed);
            ds->cursor = 0;
            return ds;
        }

        close(fd);
        #endif
    }

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "[dataset] cannot open file: %s\n", path);
        free(ds);
        return NULL;
    }

    if (fseek(fp, 0, SEEK_END) < 0) {
        fprintf(stderr, "[dataset] cannot seek file: %s\n", path);
        fclose(fp);
        free(ds);
        return NULL;
    }
    long sz = ftell(fp);
    if (sz < 0) {
        fprintf(stderr, "[dataset] cannot get file size: %s\n", path);
        fclose(fp);
        free(ds);
        return NULL;
    }
    size_t file_size = (size_t)sz;
    rewind(fp);

    if (file_size < DATASET_HEADER_BYTES) {
        fprintf(stderr, "[dataset] file too small for header check\n");
        fclose(fp);
        free(ds);
        return NULL;
    }

    uint8_t header_buf[DATASET_HEADER_BYTES];
    size_t n_read = fread(header_buf, 1, DATASET_HEADER_BYTES, fp);
    if (n_read != DATASET_HEADER_BYTES) {
        fprintf(stderr, "[dataset] cannot read header\n");
        fclose(fp);
        free(ds);
        return NULL;
    }

    uint32_t magic = read_le_u32(header_buf + 0);
    uint32_t version = read_le_u32(header_buf + 4);
    uint64_t n_tok = read_le_u64(header_buf + 8);

    size_t data_offset = 0;
    size_t data_size = 0;

    if (magic == DATASET_MAGIC) {
        if (version != DATASET_VERSION) {
            fprintf(stderr, "[dataset] unsupported version %u\n", version);
            fclose(fp);
            free(ds);
            return NULL;
        }
        ds->n_tokens = n_tok;
        data_offset = DATASET_HEADER_BYTES;
        data_size = (size_t)n_tok * sizeof(uint16_t);
    } else {
        ds->n_tokens = (size_t)file_size / sizeof(uint16_t);
        data_offset = 0;
        data_size = (size_t)file_size;
    }

    if (ds->n_tokens == 0) {
        fprintf(stderr, "[dataset] dataset empty\n");
        fclose(fp);
        free(ds);
        return NULL;
    }

    if (n_tok > SIZE_MAX / sizeof(uint16_t)) {
        fprintf(stderr, "[dataset] n_tokens overflow\n");
        fclose(fp);
        free(ds);
        return NULL;
    }

    ds->data = malloc(data_size);
    if (ds->data == NULL) {
        fprintf(stderr, "[dataset] OOM allocating data buffer\n");
        fclose(fp);
        free(ds);
        return NULL;
    }

    /* Seek to data start — always needed because we consumed 16 bytes for header check */
    {
        if (fseek(fp, (long)data_offset, SEEK_SET) < 0) {
            fprintf(stderr, "[dataset] cannot seek to data offset\n");
            free(ds->data);
            fclose(fp);
            free(ds);
            return NULL;
        }
    }

    /* Read in chunks to handle platforms where fread has size limits */
    size_t n_tokens_read = 0;
    size_t remaining = ds->n_tokens;
    while (remaining > 0) {
        size_t chunk = remaining > 1000000 ? 1000000 : remaining;
        size_t got = fread(ds->data + n_tokens_read, sizeof(uint16_t), chunk, fp);
        if (got == 0) break;
        n_tokens_read += got;
        remaining -= got;
    }
    if (n_tokens_read < ds->n_tokens) {
        fprintf(stderr, "[dataset] short read: got %zu of %zu tokens (adjusting)\n",
                n_tokens_read, ds->n_tokens);
        ds->n_tokens = n_tokens_read;
    }

    ds->is_mmap = 0;
    ds->fd = -1;
    ds->mmap_size = 0;
    fclose(fp);

    uint64_t seed = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)ds;
    rng_seed(ds->rng, seed);
    ds->cursor = 0;

    return ds;
}

void dataset_sample_batch(Dataset *ds,
                          uint16_t *batch,
                          int batch_size,
                          int seq_len) {
    #ifndef NDEBUG
    if (seq_len >= (int)ds->n_tokens) {
        fprintf(stderr, "[dataset] seq_len %d >= n_tokens %zu\n", seq_len, ds->n_tokens);
    }
    #endif

    size_t max_offset = ds->n_tokens - (size_t)seq_len;

    for (int b = 0; b < batch_size; b++) {
        uint64_t raw = rng_next(ds->rng);
        size_t offset = raw % max_offset;
        uint16_t *src = ds->data + offset;
        uint16_t *dst = batch + (size_t)b * seq_len;
        memcpy(dst, src, (size_t)seq_len * sizeof(uint16_t));
    }
}

void dataset_next_sequential(Dataset *ds,
                             uint16_t *batch,
                             int batch_size,
                             int seq_len) {
    #ifndef NDEBUG
    if (seq_len >= (int)ds->n_tokens) {
        fprintf(stderr, "[dataset] seq_len %d >= n_tokens %zu\n", seq_len, ds->n_tokens);
    }
    #endif

    size_t n = ds->n_tokens;
    size_t window = (size_t)seq_len;

    for (int b = 0; b < batch_size; b++) {
        if (ds->cursor + window > n) {
            ds->cursor = 0;
        }
        #ifndef NDEBUG
        printf("[SEQ] sequential step cursor=%llu\n", (unsigned long long)ds->cursor);
        #endif
        uint16_t *src = ds->data + ds->cursor;
        uint16_t *dst = batch + (size_t)b * window;
        memcpy(dst, src, window * sizeof(uint16_t));
        ds->cursor += window;
    }
}

void dataset_reset_cursor(Dataset *ds) {
    if (ds == NULL) return;
    ds->cursor = 0;
}

void dataset_seed(Dataset *ds, uint64_t seed) {
    if (seed == 0) {
        fprintf(stderr, "[dataset] seed=0 rejected, using seed=1\n");
        seed = 1;
    }
    rng_seed(ds->rng, seed);
}

void dataset_free(Dataset *ds) {
    if (ds == NULL) return;

    if (ds->is_mmap == 1) {
        #if !(defined(USE_MMAP) && USE_MMAP == 0)
        void *ptr = (void*)((uint8_t*)ds->data - DATASET_HEADER_BYTES);
        munmap(ptr, ds->mmap_size);
        close(ds->fd);
        #endif
    } else {
        free(ds->data);
    }

    free(ds);
}

/* R493-G04 R441 Gate-to-Task Compiler (Aspect A05) */
int dataset_classify_task(const uint16_t *tokens, int seq_len) {
    if (!tokens || seq_len <= 0) return 0;
    
    int code_token_count = 0;
    int logic_token_count = 0;
    
    for (int i = 0; i < seq_len; i++) {
        uint16_t tok = tokens[i];
        if (tok == 123 || tok == 125 || tok == 59 || tok == 40 || tok == 41) { /* {, }, ;, (, ) */
            code_token_count++;
        }
        if (tok >= 48 && tok <= 57) { /* '0'-'9' (logical/procedural metrics) */
            logic_token_count++;
        }
    }
    
    if (code_token_count > 3) return 2; /* syntactic_code */
    if (logic_token_count > 5) return 1; /* procedural_logic */
    
    return 0; /* default: linguistic_qa */
}
