#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>

#define TVOC_MAGIC       0x434F5654  /* "TVOC" little-endian */
#define TVOC_VERSION     1
#define TVOC_HEADER_SIZE 64
#define TVOC_FLAG_BYTE_LEVEL (1u << 0)

#define TVOC_OK            0
#define TVOC_ERR_IO       -1
#define TVOC_ERR_MAGIC    -2
#define TVOC_ERR_VERSION  -3
#define TVOC_ERR_ALLOC    -4
#define TVOC_ERR_OVERFLOW -5

typedef struct {
    /* Vocab storage */
    char*      vocab_data;       /* Contiguous token string data */
    uint32_t*  vocab_offsets;    /* Offset into vocab_data per token */
    uint16_t*  vocab_lens;       /* Precomputed length per token */
    int32_t    vocab_size;

    /* Special tokens */
    int32_t    pad_id;
    int32_t    unk_id;
    int32_t    bos_id;
    int32_t    eos_id;

    /* BPE merges */
    int32_t    n_merges;
    uint32_t*  merge_a;
    uint32_t*  merge_b;
    uint32_t*  merge_result;

    /* String -> token ID hash table */
    int32_t*   str_ht_values;
    uint32_t*  str_ht_hashes;
    int32_t    str_ht_capacity;

    /* (a,b) -> merge priority hash table */
    int32_t*   merge_ht_prio;
    uint64_t*  merge_ht_keys;
    int32_t    merge_ht_capacity;

    uint32_t   flags;
} tokenizer_t;

/* Public API */
tokenizer_t* tokenizer_load(const char* path);
void tokenizer_free(tokenizer_t* tok);

/* Encode text -> token IDs. Returns n_tokens or negative error. */
int32_t tokenizer_encode(const tokenizer_t* tok, const char* text,
                         int32_t* out_tokens, int32_t max_tokens);

/* Decode token IDs -> text. Returns n_bytes written or negative error. */
int32_t tokenizer_decode(const tokenizer_t* tok, const int32_t* tokens,
                         int32_t n_tokens, char* out_text, int32_t max_len);

/* Decode single token -> text (for streaming). Returns n_bytes. */
int32_t tokenizer_decode_token(const tokenizer_t* tok, int32_t token_id,
                               char* out_text, int32_t max_len);

const char* tvoc_strerror(int err);

#endif /* TOKENIZER_H */
