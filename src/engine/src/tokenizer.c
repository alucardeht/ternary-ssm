#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>

/* GPT-2 byte_to_unicode mapping */
static uint32_t byte_to_unicode_table[256];
static uint8_t unicode_to_byte_table[65536];
static int byte_to_unicode_ready = 0;

static void byte_to_unicode_init(void) {
    if (byte_to_unicode_ready) return;

    memset(unicode_to_byte_table, 0xFF, sizeof(unicode_to_byte_table));

    for (int b = 0; b < 256; b++) {
        uint32_t code = b;

        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            code = b;
        } else if (b < 33 || (b >= 127 && b <= 160) || b == 173) {
            code = 256 + b;
        }

        byte_to_unicode_table[b] = code;
        if (code < 65536) {
            unicode_to_byte_table[code] = (uint8_t)b;
        }
    }

    byte_to_unicode_ready = 1;
}

static uint32_t fnv1a(const char* data, int32_t len) {
    uint32_t h = 2166136261u;
    for (int32_t i = 0; i < len; i++) {
        h ^= (uint8_t)data[i];
        h *= 16777619u;
    }
    return h;
}

static uint32_t fnv1a_uint32(uint32_t a, uint32_t b) {
    uint32_t h = 2166136261u;
    h ^= (uint8_t)(a & 0xFF);
    h *= 16777619u;
    h ^= (uint8_t)((a >> 8) & 0xFF);
    h *= 16777619u;
    h ^= (uint8_t)((a >> 16) & 0xFF);
    h *= 16777619u;
    h ^= (uint8_t)((a >> 24) & 0xFF);
    h *= 16777619u;

    h ^= (uint8_t)(b & 0xFF);
    h *= 16777619u;
    h ^= (uint8_t)((b >> 8) & 0xFF);
    h *= 16777619u;
    h ^= (uint8_t)((b >> 16) & 0xFF);
    h *= 16777619u;
    h ^= (uint8_t)((b >> 24) & 0xFF);
    h *= 16777619u;

    return h;
}

static int32_t next_power_of_2(int32_t n) {
    if (n <= 0) return 1;
    if (n > INT32_MAX / 2) return INT32_MAX;

    int32_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static int32_t str_ht_lookup(const tokenizer_t* tok, const char* str, int32_t len) {
    if (!tok || !str) return -1;

    uint32_t h = fnv1a(str, len);
    int32_t idx = h & (tok->str_ht_capacity - 1);

    while (1) {
        if (tok->str_ht_values[idx] == -1) return -1;

        if (tok->str_ht_hashes[idx] == h) {
            int32_t stored_id = tok->str_ht_values[idx];
            if (stored_id >= 0 && stored_id < tok->vocab_size) {
                int32_t str_len = tok->vocab_lens[stored_id];
                const char* stored_str = tok->vocab_data + tok->vocab_offsets[stored_id];
                if (str_len == len && memcmp(stored_str, str, len) == 0) {
                    return stored_id;
                }
            }
        }

        idx = (idx + 1) & (tok->str_ht_capacity - 1);
    }
}

static void str_ht_build(tokenizer_t* tok) {
    tok->str_ht_capacity = next_power_of_2(2 * tok->vocab_size);

    tok->str_ht_values = (int32_t*)malloc(tok->str_ht_capacity * sizeof(int32_t));
    tok->str_ht_hashes = (uint32_t*)malloc(tok->str_ht_capacity * sizeof(uint32_t));

    if (!tok->str_ht_values || !tok->str_ht_hashes) {
        free(tok->str_ht_values);
        free(tok->str_ht_hashes);
        tok->str_ht_values = NULL;
        tok->str_ht_hashes = NULL;
        return;
    }

    for (int32_t i = 0; i < tok->str_ht_capacity; i++) {
        tok->str_ht_values[i] = -1;
    }

    for (int32_t id = 0; id < tok->vocab_size; id++) {
        const char* str = tok->vocab_data + tok->vocab_offsets[id];
        int32_t len = tok->vocab_lens[id];
        uint32_t h = fnv1a(str, len);

        int32_t idx = h & (tok->str_ht_capacity - 1);
        while (tok->str_ht_values[idx] != -1) {
            idx = (idx + 1) & (tok->str_ht_capacity - 1);
        }

        tok->str_ht_values[idx] = id;
        tok->str_ht_hashes[idx] = h;
    }
}

static void merge_ht_build(tokenizer_t* tok) {
    tok->merge_ht_capacity = next_power_of_2(2 * tok->n_merges);

    tok->merge_ht_keys = (uint64_t*)malloc(tok->merge_ht_capacity * sizeof(uint64_t));
    tok->merge_ht_prio = (int32_t*)malloc(tok->merge_ht_capacity * sizeof(int32_t));

    if (!tok->merge_ht_keys || !tok->merge_ht_prio) {
        free(tok->merge_ht_keys);
        free(tok->merge_ht_prio);
        tok->merge_ht_keys = NULL;
        tok->merge_ht_prio = NULL;
        return;
    }

    for (int32_t i = 0; i < tok->merge_ht_capacity; i++) {
        tok->merge_ht_keys[i] = UINT64_MAX;
        tok->merge_ht_prio[i] = -1;
    }

    for (int32_t prio = 0; prio < tok->n_merges; prio++) {
        uint64_t key = ((uint64_t)tok->merge_a[prio] << 32) | tok->merge_b[prio];
        int32_t idx = (int32_t)(key & (tok->merge_ht_capacity - 1));

        while (tok->merge_ht_keys[idx] != UINT64_MAX) {
            idx = (idx + 1) % tok->merge_ht_capacity;
        }

        tok->merge_ht_keys[idx] = key;
        tok->merge_ht_prio[idx] = prio;
    }
}

static int32_t merge_ht_lookup(const tokenizer_t* tok, uint32_t a, uint32_t b) {
    if (!tok || tok->merge_ht_capacity == 0) return -1;

    uint64_t key = ((uint64_t)a << 32) | b;
    int32_t idx = (int32_t)(key & (tok->merge_ht_capacity - 1));

    while (tok->merge_ht_keys[idx] != UINT64_MAX) {
        if (tok->merge_ht_keys[idx] == key) {
            return tok->merge_ht_prio[idx];
        }
        idx = (idx + 1) % tok->merge_ht_capacity;
    }

    return -1;
}

tokenizer_t* tokenizer_load(const char* path) {
    if (!path) return NULL;

    byte_to_unicode_init();

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    tokenizer_t* tok = (tokenizer_t*)calloc(1, sizeof(tokenizer_t));
    if (!tok) {
        fclose(f);
        return NULL;
    }

    uint8_t header[TVOC_HEADER_SIZE];
    if (fread(header, TVOC_HEADER_SIZE, 1, f) != 1) {
        fclose(f);
        free(tok);
        return NULL;
    }

    uint32_t magic = *(uint32_t*)(header + 0);
    if (magic != TVOC_MAGIC) {
        fclose(f);
        free(tok);
        return NULL;
    }

    uint32_t version = *(uint32_t*)(header + 4);
    if (version != TVOC_VERSION) {
        fclose(f);
        free(tok);
        return NULL;
    }

    tok->vocab_size = *(uint32_t*)(header + 8);
    tok->n_merges = *(uint32_t*)(header + 12);
    tok->pad_id = *(int32_t*)(header + 16);
    tok->unk_id = *(int32_t*)(header + 20);
    tok->bos_id = *(int32_t*)(header + 24);
    tok->eos_id = *(int32_t*)(header + 28);
    tok->flags = *(uint32_t*)(header + 32);
    uint32_t vocab_data_size = *(uint32_t*)(header + 36);

    if (tok->vocab_size <= 0 || tok->vocab_size > 1000000) {
        fclose(f);
        free(tok);
        return NULL;
    }

    tok->vocab_offsets = (uint32_t*)malloc(tok->vocab_size * sizeof(uint32_t));
    tok->vocab_lens = (uint16_t*)malloc(tok->vocab_size * sizeof(uint16_t));
    if (!tok->vocab_offsets || !tok->vocab_lens) {
        free(tok->vocab_offsets);
        free(tok->vocab_lens);
        fclose(f);
        free(tok);
        return NULL;
    }

    if (fread(tok->vocab_offsets, sizeof(uint32_t), tok->vocab_size, f) != (size_t)tok->vocab_size) {
        free(tok->vocab_offsets);
        free(tok->vocab_lens);
        fclose(f);
        free(tok);
        return NULL;
    }

    tok->vocab_data = (char*)malloc(vocab_data_size);
    if (!tok->vocab_data) {
        free(tok->vocab_offsets);
        free(tok->vocab_lens);
        fclose(f);
        free(tok);
        return NULL;
    }

    if (fread(tok->vocab_data, 1, vocab_data_size, f) != vocab_data_size) {
        free(tok->vocab_offsets);
        free(tok->vocab_lens);
        free(tok->vocab_data);
        fclose(f);
        free(tok);
        return NULL;
    }

    for (int32_t i = 0; i < tok->vocab_size; i++) {
        uint32_t start = tok->vocab_offsets[i];
        uint32_t end = (i + 1 < tok->vocab_size) ? tok->vocab_offsets[i + 1] : vocab_data_size;

        if (start > vocab_data_size || end > vocab_data_size || start > end) {
            free(tok->vocab_offsets);
            free(tok->vocab_lens);
            free(tok->vocab_data);
            fclose(f);
            free(tok);
            return NULL;
        }

        uint32_t len = end - start;
        if (len > UINT16_MAX) {
            free(tok->vocab_offsets);
            free(tok->vocab_lens);
            free(tok->vocab_data);
            fclose(f);
            free(tok);
            return NULL;
        }

        tok->vocab_lens[i] = (uint16_t)len;
    }

    if (tok->n_merges > 0) {
        tok->merge_a = (uint32_t*)malloc(tok->n_merges * sizeof(uint32_t));
        tok->merge_b = (uint32_t*)malloc(tok->n_merges * sizeof(uint32_t));
        tok->merge_result = (uint32_t*)malloc(tok->n_merges * sizeof(uint32_t));

        if (!tok->merge_a || !tok->merge_b || !tok->merge_result) {
            free(tok->merge_a);
            free(tok->merge_b);
            free(tok->merge_result);
            free(tok->vocab_offsets);
            free(tok->vocab_lens);
            free(tok->vocab_data);
            fclose(f);
            free(tok);
            return NULL;
        }

        for (int32_t i = 0; i < tok->n_merges; i++) {
            uint32_t a, b, result;
            if (fread(&a, 4, 1, f) != 1 || fread(&b, 4, 1, f) != 1 || fread(&result, 4, 1, f) != 1) {
                free(tok->merge_a);
                free(tok->merge_b);
                free(tok->merge_result);
                free(tok->vocab_offsets);
                free(tok->vocab_lens);
                free(tok->vocab_data);
                fclose(f);
                free(tok);
                return NULL;
            }
            tok->merge_a[i] = a;
            tok->merge_b[i] = b;
            tok->merge_result[i] = result;
        }

        merge_ht_build(tok);
    }

    fclose(f);

    str_ht_build(tok);

    return tok;
}

void tokenizer_free(tokenizer_t* tok) {
    if (!tok) return;

    free(tok->vocab_data);
    tok->vocab_data = NULL;

    free(tok->vocab_offsets);
    tok->vocab_offsets = NULL;

    free(tok->vocab_lens);
    tok->vocab_lens = NULL;

    free(tok->merge_a);
    tok->merge_a = NULL;

    free(tok->merge_b);
    tok->merge_b = NULL;

    free(tok->merge_result);
    tok->merge_result = NULL;

    free(tok->str_ht_values);
    tok->str_ht_values = NULL;

    free(tok->str_ht_hashes);
    tok->str_ht_hashes = NULL;

    free(tok->merge_ht_keys);
    tok->merge_ht_keys = NULL;

    free(tok->merge_ht_prio);
    tok->merge_ht_prio = NULL;

    free(tok);
}

static int32_t utf8_decode_codepoint(const char* s, int32_t len, uint32_t* out_codepoint, int32_t* out_consumed) {
    if (len < 1) return TVOC_ERR_OVERFLOW;

    uint8_t b0 = (uint8_t)s[0];

    if ((b0 & 0x80) == 0) {
        *out_codepoint = b0;
        *out_consumed = 1;
        return TVOC_OK;
    }

    if ((b0 & 0xE0) == 0xC0) {
        if (len < 2) return TVOC_ERR_OVERFLOW;
        uint8_t b1 = (uint8_t)s[1];
        *out_codepoint = (((uint32_t)(b0 & 0x1F)) << 6) | (b1 & 0x3F);
        *out_consumed = 2;
        return TVOC_OK;
    }

    if ((b0 & 0xF0) == 0xE0) {
        if (len < 3) return TVOC_ERR_OVERFLOW;
        uint8_t b1 = (uint8_t)s[1];
        uint8_t b2 = (uint8_t)s[2];
        *out_codepoint = (((uint32_t)(b0 & 0x0F)) << 12) | (((uint32_t)(b1 & 0x3F)) << 6) | (b2 & 0x3F);
        *out_consumed = 3;
        return TVOC_OK;
    }

    if ((b0 & 0xF8) == 0xF0) {
        if (len < 4) return TVOC_ERR_OVERFLOW;
        uint8_t b1 = (uint8_t)s[1];
        uint8_t b2 = (uint8_t)s[2];
        uint8_t b3 = (uint8_t)s[3];
        *out_codepoint = (((uint32_t)(b0 & 0x07)) << 18) | (((uint32_t)(b1 & 0x3F)) << 12) |
                         (((uint32_t)(b2 & 0x3F)) << 6) | (b3 & 0x3F);
        *out_consumed = 4;
        return TVOC_OK;
    }

    return TVOC_ERR_OVERFLOW;
}

static int32_t utf8_encode_codepoint(uint32_t codepoint, char* out, int32_t max_len) {
    if (codepoint < 0x80) {
        if (max_len < 1) return TVOC_ERR_OVERFLOW;
        out[0] = (char)codepoint;
        return 1;
    }

    if (codepoint < 0x800) {
        if (max_len < 2) return TVOC_ERR_OVERFLOW;
        out[0] = (char)(0xC0 | (codepoint >> 6));
        out[1] = (char)(0x80 | (codepoint & 0x3F));
        return 2;
    }

    if (codepoint < 0x10000) {
        if (max_len < 3) return TVOC_ERR_OVERFLOW;
        out[0] = (char)(0xE0 | (codepoint >> 12));
        out[1] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        out[2] = (char)(0x80 | (codepoint & 0x3F));
        return 3;
    }

    if (codepoint < 0x110000) {
        if (max_len < 4) return TVOC_ERR_OVERFLOW;
        out[0] = (char)(0xF0 | (codepoint >> 18));
        out[1] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        out[3] = (char)(0x80 | (codepoint & 0x3F));
        return 4;
    }

    return TVOC_ERR_OVERFLOW;
}

int32_t tokenizer_encode(const tokenizer_t* tok, const char* text,
                         int32_t* out_tokens, int32_t max_tokens) {
    if (!tok || !text || !out_tokens || max_tokens <= 0) return TVOC_ERR_OVERFLOW;

    int32_t text_len = (int32_t)strlen(text);
    if (text_len == 0) return 0;

    byte_to_unicode_init();

    char* mapped = (char*)malloc(text_len * 4 + 1);
    if (!mapped) return TVOC_ERR_ALLOC;

    int32_t mapped_pos = 0;
    for (int32_t i = 0; i < text_len; i++) {
        uint8_t b = (uint8_t)text[i];
        uint32_t code = byte_to_unicode_table[b];

        int32_t encoded = utf8_encode_codepoint(code, mapped + mapped_pos, text_len * 4 - mapped_pos);
        if (encoded < 0) {
            free(mapped);
            return TVOC_ERR_OVERFLOW;
        }
        mapped_pos += encoded;
    }
    mapped[mapped_pos] = '\0';

    uint32_t space_code = byte_to_unicode_table[' '];

    int32_t token_count = 0;
    int32_t* segment_tokens = (int32_t*)malloc(max_tokens * sizeof(int32_t));
    if (!segment_tokens) {
        free(mapped);
        return TVOC_ERR_ALLOC;
    }

    int32_t pos = 0;
    while (pos < mapped_pos && token_count < max_tokens) {
        uint32_t codepoint;
        int32_t consumed;

        int32_t result = utf8_decode_codepoint(mapped + pos, mapped_pos - pos, &codepoint, &consumed);
        if (result != TVOC_OK) {
            free(segment_tokens);
            free(mapped);
            return result;
        }

        if (codepoint == space_code && pos > 0) {
            pos += consumed;
            continue;
        }

        int32_t seg_start = pos;
        while (pos < mapped_pos) {
            result = utf8_decode_codepoint(mapped + pos, mapped_pos - pos, &codepoint, &consumed);
            if (result != TVOC_OK) break;

            if (codepoint == space_code && pos > seg_start) break;

            pos += consumed;
        }

        int32_t seg_len = pos - seg_start;
        int32_t seg_tokens = 0;

        pos = seg_start;
        while (pos < seg_start + seg_len && seg_tokens < max_tokens) {
            result = utf8_decode_codepoint(mapped + pos, seg_start + seg_len - pos, &codepoint, &consumed);
            if (result != TVOC_OK) break;

            char char_utf8[4];
            int32_t char_len = utf8_encode_codepoint(codepoint, char_utf8, 4);
            if (char_len <= 0) break;

            int32_t token_id = str_ht_lookup(tok, char_utf8, char_len);
            if (token_id < 0) token_id = tok->unk_id;

            segment_tokens[seg_tokens++] = token_id;
            pos += consumed;
        }

        while (seg_tokens > 1 && token_count + seg_tokens <= max_tokens) {
            int32_t best_pos = -1;
            int32_t best_prio = INT32_MAX;

            for (int32_t i = 0; i < seg_tokens - 1; i++) {
                int32_t prio = merge_ht_lookup(tok, segment_tokens[i], segment_tokens[i + 1]);
                if (prio >= 0 && prio < best_prio) {
                    best_prio = prio;
                    best_pos = i;
                }
            }

            if (best_pos < 0) break;

            segment_tokens[best_pos] = tok->merge_result[best_prio];

            for (int32_t i = best_pos + 1; i < seg_tokens - 1; i++) {
                segment_tokens[i] = segment_tokens[i + 1];
            }
            seg_tokens--;
        }

        for (int32_t i = 0; i < seg_tokens && token_count < max_tokens; i++) {
            out_tokens[token_count++] = segment_tokens[i];
        }
    }

    free(segment_tokens);
    free(mapped);

    return token_count;
}

int32_t tokenizer_decode(const tokenizer_t* tok, const int32_t* tokens,
                         int32_t n_tokens, char* out_text, int32_t max_len) {
    if (!tok || !out_text || max_len <= 0) return TVOC_ERR_OVERFLOW;
    if (n_tokens == 0) { out_text[0] = '\0'; return 0; }
    if (!tokens) return TVOC_ERR_OVERFLOW;

    byte_to_unicode_init();

    char* temp = (char*)malloc(max_len);
    if (!temp) return TVOC_ERR_ALLOC;

    int32_t pos = 0;
    for (int32_t i = 0; i < n_tokens; i++) {
        int32_t token_id = tokens[i];

        if (token_id < 0 || token_id >= tok->vocab_size) {
            free(temp);
            return TVOC_ERR_OVERFLOW;
        }

        int32_t str_len = tok->vocab_lens[token_id];
        const char* str = tok->vocab_data + tok->vocab_offsets[token_id];

        if (pos + str_len > max_len) {
            free(temp);
            return TVOC_ERR_OVERFLOW;
        }

        memcpy(temp + pos, str, str_len);
        pos += str_len;
    }

    if (tok->flags & TVOC_FLAG_BYTE_LEVEL) {
        int32_t out_pos = 0;
        int32_t i = 0;

        while (i < pos && out_pos < max_len - 1) {
            uint32_t codepoint;
            int32_t consumed;

            int32_t result = utf8_decode_codepoint(temp + i, pos - i, &codepoint, &consumed);
            if (result != TVOC_OK) break;

            if (codepoint < 65536 && unicode_to_byte_table[codepoint] != 0xFF) {
                uint8_t byte = unicode_to_byte_table[codepoint];
                out_text[out_pos++] = (char)byte;
            } else {
                int32_t encoded = utf8_encode_codepoint(codepoint, out_text + out_pos, max_len - out_pos);
                if (encoded < 0) break;
                out_pos += encoded;
            }

            i += consumed;
        }

        out_text[out_pos] = '\0';
        free(temp);
        return out_pos;
    } else {
        if (pos > max_len - 1) {
            free(temp);
            return TVOC_ERR_OVERFLOW;
        }

        memcpy(out_text, temp, pos);
        out_text[pos] = '\0';
        free(temp);
        return pos;
    }
}

int32_t tokenizer_decode_token(const tokenizer_t* tok, int32_t token_id,
                               char* out_text, int32_t max_len) {
    if (!tok || !out_text || max_len <= 0) return TVOC_ERR_OVERFLOW;

    if (token_id < 0 || token_id >= tok->vocab_size) return TVOC_ERR_OVERFLOW;

    byte_to_unicode_init();

    int32_t str_len = tok->vocab_lens[token_id];
    const char* str = tok->vocab_data + tok->vocab_offsets[token_id];

    if (str_len > max_len - 1) return TVOC_ERR_OVERFLOW;

    if (!(tok->flags & TVOC_FLAG_BYTE_LEVEL)) {
        memcpy(out_text, str, str_len);
        out_text[str_len] = '\0';
        return str_len;
    }

    int32_t out_pos = 0;
    int32_t i = 0;

    while (i < str_len && out_pos < max_len - 1) {
        uint32_t codepoint;
        int32_t consumed;

        int32_t result = utf8_decode_codepoint(str + i, str_len - i, &codepoint, &consumed);
        if (result != TVOC_OK) break;

        if (codepoint < 65536 && unicode_to_byte_table[codepoint] != 0xFF) {
            uint8_t byte = unicode_to_byte_table[codepoint];
            out_text[out_pos++] = (char)byte;
        } else {
            int32_t encoded = utf8_encode_codepoint(codepoint, out_text + out_pos, max_len - out_pos);
            if (encoded < 0) break;
            out_pos += encoded;
        }

        i += consumed;
    }

    out_text[out_pos] = '\0';
    return out_pos;
}

const char* tvoc_strerror(int err) {
    switch (err) {
        case TVOC_OK:           return "OK";
        case TVOC_ERR_IO:       return "I/O error";
        case TVOC_ERR_MAGIC:    return "bad magic bytes";
        case TVOC_ERR_VERSION:  return "unsupported version";
        case TVOC_ERR_ALLOC:    return "memory allocation failure";
        case TVOC_ERR_OVERFLOW: return "buffer overflow";
        default:                return "unknown error";
    }
}
