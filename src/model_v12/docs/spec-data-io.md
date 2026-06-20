# Specification: Data I/O Layer
# Files: src/io/dataset.c + src/io/tokenizer.c

**Status:** Spec — not yet implemented
**Architecture ref:** docs/architecture.md
**Vocab:** 32064 | **Seq len:** 512 | **Token type:** uint16_t

---

## 1. Constraints and Invariants

These apply to both files without exception:

1. **No dynamic vocab beyond uint16 range.** Vocab = 32064 < 65536. Token IDs always fit uint16_t.
2. **No hidden allocations.** Every allocation goes through explicit `malloc`/`mmap`. Caller owns all returned pointers and frees via the provided `_free()` functions.
3. **No global state.** All state lives in the structs. Functions are reentrant if different struct instances are used.
4. **Compile-time seq_len.** `SEQ_LEN` and `BATCH_SIZE` are set at compile time (CMake flags). Runtime values passed to `dataset_sample_batch` must be <= these limits.
5. **C99/C11 only.** No C++ extensions, no platform-specific intrinsics, no POSIX beyond `mmap`/`open` with a scalar fallback.

---

## 2. dataset.c — Binary Token File Loader

### 2.1 File Format

```
[header: 4 bytes magic][header: 4 bytes version][header: 8 bytes n_tokens]
[token data: n_tokens × uint16_t, little-endian, tightly packed]
```

Magic: `0x544F4B4E` ("TOKN"). Version: `0x00000001`.

If a file has no header (raw dump of uint16 values), the loader detects this by checking magic. If magic is absent, treat entire file as raw token data (n_tokens = file_size / 2).

Both cases are valid. The header format is preferred for new files.

### 2.2 Struct Definition

```c
// src/io/dataset.h

#ifndef DATASET_H
#define DATASET_H

#include <stdint.h>
#include <stddef.h>

#define DATASET_MAGIC   0x544F4B4E
#define DATASET_VERSION 1

typedef struct {
    uint16_t  *data;        // pointer to token array (mmap or malloc)
    size_t     n_tokens;    // total number of tokens
    int        is_mmap;     // 1 = mmap-backed, 0 = malloc-backed
    int        fd;          // file descriptor (mmap only, else -1)
    size_t     mmap_size;   // bytes mapped (mmap only)
    uint64_t   rng_state;   // xoshiro256** state (4 × uint64)
    // rng_state is actually [4] but kept opaque here; see impl notes
} Dataset;

Dataset*  dataset_load(const char *path);
void      dataset_sample_batch(Dataset *ds,
                               uint16_t *batch,
                               int batch_size,
                               int seq_len);
void      dataset_shuffle(Dataset *ds);
void      dataset_free(Dataset *ds);

#endif
```

### 2.3 dataset_load

```
Dataset* dataset_load(const char *path)
```

**Steps:**

1. Open file with `open(path, O_RDONLY)`.
2. `fstat` to get file size.
3. Read first 16 bytes to check magic + version + n_tokens.
   - If magic matches: use header values. Data starts at byte 16.
   - If magic absent: raw mode. `n_tokens = file_size / 2`. Data starts at byte 0.
4. Attempt `mmap(NULL, data_size, PROT_READ, MAP_PRIVATE, fd, data_offset)`.
   - On success: `ds->is_mmap = 1`, `ds->data = mmap_ptr`.
   - On failure (or Windows where mmap is unavailable): fall back to `malloc` + `fread`.
5. Seed RNG from current time XOR `(uintptr_t)ds`.
6. Return `ds`. Return `NULL` on any error; print error to stderr.

**POSIX mmap fallback for portability:**

```c
// Compile with -DUSE_MMAP=0 to force malloc path (e.g., Windows)
#if defined(USE_MMAP) && USE_MMAP == 0
  // always use malloc path
#else
  // attempt mmap, fall back to malloc on ENOTSUP or MAP_FAILED
#endif
```

This keeps the implementation portable to Windows (where `CreateFileMapping` is not used; we just take the malloc path via the flag).

### 2.4 dataset_sample_batch

```
void dataset_sample_batch(Dataset *ds,
                          uint16_t *batch,
                          int batch_size,
                          int seq_len)
```

**Contract:**
- `batch` is a caller-allocated buffer of `batch_size * seq_len` uint16_t values.
- Output layout: row-major, `batch[b * seq_len + t]` = token at position t in sample b.

**Steps:**

For each of the `batch_size` samples:
1. Pick `offset = rng_next(&ds->rng_state) % (ds->n_tokens - seq_len)`.
2. `memcpy(&batch[b * seq_len], &ds->data[offset], seq_len * sizeof(uint16_t))`.

No bounds check beyond the subtraction above. Caller must ensure `seq_len < ds->n_tokens`.

**RNG:** xoshiro256** — fast, no division, no modulo bias beyond the 2^64 truncation (acceptable for data sampling). State is 4 × uint64_t stored in the Dataset struct as `uint64_t rng[4]`. Rename the struct field to `uint64_t rng[4]` in final implementation (the header above used `rng_state` as placeholder for clarity).

### 2.5 dataset_shuffle

```
void dataset_shuffle(Dataset *ds)
```

Performs a Fisher-Yates shuffle over the token array.

**Note:** Only valid when `is_mmap == 0` (mmap is read-only). If called on a mmap-backed dataset, allocate a writable copy first, then shuffle in place. Emit a one-time `fprintf(stderr, "[dataset] mmap copy for shuffle\n")` warning.

This function is optional for training (random sampling without replacement via `dataset_sample_batch` is sufficient for most cases) but required for epoch-based curriculum.

### 2.6 dataset_free

```
void dataset_free(Dataset *ds)
```

- If `is_mmap`: `munmap(ds->data, ds->mmap_size)` + `close(ds->fd)`.
- Else: `free(ds->data)`.
- `free(ds)`.

---

## 3. tokenizer.c — BPE Tokenizer

### 3.1 Source Format: tokenizer.json

HuggingFace tokenizer format. Relevant fields:

```json
{
  "model": {
    "vocab": { "<token_string>": <id>, ... },
    "merges": [ "<a> <b>", ... ]
  },
  "added_tokens": [
    { "id": 32000, "content": "<|user|>", "special": true },
    ...
  ]
}
```

The loader parses only these fields. All other fields are ignored.

**Special tokens required by v12:**

| Token | Meaning |
|-------|---------|
| `<\|user\|>` | Start of user turn |
| `<\|assistant\|>` | Start of assistant turn |
| `<\|end\|>` | End of turn |
| `<\|pad\|>` | Padding (id = 0 if absent) |
| `<\|unk\|>` | Unknown (fallback for encode) |

If any of the above are absent from the file, assign them IDs starting at `vocab_size` and expand vocab accordingly. Never silently drop a special token.

### 3.2 JSON Parser

**Do not link an external JSON library.** Implement a minimal single-pass parser sufficient for the tokenizer.json structure. This parser needs to handle:

- Nested objects (2 levels: model → vocab/merges)
- String keys and values
- Integer values
- Array of strings (merges)
- Array of objects (added_tokens)
- Unicode escape sequences `\uXXXX` (convert to UTF-8)
- Escaped quotes `\"` inside strings

The parser does NOT need to handle:
- Numbers with exponents
- Null / boolean
- Deeply nested arbitrary JSON

A ~300-line recursive descent parser is sufficient. Keep it in a separate translation unit `src/io/json_parser.c` with internal linkage (all functions `static`), included by `tokenizer.c` via a direct `#include "json_parser.c"` or compiled as a separate object. Prefer separate object to avoid violating ODR if ever linked from multiple units.

### 3.3 Struct Definition

```c
// src/io/tokenizer.h

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#define TOK_MAX_VOCAB   65536
#define TOK_MAX_MERGES  100000
#define TOK_MAX_TOKEN_BYTES 32

typedef struct {
    // vocab: id → string
    char    **id_to_str;      // [vocab_size] — heap-allocated strings
    uint16_t  vocab_size;     // actual size (may exceed 32064 if special tokens added)

    // vocab: string → id (hash map)
    // Using open-addressing hash map keyed on string content
    // Implementation detail: see §3.4
    void     *str_to_id;      // opaque pointer to hash map

    // BPE merge table: ordered list of (a, b) → merged token
    // merges[i] = { left_id, right_id, result_id }
    uint16_t (*merges)[3];    // [n_merges][3]
    int       n_merges;

    // merge priority table: for each (a, b) pair, their rank (index in merges[])
    // Used for O(n log n) BPE encoding
    // Implementation: same hash map type as str_to_id, key = (a<<16|b), value = rank
    void     *merge_ranks;    // opaque pointer to hash map

    // special token IDs
    uint16_t  tok_user;
    uint16_t  tok_assistant;
    uint16_t  tok_end;
    uint16_t  tok_pad;
    uint16_t  tok_unk;
} Tokenizer;

Tokenizer* tokenizer_load(const char *json_path);
int        tokenizer_encode(Tokenizer *tok,
                            const char *text,
                            uint16_t *ids,
                            int max_len);
char*      tokenizer_decode(Tokenizer *tok,
                            const uint16_t *ids,
                            int len);
void       tokenizer_free(Tokenizer *tok);

#endif
```

### 3.4 Hash Map

Both `str_to_id` and `merge_ranks` use the same open-addressing hash map implementation:

```
// src/io/hashmap.c (internal)
// Key type: const char* (string) or uint32_t (pair)
// Value type: uint16_t
// Load factor: 0.65
// Hash: FNV-1a for strings, identity-scramble for uint32 keys
// Collision: linear probing
```

Keep this in `src/io/hashmap.c` with a minimal header. Capacity is next power-of-two above `vocab_size / 0.65`.

### 3.5 tokenizer_load

```
Tokenizer* tokenizer_load(const char *json_path)
```

**Steps:**

1. Read entire file into a malloc'd buffer (`fread` in one shot after `fstat`).
2. Parse JSON with the minimal parser:
   - Extract `model.vocab` → populate `id_to_str[]` and `str_to_id` hash map.
   - Extract `model.merges` (array of `"a b"` strings) → populate `merges[][3]` and `merge_ranks` hash map. Rank = index in array.
   - Extract `added_tokens` → for each with `special: true`, record in `id_to_str` and hash map.
3. Resolve special token IDs from hash map. Assign new IDs for any missing ones.
4. Free the JSON buffer.
5. Return `tok`. Return `NULL` on parse error; print reason to stderr.

**Memory budget (32K vocab, 100K merges):**
- `id_to_str`: 32064 pointers × 8 B = 256 KB + string data ~1.2 MB
- `str_to_id` hash map: ~800 KB
- `merges`: 100K × 6 B = 600 KB
- `merge_ranks` hash map: ~800 KB
- **Total: ~3.6 MB** — well within the 16 GB RAM constraint.

### 3.6 tokenizer_encode

```
int tokenizer_encode(Tokenizer *tok,
                     const char *text,
                     uint16_t *ids,
                     int max_len)
```

**Contract:**
- Writes up to `max_len` token IDs into `ids`.
- Returns the number of tokens written. Returns -1 on error.
- Input `text` is UTF-8. The encoder works at the byte level for the initial split (standard GPT-2 BPE approach).

**Algorithm (standard BPE, O(n log n) with priority queue):**

1. **Pre-tokenize:** Split `text` on whitespace boundaries using the GPT-2 regex pattern (space-prefixed words). Each "word" is a separate BPE unit. For simplicity in C, implement a character-class based splitter: split on boundaries between `[a-zA-Z0-9]` and other characters, preserving the leading space convention (i.e., non-first words get a `Ġ` prefix = `\xc4\xa0` in UTF-8, or use the byte-level BPE approach where space = token 220).
2. **Byte-level init:** Convert each character to its byte-level token ID. GPT-2 BPE uses a 256-entry byte vocabulary as the base. IDs 0–255 correspond to bytes.
3. **Merge loop:** Use a min-heap (priority queue) keyed on merge rank. For each adjacent pair in the current token sequence, look up rank in `merge_ranks`. Push all pairs. Pop minimum rank pair, merge, update neighbors, push new pairs. Repeat until no merges remain.
4. **Append to `ids`** up to `max_len`. If the result exceeds `max_len`, truncate and return `max_len`.
5. If a character has no byte mapping and no vocab entry, emit `tok->tok_unk`.

**Min-heap for merge priorities:** Implement a simple binary min-heap over a fixed-size array of `(rank, position)` pairs. Max size = 2 × initial_token_count (merges reduce length). Heap allocated on stack for typical sequences (seq_len=512).

**Stack allocation bound:** For seq_len=512, max initial tokens = 512 × 4 bytes/char = 2048 bytes. Heap size = 4096 entries × 8 B = 32 KB. This is safe for a typical 1–8 MB thread stack.

### 3.7 tokenizer_decode

```
char* tokenizer_decode(Tokenizer *tok,
                       const uint16_t *ids,
                       int len)
```

**Contract:**
- Returns a null-terminated heap-allocated UTF-8 string.
- Caller must `free()` the returned pointer.
- Returns NULL on error.

**Steps:**

1. Compute total output byte length: iterate `ids`, sum `strlen(id_to_str[ids[i]])`.
2. `malloc(total + 1)`.
3. Concatenate strings. For GPT-2 byte-level BPE, convert `Ġ` (`\xc4\xa0`) back to space.
4. Null-terminate and return.

Special tokens (`tok_user`, `tok_assistant`, `tok_end`, `tok_pad`) are decoded as their string representation (`<|user|>` etc.) unless the caller filters them out first.

### 3.8 tokenizer_free

```
void tokenizer_free(Tokenizer *tok)
```

- Free each string in `id_to_str`.
- Free `id_to_str` array.
- Free `str_to_id` hash map (via `hashmap_free`).
- Free `merges`.
- Free `merge_ranks` hash map.
- `free(tok)`.

---

## 4. File and Directory Layout

```
src/model_v12/
├── src/
│   └── io/
│       ├── dataset.h
│       ├── dataset.c
│       ├── tokenizer.h
│       ├── tokenizer.c
│       ├── hashmap.h        (internal, used by tokenizer.c only)
│       ├── hashmap.c
│       └── json_parser.c    (internal, compiled as separate object)
└── docs/
    └── spec-data-io.md      (this file)
```

No other files in `src/io/` at this stage.

---

## 5. Error Handling Contract

All functions follow the same convention:

| Return type | Error sentinel | Side effect |
|-------------|----------------|-------------|
| `Dataset*` | `NULL` | `fprintf(stderr, ...)` |
| `Tokenizer*` | `NULL` | `fprintf(stderr, ...)` |
| `int` (encode) | `-1` | `fprintf(stderr, ...)` |
| `char*` (decode) | `NULL` | `fprintf(stderr, ...)` |

No exceptions. No `exit()` calls inside library functions (caller decides what to do with NULL). The main training loop may abort on NULL but that is not the library's responsibility.

---

## 6. Dependencies on the Rest of v12

- `dataset.c` and `tokenizer.c` have **zero dependencies** on any other v12 component. They can be compiled and unit-tested in isolation.
- They depend only on: `<stdint.h>`, `<stddef.h>`, `<stdlib.h>`, `<string.h>`, `<stdio.h>`, `<fcntl.h>`, `<sys/mman.h>` (or the malloc fallback), `<sys/stat.h>`.
- The rest of v12 (model, training loop) calls into this layer via the public headers only.

---

## 7. Unit Test Sketch (not yet implemented)

```c
// test_data_io.c — smoke tests

void test_dataset_roundtrip() {
    // Write 1000 uint16 tokens to a temp .bin file
    // dataset_load → dataset_sample_batch(batch_size=4, seq_len=8)
    // Assert: all IDs in [0, 32063], no OOB read
}

void test_tokenizer_encode_decode() {
    Tokenizer *tok = tokenizer_load("data/tokenizer.json");
    const char *text = "Hello, world!";
    uint16_t ids[64];
    int n = tokenizer_encode(tok, text, ids, 64);
    assert(n > 0);
    char *decoded = tokenizer_decode(tok, ids, n);
    // decoded should equal text (up to BPE round-trip fidelity)
    free(decoded);
    tokenizer_free(tok);
}

void test_special_tokens_present() {
    Tokenizer *tok = tokenizer_load("data/tokenizer.json");
    assert(tok->tok_user      < tok->vocab_size);
    assert(tok->tok_assistant < tok->vocab_size);
    assert(tok->tok_end       < tok->vocab_size);
    tokenizer_free(tok);
}
```

---

## 8. Implementation Order

1. `hashmap.c` — needed by tokenizer; ~150 lines.
2. `json_parser.c` — needed by tokenizer; ~300 lines.
3. `tokenizer.c` — depends on 1 and 2.
4. `dataset.c` — independent; ~200 lines.
5. Unit tests in `test_data_io.c`.

Total estimated lines: ~900 across all four files.

---

## 9. Open Questions for Implementation

- **Pre-tokenization regex:** GPT-2 uses a Rust regex. The C implementation should replicate the split behavior with a hand-written character-class automaton. This is sufficient for byte-level BPE correctness; document any deviation.
- **Byte-level base vocabulary:** Confirm the tokenizer.json being used is GPT-2 / LLaMA-style (256 byte entries + merges). If it is a character-level vocab instead, the encode algorithm in §3.6 needs adjustment (skip byte-level init, start from character IDs directly).
- **Windows mmap:** Use `USE_MMAP=0` flag in CMake for Windows builds. No `CreateFileMapping` — keep the fallback malloc path as the Windows default.
