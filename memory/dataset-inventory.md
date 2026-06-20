---
name: Dataset inventory
description: All datasets available for training — sizes, locations, preparation scripts
type: reference
---

# Dataset Inventory

**Last updated:** 2026-03-23

## Available Datasets

| Dataset | Size | Tokens (est.) | Format | Location | Prep Script |
|---------|------|---------------|--------|----------|-------------|
| WikiText-103 | ~180 MB | ~100M | uint16 .bin | `data/wikitext/` | `prepare_wikitext.py` |
| Curated v11 | 2.96 MB | ~12M | uint8 .bin (byte-level) | `data/curated/full_dataset.bin` | internal pipeline |
| Curated v11 (raw) | 78 .jsonl files | 5,150 examples | .jsonl | `data/curated/` | append scripts |
| Book pipeline | varies | varies | uint32 .bin | configurable | `prepare_book.py` |
| Combined | configurable | depends on sources | uint16 .bin | configurable | `prepare_combined_dataset.py` |
| Binary generic | any text | varies | auto-detect | configurable | `prepare_binary_dataset.py` |

## Tokenization

| Vocab | Type | Used in |
|-------|------|---------|
| 256 | Byte-level | 7B PoC experiments (R121, R130) |
| 32,000 | BPE WordPiece | WikiText, standard training |

## Data Pipeline (Source->Signal)

6-stage pipeline for curated data:
1. **Parse** — raw text extraction
2. **Score** — quality scoring
3. **Extract** — facts (definitions, procedures, commands, triples)
4. **Generate** — 3 view types (statement, QA, contrastive)
5. **Filter + Dedup** — quality filtering
6. **Curriculum** — ordering by difficulty

Scripts in `src/model_v11/data_pipeline/`

## Known Issues

- `ref_losses.bin` required by `prepare_combined_dataset.py` combine mode — never generated, will cause FileNotFoundError
- Mixed dtype: uint16 (wikitext/combined) vs uint32 (book) — potential compatibility issue

## Scale Context

| What | Tokens | Ratio to ours |
|------|--------|---------------|
| Our curated dataset | ~12M | 1x |
| WikiText-103 | ~100M | ~8x |
| LLaMA 2 training | 2T | ~167,000x |
| Chinchilla optimal for 7B | ~140B | ~11,667x |
