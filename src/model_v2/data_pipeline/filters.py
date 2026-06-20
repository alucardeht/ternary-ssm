import hashlib
from typing import List, Tuple, Dict
from . import TrainingExample
from .config import PipelineConfig


def _normalize_text(text: str) -> str:
    """Normalize text for exact deduplication: lowercase, strip whitespace."""
    return text.lower().strip()


def _get_ngrams(text: str, n: int) -> set:
    """Extract word n-grams from text."""
    words = text.split()
    return set(
        ' '.join(words[i:i+n])
        for i in range(len(words) - n + 1)
    ) if len(words) >= n else set()


def _jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def filter_examples(
    examples: List[TrainingExample],
    config: PipelineConfig
) -> Tuple[List[TrainingExample], Dict]:
    """Filter and deduplicate training examples.

    Steps:
    1. Length filter: remove examples outside token bounds
    2. Exact dedup: hash-based exact match removal
    3. Near-dedup: n-gram Jaccard similarity (threshold-based)
    4. Max cap: truncate to config.max_examples if specified

    Returns:
        Tuple of (filtered_examples, stats_dict)
    """
    stats = {
        'n_input': len(examples),
        'n_length_filtered': 0,
        'n_exact_dedup': 0,
        'n_near_dedup': 0,
        'n_output': 0,
    }

    filtered = []
    seen_hashes = set()
    kept_examples = []

    for example in examples:
        text = example.text
        token_count = len(text.split())

        if token_count < config.min_example_tokens or token_count > config.max_example_tokens:
            stats['n_length_filtered'] += 1
            continue

        normalized = _normalize_text(text)
        text_hash = hashlib.md5(normalized.encode()).hexdigest()

        if text_hash in seen_hashes:
            stats['n_exact_dedup'] += 1
            continue

        seen_hashes.add(text_hash)

        ngrams_current = _get_ngrams(text, config.dedup_ngram_size)

        is_duplicate = False
        for kept_ex in kept_examples[-100:]:
            ngrams_kept = _get_ngrams(kept_ex.text, config.dedup_ngram_size)
            similarity = _jaccard_similarity(ngrams_current, ngrams_kept)
            if similarity > config.dedup_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            stats['n_near_dedup'] += 1
            continue

        filtered.append(example)
        kept_examples.append(example)

    if config.max_examples > 0:
        filtered = filtered[:config.max_examples]

    stats['n_output'] = len(filtered)

    return filtered, stats
