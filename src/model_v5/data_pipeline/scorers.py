import re
from typing import List, Tuple, Optional
from . import Chunk, ScoredChunk
from .config import PipelineConfig


def _compute_specificity(text: str) -> float:
    """Compute specificity heuristic: code, numbers, proper nouns, emphasis.

    Returns float in [0, 1].
    """
    features = 0.0

    features += text.count('`') * 2
    features += len(re.findall(r'\d+', text))
    features += len(re.findall(r'\b[A-Z][a-z]+\b', text))
    features += text.count('*') + text.count('_')

    specificity = min(features / 4.0, 1.0)
    return specificity


def score_chunks(
    chunks: List[Chunk],
    config: PipelineConfig,
    model=None,
    tokenizer=None
) -> List[ScoredChunk]:
    """Score chunks by information density.

    Two modes:
    1. Heuristic (default): specificity only
    2. With model: 0.7 * novelty + 0.3 * specificity

    Args:
        chunks: List of Chunk objects to score
        config: PipelineConfig settings
        model: Optional loaded model for perplexity scoring
        tokenizer: Optional tokenizer for encoding chunks

    Returns:
        List of ScoredChunk objects (filtered to top_k_pct by info_score)
    """
    scored = []

    if model is not None and tokenizer is not None:
        torch = __import__('torch')
        torch.no_grad = torch.no_grad

        perplexities = []
        with torch.no_grad():
            for chunk in chunks[:100]:
                tokens = tokenizer.encode(chunk.text).ids
                if not tokens or len(tokens) < 2:
                    perplexities.append(1.0)
                    continue

                input_ids = torch.tensor([tokens[:-1]], dtype=torch.long).to(model.device)
                target_ids = torch.tensor([tokens[1:]], dtype=torch.long).to(model.device)

                logits = model(input_ids).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    reduction='mean'
                )
                perplexities.append(loss.exp().item())

        baseline = sorted(perplexities)[len(perplexities) // 2] if perplexities else 1.0

        for chunk in chunks:
            specificity = _compute_specificity(chunk.text)
            tokens = tokenizer.encode(chunk.text).ids

            if not tokens or len(tokens) < 2:
                perplexity = baseline
                novelty = 0.0
            else:
                with torch.no_grad():
                    input_ids = torch.tensor([tokens[:-1]], dtype=torch.long).to(model.device)
                    target_ids = torch.tensor([tokens[1:]], dtype=torch.long).to(model.device)

                    logits = model(input_ids).logits
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1),
                        reduction='mean'
                    )
                    perplexity = loss.exp().item()

            novelty = max(0.0, (perplexity - baseline) / baseline) if baseline > 0 else 0.0
            novelty = min(novelty, 1.0)

            info_score = 0.7 * novelty + 0.3 * specificity

            scored.append(ScoredChunk(
                text=chunk.text,
                section=chunk.section,
                chunk_type=chunk.chunk_type,
                position=chunk.position,
                info_score=info_score,
                perplexity=perplexity,
                specificity=specificity
            ))
    else:
        for chunk in chunks:
            specificity = _compute_specificity(chunk.text)
            scored.append(ScoredChunk(
                text=chunk.text,
                section=chunk.section,
                chunk_type=chunk.chunk_type,
                position=chunk.position,
                info_score=specificity,
                perplexity=0.0,
                specificity=specificity
            ))

    scored.sort(key=lambda x: -x.info_score)
    cutoff_idx = int(len(scored) * config.score_top_k_pct)
    return scored[:cutoff_idx]
