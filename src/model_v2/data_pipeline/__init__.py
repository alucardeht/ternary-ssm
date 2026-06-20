from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    text: str
    section: str
    chunk_type: str
    position: int


@dataclass
class ScoredChunk:
    text: str
    section: str
    chunk_type: str
    position: int
    info_score: float
    perplexity: float
    specificity: float


@dataclass
class Fact:
    subject: str
    predicate: str
    object_: str
    fact_type: str
    source_text: str


@dataclass
class TrainingExample:
    text: str
    view_type: str
    source_fact: str
    difficulty: int
    info_score: float


@dataclass
class PipelineResult:
    n_chunks: int
    n_scored_chunks: int
    n_facts: int
    n_examples: int
    n_filtered: int
    output_path: str
    timing: dict


def run_pipeline(
    input_path: str,
    output_dir: str,
    model_path: str = None,
    tokenizer_path: str = None,
    config=None,
) -> PipelineResult:
    """Run the full Source->Signal pipeline.

    Stages:
    1. Parse document -> List[Chunk]
    2. Score chunks -> List[ScoredChunk] (filtered to top-k)
    3. Extract facts -> List[Fact]
    4. Generate views -> List[TrainingExample]
    5. Filter examples -> List[TrainingExample]
    6. Order curriculum -> List[TrainingExample]
    7. Save dataset -> output files

    Args:
        input_path: Path to input text file
        output_dir: Output directory
        model_path: Optional model checkpoint for perplexity scoring
        tokenizer_path: Path to tokenizer file
        config: PipelineConfig (uses defaults if None)

    Returns:
        PipelineResult with statistics and timing
    """
    import time
    from .config import PipelineConfig
    from .parsers import parse_document
    from .scorers import score_chunks
    from .extractors import extract_facts
    from .generators import generate_views
    from .filters import filter_examples
    from .curriculum import order_curriculum
    from .dataset import save_dataset

    if config is None:
        config = PipelineConfig()

    timing = {}

    print("[1/7] Parsing document...")
    start = time.time()
    chunks = parse_document(input_path, config)
    timing['parse'] = time.time() - start
    print(f"  -> {len(chunks)} chunks ({timing['parse']:.2f}s)")

    print("[2/7] Scoring chunks...")
    start = time.time()
    model = None
    tokenizer = None
    if model_path:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"  Warning: Could not load model {model_path}: {e}")
            model = None
            tokenizer = None

    scored_chunks = score_chunks(chunks, config, model, tokenizer)
    timing['score'] = time.time() - start
    print(f"  -> {len(scored_chunks)} scored chunks ({timing['score']:.2f}s)")

    print("[3/7] Extracting facts...")
    start = time.time()
    facts = extract_facts(scored_chunks, config)
    timing['extract'] = time.time() - start
    print(f"  -> {len(facts)} facts ({timing['extract']:.2f}s)")

    print("[4/7] Generating views...")
    start = time.time()
    examples = generate_views(facts, config, scored_chunks)
    timing['generate'] = time.time() - start
    print(f"  -> {len(examples)} examples ({timing['generate']:.2f}s)")

    print("[5/7] Filtering examples...")
    start = time.time()
    filtered_examples, filter_stats = filter_examples(examples, config)
    timing['filter'] = time.time() - start
    print(f"  -> {len(filtered_examples)} examples after filtering ({timing['filter']:.2f}s)")
    for stat_name, stat_val in filter_stats.items():
        if stat_name != 'n_output':
            print(f"     {stat_name}: {stat_val}")

    print("[6/7] Ordering curriculum...")
    start = time.time()
    curriculum_examples = order_curriculum(filtered_examples, config)
    timing['curriculum'] = time.time() - start
    print(f"  -> {len(curriculum_examples)} examples ordered ({timing['curriculum']:.2f}s)")

    print("[7/7] Saving dataset...")
    start = time.time()
    save_dataset(curriculum_examples, output_dir, tokenizer_path, config.seq_len)
    timing['save'] = time.time() - start
    print(f"  -> Saved to {output_dir} ({timing['save']:.2f}s)")

    return PipelineResult(
        n_chunks=len(chunks),
        n_scored_chunks=len(scored_chunks),
        n_facts=len(facts),
        n_examples=len(examples),
        n_filtered=len(curriculum_examples),
        output_path=str(output_dir),
        timing=timing
    )
