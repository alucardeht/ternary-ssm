from dataclasses import dataclass


@dataclass
class PipelineConfig:
    input_format: str = "text"
    chunk_strategy: str = "paragraph"
    min_chunk_tokens: int = 20
    max_chunk_tokens: int = 512

    score_top_k_pct: float = 0.25
    use_model_scoring: bool = False

    extract_triples: bool = True
    extract_procedures: bool = True
    extract_definitions: bool = True
    extract_code_blocks: bool = True

    generate_qa: bool = True
    generate_contrastive: bool = True
    generate_statement: bool = True
    views_per_fact: int = 14

    dedup_ngram_size: int = 5
    dedup_threshold: float = 0.85
    min_example_tokens: int = 10
    max_example_tokens: int = 256
    max_examples: int = 0

    curriculum_enabled: bool = True
    curriculum_strategy: str = "concept_first"
    repetition_factor: float = 0.15

    seq_len: int = 512
    seed: int = 42
