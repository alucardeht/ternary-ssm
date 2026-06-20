import random
from typing import List
from . import TrainingExample
from .config import PipelineConfig


def order_curriculum(
    examples: List[TrainingExample],
    config: PipelineConfig
) -> List[TrainingExample]:
    """Order training examples in curriculum order.

    Strategies:
    - concept_first: Sort by difficulty (1→5), then by info_score descending.
      Insert repetitions at 33% and 66% positions.
    - info_ordered: Sort purely by info_score descending
    - random: Shuffle with seed

    Args:
        examples: List of TrainingExample objects
        config: PipelineConfig with curriculum settings

    Returns:
        Ordered list of TrainingExample objects
    """
    if not config.curriculum_enabled:
        return examples

    if config.curriculum_strategy == 'random':
        rng = random.Random(config.seed)
        result = examples.copy()
        rng.shuffle(result)
        return result

    if config.curriculum_strategy == 'info_ordered':
        return sorted(examples, key=lambda ex: -ex.info_score)

    if config.curriculum_strategy == 'concept_first':
        rng = random.Random(config.seed)

        sorted_examples = sorted(
            examples,
            key=lambda ex: (ex.difficulty, -ex.info_score)
        )

        top_k = int(len(sorted_examples) * config.repetition_factor)
        top_examples = sorted(
            sorted_examples,
            key=lambda ex: -ex.info_score
        )[:top_k]

        result = sorted_examples.copy()

        insertion_points = [
            int(len(result) * 0.33),
            int(len(result) * 0.66),
        ]

        repetitions_to_add = [(pos, ex) for pos in insertion_points
                               for ex in rng.sample(top_examples, min(5, len(top_examples)))]
        repetitions_to_add.sort(key=lambda x: -x[0], reverse=False)

        for pos, example in repetitions_to_add:
            result.insert(pos, example)

        return result

    return examples
