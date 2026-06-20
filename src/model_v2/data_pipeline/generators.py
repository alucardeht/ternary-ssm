import re
import random
from typing import List
from . import Fact, TrainingExample
from .config import PipelineConfig


STATEMENT_TEMPLATES = {
    "definition": [
        "{subject} is defined as {object_}.",
        "{subject} refers to {object_}.",
        "The definition of {subject} is {object_}.",
        "{subject} means {object_}.",
    ],
    "procedure": [
        "The procedure for {subject} involves {object_}.",
        "To perform {subject}, {object_}.",
        "The steps for {subject} are: {object_}.",
        "{subject} is accomplished by {object_}.",
    ],
    "command": [
        "The command {subject} is used to {object_}.",
        "{subject} is a command that {object_}.",
        "Use {subject} to {object_}.",
        "Execute {subject} to {object_}.",
    ],
    "fact": [
        "{subject} {predicate} {object_}.",
        "It is known that {subject} {predicate} {object_}.",
        "The relationship is: {subject} {predicate} {object_}.",
    ],
}

QA_TEMPLATES = {
    "definition": [
        ("What is {subject}?", "{object_}"),
        ("Define {subject}.", "{object_}"),
        ("How would you describe {subject}?", "{object_}"),
        ("Explain what {subject} means.", "{object_}"),
        ("What does {subject} refer to?", "{object_}"),
    ],
    "procedure": [
        ("How do you {subject}?", "{object_}"),
        ("What are the steps to {subject}?", "{object_}"),
        ("Describe the process of {subject}.", "{object_}"),
        ("What is required to {subject}?", "{object_}"),
        ("How is {subject} performed?", "{object_}"),
    ],
    "command": [
        ("What command is used to {subject}?", "{subject}: {object_}"),
        ("How do you {subject} using a command?", "{subject}"),
        ("What does the command {subject} do?", "{object_}"),
        ("Execute {subject} to {object_}.", "{subject}"),
    ],
    "fact": [
        ("What does {subject} {predicate}?", "{object_}"),
        ("How does {subject} {predicate} {object_}?", "{subject} {predicate} {object_}"),
        ("Explain: {subject} {predicate} {object_}.", "{subject} {predicate} {object_}"),
        ("What is the relationship between {subject} and {object_}?", "{predicate}"),
    ],
}


def sanitize_text(text: str, max_length: int = 200) -> str:
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length].rsplit(" ", 1)[0]
    return text


def generate_statement_view(fact: Fact, config: PipelineConfig) -> List[TrainingExample]:
    templates = STATEMENT_TEMPLATES.get(fact.fact_type, STATEMENT_TEMPLATES["fact"])
    examples = []

    for template in templates:
        try:
            text = template.format(
                subject=sanitize_text(fact.subject),
                predicate=fact.predicate,
                object_=sanitize_text(fact.object_),
            )
            if len(text.split()) >= config.min_example_tokens:
                examples.append(
                    TrainingExample(
                        text=text,
                        view_type="statement",
                        source_fact=f"{fact.subject} {fact.predicate} {fact.object_}",
                        difficulty=1,
                        info_score=0.0,
                    )
                )
        except (KeyError, ValueError):
            continue

    return examples


def generate_qa_view(fact: Fact, config: PipelineConfig) -> List[TrainingExample]:
    templates = QA_TEMPLATES.get(fact.fact_type, QA_TEMPLATES["fact"])
    examples = []

    for question_template, answer_template in templates:
        try:
            question = question_template.format(
                subject=sanitize_text(fact.subject),
                predicate=fact.predicate,
                object_=sanitize_text(fact.object_),
            )
            answer = answer_template.format(
                subject=sanitize_text(fact.subject),
                predicate=fact.predicate,
                object_=sanitize_text(fact.object_),
            )

            full_text = f"{question}\n{answer}"
            if len(full_text.split()) >= config.min_example_tokens:
                examples.append(
                    TrainingExample(
                        text=full_text,
                        view_type="qa",
                        source_fact=f"{fact.subject} {fact.predicate} {fact.object_}",
                        difficulty=2,
                        info_score=0.0,
                    )
                )
        except (KeyError, ValueError):
            continue

    return examples


def apply_contrastive_modification(value: str, fact_type: str) -> str:
    if re.match(r"^\d+$", value):
        num = int(value)
        modifications = [num + 1, num - 1, num * 2, num // 2 if num > 1 else num]
        return str(random.choice([m for m in modifications if m >= 0]))

    if value.lower() in ("true", "enabled", "yes", "active"):
        return "False" if random.random() > 0.5 else "disabled"
    if value.lower() in ("false", "disabled", "no", "inactive"):
        return "True" if random.random() > 0.5 else "enabled"

    return value


def generate_contrastive_view(fact: Fact, config: PipelineConfig) -> List[TrainingExample]:
    examples = []

    modified_object = apply_contrastive_modification(fact.object_, fact.fact_type)

    if modified_object == fact.object_:
        return examples

    modified_statement = (
        f"{fact.subject} {fact.predicate} {modified_object}."
    )
    original_statement = f"{fact.subject} {fact.predicate} {fact.object_}."

    full_text = (
        f"True or false: {modified_statement}\n"
        f"Answer: False.\n"
        f"The correct statement is: {original_statement}"
    )

    if len(full_text.split()) >= config.min_example_tokens:
        examples.append(
            TrainingExample(
                text=full_text,
                view_type="contrastive",
                source_fact=f"{fact.subject} {fact.predicate} {fact.object_}",
                difficulty=3,
                info_score=0.0,
            )
        )

    return examples


def generate_views(
    facts: List[Fact], config: PipelineConfig, scored_chunks=None
) -> List[TrainingExample]:
    all_examples = []

    chunk_score_map = {}
    if scored_chunks:
        for chunk in scored_chunks:
            chunk_score_map[chunk.text] = chunk.info_score

    for fact in facts:
        examples_per_fact = []

        if config.generate_statement:
            examples_per_fact.extend(generate_statement_view(fact, config))

        if config.generate_qa:
            examples_per_fact.extend(generate_qa_view(fact, config))

        if config.generate_contrastive:
            examples_per_fact.extend(generate_contrastive_view(fact, config))

        if not examples_per_fact:
            continue

        target_views = max(1, config.views_per_fact // len(facts))

        while len(examples_per_fact) < target_views and examples_per_fact:
            template_example = random.choice(examples_per_fact)
            variation = TrainingExample(
                text=template_example.text,
                view_type=template_example.view_type,
                source_fact=template_example.source_fact,
                difficulty=template_example.difficulty,
                info_score=template_example.info_score,
            )
            examples_per_fact.append(variation)

        fact_score = chunk_score_map.get(fact.source_text, 0.5)
        for example in examples_per_fact:
            example.info_score = fact_score

        all_examples.extend(examples_per_fact)

    return all_examples
