import re
from typing import List
from . import ScoredChunk, Fact
from .config import PipelineConfig

DEFINITION_PATTERNS = [
    re.compile(r"([A-Z][a-zA-Z_\s]+)\s+is\s+(?:a|an)\s+(.{10,200}?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"([A-Z][a-zA-Z_\s]+)\s+refers\s+to\s+(.{10,200}?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"([A-Z][a-zA-Z_\s]+)\s+means?\s+(.{10,200}?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"^([A-Z][a-zA-Z_\s]+):\s+(.{10,200}?)$", re.MULTILINE | re.IGNORECASE),
]

PROCEDURE_PATTERNS = [
    re.compile(r"^(?:\d+\.|Step\s+\d+:)\s+(.{5,100}?)(?:\.|$)", re.MULTILINE | re.IGNORECASE),
    re.compile(r"To\s+(.{5,100}?),\s+(?:do|use|execute)\s+(.{5,150}?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:First|Begin|Start)\s+(.{5,150}?)(?:\.|$)", re.IGNORECASE),
]

COMMAND_PATTERNS = [
    re.compile(r"```(?:bash|shell|sh|python)?\n(.{5,300}?)\n```", re.DOTALL),
    re.compile(r"`([A-Za-z0-9\-\_\.\s]+)`", re.IGNORECASE),
    re.compile(r"(?:Run|Execute|Use|Call)\s+([A-Za-z0-9\-\_\.]+)(?:\s+(.{5,100}?))?(?:\.|$)", re.IGNORECASE),
]

VERB_WHITELIST = {
    "contains",
    "requires",
    "supports",
    "provides",
    "uses",
    "returns",
    "accepts",
    "enables",
    "produces",
    "handles",
    "processes",
    "generates",
    "receives",
    "sends",
    "transfers",
    "stores",
    "retrieves",
    "manages",
    "controls",
    "defines",
    "implements",
    "performs",
}

FACT_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z0-9_]*)\s+(" + "|".join(VERB_WHITELIST) + r")\s+(.{5,100}?)(?:\s|$|\.)",
    re.IGNORECASE,
)


def extract_definitions(chunks: List[ScoredChunk]) -> List[Fact]:
    facts = []
    for chunk in chunks:
        for pattern in DEFINITION_PATTERNS:
            matches = pattern.finditer(chunk.text)
            for match in matches:
                subject = match.group(1).strip()
                definition = match.group(2).strip()

                if len(subject) > 100 or len(definition) > 200:
                    continue

                facts.append(
                    Fact(
                        subject=subject,
                        predicate="is defined as",
                        object_=definition,
                        fact_type="definition",
                        source_text=chunk.text,
                    )
                )

    return facts


def extract_procedures(chunks: List[ScoredChunk]) -> List[Fact]:
    facts = []
    for chunk in chunks:
        for pattern in PROCEDURE_PATTERNS:
            matches = pattern.finditer(chunk.text)
            for match in matches:
                if len(match.groups()) == 1:
                    procedure = match.group(1).strip()
                    if len(procedure) > 200:
                        continue
                    facts.append(
                        Fact(
                            subject=procedure,
                            predicate="is a procedure",
                            object_=procedure,
                            fact_type="procedure",
                            source_text=chunk.text,
                        )
                    )
                else:
                    action = match.group(1).strip()
                    details = match.group(2).strip()
                    if len(action) > 100 or len(details) > 200:
                        continue
                    facts.append(
                        Fact(
                            subject=action,
                            predicate="requires",
                            object_=details,
                            fact_type="procedure",
                            source_text=chunk.text,
                        )
                    )

    return facts


def extract_commands(chunks: List[ScoredChunk]) -> List[Fact]:
    facts = []
    for chunk in chunks:
        for pattern in COMMAND_PATTERNS:
            matches = pattern.finditer(chunk.text)
            for match in matches:
                if len(match.groups()) >= 1:
                    command = match.group(1).strip()
                    if len(command) > 200:
                        continue
                    if not command or command.isspace():
                        continue
                    facts.append(
                        Fact(
                            subject=command.split()[0] if command.split() else command,
                            predicate="is a command",
                            object_=command,
                            fact_type="command",
                            source_text=chunk.text,
                        )
                    )

    return facts


def extract_triples(chunks: List[ScoredChunk]) -> List[Fact]:
    facts = []
    for chunk in chunks:
        matches = FACT_PATTERN.finditer(chunk.text)
        for match in matches:
            subject = match.group(1).strip()
            predicate = match.group(2).strip().lower()
            obj = match.group(3).strip()

            if len(subject) > 100 or len(obj) > 200:
                continue

            facts.append(
                Fact(
                    subject=subject,
                    predicate=predicate,
                    object_=obj,
                    fact_type="fact",
                    source_text=chunk.text,
                )
            )

    return facts


def extract_facts(chunks: List[ScoredChunk], config: PipelineConfig) -> List[Fact]:
    facts = []

    if config.extract_definitions:
        facts.extend(extract_definitions(chunks))

    if config.extract_procedures:
        facts.extend(extract_procedures(chunks))

    if config.extract_code_blocks:
        facts.extend(extract_commands(chunks))

    if config.extract_triples:
        facts.extend(extract_triples(chunks))

    return facts
