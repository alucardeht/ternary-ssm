import re
from typing import List
from . import Chunk
from .config import PipelineConfig

MARKDOWN_HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")
LIST_ITEM_PATTERN = re.compile(r"^\s*(?:\d+\.|[-*+])\s+", re.MULTILINE)


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def detect_chunk_type(text: str, prev_text: str = "") -> str:
    if CODE_BLOCK_PATTERN.search(text):
        return "code"
    if LIST_ITEM_PATTERN.search(text):
        return "list"
    if re.match(r"^#{1,6}\s+", text):
        return "header"
    if ":" in text and len(text.split(":")[0]) < 50:
        return "definition"
    return "paragraph"


def extract_current_section(text_before: str) -> str:
    matches = list(MARKDOWN_HEADER_PATTERN.finditer(text_before))
    if not matches:
        return "untitled"
    last_match = matches[-1]
    return last_match.group(2).strip()


def split_by_paragraph(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def split_by_section(text: str) -> List[str]:
    sections = re.split(MARKDOWN_HEADER_PATTERN, text.strip())
    chunks = []
    current_section = ""
    for i, part in enumerate(sections):
        if i % 3 == 0:
            if part.strip() and current_section:
                chunks.append(current_section)
                current_section = ""
            if part.strip():
                current_section = part.strip()
        elif i % 3 == 2:
            if current_section:
                current_section += f"\n{part.strip()}"
    if current_section.strip():
        chunks.append(current_section)
    return chunks


def chunk_text_by_tokens(
    text: str, min_tokens: int, max_tokens: int
) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        word_tokens = int(len(word) * 1.3 / 4.7) + 1
        if current_tokens + word_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_tokens += word_tokens

    if current_chunk and current_tokens >= min_tokens:
        chunks.append(" ".join(current_chunk))

    return chunks


def parse_document(file_path: str, config: PipelineConfig) -> List[Chunk]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if config.chunk_strategy == "section":
        raw_chunks = split_by_section(content)
    else:
        raw_chunks = split_by_paragraph(content)

    chunks = []
    position = 0

    for raw_chunk in raw_chunks:
        if not raw_chunk.strip():
            continue

        token_count = estimate_tokens(raw_chunk)

        if token_count > config.max_chunk_tokens:
            sub_chunks = chunk_text_by_tokens(
                raw_chunk, config.min_chunk_tokens, config.max_chunk_tokens
            )
            for sub_chunk in sub_chunks:
                if estimate_tokens(sub_chunk) >= config.min_chunk_tokens:
                    section = extract_current_section(content[: content.find(sub_chunk)])
                    chunk_type = detect_chunk_type(sub_chunk, content[:content.find(sub_chunk)])
                    chunks.append(
                        Chunk(
                            text=sub_chunk,
                            section=section,
                            chunk_type=chunk_type,
                            position=position,
                        )
                    )
                    position += 1
        elif token_count >= config.min_chunk_tokens:
            section = extract_current_section(content[: content.find(raw_chunk)])
            chunk_type = detect_chunk_type(raw_chunk, content[: content.find(raw_chunk)])
            chunks.append(
                Chunk(
                    text=raw_chunk,
                    section=section,
                    chunk_type=chunk_type,
                    position=position,
                )
            )
            position += 1

    return chunks
