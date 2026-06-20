from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MEMORY = ROOT / "memory"
BACKUP_ROOT = ROOT / ".work" / "context-heavy" / "docs-raw" / "2026-05-22-hot-memory-split"
TARGET_BYTES = 60 * 1024

TOML_LEDGER_CONFIG = [
    ("research-ledger.toml", MEMORY / "ledgers" / "research", "research"),
    ("measurement-ledger.toml", MEMORY / "ledgers" / "measurement", "measurement"),
    ("proof-object-ledger.toml", MEMORY / "ledgers" / "proof-object", "proof-object"),
]

MARKDOWN_CONFIG = [
    ("baselines.md", MEMORY / "baselines", "baseline"),
    ("experiment-index.md", MEMORY / "experiment-index", "experiment-index"),
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def backup(path: Path) -> Path:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    target = BACKUP_ROOT / path.name
    shutil.copy2(path, target)
    return target


def is_split_stub(text: str) -> bool:
    return "split_shard_dir" in text and "generated_by = \"scripts/maintenance/split_hot_memory_files.py\"" in text


def clean_dir(path: Path, pattern: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob(pattern):
        old.unlink()


def split_toml_ledger(filename: str, shard_dir: Path, prefix: str) -> dict:
    source = MEMORY / filename
    text = read_text(source)
    if is_split_stub(text):
        return {"file": str(source.relative_to(ROOT)), "status": "already_split"}

    backup_path = backup(source)
    lines = text.splitlines(keepends=True)
    meta_lines: list[str] = []
    records: list[str] = []
    in_records = False

    for line in lines:
        if line.startswith("[["):
            in_records = True
        if in_records:
            records.append(line)
        else:
            meta_lines.append(line)

    clean_dir(shard_dir, f"{prefix}-*.toml")
    shards: list[Path] = []
    current: list[str] = []
    current_bytes = 0

    for line in records:
        starts_record = line.startswith("[[")
        line_bytes = len(line.encode("utf-8"))
        if starts_record and current and current_bytes + line_bytes > TARGET_BYTES:
            shard = shard_dir / f"{prefix}-{len(shards) + 1:03d}.toml"
            write_text(shard, "".join(current))
            shards.append(shard)
            current = []
            current_bytes = 0
        current.append(line)
        current_bytes += line_bytes

    if current:
        shard = shard_dir / f"{prefix}-{len(shards) + 1:03d}.toml"
        write_text(shard, "".join(current))
        shards.append(shard)

    rel_shards = [str(path.relative_to(ROOT)).replace("\\", "/") for path in shards]
    stub = "".join(meta_lines).rstrip() + "\n\n"
    stub += "[split]\n"
    stub += 'generated_by = "scripts/maintenance/split_hot_memory_files.py"\n'
    stub += f'generated_at = "{datetime.now().isoformat(timespec="seconds")}"\n'
    stub += f'split_shard_dir = "{str(shard_dir.relative_to(ROOT)).replace("\\", "/")}"\n'
    stub += f'cold_backup = "{str(backup_path.relative_to(ROOT)).replace("\\", "/")}"\n'
    stub += "target_shard_kb = 60\n"
    stub += "shards = [\n"
    for rel in rel_shards:
        stub += f'  "{rel}",\n'
    stub += "]\n"
    write_text(source, stub)

    return {
        "file": str(source.relative_to(ROOT)),
        "status": "split",
        "backup": str(backup_path.relative_to(ROOT)),
        "shards": rel_shards,
    }


def markdown_frontmatter_and_body(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        return "", text
    end = text.find("\n---\n", 4)
    if end == -1:
        return "", text
    return text[: end + 5], text[end + 5 :].lstrip("\n")


def split_markdown(filename: str, shard_dir: Path, prefix: str) -> dict:
    source = MEMORY / filename
    text = read_text(source)
    if "<!-- split-hot-memory-stub -->" in text:
        return {"file": str(source.relative_to(ROOT)), "status": "already_split"}

    backup_path = backup(source)
    frontmatter, body = markdown_frontmatter_and_body(text)
    lines = body.splitlines(keepends=True)
    preface: list[str] = []
    sections: list[str] = []
    current: list[str] = []
    seen_section = False

    for line in lines:
        if line.startswith("## "):
            seen_section = True
            if current:
                sections.append("".join(current))
            current = [line]
        elif seen_section:
            current.append(line)
        else:
            preface.append(line)
    if current:
        sections.append("".join(current))

    clean_dir(shard_dir, f"{prefix}-*.md")
    shards: list[Path] = []
    current_sections: list[str] = []
    current_bytes = 0
    preface_text = "".join(preface).rstrip() + "\n\n"

    for section in sections:
        section_bytes = len(section.encode("utf-8"))
        if current_sections and current_bytes + section_bytes > TARGET_BYTES:
            shard = shard_dir / f"{prefix}-{len(shards) + 1:03d}.md"
            write_text(shard, preface_text + "".join(current_sections))
            shards.append(shard)
            current_sections = []
            current_bytes = 0
        current_sections.append(section)
        current_bytes += section_bytes

    if current_sections:
        shard = shard_dir / f"{prefix}-{len(shards) + 1:03d}.md"
        write_text(shard, preface_text + "".join(current_sections))
        shards.append(shard)

    rel_shards = [str(path.relative_to(ROOT)).replace("\\", "/") for path in shards]
    title = next((line.strip() for line in preface if line.startswith("# ")), f"# {filename}")
    stub_lines = [
        frontmatter.rstrip(),
        "",
        "<!-- split-hot-memory-stub -->",
        "",
        title,
        "",
        "This file is a compact index stub. The full historical detail was split",
        "into shard files so startup, compact recovery, and broad investigation do",
        "not require reading a large hot-path document.",
        "",
        f"- Cold backup: `{str(backup_path.relative_to(ROOT)).replace('\\', '/')}`",
        f"- Shard directory: `{str(shard_dir.relative_to(ROOT)).replace('\\', '/')}`",
        "- Read only the targeted shard or line range needed for the active question.",
        "",
        "## Shards",
        "",
    ]
    for rel in rel_shards:
        stub_lines.append(f"- `{rel}`")
    stub_lines.append("")
    write_text(source, "\n".join(stub_lines))

    return {
        "file": str(source.relative_to(ROOT)),
        "status": "split",
        "backup": str(backup_path.relative_to(ROOT)),
        "shards": rel_shards,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results: list[dict] = []
    for filename, shard_dir, prefix in TOML_LEDGER_CONFIG:
        results.append(split_toml_ledger(filename, shard_dir, prefix))
    for filename, shard_dir, prefix in MARKDOWN_CONFIG:
        results.append(split_markdown(filename, shard_dir, prefix))

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for item in results:
            print(f"{item['status']}: {item['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
