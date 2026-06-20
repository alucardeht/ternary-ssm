import json
import argparse
import random
from pathlib import Path
from typing import Generator, Optional
import sys
import numpy as np
from tokenizers import Tokenizer


def load_gsm8k(path: Path) -> Generator[tuple[str, str], None, None]:
    """Load GSM8K and yield (formatted_text, category)."""
    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                question = obj.get("question", "").strip()
                answer = obj.get("answer", "").strip()

                if not question or not answer:
                    continue

                text = f"<|user|>{question}<|end|><|assistant|>{answer}<|end|>"
                yield text, "math"
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] GSM8K line {line_no}: {e}", file=sys.stderr)
                continue


def load_orca_math(path: Path) -> Generator[tuple[str, str], None, None]:
    """Load Orca-Math and yield (formatted_text, category)."""
    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                question = obj.get("question", "").strip()
                answer = obj.get("answer", "").strip()

                if not question or not answer:
                    continue

                text = f"<|user|>{question}<|end|><|assistant|>{answer}<|end|>"
                yield text, "math"
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] Orca-Math line {line_no}: {e}", file=sys.stderr)
                continue


def load_mathinstruct(path: Path) -> Generator[tuple[str, str], None, None]:
    """Load MathInstruct and yield (formatted_text, category)."""
    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                instruction = obj.get("instruction", "").strip()
                output = obj.get("output", "").strip()

                if not instruction or not output:
                    continue

                text = f"<|user|>{instruction}<|end|><|assistant|>{output}<|end|>"
                yield text, "math"
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] MathInstruct line {line_no}: {e}", file=sys.stderr)
                continue


def load_fineweb_edu(path: Path) -> Generator[tuple[str, str], None, None]:
    """Load FineWeb-Edu and yield (formatted_text, category)."""
    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                text_content = obj.get("text", "").strip()

                if not text_content:
                    continue

                if len(text_content) > 1500:
                    text_content = text_content[:1500]

                text = f"<|user|>Explain the following:<|end|><|assistant|>{text_content}<|end|>"
                yield text, "knowledge"
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] FineWeb-Edu line {line_no}: {e}", file=sys.stderr)
                continue


def load_alpaca_pt(path: Path) -> Generator[tuple[str, str], None, None]:
    """Load Alpaca-PT and yield (formatted_text, category)."""
    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                instruction = obj.get("instruction", "").strip()
                input_text = obj.get("input", "").strip()
                output = obj.get("output", "").strip()

                if not instruction or not output:
                    continue

                if input_text:
                    user_part = f"{instruction}\n{input_text}"
                else:
                    user_part = instruction

                text = f"<|user|>{user_part}<|end|><|assistant|>{output}<|end|>"
                yield text, "pt"
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] Alpaca-PT line {line_no}: {e}", file=sys.stderr)
                continue


def load_opus_en_pt(path: Path) -> Generator[tuple[str, str], None, None]:
    """Load OPUS EN-PT bilingual and yield (formatted_text, category)."""
    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                en = obj.get("en", "").strip()
                pt = obj.get("pt", "").strip()

                if not en or not pt:
                    continue

                text = f"<|user|>Translate to Portuguese: {en}<|end|><|assistant|>{pt}<|end|>"
                yield text, "pt"
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] OPUS EN-PT line {line_no}: {e}", file=sys.stderr)
                continue


def load_existing_chat(path: Path, tokenizer: Optional[Tokenizer] = None) -> Generator[tuple[str, str], None, None]:
    """Load existing chat dataset from bin file and yield (formatted_text, category)."""
    if not path.exists():
        return

    try:
        with open(path, "rb") as f:
            data = f.read()

        if tokenizer:
            token_ids = np.frombuffer(data, dtype=np.uint16)

            batch_size = 512
            i = 0
            while i < len(token_ids):
                batch_ids = token_ids[i:i+batch_size]
                if len(batch_ids) > 0:
                    decoded = tokenizer.decode(batch_ids.tolist())
                    if decoded.strip():
                        yield decoded, "chat"
                i += batch_size
        else:
            text = data.decode("utf-8", errors="ignore")
            for line in text.split("\n\n"):
                if line.strip():
                    yield line, "chat"
    except Exception as e:
        print(f"[WARN] Error loading existing chat: {e}", file=sys.stderr)
        return


def is_valid_example(text: str, tokenizer: Optional[Tokenizer] = None) -> tuple[bool, int]:
    """Check if example passes filters. Returns (is_valid, token_count)."""
    if len(text) < 20:
        return False, 0

    if len(text) > 8000:
        return False, 0

    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False, 0

    if tokenizer:
        try:
            enc = tokenizer.encode(text)
            token_count = len(enc.ids)
            if token_count < 5:
                return False, 0
            if token_count > 2000:
                return False, 0
            return True, token_count
        except Exception:
            return False, 0
    else:
        token_count = len(text) // 4
        if token_count < 5:
            return False, 0
        if token_count > 2000:
            return False, 0
        return True, token_count


def balance_examples(examples: list[tuple[str, str]], target_counts: dict[str, int]) -> list[str]:
    """Balance examples by category and return formatted list."""
    by_category = {}
    for text, category in examples:
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(text)

    balanced = []
    for category in ["math", "knowledge", "pt", "chat", "code"]:
        if category in by_category and category in target_counts:
            target = target_counts[category]
            available = by_category[category]

            if len(available) >= target:
                sampled = random.sample(available, target)
            else:
                sampled = available
                if len(sampled) < target:
                    shortage = target - len(sampled)
                    extra = random.choices(available, k=shortage)
                    sampled.extend(extra)

            balanced.extend(sampled)

    return balanced


def main():
    parser = argparse.ArgumentParser(
        description="Curate and combine datasets for v11 training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/chat_v2/train.bin",
        help="Output .bin file path"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/wikitext103/tokenizer.json",
        help="Path to tokenizer.json (BPE 32K)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print sample examples"
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="Max examples per dataset (for testing)"
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_dir = Path(__file__).parent.parent.parent
    datasets_dir = base_dir / "data" / "datasets"
    chat_dir = base_dir / "data" / "chat_full"

    random.seed(args.seed)

    tokenizer = None
    if args.tokenizer and Path(args.tokenizer).exists():
        tokenizer = Tokenizer.from_file(args.tokenizer)
        chat_special = ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>",
                        "<|tool_call|>", "<|tool_result|>", "<|thinking|>"]
        tokenizer.add_special_tokens(chat_special)
        print(f"[INFO] Loaded BPE tokenizer: vocab={tokenizer.get_vocab_size()}", file=sys.stderr)
    else:
        print(f"[WARN] Tokenizer not found, using character-level estimation", file=sys.stderr)

    all_examples = []
    category_counts = {
        "math": 0,
        "knowledge": 0,
        "pt": 0,
        "chat": 0,
        "code": 0
    }

    print("[INFO] Loading GSM8K...", file=sys.stderr)
    count = 0
    for text, cat in load_gsm8k(datasets_dir / "gsm8k" / "train.jsonl"):
        if args.max_per_dataset and count >= args.max_per_dataset:
            break
        valid, tokens = is_valid_example(text, tokenizer)
        if valid:
            all_examples.append((text, cat))
            category_counts[cat] += 1
            count += 1
    print(f"[INFO] Loaded {count} GSM8K examples", file=sys.stderr)

    print("[INFO] Loading Orca-Math...", file=sys.stderr)
    count = 0
    for text, cat in load_orca_math(datasets_dir / "orca_math" / "train.jsonl"):
        if args.max_per_dataset and count >= args.max_per_dataset:
            break
        valid, tokens = is_valid_example(text, tokenizer)
        if valid:
            all_examples.append((text, cat))
            category_counts[cat] += 1
            count += 1
    print(f"[INFO] Loaded {count} Orca-Math examples", file=sys.stderr)

    print("[INFO] Loading MathInstruct...", file=sys.stderr)
    count = 0
    for text, cat in load_mathinstruct(datasets_dir / "math_instruct" / "train.jsonl"):
        if args.max_per_dataset and count >= args.max_per_dataset:
            break
        valid, tokens = is_valid_example(text, tokenizer)
        if valid:
            all_examples.append((text, cat))
            category_counts[cat] += 1
            count += 1
    print(f"[INFO] Loaded {count} MathInstruct examples", file=sys.stderr)

    print("[INFO] Loading FineWeb-Edu...", file=sys.stderr)
    count = 0
    for text, cat in load_fineweb_edu(datasets_dir / "fineweb_edu" / "sample.jsonl"):
        if args.max_per_dataset and count >= args.max_per_dataset:
            break
        valid, tokens = is_valid_example(text, tokenizer)
        if valid:
            all_examples.append((text, cat))
            category_counts[cat] += 1
            count += 1
    print(f"[INFO] Loaded {count} FineWeb-Edu examples", file=sys.stderr)

    print("[INFO] Loading Alpaca-PT...", file=sys.stderr)
    count = 0
    for text, cat in load_alpaca_pt(datasets_dir / "portuguese" / "alpaca_pt.jsonl"):
        if args.max_per_dataset and count >= args.max_per_dataset:
            break
        valid, tokens = is_valid_example(text, tokenizer)
        if valid:
            all_examples.append((text, cat))
            category_counts[cat] += 1
            count += 1
    print(f"[INFO] Loaded {count} Alpaca-PT examples", file=sys.stderr)

    print("[INFO] Loading OPUS EN-PT...", file=sys.stderr)
    count = 0
    for text, cat in load_opus_en_pt(datasets_dir / "portuguese" / "opus_en_pt.jsonl"):
        if args.max_per_dataset and count >= args.max_per_dataset:
            break
        valid, tokens = is_valid_example(text, tokenizer)
        if valid:
            all_examples.append((text, cat))
            category_counts[cat] += 1
            count += 1
    print(f"[INFO] Loaded {count} OPUS EN-PT examples", file=sys.stderr)

    existing_chat_path = chat_dir / "train.bin"
    print("[INFO] Loading existing chat dataset (10%)...", file=sys.stderr)
    count = 0
    existing_count_target = 39000
    for text, cat in load_existing_chat(existing_chat_path, tokenizer):
        if count >= existing_count_target:
            break
        valid, tokens = is_valid_example(text, tokenizer)
        if valid:
            all_examples.append((text, cat))
            category_counts[cat] += 1
            count += 1
    print(f"[INFO] Loaded {count} existing chat examples", file=sys.stderr)

    print(f"[INFO] Total examples loaded: {len(all_examples)}", file=sys.stderr)
    print(f"[INFO] Category distribution: {category_counts}", file=sys.stderr)

    target_mix = {
        "math": int(len(all_examples) * 0.40),
        "knowledge": int(len(all_examples) * 0.30),
        "pt": int(len(all_examples) * 0.15),
        "chat": int(len(all_examples) * 0.10),
        "code": int(len(all_examples) * 0.05)
    }
    print(f"[INFO] Target mix: {target_mix}", file=sys.stderr)

    print("[INFO] Balancing examples...", file=sys.stderr)
    balanced_examples = balance_examples(all_examples, target_mix)

    print("[INFO] Shuffling...", file=sys.stderr)
    random.shuffle(balanced_examples)

    if args.debug and balanced_examples:
        print("\n=== DEBUG SAMPLES ===\n", file=sys.stderr)
        for i, text in enumerate(balanced_examples[:3]):
            print(f"--- Sample {i+1} ---", file=sys.stderr)
            print(text[:200], file=sys.stderr)
            print()

    print(f"[INFO] Writing {len(balanced_examples)} examples to {output_path}...", file=sys.stderr)

    total_bytes = 0
    total_tokens = 0
    with open(output_path, "wb") as f:
        for i, text in enumerate(balanced_examples):
            if tokenizer:
                enc = tokenizer.encode(text)
                token_ids = np.array(enc.ids, dtype=np.uint16)
                token_ids.tofile(f)
                total_bytes += token_ids.nbytes
                total_tokens += len(enc.ids)
            else:
                data = text.encode("utf-8") + b"\n\n"
                f.write(data)
                total_bytes += len(data)

            if (i + 1) % 10000 == 0:
                print(f"[PROGRESS] {i + 1} examples written", file=sys.stderr)

    total_mb = total_bytes / (1024 * 1024)

    print("\n=== DATASET STATISTICS ===", file=sys.stderr)
    print(f"[CURATE] Total examples: {len(balanced_examples):,}", file=sys.stderr)
    print(f"[CURATE] Total tokens: {total_tokens:,}", file=sys.stderr)
    print(f"[CURATE] Total size: {total_mb:.2f} MB", file=sys.stderr)

    if len(balanced_examples) > 0:
        print(f"[CURATE] Mix:", file=sys.stderr)
        math_count = sum(1 for text, cat in all_examples if cat == "math")
        know_count = sum(1 for text, cat in all_examples if cat == "knowledge")
        pt_count = sum(1 for text, cat in all_examples if cat == "pt")
        chat_count = sum(1 for text, cat in all_examples if cat == "chat")
        code_count = sum(1 for text, cat in all_examples if cat == "code")

        total = math_count + know_count + pt_count + chat_count + code_count
        if total > 0:
            print(f"  - math={math_count/total*100:.1f}%", file=sys.stderr)
            print(f"  - knowledge={know_count/total*100:.1f}%", file=sys.stderr)
            print(f"  - pt={pt_count/total*100:.1f}%", file=sys.stderr)
            print(f"  - chat={chat_count/total*100:.1f}%", file=sys.stderr)
            print(f"  - code={code_count/total*100:.1f}%", file=sys.stderr)

    print(f"[CURATE] Saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
