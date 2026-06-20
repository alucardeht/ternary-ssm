import json
import argparse
import random
from pathlib import Path
from typing import Generator
import re
import sys
import numpy as np
from tokenizers import Tokenizer


def load_everyday_conversations(path: Path) -> Generator[str, None, None]:
    """Load everyday_conversations.jsonl and convert to formatted text."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                messages = obj.get("messages", [])

                if not messages:
                    continue

                text = ""
                for msg in messages:
                    role = msg.get("role", "").strip()
                    content = msg.get("content", "").strip()

                    if not content:
                        continue

                    if role == "user":
                        text += f"<|user|>{content}<|end|>"
                    elif role == "assistant":
                        text += f"<|assistant|>{content}<|end|>"
                    elif role == "system":
                        text += f"<|system|>{content}<|end|>"

                if text:
                    yield text

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] everyday_conversations.jsonl line {line_no}: {e}", file=sys.stderr)
                continue


def load_smol_smoltalk(path: Path) -> Generator[str, None, None]:
    """Load smol_smoltalk.jsonl and convert to formatted text."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                messages = obj.get("messages", [])

                if not messages:
                    continue

                text = ""
                for msg in messages:
                    role = msg.get("role", "").strip()
                    content = msg.get("content", "").strip()

                    if not content:
                        continue

                    if role == "user":
                        text += f"<|user|>{content}<|end|>"
                    elif role == "assistant":
                        text += f"<|assistant|>{content}<|end|>"
                    elif role == "system":
                        text += f"<|system|>{content}<|end|>"

                if text:
                    yield text

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] smol_smoltalk.jsonl line {line_no}: {e}", file=sys.stderr)
                continue


def parse_glaive_chat(chat_str: str) -> list[dict]:
    """Parse glaive chat string into list of {role, content}."""
    turns = []

    current_section = None
    current_content = []

    lines = chat_str.split("\n")

    for line in lines:
        line = line.rstrip()

        if line.startswith("USER:"):
            if current_section and current_content:
                turns.append({
                    "role": current_section,
                    "content": "\n".join(current_content).strip()
                })
            current_section = "user"
            current_content = [line[5:].strip()]

        elif line.startswith("ASSISTANT:"):
            if current_section and current_content:
                turns.append({
                    "role": current_section,
                    "content": "\n".join(current_content).strip()
                })
            current_section = "assistant"
            current_content = [line[10:].strip()]

        elif line.startswith("FUNCTION RESPONSE:"):
            if current_section and current_content:
                turns.append({
                    "role": current_section,
                    "content": "\n".join(current_content).strip()
                })
            current_section = "tool_result"
            current_content = [line[18:].strip()]

        elif current_section:
            if line.strip():
                current_content.append(line)

    if current_section and current_content:
        turns.append({
            "role": current_section,
            "content": "\n".join(current_content).strip()
        })

    return turns


def extract_function_call(text: str) -> dict | None:
    """Extract <functioncall> {...} from text."""
    pattern = r"<functioncall>\s*(\{.*?\})\s*"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def _parse_thinking_value(value: str) -> str:
    """
    Extract <|begin_of_thought|>...<|end_of_thought|> → <|thinking|>...<|end|>
    and <|begin_of_solution|>...<|end_of_solution|> → <|assistant|>...<|end|>
    Used by both load_openthoughts and load_bespoke_stratos.
    """
    thought_pattern = r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>"
    solution_pattern = r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>"

    thought_match = re.search(thought_pattern, value, re.DOTALL)
    solution_match = re.search(solution_pattern, value, re.DOTALL)

    out = ""
    if thought_match:
        out += f"<|thinking|>{thought_match.group(1).strip()}<|end|>"
    if solution_match:
        out += f"<|assistant|>{solution_match.group(1).strip()}<|end|>"
    else:
        remainder = re.sub(thought_pattern, "", value, flags=re.DOTALL).strip()
        if remainder:
            out += f"<|assistant|>{remainder}<|end|>"

    return out


def load_numinamath(path: Path) -> Generator[str, None, None]:
    """Load numinamath_50k.jsonl and convert to formatted text."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                messages = obj.get("messages", [])

                if not messages:
                    continue

                text = ""
                for msg in messages:
                    role = msg.get("role", "").strip()
                    content = msg.get("content", "").strip()

                    if not content:
                        continue

                    if role == "user":
                        text += f"<|user|>{content}<|end|>"
                    elif role == "assistant":
                        text += f"<|assistant|>{content}<|end|>"

                if text:
                    yield text

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] numinamath_50k.jsonl line {line_no}: {e}", file=sys.stderr)
                continue


def load_metamath(path: Path) -> Generator[str, None, None]:
    """Load metamath_30k.jsonl and convert to formatted text."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                query = obj.get("query", "").strip()
                response = obj.get("response", "").strip()

                if not query or not response:
                    continue

                text = f"<|user|>{query}<|end|><|assistant|>{response}<|end|>"
                yield text

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] metamath_30k.jsonl line {line_no}: {e}", file=sys.stderr)
                continue


def load_openthoughts(path: Path) -> Generator[str, None, None]:
    """Load openthoughts_50k.jsonl and convert to formatted text with thinking traces."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                system = obj.get("system", "").strip()
                conversations = obj.get("conversations", [])

                if not conversations:
                    continue

                text = ""
                if system:
                    text += f"<|system|>{system}<|end|>"

                for turn in conversations:
                    role = turn.get("from", "").strip()
                    value = turn.get("value", "").strip()

                    if not value:
                        continue

                    if role == "user":
                        text += f"<|user|>{value}<|end|>"
                    elif role == "assistant":
                        text += _parse_thinking_value(value)

                if text:
                    yield text

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] openthoughts_50k.jsonl line {line_no}: {e}", file=sys.stderr)
                continue


def load_code_selfoss(path: Path) -> Generator[str, None, None]:
    """Load code_selfoss_50k.jsonl and convert to formatted text."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                instruction = obj.get("instruction", "").strip()
                response = obj.get("response", "").strip()

                if not instruction or not response:
                    continue

                text = f"<|user|>{instruction}<|end|><|assistant|>{response}<|end|>"
                yield text

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] code_selfoss_50k.jsonl line {line_no}: {e}", file=sys.stderr)
                continue


def load_openhermes(path: Path) -> Generator[str, None, None]:
    """Load openhermes_50k.jsonl and convert to formatted text."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                system_prompt = obj.get("system_prompt", "")
                if system_prompt:
                    system_prompt = system_prompt.strip()

                conversations = obj.get("conversations", [])

                if not conversations:
                    continue

                text = ""
                if system_prompt:
                    text += f"<|system|>{system_prompt}<|end|>"

                for turn in conversations:
                    role = turn.get("from", "").strip()
                    content = turn.get("value", "").strip()

                    if not content:
                        continue

                    if role == "human":
                        text += f"<|user|>{content}<|end|>"
                    elif role == "gpt":
                        text += f"<|assistant|>{content}<|end|>"

                if text:
                    yield text

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] openhermes_50k.jsonl line {line_no}: {e}", file=sys.stderr)
                continue


def load_bespoke_stratos(path: Path) -> Generator[str, None, None]:
    """Load bespoke_stratos_17k.jsonl and convert to formatted text with thinking traces."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                system = obj.get("system", "").strip()
                conversations = obj.get("conversations", [])

                if not conversations:
                    continue

                text = ""
                if system:
                    text += f"<|system|>{system}<|end|>"

                for turn in conversations:
                    role = turn.get("from", "").strip()
                    value = turn.get("value", "").strip()

                    if not value:
                        continue

                    if role == "user":
                        text += f"<|user|>{value}<|end|>"
                    elif role == "assistant":
                        text += _parse_thinking_value(value)

                if text:
                    yield text

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] bespoke_stratos_17k.jsonl line {line_no}: {e}", file=sys.stderr)
                continue


def load_systemchat(path: Path) -> Generator[str, None, None]:
    """Load systemchat_20k.jsonl and convert to formatted text."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                messages = obj.get("messages", [])

                if not messages:
                    continue

                text = ""
                for msg in messages:
                    role = msg.get("role", "").strip()
                    content = msg.get("content", "").strip()

                    if not content:
                        continue

                    if role == "system":
                        text += f"<|system|>{content}<|end|>"
                    elif role == "user":
                        text += f"<|user|>{content}<|end|>"
                    elif role == "assistant":
                        text += f"<|assistant|>{content}<|end|>"

                if text:
                    yield text

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] systemchat_20k.jsonl line {line_no}: {e}", file=sys.stderr)
                continue


def load_glaive_function_calling(path: Path) -> Generator[str, None, None]:
    """Load glaive_function_calling.jsonl and convert to formatted text."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                system = obj.get("system", "").strip()
                chat = obj.get("chat", "").strip()

                if not chat:
                    continue

                turns = parse_glaive_chat(chat)

                if not turns:
                    continue

                text = ""

                if system:
                    system_clean = system.replace("<|endoftext|>", "").strip()
                    text += f"<|system|>{system_clean}<|end|>"

                for turn in turns:
                    role = turn.get("role", "").strip()
                    content = turn.get("content", "").strip()

                    if not content:
                        continue

                    content = content.replace("<|endoftext|>", "").strip()

                    if role == "user":
                        text += f"<|user|>{content}<|end|>"

                    elif role == "assistant":
                        func_call = extract_function_call(content)
                        if func_call:
                            text += f"<|tool_call|>{json.dumps(func_call)}<|end|>"
                            content_without_call = re.sub(
                                r"<functioncall>\s*\{.*?\}\s*<\|endoftext\|\>?\s*",
                                "",
                                content,
                                flags=re.DOTALL
                            ).strip()
                            if content_without_call:
                                text += f"<|assistant|>{content_without_call}<|end|>"
                        else:
                            text += f"<|assistant|>{content}<|end|>"

                    elif role == "tool_result":
                        text += f"<|tool_result|>{content}<|end|>"

                if text:
                    yield text

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARN] glaive_function_calling.jsonl line {line_no}: {e}", file=sys.stderr)
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Format chat datasets for byte-level training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/chat_100m/train.bin",
        help="Output .bin file path"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print sample formatted examples"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json (BPE 32K). If None, byte-level."
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_dir = Path(__file__).parent.parent.parent
    everyday_path = base_dir / "data" / "downloads" / "everyday_conversations.jsonl"
    smoltak_path = base_dir / "data" / "downloads" / "smol_smoltalk.jsonl"
    glaive_path = base_dir / "data" / "downloads" / "glaive_function_calling.jsonl"

    numinamath_path = base_dir / "data" / "downloads" / "numinamath_50k.jsonl"
    metamath_path = base_dir / "data" / "downloads" / "metamath_30k.jsonl"
    openthoughts_path = base_dir / "data" / "downloads" / "openthoughts_50k.jsonl"
    code_selfoss_path = base_dir / "data" / "downloads" / "code_selfoss_50k.jsonl"
    openhermes_path = base_dir / "data" / "downloads" / "openhermes_50k.jsonl"
    bespoke_stratos_path = base_dir / "data" / "downloads" / "bespoke_stratos_17k.jsonl"
    systemchat_path = base_dir / "data" / "downloads" / "systemchat_20k.jsonl"

    examples = []
    sample_count = {
        "everyday": 0, "smol": 0, "glaive": 0,
        "numinamath": 0, "metamath": 0, "openthoughts": 0,
        "code_selfoss": 0, "openhermes": 0, "bespoke_stratos": 0, "systemchat": 0
    }
    debug_samples = {
        "everyday": [], "smol": [], "glaive": [],
        "numinamath": [], "metamath": [], "openthoughts": [],
        "code_selfoss": [], "openhermes": [], "bespoke_stratos": [], "systemchat": []
    }

    max_ex = args.max_examples

    print("[INFO] Loading everyday_conversations.jsonl...", file=sys.stderr)
    for text in load_everyday_conversations(everyday_path):
        if max_ex and len(examples) >= max_ex:
            break
        examples.append(text)
        sample_count["everyday"] += 1
        if args.debug and len(debug_samples["everyday"]) < 3:
            debug_samples["everyday"].append(text)

    print(
        f"[INFO] Loaded {sample_count['everyday']} everyday examples",
        file=sys.stderr
    )

    print("[INFO] Loading smol_smoltalk.jsonl...", file=sys.stderr)
    for text in load_smol_smoltalk(smoltak_path):
        if max_ex and len(examples) >= max_ex:
            break
        examples.append(text)
        sample_count["smol"] += 1
        if args.debug and len(debug_samples["smol"]) < 3:
            debug_samples["smol"].append(text)

    print(f"[INFO] Loaded {sample_count['smol']} smol examples", file=sys.stderr)

    print("[INFO] Loading glaive_function_calling.jsonl...", file=sys.stderr)
    for text in load_glaive_function_calling(glaive_path):
        if max_ex and len(examples) >= max_ex:
            break
        examples.append(text)
        sample_count["glaive"] += 1
        if args.debug and len(debug_samples["glaive"]) < 3:
            debug_samples["glaive"].append(text)

    print(f"[INFO] Loaded {sample_count['glaive']} glaive examples", file=sys.stderr)

    if numinamath_path.exists():
        print("[INFO] Loading numinamath_50k.jsonl...", file=sys.stderr)
        for text in load_numinamath(numinamath_path):
            if max_ex and len(examples) >= max_ex:
                break
            examples.append(text)
            sample_count["numinamath"] += 1
            if args.debug and len(debug_samples["numinamath"]) < 3:
                debug_samples["numinamath"].append(text)
        print(f"[INFO] Loaded {sample_count['numinamath']} numinamath examples", file=sys.stderr)
    else:
        print(f"[WARN] numinamath_50k.jsonl not found at {numinamath_path}", file=sys.stderr)

    if metamath_path.exists():
        print("[INFO] Loading metamath_30k.jsonl...", file=sys.stderr)
        for text in load_metamath(metamath_path):
            if max_ex and len(examples) >= max_ex:
                break
            examples.append(text)
            sample_count["metamath"] += 1
            if args.debug and len(debug_samples["metamath"]) < 3:
                debug_samples["metamath"].append(text)
        print(f"[INFO] Loaded {sample_count['metamath']} metamath examples", file=sys.stderr)
    else:
        print(f"[WARN] metamath_30k.jsonl not found at {metamath_path}", file=sys.stderr)

    if openthoughts_path.exists():
        print("[INFO] Loading openthoughts_50k.jsonl...", file=sys.stderr)
        for text in load_openthoughts(openthoughts_path):
            if max_ex and len(examples) >= max_ex:
                break
            examples.append(text)
            sample_count["openthoughts"] += 1
            if args.debug and len(debug_samples["openthoughts"]) < 3:
                debug_samples["openthoughts"].append(text)
        print(f"[INFO] Loaded {sample_count['openthoughts']} openthoughts examples", file=sys.stderr)
    else:
        print(f"[WARN] openthoughts_50k.jsonl not found at {openthoughts_path}", file=sys.stderr)

    if code_selfoss_path.exists():
        print("[INFO] Loading code_selfoss_50k.jsonl...", file=sys.stderr)
        for text in load_code_selfoss(code_selfoss_path):
            if max_ex and len(examples) >= max_ex:
                break
            examples.append(text)
            sample_count["code_selfoss"] += 1
            if args.debug and len(debug_samples["code_selfoss"]) < 3:
                debug_samples["code_selfoss"].append(text)
        print(f"[INFO] Loaded {sample_count['code_selfoss']} code_selfoss examples", file=sys.stderr)
    else:
        print(f"[WARN] code_selfoss_50k.jsonl not found at {code_selfoss_path}", file=sys.stderr)

    if openhermes_path.exists():
        print("[INFO] Loading openhermes_50k.jsonl...", file=sys.stderr)
        for text in load_openhermes(openhermes_path):
            if max_ex and len(examples) >= max_ex:
                break
            examples.append(text)
            sample_count["openhermes"] += 1
            if args.debug and len(debug_samples["openhermes"]) < 3:
                debug_samples["openhermes"].append(text)
        print(f"[INFO] Loaded {sample_count['openhermes']} openhermes examples", file=sys.stderr)
    else:
        print(f"[WARN] openhermes_50k.jsonl not found at {openhermes_path}", file=sys.stderr)

    if bespoke_stratos_path.exists():
        print("[INFO] Loading bespoke_stratos_17k.jsonl...", file=sys.stderr)
        for text in load_bespoke_stratos(bespoke_stratos_path):
            if max_ex and len(examples) >= max_ex:
                break
            examples.append(text)
            sample_count["bespoke_stratos"] += 1
            if args.debug and len(debug_samples["bespoke_stratos"]) < 3:
                debug_samples["bespoke_stratos"].append(text)
        print(f"[INFO] Loaded {sample_count['bespoke_stratos']} bespoke_stratos examples", file=sys.stderr)
    else:
        print(f"[WARN] bespoke_stratos_17k.jsonl not found at {bespoke_stratos_path}", file=sys.stderr)

    if systemchat_path.exists():
        print("[INFO] Loading systemchat_20k.jsonl...", file=sys.stderr)
        for text in load_systemchat(systemchat_path):
            if max_ex and len(examples) >= max_ex:
                break
            examples.append(text)
            sample_count["systemchat"] += 1
            if args.debug and len(debug_samples["systemchat"]) < 3:
                debug_samples["systemchat"].append(text)
        print(f"[INFO] Loaded {sample_count['systemchat']} systemchat examples", file=sys.stderr)
    else:
        print(f"[WARN] systemchat_20k.jsonl not found at {systemchat_path}", file=sys.stderr)

    if args.debug:
        print("\n=== DEBUG SAMPLES ===\n", file=sys.stderr)

        if debug_samples["everyday"]:
            print("--- Everyday Conversation Sample ---", file=sys.stderr)
            print(debug_samples["everyday"][0][:300], file=sys.stderr)
            print()

        if debug_samples["smol"]:
            print("--- Smol Chat Sample ---", file=sys.stderr)
            print(debug_samples["smol"][0][:300], file=sys.stderr)
            print()

        if debug_samples["glaive"]:
            print("--- Tool Use Sample ---", file=sys.stderr)
            print(debug_samples["glaive"][0][:300], file=sys.stderr)
            print()

        if debug_samples["numinamath"]:
            print("--- Numinamath Sample ---", file=sys.stderr)
            print(debug_samples["numinamath"][0][:300], file=sys.stderr)
            print()

        if debug_samples["metamath"]:
            print("--- Metamath Sample ---", file=sys.stderr)
            print(debug_samples["metamath"][0][:300], file=sys.stderr)
            print()

        if debug_samples["openthoughts"]:
            print("--- Openthoughts Sample ---", file=sys.stderr)
            print(debug_samples["openthoughts"][0][:300], file=sys.stderr)
            print()

        if debug_samples["code_selfoss"]:
            print("--- Code Selfoss Sample ---", file=sys.stderr)
            print(debug_samples["code_selfoss"][0][:300], file=sys.stderr)
            print()

        if debug_samples["openhermes"]:
            print("--- Openhermes Sample ---", file=sys.stderr)
            print(debug_samples["openhermes"][0][:300], file=sys.stderr)
            print()

        if debug_samples["bespoke_stratos"]:
            print("--- Bespoke Stratos Sample ---", file=sys.stderr)
            print(debug_samples["bespoke_stratos"][0][:300], file=sys.stderr)
            print()

        if debug_samples["systemchat"]:
            print("--- Systemchat Sample ---", file=sys.stderr)
            print(debug_samples["systemchat"][0][:300], file=sys.stderr)
            print()

    print("[INFO] Shuffling examples...", file=sys.stderr)
    random.seed(args.seed)
    random.shuffle(examples)

    tokenizer = None
    if args.tokenizer:
        tokenizer = Tokenizer.from_file(args.tokenizer)
        chat_special = ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>",
                        "<|tool_call|>", "<|tool_result|>", "<|thinking|>"]
        tokenizer.add_special_tokens(chat_special)
        print(f"[INFO] Loaded BPE tokenizer: vocab={tokenizer.get_vocab_size()}", file=sys.stderr)

    print(f"[INFO] Writing {len(examples)} examples to {output_path}...", file=sys.stderr)

    total_bytes = 0
    total_tokens = 0
    with open(output_path, "wb") as f:
        for i, text in enumerate(examples):
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

    if tokenizer:
        total_tokens_estimate = total_tokens
        encoding_unit = "tokens (BPE)"
    else:
        total_tokens_estimate = total_bytes // 4
        encoding_unit = "estimated tokens (1 token ≈ 4 bytes)"

    total_mb = total_bytes / (1024 * 1024)

    print("\n=== DATASET STATISTICS ===", file=sys.stderr)
    print(f"Total examples: {len(examples)}", file=sys.stderr)
    print(f"  - Everyday: {sample_count['everyday']}", file=sys.stderr)
    print(f"  - Smol: {sample_count['smol']}", file=sys.stderr)
    print(f"  - Glaive: {sample_count['glaive']}", file=sys.stderr)
    print(f"  - Numinamath: {sample_count['numinamath']}", file=sys.stderr)
    print(f"  - Metamath: {sample_count['metamath']}", file=sys.stderr)
    print(f"  - Openthoughts: {sample_count['openthoughts']}", file=sys.stderr)
    print(f"  - Code Selfoss: {sample_count['code_selfoss']}", file=sys.stderr)
    print(f"  - Openhermes: {sample_count['openhermes']}", file=sys.stderr)
    print(f"  - Bespoke Stratos: {sample_count['bespoke_stratos']}", file=sys.stderr)
    print(f"  - Systemchat: {sample_count['systemchat']}", file=sys.stderr)
    print(f"Total size: {total_mb:.2f} MB", file=sys.stderr)
    print(f"{encoding_unit}: {total_tokens_estimate:,}", file=sys.stderr)
    print(f"Output file: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
