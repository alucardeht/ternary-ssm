import numpy as np
import random
from pathlib import Path
import sys


def main():
    base_dir = Path(__file__).parent.parent.parent
    path_full = base_dir / "data" / "chat_full" / "train.bin"
    path_v2 = base_dir / "data" / "chat_v2" / "train.bin"
    output_path = base_dir / "data" / "chat_combined" / "train.bin"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading chat_full/train.bin...", file=sys.stderr)
    with open(path_full, "rb") as f:
        data_full = np.frombuffer(f.read(), dtype=np.uint16)
    print(f"[INFO] chat_full: {len(data_full):,} tokens", file=sys.stderr)

    print("[INFO] Loading chat_v2/train.bin...", file=sys.stderr)
    with open(path_v2, "rb") as f:
        data_v2 = np.frombuffer(f.read(), dtype=np.uint16)
    print(f"[INFO] chat_v2: {len(data_v2):,} tokens", file=sys.stderr)

    combined = np.concatenate([data_full, data_v2])
    total_tokens = len(combined)
    print(f"[INFO] Combined: {total_tokens:,} tokens", file=sys.stderr)

    block_size = 512
    num_blocks = total_tokens // block_size
    blocks = [combined[i*block_size:(i+1)*block_size] for i in range(num_blocks)]
    remainder = combined[num_blocks*block_size:]

    random.seed(42)
    random.shuffle(blocks)

    shuffled = np.concatenate(blocks + ([remainder] if len(remainder) > 0 else []))

    print(f"[INFO] Writing {total_tokens:,} tokens to {output_path}...", file=sys.stderr)
    shuffled.astype(np.uint16).tofile(output_path)

    total_mb = len(shuffled) * 2 / (1024 * 1024)
    print(f"\n=== COMBINE STATISTICS ===", file=sys.stderr)
    print(f"[COMBINE] Total tokens: {total_tokens:,}", file=sys.stderr)
    print(f"[COMBINE] Total size: {total_mb:.2f} MB", file=sys.stderr)
    print(f"[COMBINE] Saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
