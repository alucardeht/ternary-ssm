from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constructive_imagination import run_all_trainable_probes


def main() -> int:
    parser = argparse.ArgumentParser(description="Run R439B trainable toy probes.")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = run_all_trainable_probes(seed=args.seed)
    payload = {
        "seed": args.seed,
        "results": [result.as_dict() for result in results],
        "passed": all(result.passed for result in results),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
