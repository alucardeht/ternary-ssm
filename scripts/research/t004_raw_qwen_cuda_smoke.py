"""Attempt a raw GPU-only Qwen3-32B load and record the concrete outcome."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ACTIVATION = "T004_RAW_QWEN_CUDA_LOAD_ATTEMPT"


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _cuda_snapshot() -> dict[str, object]:
    data: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        data.update(
            {
                "device": torch.cuda.get_device_name(0),
                "free_gib": free / 1024**3,
                "total_gib": total / 1024**3,
                "max_allocated_gib": torch.cuda.max_memory_allocated(0) / 1024**3,
            }
        )
    return data


def run_raw_cuda_smoke(model_dir: Path, *, max_new_tokens: int, prompt: str) -> dict[str, object]:
    started = datetime.now(UTC)
    result: dict[str, object] = {
        "probe": "t004_raw_qwen_cuda_smoke",
        "activation": ACTIVATION,
        "timestamp": started.isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "cuda_before": _cuda_snapshot(),
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        generate_started = datetime.now(UTC)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        generate_ended = datetime.now(UTC)
        generated_tokens = int(output.shape[-1] - inputs["input_ids"].shape[-1])
        seconds = (generate_ended - generate_started).total_seconds()
        result.update(
            {
                "ok": True,
                "status": "generation_done",
                "generated_tokens": generated_tokens,
                "seconds": seconds,
                "tokens_per_second": generated_tokens / seconds if seconds > 0 else None,
                "text": tokenizer.decode(output[0], skip_special_tokens=True),
            }
        )
    except Exception as exc:  # This is the expected path for raw BF16 on 12 GB.
        result.update(
            {
                "ok": False,
                "status": "load_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
            }
        )
    finally:
        result["cuda_after"] = _cuda_snapshot()
        result["elapsed_seconds"] = (datetime.now(UTC) - started).total_seconds()
    return result


def write_report(result: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T004 Raw Qwen CUDA Smoke",
        "",
        f"Activation: `{result['activation']}`",
        f"Status: `{result['status']}`",
        f"OK: `{result['ok']}`",
        f"Model dir: `{result['model_dir']}`",
        "",
        "## CUDA Before",
        "",
    ]
    for key, value in dict(result["cuda_before"]).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## CUDA After", ""])
    for key, value in dict(result["cuda_after"]).items():
        lines.append(f"- `{key}`: `{value}`")
    if not result["ok"]:
        lines.extend(
            [
                "",
                "## Error",
                "",
                f"- type: `{result['error_type']}`",
                f"- message: `{result['error']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Generation",
                "",
                f"- generated tokens: `{result['generated_tokens']}`",
                f"- seconds: `{result['seconds']}`",
                f"- tok/s: `{result['tokens_per_second']}`",
            ]
        )
    lines.extend(
        [
            "",
            "This smoke uses the official raw model weights with GPU-only placement. It does not exercise the T001-T003 compressed residency stack.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Write one short sentence about local inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_raw_cuda_smoke(args.model_dir, max_new_tokens=args.max_new_tokens, prompt=args.prompt)
    write_report(result, args.out_dir)
    print(ACTIVATION)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "ok": result["ok"],
                "status": result["status"],
                "error_type": result.get("error_type"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
