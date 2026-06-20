"""Run a repeatable external Ollama Qwen smoke for local inference evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ACTIVATION = "T004_OLLAMA_QWEN_RUNTIME_SMOKE_ACTIVE"


def _find_ollama() -> Path | None:
    found = shutil.which("ollama")
    if found:
        return Path(found)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            return candidate
    return None


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _start_ollama_service(ollama: Path) -> int | None:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        [str(ollama), "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    return process.pid


def _wait_for_service(base_url: str, seconds: int) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            _http_json(f"{base_url}/api/tags", timeout=5)
            return True
        except Exception:
            time.sleep(1)
    return False


def _ollama_run(args: list[str], *, timeout: int) -> dict[str, Any]:
    started = time.time()
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return {
        "returncode": completed.returncode,
        "elapsed_seconds": time.time() - started,
        "stdout_tail": stdout.splitlines()[-20:],
        "stderr_tail": stderr.splitlines()[-20:],
    }


def run_ollama_smoke(
    *,
    model: str,
    prompt: str,
    num_predict: int,
    out_dir: Path,
    pull: bool,
    start_service: bool,
    base_url: str,
    pull_timeout_seconds: int,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "probe": "t004_ollama_qwen_smoke",
        "activation": ACTIVATION,
        "timestamp": started.isoformat(timespec="seconds"),
        "model": model,
        "prompt": prompt,
        "num_predict": num_predict,
        "base_url": base_url,
        "pull_requested": pull,
        "service_start_requested": start_service,
        "ok": False,
        "status": "not_started",
    }

    ollama = _find_ollama()
    result["ollama_path"] = str(ollama) if ollama is not None else None
    if ollama is None:
        result.update({"status": "ollama_cli_missing", "blocker": "ollama_cli_missing"})
        return result

    service_ready = _wait_for_service(base_url, 3)
    if not service_ready and start_service:
        result["started_service_pid"] = _start_ollama_service(ollama)
        service_ready = _wait_for_service(base_url, 30)
    result["service_ready"] = service_ready
    if not service_ready:
        result.update({"status": "ollama_service_unreachable", "blocker": "ollama_service_unreachable"})
        return result

    if pull:
        pull_result = _ollama_run([str(ollama), "pull", model], timeout=pull_timeout_seconds)
        result["pull"] = pull_result
        if pull_result["returncode"] != 0:
            result.update({"status": "pull_failed", "blocker": "ollama_pull_failed"})
            return result

    list_result = _ollama_run([str(ollama), "list"], timeout=30)
    result["list"] = list_result

    generation_payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0},
    }
    try:
        generation_started = time.time()
        generation = _http_json(f"{base_url}/api/generate", generation_payload, timeout=600)
        wall_seconds = time.time() - generation_started
    except urllib.error.HTTPError as exc:
        result.update({"status": "generate_http_failed", "blocker": f"http_{exc.code}", "error": str(exc)})
        return result
    except Exception as exc:
        result.update({"status": "generate_failed", "blocker": type(exc).__name__, "error": str(exc)})
        return result

    eval_count = int(generation.get("eval_count") or 0)
    eval_duration_ns = int(generation.get("eval_duration") or 0)
    prompt_eval_count = int(generation.get("prompt_eval_count") or 0)
    prompt_eval_duration_ns = int(generation.get("prompt_eval_duration") or 0)
    result.update(
        {
            "ok": True,
            "status": "generation_done",
            "response": generation.get("response"),
            "done_reason": generation.get("done_reason"),
            "eval_count": eval_count,
            "eval_duration_seconds": eval_duration_ns / 1e9,
            "tokens_per_second": eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else None,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration_seconds": prompt_eval_duration_ns / 1e9,
            "prompt_tokens_per_second": prompt_eval_count / (prompt_eval_duration_ns / 1e9)
            if prompt_eval_duration_ns > 0
            else None,
            "load_duration_seconds": int(generation.get("load_duration") or 0) / 1e9,
            "total_duration_seconds": int(generation.get("total_duration") or 0) / 1e9,
            "wall_seconds": wall_seconds,
        }
    )
    return result


def write_report(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# T004 Ollama Qwen Runtime Smoke",
        "",
        f"Activation: `{result['activation']}`",
        f"Status: `{result['status']}`",
        f"OK: `{result['ok']}`",
        f"Model: `{result['model']}`",
        f"Ollama path: `{result['ollama_path']}`",
        f"Service ready: `{result.get('service_ready')}`",
    ]
    if result.get("started_service_pid"):
        lines.append(f"Started service PID: `{result['started_service_pid']}`")
    if result["ok"]:
        lines.extend(
            [
                "",
                "## Generation",
                "",
                f"- eval tokens: `{result['eval_count']}`",
                f"- eval seconds: `{result['eval_duration_seconds']}`",
                f"- tok/s: `{result['tokens_per_second']}`",
                f"- prompt tokens: `{result['prompt_eval_count']}`",
                f"- prompt tok/s: `{result['prompt_tokens_per_second']}`",
                f"- load seconds: `{result['load_duration_seconds']}`",
                f"- total seconds: `{result['total_duration_seconds']}`",
                f"- response: `{result['response']}`",
            ]
        )
    else:
        lines.extend(["", "## Blocker", "", f"- `{result.get('blocker')}`", f"- error: `{result.get('error')}`"])
    lines.extend(
        [
            "",
            "This is an external runtime smoke. It does not exercise the project compressed residency stack.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:32b")
    parser.add_argument("--prompt", default="Reply with exactly five words: local Qwen smoke test")
    parser.add_argument("--num-predict", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--start-service", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--pull-timeout-seconds", type=int, default=7200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_ollama_smoke(
        model=args.model,
        prompt=args.prompt,
        num_predict=args.num_predict,
        out_dir=args.out_dir,
        pull=args.pull,
        start_service=args.start_service,
        base_url=args.base_url,
        pull_timeout_seconds=args.pull_timeout_seconds,
    )
    write_report(result, args.out_dir)
    print(ACTIVATION)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "ok": result["ok"],
                "status": result["status"],
                "tokens_per_second": result.get("tokens_per_second"),
                "blocker": result.get("blocker"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
