"""Part 5.2 — Model Quantization & Performance Profiling.

Benchmarks three Ollama-served Gemma-3 variants across quantization levels:
  gemma3:270m      Q8_0   268 M params
  gemma3:1b        Q4_K_M 1 B params
  gemma3:1b-it-qat Q4_0   1 B params (quantization-aware training)

Metrics collected per model:
  - Tokens per second (TPS) — from Ollama's eval_duration / eval_count
  - Peak VRAM (MiB)         — nvidia-smi polled every 100 ms during generation
  - Peak RAM  (MiB)         — psutil RSS polled every 100 ms during generation
  - Output quality          — stored for manual inspection

Usage:
    python src/part5_2/benchmark.py [--runs N] [--output results.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict

import psutil
import requests
from tabulate import tabulate

OLLAMA_BASE = "http://localhost:11434"

MODELS = [
    {"tag": "gemma3:270m",      "quant": "Q8_0",   "params": "268 M"},
    {"tag": "gemma3:1b",        "quant": "Q4_K_M", "params": "1 B"},
    {"tag": "gemma3:1b-it-qat", "quant": "Q4_0",   "params": "1 B (QAT)"},
]

PROMPTS = [
    "Explain the difference between supervised and unsupervised learning in two sentences.",
    "What is the capital of France and why is it historically significant?",
    "Write a Python function that returns the nth Fibonacci number.",
]


def _vram_mib() -> float:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip()) if r.returncode == 0 else 0.0


def _ram_mib() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


@dataclass
class RunResult:
    model: str
    quant: str
    params: str
    prompt_tokens: int = 0
    eval_tokens: int = 0
    tps: float = 0.0
    peak_vram_mib: float = 0.0
    peak_ram_mib: float = 0.0
    outputs: list[str] = field(default_factory=list)


def _poll_resources(stop: threading.Event, peak: dict) -> None:
    while not stop.is_set():
        peak["vram"] = max(peak["vram"], _vram_mib())
        peak["ram"]  = max(peak["ram"],  _ram_mib())
        time.sleep(0.1)


def benchmark_model(tag: str, quant: str, params: str, runs: int) -> RunResult:
    result = RunResult(model=tag, quant=quant, params=params)

    # Warm-up: one short call so the model is loaded before we measure
    requests.post(f"{OLLAMA_BASE}/api/generate",
                  json={"model": tag, "prompt": "hi", "stream": False},
                  timeout=120)

    tps_list: list[float] = []

    for prompt in PROMPTS[:runs]:
        peak = {"vram": 0.0, "ram": 0.0}
        stop = threading.Event()
        poller = threading.Thread(target=_poll_resources, args=(stop, peak), daemon=True)
        poller.start()

        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": tag, "prompt": prompt, "stream": False},
            timeout=300,
        )
        stop.set()
        poller.join()

        data = resp.json()
        eval_count    = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 1)   # nanoseconds
        tps = eval_count / (eval_duration / 1e9) if eval_duration else 0.0

        tps_list.append(tps)
        result.prompt_tokens += data.get("prompt_eval_count", 0)
        result.eval_tokens   += eval_count
        result.peak_vram_mib  = max(result.peak_vram_mib, peak["vram"])
        result.peak_ram_mib   = max(result.peak_ram_mib,  peak["ram"])
        result.outputs.append(data.get("response", "").strip())

    result.tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs",   type=int, default=len(PROMPTS),
                    help="Number of prompts to run per model (max 3)")
    ap.add_argument("--output", default="src/part5_2/results.json",
                    help="Path to write raw JSON results")
    args = ap.parse_args()
    runs = min(args.runs, len(PROMPTS))

    results: list[RunResult] = []
    for m in MODELS:
        print(f"\n▶ Benchmarking {m['tag']} ({m['quant']}) …")
        r = benchmark_model(m["tag"], m["quant"], m["params"], runs)
        results.append(r)
        print(f"  TPS={r.tps:.1f}  VRAM={r.peak_vram_mib:.0f} MiB  RAM={r.peak_ram_mib:.0f} MiB")

    # ── Performance table ──────────────────────────────────────────────────────
    headers = ["Model", "Params", "Quantization", "Avg TPS", "Peak VRAM (MiB)", "Peak RAM (MiB)"]
    rows = [
        [r.model, r.params, r.quant, f"{r.tps:.1f}", f"{r.peak_vram_mib:.0f}", f"{r.peak_ram_mib:.0f}"]
        for r in results
    ]
    print("\n\n" + "=" * 70)
    print("PERFORMANCE REPORT — Gemma-3 Quantization Comparison")
    print("=" * 70)
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # ── Quality samples ────────────────────────────────────────────────────────
    print("\n\nQUALITY SAMPLES (prompt 1)")
    print("-" * 70)
    for r in results:
        print(f"\n[{r.model} / {r.quant}]")
        print(r.outputs[0] if r.outputs else "(no output)")

    # ── Save raw results ───────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n\nRaw results saved → {args.output}")


if __name__ == "__main__":
    main()
