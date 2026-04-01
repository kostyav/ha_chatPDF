# Part 5.2 — Model Quantization & Performance Profiling Report

## Setup

| Item | Detail |
|------|--------|
| Hardware | NVIDIA Tesla T4 (15 360 MiB VRAM), 29 GB RAM |
| Inference engine | Ollama 0.19.0 |
| Model family | Gemma-3 (Google) |
| Prompts per model | 3 (same prompts for all models) |
| Measurement | TPS from Ollama `eval_count / eval_duration`; VRAM/RAM polled every 100 ms via `nvidia-smi` / `psutil` |

---

## Performance Report

| Model | Params | Quantization | Avg TPS | Peak VRAM (MiB) | Peak RAM (MiB) |
|---|---|---|---|---|---|
| gemma3:270m | 268 M | Q8_0 | **216.1** | 5 287 | 30 |
| gemma3:1b | 1 B | Q4_K_M | 135.9 | 6 297 | 30 |
| gemma3:1b-it-qat | 1 B (QAT) | Q4_0 | 114.7 | **7 493** | 30 |

> RAM stays flat at ~30 MiB because Ollama loads weights entirely onto the GPU; the host process is just a thin HTTP proxy.

---

## Quality Samples

All three models were given the same three prompts. Representative output for
*"Explain the difference between supervised and unsupervised learning in two sentences."*:

**gemma3:270m / Q8_0**
> Supervised learning uses labeled data, where the model learns from a dataset with known correct answers, while unsupervised learning uses unlabeled data, where the model explores the data to discover patterns and relationships.

**gemma3:1b / Q4_K_M**
> Supervised learning uses labeled data to train a model to predict outcomes, like predicting house prices based on features. Unsupervised learning, on the other hand, explores unlabeled data to discover patterns and groupings without any predefined outcomes.

**gemma3:1b-it-qat / Q4_0**
> **Supervised learning** involves training a model with labeled data – meaning the data includes both input and the desired output. **Unsupervised learning**, on the other hand, utilizes unlabeled data and aims to discover patterns or structures within the data itself, such as grouping similar items or reducing dimensions.

All three answers are factually correct and well-formed. The 1 B models produce
slightly richer prose; the 270 M model is more concise.

---

## Analysis: Speed vs. Quality Trade-offs

### Speed

`gemma3:270m` (Q8_0) is the fastest at **216 TPS** — 59 % faster than the
standard 1 B model — despite using a *higher* quantization precision (8-bit).
The speed advantage comes entirely from having ~3.7× fewer parameters, which
means fewer matrix multiplications per token regardless of bit-width.

`gemma3:1b-it-qat` (Q4_0) is the *slowest* of the three at **115 TPS**, even
though Q4_0 is nominally a lower-precision format than Q4_K_M. Q4_0 uses a
simpler, uniform 4-bit scheme that is less hardware-friendly on the T4's tensor
cores than the mixed-precision K-quant (Q4_K_M). The QAT model also generates
longer outputs on average (1 218 eval tokens vs. 617 for the 270 M model),
which inflates wall-clock time.

### VRAM

VRAM scales with `params × bits_per_weight`:

| Model | Approx. weight bytes | Measured VRAM |
|---|---|---|
| 268 M × 8 bit = 268 MB | ~268 MB weights | 5 287 MiB (includes KV cache + runtime) |
| 1 B × 4.5 bit ≈ 562 MB | ~562 MB weights | 6 297 MiB |
| 1 B × 4 bit = 500 MB | ~500 MB weights | 7 493 MiB |

The QAT model's higher VRAM despite lower bit-width is explained by its larger
KV-cache footprint: it generates longer sequences, keeping more activations
resident during generation.

### Output Quality

All three models produce coherent, accurate answers across the three test
prompts (ML concepts, geography/history, Python code). Qualitative differences:

- **gemma3:270m Q8_0** — concise, correct, occasionally omits detail (e.g.,
  Paris answer was a single sentence vs. multi-paragraph for the 1 B models).
- **gemma3:1b Q4_K_M** — best balance: detailed, well-structured, iterative
  Fibonacci implementation with O(n) complexity noted explicitly.
- **gemma3:1b-it-qat Q4_0** — most verbose; adds input validation
  (`ValueError`) to the Fibonacci function; slightly more hallucination risk in
  the Paris history answer (incorrect founding date attributed to Augustus).

### Summary

| Goal | Recommended variant |
|---|---|
| Maximum throughput / real-time UX | `gemma3:270m` Q8_0 — 216 TPS |
| Best quality per VRAM dollar | `gemma3:1b` Q4_K_M — 136 TPS, richest output |
| Lowest VRAM footprint | `gemma3:270m` Q8_0 — 5 287 MiB |

The key insight is that **parameter count dominates speed more than bit-width**
on a single T4. Halving the parameters (1 B → 270 M) gains +59 % TPS even when
moving from 4-bit to 8-bit precision. Within the same parameter count, Q4_K_M
outperforms Q4_0 because the K-quant format is better aligned with GPU tensor
core arithmetic.
