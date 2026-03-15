# Power-SMC

Codebase for experiments related to:

**Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM Reasoning**  
arXiv: https://arxiv.org/abs/2602.10273

---

## Overview

This repository focuses on **training-free reasoning-time sampling** methods for LLMs, including:

- Standard sampling baselines
- Power sampling via autoregressive MCMC
- Sequence-level SMC power sampling (memory-optimized implementation)

Benchmarks in this folder include:

- **MATH500**
- **GSM8K**
- **GPQA-Diamond**





---

## Method Summary

The target sequence-level distribution is:

$$
\pi_\alpha(y\mid x) \propto p_\theta(y\mid x)^\alpha,\quad \alpha > 1
$$

This code compares practical samplers that approximate or target this sharpened distribution with different latency/quality trade-offs.

---

## Environment Setup

### 1) Python environment

Use Python 3.10+ (recommended), then install dependencies:

- `torch`
- `transformers`
- `datasets`
- `numpy`
- `pandas`
- `tqdm`
- `sympy`
- `pylatexenc`

Optional formatting/dev tools:

- `ruff`

### 2) Model access

The scripts load models from Hugging Face Hub (e.g., Qwen, Phi, Tulu, Llama variants). Make sure your environment has access credentials if needed.

---

## Data

Expected local files/datasets include:

- `data/MATH500.json`
- `openai/gsm8k` via `datasets`
- `fingertap/GPQA-Diamond` via `datasets`


---

## Running Experiments

From [power-smc](power-smc):

- MATH:
  - `python -u power_samp_math.py --model qwen --temperature 0.25 --mcmc_steps 10`
- GSM8K:
  - `python -u power_samp_gsm.py --model qwen --temperature 0.25 --mcmc_steps 10`
- GPQA:
  - `python -u power_samp_gpqa.py --model qwen --temperature 0.25 --mcmc_steps 10`

---

## Key Configuration Knobs

Common options across scripts:

- `--model`
- `--temperature`
- `--mcmc_steps`
- `--cot`
- `--device`
- `--batch_idx`

SMC-specific settings are defined in `SMCSamplingConfig` in:

- [power-smc/smc_samp_utils_noschedule_plus_halt_opt.py](power-smc/smc_samp_utils_noschedule_plus_halt_opt.py)


---

## GitHub Readiness and Cleanup Policy

This repository was prepared with **non-functional cleanup only**:

- formatting
- comments/docstrings
- file organization/readability
- README and repository hygiene files

No logic, algorithm, or core experimental behavior was intentionally changed.



---

## License

Add your preferred repository license file (`LICENSE`) before public release.
