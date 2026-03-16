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

Power-SMC targets the **power distribution** of an autoregressive language model:

$$\pi(y \mid x) \propto p_\theta(y \mid x)^\alpha, \quad \alpha > 1$$

where $p_\theta$ is the base model and $\alpha$ controls the sharpness of the distribution. Sampling from $\pi$ is intractable because the normalizing constant $Z_\alpha(x) = \sum_y p_\theta(y \mid x)^\alpha$ involves a sum over the exponentially large space of all token sequences. Standard approaches like Metropolis-Hastings (MH) require expensive full-sequence proposals and suffer from high rejection rates, resulting in 8–28× latency overhead.

Power-SMC reformulates power sampling as **Sequential Monte Carlo (SMC) inference**. We maintain $N$ particles (partial sequences), each extended one token at a time from a proposal distribution $q$. At each step $t$, the incremental importance weight for particle $i$ is:

$$w_t^{(i)} = \frac{p_\theta(y_t^{(i)} \mid y_{\lt t}^{(i)}, x)^{\alpha_t}}{q(y_t^{(i)} \mid y_{\lt t}^{(i)}, x)}$$

When the effective sample size (ESS) drops below a threshold $\kappa N$, we **resample**: duplicating high-weight particles and pruning low-weight ones. This concentrates computation on promising reasoning paths without waiting for complete sequences.

**Key design choices:**

- **Adaptive proposal temperature.** Setting the proposal temperature to $T_t = 1/\alpha_t$ minimizes per-step weight variance. When $q(y_t) \propto p_\theta(y_t)^{1/T_t} = p_\theta(y_t)^{\alpha_t}$, the token-level importance ratio becomes constant across tokens, and weight variance arises only from the $\alpha$-annealing schedule.

- **$\alpha$-annealing.** Rather than targeting $p^\alpha$ from the start, we linearly ramp $\alpha$ from 1 to $\alpha_{\text{final}}$ over the first $T_{\text{ramp}}$ tokens. This avoids early weight degeneracy and allows particles to establish diverse reasoning paths before sharpening begins.

- **Block-level resampling.** ESS is checked every `block_size` tokens rather than at every step, amortizing the overhead of KV cache reordering and reducing particle impoverishment from excessive resampling.

- **Copy-on-Write (CoW) KV cache.** After resampling, typically only $U \ll N$ unique ancestors survive (e.g., $U \approx 12$ out of $N = 64$). We keep only $U$ physical KV caches and expand logits to $N$ particles via an index mapping, deferring the full cache expansion until particles diverge at the next forward pass. This reduces transient peak memory from $2N$ to $N + U$ cache equivalents.

- **Shared prompt cache.** The prompt is processed once at batch size 1 and the resulting KV cache is replicated to $N$ particles, avoiding redundant computation.

After generation completes, a final sample is drawn from the $N$ particles proportional to their importance weights.

For settings where $N$ particles do not fit in GPU memory, we provide a **multi-round variant** that runs $R$ independent SMC sweeps with a smaller particle count $M$ and combines the results via importance-weighted selection across all $R \times M$ particles. Peak memory scales with $M$ rather than the effective particle count.


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

- [smc_samp_utils.py](smc_samp_utils.py)


---

## Inference

The following example runs Power-SMC inference on a single prompt using a HuggingFace model:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from smc_samp_utils import smc_power_sample_memopt, SMCSamplingConfig

# Load model
model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
)
model.eval()

# Prepare prompt
prompt = "Solve the following problem step by step.\n\nQuestion: What is the sum of all integer values of $n$ such that $\\frac{20}{2n-1}$ is an integer?\n\nSolution:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Configure sampler
cfg = SMCSamplingConfig(
    max_new_tokens=2048,
    alpha=4.0,               # sharpening exponent
    n_particles=64,           # number of SMC particles
    ess_threshold=0.5,        # resample when ESS < 0.5 * N
    temperature=0.25,         # proposal temperature (1/alpha)
    block_size=64,           # check ESS every 128 tokens
    alpha_ramp_tokens=400,    # anneal alpha from 1 to alpha over 400 tokens
    use_cow_cache=True,       # memory-optimized resampling
    shared_prompt_cache=True, # process prompt at batch=1
    stop_on_boxed=True,       # stop particle when \boxed{} detected
)

# Run inference
with torch.no_grad():
    result = smc_power_sample_memopt(model, tokenizer, input_ids, cfg)

# Decode the chosen sequence
output = tokenizer.decode(
    result["chosen_sequence"][input_ids.size(1):],
    skip_special_tokens=True,
)
print(output)

# Access diagnostics
stats = result["stats"]
print(f"Resample events: {stats['resample_count']}")
print(f"Unique ancestors per resample: {stats['unique_ancestors_history']}")
print(f"Generation stopped at token: {stats['done_at']}")
```

For memory-constrained settings, use the multi-round variant:

```python
from smc_power_sample_memopt import smc_power_sample_multiround

result = smc_power_sample_multiround(
    model, tokenizer, input_ids, cfg,
    physical_batch=16,  # particles per round (fits in GPU memory)
    n_rounds=4,         # total effective particles = 16 * 4 = 64
)

output = tokenizer.decode(
    result["chosen_sequence"][input_ids.size(1):],
    skip_special_tokens=True,
)
print(output)
print(f"Selected from round {result['round_idx']}, particle {result['particle_idx']}")
```


