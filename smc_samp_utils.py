"""
Power-SMC with memory-optimized KV cache management.

Key optimizations over the baseline smc_power_sample:
  1. Copy-on-Write (CoW) cache deduplication after resampling:
     - After systematic resampling, only U unique ancestors survive (typically U << N).
     - We keep only U physical KV caches until the next forward pass.
     - We expand logits U->N for sampling, then expand caches U->N before forward.
     - Mathematically identical to the original; no approximation.

  2. Shared prompt cache:
     - Process prompt once at batch=1, then broadcast the KV cache.
     - Saves prompt_len/total_len fraction of initial memory.

  3. Multi-round SMC (smc_power_sample_multiround):
     - Run multiple independent SMC runs with smaller particle counts.
     - Peak memory = physical_batch KV caches instead of N.
     - Combine via importance-weighted selection across rounds.

Usage:
  from smc_power_sample_memopt import smc_power_sample_memopt, SMCSamplingConfig
  result = smc_power_sample_memopt(model, tokenizer, input_ids, cfg)
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ============================================================
# Import shared utilities from original (or inline them)
# ============================================================


def apply_repetition_penalty_(logits, prev_tokens, penalty):
    if penalty is None or penalty == 1.0:
        return logits
    N, V = logits.shape
    for i in range(N):
        toks = prev_tokens[i]
        if toks.numel() == 0:
            continue
        uniq = torch.unique(toks)
        row = logits[i]
        vals = row.index_select(0, uniq)
        vals = torch.where(vals < 0, vals * penalty, vals / penalty)
        row.index_copy_(0, uniq, vals)
    return logits


def top_k_filter_(logits, top_k):
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    kth_vals = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
    logits.masked_fill_(logits < kth_vals, -float("inf"))
    return logits


def top_p_filter_(logits, top_p, min_tokens_to_keep=1):
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumprobs > top_p
    if min_tokens_to_keep > 1:
        sorted_mask[:, :min_tokens_to_keep] = False
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
    sorted_mask[:, 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_idx, src=sorted_mask)
    logits.masked_fill_(mask, -float("inf"))
    return logits


def effective_sample_size(w, eps=1e-12):
    w = w.clamp_min(eps)
    return float(1.0 / torch.sum(w * w).item())


def systematic_resample(w, generator=None):
    N = w.numel()
    device = w.device
    if generator is None:
        u0 = torch.rand((), device=device)
    else:
        u0 = torch.rand((), device=device, generator=generator)
    positions = (u0 + torch.arange(N, device=device)) / N
    cdf = torch.cumsum(w, dim=0)
    cdf[-1] = 1.0
    idx = torch.searchsorted(cdf, positions, right=False)
    idx = idx.clamp_max(N - 1).to(torch.long)
    return idx


# ============================================================
# KV Cache utilities: reorder, clone-subset, quantize
# ============================================================


def _reorder_tensor_batchdim(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return t.index_select(0, idx)


def _recursive_reorder(obj, idx):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if obj.dim() >= 1 and obj.size(0) == idx.numel():
            return _reorder_tensor_batchdim(obj, idx)
        return obj
    if isinstance(obj, tuple):
        return tuple(_recursive_reorder(x, idx) for x in obj)
    if isinstance(obj, list):
        return [_recursive_reorder(x, idx) for x in obj]
    if isinstance(obj, dict):
        return {k: _recursive_reorder(v, idx) for k, v in obj.items()}
    return obj


def reorder_past_key_values(model, past_key_values, idx):
    if past_key_values is None:
        return None
    if hasattr(model, "_reorder_cache") and callable(getattr(model, "_reorder_cache")):
        try:
            return model._reorder_cache(past_key_values, idx)
        except Exception:
            pass
    if hasattr(past_key_values, "reorder_cache") and callable(
        getattr(past_key_values, "reorder_cache")
    ):
        try:
            past_key_values.reorder_cache(idx)
            return past_key_values
        except Exception:
            pass
    return _recursive_reorder(past_key_values, idx)


def _recursive_select_batch(obj, idx):
    """Select a subset of the batch dimension (idx is a 1-D long tensor of unique indices)."""
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if idx.numel() == 0:
            return obj
        if obj.dim() >= 1 and obj.size(0) >= idx.max().item() + 1:
            return obj.index_select(0, idx).contiguous()
        return obj
    if isinstance(obj, tuple):
        return tuple(_recursive_select_batch(x, idx) for x in obj)
    if isinstance(obj, list):
        return [_recursive_select_batch(x, idx) for x in obj]
    if isinstance(obj, dict):
        return {k: _recursive_select_batch(v, idx) for k, v in obj.items()}
    return obj


def _recursive_expand_batch(obj, expand_idx):
    """
    Expand from U unique caches to N particles.
    expand_idx: [N] tensor where expand_idx[i] = index into the U-sized batch.
    """
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if expand_idx.numel() == 0:
            return obj
        U = int(expand_idx.max().item()) + 1
        if obj.dim() >= 1 and obj.size(0) == U:
            return obj.index_select(0, expand_idx)
        return obj
    if isinstance(obj, tuple):
        return tuple(_recursive_expand_batch(x, expand_idx) for x in obj)
    if isinstance(obj, list):
        return [_recursive_expand_batch(x, expand_idx) for x in obj]
    if isinstance(obj, dict):
        return {k: _recursive_expand_batch(v, expand_idx) for k, v in obj.items()}
    return obj


def select_cache_subset(model, past, idx):
    """
    Extract KV cache for a subset of particles (unique ancestors).
    This MODIFIES past in-place for HF DynamicCache (which is what we want:
    we're shrinking from N -> U and don't need the old N-batch cache).
    """
    if past is None:
        return None
    # HF DynamicCache.reorder_cache does in-place index_select — perfect for shrinking
    if hasattr(past, "reorder_cache") and callable(getattr(past, "reorder_cache")):
        try:
            past.reorder_cache(idx)
            return past
        except Exception:
            pass
    # Model-level hook (some older HF models)
    if hasattr(model, "_reorder_cache") and callable(getattr(model, "_reorder_cache")):
        try:
            return model._reorder_cache(past, idx)
        except Exception:
            pass
    return _recursive_select_batch(past, idx)


def expand_cache(model, past, expand_idx):
    """
    Expand cache from U unique entries to N particles using expand_idx.
    expand_idx: [N] tensor mapping logical particle -> physical cache slot.
    This MODIFIES past in-place for HF DynamicCache (replacing U-batch with N-batch).
    """
    if past is None:
        return None
    # HF DynamicCache.reorder_cache with expand_idx does index_select in-place
    if hasattr(past, "reorder_cache") and callable(getattr(past, "reorder_cache")):
        try:
            past.reorder_cache(expand_idx)
            return past
        except Exception:
            pass
    if hasattr(model, "_reorder_cache") and callable(getattr(model, "_reorder_cache")):
        try:
            return model._reorder_cache(past, expand_idx)
        except Exception:
            pass
    return _recursive_expand_batch(past, expand_idx)


# ============================================================
# Shared prompt cache: process prompt once, replicate cache
# ============================================================


def _build_prompt_cache(model, input_ids_1: torch.Tensor, N: int):
    """
    Process prompt at batch=1, then replicate the cache to N particles.
    Returns: (past_key_values for batch N, last_logits [N, V])
    """
    # Forward pass with batch=1
    out = model(input_ids=input_ids_1, use_cache=True)
    past_1 = getattr(out, "past_key_values", None)
    logits_1 = out.logits[:, -1, :]  # [1, V]

    # Expand to N particles: replicate the single cache entry N times
    # NOTE: must use expand_cache, NOT reorder_past_key_values, because
    # the recursive fallback in reorder_past_key_values checks
    # obj.size(0) == idx.numel() (1 != N → no-op). expand_cache's
    # fallback checks obj.size(0) == U (1 == 1 → correct).
    expand_idx = torch.zeros(N, dtype=torch.long, device=input_ids_1.device)
    past_N = expand_cache(model, past_1, expand_idx)
    logits_N = logits_1.expand(N, -1).contiguous()

    del out
    torch.cuda.empty_cache()

    return past_N, logits_N


# ============================================================
# Config (same as original + memory options)
# ============================================================


@dataclass
class SMCSamplingConfig:
    max_new_tokens: int = 3072
    alpha: float = 2.0
    n_particles: int = 32
    ess_threshold: float = 0.5
    temperature: float = 0.25
    block_size: int = 128
    force_eos_after_done: bool = True

    alpha_ramp_tokens: int = 400
    min_new_tokens: int = 100
    repetition_penalty: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    min_tokens_to_keep: int = 1
    penalize_prompt: bool = False
    hard_truncation: bool = True
    soft_truncation_value: float = -50.0

    stop_on_boxed: bool = True
    boxed_check_window_tokens: int = 256

    # ---- Memory optimization options ----
    use_cow_cache: bool = True  # Copy-on-Write cache after resampling
    shared_prompt_cache: bool = True  # Process prompt at batch=1 then replicate


# ============================================================
# Helpers
# ============================================================


def _logsumexp(x, dim=0):
    return torch.logsumexp(x, dim=dim)


def _alpha_ramp(t, alpha_final, ramp_T):
    if ramp_T <= 1:
        return float(alpha_final)
    if t < ramp_T:
        frac = float(t + 1) / float(ramp_T)
        return 1.0 + (float(alpha_final) - 1.0) * frac
    return float(alpha_final)


_BOX_ONLY_RE = re.compile(r"^[\s{}]*$")


def _has_nonempty_boxed(text):
    for macro in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(macro, start)
            if idx < 0:
                break
            j = idx + len(macro)
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] == "{":
                depth = 0
                right = None
                for k in range(j, len(text)):
                    c = text[k]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            right = k
                            break
                if right is not None:
                    content = text[j + 1 : right].strip()
                    while (
                        len(content) >= 2 and content[0] == "{" and content[-1] == "}"
                    ):
                        inner = content[1:-1].strip()
                        if inner == content:
                            break
                        content = inner
                    if content and (_BOX_ONLY_RE.fullmatch(content) is None):
                        return True
            start = idx + len(macro)
    return False


# ============================================================
# Core: Memory-optimized SMC power sampler
# ============================================================


@torch.no_grad()
def smc_power_sample_memopt(
    model,
    tokenizer,
    input_ids: torch.Tensor,  # [1, L]
    cfg: SMCSamplingConfig,
) -> Dict[str, Any]:
    """
    Power-SMC with Copy-on-Write KV cache deduplication.

    After resampling, instead of immediately replicating KV caches to all N particles,
    we identify unique ancestors (U << N) and run the next forward pass at batch size U.
    We then sample N tokens from U sets of logits (each ancestor's logits shared across
    its child particles), and expand the cache only when particles diverge.

    Memory profile:
      - Peak KV cache: N caches (same as baseline)
      - Post-resampling KV cache: U caches (typically U = 8-20 for N=64)
      - Average savings: depends on resampling frequency and block_size
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1
    device = input_ids.device
    g = torch.Generator(device=device)

    N = int(cfg.n_particles)
    prompt_len = input_ids.size(1)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is required.")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    # ---- Build prompt cache ----
    if cfg.shared_prompt_cache:
        past, last_logits = _build_prompt_cache(model, input_ids, N)
    else:
        prompt = input_ids.expand(N, -1).contiguous()
        out = model(input_ids=prompt, use_cache=True)
        past = getattr(out, "past_key_values", None)
        last_logits = out.logits[:, -1, :]

    # ---- Preallocate ----
    total_len = prompt_len + int(cfg.max_new_tokens)
    seqs = torch.full((N, total_len), pad_id, dtype=torch.long, device=device)
    seqs[:, :prompt_len] = input_ids.expand(N, -1)
    cur = prompt_len

    done = torch.zeros(N, dtype=torch.bool, device=device)
    cum_logp = torch.zeros(N, dtype=torch.float32, device=device)
    log_w = torch.zeros(N, dtype=torch.float32, device=device)

    alpha_final = float(cfg.alpha)
    ramp_T = max(int(cfg.alpha_ramp_tokens), 1)
    prev_alpha = 1.0

    stats = {
        "resample_count": 0,
        "ess_history": [],
        "mean_logw_history": [],
        "max_logw_history": [],
        "unique_ancestors_history": [],
        "peak_batch_size_history": [],
        "done_at": None,
    }

    Tmax = int(cfg.max_new_tokens)
    min_new = max(int(cfg.min_new_tokens), 0)

    # ---- CoW state ----
    # When in "compressed" mode after resampling, we track the mapping
    # from logical particles to physical cache slots.
    # physical_to_logical[phys_idx] = list of logical particle indices
    # logical_to_physical[log_idx] = physical cache index
    cow_active = False
    logical_to_physical = None  # [N] -> index into physical batch
    physical_to_logical = None  # list of lists
    n_physical = N  # current physical batch size

    for t in range(Tmax):
        # alpha_t = _alpha_ramp(t=t, alpha_final=alpha_final, ramp_T=ramp_T)
        # T_t = 1.0 / alpha_t  # adaptive proposal temperature
        # ---- Compute base logprobs and proposal ----
        # last_logits shape: [n_physical, V] (may be < N if CoW active)
        base_logprobs_phys = F.log_softmax(last_logits.float(), dim=-1)

        if cfg.temperature == 1.0:
            prop_logits_phys = last_logits.float()
        else:
            prop_logits_phys = last_logits.float() / cfg.temperature

        # ---- Expand logits/logprobs from physical to logical if CoW ----
        if cow_active and logical_to_physical is not None:
            # Map physical logits -> logical particles
            base_logprobs = base_logprobs_phys[logical_to_physical]  # [N, V]
            prop_logits = prop_logits_phys[logical_to_physical]  # [N, V]
        else:
            base_logprobs = base_logprobs_phys
            prop_logits = prop_logits_phys

        # ---- Apply proposal modifications (at logical = N level) ----
        if t < min_new:
            prop_logits[:, eos_id] = -float("inf")

        if cfg.repetition_penalty is not None and cfg.repetition_penalty != 1.0:
            if cfg.penalize_prompt:
                history = seqs[:, :cur]
            else:
                history = seqs[:, prompt_len:cur]
            apply_repetition_penalty_(prop_logits, history, cfg.repetition_penalty)

        if cfg.top_k is not None and cfg.top_k > 0:
            top_k_filter_(prop_logits, cfg.top_k)
        if cfg.top_p is not None and cfg.top_p < 1.0:
            top_p_filter_(
                prop_logits, cfg.top_p, min_tokens_to_keep=cfg.min_tokens_to_keep
            )

        if not cfg.hard_truncation:
            prop_logits = torch.where(
                torch.isinf(prop_logits),
                torch.full_like(prop_logits, float(cfg.soft_truncation_value)),
                prop_logits,
            )

        prop_logprobs = F.log_softmax(prop_logits, dim=-1)  # [N, V]

        # ---- Sample N tokens ----
        next_tokens = torch.empty(N, dtype=torch.long, device=device)
        if cfg.force_eos_after_done:
            active = ~done
            if active.any():
                probs_active = torch.exp(prop_logprobs[active])
                next_tokens[active] = torch.multinomial(
                    probs_active, num_samples=1, generator=g
                ).squeeze(-1)
            next_tokens[done] = eos_id
        else:
            probs = torch.exp(prop_logprobs)
            next_tokens = torch.multinomial(probs, num_samples=1, generator=g).squeeze(
                -1
            )

        # ---- Gather per-particle logp, logq ----
        idx_tok = next_tokens.view(N, 1)
        token_logp = torch.gather(base_logprobs, dim=-1, index=idx_tok).squeeze(-1)
        token_logq = torch.gather(prop_logprobs, dim=-1, index=idx_tok).squeeze(-1)

        if cfg.force_eos_after_done:
            token_logp = torch.where(done, torch.zeros_like(token_logp), token_logp)
            token_logq = torch.where(done, torch.zeros_like(token_logq), token_logq)

        # ---- Alpha ramp + weight update (same as original) ----
        cum_logp_prev = (
            cum_logp  # no clone needed: cum_logp + token_logp creates new tensor
        )
        alpha_t = _alpha_ramp(t=t, alpha_final=alpha_final, ramp_T=ramp_T)
        delta = float(alpha_t - prev_alpha)
        if delta != 0.0:
            log_w = log_w + delta * cum_logp_prev
            prev_alpha = float(alpha_t)

        log_w = log_w + (float(alpha_t) - 1.0) * token_logp
        log_w = log_w + (token_logp - token_logq)
        cum_logp = cum_logp + token_logp

        # ---- Write tokens ----
        seqs[:, cur] = next_tokens
        cur += 1

        # ---- Update done ----
        newly_done = (next_tokens == eos_id) & (~done)
        done = done | newly_done

        if cfg.stop_on_boxed:
            active = ~done
            if active.any():
                start_tok = max(prompt_len, cur - int(cfg.boxed_check_window_tokens))
                active_idx = torch.nonzero(active, as_tuple=False).flatten().tolist()
                for i_part in active_idx:
                    gen_text_tail = tokenizer.decode(
                        seqs[i_part, start_tok:cur].tolist(), skip_special_tokens=True
                    )
                    if _has_nonempty_boxed(gen_text_tail):
                        done[i_part] = True

        if stats["done_at"] is None and done.all():
            stats["done_at"] = t + 1

        if done.all():
            break

        # ---- Expand cache from physical to logical BEFORE the forward pass ----
        # If CoW was active, we now need all N particles to have their own cache
        # since they (potentially) sampled different tokens.
        if cow_active and logical_to_physical is not None:
            past = expand_cache(model, past, logical_to_physical)
            cow_active = False
            logical_to_physical = None
            physical_to_logical = None
            n_physical = N

        # ---- Forward pass at batch N ----
        out = model(
            input_ids=next_tokens.view(N, 1),
            past_key_values=past,
            use_cache=True,
        )
        past = getattr(out, "past_key_values", None)
        last_logits = out.logits[:, -1, :]

        # ---- Resampling check ----
        is_block_end = ((t + 1) % int(cfg.block_size) == 0) or (t == Tmax - 1)
        if is_block_end:
            lw = log_w - _logsumexp(log_w, dim=0)
            w = torch.exp(lw)
            ess = effective_sample_size(w)
            stats["ess_history"].append(ess)
            stats["mean_logw_history"].append(float(log_w.mean().item()))
            stats["max_logw_history"].append(float(log_w.max().item()))

            if ess < float(cfg.ess_threshold) * N:
                idx_rs = systematic_resample(w, generator=g)

                # ---- CoW deduplication ----
                if cfg.use_cow_cache:
                    unique_ancestors, inverse = torch.unique(
                        idx_rs, return_inverse=True
                    )
                    U = unique_ancestors.numel()
                    stats["unique_ancestors_history"].append(U)

                    # Reorder logical state (seqs, done, cum_logp) to full N
                    seqs = seqs.index_select(0, idx_rs)
                    done = done.index_select(0, idx_rs)
                    cum_logp = cum_logp.index_select(0, idx_rs)

                    # But only keep U physical caches
                    past = select_cache_subset(model, past, unique_ancestors)
                    last_logits = last_logits.index_select(0, unique_ancestors)

                    # Build logical-to-physical mapping
                    logical_to_physical = inverse  # [N] -> [0, U)
                    cow_active = True
                    n_physical = U
                else:
                    # Standard full resampling (original behavior)
                    seqs = seqs.index_select(0, idx_rs)
                    done = done.index_select(0, idx_rs)
                    cum_logp = cum_logp.index_select(0, idx_rs)
                    past = reorder_past_key_values(model, past, idx_rs)
                    last_logits = last_logits.index_select(0, idx_rs)
                    stats["unique_ancestors_history"].append(N)

                log_w = torch.zeros_like(log_w)
                stats["resample_count"] += 1

                stats["peak_batch_size_history"].append(n_physical)

    # ---- Final alpha catch-up ----
    if prev_alpha != alpha_final:
        log_w = log_w + float(alpha_final - prev_alpha) * cum_logp
        prev_alpha = alpha_final

    seqs_out = seqs[:, :cur]
    lw = log_w - _logsumexp(log_w, dim=0)
    w = torch.exp(lw)
    chosen_idx = int(torch.multinomial(w, 1, generator=g).item())
    chosen_sequence = seqs_out[chosen_idx]

    return {
        "sequences": seqs_out,
        "log_w": log_w,
        "w": w,
        "chosen_idx": chosen_idx,
        "chosen_sequence": chosen_sequence,
        "stats": stats,
    }


# ============================================================
# Bonus: Multi-round SMC for extreme memory constraints
# ============================================================


@torch.no_grad()
def smc_power_sample_multiround(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    cfg: SMCSamplingConfig,
    physical_batch: int = 16,
    n_rounds: int = 2,
) -> Dict[str, Any]:
    """
    Multi-round SMC for when even N/2 particles don't fit in memory.

    Strategy:
      - Run `n_rounds` independent SMC runs, each with `physical_batch` particles.
      - Collect the final weighted particle sets from all rounds.
      - Do a final weighted selection across all n_rounds * physical_batch particles.

    This gives you effective N = n_rounds * physical_batch diversity
    while only ever holding `physical_batch` KV caches in GPU memory.

    Trade-off: you lose inter-round resampling (particles from different rounds
    can't exchange information). But empirically, if each round already has
    reasonable particle count (16-32), this works well.

    Total effective particles: n_rounds * physical_batch
    Peak GPU memory: physical_batch KV caches
    """
    all_sequences = []
    all_log_w = []

    # Temporarily override n_particles
    round_cfg = SMCSamplingConfig(
        **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    )
    round_cfg.n_particles = physical_batch

    for r in range(n_rounds):
        result = smc_power_sample_memopt(model, tokenizer, input_ids, round_cfg)

        # Pad sequences to same length for stacking
        all_sequences.append(result["sequences"])
        all_log_w.append(result["log_w"])

        # Free GPU memory between rounds
        del result
        torch.cuda.empty_cache()

    # ---- Combine across rounds ----
    # Normalize log-weights across ALL particles from all rounds
    combined_log_w = torch.cat(all_log_w, dim=0)  # [n_rounds * physical_batch]
    lw = combined_log_w - _logsumexp(combined_log_w, dim=0)
    w = torch.exp(lw)

    # Sample final particle
    g = torch.Generator(device=input_ids.device)
    chosen_global = int(torch.multinomial(w, 1, generator=g).item())

    round_idx = chosen_global // physical_batch
    particle_idx = chosen_global % physical_batch

    chosen_sequence = all_sequences[round_idx][particle_idx]

    return {
        "chosen_sequence": chosen_sequence,
        "w": w,
        "chosen_global_idx": chosen_global,
        "round_idx": round_idx,
        "particle_idx": particle_idx,
        "n_total_particles": n_rounds * physical_batch,
    }
