"""Benchmark script for MATH500 using naive, standard, MCMC, and SMC sampling."""

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from glob import glob

import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TextStreamer

from constants import *
from grader_utils.math_grader import *
from grader_utils.math_normalize import *
from grader_utils.parse_utils import *
from power_samp_utils import *
from smc_samp_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run power-sampling comparisons on the MATH500 dataset."
    )
    parser.add_argument(
        "--save_str", action="store", type=str, default="results/", dest="save_str"
    )
    parser.add_argument(
        "--model",
        action="store",
        default="qwen",
        type=str,
        choices=[
            "qwen",
            "qwen_math",
            "phi",
            "tulu",
            "qwen_math_grpo",
            "phi_grpo",
            "llama",
        ],
    )
    parser.add_argument(
        "--temperature", action="store", default=0.25, type=float, dest="temperature"
    )
    parser.add_argument("--dataset", action="store", default="MATH", type=str)
    parser.add_argument("--cot", action="store", type=bool, default=True)
    parser.add_argument("--mcmc_steps", action="store", type=int, default=10)
    parser.add_argument(
        "--device",
        action="store",
        type=str,
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch_idx", action="store", type=int, default=0)
    args = parser.parse_args()

    # Runtime configuration.
    model = args.model
    device = args.device
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    mcmc_steps = args.mcmc_steps

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)

    # Model selection.
    if model == "qwen":
        model_str = "Qwen/Qwen2.5-7B"
    elif model == "qwen_math":
        model_str = "Qwen/Qwen2.5-Math-7B"
    elif model == "qwen_math_grpo":
        model_str = "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150"
    elif model == "phi":
        model_str = "microsoft/Phi-3.5-mini-instruct"
    elif model == "tulu":
        model_str = "allenai/Llama-3.1-Tulu-3-8B-DPO"
    elif model == "llama":
        model_str = "meta-llama/Llama-3.1-8B-Instruct"

    if dataset_name == "MATH":
        json_file = "data/MATH500.json"
        dataset = json.load(open(json_file, "r"))

    # Load tokenizer/model and construct sampling wrapper.
    print("dataset done")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_str, trust_remote_code=False
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=False,
    ).to(device)
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print("loaded models")
    results = []

    start = 100 * (args.batch_idx)
    end = 100 * (args.batch_idx + 5)
    count = 0

    naive_time = []
    std_time = []
    smc_time = []
    mcmc_time = []

    naive_acc = 0
    std_acc = 0
    smc_acc = 0
    mcmc_acc = 0

    naive_mem = []
    std_mem = []
    smc_mem = []
    mcmc_mem = []

    # Main benchmark loop.
    for problem, data in tqdm(
        enumerate(dataset[start + 3 : end]), desc="Benchmark on MATH"
    ):
        question = data["prompt"]
        answer = data["answer"]
        count += 1
        input_text = format_prompt(question, model, tokenizer, cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(
            hf_model.device
        )
        prefx = [idx.item() for idx in input_ids[0]]

        t1 = time.time()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        naive_temp_output = hf_model.generate(
            input_ids,
            max_new_tokens=3072,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            temperature=temp,
            repetition_penalty=1.0,
        )

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        naive_mem.append(peak)
        t2 = time.time()
        naive_time.append(t2 - t1)

        t1 = time.time()
        std_output = hf_model.generate(
            input_ids,
            max_new_tokens=3072,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            repetition_penalty=1.0,
        )
        t2 = time.time()
        std_time.append(t2 - t1)

        alpha = 1.0 / temp  # consistent with their naming
        t1 = time.time()

        cfg = SMCSamplingConfig(
            max_new_tokens=3072,
            alpha=alpha,  # since their temp argument means alpha = 1/temp
            n_particles=64,  # start small
            ess_threshold=0.5,
            temperature=temp,  # proposal = base for clean baseline; later you can set !=1 and keep correct>
            block_size=64,
            stop_on_boxed=True,  # <-- enabled
            boxed_check_window_tokens=256,  #
        )

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        smc_out = smc_power_sample_memopt(hf_model, tokenizer, input_ids, cfg)

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        smc_mem.append(peak)
        t2 = time.time()
        smc_time.append(t2 - t1)

        smc_ids = smc_out["chosen_sequence"][len(input_ids[0]) :].to("cpu")
        smc_completion = tokenizer.decode(smc_ids, skip_special_tokens=True)

        t1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        mcmc_power_samp_output, _, _, acceptance_ratio = mcmc_power_samp(
            autoreg_sampler, prefx, temp, mcmc_steps, max_new_tokens=3072
        )
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024**3)

        mcmc_mem.append(peak)
        t2 = time.time()
        mcmc_time.append(t2 - t1)

        naive_generated_ids = (
            naive_temp_output[0][:, len(input_ids[0]) :].squeeze().to("cpu")
        )
        std_generated_ids = std_output[0][:, len(input_ids[0]) :].squeeze().to("cpu")
        mcmc_power_samp_ids = (
            torch.tensor([mcmc_power_samp_output], dtype=torch.long, device=device)
            .squeeze()
            .to("cpu")
        )

        naive_completion = tokenizer.decode(
            naive_generated_ids, skip_special_tokens=True
        )
        std_completion = tokenizer.decode(std_generated_ids, skip_special_tokens=True)
        mcmc_completion = tokenizer.decode(
            mcmc_power_samp_ids, skip_special_tokens=True
        )

        naive_answer = parse_answer(naive_completion)
        std_answer = parse_answer(std_completion)
        mcmc_answer = parse_answer(mcmc_completion)
        smc_answer = parse_answer(smc_completion)

        if grade_answer(naive_answer, answer):
            naive_acc += 1

        if grade_answer(std_answer, answer):
            std_acc += 1

        if grade_answer(smc_answer, answer):
            smc_acc += 1

        if grade_answer(mcmc_answer, answer):
            mcmc_acc += 1

        print("Question: ", question)
        print("Gold answer:", answer)

        print("Naive Acc: ", naive_acc / count)
        print("Naive time: ", numpy.mean(np.array(naive_time)))
        print("Naive answer:", naive_answer)
        print("Naive mem: ", numpy.mean(np.array(naive_mem)))

        print("STD Acc: ", std_acc / count)
        print("STD time: ", numpy.mean(np.array(std_time)))
        print("STD answer: ", std_answer)

        print("SMC Acc: ", smc_acc / count)
        print("SMC time: ", numpy.mean(np.array(smc_time)))
        print("SMC answer:", smc_answer)
        print("SMC mem: ", numpy.mean(np.array(smc_mem)))

        print("MCMC Acc: ", mcmc_acc / count)
        print("MCMC time: ", numpy.mean(np.array(mcmc_time)))
        print("MCMC answer:", mcmc_answer)
        print("MCMC mem: ", numpy.mean(np.array(mcmc_mem)))

        print(
            "***************************************************************************************************"
        )
