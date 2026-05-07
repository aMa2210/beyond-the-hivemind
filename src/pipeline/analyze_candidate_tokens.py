"""
Analyze the average number of candidate tokens at each decoding step
for top_p=0.9 vs min_p=0.1, to test whether peaked distributions
explain min_p's lower diversity.

Runs 5 prompts × 5 responses each, reports average candidate token count
per decoding step for each sampling method.

Usage:
    python -m src.pipeline.analyze_candidate_tokens --model Qwen/Qwen2.5-14B-Instruct
"""
import argparse
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    LogitsProcessor, LogitsProcessorList,
    TemperatureLogitsWarper, TopPLogitsWarper,
)
from src.utils.main_utils import load_standard_data
from src.pipeline.sampling_configs import SUPPORTED_MODELS
from src.pipeline.logits_processors import MinPLogitsWarper
from src.pipeline.step2_generate_responses import format_prompt

N_PROMPTS = 5
N_RESPONSES = 5
MAX_NEW_TOKENS = 200  # shorter for speed


class CandidateCounter(LogitsProcessor):
    """Wraps a filtering processor and records non-masked token count per step."""
    def __init__(self, inner: LogitsProcessor):
        self.inner = inner
        self.counts = []  # one entry per (response, decoding step)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        filtered = self.inner(input_ids, scores)
        # Count non-inf tokens for each sequence in the batch, then average
        n_candidates = (filtered != float("-inf")).sum(dim=-1).float()
        self.counts.extend(n_candidates.tolist())
        return filtered

    def mean_candidates(self):
        return float(np.mean(self.counts)) if self.counts else 0.0


def build_processor(method: str):
    """Return (LogitsProcessorList, CandidateCounter) for the given method."""
    if method == "top_p":
        # Temperature=1.0 (identity) then top_p filter
        inner = TopPLogitsWarper(top_p=0.9)
    elif method == "min_p":
        inner = MinPLogitsWarper(min_p=0.1, filter_temperature=1.0)
    else:
        raise ValueError(method)

    counter = CandidateCounter(inner)
    return LogitsProcessorList([counter]), counter


def generate_and_count(prompts, tokenizer, model, method):
    processor_list, counter = build_processor(method)

    for prompt in prompts:
        formatted = format_prompt(prompt, tokenizer)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_len = input_ids.shape[1]

        # Generate N_RESPONSES in one batch
        batched_ids = input_ids.repeat(N_RESPONSES, 1)
        batched_mask = attention_mask.repeat(N_RESPONSES, 1)

        with torch.no_grad():
            model.generate(
                batched_ids,
                attention_mask=batched_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                logits_processor=processor_list,
                pad_token_id=tokenizer.pad_token_id,
            )

    return counter.mean_candidates()


def main(model_id: str = "Qwen/Qwen2.5-14B-Instruct"):
    # Load first N_PROMPTS from dataset
    prompts_data = load_standard_data("data/prompts/infinite_chats_eval.jsonl")
    prompts = [d["prompt"] for d in prompts_data[:N_PROMPTS]]

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    print(f"\nRunning {N_PROMPTS} prompts × {N_RESPONSES} responses × {MAX_NEW_TOKENS} steps\n")

    results = {}
    for method in ["top_p", "min_p"]:
        print(f"Testing [{method}]...")
        avg = generate_and_count(prompts, tokenizer, model, method)
        results[method] = avg
        print(f"  Average candidate tokens per step: {avg:.1f}")

    print("\n===== Summary =====")
    for method, avg in results.items():
        print(f"  {method:20s}: {avg:.1f} candidate tokens/step")

    ratio = results["top_p"] / results["min_p"] if results["min_p"] > 0 else float("inf")
    print(f"\n  top_p allows {ratio:.2f}x more candidates than min_p on average")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", choices=SUPPORTED_MODELS)
    args = parser.parse_args()
    main(model_id=args.model)
