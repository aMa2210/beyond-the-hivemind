"""
Check extraction failure rate per model per config.

Applies the same extraction logic as step3_compute_embeddings to count
how many responses fail to parse (unclosed <think>, missing Answer:, etc.)

Usage:
  python -m src.pipeline.check_fail_rate
"""
import json
import os
import re

from src.pipeline.sampling_configs import SUPPORTED_MODELS, CONFIG_NAMES, make_configs


def strip_think_block(response: str) -> str | None:
    if '<think>' in response and '</think>' not in response:
        return None
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()


def extract_answer(response: str) -> str | None:
    match = re.search(r'\{(.*)\}', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'Answer:\s*(.*)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def count_failures(path: str, extract_answer_flag: bool, extract_think_flag: bool, n_examples: int = 0) -> tuple[int, int, list[str]]:
    """Returns (fail_count, total_count, examples)."""
    if not os.path.exists(path):
        return 0, 0, []

    fail, total = 0, 0
    examples = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            for resp in record.get("responses", []):
                total += 1
                r = resp

                is_fail = False
                if extract_think_flag:
                    r = strip_think_block(r)
                    if r is None or not r:
                        is_fail = True
                        r = resp  # keep original for display

                if not is_fail and extract_answer_flag:
                    if not extract_answer(r):
                        is_fail = True

                if is_fail:
                    fail += 1
                    if len(examples) < n_examples:
                        examples.append(resp)

    return fail, total, examples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=0, help="Number of failure examples to show per config (default: 0)")
    args = parser.parse_args()

    header = f"{'Model':<35} {'Config':<12} {'Fail':>8} {'Total':>8} {'Rate':>8}"
    print(header)
    print("-" * len(header))

    for model_id in SUPPORTED_MODELS:
        configs = make_configs(model_id)
        for config_name in CONFIG_NAMES:
            cfg = configs.get(config_name)
            if cfg is None:
                continue
            path = cfg["generations_path"]
            extract_answer_flag = cfg.get("extract_answer", False)
            extract_think_flag = cfg.get("extract_think", False)

            fail, total, examples = count_failures(path, extract_answer_flag, extract_think_flag, n_examples=args.examples)
            if total == 0:
                rate_str = "N/A"
            else:
                rate_str = f"{fail / total:.2%}"

            print(f"{model_id:<35} {config_name:<12} {fail:>8} {total:>8} {rate_str:>8}")
            for i, ex in enumerate(examples):
                print(f"  [example {i+1}] {ex!r}")
        print()


if __name__ == "__main__":
    main()
