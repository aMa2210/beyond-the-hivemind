"""
Step 4: Evaluate quality of generated responses using GPT-4o-mini as a judge.

For each prompt, scores all 50 responses on a 1-10 scale using a detailed rubric.
Supports resume: skips prompts that already have scores.

Usage:
  python -m src.pipeline.step4_quality_scores --model Qwen/Qwen2.5-14B-Instruct --config baseline
  python -m src.pipeline.step4_quality_scores --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --config persona_t4
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai

from src.utils.main_utils import load_standard_data, write_standard_data
from src.pipeline.sampling_configs import make_configs, SUPPORTED_MODELS, CONFIG_NAMES
from src.pipeline.step3_compute_embeddings import strip_think_block

MODEL = "gpt-4o-mini"
MAX_WORKERS = 20

JUDGE_SYSTEM_PROMPT = """You are a strict but fair evaluator of language model responses.
Score the response on a scale from 1 to 10 using the following rubric:

1-2: Very poor. The response is off-topic, incoherent, or completely unhelpful. May contain nonsense or refuse to engage.
3-4: Below average. Partially addresses the question but has significant issues: major gaps, poor reasoning, or low clarity.
5-6: Average. Responds to the question adequately but lacks depth, precision, or usefulness. Gets the job done minimally.
7-8: Good. Clear, relevant, and helpful response. Minor issues only — slight verbosity, small inaccuracies, or missed nuance.
9-10: Excellent. Fully addresses the question with clarity, accuracy, and genuine usefulness. Well-structured and insightful.

Output valid JSON only, in this exact format:
{"reason": "<one sentence justification>", "score": <integer 1-10>}"""


def build_judge_prompt(prompt: str, response: str) -> str:
    return f"Question: {prompt}\n\nResponse: {response}"


def parse_score(raw: str) -> int | None:
    try:
        data = json.loads(raw)
        score = int(data["score"])
        if 1 <= score <= 10:
            return score
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    match = re.search(r'"score"\s*:\s*(\d+)', raw)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 10:
            return score
    return None


def score_single(client: openai.OpenAI, prompt: str, response: str) -> int | None:
    try:
        result = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": build_judge_prompt(prompt, response)},
            ],
            max_tokens=100,
            temperature=0.0,
        )
        raw = result.choices[0].message.content
        return parse_score(raw)
    except Exception as e:
        print(f"[WARNING] GPT call failed: {e}")
        return None


def score_all_responses(
    client: openai.OpenAI,
    prompt: str,
    responses: list[str],
    max_workers: int = MAX_WORKERS,
) -> list[int | None]:
    scores = [None] * len(responses)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(score_single, client, prompt, resp): i
            for i, resp in enumerate(responses)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            scores[idx] = future.result()
    return scores


def main(config_name: str = "baseline", configs: dict | None = None):
    if configs is None:
        configs = make_configs("Qwen/Qwen2.5-14B-Instruct")
    sampling_config = configs[config_name]
    generations_path = sampling_config["generations_path"]
    save_path = sampling_config["quality_path"]
    extract_think = sampling_config.get("extract_think", False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = load_standard_data(generations_path)

    results_dict: dict[str, dict] = {}
    if os.path.exists(save_path):
        for d in load_standard_data(save_path, is_print=False):
            results_dict[d["prompt"]] = d
    done_prompts = {p for p, d in results_dict.items() if "scores" in d}

    to_score = [d for d in data if d["prompt"] not in done_prompts]
    print(f"Config: {config_name} | {len(done_prompts)} done, {len(to_score)} remaining")

    if not to_score:
        print("All prompts already scored.")
        _print_summary(results_dict)
        return

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for item in tqdm(to_score, desc=f"Scoring [{config_name}]"):
        prompt = item["prompt"]
        responses = item["responses"]

        # For DeepSeek-R1, score only the answer portion (after <think> block)
        if extract_think:
            responses = [strip_think_block(r) for r in responses]

        scores = score_all_responses(client, prompt, responses)

        valid_scores = [s for s in scores if s is not None]
        mean_score = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None

        if len(valid_scores) < len(scores):
            print(f"[WARNING] {len(scores) - len(valid_scores)} scoring failures for prompt: {prompt[:60]!r}")

        results_dict[prompt] = {
            "prompt": prompt,
            "config": config_name,
            "scores": scores,
            "mean_score": mean_score,
        }
        write_standard_data(list(results_dict.values()), save_path)

    print(f"Done. Saved {len(results_dict)} records to {save_path}")
    _print_summary(results_dict)


def _print_summary(results_dict: dict):
    valid_means = [d["mean_score"] for d in results_dict.values() if d.get("mean_score") is not None]
    if valid_means:
        overall = sum(valid_means) / len(valid_means)
        print(f"Overall mean quality score: {overall:.4f} (across {len(valid_means)} prompts)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--config", required=True, choices=CONFIG_NAMES)
    args = parser.parse_args()
    configs = make_configs(args.model)
    main(config_name=args.config, configs=configs)
