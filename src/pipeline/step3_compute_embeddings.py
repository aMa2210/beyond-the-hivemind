"""
Step 3: Compute embeddings for all generated responses using
OpenAI text-embedding-3-small.

Usage:
  python -m src.pipeline.step3_compute_embeddings --model Qwen/Qwen2.5-14B-Instruct --config baseline
  python -m src.pipeline.step3_compute_embeddings --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --config persona_t4
"""
import argparse
import json
import os
import re
from tqdm import tqdm
from src.utils.main_utils import load_standard_data, write_standard_data
from src.utils.chat_models import embed_texts
from src.pipeline.sampling_configs import make_configs, SUPPORTED_MODELS, CONFIG_NAMES

EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 512

# Maps embedding model name -> data directory root
EMB_MODEL_TO_DIR = {
    "text-embedding-3-small": "data/embeddings",
    "text-embedding-3-large": "data/embeddings_large",
    "gemini-embedding-2":     "data/embeddings_gemini2",
}


def _remap_embeddings_path(original_path: str, embedding_model: str) -> str:
    """Redirect embeddings_path to the correct root for the given embedding model."""
    from pathlib import Path
    new_root = EMB_MODEL_TO_DIR[embedding_model]
    rel = Path(original_path).relative_to("data/embeddings")
    return str(Path(new_root) / rel)


def strip_think_block(response: str) -> str | None:
    """Strip <think>...</think> from DeepSeek-R1 outputs. Returns None if <think> is unclosed."""
    if '<think>' in response and '</think>' not in response:
        return None
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()


def extract_answer(response: str) -> str | None:
    """Extract answer from a persona response.

    Priority:
    1. If curly braces present, extract content inside first { … last }
    2. Otherwise, extract text after 'Answer:'
    """
    match = re.search(r'\{(.*)\}', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'Answer:\s*(.*)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_persona(response: str) -> str | None:
    """Extract persona description (text between 'Persona:' and 'Answer:')."""
    match = re.search(r'Persona:\s*(.*?)\s*Answer:', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def main(config_name: str = "baseline", configs: dict | None = None, embedding_model: str = EMBEDDING_MODEL):
    if configs is None:
        configs = make_configs("Qwen/Qwen2.5-14B-Instruct")
    sampling_config = configs[config_name]
    generations_path = sampling_config["generations_path"]
    save_path = (
        _remap_embeddings_path(sampling_config["embeddings_path"], embedding_model)
        if embedding_model != EMBEDDING_MODEL
        else sampling_config["embeddings_path"]
    )
    should_extract = sampling_config.get("extract_answer", False)
    extract_think = sampling_config.get("extract_think", False)

    # Select embed function based on model
    if embedding_model == "gemini-embedding-2":
        from src.utils.chat_models import embed_texts_gemini
        def _embed(texts):
            return embed_texts_gemini(texts, model=embedding_model)
    else:
        def _embed(texts):
            return embed_texts(texts, model=embedding_model, batch_size=EMBED_BATCH_SIZE)

    data = load_standard_data(generations_path)

    results_dict: dict[str, dict] = {}
    if os.path.exists(save_path):
        for d in load_standard_data(save_path, is_print=False):
            results_dict[d["prompt"]] = d

    if should_extract:
        done_answer = {p for p, d in results_dict.items() if "embeddings" in d}
        done_persona = {p for p, d in results_dict.items() if "persona_embeddings" in d}
        to_embed = [d for d in data if d["prompt"] not in done_answer or d["prompt"] not in done_persona]
    else:
        done_answer = {p for p, d in results_dict.items() if "embeddings" in d}
        done_persona = set()
        to_embed = [d for d in data if d["prompt"] not in done_answer]

    print(f"Config: {config_name} | answer done={len(done_answer)}, persona done={len(done_persona)}, remaining={len(to_embed)}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_extraction_failures: list[dict] = []

    for item in tqdm(to_embed, desc=f"Embedding [{config_name}]"):
        prompt = item["prompt"]
        responses = item["responses"]
        need_answer = prompt not in done_answer
        need_persona = should_extract and prompt not in done_persona

        if should_extract:
            answer_texts, answer_indices = [], []
            persona_texts, persona_indices = [], []
            failed_indices = []

            for i, r in enumerate(responses):
                # Strip <think> block before any extraction (DeepSeek-R1)
                if extract_think:
                    r = strip_think_block(r)
                    if r is None:
                        print(f"[WARNING] Unclosed <think> block, skipping | prompt: {prompt[:60]!r} | response index: {i}")
                        failed_indices.append(i)
                        continue

                ans = extract_answer(r) if need_answer else None
                per = extract_persona(r) if need_persona else None

                ans_ok = bool(ans)
                per_ok = bool(per)

                if need_answer and ans_ok:
                    answer_texts.append(ans)
                    answer_indices.append(i)
                if need_persona and per_ok:
                    persona_texts.append(per)
                    persona_indices.append(i)
                if (need_answer and not ans_ok) or (need_persona and not per_ok):
                    failed_indices.append(i)

            if failed_indices:
                for idx in set(failed_indices):
                    print(f"[WARNING] Extraction failed | prompt: {prompt[:60]!r} | response index: {idx}")
                all_extraction_failures.extend(
                    {"prompt": prompt, "response_index": idx} for idx in set(failed_indices)
                )

            if need_answer and not answer_texts:
                print(f"[WARNING] All answer extractions failed for prompt: {prompt[:60]!r}, skipping.")
                continue
            if need_persona and not persona_texts:
                print(f"[WARNING] All persona extractions failed for prompt: {prompt[:60]!r}, skipping persona.")
                need_persona = False
        else:
            if extract_think:
                stripped_pairs = []
                for i, r in enumerate(responses):
                    s = strip_think_block(r)
                    if s is None:
                        print(f"[WARNING] Unclosed <think> block, skipping | prompt: {prompt[:60]!r} | response index: {i}")
                    elif s:
                        stripped_pairs.append((i, s))
                if stripped_pairs:
                    answer_indices, answer_texts = zip(*stripped_pairs)
                    answer_indices, answer_texts = list(answer_indices), list(answer_texts)
                else:
                    answer_indices, answer_texts = [], []
            else:
                answer_texts = responses
                answer_indices = list(range(len(responses)))

        existing = results_dict.get(prompt, {"prompt": prompt, "responses": responses})

        if need_answer:
            answer_embeddings = _embed(answer_texts)
            existing["embedded_indices"] = answer_indices
            existing["embeddings"] = answer_embeddings

        if need_persona:
            persona_embeddings = _embed(persona_texts)
            existing["persona_embedded_indices"] = persona_indices
            existing["persona_embeddings"] = persona_embeddings

        results_dict[prompt] = existing
        write_standard_data(list(results_dict.values()), save_path)

    if all_extraction_failures:
        failure_path = os.path.join(os.path.dirname(save_path), "extraction_failures.jsonl")
        with open(failure_path, "w") as f:
            for record in all_extraction_failures:
                f.write(json.dumps(record) + "\n")
        print(f"[WARNING] {len(all_extraction_failures)} extraction failures saved to {failure_path}")

    all_results = list(results_dict.values())
    print(f"Done. Saved {len(all_results)} records to {save_path}")
    if all_results:
        first_emb = all_results[0].get("embeddings")
        if first_emb:
            print(f"Embedding dimension: {len(first_emb[0])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--config", required=True, choices=CONFIG_NAMES)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL, choices=list(EMB_MODEL_TO_DIR.keys()))
    args = parser.parse_args()
    configs = make_configs(args.model)
    main(config_name=args.config, configs=configs, embedding_model=args.embedding_model)
