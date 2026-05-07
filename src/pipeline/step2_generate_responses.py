"""
Step 2: Generate 50 responses per prompt using a local HuggingFace model on GPU (bf16).

Usage:
  python -m src.pipeline.step2_generate_responses --model Qwen/Qwen2.5-14B-Instruct --config baseline
  python -m src.pipeline.step2_generate_responses --model meta-llama/Llama-3.1-8B-Instruct --config persona_t4

Supports resume: skips prompts that already have 50 responses.
"""
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from tqdm import tqdm
from src.utils.main_utils import load_standard_data, write_standard_data
from src.pipeline.sampling_configs import make_configs, SUPPORTED_MODELS, CONFIG_NAMES
from src.pipeline.logits_processors import MinPLogitsWarper, MinPRescaledTemperatureWarper, TopPRescaledTemperatureWarper

PROMPTS_PATH = "data/prompts/infinite_chats_eval.jsonl"
N_RESPONSES = 50

PERSONA_TEMPLATE = (
    "Randomly select a persona and briefly describe it. "
    "Then, as that persona, answer the following question: {question}\n\n"
    "Your response must follow this structure:\n"
    "Persona: [describe your persona here]\n"
    "Answer: {{your answer as the persona}}"
)

def load_model(model_id: str):
    print(f"Loading {model_id} with bf16...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def build_generate_kwargs(sampling_config: dict, tokenizer) -> dict:
    method = sampling_config["method"]
    base = dict(
        max_new_tokens=sampling_config["max_new_tokens"],
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    if method == "top_p":
        base["temperature"] = sampling_config["temperature"]
        base["top_p"] = sampling_config["top_p"]
    elif method == "min_p":
        base["logits_processor"] = LogitsProcessorList([
            MinPLogitsWarper(
                min_p=sampling_config["min_p"],
                filter_temperature=sampling_config["filter_temperature"],
            )
        ])
    elif method == "min_p_rescaled":
        base["logits_processor"] = LogitsProcessorList([
            MinPRescaledTemperatureWarper(
                min_p=sampling_config["min_p"],
                filter_temperature=sampling_config["filter_temperature"],
                resample_temperature=sampling_config["resample_temperature"],
            )
        ])
    elif method == "top_p_rescaled":
        base["logits_processor"] = LogitsProcessorList([
            TopPRescaledTemperatureWarper(
                top_p=sampling_config["top_p"],
                resample_temperature=sampling_config["resample_temperature"],
            )
        ])
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    return base


def format_prompt(prompt, tokenizer, use_persona: bool = False):
    if use_persona:
        content = PERSONA_TEMPLATE.format(question=prompt)
    else:
        content = prompt
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return content


def generate_responses(prompt, tokenizer, model, generate_kwargs, n_responses, batch_size, use_persona: bool = False):
    formatted = format_prompt(prompt, tokenizer, use_persona=use_persona)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_len = input_ids.shape[1]

    all_responses = []
    remaining = n_responses
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        batched_input_ids = input_ids.repeat(current_batch, 1)
        batched_attention_mask = attention_mask.repeat(current_batch, 1)
        with torch.no_grad():
            output_ids = model.generate(
                batched_input_ids,
                attention_mask=batched_attention_mask,
                **generate_kwargs,
            )
        new_tokens = output_ids[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        # Strip residual BPE space/newline chars (e.g. Ġ=\u0120, Ċ=\u010a) that
        # some tiktoken-based tokenizers (Qwen/DeepSeek) leave in decoded output.
        decoded = [t.replace('\u0120', ' ').replace('\u010a', '\n') for t in decoded]
        all_responses.extend(decoded)
        remaining -= current_batch

    return all_responses


def main(config_name: str = "baseline", configs: dict | None = None,
         tokenizer=None, model=None):
    if configs is None:
        configs = make_configs("Qwen/Qwen2.5-14B-Instruct")
    sampling_config = configs[config_name]
    model_id = sampling_config["model_id"]
    save_path = sampling_config["generations_path"]
    use_persona = sampling_config.get("persona", False)
    batch_size = sampling_config["batch_size"]

    prompts_data = load_standard_data(PROMPTS_PATH)

    results_dict: dict[str, dict] = {}
    if os.path.exists(save_path):
        for d in load_standard_data(save_path, is_print=False):
            results_dict[d["prompt"]] = d
    done_prompts = {p for p, d in results_dict.items() if len(d.get("responses", [])) >= N_RESPONSES}

    prompts_to_run = [d for d in prompts_data if d["prompt"] not in done_prompts]
    print(f"Config: {config_name} | {len(done_prompts)} done, {len(prompts_to_run)} remaining")

    if not prompts_to_run:
        print("All prompts already processed.")
        return tokenizer, model

    if tokenizer is None or model is None:
        tokenizer, model = load_model(model_id)

    generate_kwargs = build_generate_kwargs(sampling_config, tokenizer)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for item in tqdm(prompts_to_run, desc=f"Generating [{config_name}]"):
        prompt = item["prompt"]
        responses = generate_responses(
            prompt, tokenizer, model, generate_kwargs, N_RESPONSES, batch_size,
            use_persona=use_persona,
        )
        results_dict[prompt] = {"prompt": prompt, "responses": responses}
        write_standard_data(list(results_dict.values()), save_path)

    print(f"Done. Saved {len(results_dict)} records to {save_path}")
    return tokenizer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--config", required=True, choices=CONFIG_NAMES)
    args = parser.parse_args()
    configs = make_configs(args.model)
    main(config_name=args.config, configs=configs)
