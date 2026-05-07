"""
Step 1: Load prompts from liweijiang/infinite-chats-eval (HuggingFace)
and save to data/prompts/infinite_chats_eval.jsonl
"""
import os
from datasets import load_dataset
from src.utils.main_utils import write_standard_data

SAVE_PATH = "data/prompts/infinite_chats_eval.jsonl"


def main():
    dataset = load_dataset("liweijiang/infinite-chats-eval", split="train")
    prompts = [{"prompt": row["query"]} for row in dataset]
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    write_standard_data(prompts, SAVE_PATH)
    print(f"Saved {len(prompts)} prompts to {SAVE_PATH}")


if __name__ == "__main__":
    main()
