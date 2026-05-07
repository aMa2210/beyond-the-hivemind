"""
Orchestrator: run the full pipeline for all configs of a given model.

Loads the model once and runs all 12 configs sequentially.
Configs: no_persona_t1/t2/t4/t8/t16/t32 and persona_t1/t2/t4/t8/t16/t32.

Usage:
  python -m src.pipeline.run_pipeline --model Qwen/Qwen2.5-14B-Instruct
  python -m src.pipeline.run_pipeline --model meta-llama/Llama-3.1-8B-Instruct --config no_persona_t2
  python -m src.pipeline.run_pipeline --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --skip-quality
"""
import argparse
from src.pipeline.sampling_configs import make_configs, SUPPORTED_MODELS, CONFIG_NAMES
import src.pipeline.step1_load_prompts as step1
import src.pipeline.step2_generate_responses as step2
import src.pipeline.step3_compute_embeddings as step3


def main(
    model_id: str,
    config_names: list[str] | None = None,
    skip_quality: bool = False,
):
    if config_names is None:
        config_names = CONFIG_NAMES

    configs = make_configs(model_id)

    print("=" * 60)
    print("Step 1: Load prompts")
    print("=" * 60)
    step1.main()

    print(f"\nLoading model: {model_id}")
    tokenizer, model = step2.load_model(model_id)

    for config_name in config_names:
        print("\n" + "=" * 60)
        print(f"Config: {config_name}")
        print("=" * 60)

        print("Step 2: Generate responses")
        step2.main(config_name=config_name, configs=configs, tokenizer=tokenizer, model=model)

        print("Step 3: Compute embeddings")
        step3.main(config_name=config_name, configs=configs)

        if not skip_quality:
            print("Step 4: Quality scoring")
            from src.pipeline.step4_quality_scores import main as quality_main
            quality_main(config_name=config_name, configs=configs)

    print("\nPipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--config", choices=CONFIG_NAMES, default=None,
                        help="Single config to run (default: all 12)")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality scoring")
    args = parser.parse_args()
    config_names = [args.config] if args.config else None
    main(
        model_id=args.model,
        config_names=config_names,
        skip_quality=args.skip_quality,
    )
