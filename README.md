# Beyond the Hivemind

Code for the paper **"Beyond the Hivemind: Escaping LLM Homogeneity via Meta-Persona Anchoring and Sequential Temperature Scaling"** (Fu et al., 2026).

We propose a two-stage framework — **Filtered Temperature Scaling (FTS)** combined with **Meta-Persona Anchoring** — that reduces semantic homogeneity in instruction-tuned LLMs. Average pairwise cosine similarity drops from ≈ 0.85 to ≈ 0.65 across five sub-20B open-weight models without compromising coherence.

## Models evaluated

- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-14B-Instruct`
- `google/gemma-4-E4B-it`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`

## Installation

```bash
git clone https://github.com/<user>/beyond-the-hivemind.git
cd beyond-the-hivemind
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit .env to add your OPENAI_API_KEY
```

A CUDA GPU (≥ 24 GB recommended for 14B models) is required for response generation.

## Data

The raw generations, embeddings (text-embedding-3-small), and quality scores used in the paper are archived on Zenodo:

> **Zenodo DOI**: *(to be added on release)*

Download the archive and unpack it as `data/` at the repo root:
```
data/
  generations/<model>/<config>.jsonl
  embeddings/<model>/<config>.jsonl
  quality_scores/<model>/<config>.jsonl
```

The prompt set is downloaded automatically from HuggingFace (`liweijiang/infinite-chats-eval`) by `step1_load_prompts.py`.

## Reproducing the paper

### End-to-end (one model at a time)

```bash
python -m src.pipeline.run_pipeline --model Qwen/Qwen2.5-14B-Instruct
```

This loads the model once and runs all 12 sampling configurations:

| Config            | Persona | Resampling temperature |
|-------------------|---------|------------------------|
| `no_persona_t1`   | no      | 1 (top-p only)         |
| `no_persona_t{2,4,8,16,32}` | no | 2, 4, 8, 16, 32 (FTS)  |
| `persona_t{1,2,4,8,16,32}`  | yes | 1, 2, 4, 8, 16, 32 (FTS) |

All configs use `top_p = 0.9`. FTS variants apply top-p filtering at T = 1 then rescale logits at the resampling temperature.

### Single config

```bash
python -m src.pipeline.run_pipeline --model Qwen/Qwen2.5-14B-Instruct --config persona_t32
```

### Skip quality scoring (saves OpenAI API cost)

```bash
python -m src.pipeline.run_pipeline --model Qwen/Qwen2.5-14B-Instruct --skip-quality
```

### Individual pipeline steps

```bash
python -m src.pipeline.step1_load_prompts
python -m src.pipeline.step2_generate_responses --model Qwen/Qwen2.5-14B-Instruct --config persona_t32
python -m src.pipeline.step3_compute_embeddings --model Qwen/Qwen2.5-14B-Instruct --config persona_t32
python -m src.pipeline.step4_quality_scores    --model Qwen/Qwen2.5-14B-Instruct --config persona_t32
```

### Generating paper figures

After `data/` is populated:

```bash
python -m src.pipeline.plot_all_figs            # heatmaps (Figures 3–8 in paper)
python -m src.pipeline.plot_fig_pca             # Figure 1 (PCA, cleanest prompt)
python -m src.pipeline.plot_fig_pca --rank 2    # Figure 1 (second prompt)
```

Output PDFs are written to `figures/`. Reference outputs are committed there for comparison.

## Diagnostics

```bash
python -m src.pipeline.check_fail_rate          # extraction failure rates per (model, config)
python -m src.pipeline.analyze_candidate_tokens # average candidate count per decoding step
```

## Repository layout

```
src/
  pipeline/
    sampling_configs.py          # 12 configs × 5 models registry
    logits_processors.py         # custom MinP/TopP rescaled-temperature warpers
    step1_load_prompts.py        # download INFINITY-CHAT prompts
    step2_generate_responses.py  # local GPU inference (bf16)
    step3_compute_embeddings.py  # OpenAI text-embedding-3-small
    step4_quality_scores.py      # GPT-4o-mini judge
    run_pipeline.py              # end-to-end orchestrator
    fig_utils.py                 # shared plotting helpers
    plot_fig*.py                 # one script per paper figure
    plot_all_figs.py             # batch all heatmaps
    check_fail_rate.py           # diagnostic
    analyze_candidate_tokens.py  # diagnostic
  utils/
    chat_models.py               # OpenAI client + embed_texts helper
    main_utils.py                # JSONL I/O
figures/                         # reference paper figure PDFs
```

## License

MIT — see `LICENSE`.

## Citation

See `CITATION.cff`. BibTeX will be added once the paper is published on arXiv.
