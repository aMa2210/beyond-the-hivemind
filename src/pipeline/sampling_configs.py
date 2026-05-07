"""
Sampling configuration registry.

make_configs(model_id) returns 12 configs per model:
  - no_persona_t1:               top-p=0.9, T=1 (reuses baseline data path)
  - no_persona_t2/t4/t8/t16/t32: top-p=0.9 rescaled, no persona
  - persona_t1/t2/t4/t8/t16/t32: top-p=0.9 rescaled, with persona

Per-model parameters (batch_size, max_new_tokens, extract_think) live in MODEL_PARAMS.
"""

MODEL_PARAMS = {
    "meta-llama/Llama-3.1-8B-Instruct":         {"batch_size": 20, "max_new_tokens": 512},
    "mistralai/Mistral-7B-Instruct-v0.3":       {"batch_size": 20, "max_new_tokens": 512},
    "Qwen/Qwen2.5-14B-Instruct":                {"batch_size": 10, "max_new_tokens": 512},
    "google/gemma-4-E4B-it":                    {"batch_size": 20, "max_new_tokens": 512},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {"batch_size": 8,  "max_new_tokens": 2048, "extract_think": True},
}

SUPPORTED_MODELS = list(MODEL_PARAMS.keys())

CONFIG_NAMES = [
    "no_persona_t1", "no_persona_t2", "no_persona_t4", "no_persona_t8", "no_persona_t16", "no_persona_t32",
    "persona_t1",    "persona_t2",    "persona_t4",    "persona_t8",    "persona_t16",    "persona_t32",
]


def make_configs(model_id: str) -> dict:
    """Return 12 experimental configs for the given model_id."""
    params = MODEL_PARAMS[model_id]
    _mid = model_id.replace("/", "_")

    shared = {"model_id": model_id, "batch_size": params["batch_size"], "max_new_tokens": params["max_new_tokens"]}
    if params.get("extract_think"):
        shared["extract_think"] = True

    def _paths(name):
        return {
            "generations_path": f"data/generations/{_mid}/{name}.jsonl",
            "embeddings_path":  f"data/embeddings/{_mid}/{name}.jsonl",
            "quality_path":     f"data/quality_scores/{_mid}/{name}.jsonl",
        }

    def _no_persona(t, file_name):
        return {
            **shared,
            "method": "top_p_rescaled",
            "top_p": 0.9,
            "resample_temperature": float(t),
            "decoding_method": f"no_persona_t{t}",
            **_paths(file_name),
        }

    def _persona(t, file_name):
        return {
            **shared,
            "method": "top_p_rescaled",
            "top_p": 0.9,
            "resample_temperature": float(t),
            "persona": True,
            "extract_answer": True,
            "decoding_method": f"persona_t{t}",
            **_paths(file_name),
        }

    return {
        # no_persona_t1 reuses baseline data (top_p T=1, accepted minor method difference)
        "no_persona_t1": {
            **shared,
            "method": "top_p",
            "temperature": 1.0,
            "top_p": 0.9,
            "decoding_method": "no_persona_t1",
            **_paths("baseline"),
        },
        "no_persona_t2":  _no_persona(2,  "no_persona_t2"),
        "no_persona_t4":  _no_persona(4,  "no_persona_t4"),
        "no_persona_t8":  _no_persona(8,  "no_persona_t8"),
        "no_persona_t16": _no_persona(16, "no_persona_t16"),
        "no_persona_t32": _no_persona(32, "no_persona_t32"),
        "persona_t1":  _persona(1,  "persona_t1"),
        "persona_t2":  _persona(2,  "persona_t2"),
        "persona_t4":  _persona(4,  "persona_t4"),
        "persona_t8":  _persona(8,  "persona_t8"),
        "persona_t16": _persona(16, "persona_t16"),
        "persona_t32": _persona(32, "persona_t32"),
    }
