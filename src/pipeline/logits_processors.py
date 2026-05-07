"""
Custom logits processors for non-standard sampling strategies.

MinPLogitsWarper:
    Filters tokens where p(token) < min_p * p(max_token).

MinPRescaledTemperatureWarper:
    Filters with min-p, then re-scales candidate logits with resample_temperature.

TopPRescaledTemperatureWarper:
    Filters with top-p (nucleus), then re-scales candidate logits with resample_temperature.
"""

import torch
from transformers import LogitsProcessor, TopPLogitsWarper


class MinPLogitsWarper(LogitsProcessor):
    def __init__(self, min_p: float, filter_temperature: float = 1.0):
        self.min_p = min_p
        self.filter_temperature = filter_temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: (batch_size, vocab_size) raw logits
        probs = torch.softmax(scores / self.filter_temperature, dim=-1)
        p_max = probs.max(dim=-1, keepdim=True).values
        mask = probs < self.min_p * p_max
        scores = scores.masked_fill(mask, float("-inf"))
        return scores


class MinPRescaledTemperatureWarper(LogitsProcessor):
    def __init__(self, min_p: float, filter_temperature: float = 1.0, resample_temperature: float = 2.0):
        self.min_p = min_p
        self.filter_temperature = filter_temperature
        self.resample_temperature = resample_temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Step 1: determine candidate set using filter_temperature
        probs = torch.softmax(scores / self.filter_temperature, dim=-1)
        p_max = probs.max(dim=-1, keepdim=True).values
        mask = probs < self.min_p * p_max  # tokens to exclude

        # Step 2: re-scale raw logits with resample_temperature and exclude non-candidates
        rescaled = scores / self.resample_temperature
        rescaled = rescaled.masked_fill(mask, float("-inf"))
        return rescaled


class TopPRescaledTemperatureWarper(LogitsProcessor):
    def __init__(self, top_p: float = 0.9, resample_temperature: float = 2.0):
        self.top_p = top_p
        self.resample_temperature = resample_temperature
        self._top_p_warper = TopPLogitsWarper(top_p=top_p)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Step 1: get the top-p mask by applying HF's warper and checking which tokens became -inf
        filtered = self._top_p_warper(input_ids, scores.clone())
        mask = filtered == float("-inf")  # tokens excluded by top-p

        # Step 2: re-scale raw logits with resample_temperature, exclude non-candidates
        rescaled = scores / self.resample_temperature
        rescaled = rescaled.masked_fill(mask, float("-inf"))
        return rescaled
