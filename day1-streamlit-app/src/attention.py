"""Attention extraction/plotting helpers, ported from notebook 01."""

import numpy as np


def get_attentions(tokenizer, model, sentence: str):
    """Returns (tokens, attentions) where attentions is a tuple of layer
    tensors shaped [1, num_heads, seq_len, seq_len]."""
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    outputs = model(**inputs)
    return tokens, outputs.attentions


def layer_head_matrix(attentions, layer: int, head: int) -> np.ndarray:
    return attentions[layer][0, head].detach().numpy()


def avg_last_layer_attention_to_token(tokens, attentions, target_token: str) -> list[tuple[str, float]]:
    """Averages attention across all heads in the last layer, ranks source
    tokens by how much attention they pay to `target_token`."""
    if target_token not in tokens:
        return []
    target_idx = tokens.index(target_token)
    last_layer = attentions[-1][0]  # [num_heads, seq_len, seq_len]
    avg_heads = last_layer.mean(dim=0).detach().numpy()  # [seq_len, seq_len]
    scores = avg_heads[:, target_idx]
    ranked = sorted(zip(tokens, scores), key=lambda x: x[1], reverse=True)
    return ranked


def causal_mask_demo(tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Hand-rolled toy causal-masking demo. Returns (raw_scores, masked_softmax)."""
    n = len(tokens)
    rng = np.random.default_rng(42)
    raw_scores = rng.random((n, n))
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    masked_scores = np.where(mask, -np.inf, raw_scores)
    # diagonal is always unmasked, so each row has a finite max -> safe softmax
    row_max = masked_scores.max(axis=1, keepdims=True)
    exp = np.exp(masked_scores - row_max)
    softmaxed = exp / exp.sum(axis=1, keepdims=True)
    return raw_scores, softmaxed


def positional_encoding(seq_len: int = 50, dim: int = 64) -> np.ndarray:
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    pe = np.zeros((seq_len, dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe
