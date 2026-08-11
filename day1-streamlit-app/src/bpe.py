"""Hand-rolled BPE merge algorithm, ported from notebook 02."""

from collections import defaultdict


def word_to_symbols(word: str) -> tuple[str, ...]:
    return tuple(word) + ("</w>",)


def get_pair_counts(vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
    pairs: dict[tuple[str, str], int] = defaultdict(int)
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(pair: tuple[str, str], vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
    new_vocab: dict[tuple[str, ...], int] = {}
    a, b = pair
    merged = a + b
    for symbols, freq in vocab.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = new_vocab.get(tuple(new_symbols), 0) + freq
    return new_vocab


def run_bpe_merges(word_freqs: dict[str, int], num_merges: int) -> list[dict]:
    """Runs `num_merges` BPE merge steps, returns a list of step records:
    {step, pair, merged_symbol, vocab_snapshot}."""
    vocab = {word_to_symbols(word): freq for word, freq in word_freqs.items()}
    steps = []
    for step in range(1, num_merges + 1):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_pair(best_pair, vocab)
        steps.append({
            "step": step,
            "pair": best_pair,
            "merged_symbol": best_pair[0] + best_pair[1],
            "count": pairs[best_pair],
            "vocab_snapshot": {" ".join(k): v for k, v in vocab.items()},
        })
    return steps
