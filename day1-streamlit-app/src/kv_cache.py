"""KV-cache timing benchmark helpers, ported from notebook 04."""

import time


def timed_generate(tokenizer, model, prompt: str, max_new_tokens: int, use_cache: bool) -> tuple[float, str]:
    inputs = tokenizer(prompt, return_tensors="pt")
    start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return elapsed, text


def sweep_prompt_lengths(tokenizer, model, base_prompt: str, repeats: list[int], max_new_tokens: int = 30):
    """For each repeat count, times generation with and without cache on the
    prompt repeated that many times. Returns list of dicts."""
    results = []
    for r in repeats:
        prompt = (base_prompt + " ") * r
        n_tokens = len(tokenizer(prompt)["input_ids"])
        cached_time, _ = timed_generate(tokenizer, model, prompt, max_new_tokens, use_cache=True)
        uncached_time, _ = timed_generate(tokenizer, model, prompt, max_new_tokens, use_cache=False)
        results.append({
            "repeats": r,
            "prompt_tokens": n_tokens,
            "cached_seconds": cached_time,
            "uncached_seconds": uncached_time,
            "speedup": uncached_time / cached_time if cached_time > 0 else float("nan"),
        })
    return results
