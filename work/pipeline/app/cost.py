import os

# Default Gemini 3.8 Flash pricing (USD per 1,000,000 tokens) from cost file.py
DEFAULT_MODEL = os.getenv("GOOGLE_GENAI_MODEL", "gemini-3.8-flash")
INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 3.75

# (input, cached_input, output) USD per 1,000,000 tokens
MODEL_PRICING_PER_MILLION: dict[str, tuple[float, float, float]] = {
    "gemini-3.8-flash": (0.75, 0.75, 3.75),
    "gpt-5-mini": (0.25, 0.03, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5": (1.25, 0.125, 10.00),
}
_SORTED_PRICING_KEYS = sorted(MODEL_PRICING_PER_MILLION, key=len, reverse=True)


def _lookup_model_pricing(model: str | None) -> tuple[float, float, float] | None:
    if not model:
        return None
    normalized = model.strip().lower()
    if normalized in MODEL_PRICING_PER_MILLION:
        return MODEL_PRICING_PER_MILLION[normalized]
    for prefix in _SORTED_PRICING_KEYS:
        if normalized.startswith(prefix):
            return MODEL_PRICING_PER_MILLION[prefix]
    return None


def compute_token_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float = INPUT_PRICE_PER_MILLION,
    output_price_per_million: float = OUTPUT_PRICE_PER_MILLION,
    *,
    model: str | None = None,
    cached_input_tokens: int = 0,
) -> float:
    """Compute USD cost for LLM token usage with optional model-specific and cached pricing."""
    matched = _lookup_model_pricing(model)
    if matched is not None:
        in_rate, cached_rate, out_rate = matched
    else:
        in_rate = input_price_per_million
        cached_rate = input_price_per_million
        out_rate = output_price_per_million

    safe_in = max(0, input_tokens)
    safe_cached = min(max(0, cached_input_tokens), safe_in)
    uncached_in = safe_in - safe_cached
    safe_out = max(0, output_tokens)

    input_cost = (uncached_in / 1_000_000) * in_rate + (safe_cached / 1_000_000) * cached_rate
    output_cost = (safe_out / 1_000_000) * out_rate
    return input_cost + output_cost
