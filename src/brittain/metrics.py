"""Metrics shared by the evaluation scripts.

These live here rather than inside one script because `compare.py` and
`novice.py` both report them and must report the SAME number. A metric defined
twice is a metric that eventually disagrees with itself.
"""
from __future__ import annotations


def distinct_ngram_ratio(token_ids: list[int], order: int = 4) -> float:
    """Unique n-grams over total n-grams. 1.0 means no repetition at all."""
    if len(token_ids) < order + 1:
        return 1.0
    grams = [tuple(token_ids[i:i + order]) for i in range(len(token_ids) - order + 1)]
    return len(set(grams)) / len(grams)


def repetition_collapse(
    token_ids: list[int], order: int = 4, threshold: float = 0.40, minimum: int = 32
) -> bool:
    """True when a generation has degenerated into a repeating loop.

    Brittain2's known failure is collapsing into repeated words. That failure is
    invisible to syntax validity — `x = 1` repeated eighty times parses fine — and
    invisible to BPB, which is measured on held-out text the model never generates.
    It needs its own metric or the pilot cannot report "no collapse" honestly.

    Healthy code stays well above 0.4 distinct-4 even with Python's boilerplate.
    Generations shorter than `minimum` tokens are not judged: too few n-grams for
    the ratio to carry information.
    """
    if len(token_ids) < minimum:
        return False
    return distinct_ngram_ratio(token_ids, order) < threshold


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from Chen et al. 2021: 1 - C(n-c, k)/C(n, k).

    `n` samples drawn, `c` of them correct. Computed as a running product to
    avoid overflowing large binomial coefficients.
    """
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    product = 1.0
    for i in range(n - c + 1, n + 1):
        product *= 1.0 - k / i
    return 1.0 - product
