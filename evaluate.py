"""SUPER Expert scoring — type-aware exact match with float tolerance.

Ported from AstaBench's super/task.py evaluate() function.

Comparison rules:
  - float: |predicted - gold| < epsilon (default 1e-2)
  - str: strip whitespace, then exact match
  - list: element-wise match, averaged
  - dict: key-wise match, averaged

Returns a score in [0.0, 1.0] — partial credit for dicts/lists.
"""

import json
from typing import Any


def evaluate(predicted: Any, gold: Any, float_epsilon: float = 1e-2) -> float:
    """Type-aware exact match evaluation.

    Args:
        predicted: Agent's submitted answer.
        gold: Gold-standard answer.
        float_epsilon: Tolerance for float comparison.

    Returns:
        Match score in [0.0, 1.0].
    """
    # Normalize int → float for comparison
    if isinstance(gold, int):
        gold = float(gold)
    if isinstance(predicted, int):
        predicted = float(predicted)

    # Type mismatch → 0
    if not isinstance(gold, type(predicted)):
        return 0.0

    if isinstance(gold, list):
        if len(gold) == 0:
            return 1.0 if len(predicted) == 0 else 0.0
        return sum(evaluate(p, g, float_epsilon) for p, g in zip(predicted, gold)) / len(gold)

    if isinstance(gold, dict):
        if len(gold) == 0:
            return 1.0 if len(predicted) == 0 else 0.0
        return sum(
            evaluate(predicted.get(gk), gv, float_epsilon)
            for gk, gv in gold.items()
        ) / len(gold)

    if isinstance(gold, str):
        return float(predicted.strip() == gold.strip())

    if isinstance(gold, float):
        return float(abs(predicted - gold) < float_epsilon)

    return 0.0


def parse_answer(raw: Any) -> Any:
    """Parse a submitted answer, attempting JSON decode if it's a string."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    return raw
