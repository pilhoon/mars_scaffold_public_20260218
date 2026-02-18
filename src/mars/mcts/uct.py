from __future__ import annotations
import math


def uct_score(parent_visits: int, child_visits: int, child_value_sum: float, c_uct: float) -> float:
    """Upper Confidence Bound for Trees (UCT).
    Mirrors Eq (6) in the paper: Q + c * sqrt(ln N(s) / N(s,a)).
    Here Q is mean value (value_sum / child_visits).
    """
    if child_visits <= 0:
        return float("inf")  # force exploration of unvisited child

    q = child_value_sum / child_visits
    return q + c_uct * math.sqrt(max(0.0, math.log(max(1, parent_visits)) / child_visits))
