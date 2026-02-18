from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MetricHistory:
    m_min: Optional[float] = None
    m_max: Optional[float] = None

    def update(self, metric: float) -> None:
        if self.m_min is None or metric < self.m_min:
            self.m_min = metric
        if self.m_max is None or metric > self.m_max:
            self.m_max = metric

    def normalize(self, metric: float) -> float:
        # Eq (3) from the paper.
        if self.m_min is None or self.m_max is None:
            return 0.5
        if self.m_max == self.m_min:
            return 0.5
        return (metric - self.m_min) / (self.m_max - self.m_min)


def efficiency_guided_reward(norm_score: float, exec_time_sec: float, time_limit_sec: float, w: float) -> float:
    """Eq (4): R(v) = G(v) * (t/L)^w, where w < 0 penalizes long runs."""
    if time_limit_sec <= 0:
        return norm_score
    ratio = max(1e-9, exec_time_sec / time_limit_sec)
    return norm_score * (ratio ** w)
