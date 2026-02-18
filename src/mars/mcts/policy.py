from __future__ import annotations

from mars.types import ActionType, NodeStatus, NodeRecord
from mars.config import MCTSConfig


def decide_action(node: NodeRecord, is_root: bool, cfg: MCTSConfig) -> ActionType:
    """Domain-specific expansion operator selection.
    Paper uses actions {Draft, Improve, Debug}. We mirror that.
    - Root expansion => Draft
    - Buggy node => Debug (until fixed or max attempts)
    - Valid node => Improve (limited children)
    """
    if is_root:
        return ActionType.DRAFT
    if node.status == NodeStatus.BUGGY:
        return ActionType.DEBUG
    return ActionType.IMPROVE


def expansion_limit(node: NodeRecord, is_root: bool, cfg: MCTSConfig) -> int:
    """Return the max number of children to expand from this node.

    Root: generate a bounded number of independent draft seeds.
    Buggy nodes: allow multiple debug attempts.
    Valid nodes: allow bounded improve branches.
    """
    if is_root:
        return max(1, int(cfg.max_root_drafts))
    if node.status == NodeStatus.BUGGY:
        return max(1, int(cfg.max_debug_attempts))
    return max(1, int(cfg.max_improve_children))
