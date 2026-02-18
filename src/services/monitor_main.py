from __future__ import annotations
import argparse
import time
from pathlib import Path

from mars.store import connect, list_nodes, get_kv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--interval-sec", type=float, default=2.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    conn = connect(run_dir)

    while True:
        best = get_kv(conn, "best_node_id", default="(none)")
        nodes = list_nodes(conn, limit=50)
        last = nodes[-1] if nodes else None
        print(f"[monitor] best_node_id={best} nodes={len(nodes)} last={last.node_id if last else None} last_status={last.status.value if last else None} last_metric={last.metric if last else None}")
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
