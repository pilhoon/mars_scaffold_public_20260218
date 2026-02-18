from __future__ import annotations
import argparse
import logging
from pathlib import Path

from mars.controller import ControllerContext, run_controller


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("controller_main")

    ap = argparse.ArgumentParser()
    ap.add_argument("--task-yaml", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--default-conf", default=str(Path(__file__).resolve().parents[2] / "conf" / "default.yaml"))
    ap.add_argument("--max-iters", type=int, default=100)
    args = ap.parse_args()

    ctx = ControllerContext(
        run_dir=Path(args.run_dir),
        worker_id=args.worker_id,
        task_yaml=Path(args.task_yaml),
        default_conf_yaml=Path(args.default_conf),
    )
    logger.info(
        "controller start run_dir=%s task_yaml=%s default_conf=%s max_iters=%s worker_id=%s",
        ctx.run_dir,
        ctx.task_yaml,
        ctx.default_conf_yaml,
        args.max_iters,
        args.worker_id,
    )
    run_controller(ctx, max_iters=args.max_iters)
    logger.info("controller finished run_dir=%s", ctx.run_dir)


if __name__ == "__main__":
    main()
