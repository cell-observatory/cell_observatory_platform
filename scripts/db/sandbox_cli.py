from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.cli import argparser

import sandbox_ops


def _build_parser():
    p = argparser()
    p.prog = "sandbox_cli.py"

    sub = p.add_subparsers(dest="command", required=True)

    pb = sub.add_parser(
        "build",
        help="Build Apptainer sandbox image from docker://postgres:17 and apply my-postgres.conf.",
    )
    pb.add_argument(
        "--bind",
        action="append",
        dest="apptainer_binds",
        metavar="SRC:DST",
        help="Apptainer build --bind (repeatable). After SANDBOX_APPTAINER_BUILDBIND from .env.",
    )
    sub.add_parser(
        "run",
        help="Start sandbox Postgres via Apptainer on localhost:5433.",
    )

    sub.add_parser(
        "snapshot",
        help="Create a one-time copy of the production database into a new sandbox.tar.zst file that can be used for training sessions.",
    )
    sub.add_parser(
        "clone",
        help="Create a longterm read replica of the production database on your local machine.",
    )
    ps = sub.add_parser(
        "sync",
        help="Update read replica with the latest changes from the production database.",
    )
    ps.add_argument(
        "--no-copy-sequences",
        action="store_true",
        help="Skip pgcopydb copy sequences after sync exits",
    )
    ps.add_argument(
        "--continuous",
        action="store_true",
        help="Keep streaming changes instead of one-shot catch-up",
    )
    sub.add_parser(
        "cleanup",
        help="Reset the sandbox database and start over.",
    )

    sub.add_parser(
        "archive",
        help="Create scripts/db/YYYY_MM_DD_sandbox.tar.zst and copy to DATABASE_DIR",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args, unknown = parser.parse_known_args()

    cmd = args.command
    if cmd == "build":
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        binds = getattr(args, "apptainer_binds", None) or []
        sys.exit(sandbox_ops.build(cli_binds=binds))
    if cmd == "run":
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        sys.exit(sandbox_ops.run())
    if cmd == "snapshot":
        sys.exit(sandbox_ops.snapshot(unknown))
    if cmd == "clone":
        sys.exit(sandbox_ops.clone(unknown))
    if cmd == "sync":
        sys.exit(
            sandbox_ops.sync(
                unknown,
                copy_sequences_after=not args.no_copy_sequences,
                continuous=args.continuous,
            )
        )
    if cmd == "cleanup":
        sys.exit(sandbox_ops.cleanup(unknown))
    if cmd == "archive":
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        sys.exit(sandbox_ops.archive())
    raise AssertionError(f"unhandled command: {cmd}")


if __name__ == "__main__":
    main()
