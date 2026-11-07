"""Generate the bundled experiments catalog from the Python source of truth.

Usage::

    uv run python scripts/generate_experiments_catalog.py
    uv run python scripts/generate_experiments_catalog.py --check

The source of truth is ``src/kconv/experiments.py``.  The packaged JSON exists
so wheels expose the curated catalog without importing the engine and so a
stale asset fails a deterministic sync test instead of drifting silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kconv.experiments import catalog_payload  # noqa: E402

OUTPUT = ROOT / "src" / "kconv" / "assets" / "experiments.json"


def _serialize(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 when the packaged asset differs from the Python source.",
    )
    args = parser.parse_args()

    expected = _serialize(catalog_payload())
    output = args.output.resolve()
    if args.check:
        if not output.is_file():
            print(f"missing experiment catalog: {output}", file=sys.stderr)
            return 1
        actual = output.read_text(encoding="utf-8")
        if actual != expected:
            print(f"stale experiment catalog: {output}", file=sys.stderr)
            print("run: uv run python scripts/generate_experiments_catalog.py", file=sys.stderr)
            return 1
        print(f"experiment catalog is in sync: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8", newline="\n")
    count = len(catalog_payload()["experiments"])
    print(f"wrote {count} experiments to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
