"""Audit recipe signatures and duplicate preset/pipeline variants.

Examples:
    uv run python scripts/audit_recipes.py
    uv run python scripts/audit_recipes.py --strict --json
"""

from __future__ import annotations

import argparse
import json
import sys

from kconv.errors import RecipeError
from kconv.recipe_audit import audit_recipes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Fail on duplicate executable pipelines.")
    parser.add_argument("--json", action="store_true", help="Emit an ASCII-safe JSON payload.")
    args = parser.parse_args(argv)

    try:
        result = audit_recipes(strict=args.strict)
    except RecipeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=True))
        return 0

    print("recipe audit: " + ", ".join(f"{kind}={count}" for kind, count in result.kind_counts.items()))
    if not result.duplicate_groups:
        print("no duplicate signatures")
        return 0
    for group in result.duplicate_groups:
        flag = "pipeline" if group.has_pipeline else "preset"
        print(f"[{flag}] {', '.join(group.recipe_ids)}")
    if result.has_pipeline_duplicates:
        print("duplicate executable pipelines found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
