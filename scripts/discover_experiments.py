"""Read-only experiment candidate scanner for kaggle-toolkit.

The scanner never imports or executes third-party code.  It walks text files,
looks for unusual-solution keywords, and emits line-level evidence for human or
LLM triage.  Output is a candidate list, not a recipe catalog.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

TEXT_SUFFIXES = {".md", ".py", ".ipynb", ".yaml", ".yml", ".txt", ".rst"}
SKIP_PARTS = {".git", ".venv", "node_modules", "data", "datasets", "input", "output", "__pycache__"}
MAX_FILE_BYTES = 1_500_000

PATTERNS: dict[str, str] = {
    "pseudo-labeling": r"pseudo[\s_-]?label",
    "adversarial-sample-weighting": r"adversarial.{0,40}(weight|sample)|sample.{0,20}weight",
    "feature-crosses": r"feature[\s_-]?cross|cross[\s_-]?feature|interaction feature",
    "row-statistics": r"row[\s_-]?(mean|std|min|max|stat)",
    "target-encoding-oof": r"target[\s_-]?encod|out[\s_-]?of[\s_-]?fold.{0,30}encod|\bwoe\b",
    "rank-blend": r"rank.{0,20}(blend|average|mean)|blend.{0,20}rank",
    "oof-stacking": r"stack(ing|er)|meta[\s_-]?model|out[\s_-]?of[\s_-]?fold.{0,30}stack",
    "seed-averaging": r"seed.{0,20}(averag|bag|ensemble)",
    "group-aware-cv": r"group[\s_-]?(kfold|cv|split)|group.{0,20}leak",
    "time-series-gap": r"time[\s_-]?series.{0,30}(gap|split)|gap.{0,20}time",
    "text-metadata": r"text.{0,20}(length|word count|metadata)|word[\s_-]?count",
    "adversarial-validation": r"adversarial[\s_-]?(validation|valid|classifier)",
    "llm-feature-proposals": r"\bllm\b|gpt|language model|prompt.{0,20}feature",
    "multi-seed-snapshot": r"snapshot.{0,20}ensemble|checkpoint.{0,20}ensemble",
}


def _iter_text_files(root: Path, max_files: int) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if len(files) >= max_files:
            break
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        files.append(path)
    return files


def scan(toolkit: Path, *, max_files: int = 500) -> list[dict[str, object]]:
    root = toolkit / "templates"
    if not root.is_dir():
        root = toolkit
    compiled = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in PATTERNS.items()}
    candidates: list[dict[str, object]] = []
    for path in _iter_text_files(root, max_files):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        relative = path.relative_to(root)
        repo = relative.parts[0] if relative.parts else path.name
        lines = text.splitlines()
        for experiment, regex in compiled.items():
            hits = 0
            for line_number, line in enumerate(lines, 1):
                if len(line) > 500 or '"image/png"' in line or "iVBOR" in line:
                    # Notebook image payloads frequently contain random keywords.
                    continue
                if not regex.search(line):
                    continue
                candidates.append(
                    {
                        "experiment": experiment,
                        "repo": repo,
                        "path": str(relative).replace("\\", "/"),
                        "line": line_number,
                        "snippet": line.strip()[:240],
                    }
                )
                hits += 1
                if hits >= 3:
                    break
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, default=Path("kaggle-toolkit"))
    parser.add_argument("--output", type=Path, default=Path("experiments_candidates.json"))
    parser.add_argument("--max-files", type=int, default=500)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    if args.max_files < 1:
        parser.error("--max-files must be positive")
    candidates = scan(args.toolkit, max_files=args.max_files)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "toolkit": str(args.toolkit),
        "patterns": PATTERNS,
        "candidates": candidates,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    print(f"wrote {len(candidates)} candidates to {args.output}")


if __name__ == "__main__":
    main()
