"""Build the bundled kconv source catalog from the local kaggle-toolkit index.

The catalog is metadata only.  It never copies or executes code from the
indexed repositories.  ``download_manifest.json`` describes the 211 locally
available templates; ``repositories_found_all.json`` describes all 1169
discovered repositories.  Both are merged by repository name so the API can
distinguish "indexed" from "downloaded".
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLKIT = ROOT.parent / "kaggle-toolkit"
OUTPUT = ROOT / "src" / "kconv" / "assets" / "catalog.json"

# Repositories with a concrete, original kconv recipe.  The recipe is a
# reimplementation of the methodology, not a vendored copy of the repository.
RECIPE_BY_REPO = {
    "arnabbiswas1/kaggle_pipeline_tps_aug_22": "arnabbiswas1/staged",
    "pesnik/kaggle-starter-kit": "pesnik/baseline",
    "sergeyshilin/kaggle-competitions-framework": "sergeyshilin/contracts",
    "congmnguyen/tabular-ml-lab": "congmnguyen/leakage-safe",
    "minerva-ml/open-solution-home-credit": "minerva/stacking",
    "kozodoi/Kaggle_Home_Credit": "kozodoi/tabular",
    "cdeotte/KGMON-Playbook-2026": "cdeotte/tabular",
    "StatsMaster/kaggle-grandmaster-templates": "statsmaster/tabular",
    "nyanp/nyaggle": "nyaggle/oof-ensemble",
    "jeongyoonlee/Kaggler": "kaggler/target-encoding",
    "kozodoi/dptools": "dptools/row-features",
    "upura/knnFeat": "knnfeat/knn-features",
    "kaz-Anova/ensemble_amazon": "ensemble-amazon/stacking",
    "RobMulla/kaggle-ieee-fraud-detection": "kaggle-ieee/fraud-pipeline",
}

# Stage coverage from kaggle-toolkit/PIPELINES.md.  ``full`` stages are listed
# in pipeline_stages; ``~`` stages are listed in partial_stages.  The source
# index remains useful even when a repository has no pipeline entry.
PIPELINE_STAGES: dict[str, dict[str, list[str]]] = {
    "arnabbiswas1/kaggle_pipeline_tps_aug_22": {
        "full": ["eda", "features", "cv", "model", "hpo", "ensemble", "submit"],
        "partial": [],
    },
    "RobMulla/kaggle-ieee-fraud-detection": {
        "full": ["eda", "features", "cv", "model", "submit"],
        "partial": ["hpo", "ensemble"],
    },
    "chammami/kaggle-competition-pipeline": {
        "full": ["features", "cv", "model", "ensemble", "submit"],
        "partial": [],
    },
    "randxie/kaggle-pipeline": {
        "full": ["features", "cv", "model", "ensemble", "submit"],
        "partial": [],
    },
    "sakami0000/kaggle_pipeline": {
        "full": ["features", "cv", "model", "submit"],
        "partial": ["eda", "hpo", "ensemble"],
    },
    "cttsai1985/Kaggle-Home-Credit-Default-Risk-Pipeline": {
        "full": ["features", "cv", "model", "hpo", "ensemble", "submit"],
        "partial": [],
    },
    "mratsim/home-credit-default-risk": {
        "full": ["features", "model", "submit"],
        "partial": ["cv"],
    },
    "killthekitten/kaggle-ds-bowl-2018-baseline": {
        "full": ["eda", "model", "submit"],
        "partial": ["features", "cv"],
    },
    "Julienbeaulieu/kaggle-computer-vision-competition": {
        "full": ["eda", "features", "cv", "model", "submit"],
        "partial": ["hpo", "ensemble"],
    },
    "5m0k3/gwd-yolov5-pytorch": {
        "full": ["cv", "model", "submit"],
        "partial": ["features", "ensemble"],
    },
    "Trigram19/m5-python-starter": {
        "full": ["features", "model"],
        "partial": ["cv", "submit"],
    },
    "congmnguyen/tabular-ml-lab": {
        "full": ["eda", "features", "cv", "model", "ensemble", "submit"],
        "partial": ["hpo"],
    },
    "ajisamudra/kaggle-otto-multi-objective-recsys": {
        "full": ["features", "cv", "model", "submit"],
        "partial": ["eda", "hpo", "ensemble"],
    },
    "sergeyshilin/kaggle-competitions-framework": {
        "full": ["cv", "model", "ensemble"],
        "partial": ["hpo", "submit"],
    },
    "bangdasun/kaggle": {
        "full": ["eda", "features", "cv", "model", "submit"],
        "partial": ["hpo", "ensemble"],
    },
    "minerva-ml/open-solution-cdiscount-starter": {
        "full": ["features", "cv", "model", "ensemble", "submit"],
        "partial": ["hpo"],
    },
    "Quantyra/kaggle-icr-challenge": {
        "full": ["eda", "model"],
        "partial": ["features", "cv", "submit"],
    },
    "WillKoehrsen/kaggle-automated-feature-engineering": {
        "full": ["features"],
        "partial": ["model"],
    },
    "cdeotte/Kaggle-OTTO-Comp": {
        "full": ["features", "cv", "model", "ensemble", "submit"],
        "partial": [],
    },
    "cdeotte/KGMON-Playbook-2026": {
        "full": ["eda", "features", "model", "ensemble"],
        "partial": ["cv", "submit"],
    },
}


def _load(path: Path) -> Any:
    # PowerShell-generated indexes may carry a UTF-8 BOM.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _license_class(value: str | None, downloaded: bool) -> str:
    text = (value or "").upper()
    if not text:
        return "unlicensed" if downloaded else "unknown"
    if "NO LICENSE" in text:
        return "unlicensed"
    if "GPL" in text or "GNU GENERAL PUBLIC" in text:
        return "copyleft"
    if "MIT" in text or "APACHE" in text or "BSD" in text:
        return "permissive"
    if "CC-BY-NC" in text or "NONCOMMERCIAL" in text:
        return "restricted"
    return "unknown"


def _reason(repo: str, recipe_id: str | None, license_class: str, downloaded: bool) -> str:
    if recipe_id:
        return f"kconv recipe '{recipe_id}' available; source code is not vendored"
    if not downloaded:
        return "indexed only; not downloaded, no adapter"
    if license_class == "copyleft":
        return "copyleft source; metadata/reference only, no code copied"
    if license_class == "unlicensed":
        return "unlicensed source; ideas/metadata only, no code copied"
    return "downloaded but not adapted to the tabular engine"


def build(toolkit: Path) -> dict[str, Any]:
    manifest = _load(toolkit / "index" / "download_manifest.json")
    discovered = _load(toolkit / "index" / "repositories_found_all.json")
    downloaded = {entry["repo"]: entry for entry in manifest}
    by_repo: dict[str, dict[str, Any]] = {}

    for entry in discovered:
        repo = entry.get("name")
        if not repo:
            continue
        by_repo[repo] = {
            "id": repo.replace("/", "__"),
            "repo": repo,
            "url": entry.get("url") or f"https://github.com/{repo}",
            "description": entry.get("desc") or "",
            "stars": entry.get("stars"),
            "language": entry.get("lang"),
            "updated": entry.get("updated"),
            "downloaded": repo in downloaded,
            "local_dir": downloaded.get(repo, {}).get("dir"),
            "category": downloaded.get(repo, {}).get("category"),
            "license": downloaded.get(repo, {}).get("license"),
            "files": downloaded.get(repo, {}).get("files"),
            "bytes": downloaded.get(repo, {}).get("bytes"),
        }

    # The manifest is authoritative for locally downloaded repositories.
    for repo, entry in downloaded.items():
        current = by_repo.setdefault(
            repo,
            {
                "id": entry["dir"],
                "repo": repo,
                "url": f"https://github.com/{repo}",
                "description": "",
                "stars": None,
                "language": None,
                "updated": None,
                "downloaded": True,
                "local_dir": entry["dir"],
                "category": entry.get("category"),
                "license": entry.get("license"),
                "files": entry.get("files"),
                "bytes": entry.get("bytes"),
            },
        )
        current.update(
            {
                "downloaded": True,
                "local_dir": entry.get("dir"),
                "category": entry.get("category"),
                "license": entry.get("license"),
                "files": entry.get("files"),
                "bytes": entry.get("bytes"),
            }
        )

    sources: list[dict[str, Any]] = []
    for repo, item in sorted(by_repo.items()):
        recipe_id = RECIPE_BY_REPO.get(repo)
        license_class = _license_class(item.get("license"), bool(item["downloaded"]))
        coverage = PIPELINE_STAGES.get(repo, {"full": [], "partial": []})
        item.update(
            {
                "license_class": license_class,
                "recipe_id": recipe_id,
                "executable": recipe_id is not None,
                "pipeline_stages": coverage["full"],
                "partial_stages": coverage["partial"],
                "reason": _reason(repo, recipe_id, license_class, bool(item["downloaded"])),
            }
        )
        sources.append(item)

    return {
        "schema_version": 1,
        "generated_from": {
            "manifest": "index/download_manifest.json",
            "discovered": "index/repositories_found_all.json",
            "pipeline_index": "PIPELINES.md",
        },
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--toolkit",
        type=Path,
        default=Path(os.environ.get("KCONV_TOOLKIT_DIR", DEFAULT_TOOLKIT)),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = build(args.toolkit.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(payload['sources'])} sources to {args.output}")


if __name__ == "__main__":
    main()
