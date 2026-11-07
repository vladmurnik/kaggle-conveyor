"""Runnable kconv API examples.

This file is intentionally safe by default: it only reads bundled catalogs.
Project creation/run happens only when ``--data`` is supplied, and a full run
needs ``--run`` in addition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from kconv.api import (
    KConv,
    list_experiments,
    list_models,
    list_pipelines,
    list_runtime_profiles,
    list_sources,
    run_dataset,
)
from kconv.features.llm_expr import materialize_plan
from kconv.llm_features import propose_features


def show_catalog() -> None:
    """Discover models, recipes, experiments, profiles and sources."""
    print("models:", len(list_models()))
    print("binary models:", len(list_models(task="binary")))
    print("regression models:", len(list_models(task="regression")))
    print("pipelines:", [pipeline.id for pipeline in list_pipelines()])
    print("experiments:", [experiment.id for experiment in list_experiments(status="implemented")])
    print("profiles:", [(profile.level, profile.name, profile.device) for profile in list_runtime_profiles()])
    print("sources (downloaded / total):", len(list_sources()), "/", len(list_sources(downloaded=None)))


def show_facade() -> None:
    """Use the KConv facade when an application wants one object."""
    kconv = KConv()
    print("facade models:", len(kconv.models()))
    print("facade profile 1:", kconv.profile(1).to_dict())
    print("facade profile models:", kconv.profile_models(1, "binary"))
    print("facade experiment:", kconv.experiment("llm-feature-proposals").status)


def show_source(identifier: str) -> None:
    """Inspect one source without importing its code."""
    source = KConv().source(identifier)
    print(json.dumps(source.to_dict(), ensure_ascii=False, indent=2))
    print("local_path:", source.local_path())
    if source.executable:
        print("recipe:", source.recipe_id)
    else:
        print("not executable: use it as metadata/reference only")


def show_llm_features() -> None:
    """Validate a fake LLM answer without a network call."""
    schema = {
        "columns": ["amount", "term", "rate"],
        "notes": "target and IDs intentionally omitted",
    }
    sample_rows = [
        {"amount": 1200, "term": 12, "rate": 0.18},
        {"amount": 800, "term": 6, "rate": 0.22},
    ]

    def fake_llm(prompt: str) -> str:
        assert "amount" in prompt
        return json.dumps(
            {
                "features": [
                    {
                        "name": "amount_per_term",
                        "kind": "numeric",
                        "expression": "amount / term",
                        "rationale": "relative monthly burden",
                    },
                    {
                        "name": "rate_x_term",
                        "kind": "numeric",
                        "expression": "rate * term",
                        "rationale": "total rate exposure",
                    },
                ]
            }
        )

    plan = propose_features(schema, sample_rows, fake_llm, max_features=5)
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))

    # Materialize with the whitelist AST evaluator; no eval/exec is involved.
    frame = pd.DataFrame(sample_rows)
    augmented = materialize_plan(plan, frame)
    print("augmented columns:", list(augmented.columns))


def build_or_run(data: str, target: str | None, profile: int, run: bool) -> None:
    """Create a config; optionally execute all stages."""
    from kconv.api import build_project

    project_dir, config_path, config = build_project(
        Path(data),
        target=target,
        recipe="kconv/linear-stack",
        profile=profile,
        name="api-example",
    )
    print("project:", project_dir)
    print("config:", config_path)
    print("strategy:", config.strategy)
    print("models:", [model.name for model in config.models])
    print("runtime:", config.runtime.to_dict() if config.runtime else None)
    if run:
        context = run_dataset(
            Path(data),
            target=target,
            recipe="kconv/linear-stack",
            profile=profile,
            name="api-example-run",
            until="report",
        )
        print("run_dir:", context.run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="kconv API usage examples")
    parser.add_argument("--data", help="Dataset path; enables build/run example")
    parser.add_argument("--target", help="Target column for --data")
    parser.add_argument("--profile", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--run", action="store_true", help="Actually execute the pipeline")
    parser.add_argument(
        "--source",
        default="arnabbiswas1__kaggle_pipeline_tps_aug_22",
        help="Source id to inspect",
    )
    args = parser.parse_args()

    show_catalog()
    show_facade()
    show_source(args.source)
    show_llm_features()
    if args.data:
        build_or_run(args.data, args.target, args.profile, args.run)


if __name__ == "__main__":
    main()
