"""Recipe catalog: named professional methodologies that resolve to config defaults."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from kconv.config import EnsembleConfig, FeatureConfig, HPOConfig, ValidationConfig
from kconv.errors import RecipeError

_VALID_TASKS = frozenset({"binary", "multiclass", "regression"})
_ALLOWED_BLOCK_TYPES = frozenset(
    {
        "base",
        "target_encoding",
        "frequency_encoding",
        "hash_encoding",
        "group_statistics",
        "categorical_interaction",
        "missingness",
        "datetime_cyclical",
        "numeric_interactions",
        "clustering",
        "anomaly",
        "time_series",
        "row_statistics",
        "modulo",
        "fold_plan",
        "feature_selection",
        "knn_class_distance",
    }
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecipeDefaults(_Strict):
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: list[str] = Field(default_factory=lambda: ["lgbm"])
    model_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    hpo: HPOConfig = Field(default_factory=HPOConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)

    @field_validator("models")
    @classmethod
    def _models_nonempty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("defaults.models must contain at least one model")
        return value


class RecipeBlock(_Strict):
    """One feature block of an executable pipeline recipe."""

    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        if value not in _ALLOWED_BLOCK_TYPES:
            allowed = ", ".join(sorted(_ALLOWED_BLOCK_TYPES))
            raise ValueError(
                f"unknown block type '{value}'; allowed feature block types: {allowed}"
            )
        return value


class RecipePipeline(_Strict):
    """Declarative, executable feature-block plan for ``kind: pipeline``."""

    blocks: list[RecipeBlock] = Field(default_factory=list)

    @model_validator(mode="after")
    def _block_order(self) -> RecipePipeline:
        if self.blocks:
            if self.blocks[0].type != "base":
                raise ValueError("pipeline.blocks must start with the 'base' block")
            if sum(block.type == "base" for block in self.blocks) != 1:
                raise ValueError("pipeline.blocks must contain 'base' exactly once")
            duplicated = sorted({block.type for block in self.blocks if sum(item.type == block.type for item in self.blocks) > 1})
            if duplicated:
                raise ValueError(
                    "pipeline.blocks must contain each singleton block once: "
                    + ", ".join(duplicated)
                )
        return self


class Recipe(_Strict):
    id: str
    kind: Literal["preset", "pipeline"] = "preset"
    derived_from: str | None = None
    title: str
    author: str
    source: str
    license: str
    group: str = "general"
    tasks: list[str] = Field(default_factory=lambda: ["binary", "multiclass", "regression"])
    best_for: str
    defaults: RecipeDefaults = Field(default_factory=RecipeDefaults)
    pipeline: RecipePipeline | None = None
    attribution: str

    @field_validator("derived_from")
    @classmethod
    def _derived_from_nonempty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("derived_from must be a non-empty string when provided")
        return value

    @model_validator(mode="after")
    def _kind_consistency(self) -> Recipe:
        if self.kind == "preset" and self.pipeline is not None:
            raise ValueError(
                "kind 'preset' must not declare pipeline; set kind: pipeline to use blocks"
            )
        if self.kind == "pipeline" and (self.pipeline is None or not self.pipeline.blocks):
            raise ValueError("kind 'pipeline' requires a non-empty pipeline.blocks list")
        return self

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("id must not be empty")
        return value

    @field_validator("tasks")
    @classmethod
    def _tasks_valid(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("tasks must contain at least one task")
        unknown = sorted(set(value) - _VALID_TASKS)
        if unknown:
            raise ValueError(f"unknown task(s): {', '.join(unknown)}")
        return list(dict.fromkeys(value))


def _recipe_files(directory: Path) -> list[Path]:
    """Return top-level YAML files and ``<id>/recipe.yaml`` files."""
    if not directory.is_dir():
        return []
    paths = list(directory.glob("*.yaml"))
    paths.extend(directory.glob("**/recipe.yaml"))
    return sorted(set(paths), key=lambda path: (len(path.relative_to(directory).parts), str(path)))


def _package_files() -> list[Any]:
    return [p for p in resources.files("kconv.recipes").iterdir() if str(p).endswith(".yaml")]


def _candidate_paths(recipe_id: str) -> list[Path]:
    safe = recipe_id.replace("/", "__")
    candidates = [
        Path.cwd() / "recipes" / f"{safe}.yaml",
        Path.cwd() / "recipes" / f"{Path(recipe_id).name}.yaml",
        Path.cwd() / "recipes" / recipe_id / "recipe.yaml",
        Path.home() / ".kconv" / "recipes" / f"{safe}.yaml",
        Path.home() / ".kconv" / "recipes" / recipe_id / "recipe.yaml",
    ]
    return candidates


def _read_yaml(path_or_resource: Any) -> dict[str, Any]:
    try:
        text = path_or_resource.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
    except (OSError, yaml.YAMLError) as exc:
        raise RecipeError(f"Cannot read recipe: {exc}", "Check the YAML file.") from exc
    if not isinstance(raw, dict):
        raise RecipeError("Recipe YAML root must be a mapping.", "See docs/recipes.md for the schema.")
    return raw


def _validate(raw: dict[str, Any], origin: str) -> Recipe:
    try:
        return Recipe.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        raise RecipeError(
            f"Invalid recipe '{origin}' at '{loc}': {first.get('msg', 'invalid value')}",
            "Fix the recipe fields shown in docs/recipes.md.",
        ) from exc


def load_recipe(recipe_id: str) -> Recipe:
    if not recipe_id or not isinstance(recipe_id, str):
        raise RecipeError("Recipe id must be a non-empty string.", "Run 'kconv recipes' to list ids.")
    for path in _candidate_paths(recipe_id):
        if path.is_file():
            return _validate(_read_yaml(path), str(path))
    # Users may name files freely (for example ``recipes/my-team.yaml`` while
    # the recipe id is ``my-team/tabular``). Match by the id inside the file.
    for directory in (Path.cwd() / "recipes", Path.home() / ".kconv" / "recipes"):
        for path in _recipe_files(directory):
            recipe = _validate(_read_yaml(path), str(path))
            if recipe.id == recipe_id:
                return recipe
    safe = recipe_id.replace("/", "__")
    try:
        resource = resources.files("kconv.recipes").joinpath(f"{safe}.yaml")
        if resource.is_file():
            return _validate(_read_yaml(resource), f"kconv.recipes/{safe}.yaml")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise RecipeError(f"Recipe package is unavailable: {exc}", "Reinstall kconv.") from exc
    raise RecipeError(
        f"Unknown recipe '{recipe_id}'.",
        "Run 'kconv recipes' to list built-in recipes.",
    )


def list_recipes() -> list[Recipe]:
    found: dict[str, Recipe] = {}
    # Built-ins first, then user catalogs.  Later entries win, matching
    # load_recipe(), where cwd recipes override home and packaged recipes.
    for resource in _package_files():
        recipe = _validate(_read_yaml(resource), str(resource))
        found[recipe.id] = recipe
    home_dir = Path.home() / ".kconv" / "recipes"
    for path in _recipe_files(home_dir):
        recipe = _validate(_read_yaml(path), str(path))
        found[recipe.id] = recipe
    user_dir = Path.cwd() / "recipes"
    for path in _recipe_files(user_dir):
        recipe = _validate(_read_yaml(path), str(path))
        found[recipe.id] = recipe
    return sorted(found.values(), key=lambda r: r.id)


def recipe_ids() -> list[str]:
    return [recipe.id for recipe in list_recipes()]
