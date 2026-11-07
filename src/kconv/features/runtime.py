"""Portable, dependency-light feature transform for inference bundles.

This module deliberately imports only numpy and pandas.  ``export_inference``
copies it into the bundle as ``feature_runtime.py`` and stores a plain state
dictionary produced by :meth:`kconv.features.pipeline.FeatureBuilder.to_runtime_state`.
The generated bundle therefore does not need the kconv package installed.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _as_str_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__missing__").astype(str)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


class FeatureRuntime:
    """Apply a fitted FeatureBuilder state to raw rows."""

    def __init__(self, state: dict) -> None:
        self.state = state
        self.family = str(state.get("family", "tree"))
        self.feature_columns = [str(value) for value in state.get("feature_columns", [])]
        self.numeric = [str(value) for value in state.get("numeric", [])]
        self.categorical = [str(value) for value in state.get("categorical", [])]
        self.datetime = [str(value) for value in state.get("datetime", [])]
        self.text = [str(value) for value in state.get("text", [])]
        self.frequency = [str(value) for value in state.get("frequency", [])]
        self.hash_encode = [str(value) for value in state.get("hash_encode", [])]
        self.hash_features = int(state.get("hash_features", 16))
        self.target_encode = [str(value) for value in state.get("target_encode", [])]
        self.interactions = list(state.get("interactions", []))
        self.keep = [str(value) for value in state.get("keep", [])]
        self.numeric_fill = dict(state.get("numeric_fill", {}))
        self.numeric_mean = dict(state.get("numeric_mean", {}))
        self.numeric_std = dict(state.get("numeric_std", {}))
        self.cat_categories = dict(state.get("cat_categories", {}))
        self.cat_maps = dict(state.get("cat_maps", {}))
        self.datetime_fill = dict(state.get("datetime_fill", {}))
        self.text_fill = dict(state.get("text_fill", {}))
        self.freq_maps = dict(state.get("freq_maps", {}))
        self.te_maps = dict(state.get("te_maps", {}))
        self.te_prior = dict(state.get("te_prior", {}))
        self.interaction_maps = dict(state.get("interaction_maps", {}))
        self.interaction_prior = dict(state.get("interaction_prior", {}))
        self.missingness = dict(state.get("missingness", {}))
        self.datetime_cyclical = dict(state.get("datetime_cyclical", {}))
        self.row_statistics = bool(state.get("row_statistics", False))
        self.row_statistics_stats = [str(value) for value in state.get("row_statistics_stats", [])]
        self.modulo_features = bool(state.get("modulo_features", False))
        self.modulo_moduli = [int(value) for value in state.get("modulo_moduli", [])]
        self.modulo_steps = [int(value) for value in state.get("modulo_steps", [])]
        self.scale = str(state.get("scale", "auto"))

    @staticmethod
    def _parse_datetime(values: pd.Series) -> pd.Series:
        try:
            return pd.to_datetime(values, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(values, errors="coerce")

    def _numeric_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        scale = self.scale == "standard" or (self.scale == "auto" and self.family == "linear")
        for column in self.numeric:
            values = _safe_numeric(df[column]).fillna(self.numeric_fill[column])
            if scale:
                values = (values - self.numeric_mean[column]) / self.numeric_std[column]
            out[column] = values
        return out

    def _categorical_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for column in self.categorical:
            values = _as_str_series(df[column])
            if self.family == "linear":
                for category in self.cat_categories.get(column, []):
                    out[f"{column}__{category}"] = (values == category).astype(float)
            else:
                out[column] = values.map(self.cat_maps.get(column, {})).fillna(-1).astype(float)
        return out

    def _datetime_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for column in self.datetime:
            parsed = self._parse_datetime(df[column])
            ordinal = parsed.map(lambda value: value.toordinal() if pd.notna(value) else np.nan).astype(float)
            ordinal = ordinal.fillna(self.datetime_fill.get(column, 0.0))
            out[f"{column}__ordinal"] = ordinal
            out[f"{column}__year"] = parsed.dt.year.astype(float).fillna(0.0)
            out[f"{column}__month"] = parsed.dt.month.astype(float).fillna(0.0)
            out[f"{column}__day"] = parsed.dt.day.astype(float).fillna(0.0)
            out[f"{column}__dayofweek"] = parsed.dt.dayofweek.astype(float).fillna(0.0)
        return out

    def _text_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for column in self.text:
            values = _as_str_series(df[column])
            length = values.str.len().astype(float)
            words = values.str.split().str.len().astype(float)
            out[f"{column}__len"] = length.fillna(self.text_fill.get(f"{column}__len", 0.0))
            out[f"{column}__words"] = words.fillna(self.text_fill.get(f"{column}__words", 0.0))
        return out

    def _frequency_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for column in self.frequency:
            values = _as_str_series(df[column])
            out[f"{column}__freq"] = values.map(self.freq_maps.get(column, {})).fillna(0.0).astype(float)
        return out

    def _hash_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if not self.hash_encode:
            return {}
        matrix = np.zeros((len(df), self.hash_features), dtype=float)
        positions = np.arange(len(df))
        for column in self.hash_encode:
            values = _as_str_series(df[column])
            hashed = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64)
            salt = int.from_bytes(hashlib.blake2b(column.encode("utf-8"), digest_size=8).digest(), "big")
            buckets = ((hashed ^ np.uint64(salt)) % np.uint64(self.hash_features)).astype(int)
            matrix[positions, buckets] += 1.0
        return {
            f"hash__{index}": pd.Series(matrix[:, index], index=df.index, dtype=float)
            for index in range(self.hash_features)
        }

    def _target_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for column in self.target_encode:
            values = _as_str_series(df[column])
            out[f"{column}__target"] = (
                values.map(self.te_maps.get(column, {}))
                .fillna(self.te_prior.get(column, 0.0))
                .astype(float)
            )
        return out

    def _interaction_series(self, df: pd.DataFrame, columns: list[str]) -> pd.Series:
        parts = [_as_str_series(df[column]) for column in columns]
        combined = parts[0]
        for part in parts[1:]:
            combined = combined + "\x1f" + part
        return combined.astype(str)

    def _interaction_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for interaction in self.interactions:
            columns = [str(value) for value in interaction.get("columns", [])]
            encoding = str(interaction.get("encoding", "frequency"))
            name = "__x__".join(columns) + f"__{encoding}"
            values = self._interaction_series(df, columns)
            if encoding == "frequency":
                out[name] = values.map(self.interaction_maps.get(name, {})).fillna(0.0).astype(float)
            else:
                out[name] = (
                    values.map(self.interaction_maps.get(name, {}))
                    .fillna(self.interaction_prior.get(name, 0.0))
                    .astype(float)
                )
        return out

    def _missingness_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if not self.missingness.get("enabled") or not self.missingness.get("columns"):
            return {}
        out: dict[str, pd.Series] = {}
        selected = [str(value) for value in self.missingness.get("columns", [])]
        if self.missingness.get("row_count"):
            out["missing__count"] = df.loc[:, selected].isna().sum(axis=1).astype(float)
        if self.missingness.get("indicators"):
            for column in selected:
                out[f"{column}__missing"] = df[column].isna().astype(float)
        if self.missingness.get("zero_count"):
            numeric_selected = [column for column in selected if column in self.numeric]
            if numeric_selected:
                matrix = pd.concat([_safe_numeric(df[column]) for column in numeric_selected], axis=1)
                out["zero__count"] = (matrix == 0).sum(axis=1).astype(float)
        return out

    def _datetime_cyclical_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if not self.datetime_cyclical.get("enabled") or not self.datetime_cyclical.get("columns"):
            return {}
        periods = {
            "month": 12.0,
            "dayofweek": 7.0,
            "dayofyear": 365.25,
            "hour": 24.0,
        }
        out: dict[str, pd.Series] = {}
        for column in [str(value) for value in self.datetime_cyclical.get("columns", [])]:
            parsed = self._parse_datetime(df[column])
            for period in [str(value) for value in self.datetime_cyclical.get("periods", [])]:
                if period not in periods:
                    continue
                values = getattr(parsed.dt, period).astype(float)
                radians = 2.0 * np.pi * values / periods[period]
                out[f"{column}__{period}_sin"] = pd.Series(
                    np.sin(radians), index=df.index, dtype=float
                ).fillna(0.0)
                out[f"{column}__{period}_cos"] = pd.Series(
                    np.cos(radians), index=df.index, dtype=float
                ).fillna(0.0)
        return out

    def _row_statistics_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if not self.row_statistics or not self.numeric:
            return {}
        matrix = pd.concat([_safe_numeric(df[column]) for column in self.numeric], axis=1)
        matrix.columns = self.numeric
        mapping = {
            "mean": matrix.mean(axis=1),
            "std": matrix.std(axis=1, ddof=0),
            "min": matrix.min(axis=1),
            "max": matrix.max(axis=1),
            "median": matrix.median(axis=1),
        }
        return {
            f"row_{stat}": mapping[stat].fillna(0.0).astype(float)
            for stat in self.row_statistics_stats
            if stat in mapping
        }

    def _modulo_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if not self.modulo_features:
            return {}
        out: dict[str, pd.Series] = {}
        for column in self.numeric:
            values = _safe_numeric(df[column])
            for modulus in self.modulo_moduli:
                out[f"{column}__mod_{modulus}"] = (values % modulus).fillna(0.0).astype(float)
            for step in self.modulo_steps:
                out[f"{column}__div_{step}"] = np.floor(values / step).fillna(0.0).astype(float)
        return out

    def _assemble(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in df.columns]
        if missing:
            raise ValueError(
                "Test data is missing feature columns: " + ", ".join(missing)
            )
        parts: list[pd.DataFrame] = []
        for frame in (
            self._numeric_frame(df),
            self._categorical_frame(df),
            self._datetime_frame(df),
            self._datetime_cyclical_frame(df),
            self._text_frame(df),
            self._frequency_frame(df),
            self._hash_frame(df),
            self._target_frame(df),
            self._interaction_frame(df),
            self._missingness_frame(df),
            self._row_statistics_frame(df),
            self._modulo_frame(df),
        ):
            if frame:
                parts.append(pd.DataFrame(frame, index=df.index))
        if not parts:
            raise ValueError("No usable features were produced from the test data.")
        matrix = pd.concat(parts, axis=1)
        if matrix.columns.duplicated().any():
            duplicates = sorted(set(matrix.columns[matrix.columns.duplicated()].astype(str)))
            raise ValueError("Duplicate engineered feature names: " + ", ".join(duplicates))
        return matrix.reindex(sorted(matrix.columns), axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        matrix = self._assemble(df)
        return matrix.reindex(columns=self.keep, fill_value=0.0).reset_index(drop=True)
