"""Typed errors with actionable, user-facing messages."""

from __future__ import annotations


class KconvError(Exception):
    """Base class for all expected kconv failures."""

    exit_code = 1

    def __init__(self, message: str, hint: str | None = None) -> None:
        self.message = message
        self.hint = hint
        super().__init__(f"{message}\nHint: {hint}" if hint else message)


class ConfigError(KconvError):
    exit_code = 2


class DataError(KconvError):
    exit_code = 2


class RecipeError(KconvError):
    exit_code = 2


class ModelError(KconvError):
    exit_code = 3


class StageError(KconvError):
    exit_code = 3


class SubmissionError(KconvError):
    exit_code = 3
