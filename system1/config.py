"""Load and validate config/pipeline.yaml.

Kept free of torch/laya imports so tests and CI can read the config on any machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "pipeline.yaml"

REQUIRED_QUESTIONS = ("guardrail", "classification", "deliberation", "grounded", "declined")
VALID_PRIMITIVES = ("noul", "choice", "score")


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]

    @property
    def llm(self) -> dict[str, Any]:
        return self.raw["models"]["llm"]

    @property
    def laya(self) -> dict[str, Any]:
        return self.raw["models"]["laya"]

    @property
    def server(self) -> dict[str, Any]:
        return self.raw["server"]

    @property
    def questions(self) -> dict[str, Any]:
        return self.raw["questions"]

    @property
    def policy(self) -> dict[str, Any]:
        return self.raw["policy"]

    def gate_questions(self) -> dict[str, Any]:
        """The three questions answered in one forward pass on the user's request."""
        return {k: self.questions[k] for k in ("guardrail", "classification", "deliberation")}

    def verify_questions(self) -> dict[str, Any]:
        """The verifier questions, answered against a draft instead of the request."""
        return {k: self.questions[k] for k in ("grounded", "declined")}


def _validate(cfg: Config) -> None:
    questions = cfg.questions
    missing = [q for q in REQUIRED_QUESTIONS if q not in questions]
    if missing:
        raise ValueError(f"{cfg.path}: missing questions {missing}")

    for name, q in questions.items():
        kind = q.get("type")
        if kind not in VALID_PRIMITIVES:
            raise ValueError(f"{cfg.path}: question {name!r} has unknown type {kind!r}")
        if kind == "choice" and not isinstance(q.get("criteria"), dict):
            raise ValueError(f"{cfg.path}: choice question {name!r} needs a criteria mapping")
        if kind == "score" and not isinstance(q.get("criteria"), list):
            raise ValueError(f"{cfg.path}: score question {name!r} needs a criteria list")
        if not q.get("instructions"):
            raise ValueError(f"{cfg.path}: question {name!r} has no instructions")

    # Every policy threshold is compared against a probability, so a threshold
    # outside [0,1] would silently make the branch unreachable.
    for name, value in cfg.policy.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{cfg.path}: policy.{name} must be a probability, got {value!r}")


def load(path: str | os.PathLike[str] | None = None) -> Config:
    resolved = Path(path) if path else DEFAULT_CONFIG
    with resolved.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{resolved}: top level must be a mapping")
    cfg = Config(path=resolved, raw=raw)
    _validate(cfg)
    return cfg
