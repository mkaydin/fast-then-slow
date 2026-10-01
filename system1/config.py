"""Load and validate the pipeline configuration.

Two files, deliberately separate:

  pipeline.yaml    portable. Which question plays which role, the threshold each is
                   judged at, the question wording, the uncertainty floor. This is
                   the part a project owns and a fine-tune is trained against.

  deployment.yaml  machine-specific. Which checkpoints, which GPUs, how much VRAM,
                   which ports. Gitignored, with deployment.example.yaml committed.

Keeping them apart is the difference between a repo that ports to the next machine
and one that does not. Kept free of torch/laya so tests and CI can read config
anywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .policy import ALL_ROLES, GATE_ROLES, SYSTEM1, THINK, VERIFY_ROLES, Decision

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_PIPELINE = CONFIG_DIR / "pipeline.yaml"
DEFAULT_DEPLOYMENT = CONFIG_DIR / "deployment.yaml"

VALID_PRIMITIVES = ("noul", "choice", "score")


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]
    deployment: dict[str, Any]
    decisions: tuple[Decision, ...]
    uncertain_below: float

    # --- policy surface -----------------------------------------------------

    @property
    def questions(self) -> dict[str, Any]:
        return self.raw.get("questions") or {}

    @property
    def by_role(self) -> dict[str, Decision]:
        """Role -> Decision. Built once; policy lookup is on the request path."""
        return {decision.role: decision for decision in self.decisions}

    @property
    def gate_questions(self) -> dict[str, Any]:
        """The questions answered in one forward pass on the user's request."""
        return {
            d.question: self.questions[d.question]
            for d in self.decisions
            if d.role in GATE_ROLES
        }

    @property
    def verify_questions(self) -> dict[str, Any]:
        """The questions answered in one forward pass against a draft."""
        return {
            d.question: self.questions[d.question]
            for d in self.decisions
            if d.role in VERIFY_ROLES
        }

    @property
    def system1_questions(self) -> list[str]:
        """Question ids surfaced in a System-1-only reply, in config order."""
        return [
            d.question for d in self.decisions if d.role in (SYSTEM1, THINK)
        ]

    @property
    def guardrail_roles(self) -> tuple[str, ...]:
        """Roles that stop a request before the model is called."""
        return tuple(d.question for d in self.decisions if d.role == "block")

    # --- deployment surface -------------------------------------------------

    @property
    def llm(self) -> dict[str, Any]:
        return self.deployment.get("models", {}).get("llm", {})

    @property
    def laya(self) -> dict[str, Any]:
        return self.deployment.get("models", {}).get("laya", {})

    @property
    def server(self) -> dict[str, Any]:
        return self.deployment.get("server", {})


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def _validate_questions(path: Path, questions: dict[str, Any]) -> None:
    for name, question in questions.items():
        kind = question.get("type")
        if kind not in VALID_PRIMITIVES:
            raise ValueError(f"{path}: question {name!r} has unknown type {kind!r}")
        if kind == "choice" and not isinstance(question.get("criteria"), dict):
            raise ValueError(f"{path}: choice question {name!r} needs a criteria mapping")
        if kind == "score" and not isinstance(question.get("criteria"), list):
            raise ValueError(f"{path}: score question {name!r} needs a criteria list")
        if not question.get("instructions"):
            raise ValueError(f"{path}: question {name!r} has no instructions")


def _parse_decisions(path: Path, raw: Any, questions: dict[str, Any]) -> tuple[Decision, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: `decisions` must be a non-empty list")

    parsed: list[Decision] = []
    seen: set[str] = set()
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: decisions[{position}] must be a mapping")

        role = entry.get("role")
        if role not in ALL_ROLES:
            raise ValueError(
                f"{path}: decisions[{position}] has role {role!r}; "
                f"known roles are {sorted(ALL_ROLES)}"
            )
        if role in seen:
            raise ValueError(f"{path}: role {role!r} is configured twice")
        seen.add(role)

        question = entry.get("question")
        if question not in questions:
            raise ValueError(
                f"{path}: decisions[{position}] references unknown question {question!r}"
            )

        threshold = entry.get("threshold")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ValueError(
                f"{path}: decisions[{position}] threshold must be a probability, "
                f"got {threshold!r}"
            )
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(
                f"{path}: decisions[{position}] threshold must be within [0,1], "
                f"got {threshold!r}"
            )
        parsed.append(Decision(role=role, question=question, threshold=float(threshold)))
    return tuple(parsed)


def load(
    pipeline: str | os.PathLike[str] | None = None,
    deployment: str | os.PathLike[str] | None = None,
) -> Config:
    pipeline_path = Path(pipeline) if pipeline else DEFAULT_PIPELINE
    deployment_path = Path(deployment) if deployment else DEFAULT_DEPLOYMENT

    raw = _read(pipeline_path)
    questions = raw.get("questions") or {}
    if not questions:
        raise ValueError(f"{pipeline_path}: no questions defined")
    _validate_questions(pipeline_path, questions)

    decisions = _parse_decisions(pipeline_path, raw.get("decisions"), questions)

    floor = raw.get("uncertain_below", 0.5)
    if not isinstance(floor, (int, float)) or isinstance(floor, bool) or not 0.0 <= floor <= 1.0:
        raise ValueError(f"{pipeline_path}: uncertain_below must be a probability, got {floor!r}")

    # A missing deployment file is not fatal: the policy half is portable on its own.
    deployment_raw = _read(deployment_path) if deployment_path.exists() else {}

    return Config(
        path=pipeline_path,
        raw=raw,
        deployment=deployment_raw,
        decisions=decisions,
        uncertain_below=float(floor),
    )