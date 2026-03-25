"""Logger module to general training and inference logging."""

import logging
import secrets
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from pathlib import Path
from typing import Any, TypedDict, TypeVar, override

import pandas as pd
from torch import nn

from birdy.utils import ensure_dir_exist

type Step_T = dict[str, int] | Sequence[str] | None
type MetricUpdate_T = dict[str, float]
type Artifact_T = nn.Module | pd.DataFrame


class MetricItem(TypedDict):
    """Metric item including information about metric and steps when it was logged."""

    data: dict[str, float]
    step: dict[str, int]


class MetricLogger(ABC):
    """Base class for metric loggers."""

    def __init__(
        self,
        workspace_dir: Path,
        default_step: str | None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize metric logger.

        Args:
            workspace_dir: path to workspace directory
            default_step: default step (this would be used as index when saving)
        """
        self._workspace_dir: Path = workspace_dir
        self._default_step: str = default_step if default_step is not None else "step"

        ensure_dir_exist(self._workspace_dir)

    @abstractmethod
    def log_metrics(self, metric_update: MetricUpdate_T, step_update: Step_T = None) -> MetricItem:
        """Log metrics at specific steps.

        Args:
            metric_update: a dict containing which metric to update
            step_update: which step to update.
                Omit this would fall back to increase the default step by 1.
                If step_update is a sequence, each step in the set would be increased by 1.
                If step_update is a dict, metric would be updated at the specific step in the dict.

        Return:
            true if the logging is success
        """

    @abstractmethod
    def log_artifact(self, artifact: Artifact_T, name: str | None = None) -> str:
        """Log metrics at specific steps.

        Args:
            artifact: the artifact to log.
            name: name of the artifact. If omitted, a random string would be used as name.

        Return:
            name of the artifact
        """

    @abstractmethod
    def load_artifact(self, name: str) -> Artifact_T:
        """Log metrics at specific steps.

        Args:
            name: name of the artifact
        """


class LocalMetricLogger(MetricLogger):
    """Local metric logger. This would log the metric and file directly to disk."""

    @override
    def __init__(self, workspace_dir: Path, default_step: str | None = None) -> None:
        super().__init__(workspace_dir, default_step)

        self._step_dict: dict[str, int] = {self._default_step: -1}

        self._metrics_items: dict[int, MetricItem] = {}
        self._metrics_rows: dict[int, dict[str, float | int]] = {}
        self._artifact_dict: dict[str, str] = {}

        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    def _save_metric(self) -> None:
        df = pd.DataFrame(
            list(self._metrics_rows.values()),
            index=list(self._metrics_rows.keys()),
        )
        df.to_csv(self._workspace_dir / "metric.csv")

    @override
    def log_metrics(self, metric_update: MetricUpdate_T, step_update: Step_T = None) -> MetricItem:
        if step_update is not None:
            for su in step_update:
                if su not in self._step_dict:
                    self._step_dict[su] = -1

        if step_update is None:
            self._step_dict[self._default_step] += 1
            metric_step = self._step_dict
        elif isinstance(step_update, Sequence):
            step_set: set[str] = {self._default_step}
            step_set.update(step_update)
            for ss in step_set:
                self._step_dict[ss] += 1
            metric_step = self._step_dict
        else:
            if self._default_step not in step_update:
                self._step_dict[self._default_step] += 1

            for su_k in step_update:
                self._step_dict[su_k] = max(self._step_dict[su_k], step_update[su_k])

            metric_step = {**self._step_dict, **step_update}

        cur_step = self._step_dict[self._default_step]

        metric_item = MetricItem(data=metric_update, step=metric_step)

        self._metrics_items[cur_step] = metric_item
        self._metrics_rows[cur_step] = {**metric_step, **metric_update}

        log_parts: list[str] = []
        for k, v in sorted(metric_update.items(), key=lambda x: x[0]):
            log_parts.append(f"[{k}: {v:.4f}]")
        self._logger.info("Step %d: %s", cur_step, " ".join(log_parts))
        self._save_metric()

        return copy(metric_item)

    @override
    def log_artifact(self, artifact: Artifact_T, name: str | None = None) -> str:
        return super().log_artifact(artifact, name)

    @override
    def load_artifact(self, name: str) -> Artifact_T:
        return super().load_artifact(name)
