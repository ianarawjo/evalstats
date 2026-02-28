"""Core data types for benchpress."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BenchmarkResult:
    """Container for benchmark scores across templates and inputs.

    The fundamental input to all benchpress analyses. Wraps a score matrix
    where every template has been evaluated on every input (complete design).

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape (N_templates, M_inputs). All values must be
        finite numeric. If a 3D array of shape (N, M, K) is provided, it is
        stored as-is and must be aggregated before analysis.
    template_labels : list[str]
        Human-readable names for each template. Length must match axis 0.
    input_labels : list[str]
        Identifiers for each benchmark input. Length must match axis 1.
    evaluator_names : list[str], optional
        Names of each evaluator. Required if scores is 3D (axis 2).
    input_metadata : pd.DataFrame, optional
        Metadata for each input (e.g., category, difficulty). Length must
        match axis 1.
    baseline_template : str, optional
        Label of a designated baseline template for comparison.
    """

    scores: np.ndarray
    template_labels: list[str]
    input_labels: list[str]
    evaluator_names: list[str] = field(default_factory=lambda: ["score"])
    input_metadata: Optional[pd.DataFrame] = None
    baseline_template: Optional[str] = None

    def __post_init__(self):
        self.scores = np.asarray(self.scores, dtype=np.float64)
        self._validate()

    def _validate(self):
        s = self.scores
        if s.ndim == 2:
            n_templates, n_inputs = s.shape
        elif s.ndim == 3:
            n_templates, n_inputs, n_evals = s.shape
            if len(self.evaluator_names) != n_evals:
                raise ValueError(
                    f"evaluator_names length ({len(self.evaluator_names)}) "
                    f"does not match scores axis 2 ({n_evals})"
                )
        else:
            raise ValueError(
                f"scores must be 2D (N, M) or 3D (N, M, K), got {s.ndim}D"
            )

        if len(self.template_labels) != n_templates:
            raise ValueError(
                f"template_labels length ({len(self.template_labels)}) "
                f"does not match scores axis 0 ({n_templates})"
            )
        if len(self.input_labels) != n_inputs:
            raise ValueError(
                f"input_labels length ({len(self.input_labels)}) "
                f"does not match scores axis 1 ({n_inputs})"
            )
        if len(self.template_labels) != len(set(self.template_labels)):
            raise ValueError("template_labels must be unique")
        if len(self.input_labels) != len(set(self.input_labels)):
            raise ValueError("input_labels must be unique")

        if not np.all(np.isfinite(s)):
            raise ValueError("scores contain NaN or infinite values")

        if self.input_metadata is not None:
            if len(self.input_metadata) != n_inputs:
                raise ValueError(
                    f"input_metadata length ({len(self.input_metadata)}) "
                    f"does not match number of inputs ({n_inputs})"
                )

        if self.baseline_template is not None:
            if self.baseline_template not in self.template_labels:
                raise ValueError(
                    f"baseline_template '{self.baseline_template}' "
                    f"not found in template_labels"
                )

        # Warnings
        for i, label in enumerate(self.template_labels):
            if s.ndim == 2:
                row = s[i]
            else:
                row = s[i].mean(axis=-1)  # average across evaluators
            if np.std(row) == 0:
                import warnings
                warnings.warn(
                    f"Template '{label}' has zero variance across inputs "
                    f"(all scores identical). This may indicate a problem.",
                    stacklevel=2,
                )

    @property
    def n_templates(self) -> int:
        return self.scores.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.scores.shape[1]

    @property
    def is_aggregated(self) -> bool:
        """Whether scores are already 2D (aggregated across evaluators)."""
        return self.scores.ndim == 2

    def get_2d_scores(self) -> np.ndarray:
        """Return 2D score matrix, averaging across evaluators if needed."""
        if self.scores.ndim == 2:
            return self.scores
        return self.scores.mean(axis=2)

    def template_index(self, label: str) -> int:
        """Get the index of a template by label."""
        try:
            return self.template_labels.index(label)
        except ValueError:
            raise KeyError(f"Template '{label}' not found")
