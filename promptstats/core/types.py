"""Core data types for benchpress."""

from __future__ import annotations

import warnings
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


@dataclass
class MultiModelBenchmark:
    """Benchmark scores across multiple models, templates, and inputs.

    Extends BenchmarkResult to a four-dimensional structure that adds a
    model axis as the outermost dimension. Every (model, template) pair
    must be evaluated on the same complete set of inputs.

    Parameters
    ----------
    scores : np.ndarray
        Score tensor of shape (N_models, N_templates, M_inputs) or
        (N_models, N_templates, M_inputs, K_evaluators). All values
        must be finite numeric.
    model_labels : list[str]
        Human-readable names for each model. Length must match axis 0.
        Must have at least 2 entries.
    template_labels : list[str]
        Human-readable names for each prompt template. Length must
        match axis 1.
    input_labels : list[str]
        Identifiers for each benchmark input. Length must match axis 2.
    evaluator_names : list[str], optional
        Names of each evaluator. Required if scores is 4D (axis 3).
    input_metadata : pd.DataFrame, optional
        Metadata for each input (e.g., category, difficulty). Length
        must match axis 2.
    """

    scores: np.ndarray
    model_labels: list[str]
    template_labels: list[str]
    input_labels: list[str]
    evaluator_names: list[str] = field(default_factory=lambda: ["score"])
    input_metadata: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.scores = np.asarray(self.scores, dtype=np.float64)
        self._validate()

    def _validate(self):
        s = self.scores
        if s.ndim == 3:
            n_models, n_templates, n_inputs = s.shape
        elif s.ndim == 4:
            n_models, n_templates, n_inputs, n_evals = s.shape
            if len(self.evaluator_names) != n_evals:
                raise ValueError(
                    f"evaluator_names length ({len(self.evaluator_names)}) "
                    f"does not match scores axis 3 ({n_evals})"
                )
        else:
            raise ValueError(
                f"scores must be 3D (N_models, N_templates, M_inputs) or "
                f"4D (N_models, N_templates, M_inputs, K_evaluators), got {s.ndim}D"
            )

        if n_models < 2:
            raise ValueError(
                f"MultiModelBenchmark requires at least 2 models; got {n_models}. "
                "Use BenchmarkResult for single-model benchmarks."
            )

        if len(self.model_labels) != n_models:
            raise ValueError(
                f"model_labels length ({len(self.model_labels)}) "
                f"does not match scores axis 0 ({n_models})"
            )
        if len(self.template_labels) != n_templates:
            raise ValueError(
                f"template_labels length ({len(self.template_labels)}) "
                f"does not match scores axis 1 ({n_templates})"
            )
        if len(self.input_labels) != n_inputs:
            raise ValueError(
                f"input_labels length ({len(self.input_labels)}) "
                f"does not match scores axis 2 ({n_inputs})"
            )
        if len(self.model_labels) != len(set(self.model_labels)):
            raise ValueError("model_labels must be unique")
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

        # Warn about zero-variance (model, template) cells.
        scores_3d = s if s.ndim == 3 else s.mean(axis=-1)
        for m_idx, model_label in enumerate(self.model_labels):
            for t_idx, template_label in enumerate(self.template_labels):
                if np.std(scores_3d[m_idx, t_idx]) == 0:
                    warnings.warn(
                        f"Template '{template_label}' for model '{model_label}' "
                        f"has zero variance across inputs (all scores identical). "
                        f"This may indicate a problem.",
                        stacklevel=2,
                    )

    # ------------------------------------------------------------------
    # Shape properties
    # ------------------------------------------------------------------

    @property
    def n_models(self) -> int:
        return self.scores.shape[0]

    @property
    def n_templates(self) -> int:
        return self.scores.shape[1]

    @property
    def n_inputs(self) -> int:
        return self.scores.shape[2]

    @property
    def is_aggregated(self) -> bool:
        """True when scores are 3D (no evaluator axis)."""
        return self.scores.ndim == 3

    # ------------------------------------------------------------------
    # Slicing helpers
    # ------------------------------------------------------------------

    def get_model_result(self, model: str) -> BenchmarkResult:
        """Slice out one model's scores as a BenchmarkResult.

        Parameters
        ----------
        model : str
            Label of the model to extract.

        Returns
        -------
        BenchmarkResult
            Shape (N_templates, M_inputs) or (N_templates, M_inputs, K_evaluators).
        """
        try:
            idx = self.model_labels.index(model)
        except ValueError:
            raise KeyError(f"Model '{model}' not found in model_labels")
        return BenchmarkResult(
            scores=self.scores[idx],
            template_labels=self.template_labels,
            input_labels=self.input_labels,
            evaluator_names=self.evaluator_names,
            input_metadata=self.input_metadata,
        )

    def get_flat_result(self, sep: str = " / ") -> BenchmarkResult:
        """Flatten all (model, template) pairs into a single template axis.

        Returns a BenchmarkResult whose 'templates' are the N_models *
        N_templates cross-product pairs, ordered model-first (all templates
        for model 0, then all templates for model 1, …). Labels are
        formatted as ``"<model><sep><template>"``.

        This is the basis for cross-model ranking: all existing pairwise
        and rank-distribution machinery operates on the flat pairs directly,
        letting the best (model, template) pair emerge naturally.

        Parameters
        ----------
        sep : str
            Separator between model and template name in compound labels
            (default ' / ').

        Returns
        -------
        BenchmarkResult
            Shape (N_models * N_templates, M_inputs[, K_evaluators]).
        """
        n_flat = self.n_models * self.n_templates
        if self.scores.ndim == 3:
            flat_scores = self.scores.reshape(n_flat, self.n_inputs)
        else:
            flat_scores = self.scores.reshape(n_flat, self.n_inputs, self.scores.shape[3])

        flat_labels = [
            f"{m}{sep}{t}"
            for m in self.model_labels
            for t in self.template_labels
        ]
        return BenchmarkResult(
            scores=flat_scores,
            template_labels=flat_labels,
            input_labels=self.input_labels,
            evaluator_names=self.evaluator_names,
            input_metadata=self.input_metadata,
        )

    def get_model_mean_result(self) -> BenchmarkResult:
        """Aggregate each model's scores by averaging across templates.

        Returns a BenchmarkResult where each 'template' represents one
        model, scored by its mean performance over all prompt templates.
        Useful for an overall model-level comparison that is independent
        of any single prompt choice.

        Returns
        -------
        BenchmarkResult
            Shape (N_models, M_inputs[, K_evaluators]).
        """
        return BenchmarkResult(
            scores=self.scores.mean(axis=1),
            template_labels=self.model_labels,
            input_labels=self.input_labels,
            evaluator_names=self.evaluator_names,
            input_metadata=self.input_metadata,
        )
