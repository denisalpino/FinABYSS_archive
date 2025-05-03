from __future__ import annotations

from collections.abc import KeysView
import functools
import math

import numpy as np
from scipy import stats

import optuna
from optuna.pruners import PercentilePruner, PatientPruner
from optuna.study._study_direction import StudyDirection
from optuna.trial._state import TrialState
from optuna.logging import _get_library_root_logger
logger = _get_library_root_logger()


def _get_percentile_intermediate_result_over_trials(
    completed_trials: list["optuna.trial.FrozenTrial"],
    direction: StudyDirection,
    step: int,
    steps_weights: list,
    percentile: float,
    n_min_trials: int
) -> float:
    if len(completed_trials) == 0:
        raise ValueError("No trials have been completed.")

    intermediate_values = []
    for t in completed_trials:
        if step in t.intermediate_values:
            # Collect all intermediate values untill current step (inclusively for steps started from 1)
            intermediate_values_till_step = list(t.intermediate_values.values())[:step]
            weighted_average = 0

            for val, weight in zip(intermediate_values_till_step, steps_weights):
                weighted_average += weight * val

            intermediate_values.append(weighted_average)

    if len(intermediate_values) < n_min_trials:
        return math.nan

    if direction == StudyDirection.MAXIMIZE:
        percentile = 100 - percentile

    p = float(np.nanpercentile(np.array(intermediate_values, dtype=float), percentile,))
    return p


def _is_first_in_interval_step(
    step: int, intermediate_steps: KeysView[int], n_warmup_steps: int, interval_steps: int
) -> bool:
    nearest_lower_pruning_step = (
        step - n_warmup_steps
    ) // interval_steps * interval_steps + n_warmup_steps
    assert nearest_lower_pruning_step >= 0

    # `intermediate_steps` may not be sorted so we must go through all elements.
    second_last_step = functools.reduce(
        lambda second_last_step, s: s if s > second_last_step and s != step else second_last_step,
        intermediate_steps,
        -1,
    )

    return second_last_step < nearest_lower_pruning_step


class AdaptiveStablePercentilePruner(PercentilePruner):
    def __init__(
        self,
        percentiles: list[list],
        steps_weights: list,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        *,
        n_min_trials: int = 1,
    ) -> None:
        """
        percentiles starts work from `n_startup_trials + 1` trial, so there must be `n_trials - n_startup_trials` columns.
        The shape of `prcentiles` must be `n_steps X n_trials`
        """
        if n_startup_trials < 0:
            raise ValueError(
                "Number of startup trials cannot be negative but got {}.".format(n_startup_trials)
            )
        if n_warmup_steps < 0:
            raise ValueError(
                "Number of warmup steps cannot be negative but got {}.".format(n_warmup_steps)
            )
        if interval_steps < 1:
            raise ValueError(
                "Pruning interval steps must be at least 1 but got {}.".format(interval_steps)
            )
        if n_min_trials < 1:
            raise ValueError(
                "Number of trials for pruning must be at least 1 but got {}.".format(n_min_trials)
            )

        self._percentiles = np.asarray(percentiles)
        self._n_trials = len(percentiles[0]) + n_startup_trials
        self._steps_weights = steps_weights
        self._n_startup_trials = n_startup_trials
        self._n_warmup_steps = n_warmup_steps
        self._interval_steps = interval_steps
        self._n_min_trials = n_min_trials

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        n_trials = len(completed_trials)

        if n_trials == 0:
            return False

        if n_trials < self._n_startup_trials:
            return False

        step = trial.last_step
        if step is None:
            return False

        n_warmup_steps = self._n_warmup_steps
        if step < n_warmup_steps:
            return False

        if not _is_first_in_interval_step(
            step, trial.intermediate_values.keys(), n_warmup_steps, self._interval_steps
        ):
            return False

        direction = study.direction

        values = np.asarray(list(trial.intermediate_values.values()), dtype=float)
        # Normalize weights untill step (inclusively for steps started from 1)
        weights = np.asarray(self._steps_weights[:step], dtype=float)
        normalized_weights = weights / np.sum(weights)
        weighted_mean_intermediate_result = np.average(values, weights=normalized_weights)
        logger.info(f"Trial {trial.number} WEIGHTED CUMMULATIVE METRIC for step {step} = {weighted_mean_intermediate_result:.4f}")

        if math.isnan(weighted_mean_intermediate_result):
            return True

        current_percentile = self._percentiles[
            step - n_warmup_steps,
            min(trial.number, self._n_trials - 1) - self._n_startup_trials
        ]

        p = _get_percentile_intermediate_result_over_trials(
            completed_trials, direction, step, normalized_weights,
            current_percentile, self._n_min_trials,
        )
        logger.info(f"Trial {trial.number} TRESHOLD PERCENTILE for step {step} = {p:.4f}")

        if math.isnan(p):
            return False

        if direction == StudyDirection.MAXIMIZE:
            return weighted_mean_intermediate_result < p
        return weighted_mean_intermediate_result > p


class CustomPatientPruner(PatientPruner):
    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        step = trial.last_step
        if step is None:
            return False

        intermediate_values = trial.intermediate_values
        steps = np.asarray(list(intermediate_values.keys()))

        # Do not prune if number of step to determine are insufficient.
        if steps.size < self._patience + 1:
            return False

        steps.sort()
        # This is the score patience steps ago
        step_before_patience = steps[-self._patience - 1]
        score_before_patience = intermediate_values[step_before_patience]
        # And these are the scores after that
        steps_after_patience = steps[-self._patience:]
        scores_after_patience = np.asarray(
            list(intermediate_values[step] for step in steps_after_patience)
        )

        direction = study.direction
        if direction == StudyDirection.MINIMIZE:
            maybe_prune = score_before_patience - self._min_delta < np.nanmin(
                scores_after_patience
            )
        else:
            maybe_prune = score_before_patience + self._min_delta > np.nanmax(
                scores_after_patience
            )

        if maybe_prune:
            if self._wrapped_pruner is not None:
                return self._wrapped_pruner.prune(study, trial)
            else:
                return True
        else:
            return False


# == == == == == == == == == == == == == == == == == == == == #
#class NormalPruner(BasePruner):
#    def __init__(
#        self,
#        percentile: float,
#        n_last_trials: int = 10,
#        n_warmup_steps: int = 0
#    ) -> None:
#        if n_warmup_steps < 0:
#            raise ValueError(
#                "Number of warmup steps cannot be negative but got {}.".format(n_warmup_steps)
#            )
#
#        self._percentile = percentile
#        self.n_last_trials = n_last_trials
#        self._n_warmup_steps = n_warmup_steps
#
#    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
#        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
#        n_trials = len(completed_trials)
#
#        if n_trials == 0:
#            return False
#
#        if n_trials <= self.n_last_trials:
#            return False
#
#        step = trial.last_step
#        if step is None:
#            return False
#
#        if step < self._n_warmup_steps:
#            return False
#
#        direction = study.direction
#        last_score = trial.intermediate_values.values()[-1]
#
#        if math.isnan(last_score):
#            return True
#
#        if len(completed_trials) == 0:
#            raise ValueError("No trials have been completed.")
#
#        intermediate_values = [t.intermediate_values[step] for t in completed_trials if step in t.intermediate_values]
#
#        if direction == StudyDirection.MAXIMIZE:
#            percentile = 100 - self._percentile
#        p = stats.norm(0, 1).ppf(percentile)
#
#        # А нормально ли распределены метрики?
#        z_score = last_score - np.mean(intermediate_values) / np.var(intermediate_values)
#
#        if direction == StudyDirection.MAXIMIZE:
#            return z_score < p
#        return z_score > p