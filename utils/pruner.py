from __future__ import annotations

from collections.abc import KeysView
import functools
import math

import numpy as np
from scipy import stats

import optuna
from optuna.pruners import PercentilePruner
from optuna.study._study_direction import StudyDirection
from optuna.trial._state import TrialState


def _get_percentile_intermediate_result_over_trials(
    completed_trials: list["optuna.trial.FrozenTrial"],
    direction: StudyDirection,
    step: int,
    percentile: float,
    n_min_trials: int,
) -> float:
    if len(completed_trials) == 0:
        raise ValueError("No trials have been completed.")

    intermediate_values = [
        sum(t.intermediate_values) / len(t.intermediate_values)
        for t in completed_trials if step in t.intermediate_values
    ]

    if len(intermediate_values) < n_min_trials:
        return math.nan

    if direction == StudyDirection.MAXIMIZE:
        percentile = 100 - percentile

    return float(
        np.nanpercentile(
            np.array(intermediate_values, dtype=float),
            percentile,
        )
    )


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
        percentile_range: list,
        n_trials: int,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        *,
        n_min_trials: int = 1,
    ) -> None:
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

        self._percentiles = np.linspace(percentile_range[0], percentile_range[1], n_trials - n_startup_trials)
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
        mean_intermediate_result = np.nanmean(values)
        if math.isnan(mean_intermediate_result):
            return True

        p = _get_percentile_intermediate_result_over_trials(
            completed_trials, direction, step, self._percentiles[trial.number - self._n_startup_trials], self._n_min_trials
        )
        if math.isnan(p):
            return False

        if direction == StudyDirection.MAXIMIZE:
            return mean_intermediate_result < p
        return mean_intermediate_result > p


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