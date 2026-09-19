import numpy as np
import torch

from redqueen.analysis import run_experiment, trend_test
from redqueen.config import Config


def test_trend_test_detects_clear_increasing_trend():
    series = list(np.arange(30, dtype=float) + np.random.RandomState(0).randn(30) * 0.1)
    result = trend_test(series)
    assert result is not None
    assert result.trend == "increasing"
    assert result.significant
    assert result.p_value < 0.05


def test_trend_test_no_trend_on_periodic_series():
    # A symmetric oscillation has as many rising pairs as falling ones over
    # whole periods, so Mann-Kendall's S statistic should stay near zero
    # regardless of noise realization - avoids relying on a "lucky" random seed.
    t = np.arange(40)
    series = list(np.sin(t * (2 * np.pi / 8)))
    result = trend_test(series)
    assert result is not None
    assert not result.significant
    assert result.trend == "no trend"


def test_trend_test_returns_none_for_short_series():
    assert trend_test([1.0, 2.0, 3.0]) is None


def test_trend_test_constant_series_is_not_significant():
    result = trend_test([1.0, 1.0, 1.0, 1.0, 1.0])
    assert result is not None
    assert not result.significant
    assert result.trend == "no trend"


def _small_cfg(**overrides) -> Config:
    cfg = Config(
        world_size=50.0,
        max_prey=200,
        max_predators=60,
        food_max_patches=100,
        k_neighbours=2,
        hidden_dim=4,
        eat_radius=1.5,
        reproduction_threshold=1.05,
        initial_energy=1.2,
        reference_burn_in_steps=20,
        competence_sample_interval=20,
        competence_n_trials=4,
        competence_trial_steps=15,
        device="cpu",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_run_experiment_end_to_end_small():
    cfg = _small_cfg()
    result = run_experiment(cfg, seeds=[0, 1], n_steps=90, log_interval=10)

    assert len(result.seed_results) == 2
    for seed_result in result.seed_results:
        assert len(seed_result.predator_competence_series) == len(seed_result.log.competence_steps)
        assert len(seed_result.log.realized_predator_catch_rate) == len(
            seed_result.log.realized_rate_steps
        )

    verdict = result.arms_race_verdict()
    assert isinstance(verdict, str)
    assert "seeds show" in verdict
