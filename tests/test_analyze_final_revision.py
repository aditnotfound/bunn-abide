from __future__ import annotations

import numpy as np

from scripts.analyze_final_revision import (
    bootstrap_weighted_mean,
    exact_two_sided_sign_flip_p,
    holm_adjust,
    positive_topk_adjacency,
)
from src.neural_operators import random_walk_laplacian


def test_positive_topk_adjacency_uses_stable_upper_triangle_tie_break() -> None:
    connectome = np.zeros((4, 4), dtype=float)
    connectome[0, 1] = connectome[1, 0] = 3.0
    connectome[0, 2] = connectome[2, 0] = 2.0
    connectome[0, 3] = connectome[3, 0] = 2.0
    connectome[1, 2] = connectome[2, 1] = 1.0
    adjacency = positive_topk_adjacency(connectome, 0.34)
    assert adjacency.sum() == 6
    assert adjacency[0, 1] == 1
    assert adjacency[0, 2] == 1
    assert adjacency[0, 3] == 1


def test_exact_sign_flip_minimum_when_every_sign_agrees() -> None:
    values = np.ones(10)
    assert exact_two_sided_sign_flip_p(values) == 2 / (2**10)


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_weighted_bootstrap_point_estimate_uses_given_weights() -> None:
    estimate, low, high = bootstrap_weighted_mean(
        np.array([0.0, 1.0]), np.array([1.0, 3.0]), resamples=2000,
        rng=np.random.default_rng(7),
    )
    assert estimate == 0.75
    assert low <= estimate <= high


def test_isolated_nodes_have_zero_random_walk_laplacian() -> None:
    import torch

    adjacency = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    laplacian = random_walk_laplacian(adjacency).numpy()[0]
    assert np.allclose(laplacian[2], 0.0)
    degree = adjacency.numpy()[0].sum(axis=1)
    active = degree > 0
    inverse_sqrt = np.zeros_like(degree)
    inverse_sqrt[active] = 1.0 / np.sqrt(degree[active])
    symmetric = np.diag(active.astype(float)) - inverse_sqrt[:, None] * adjacency.numpy()[0] * inverse_sqrt[None, :]
    assert np.allclose(np.sort(np.linalg.eigvals(laplacian)), np.linalg.eigvalsh(symmetric))
