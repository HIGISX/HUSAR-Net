from pathlib import Path

import numpy as np
import pytest

from husarnet import (
    Instance,
    load_instance,
    load_policy,
    make_synthetic,
    save_instance,
    save_policy,
    solve,
    train_policy,
)


def make_small_instance(seed: int = 7) -> Instance:
    rng = np.random.default_rng(seed)
    n_candidates = 12
    city = np.repeat(np.arange(4), 3)
    city_prov = np.array([0, 0, 1, 1], dtype=np.int64)
    prov = city_prov[city]
    q = rng.uniform(0.2, 2.0, size=(n_candidates, 6))
    return Instance(
        q=q,
        city=city,
        prov=prov,
        city_prov=city_prov,
        edges=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        rho=np.array([0.8, 1.0, 1.2, 0.9]),
        a_city=np.array([1.0, 0.9, 1.1, 1.0]),
        a_prov=np.array([1.0, 1.2]),
        K=4,
    )


def test_user_instance_solves_and_evaluates_consistently():
    instance = make_small_instance()

    allocation = solve(instance, swap_seconds=0.0, seed=77)

    assert len(allocation.selected) == instance.K
    assert len(set(allocation.selected)) == instance.K
    assert all(0 <= index < instance.N for index in allocation.selected)
    objective = instance.evaluate(np.asarray(allocation.selected, dtype=np.int64))
    assert allocation.objective == pytest.approx(objective[0])
    assert allocation.normalized_return == pytest.approx(objective[1])
    assert allocation.hierarchical_coverage == pytest.approx(objective[2])
    assert allocation.spatial_protection == pytest.approx(objective[3])


def test_local_exchange_does_not_reduce_the_objective():
    instance = make_small_instance()

    greedy_only = solve(instance, swap_seconds=0.0, seed=123)
    refined = solve(instance, swap_seconds=0.05, seed=123)

    assert refined.objective >= greedy_only.objective - 1e-12
    assert len(refined.selected) == instance.K
    assert len(set(refined.selected)) == instance.K


def test_npz_round_trip_preserves_the_instance(tmp_path: Path):
    instance = make_small_instance()
    path = tmp_path / "small_instance.npz"

    save_instance(instance, path)
    restored = load_instance(path)

    for name in ("q", "city", "prov", "city_prov", "edges", "rho", "a_city", "a_prov"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(instance, name))
    assert restored.K == instance.K
    assert restored.evaluate(np.arange(instance.K))[0] == pytest.approx(
        instance.evaluate(np.arange(instance.K))[0]
    )


def test_invalid_hierarchy_and_duplicate_adjacency_are_rejected():
    instance = make_small_instance()
    with pytest.raises(ValueError, match="prov must agree"):
        Instance(
            q=instance.q,
            city=instance.city,
            prov=np.zeros(instance.N, dtype=np.int64),
            city_prov=instance.city_prov,
            edges=instance.edges,
            K=instance.K,
        )

    with pytest.raises(ValueError, match="list each undirected edge once"):
        Instance(
            q=instance.q,
            city=instance.city,
            prov=instance.prov,
            city_prov=instance.city_prov,
            edges=np.array([[0, 1], [1, 0], [2, 3]], dtype=np.int64),
            K=instance.K,
        )


def test_one_round_training_checkpoint_can_be_saved_and_loaded(tmp_path: Path):
    training = make_synthetic(seed=701, N=40, G=8, P=2, B=8, K=6)
    validation = make_synthetic(seed=702, N=40, G=8, P=2, B=8, K=6)

    policy = train_policy(
        [training],
        [validation],
        imitation_epochs=1,
        ppo_rounds=1,
        seed=19,
    )
    path = tmp_path / "policy.pt"
    save_policy(policy, path)
    restored = load_policy(path)

    allocation = solve(validation, restored, swap_seconds=0.0, seed=19)
    assert len(allocation.selected) == validation.K
    assert len(set(allocation.selected)) == validation.K
    assert allocation.objective == pytest.approx(
        validation.evaluate(np.asarray(allocation.selected))[0]
    )
