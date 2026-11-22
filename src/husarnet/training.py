"""Train the mean-risk policy on independent allocation instances."""

from collections.abc import Iterable

import torch

from ._objective import Instance, greedy
from ._policy import HUSARNet, NetConfig, expert_order, imitation_train, make_cache, ppo_train


def train_policy(
    training: Iterable[Instance],
    validation: Iterable[Instance],
    *,
    imitation_epochs: int = 12,
    ppo_rounds: int = 16,
    seed: int = 2026,
    verbose: bool = False,
) -> HUSARNet:
    """Imitate greedy teachers and refine the policy with PPO on the full objective."""
    training, validation = tuple(training), tuple(validation)
    if not training or not validation:
        raise ValueError("supply at least one training instance and one validation instance")
    if imitation_epochs < 1 or ppo_rounds < 0:
        raise ValueError("imitation_epochs must be positive and ppo_rounds nonnegative")

    teachers = [expert_order(instance, greedy(instance, screen=1500)) for instance in training]
    train_caches = [make_cache(instance) for instance in training]
    validation_caches = [make_cache(instance) for instance in validation]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = HUSARNet(NetConfig())
        imitation_train(
            model, training, teachers, caches=train_caches, epochs=imitation_epochs,
            seed=seed, verbose=verbose,
        )
        if ppo_rounds:
            ppo_train(
                model, training, validation, rounds=ppo_rounds, seed=seed,
                train_caches=train_caches, val_caches=validation_caches, verbose=verbose,
            )
    return model.eval()
