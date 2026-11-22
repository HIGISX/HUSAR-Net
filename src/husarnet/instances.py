"""Load and save allocation instances in NumPy's NPZ format."""

from pathlib import Path

import numpy as np

from ._objective import Instance

_REQUIRED = ("q", "city", "prov", "city_prov", "edges", "K")
_OPTIONAL = ("rho", "a_city", "a_prov", "alpha", "beta")


def load_instance(path: str | Path) -> Instance:
    """Load candidate risks, the region hierarchy, and the budget from an NPZ file."""
    with np.load(path, allow_pickle=False) as archive:
        fields = set(archive.files)
        missing = set(_REQUIRED) - fields
        if missing:
            raise ValueError(f"missing instance fields: {', '.join(sorted(missing))}")
        inputs = {key: archive[key] for key in _REQUIRED + _OPTIONAL if key in fields}
    for key in ("K", "alpha", "beta"):
        if key in inputs:
            inputs[key] = inputs[key].item()
    return Instance(**inputs)


def save_instance(instance: Instance, path: str | Path) -> None:
    """Save an instance using its original scenario precision."""
    inputs = {key: getattr(instance, key) for key in _REQUIRED + _OPTIONAL}
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **inputs)
