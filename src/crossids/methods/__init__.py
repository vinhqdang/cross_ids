"""Registry of named methods: name -> factory(seed)."""

from __future__ import annotations

from collections.abc import Callable

from .base import Domain, Method, MethodUnavailable
from .baselines import BASELINE_CLASSES
from .coral import Coral
from .dann import Dann
from .rift import Rift, RiftConfig

RIFT_VARIANTS: dict[str, RiftConfig] = {
    "rift": RiftConfig(),
    "rift-noDN": RiftConfig(dn="global"),
    "rift-noAnchor": RiftConfig(anchor="none"),
    "rift-noIFS": RiftConfig(ifs=False),
    "rift-noLSC": RiftConfig(lsc=False),
    "rift-robust": RiftConfig(dn="robust"),
    "rift-standard": RiftConfig(dn="standard"),
}


def _factory(name: str) -> Callable[[int], Method]:
    if name in BASELINE_CLASSES:
        cls = BASELINE_CLASSES[name]
        return lambda seed: cls(seed=seed)
    if name == "coral":
        return lambda seed: Coral(seed=seed)
    if name == "dann":
        return lambda seed: Dann(seed=seed)
    if name in RIFT_VARIANTS:
        config = RIFT_VARIANTS[name]
        return lambda seed: Rift(config=config, seed=seed, name=name)
    raise KeyError(f"unknown method {name!r}")


METHOD_NAMES: tuple[str, ...] = (
    "lr",
    "rf",
    "hgb",
    "catboost",
    "mlp",
    "coral",
    "dann",
    *RIFT_VARIANTS,
)


def make_method(name: str, seed: int = 0) -> Method:
    """Instantiate a registered method. Raises MethodUnavailable when it cannot run here."""
    return _factory(name)(seed)


__all__ = [
    "Domain",
    "Method",
    "MethodUnavailable",
    "METHOD_NAMES",
    "make_method",
]
