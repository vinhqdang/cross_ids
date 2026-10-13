"""Registry of named methods: name -> factory(seed, threads, device)."""

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
    "rift-st": RiftConfig(self_train=True),
}

Factory = Callable[[int, int | None, str], Method]


def _factory(name: str) -> Factory:
    if name in BASELINE_CLASSES:
        cls = BASELINE_CLASSES[name]
        return lambda seed, threads, device: cls(seed=seed, threads=threads, device=device)
    if name == "coral":
        return lambda seed, threads, device: Coral(seed=seed)
    if name == "dann":
        return lambda seed, threads, device: Dann(seed=seed)
    if name in RIFT_VARIANTS:
        config = RIFT_VARIANTS[name]
        return lambda seed, threads, device: Rift(config=config, seed=seed, name=name)
    raise KeyError(f"unknown method {name!r}")


METHOD_NAMES: tuple[str, ...] = (
    "lr",
    "rf",
    "hgb",
    "catboost",
    "xgboost",
    "mlp",
    "coral",
    "dann",
    *RIFT_VARIANTS,
)


def make_method(name: str, seed: int = 0, threads: int | None = None, device: str = "cpu") -> Method:
    """Instantiate a registered method. Raises MethodUnavailable when it cannot run here."""
    return _factory(name)(seed, threads, device)


__all__ = ["Domain", "Method", "MethodUnavailable", "METHOD_NAMES", "make_method"]
