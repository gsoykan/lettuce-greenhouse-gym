"""Import an optional dependency, or explain which extra installs it."""

import importlib
from types import ModuleType

# Which extra in pyproject.toml provides each optional module. One place to keep in sync.
EXTRAS: dict[str, str] = {
    "stable_baselines3": "train",
    "torch": "train",
    "yaml": "train",
    "tensorboard": "train",
    "wandb": "train",
    "matplotlib": "plot",
}


def require(module: str) -> ModuleType:
    """Return ``module`` if importable, else raise with the ``pip install`` command that fixes it."""
    extra = EXTRAS[module]  # a KeyError here is a programming error: register the module above
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"{module!r} is not installed; it is part of the {extra!r} extra: "
            f'pip install "lettuce-greenhouse-gym[{extra}]"'
        ) from e
