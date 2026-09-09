"""The optional-import helper names the extra that fixes the problem."""

import pytest

from lettuce_greenhouse_gym._optional import EXTRAS, require


def test_require_returns_the_module_when_present():
    pytest.importorskip("matplotlib")
    assert require("matplotlib").__name__ == "matplotlib"


def test_require_names_the_extra_when_missing(monkeypatch):
    monkeypatch.setitem(EXTRAS, "no_such_module_xyz", "plot")
    with pytest.raises(ImportError, match=r'pip install "lettuce-greenhouse-gym\[plot\]"'):
        require("no_such_module_xyz")


def test_unregistered_modules_are_a_programming_error():
    with pytest.raises(KeyError):
        require("json")


def test_registry_matches_pyproject_extras():
    """Every registered extra must exist in pyproject, so the install hint is always valid."""
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    declared = set(pyproject["project"]["optional-dependencies"])
    assert set(EXTRAS.values()) <= declared
