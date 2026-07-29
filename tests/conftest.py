from __future__ import annotations

import importlib.metadata
import importlib.util
import pathlib
import sys

import pytest


@pytest.fixture(scope="session")
def upstream():
    distribution = importlib.metadata.distribution("textdistance")
    root = pathlib.Path(distribution.locate_file("textdistance"))
    spec = importlib.util.spec_from_file_location(
        "upstream_textdistance",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
