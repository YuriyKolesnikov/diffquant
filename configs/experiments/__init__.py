# configs/experiments/__init__.py
"""Experiment registry for DiffQuant."""

import importlib.util


def load_config(path: str):
    """Load a MasterConfig instance from an experiment file path."""
    spec = importlib.util.spec_from_file_location("_experiment_cfg", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "cfg"):
        raise AttributeError(
            f"Experiment file '{path}' must define a top-level 'cfg' variable."
        )
    return mod.cfg

