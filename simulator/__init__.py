# simulator/__init__.py
from simulator.diff_simulator import DiffSimulator, smooth_abs
from simulator.losses         import compute_loss

__all__ = ["DiffSimulator", "smooth_abs", "compute_loss"]

