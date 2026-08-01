"""Small helpers shared only by executable examples."""

from types import SimpleNamespace

import numpy as np


class GridSearchOptimizer:
    """Dependency-free one-parameter optimizer for short VQE examples."""

    def __init__(self, points=33):
        self.points = int(points)

    def minimize(self, fun, x0, args=()):
        initial = np.asarray(x0, dtype=float).reshape(-1)
        if initial.size != 1:
            raise ValueError("GridSearchOptimizer examples require one parameter.")
        candidates = np.linspace(-np.pi, np.pi, self.points)
        evaluations = [(float(fun(np.array([theta]), *args)), theta) for theta in candidates]
        energy, theta = min(evaluations, key=lambda item: item[0])
        return SimpleNamespace(fun=energy, x=np.array([theta]))
