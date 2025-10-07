from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class RNGManager:
    seed: int

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)
        self._np_random = np.random.default_rng(self.seed)

    @property
    def py_random(self) -> random.Random:
        return self._random

    @property
    def np_random(self) -> np.random.Generator:
        return self._np_random

    def reseed(self, seed: int) -> None:
        self.seed = seed
        self._random.seed(seed)
        self._np_random = np.random.default_rng(seed)
