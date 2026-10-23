"""Game of 24 (Yao et al. 2023 ToT data). Split BY NUMBER SET as in CFRL:
1,076 train sets -> all orderings; 269 held-out sets.

Prompt:   "24 : a b c d"
Response: chain reducing to 24 (see arith.py).
"""
from __future__ import annotations

import csv
import itertools
import random
from dataclasses import dataclass
from pathlib import Path

from . import arith

DATA = Path(__file__).resolve().parents[2] / "data" / "game24.csv"


@dataclass
class Game24Example:
    nums: list[int]
    solution: list[str] | None = None
    target: int = 24

    @property
    def prompt(self) -> str:
        return "24 : " + " ".join(str(n) for n in self.nums)

    @property
    def response(self) -> str:
        assert self.solution is not None
        return arith.format_chain(self.solution)


class Game24Task:
    name = "game24"

    def __init__(self, path: Path = DATA, holdout_frac: float = 0.2, split_seed: int = 0):
        sets: list[tuple[int, ...]] = []
        with open(path) as f:
            for row in csv.DictReader(f):
                sets.append(tuple(int(x) for x in row["Puzzles"].split()))
        rng = random.Random(split_seed)
        idx = list(range(len(sets)))
        rng.shuffle(idx)
        n_hold = round(len(sets) * holdout_frac)
        self.holdout_sets = [sets[i] for i in sorted(idx[:n_hold])]
        self.train_sets = [sets[i] for i in sorted(idx[n_hold:])]

    @staticmethod
    def _orderings(s: tuple[int, ...]) -> list[list[int]]:
        return [list(p) for p in sorted(set(itertools.permutations(s)))]

    def _expand(self, sets, with_solutions: bool) -> list[Game24Example]:
        out = []
        for s in sets:
            for order in self._orderings(s):
                sol = arith.solve(order, 24) if with_solutions else None
                if with_solutions and sol is None:
                    continue  # ToT list has a few sets unsolvable under integer-only ops
                out.append(Game24Example(order, sol))
        return out

    def train(self) -> list[Game24Example]:
        return self._expand(self.train_sets, with_solutions=True)

    def holdout(self) -> list[Game24Example]:
        return self._expand(self.holdout_sets, with_solutions=False)

    @staticmethod
    def verify(example: Game24Example, response: str) -> bool:
        return arith.verify_chain(response, example.nums, 24)
