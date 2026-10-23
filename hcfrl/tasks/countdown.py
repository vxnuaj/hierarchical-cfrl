"""Countdown task (Gandhi et al. 2024 formulation; CFRL setup).

Prompt:   "<target> : <n1> <n2> <n3> <n4>"      (4-5 numbers in 1..99, target 10..100)
Response: "a OP b = c , ..."                     (chain; see arith.py)
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from . import arith


@dataclass
class CountdownExample:
    nums: list[int]
    target: int
    solution: list[str] | None = None

    @property
    def prompt(self) -> str:
        return f"{self.target} : " + " ".join(str(n) for n in self.nums)

    @property
    def response(self) -> str:
        assert self.solution is not None
        return arith.format_chain(self.solution)


class CountdownTask:
    name = "countdown"

    def __init__(self, min_nums: int = 4, max_nums: int = 5, max_target: int = 100,
                 min_target: int = 10, max_num: int = 99):
        self.min_nums, self.max_nums = min_nums, max_nums
        self.min_target, self.max_target = min_target, max_target
        self.max_num = max_num

    def sample(self, rng: random.Random) -> CountdownExample:
        """Sample a solvable instance with one solution chain."""
        while True:
            k = rng.randint(self.min_nums, self.max_nums)
            target = rng.randint(self.min_target, self.max_target)
            nums = [rng.randint(1, self.max_num) for _ in range(k)]
            sol = arith.solve(nums, target)
            if sol is not None:
                return CountdownExample(nums, target, sol)

    def generate(self, n: int, seed: int) -> list[CountdownExample]:
        rng = random.Random(seed)
        return [self.sample(rng) for _ in range(n)]

    @staticmethod
    def verify(example: CountdownExample, response: str) -> bool:
        return arith.verify_chain(response, example.nums, example.target)

    @staticmethod
    def parse_prompt(prompt: str) -> CountdownExample:
        t, rest = prompt.split(":")
        return CountdownExample([int(x) for x in rest.split()], int(t))
