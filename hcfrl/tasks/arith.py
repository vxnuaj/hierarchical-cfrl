"""Shared arithmetic-chain verifier used by Countdown and Game24.

Response format (Stream-of-Search style, as used by CFRL, arXiv 2610.19080):
    "12-5=7 , 4*7=28 , 28-3=25"
Each step is `a OP b = c` with integer a, b, c; operands must come from the
current multiset of available numbers (the inputs, plus results of earlier
steps); each number is consumed once; the last result must equal the target.
"""
from __future__ import annotations

import itertools
import re
from typing import Iterable

_STEP = re.compile(r"^\s*(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)\s*$")
_OPS = "+-*/"


def _apply(a: int, op: str, b: int) -> int | None:
    if op == "+":
        return a + b
    if op == "*":
        return a * b
    if op == "-":
        return a - b if a >= b else None
    if op == "/":
        return a // b if b != 0 and a % b == 0 else None
    return None


def verify_chain(text: str, nums: Iterable[int], target: int) -> bool:
    """Return True iff `text` is a valid chain reducing `nums` to `target`."""
    avail = sorted(int(n) for n in nums)
    steps = [s for s in text.split(",")]
    if not steps or len(steps) != len(avail) - 1:
        return False
    last = None
    for s in steps:
        m = _STEP.match(s)
        if m is None:
            return False
        a, op, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        if a not in avail:
            return False
        avail.remove(a)
        if b not in avail:
            avail.append(a)
            return False
        avail.remove(b)
        r = _apply(a, op, b)
        if r is None or r != c:
            return False
        avail.append(c)
        last = c
    return len(avail) == 1 and last == target


def _combine(a: int, b: int):
    """All (result, step_string) pairs from combining a and b (SoS convention)."""
    out = [(a + b, f"{a}+{b}={a + b}"), (a * b, f"{a}*{b}={a * b}")]
    hi, lo = (b, a) if a <= b else (a, b)
    out.append((hi - lo, f"{hi}-{lo}={hi - lo}"))
    if lo != 0 and hi % lo == 0:
        out.append((hi // lo, f"{hi}/{lo}={hi // lo}"))
    return out


def solve(nums: list[int], target: int, _ops: list[str] | None = None) -> list[str] | None:
    """DFS for one solution chain, or None. Same search order as SoS."""
    ops = _ops or []
    if len(nums) == 1:
        return ops if nums[0] == target else None
    for i, j in itertools.combinations(range(len(nums)), 2):
        rest = [nums[k] for k in range(len(nums)) if k not in (i, j)]
        for r, step in _combine(nums[i], nums[j]):
            sol = solve(rest + [r], target, ops + [step])
            if sol is not None:
                return sol
    return None


def all_solutions(nums: list[int], target: int, limit: int = 10_000) -> list[list[str]]:
    """Enumerate solution chains (bounded)."""
    found: list[list[str]] = []

    def rec(ns, ops):
        if len(found) >= limit:
            return
        if len(ns) == 1:
            if ns[0] == target:
                found.append(ops)
            return
        for i, j in itertools.combinations(range(len(ns)), 2):
            rest = [ns[k] for k in range(len(ns)) if k not in (i, j)]
            for r, step in _combine(ns[i], ns[j]):
                rec(rest + [r], ops + [step])

    rec(list(nums), [])
    return found


def format_chain(steps: list[str]) -> str:
    return " , ".join(steps)
