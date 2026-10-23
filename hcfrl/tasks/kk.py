"""Knights-and-Knaves (Xie et al. 2024 style, CFRL format).

n in {4,5,6} characters (A, B, C, ...), each makes one statement about others:
  "A: C knave"      -> A claims C is a knave
  "A: C knight"     -> A claims C is a knight
  "B: A same C"     -> B claims A and C are the same type
  "C: D diff B"     -> C claims D and B are different types
Knights tell the truth, knaves lie. Prompt = statements joined by " ; ". Response = assignment
"A knight B knave C knight D knave". Generator keeps only puzzles with a unique solution.
Sequence length 96 (56 condition) in CFRL.
"""
from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass

NAMES = "ABCDEFG"


@dataclass
class KKExample:
    n: int
    statements: list[tuple[str, str, str, str | None]]   # (speaker, kind, x, y)   kind in {knight, knave, same, diff}
    solution: dict[str, bool] | None = None               # True = knight

    @property
    def prompt(self) -> str:
        parts = []
        for s, kind, x, y in self.statements:
            parts.append(f"{s}: {x} {kind}" if y is None else f"{s}: {x} {kind} {y}")
        return " ; ".join(parts)

    @property
    def response(self) -> str:
        assert self.solution is not None
        return " ".join(f"{c} {'knight' if self.solution[c] else 'knave'}" for c in NAMES[: self.n])


def _holds(kind: str, x: str, y: str | None, a: dict[str, bool]) -> bool:
    if kind == "knight":
        return a[x]
    if kind == "knave":
        return not a[x]
    if kind == "same":
        return a[x] == a[y]
    if kind == "diff":
        return a[x] != a[y]
    raise ValueError(kind)


def consistent(statements, a: dict[str, bool]) -> bool:
    """Every speaker's statement is true iff the speaker is a knight."""
    return all(_holds(kind, x, y, a) == a[s] for s, kind, x, y in statements)


def solutions(n: int, statements) -> list[dict[str, bool]]:
    chars = NAMES[:n]
    out = []
    for bits in itertools.product([True, False], repeat=n):
        a = dict(zip(chars, bits))
        if consistent(statements, a):
            out.append(a)
    return out


class KKTask:
    name = "kk"

    def __init__(self, min_n: int = 4, max_n: int = 6):
        self.min_n, self.max_n = min_n, max_n

    def sample(self, rng: random.Random) -> KKExample:
        while True:
            n = rng.randint(self.min_n, self.max_n)
            chars = NAMES[:n]
            stmts = []
            for s in chars:
                others = [c for c in chars if c != s]
                kind = rng.choice(["knight", "knave", "same", "diff"])
                if kind in ("knight", "knave"):
                    stmts.append((s, kind, rng.choice(others), None))
                else:
                    x, y = rng.sample(others, 2)
                    stmts.append((s, kind, x, y))
            sols = solutions(n, stmts)
            if len(sols) == 1:
                return KKExample(n, stmts, sols[0])

    def generate(self, k: int, seed: int) -> list[KKExample]:
        rng = random.Random(seed)
        return [self.sample(rng) for _ in range(k)]

    # ---- verifier ----------------------------------------------------------
    _STMT = re.compile(r"^\s*([A-G])\s*:\s*([A-G])\s+(knight|knave|same|diff)(?:\s+([A-G]))?\s*$")
    _ASSIGN = re.compile(r"([A-G])\s+(knight|knave)")

    @classmethod
    def parse_prompt(cls, prompt: str) -> KKExample:
        stmts = []
        for part in prompt.split(";"):
            m = cls._STMT.match(part)
            if m is None:
                raise ValueError(f"bad statement: {part!r}")
            s, x, kind, y = m.groups()
            stmts.append((s, kind, x, y))
        n = len(stmts)
        return KKExample(n, stmts)

    @classmethod
    def verify(cls, example: KKExample, response: str) -> bool:
        pairs = cls._ASSIGN.findall(response)
        chars = NAMES[: example.n]
        a = {}
        for c, t in pairs:
            if c in a:        # duplicate assignment -> reject
                return False
            a[c] = (t == "knight")
        if sorted(a) != list(chars):
            return False
        return consistent(example.statements, a)
