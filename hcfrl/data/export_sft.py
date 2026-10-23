"""Export task SFT data as JSONL {input, output} for ELF's load_jsonl_dataset.

usage: python -m hcfrl.data.export_sft game24 data/game24
       python -m hcfrl.data.export_sft countdown data/countdown --n_train 500000 --n_heldout 2000 --n_pool 50000
Held-out split follows CFRL: even index = val (ckpt selection), odd index = test.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from hcfrl.tasks import CountdownTask, Game24Task


def dump(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{path}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["game24", "countdown", "kk"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--n_train", type=int, default=500_000)
    ap.add_argument("--n_heldout", type=int, default=2_000)
    ap.add_argument("--n_pool", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.task == "game24":
        t = Game24Task(split_seed=a.seed)
        train = [{"input": e.prompt, "output": e.response} for e in t.train()]
        rng = random.Random(a.seed)
        rng.shuffle(train)
        held = [{"input": e.prompt, "nums": e.nums, "target": 24} for e in t.holdout()]
    elif a.task == "kk":
        from hcfrl.tasks.kk import KKTask
        t = KKTask()
        train = [{"input": e.prompt, "output": e.response, "task": "kk"} for e in t.generate(a.n_train, a.seed)]
        held = [{"input": e.prompt, "task": "kk"} for e in t.generate(a.n_heldout, a.seed + 1)]
        pool = [{"input": e.prompt, "task": "kk"} for e in t.generate(a.n_pool, a.seed + 2)]
        dump(pool, a.out / "pool.jsonl")
        dump([{"input": r["input"], "task": "kk"} for r in train], a.out / "train_prompts.jsonl")
    else:
        t = CountdownTask()
        # disjoint seeds for train / heldout / unlabelled pool
        train = [{"input": e.prompt, "output": e.response} for e in t.generate(a.n_train, a.seed)]
        held = [{"input": e.prompt, "nums": e.nums, "target": e.target} for e in t.generate(a.n_heldout, a.seed + 1)]
        pool = [{"input": e.prompt, "nums": e.nums, "target": e.target} for e in t.generate(a.n_pool, a.seed + 2)]
        dump(pool, a.out / "pool.jsonl")
    dump(train, a.out / "train.jsonl")
    dump(held[0::2], a.out / "val.jsonl")
    dump(held[1::2], a.out / "test.jsonl")


if __name__ == "__main__":
    main()
