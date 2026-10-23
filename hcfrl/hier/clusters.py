"""Coarse map Gamma: vocab -> n clusters, from k-means over the frozen T5-small input embeddings
(H-CDLM recipe: sklearn KMeans, k-means++, n_init=10, max_iter=300, tol=1e-4, seed 42; rows unit-normalized).
Special tokens (pad/eos/unk and extra_ids) go to the reserved structural cluster n-1.

usage: python -m hcfrl.hier.clusters --n 16 --out data/clusters_t5small_n16.pt [--inspect]
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, T5EncoderModel


def build(n: int, model_name: str = "t5-small", seed: int = 42):
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = T5EncoderModel.from_pretrained(model_name)
    E = enc.get_input_embeddings().weight.detach().float().numpy()        # (V, 512)
    V = tok.vocab_size
    E = E[:V]
    special = set(tok.all_special_ids)
    keep = np.array([i for i in range(V) if i not in special])
    X = E[keep] / np.linalg.norm(E[keep], axis=1, keepdims=True)
    km = KMeans(n_clusters=n - 1, init="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=seed).fit(X)
    gamma = np.full(V, n - 1, dtype=np.int64)
    gamma[keep] = km.labels_
    return tok, gamma, km.cluster_centers_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--out", required=True)
    ap.add_argument("--inspect", action="store_true"); ap.add_argument("--model", default="t5-small")
    a = ap.parse_args()
    tok, gamma, C = build(a.n, a.model)
    torch.save({"gamma": torch.from_numpy(gamma), "centers": torch.from_numpy(C), "n": a.n, "model": a.model}, a.out)
    sizes = Counter(gamma.tolist())
    print("cluster sizes:", dict(sorted(sizes.items())))
    if a.inspect:
        # what cluster do the task tokens land in?
        for text in ["24 : 7 9 1 7", "7+9=16 , 1+7=8 , 16+8=24", "46*28=1288 , 75-65=10 , 1288-10=1278 , 1278/18=71"]:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            print(text); print("   ", [(tok.convert_ids_to_tokens(i), int(gamma[i])) for i in ids])
        # sample members of each cluster
        rng = np.random.default_rng(0)
        for c in range(a.n):
            m = np.nonzero(gamma == c)[0]
            s = rng.choice(m, size=min(12, len(m)), replace=False)
            print(f"c{c:2d} ({len(m):5d}):", " ".join(tok.convert_ids_to_tokens(int(i)) for i in s))
        # number tokens: how are they spread?
        num_ids = [i for i in range(tok.vocab_size) if tok.convert_ids_to_tokens(i).lstrip("▁").isdigit()]
        print("number-like tokens:", len(num_ids), "cluster hist:", dict(Counter(int(gamma[i]) for i in num_ids)))




# ---------------------------------------------------------------------------------------------
# Task-informed abstraction for arithmetic tasks (Gamma_task), n = 16:
#   0-7  number tokens by magnitude bin of their numeric value:
#        [0-9], [10-19], [20-49], [50-99], [100-299], [300-999], [1000-9999], [>=10000]
#   8-11 operator tokens + , - , * , /   (merged op+number tokens like "-10" or "/18" -> the op's class)
#   12   "="   13   separators ("," and lone "▁")   14   everything else   15   special tokens
# Deterministic function of the token id, so the joint sampler stays an exact token sampler.
# ---------------------------------------------------------------------------------------------
import re as _re

_BINS = [10, 20, 50, 100, 300, 1000, 10000]


def _mag_bin(v: int) -> int:
    for b, hi in enumerate(_BINS):
        if v < hi:
            return b
    return len(_BINS)


def build_task_map(tok) -> np.ndarray:
    V = tok.vocab_size
    gamma = np.full(V, 14, dtype=np.int64)
    special = set(tok.all_special_ids)
    for i in range(V):
        if i in special:
            gamma[i] = 15; continue
        s = tok.convert_ids_to_tokens(i).replace("▁", "")
        if s == "":
            gamma[i] = 13
        elif s.isdigit():
            gamma[i] = _mag_bin(int(s))
        elif _re.fullmatch(r"[+\-*/]\d*", s):
            gamma[i] = 8 + "+-*/".index(s[0])
        elif s == "=":
            gamma[i] = 12
        elif s == ",":
            gamma[i] = 13
    return gamma


def main_task():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="t5-small")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    gamma = build_task_map(tok)
    torch.save({"gamma": torch.from_numpy(gamma), "n": 16, "model": a.model, "kind": "task_arith"}, a.out)
    print("cluster sizes:", dict(sorted(Counter(gamma.tolist()).items())))
    for text in ["24 : 7 9 1 7", "46*28=1288 , 75-65=10 , 1288-10=1278 , 1278/18=71"]:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        print(text); print("   ", [(tok.convert_ids_to_tokens(i), int(gamma[i])) for i in ids])


if __name__ == "__main__":
    import sys
    main_task() if "--task" in sys.argv and sys.argv.remove("--task") is None else main()
