"""Convert {input, output} JSONL into the pre-tokenized on-disk HF dataset ELF's train.py expects
(columns: condition_input_ids, input_ids, plus input/target text for logging).
usage: python -m hcfrl.data.to_hf data/game24/train.jsonl data/game24/train_hf [--tokenizer t5-small]
"""
import argparse, json
from datasets import Dataset
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--tokenizer", default="t5-small")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
rows = [json.loads(l) for l in open(a.src) if l.strip()]
ds = Dataset.from_dict({
    "index": list(range(len(rows))),
    "input": [r["input"] for r in rows],
    "target": [r.get("output", "") for r in rows],
    "condition_input_ids": [tok(r["input"], add_special_tokens=False)["input_ids"] for r in rows],
    "input_ids": [tok(r.get("output", ""), add_special_tokens=False)["input_ids"] for r in rows],
})
ds.save_to_disk(a.dst)
print(a.dst, len(ds), "rows; max cond", max(len(x) for x in ds["condition_input_ids"]), "max out", max(len(x) for x in ds["input_ids"]))
