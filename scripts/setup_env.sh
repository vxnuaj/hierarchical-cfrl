#!/bin/bash
# One-time environment setup on CARC (login node OK: downloads + pip only).
set -e
cd /scratch1/verajuan/hcfrl
source /etc/profile.d/modules.sh 2>/dev/null; module purge; module load gcc/13.3.0 python/3.11.9 cuda/12.6.3 2>/dev/null || module load python/3.11.9
export HF_HOME=/scratch1/verajuan/hf_cache TMPDIR=/scratch1/verajuan/tmp PIP_CACHE_DIR=/scratch1/verajuan/tmp/pip
[ -x .venv/bin/python ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu126
pip install -q "numpy>=2" PyYAML tqdm einops "transformers>=4.41.2,<4.45.0" datasets huggingface-hub scikit-learn sacrebleu rouge-score wandb muon-optimizer
python - <<'PY'
import torch; print("torch", torch.__version__, "cuda build", torch.version.cuda)
from huggingface_hub import snapshot_download
print(snapshot_download("embedded-language-flows/ELF-B-owt-torch"))
from transformers import AutoTokenizer, T5EncoderModel
AutoTokenizer.from_pretrained("t5-small"); T5EncoderModel.from_pretrained("t5-small"); print("t5-small cached")
PY
for s in train val; do python -m hcfrl.data.to_hf data/game24/$s.jsonl data/game24/${s}_hf; done
echo SETUP_DONE
