#!/bin/bash
# One-time environment setup on CARC with uv (login node OK: downloads only).
# Reproduces the exact package set of the reported runs from uv.lock (torch 2.14.1+cu126, transformers 4.44.2, ...).
set -e
cd /scratch1/verajuan/hcfrl
source /etc/profile.d/modules.sh 2>/dev/null; module purge; module load gcc/13.3.0 python/3.11.9 cuda/12.6.3 2>/dev/null || module load python/3.11.9
export HF_HOME=/scratch1/verajuan/hf_cache TMPDIR=/scratch1/verajuan/tmp
export UV_CACHE_DIR=/scratch1/verajuan/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/scratch1/verajuan/hcfrl/.venv-uv
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=$HOME/.local/bin sh; export PATH=$HOME/.local/bin:$PATH; }
uv sync --frozen --no-dev --python "$(which python3)"
uv run --no-sync python - <<'PY'
import torch; print("torch", torch.__version__, "cuda build", torch.version.cuda)
from huggingface_hub import snapshot_download
print(snapshot_download("embedded-language-flows/ELF-B-owt-torch"))
from transformers import AutoTokenizer, T5EncoderModel
AutoTokenizer.from_pretrained("t5-small"); T5EncoderModel.from_pretrained("t5-small"); print("t5-small cached")
PY
for s in train val; do uv run --no-sync python -m hcfrl.data.to_hf data/game24/$s.jsonl data/game24/${s}_hf; done
touch $UV_PROJECT_ENVIRONMENT/READY
echo SETUP_DONE
