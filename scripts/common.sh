# sourced by Slurm jobs
cd /scratch1/verajuan/hcfrl
source /etc/profile.d/modules.sh 2>/dev/null; module purge; module load gcc/13.3.0 python/3.11.9 cuda/12.6.3 2>/dev/null || module load python/3.11.9
# uv-managed environment (uv sync --frozen, see scripts/setup_env.sh); falls back to the legacy venv if absent
if [ -f /scratch1/verajuan/hcfrl/.venv-uv/READY ]; then   # marker written by setup_env.sh after a verified build
  source /scratch1/verajuan/hcfrl/.venv-uv/bin/activate
else
  source .venv/bin/activate
fi
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/scratch1/verajuan/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch1/verajuan/hcfrl:$PYTHONPATH
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
