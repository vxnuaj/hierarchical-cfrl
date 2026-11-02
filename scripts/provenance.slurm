#!/bin/bash
# Environment snapshot + checksums of data, configs and the checkpoints the results depend on.
#   sbatch scripts/provenance.slurm
#SBATCH --job-name=provenance
#SBATCH --account=irahbari_832
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
mkdir -p outputs/provenance
{ echo "# environment snapshot $(date -Is)"; python -V; python -c 'import torch;print("torch",torch.__version__,"cuda",torch.version.cuda,"cudnn",torch.backends.cudnn.version())';
  module list 2>&1; uv pip freeze 2>/dev/null || pip freeze; } > outputs/provenance/environment.txt
# data and configs
find data configs -type f \( -name '*.jsonl' -o -name '*.json' -o -name '*.pt' -o -name '*.yml' \) -print0 | sort -z | xargs -0 sha256sum > outputs/provenance/sha256_data_configs.txt
# checkpoints: selected SFT bases, every critic's selected epoch, policy_it0/best/final of every RL run
{
for t in outputs/eval/*_ckpt.txt; do p=$(sed 's/^CKPT=//' $t); [ -d "$p" ] && find "$p" -type f -name '*.pt' -o -type f -name '*.safetensors' | head -20; done
for d in outputs/critic/*_2d; do b=$(python -c "import json,sys;print(json.load(open('$d/log.json'))['best_epoch'])" 2>/dev/null); [ -n "$b" ] && echo $d/critic_ep$b.pt; done
for d in outputs/rl/*; do for f in policy_it0.pt policy_best.pt policy_it1500.pt policy_it1000.pt; do [ -f $d/$f ] && echo $d/$f; done; done
} | sort -u > outputs/provenance/ckpt_list.txt
xargs -a outputs/provenance/ckpt_list.txt -d '\n' sha256sum > outputs/provenance/sha256_checkpoints.txt
echo PROVENANCE_DONE
