#!/bin/bash
# Critic sharpness at the base, per model: rsa_gap = l_phi(state) - l_phi(state + delta) at the one-step worst-case
# displacement with |delta| = rho |state| (train_svg --rsa_rho rho --rsa_apply 0), 5 actor iterations per base, no RL effect
# (lr 0). Usage: sbatch scripts/sharpness.slurm <rho> <tag>[:<config>] ...   e.g. game24cd_flat:configs/elf_b_game24_cd.yml
#SBATCH --job-name=sharp
#SBATCH --account=irahbari_832
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
set -x
RHO=$1; shift
for spec in "$@"; do
  TAG=${spec%%:*}; CFG=${spec#*:}
  CKPT=$(sed 's/^CKPT=//' outputs/eval/${TAG}_ckpt.txt)
  BEST=$(python - <<PY
import json, glob, os, torch
d = "outputs/critic/${TAG}_2d"
if os.path.exists(f"{d}/log.json"): print(json.load(open(f"{d}/log.json"))["best_epoch"])
else: print(int(max(glob.glob(f"{d}/critic_ep*.pt"), key=lambda f: torch.load(f, map_location="cpu")["eval"]["field_score"]).split("critic_ep")[1].split(".pt")[0]))
PY
)
  python -m hcfrl.rl.train_svg --config $CFG --ckpt $CKPT --critic outputs/critic/${TAG}_2d/critic_ep${BEST}.pt \
      --train_prompts data/game24/train.jsonl --val_prompts data/game24/val.jsonl --n_val 8 --eval_every 1000 --save_every 0 \
      --out outputs/analysis/sharp_${TAG}_rho${RHO} --K 64 --gamma 56 --iters 6 --B 32 --NB 256 --Nq 32 --nq 2 \
      --lr_actor 0 --lr_actor_min 0 --rsa_rho $RHO --rsa_apply 0 --seed 0
done
echo SHARP_DONE
