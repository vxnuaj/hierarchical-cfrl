#!/bin/bash
# After the per-prompt screens finish: build the difficulty-matched prompt sets and launch critic + RL chains for ELF-M.
#SBATCH --job-name=launch-hard
#SBATCH --account=irahbari_832
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.out
source scripts/common.sh
set -x
for M in flat hier; do
  if [ $M = flat ]; then CFG=configs/elfM_b_game24_cd.yml; CH="0 1 2"; else CFG=configs/helfM_b_game24_cd.yml; CH="0 1 2 3 4 5"; fi
  ok=1; for i in $CH; do [ -f outputs/screen/M_${M}_$i/summary.txt ] || ok=0; done
  [ $ok = 1 ] || { echo "screens for $M incomplete"; continue; }
  python -m hcfrl.analysis.offline.build_hard_set --chunks $(for i in $CH; do echo data/game24_screen/chunk_$i.jsonl; done) \
      --screens $(for i in $CH; do echo outputs/screen/M_${M}_$i; done) --repeat 2 --target 0.75 \
      --train data/game24/train.jsonl --src data/game24 --out data/game24_hardM_${M} || continue
  CK=$(sed 's/^CKPT=//' outputs/eval/game24cdM_${M}_ckpt.txt)
  g=$(sbatch --parsable -J gate0a-hardM-$M scripts/gate0a_hard_p100.slurm $CFG $CK data/game24_hardM_${M} game24cdMhard_${M} 0 4)
  echo "critic $M $g"
  for s in 0 1 2; do
    echo "rl $M s$s $(sbatch --parsable -d afterok:$g -J p100-Mhard-$M-s$s scripts/gate0b_p100.slurm $CFG $CK data/game24_hardM_${M} game24cdMhard_${M} $s 64 56 1500 game24cdMhard_${M}_svg2d_s$s --actor_chunk 2)"
  done
done
