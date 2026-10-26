#!/bin/bash
# Copy the checkpoints the paper's tables depend on from scratch (purged) to the user's home directory (backed up, 100 GB quota).
# Runs ON the cluster:  nohup bash scripts/backup_checkpoints.sh > outputs/backup.log 2>&1 &
# Lean set (~15 GB): selected SFT bases, best critic epoch per critic, final + best policy of every Game24-cd / K&K / Countdown run.
set -u
SRC=/scratch1/verajuan/hcfrl/outputs
DST=$HOME/hcfrl_backup   # own home quota (backed up, not purged); never the group project space
mkdir -p $DST/sft $DST/critic $DST/rl
cp_sft() { d=$1; [ -d "$SRC/$d" ] || return; mkdir -p $DST/sft/$d; rsync -a $SRC/$d/config.yml $DST/sft/$d/ 2>/dev/null; rsync -a "$SRC/$d/$2" $DST/sft/$d/; }
# selected bases (gate0a wrote the chosen checkpoint into outputs/eval/<tag>_ckpt.txt)
for t in $SRC/eval/*_ckpt.txt; do
  p=$(sed 's/^CKPT=//' $t); rel=${p#outputs/}; d=$(dirname $rel); c=$(basename $rel)
  [ "$d" = "." ] && { d=$rel; c=$(ls -t $SRC/$d | grep checkpoint_ | head -1); }
  cp_sft $d $c
done
# best critic epoch per critic dir
for d in $SRC/critic/*_2d; do
  n=$(basename $d); mkdir -p $DST/critic/$n; rsync -a $d/log.json $DST/critic/$n/ 2>/dev/null
  best=$(python - $d <<'PY'
import glob, sys, torch
d = sys.argv[1]
fs = glob.glob(f"{d}/critic_ep*.pt")
print(max(fs, key=lambda f: torch.load(f, map_location="cpu")["eval"]["field_score"]) if fs else "")
PY
)
  [ -n "$best" ] && rsync -a $best $DST/critic/$n/
done
# RL runs: log + final + best policy (policy_it0 equals the base and is skipped)
for d in $SRC/rl/game24cd_* $SRC/rl/kk_* $SRC/rl/countdown_*; do
  n=$(basename $d); mkdir -p $DST/rl/$n; rsync -a $d/log.jsonl $DST/rl/$n/ 2>/dev/null
  last=$(ls $d/policy_it*.pt 2>/dev/null | sed 's/.*policy_it//; s/\.pt//' | sort -n | tail -1)
  [ -n "$last" ] && [ "$last" != "0" ] && rsync -a $d/policy_it$last.pt $DST/rl/$n/
  [ -f $d/policy_best.pt ] && rsync -a $d/policy_best.pt $DST/rl/$n/
done
rsync -a $SRC/analysis $SRC/eval $DST/ 2>/dev/null
echo "BACKUP_DONE $(date)"; du -sh $DST
