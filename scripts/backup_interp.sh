#!/bin/bash
# Copy the checkpoints needed for the deferred interpretability analyses (notes/results.md 14:30 10-28)
# from purgeable /scratch1 to the persistent home directory. Policies: it500, it1500, best. Critics: best epoch.
set -u
cd /scratch1/verajuan/hcfrl
DST=/home1/verajuan/hcfrl_backup/interp
mkdir -p $DST/rl $DST/critic
for r in game24cd_flat_svg2d_s0 game24cd_flat_svg2d_s1 game24cd_flat_svg2d_s2 game24cd_swapMA_svg2d_s0 game24cd_swapMA_svg2d_s1 \
         game24cd_swapMA_svg2d_s2 game24cd_swapAM_svg2d_s0 game24cd_swapMM43_svg2d_s0 game24cd_swapMA43_svg2d_s0 game24cdA_flat_svg2d_s0 game24cd_swapMA44_svg2d_s0 game24cd_swapHMA_svg2d_s0 game24cd_swapHMA_svg2d_s1 game24cdK_hier_svg2d_s0 game24cdK_hier_svg2d_s1 game24cdK_hier_svg2d_s2 kk_flat_svg2d_s0 kk_swapMM7_svg2d_s0 kk_swapMM7_svg2d_s1 kk_swapMM7_svg2d_s2; do
  mkdir -p $DST/rl/$r
  for f in policy_it500.pt policy_it1500.pt policy_best.pt log.jsonl; do
    [ -f outputs/rl/$r/$f ] && rsync -a --ignore-existing outputs/rl/$r/$f $DST/rl/$r/
  done
done
for c in game24cd_flat game24cd_swapMA game24cd_swapAM game24cd_swapMM43 game24cd_swapMA43 game24cdA_flat game24cd_swapMA44 game24cd_swapHMA kk_flat kk_swapMM7; do
  d=outputs/critic/${c}_2d; [ -d $d ] || continue
  best=$(python3 -c "import json,sys; print(json.load(open('$d/log.json'))['best_epoch'])" 2>/dev/null)
  mkdir -p $DST/critic/${c}_2d
  if [ -n "$best" ]; then rsync -a --ignore-existing $d/critic_ep${best}.pt $d/log.json $DST/critic/${c}_2d/; else rsync -a --ignore-existing $d/ $DST/critic/${c}_2d/; fi
done
du -sh $DST; echo BACKUP_DONE
