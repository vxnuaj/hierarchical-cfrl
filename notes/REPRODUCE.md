# Reproducing every number

Every number in `notes/results.md` (and any table built from it) comes from a run whose record is in this
repository under `runs/`, and whose checkpoints are on the cluster (scratch) with a lean copy on persistent
project storage. This file maps results to runs, runs to commands, and commands to configs.

## Where the record lives

| what | where | notes |
|---|---|---|
| per-iteration training metrics + evals of every RL run | `runs/rl/<tag>/log.jsonl` | one JSON per line; eval records have `val_acc`/`test_acc` every 250 iters (100 for GRR) |
| run → config / base ckpt / critic / seed / args / job id / curve | `runs/manifest.json` | built by `scripts/build_manifest.py` from the Slurm logs' `set -x` command lines |
| critic pre-training curves (per epoch, per step k: AUC, Pearson, pick@V) | `runs/critic/<tag>_2d/log.json` | |
| base-model evaluations and which SFT checkpoint was selected | `runs/eval/<tag>_ckpt.txt`, `runs/eval/*` | |
| diagnostics (one-step, gradient SNR, failure modes, commitment, probe, trace, Jacobian, survival) | `runs/analysis/*.json` | each produced by a script in `hcfrl/analysis/` |
| every Slurm log (exact command lines, tracebacks, timings) | `runs/slurm/<jobname>-<jobid>.out.gz` | |
| checkpoints: selected SFT bases, best critic epochs, final + best policy of each run | `carc:/project2/irahbari_832/hcfrl_backup/` (persistent) and `carc:/scratch1/verajuan/hcfrl/outputs/` (all intermediate checkpoints, purgeable) | `scripts/backup_checkpoints.sh` |
| running log of experiments with interpretation, dated | `notes/results.md` | |

Refresh the mirror from a laptop with `bash scripts/archive_runs.sh` (needs the `carc` ssh alias).

## Pipeline (one model, one task)

```
# 1. supervised base (Muon recipe; AdamW recipe = configs/*A_* ; ELF-M = configs/*M_* ; random clusters = helfR ; k-means clusters = helfK)
sbatch -J sft-cd-<m>   scripts/sft.slurm configs/<m>_b_countdown.yml          # Countdown, 60 epochs
sbatch -J sft-g24cd-<m> -d afterok:<id> scripts/sft.slurm configs/<m>_b_game24_cd.yml   # Game24 from the Countdown model

# 2. base eval + rollout collection + critic pre-training (writes outputs/eval/<tag>_ckpt.txt and outputs/critic/<tag>_2d/)
sbatch -J gate0a-<tag> -d afterok:<id> scripts/gate0a.slurm <config> <sft_dir> data/game24 <tag> 0 64 56 1
#   args: config sft_ckpt_dir task_dir tag seed K gamma select_every(1 = pick SFT ckpt by val acc) [select_target] [crit_K crit_gamma crit_steps crit_n]
#   16k critic:  ... <tag>_bigcrit 0 64 56 1 "" 64 56 1,8,16,24,32,40,48 4000
#   Countdown cheap-sampler critic (CFRL): ... countdown_<m>_c16 0 64 56 0 16 4 1,2,4,6,8,10,12,14 4000

# 3. pathwise RL (CFRL SVG-2D), seed s
sbatch -J gate0b-<tag>-s<s> -d afterok:<id> scripts/gate0b.slurm <config> <sft_dir> data/game24 <tag> <s> 64 56 1500 [out_tag] [train_svg extra args]
#   ablations: extra args --actor_channels tokens|clusters ; --lr_actor 3e-7 --lr_actor_min 3e-8 ; --actor_mode random|none
#   cluster-blind critic: pre-train with `pretrain_critic --blind_clusters` (tag <tag>_blind), then gate0b with that tag

# 4. score-function baseline (CFRL-GRR)
sbatch -J gate0c-<tag> -d afterok:<id> scripts/gate0c.slurm <config> <sft_dir> data/game24 <tag> 0 64 56 600 <out_tag> --lr 3e-7
```

K&K: `configs/{elf,helf}_b_kk.yml`, task dir `data/kk`, epoch-4 SFT checkpoint, 1000 iterations.
Countdown: `configs/{elf,helf}_b_countdown.yml`, task dir `data/countdown`.

## Results → runs

| result (notes/results.md) | run tags (`runs/rl/`) |
|---|---|
| Game24-cd, SVG-2D, 3 seeds (flat −11.7 / hier +4.8) | `game24cd_{flat,hier}_svg2d_s{0,1,2}` |
| K&K, SVG-2D, 3 seeds (−8.7 / −1.5) | `kk_{flat,hier}_svg2d_s{0,1,2}` |
| Countdown, cheap-sampler critic (+2.6 / +1.3) | `countdown_{flat,hier}_c16_svg2d_s0` ; deployment-sampler critic (both collapse): `countdown_{flat,hier}_svg2d_s0` |
| GRR lr 3e-7 (+4.6 / +4.1; Countdown neutral) | `game24cd_{flat,hier}_grr_lr3e-7_s0`, `countdown_*_grr_lr3e-7_s0` ; paper lr (diverges): `*_grr_s0` |
| 16k critic (flat −5.7 mean; hier +0.2) | `game24cd_flat_bigcrit_svg2d_s{0,1,2}`, `game24cd_hier_bigcrit_svg2d_s{0,1,2}` |
| cluster-blind critic (hier +1.8 mean) | `game24cd_hier_blind_svg2d_s{0,1,2}` |
| channel ablations (tokens-only +3.4, clusters-only −1.7) | `game24cd_hier_svg2d_tokgrad_s0`, `game24cd_hier_svg2d_clgrad_s0` |
| learning-rate checks | `game24cd_flat_svg2d_lr3e-7_s0`, `game24cd_hier_svg2d_lr3e-{7,6}_s0` |
| random-direction / no-actor diagnostics (K&K) | `kk_hier_diag_random`, `kk_hier_diag_none` |
| random-cluster control | `game24cd_rand_svg2d_s{0,1,2}` (running) |
| k-means hierarchy-design control | `game24cdK_hier_svg2d_s{0,1,2}` (queued) |
| AdamW-recipe bases | `game24cdA_{flat,hier}_*` (queued) |
| ELF-M | `game24cdM_{flat,hier}_svg2d_s0` (queued) |
| one-step test | `runs/analysis/onestep_*.json` (`hcfrl/analysis/one_step.py`, `scripts/one_step.slurm`) |
| gradient SNR / channel share | `runs/analysis/gsnr_*.json` (`grad_snr.py`, `grad_channels.py`) |
| failure modes | `runs/analysis/fm_*.json` (`failure_modes.py`) |
| commitment profiles | `runs/analysis/commit_*.json` (`commitment.py`) |
| critic probe / consistency trace / survival | `runs/analysis/cons_probe_*.json`, `cons_trace*.json`, `cons_survive_*.json` (`consistency.py`, `scripts/consistency.slurm {probe,trace,trace2,survive}`) |
| Jacobian-step test | `runs/analysis/jac_*.json` (`jacobian_step.py`, `scripts/jacobian.slurm`) |

## Fixed choices that matter

- Sampler γ56@64, guidance 2, self-conditioning guidance 1 (CFRL deployment sampler); critic copy evaluated without guidance.
- Critic: 2D head, 8,000 rollouts (2,000 prompts × 4) at steps {1,8,16,24,32,40,48}, 6 epochs, best epoch by held-out field score.
- RL: CFRL Algorithm 1 defaults (`hcfrl/rl/train_svg.py` argparse defaults = paper values); evaluation on the full held-out test set under the deployment sampler, final checkpoint reported.
- Seeds: RL seed s controls prompt order, noise and the critic buffer; the evaluation noise is fixed (seed 777) for paired comparisons in the analyses.
- Test sets: Game24 2,238 prompts (269 held-out number sets), K&K and Countdown as in `data/`.
- Deviations from CFRL: Muon SFT (AdamW chains in progress); 8k critic (16k control done); Countdown data self-generated.
