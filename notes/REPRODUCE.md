# Reproducing every number

This file maps every result to the runs behind it, every run to the exact command that produced it, and every
command to the code, data, configuration and environment it used. It also states what is and is not
bit-reproducible, with measured magnitudes. `notes/results.md` is the dated laboratory log; this file is the index.

## 1. Where the record lives

| what | where |
|---|---|
| per-iteration metrics and evaluations of every RL run | `runs/rl/<tag>/log.jsonl` (also bundled in `runs/rl/rl_logs_all.json.gz`) |
| run → config, base checkpoint, critic, seed, extra args, Slurm job ids, eval curve | `runs/manifest.json` (`scripts/build_manifest.py`) |
| every `python -m hcfrl...` command any job executed, with job id, GPU and (from 2026-11-02) code fingerprint | `runs/commands.jsonl` (`scripts/build_command_ledger.py`) |
| every Slurm log (full command lines via `set -x`, tracebacks, timings) | `runs/slurm/<jobname>-<jobid>.out.gz` |
| critic pre-training curves (per epoch: field score, AUC, Pearson, pick@V per denoising step) | `runs/critic/<tag>_2d/log.json` |
| effective configuration of every supervised run (config file plus command-line overrides such as `seed=7`), as saved by the trainer | `runs/sft_configs/<output_dir>/config.yml` |
| base evaluations and the SFT checkpoint each tag uses | `runs/eval/<tag>_ckpt.txt`, `runs/eval/*` |
| analysis outputs | `runs/analysis/*` (each produced by a script in `hcfrl/analysis/`, see §6) |
| theory sandboxes and closed-form checks | `runs/theory/*` (`hcfrl/theory/`) |
| environment snapshot, sha256 of data, configs and checkpoints | `runs/provenance/` (`scripts/provenance.slurm`) |
| checkpoints | `carc:/scratch1/verajuan/hcfrl/outputs/` (all, purgeable) and `carc:~/hcfrl_backup/` (lean set, backed up; `scripts/backup_checkpoints.sh`) |
| dated log of every experiment and decision | `notes/results.md` |
| theory (working notes) | `notes/theory.md`; paper version `paper/theory.tex` |

Refresh the mirror from a laptop with `bash scripts/archive_runs.sh` (rsyncs logs, analyses, critic logs and Slurm
logs, then rebuilds the manifest and the command ledger). Storage rule: write only to `/scratch1/verajuan` and
`/home1/verajuan`; the group project space is not used.

## 2. Environment

- Cluster: USC CARC, Slurm account `irahbari_832`. Modules `gcc/13.3.0 python/3.11.9 cuda/12.6.3`.
- Python environment: `uv sync --frozen` from `uv.lock` (`scripts/setup_env.sh`), activated by `scripts/common.sh`.
  The exact package list and torch/CUDA/cuDNN versions are in `runs/provenance/environment.txt`.
- Supervised-training jobs echo their command from 2026-11-02; before that, the trainer's saved `config.yml` in each output directory (mirrored to `runs/sft_configs/`) is the record of the effective configuration, including overrides.
- Every job sources `scripts/common.sh`. From 2026-11-02 it prints a `PROVENANCE` line with a 16-hex fingerprint of all
  code, scripts and configs (`sha256` over `hcfrl/ scripts/ configs/`), host, python, torch and the precision flag.
  Earlier jobs are tied to code by date: the repository history (`git log`) is the code record, and
  `runs/commands.jsonl` gives each job's command and id (Slurm ids increase with submission time).
- Precision. A40 runs use bf16 autocast. P100 runs set `HCFRL_FP32=1` (`hcfrl/__init__.py` disables every autocast
  region), because P100s have no bf16. The GPU of each job is recorded in its log and in `runs/commands.jsonl`.

| experiment family | hardware | precision |
|---|---|---|
| all ELF-B supervised bases, critics and RL (Game24, K&K, Countdown), all seeds 0–4 | A40 | bf16 |
| ELF-B mechanism analyses (attacks, cross-critic, refit, pull, spectra, offspan, T1) | A40 (CPU for spectra) | bf16 |
| T2 refit feedback | P100 | fp32 |
| ELF-M supervised bases | A40 (debug and gpu partitions) | bf16 |
| ELF-M base eval, checkpoint selection, critics, RL, screens, hard-set critics and RL | P100 | fp32 |

## 3. Data

- Countdown, Game24 and Knights & Knaves files are in `data/` (generators and exporters in `hcfrl/tasks/`). Game24 is
  split by number set; test has 2,238 prompts. `data/game24/train.jsonl` (rows `{input, output}`) feeds supervised
  training and RL; `data/game24/train_prompts.jsonl` (rows `{input, nums, target}`) feeds rollout collection.
- Coarse maps: `data/clusters_task_arith_n16.pt` (Γ_task), `data/clusters_t5small_n16.pt` (Γ_kmeans), the random map
  used by `helfR` configs.
- Difficulty-matched ELF-M prompt sets: `data/game24_hardM_{flat,hier}/` with `stats.json`, built by
  `hcfrl/analysis/offline/build_hard_set.py` from the per-prompt screens in `outputs/screen/` (chunks in
  `data/game24_screen/`).
- sha256 of every data and config file: `runs/provenance/sha256_data_configs.txt`.

## 4. Pipeline commands

```
# 1. supervised base. Muon: configs/{elf,helf}_b_*; AdamW: *A_*; AdamW+wd: elfAwd_*; ELF-M: *M_*; random/k-means
#    coarse maps: helfR_*, helfK_*; independent Muon model (seed 7): config seed field
sbatch -J sft-cd-<m>    scripts/sft.slurm configs/<m>_b_countdown.yml                 # Countdown, 60 epochs
sbatch -J sft-g24cd-<m> -d afterok:<id> scripts/sft.slurm configs/<m>_b_game24_cd.yml # Game24 from Countdown, 40 epochs

# 2. checkpoint selection (600 val prompts) + base test eval + critic data (2000 prompts x 4) + critic pre-training
sbatch scripts/gate0a.slurm <config> <sft_dir> data/game24 <tag> 0 64 56 1     # select_every=1 picks the SFT checkpoint
#    16k critic: ... <tag>_bigcrit 0 64 56 1 "" 64 56 1,8,16,24,32,40,48 4000
#    P100 / large models: CRIT_EXTRA="--accum 4" ; critic stage alone: scripts/critic_pretrain.slurm <config> <tag> 0 --accum 4

# 3. critic swap: critic data of policy P, trunk from model T
sbatch scripts/critic_swap.slurm <config> <policy_tag> <data_tag> <out_tag> [pretrain_critic extra args]
#    trunk refined with AdamW (+ weight decay): ... --lr_trunk 1e-4 --epochs 12 --wd_trunk {0,1,5,20}

# 4. pathwise RL (SVG-2D), seed s. Debug A40 with self-relaying 1 h links, or P100 2-day jobs
sbatch scripts/gate0b_debug.slurm <config> <sft_dir> data/game24 <tag> <s> 64 56 1500 [out_tag] [train_svg args]
sbatch scripts/gate0b_p100.slurm  <config> <sft_dir> <task_dir>  <tag> <s> 64 56 1500 [out_tag] [train_svg args]
#    ELF-M on 16 GB GPUs: --actor_chunk 2 (micro-batched actor step, same objective)
#    ablations: --lr_actor, --norm_pen, --rel_step, --rsa_rho, --actor_channels, --actor_mode, --log_gcos, frozen head (--mc 0 --mc_extra 0 --a0 0)

# 5. critic-free baseline (CFRL-GRR)
sbatch scripts/gate0c.slurm <config> <sft_dir> data/game24 <tag> 0 64 56 600 <out_tag> --lr 3e-7

# 6. difficulty-matched ELF-M (pre-registered in notes/results.md, 2026-11-01 ~13:45)
sbatch scripts/screen_p100.slurm <config> <ckpt> data/game24_screen/chunk_<i>.jsonl outputs/screen/M_<m>_<i> 2
sbatch -d afterany:<screens> scripts/launch_hard.slurm   # builds data/game24_hardM_<m>, then gate0a_hard_p100 + RL x3
```

The exact command of every run is in `runs/manifest.json` (RL) and `runs/commands.jsonl` (everything).

## 5. Results → runs

| result | run tags in `runs/rl/` |
|---|---|
| headline, Game24, 5 seeds: native, independent-Muon critic, AdamW-trunk critic, AdamW+wd base, H-ELF | `game24cd_{flat,swapMM7,swapMA,hier}_svg2d_s{0..4}`, `game24cdAwd_flat_svg2d_s{0..4}` |
| other Muon bases | `game24cd_{flat,hier,swapMA}{_ft43,_ft44,43,44}_svg2d_s0` (see manifest) |
| K&K | `kk_{flat,hier,swapMM7}_svg2d_s{0,1,2}`, `kk_swapMA*` (weak-critic, stopped) |
| Countdown | `countdown_{flat,hier}{,_c16}_svg2d_s0` |
| critic-free GRR | `*_grr_lr3e-7_s0` (used), `*_grr_s0` (published lr, diverges) |
| remedies that fail | `game24cd_flat_svg2d_{lr1e-7,lr3e-7,relstep,np10,np100,rsa1e-3}_s0`, `game24cd_flat_bigcrit_svg2d_s{0,1,2}`, `game24cd_flat_svg2d_refit8_s{0,1,2}` |
| step-size margin | above plus `game24cd_hier_svg2d_lr{1e-7,3e-7,3e-6}_s0`, `game24cdA_flat_svg2d_{s0,s1,s2,lr1e-5_s0}`, `game24cdA_flat60_svg2d_s0` |
| AdamW policies | `game24cdA_{flat,hier}_svg2d_s{0,1,2}`, `game24cdA_*_grr_*`, reverse pairing `game24cd_swapAM_svg2d_s0` |
| hierarchy controls | `game24cd_hier_{bigcrit,blind}_svg2d_s{0,1,2}`, `game24cd_hier_svg2d_{tokgrad,clgrad}_s0`, `game24cdK_hier_svg2d_s{0,1,2}`, `game24cd_{rand,randm}_svg2d_s{0,1,2}`, `game24cd_swapHMA_svg2d_s{0,1}` |
| refined / weight-decayed critic trunks | `game24cd_flatAT_svg2d_s0`, `game24cd_flatATwd{,5,20}_svg2d_s*` |
| frozen head, gradient coherence | `game24cd_swapMM7_frozen_s0`, `game24cd_{flat,swapMA}_gcos_s0` |
| sibling Muon trunk | `game24cd_swapMM43_svg2d_s0` (stopped at ≈ it 430) |
| early SFT checkpoint | `game24cd_flat_early_svg2d_s0` |
| ELF-M same recipe | `game24cdM_{flat,hier}_svg2d_s{0,1,2}` |
| ELF-M matched reward variance | `game24cdMhard_{flat,hier}_svg2d_s{0,1,2}` |
| replicate runs (nondeterminism, §7) | `repro_{flat,swapMA}_s0_{a,b}` |

Per-seed values, every evaluation and best-validation checkpoints: `python -m hcfrl.analysis.offline.seed_table <tag> ...`
(Game24 headline), or the full table built by `scripts/make_appendix_figs.py`.

## 6. Analyses → scripts → outputs

| analysis | script | output in `runs/analysis/` |
|---|---|---|
| frozen-critic attacks, attack transfer, self-consistency (T1) | `hcfrl/analysis/attack_transfer.py` (`--self_consistency 1` for T1) | `attack600_*.json`, `attack_*.json`, `T1_selfconsistency_native.json` |
| refit feedback along the drift (T2) | `refit_feedback.py` | `T2_refit_feedback.json` |
| critics on every policy's states, gradient agreement | `cross_critic.py`, summary `offline/cross_summary.py` | `cross_critic_g24.pt` (cluster only, 190 MB), `cross_critic_g24_summary.txt` |
| one actor step along each critic, parameter-space agreement | `multi_onestep.py` | `multi_onestep_g24.json` |
| pull effects | `pull_effects.py` | `pull_effects_g24.json` |
| refit churn | `refit_churn.py` | `refit_churn_g24.json` |
| weight and update spectra | `spectra.py` | `spectra_g24.pt`, `spectra_flatAT.pt` |
| off-manifold geometry | `offspan.py` | `offspan_g24.pt` |
| one-step test, gradient SNR, failure modes, commitment, consistency, Jacobian step | `one_step.py`, `grad_snr.py`, `failure_modes.py`, `commitment.py`, `consistency.py`, `jacobian_step.py` | `onestep_*`, `gsnr_*`, `fm_*`, `commit_*`, `cons_*`, `jac_*` |
| base critic sensitivity | `scripts/sharpness.slurm` (train_svg at lr 0 with `--rsa_rho 1e-3 --rsa_apply 0`) | `sharp_*/log.jsonl` |
| parameter displacement per iteration | `offline/step_size.py` | Slurm log `an-stepsize-*` |
| theory sandboxes | `hcfrl/theory/*.py` | `runs/theory/*` |

Figures and generated tables of the paper: `scripts/fig_seed_curves.py` (Fig. 1) and `scripts/make_appendix_figs.py`
(all appendix figures and the every-run table), both reading only `runs/`.

## 7. What is reproducible, and how exactly

Seeded: supervised training (config `seed`), rollout collection (`--seed`), critic pre-training (`--seed`, epoch
batch order derived from it), RL (`--seed` sets Python `random`, the torch CPU generator used for all sampling noise,
and prompt order; evaluation uses a separate fixed generator so every evaluation of a run sees the same noise).
Resumed RL runs restore every random-number state from `state.pt` (saved every 50 iterations), so a restart continues
the same stream.

Not bit-reproducible:
1. GPU kernels. Deterministic algorithms are not enabled, and bf16 accumulation order varies between runs, so a
   rerun with the same seed reproduces a run statistically, not bit for bit. The size of this effect is measured by
   rerunning `game24cd_flat_svg2d_s0` and `game24cd_swapMA_svg2d_s0` twice each with identical seeds to iteration 250
   (`repro_*`, see `notes/results.md`).
2. Code changes during the project that alter numerics for runs started after them:
   - 2026-10-25 (`d55f47c`): fix of the channel-ablation gradient (ablation runs before it were void and were rerun).
   - 2026-10-27 (`5eb3d78`): critic-feature replay buffer fp16 → bf16 (runs before this used fp16; Muon trunks are
     within fp16 range, the AdamW trunk overflowed and its run was rerun).
   - 2026-11-01 (`02449ae`): micro-batched actor step for ELF-M (same objective, different summation order and
     random-number consumption).
   - The iteration-0 test accuracy of the same seed-42 flat base is 64.2 in the run of 2026-10-24 and 63.9 in later
     runs, a 7-prompt difference attributed to the changes above; every Δ is measured against the run's own iteration 0.
3. Hardware and precision differ between experiment families (§2). Comparisons are always made within a family.

## 8. Fixed choices that matter

- Sampler γ56@64, guidance 2 (critic evaluated without guidance); argmax decoding.
- Critic: 2D head on a copy of a trunk, 8,000 rollouts (2,000 prompts × 4) at steps {1,8,16,24,32,40,48}, 6 epochs,
  best epoch by held-out field score; RL loads that epoch.
- RL: CFRL Algorithm 1 defaults (`hcfrl/rl/train_svg.py` argparse defaults), actor lr 1e-6 → 1e-7 cosine,
  32 prompts per iteration, 1,500 iterations (1,000 on K&K), evaluation every 250 iterations on the full test set.
- Deviations from CFRL: Muon supervised training (AdamW comparisons run separately), self-generated Countdown data,
  GRR at lr 3e-7 because the published 3e-6 diverges on our models.

## 9. Re-running from scratch

1. `git clone` the repository, `bash scripts/setup_env.sh` on the cluster, download the ELF-B/ELF-M OpenWebText
   checkpoints into `HF_HOME`.
2. Check data and configs against `runs/provenance/sha256_data_configs.txt`.
3. Run §4 for the family of interest; the manifest gives the exact arguments of every run.
4. Compare per-evaluation test accuracy with `runs/rl/<tag>/log.jsonl`; differences within the replicate spread of §7
   are expected.
