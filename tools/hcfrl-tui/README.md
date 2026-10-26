# hcfrl — experiment dashboard

Interactive terminal dashboard for every job and run of the hierarchical-cfrl project on CARC.

```
uv tool install --editable tools/hcfrl-tui   # once (from the repo root); needs `uv`
hcfrl                                        # open it (needs the USC VPN; falls back to the cached snapshot)
hcfrl --offline                              # cached snapshot only
hcfrl --plain                                # one-shot text summary
```

Keys: `↑/↓` move · `Enter` open the selected run / model / critic / job · `Space` mark a run · `c` compare marked runs (or the run under the cursor with its flat/hier twin) · `/` filter · `r` refresh · `1`–`5` switch tabs · `a` show smoke/probe runs · `Esc` back · `q` quit.

Tabs: **runs** (RL runs: base → last accuracy, Δ, sparkline, live/done state, buffer success, grad norm), **sft** (checkpoint progress and loss), **critics** (pretrained value heads with held-out AUC), **queue** (Slurm, Enter shows the job's log tail), **analysis** (diagnostic JSONs).

Data comes from `scripts/status.py --remote --full` run on the login node over ssh (stdlib only, Python 3.6 safe); it auto-refreshes every 5 minutes (`HCFRL_REFRESH` seconds) and caches to `~/.cache/hcfrl/last.json`.
