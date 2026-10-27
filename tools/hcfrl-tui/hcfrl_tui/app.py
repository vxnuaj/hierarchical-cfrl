"""hcfrl — interactive terminal dashboard for the hierarchical-cfrl experiments on CARC.

    hcfrl                 open the dashboard (fetches from the cluster over ssh, auto-refreshes)
    hcfrl --offline       render the last cached snapshot without touching the cluster
    hcfrl --plain         one-shot plain-text summary (no TUI)

Keys: ↑/↓ move · Enter open · Space mark · c compare marked runs · / filter · r refresh · 1-5 tabs · a show all · q quit
"""
from __future__ import annotations

import math
import argparse, json, os, subprocess, sys, time
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Input, Static, TabbedContent, TabPane
from textual_plotext import PlotextPlot

SSH_HOST = os.environ.get("HCFRL_HOST", "carc")
REMOTE_DIR = os.environ.get("HCFRL_DIR", "/scratch1/verajuan/hcfrl")
CACHE = Path(os.environ.get("HCFRL_CACHE", Path.home() / ".cache" / "hcfrl" / "last.json"))
REFRESH_S = int(os.environ.get("HCFRL_REFRESH", "300"))
BARS = "▁▂▃▄▅▆▇█"

# ----------------------------------------------------------------------------- data
def fetch_remote(timeout=90) -> dict:
    cmd = ["ssh", "-o", "ConnectTimeout=20", SSH_HOST, f"cd {REMOTE_DIR} && python3 scripts/status.py --remote --full"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode().strip().splitlines()[-1] if p.stderr.strip() else "ssh failed")
    d = json.loads(p.stdout.decode())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d))
    return d

def load_cache() -> dict | None:
    try: return json.loads(CACHE.read_text())
    except (OSError, ValueError): return None

def fmt_age(a):
    if a is None: return "?"
    a = int(a)
    if a < 60: return f"{a}s"
    if a < 3600: return f"{a // 60}m"
    if a < 86400: return f"{a / 3600:.1f}h"
    return f"{a / 86400:.1f}d"

def spark(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return ""
    lo, hi = min(vals), max(vals); rng = (hi - lo) or 1e-9
    return "".join(BARS[min(7, int((v - lo) / rng * 7.999))] for v in vals)

def target_iters(name):
    if "grr" in name: return 600
    if name.startswith("kk"): return 1000
    return 1500

def task_of(name): return name.split("_")[0]

def fmt_dur(sec):
    if sec is None: return "?"
    sec = int(max(0, sec))
    if sec < 3600: return f"{sec // 60}m"
    if sec < 86400: return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"
    return f"{sec // 86400}d{(sec % 86400) // 3600}h"

def bar(frac, width=10):
    frac = max(0.0, min(1.0, frac)); n = int(frac * width)
    return "█" * n + "░" * (width - n) + f" {frac * 100:3.0f}%"

def run_eta(r):
    """(fraction done, seconds left) from the last train record's iter/elapsed; None if unknown"""
    l = r.get("last") or {}
    it, el = l.get("iter"), l.get("elapsed"); tgt = target_iters(r["name"])
    if not it or not el: return None, None
    return min(1.0, (it + 1) / tgt), max(0.0, (tgt - it - 1) * el / (it + 1))

def sft_eta(s):
    tot, sps = s.get("total"), s.get("sps")
    if not tot: return None, None
    frac = min(1.0, s["step"] / tot)
    return frac, ((tot - s["step"]) / sps if sps else None)

def parse_slurm_time(t):
    """'1-16:00:00' / '2:51:08' / '26:52' -> seconds"""
    try:
        d = 0
        if "-" in t: d, t = t.split("-"); d = int(d)
        parts = [int(x) for x in t.split(":")]
        while len(parts) < 3: parts.insert(0, 0)
        return d * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    except ValueError: return None

def run_state(r) -> tuple[str, str]:
    """(label, rich style)"""
    ev = [e for e in r["evals"] if e[1] is not None]
    it = ev[-1][0] if ev else 0
    age = r.get("age")
    if it >= target_iters(r["name"]): return "DONE", "blue"
    if age is not None and age < 1200: return "LIVE", "bold green"
    if age is not None and age > 6 * 3600: return "ENDED", "dim"
    return "STALL", "yellow"

def delta_text(x):
    t = Text(f"{x:+.1f}")
    t.stylize("green" if x > 1 else "red" if x < -1 else "")
    return t

# ----------------------------------------------------------------------------- plots
def _xy(points, xi, yi):
    xs, ys = [], []
    for p in points:
        if p[xi] is not None and p[yi] is not None: xs.append(p[xi]); ys.append(p[yi])
    return xs, ys

PALETTE = ["cyan+", "orange+", "green+", "magenta+", "yellow+", "blue+", "red+", "white"]

def plot_lines(widget: PlotextPlot, series, title, xlabel="", ylabel="", ylog=False, dots=False):
    """series: list of (label, xs, ys); dots=True also marks the actual data points (sparse evals)"""
    plt = widget.plt
    plt.clear_figure(); plt.clear_data()
    any_data = False
    for i, (label, xs, ys) in enumerate(series):
        if ylog:   # plotext 5.3 applies yscale("log") in place on every build (and the widget rebuilds on each
            # render/resize), which raises a math domain error the second time; so take log10 ourselves and
            # draw on a linear axis
            pts = [(x, math.log10(y)) for x, y in zip(xs, ys) if y is not None and y > 0]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if xs:
            any_data = True; col = PALETTE[i % len(PALETTE)]
            plt.plot(xs, ys, label=label, marker="braille", color=col)
            if dots: plt.scatter(xs, ys, marker="●", color=col)
    plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if ylog: plt.ylabel(("log10 " + ylabel).strip())
    if not any_data: plt.title(title + "  (no data)")
    widget.refresh()

# ----------------------------------------------------------------------------- screens
class RunDetail(Screen):
    BINDINGS = [Binding("escape,q", "app.pop_screen", "back"), Binding("1,2,3", "tab", "tab", show=False)]

    def __init__(self, run: dict):
        super().__init__(); self.run = run

    def compose(self) -> ComposeResult:
        r = self.run; ev = [e for e in r["evals"] if e[1] is not None]
        base = ev[0][1] * 100 if ev else float("nan"); last = ev[-1][1] * 100 if ev else float("nan")
        st, _ = run_state(r)
        l = dict(r.get("last") or {})
        tr = r.get("train") or []
        aucs = [p[4] for p in tr if p[4] is not None]
        if aucs: l["critic_auc"] = aucs[-1]
        frac, left = run_eta(r)
        prog = f"  {bar(frac)}  ~{fmt_dur(left)} left" if (st == "LIVE" and frac is not None) else ""
        yield Static(Text.assemble((f" {r['name']}  ", "bold"), (f"{st}  ", "cyan"), (prog, "green"),
                                   f"test {base:.1f} → {last:.1f}  (Δ {last - base:+.1f})  iter {l.get('iter', '?')}  "
                                   f"buf succ {l.get('buf_succ') or 0:.2f}  grad {l.get('grad_norm') or 0:.0f}  "
                                   f"critic auc {l.get('critic_auc') or 0:.2f}  last write {fmt_age(r.get('age'))} ago"),
                     id="dtitle")
        with TabbedContent(id="dtabs"):
            with TabPane("Accuracy", id="t1"): yield PlotextPlot(id="p_acc")
            with TabPane("Actor", id="t2"):
                with Vertical():
                    yield PlotextPlot(id="p_obj"); yield PlotextPlot(id="p_grad")
            with TabPane("Critic", id="t3"):
                with Vertical():
                    yield PlotextPlot(id="p_buf"); yield PlotextPlot(id="p_auc")
        yield Footer()

    def on_mount(self):
        r = self.run; ev = [e for e in r["evals"] if e[1] is not None]; tr = r.get("train") or []
        its = [e[0] for e in ev]
        plot_lines(self.query_one("#p_acc", PlotextPlot),
                   [("test", its, [e[1] * 100 for e in ev]), ("val", its, [e[2] * 100 for e in ev if e[2] is not None])],
                   "accuracy (%) vs iteration", "iter", dots=True)
        plot_lines(self.query_one("#p_obj", PlotextPlot), [("actor objective (pooled logit)", *_xy(tr, 0, 1))], "actor objective", "iter")
        plot_lines(self.query_one("#p_grad", PlotextPlot), [("grad norm (pre-clip)", *_xy(tr, 0, 2))], "gradient norm", "iter", ylog=True)
        plot_lines(self.query_one("#p_buf", PlotextPlot), [("buffer success (batch)", *_xy(tr, 0, 3))], "replay-buffer success rate", "iter")
        plot_lines(self.query_one("#p_auc", PlotextPlot), [("critic AUC", *_xy(tr, 0, 4)), ("critic loss", *_xy(tr, 0, 5))], "critic refit", "iter")

    def action_tab(self):
        pass

    def on_key(self, event):
        if event.key in ("1", "2", "3"):
            self.query_one("#dtabs", TabbedContent).active = f"t{event.key}"


class CompareScreen(Screen):
    BINDINGS = [Binding("escape,q", "app.pop_screen", "back")]

    def __init__(self, runs: list[dict]):
        super().__init__(); self.runs = runs

    def compose(self) -> ComposeResult:
        yield Static(Text(" compare: " + ", ".join(r["name"] for r in self.runs), style="bold"), id="dtitle")
        with Vertical():
            yield PlotextPlot(id="c_acc"); yield PlotextPlot(id="c_buf")
        yield Footer()

    def on_mount(self):
        acc, buf = [], []
        for r in self.runs:
            ev = [e for e in r["evals"] if e[1] is not None]
            acc.append((r["name"], [e[0] for e in ev], [e[1] * 100 for e in ev]))
            buf.append((r["name"], *_xy(r.get("train") or [], 0, 3)))
        plot_lines(self.query_one("#c_acc", PlotextPlot), acc, "test accuracy (%) vs iteration", "iter", dots=True)
        plot_lines(self.query_one("#c_buf", PlotextPlot), buf, "replay-buffer success rate", "iter")


class SftDetail(Screen):
    BINDINGS = [Binding("escape,q", "app.pop_screen", "back")]

    def __init__(self, s: dict):
        super().__init__(); self.s = s

    def compose(self) -> ComposeResult:
        s = self.s; tot = f"/{s['total']}" if s.get("total") else ""
        frac, left = sft_eta(s); live = s.get("age") is not None and s["age"] < 1200
        prog = f"   {bar(frac, 20)}" + (f"   ~{fmt_dur(left)} left" if (live and left is not None and frac < 0.999) else "") if frac is not None else ""
        yield Static(Text(f" {s['name']}   step {s['step']}{tot}{prog}   last write {fmt_age(s.get('age'))} ago", style="bold"), id="dtitle")
        with Vertical():
            yield PlotextPlot(id="s_loss"); yield PlotextPlot(id="s_lr")
        yield Footer()

    def on_mount(self):
        pts = self.s.get("loss") or []
        plot_lines(self.query_one("#s_loss", PlotextPlot), [("loss", *_xy(pts, 0, 1)), ("ce", *_xy(pts, 0, 2))],
                   "SFT loss (log scale)", "step", ylog=True)
        plot_lines(self.query_one("#s_lr", PlotextPlot), [("lr", *_xy(pts, 0, 3))], "learning rate", "step")


class CriticDetail(Screen):
    BINDINGS = [Binding("escape,q", "app.pop_screen", "back")]

    def __init__(self, c: dict):
        super().__init__(); self.c = c

    def compose(self) -> ComposeResult:
        c = self.c
        yield Static(Text(f" critic {c['name']}   epochs {c['epochs']}   best epoch {c.get('best')}", style="bold"), id="dtitle")
        with Vertical():
            yield PlotextPlot(id="k_auc"); yield PlotextPlot(id="k_pear")
        yield Footer()

    def on_mount(self):
        auc, pear = [], []
        for e in self.c.get("log") or []:
            ks = sorted(e["ks"].items(), key=lambda kv: int(kv[0][1:]))
            xs = [int(k[1:]) for k, _ in ks]
            auc.append((f"ep{e['epoch']}", xs, [v["auc"] or 0 for _, v in ks]))
            pear.append((f"ep{e['epoch']}", xs, [v["pearson"] or 0 for _, v in ks]))
        plot_lines(self.query_one("#k_auc", PlotextPlot), auc, "held-out AUC by sampler step k (one line per epoch)", "k")
        plot_lines(self.query_one("#k_pear", PlotextPlot), pear, "Pearson(value, outcome) by k", "k")


class JobDetail(ModalScreen):
    BINDINGS = [Binding("escape,q", "app.pop_screen", "back")]

    def __init__(self, job: dict, tail: list[str]):
        super().__init__(); self.job = job; self.tail = tail

    def compose(self) -> ComposeResult:
        j = self.job
        body = "\n".join(self.tail) if self.tail else "(no log yet — job has not started)"
        yield Vertical(Static(Text(f" {j['name']}  job {j['id']}  {j['state']}  {j['time']} / {j['limit']}  {j['reason']}", style="bold")),
                       Static(body, id="logtail"), id="jobbox")


class FilterInput(Input):
    pass

# ----------------------------------------------------------------------------- main app
class HcfrlApp(App):
    TITLE = "hcfrl"
    CSS = """
    #status { height: 1; background: $primary-background; color: $text; padding: 0 1; }
    #filter { display: none; height: 3; }
    #filter.shown { display: block; }
    DataTable { height: 1fr; }
    #dtitle { height: 1; background: $primary-background; padding: 0 1; }
    #dtabs { height: 1fr; }
    #dtabs ContentSwitcher { height: 1fr; }
    #dtabs TabPane { height: 1fr; padding: 0; }
    PlotextPlot { height: 1fr; }
    #jobbox { width: 90%; height: 80%; border: round $accent; background: $surface; padding: 1 2; }
    #logtail { height: 1fr; overflow-y: auto; }
    """
    BINDINGS = [
        Binding("q", "quit", "quit"), Binding("r", "refresh", "refresh"), Binding("slash", "filter", "filter"),
        Binding("enter", "open", "open"), Binding("space", "mark", "mark"), Binding("c", "compare", "compare marked"),
        Binding("a", "toggle_all", "show all", show=False),
        Binding("1", "tab('runs')", "runs", show=False), Binding("2", "tab('sft')", "sft", show=False),
        Binding("3", "tab('critics')", "critics", show=False), Binding("4", "tab('queue')", "queue", show=False),
        Binding("5", "tab('analysis')", "analysis", show=False),
    ]

    def __init__(self, offline=False):
        super().__init__()
        self.offline = offline; self.data: dict | None = None; self.marks: set[str] = set()
        self.filter = ""; self.show_all = False; self.busy = False

    def compose(self) -> ComposeResult:
        yield Static("", id="status")
        yield FilterInput(placeholder="filter by name (esc to close)", id="filter")
        with TabbedContent(id="tabs"):
            with TabPane("1 runs", id="runs"): yield DataTable(id="t_runs", cursor_type="row", zebra_stripes=True)
            with TabPane("2 sft", id="sft"): yield DataTable(id="t_sft", cursor_type="row", zebra_stripes=True)
            with TabPane("3 critics", id="critics"): yield DataTable(id="t_critics", cursor_type="row", zebra_stripes=True)
            with TabPane("4 queue", id="queue"): yield DataTable(id="t_queue", cursor_type="row", zebra_stripes=True)
            with TabPane("5 analysis", id="analysis"): yield DataTable(id="t_analysis", cursor_type="row", zebra_stripes=True)
        yield Footer()

    def on_mount(self):
        self.query_one("#t_runs", DataTable).add_columns("", "run", "base", "last", "iter", "Δ", "trajectory", "state", "progress", "left", "buf", "grad", "write")
        self.query_one("#t_sft", DataTable).add_columns("model", "step", "progress", "left", "loss", "ce", "lr", "write", "state")
        self.query_one("#t_critics", DataTable).add_columns("critic", "epochs", "best", "auc@k8", "auc@k24", "auc@k40", "state", "left", "write")
        self.query_one("#t_queue", DataTable).add_columns("job", "name", "st", "elapsed", "limit", "left / est. start", "node / reason")
        self.query_one("#t_analysis", DataTable).add_columns("file", "age")
        cached = load_cache()
        if cached: self.data = cached; self.render_all()
        if not self.offline:
            self.action_refresh(); self.set_interval(REFRESH_S, self.action_refresh)
        self.query_one("#t_runs", DataTable).focus()

    # ---- data refresh
    def action_refresh(self):
        if self.busy or self.offline: return
        self.busy = True; self.update_status("refreshing…")
        self.do_fetch()

    @work(thread=True, exclusive=True)
    def do_fetch(self):
        try:
            d = fetch_remote(); err = None
        except Exception as e:
            d = None; err = str(e)[:100]
        self.call_from_thread(self.on_fetched, d, err)

    def on_fetched(self, d, err):
        self.busy = False
        if d: self.data = d; self.render_all()
        else: self.update_status(f"cluster unreachable ({err}) — on USC VPN? showing cached data", error=True)

    def update_status(self, note="", error=False):
        d = self.data; parts = [Text("hcfrl", style="bold")]
        if d:
            q = d["queue"]; nr = sum(j["state"] == "R" for j in q); npd = sum(j["state"] == "PD" for j in q)
            parts += [Text(f"  snapshot {time.strftime('%H:%M', time.localtime(d['ts']))} ({fmt_age(time.time() - d['ts'])} old)"),
                      Text(f"  running {nr}  pending {npd}", style="cyan"),
                      Text(f"  marked {len(self.marks)}" if self.marks else "")]
        if self.filter: parts.append(Text(f"  filter: {self.filter}", style="yellow"))
        if note: parts.append(Text("  " + note, style="red" if error else "dim"))
        self.query_one("#status", Static).update(Text.assemble(*parts))

    # ---- tables
    def _keep(self, name): return (not self.filter or self.filter in name)

    def render_all(self):
        d = self.data
        if not d: return
        self.update_status()
        t = self.query_one("#t_runs", DataTable); cur = t.cursor_row; t.clear()
        rl = [r for r in d["rl"] if self._keep(r["name"]) and (self.show_all or not r["name"].startswith(("probe", "smoke")))]
        for r in sorted(rl, key=lambda r: (task_of(r["name"]), r["name"])):
            ev = [e for e in r["evals"] if e[1] is not None]
            st, style = run_state(r); l = r.get("last") or {}
            if ev:
                base, last, it = ev[0][1] * 100, ev[-1][1] * 100, ev[-1][0]
                cells = (f"{base:.1f}", f"{last:.1f}", str(it), delta_text(last - base), spark([e[1] for e in ev]))
            else:
                cells = ("", "", "", Text(""), "")
            frac, left = run_eta(r); live = st == "LIVE"
            t.add_row("●" if r["name"] in self.marks else "", r["name"], *cells, Text(st, style=style),
                      bar(frac) if (live and frac is not None) else "", fmt_dur(left) if (live and left is not None) else "",
                      f"{l.get('buf_succ'):.2f}" if l.get("buf_succ") is not None else "",
                      f"{l.get('grad_norm'):.0f}" if l.get("grad_norm") is not None else "", fmt_age(r.get("age")), key=r["name"])
        if t.row_count: t.move_cursor(row=min(cur, t.row_count - 1))

        t = self.query_one("#t_sft", DataTable); cur = t.cursor_row; t.clear()
        for s in d["sft"]:
            if not self._keep(s["name"]): continue
            live = s.get("age") is not None and s["age"] < 1200
            tot = s.get("total"); frac, left = sft_eta(s)
            lp = (s.get("loss") or [None])[-1]
            t.add_row(s["name"], f"{s['step']}" + (f"/{tot}" if tot else ""), bar(frac, 20) if frac is not None else "",
                      fmt_dur(left) if (live and left is not None and frac < 0.999) else "",
                      f"{lp[1]:.4f}" if lp else "", f"{lp[2]:.4f}" if lp else "", f"{lp[3]:.1e}" if lp else "",
                      fmt_age(s.get("age")), Text("training", style="bold green") if live else Text("done" if frac and frac >= 0.999 else "idle", style="dim"), key=s["name"])
        if t.row_count: t.move_cursor(row=min(cur, t.row_count - 1))

        t = self.query_one("#t_critics", DataTable); cur = t.cursor_row; t.clear()
        for c in d["critic"]:
            if not self._keep(c["name"]): continue
            best = next((e for e in (c.get("log") or []) if e["epoch"] == c.get("best")), None)
            def a(k): v = (best or {}).get("ks", {}).get(k); return f"{v['auc']:.2f}" if v and v.get("auc") is not None else ""
            left = (6 - c["epochs"]) * c["epoch_sec"] if (not c["done"] and c.get("epoch_sec")) else None
            t.add_row(c["name"], f"{c['epochs']}/6", str(c.get("best") or ""), a("k8"), a("k24"), a("k40"),
                      Text("done", style="blue") if c["done"] else Text("training", style="bold green"),
                      fmt_dur(left) if left is not None else "", fmt_age(c.get("age")), key=c["name"])
        if t.row_count: t.move_cursor(row=min(cur, t.row_count - 1))

        t = self.query_one("#t_queue", DataTable); cur = t.cursor_row; t.clear()
        for j in sorted(d["queue"], key=lambda j: (j["state"] != "R", j["name"])):
            if not self._keep(j["name"]): continue
            if j["state"] == "R":
                used, lim = parse_slurm_time(j["time"]), parse_slurm_time(j["limit"])
                eta = f"{fmt_dur(lim - used)} to limit" if used is not None and lim is not None else ""
            else:
                eta = ("starts ~" + j["start"].replace("T", " ")[5:16]) if j.get("start") else ""
            t.add_row(j["id"], j["name"], Text(j["state"], style="bold green" if j["state"] == "R" else "yellow"),
                      j["time"], j["limit"], eta, j["reason"], key=j["id"])
        if t.row_count: t.move_cursor(row=min(cur, t.row_count - 1))

        t = self.query_one("#t_analysis", DataTable); cur = t.cursor_row; t.clear()
        for x in reversed(d["analysis"]):
            if self._keep(x["name"]): t.add_row(x["name"], fmt_age(x["age"]), key=x["name"])
        if t.row_count: t.move_cursor(row=min(cur, t.row_count - 1))
        if d.get("failed"):
            self.update_status(f"{len(d['failed'])} slurm log(s) with errors: " + ", ".join(f["name"] for f in d["failed"][:3]), error=True)

    # ---- actions
    def _active(self) -> tuple[str, DataTable]:
        tab = self.query_one("#tabs", TabbedContent).active
        return tab, self.query_one(f"#t_{tab}", DataTable)

    def _row_key(self, t: DataTable):
        if not t.row_count: return None
        return t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value

    def action_tab(self, name):
        self.query_one("#tabs", TabbedContent).active = name
        self.query_one(f"#t_{name}", DataTable).focus()

    def action_open(self):
        if not self.data: return
        tab, t = self._active(); key = self._row_key(t)
        if key is None: return
        if tab == "runs":
            r = next(r for r in self.data["rl"] if r["name"] == key); self.push_screen(RunDetail(r))
        elif tab == "sft":
            s = next(s for s in self.data["sft"] if s["name"] == key); self.push_screen(SftDetail(s))
        elif tab == "critics":
            c = next(c for c in self.data["critic"] if c["name"] == key); self.push_screen(CriticDetail(c))
        elif tab == "queue":
            j = next(j for j in self.data["queue"] if j["id"] == key)
            self.push_screen(JobDetail(j, (self.data.get("tails") or {}).get(key, [])))

    def action_mark(self):
        tab, t = self._active(); key = self._row_key(t)
        if tab != "runs" or key is None: return
        self.marks.symmetric_difference_update({key})
        self.render_all()

    def action_compare(self):
        if not self.data: return
        names = self.marks or set()
        if not names:
            tab, t = self._active(); key = self._row_key(t)
            if tab == "runs" and key:   # default: compare this run with its flat/hier twin
                twin = key.replace("_hier_", "_flat_") if "_hier_" in key else key.replace("_flat_", "_hier_")
                names = {key, twin}
        runs = [r for r in self.data["rl"] if r["name"] in names]
        if runs: self.push_screen(CompareScreen(runs))

    def action_toggle_all(self):
        self.show_all = not self.show_all; self.render_all()

    def action_filter(self):
        f = self.query_one("#filter", FilterInput); f.add_class("shown"); f.focus()

    def on_input_changed(self, ev: Input.Changed):
        self.filter = ev.value.strip(); self.render_all()

    def on_input_submitted(self, ev: Input.Submitted):
        self._close_filter()

    def on_key(self, event):
        if event.key == "escape" and self.query_one("#filter", FilterInput).has_focus:
            self._close_filter()

    def _close_filter(self):
        f = self.query_one("#filter", FilterInput); f.remove_class("shown")
        self._active()[1].focus()

    def on_data_table_row_selected(self, ev: DataTable.RowSelected):
        self.action_open()

    def on_data_table_header_selected(self, ev: DataTable.HeaderSelected):
        ev.data_table.sort(ev.column_key, key=lambda v: (str(v) if not isinstance(v, Text) else v.plain))


# ----------------------------------------------------------------------------- entry
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="use the cached snapshot, do not ssh")
    ap.add_argument("--plain", action="store_true", help="one-shot plain text summary instead of the TUI")
    a = ap.parse_args()
    if a.plain:
        repo = os.environ.get("HCFRL_REPO", str(Path.home() / "Documents/code/research/hierarchical-cfrl"))
        sys.exit(subprocess.call([sys.executable, f"{repo}/scripts/status.py"]))
    HcfrlApp(offline=a.offline).run()

if __name__ == "__main__":
    main()
