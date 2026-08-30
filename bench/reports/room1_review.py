# Usage: PYTHONPATH=. uv run python bench/reports/room1_review.py . out.html  (renders the cuVSLAM-diagnosis -> podslam review from bench/results)
"""HTML review: cuVSLAM diagnosis -> podslam. Reads bench/results/*, computes the
same metrics for every candidate (bench.sweep_summary.analyze) and renders charts as inline SVG."""
import json, sys, html
from pathlib import Path
import numpy as np
from bench.evaluate import load_tum, associate, umeyama
from bench.sweep_summary import analyze
from bench.collect_results import find_trajectory

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
gt = load_tum(str(ROOT / "bench/results/room1/gt.txt")); t0 = gt[0, 0]
CONDS = [("day", "Day"), ("night", "Night + IR"), ("transition", "Day → night")]
CANDS = [("openvins", "OpenVINS", "s1", "GPL yardstick"), ("basalt", "Basalt", "s2", "BSD"),
         ("cuvslam", "cuVSLAM 17", "s3", "closed core"), ("podslam", "podslam", "s4", "ours, Apache-2.0")]

def est_path(c, k):
    if c == "podslam":
        p = ROOT / f"bench/results/room1_sweep/podslam_klt_{k}/est.tum"
    else:
        p = find_trajectory(ROOT / f"bench/results/room1/{c}_{k}")
    return p if p and p.exists() and p.stat().st_size > 0 else None

rows = {}
for c, *_ in CANDS:
    for k, _ in CONDS:
        p = est_path(c, k)
        rows[f"{c}_{k}"] = analyze(gt, p) if p else {"status": "not run"}
sweep = json.loads((ROOT / "bench/results/room1_sweep/summary.json").read_text()) if (ROOT / "bench/results/room1_sweep/summary.json").exists() else {}

def series(c, k, win=5.0):
    p = est_path(c, k)
    if not p: return None
    est = load_tum(str(p)); gi, ei = associate(gt[:, 0], est[:, 0], 0.02)
    if len(gi) < 10: return None
    s, R, t = umeyama(est[ei, 1:4], gt[gi, 1:4]); al = (R @ est[ei, 1:4].T).T + t
    e = np.linalg.norm(al - gt[gi, 1:4], axis=1); tt = est[ei, 0] - t0
    edges = np.arange(0, 141, win)
    return [(float(a), float(np.sqrt(np.mean(e[(tt >= a) & (tt < a + win)] ** 2)) * 100) if np.any((tt >= a) & (tt < a + win)) else None) for a in edges]

# ---------- SVG helpers ----------
def bars(rowsdef, series_def, vmax, ticks, fmt, ident):
    bar_h, gap, pad, left, width = 13, 2, 14, 150, 720; plot_w = width - left - 70
    n = len(series_def); gh = n * bar_h + (n - 1) * gap; height = 26 + len(rowsdef) * (gh + pad) + 30
    sx = lambda v: left + plot_w * min(v, vmax) / vmax
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img"><title>{html.escape(ident)}</title>']
    for tk in ticks:
        x = sx(tk); out.append(f'<line class="grid" x1="{x:.1f}" y1="14" x2="{x:.1f}" y2="{height-26}"/><text class="tick" x="{x:.1f}" y="{height-10}" text-anchor="middle">{fmt(tk)}</text>')
    y = 22
    for label, vals in rowsdef:
        out.append(f'<text class="rowlab" x="{left-10}" y="{y+gh/2+4}" text-anchor="end">{html.escape(label)}</text>')
        for i, (key, name, cls) in enumerate(series_def):
            v = vals.get(key); by = y + i * (bar_h + gap)
            if v is None:
                out.append(f'<text class="missing" x="{left+6}" y="{by+bar_h-3}">not run</text>'); continue
            x1 = sx(v); r = 4
            out.append(f'<g class="bar {cls}" tabindex="0" data-tip="{html.escape(f"{name} · {label}: {fmt(v)}")}"><rect class="hit" x="{left}" y="{by-1}" width="{plot_w}" height="{bar_h+2}"/>'
                       f'<path d="M{left},{by} H{x1-r:.1f} a{r},{r} 0 0 1 {r},{r} V{by+bar_h-r} a{r},{r} 0 0 1 -{r},{r} H{left} Z"/>'
                       f'<text class="val" x="{x1+6:.1f}" y="{by+bar_h-3}">{fmt(v)}{"+" if v > vmax else ""}</text></g>')
        y += gh + pad
    out.append(f'<line class="axis" x1="{left}" y1="14" x2="{left}" y2="{height-26}"/></svg>')
    return "\n".join(out)

def lines(series_map, ident, ymax=40, ylabel="cm"):
    """error-over-time lines: series_map name -> (cls, [(t, v)])"""
    width, height, left, top, right, bottom = 720, 250, 44, 14, 16, 30
    pw, ph = width - left - right, height - top - bottom
    sx = lambda t: left + pw * t / 140.0; sy = lambda v: top + ph * (1 - min(v, ymax) / ymax)
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img"><title>{html.escape(ident)}</title>']
    for v in range(0, ymax + 1, 10):
        out.append(f'<line class="grid" x1="{left}" y1="{sy(v):.1f}" x2="{width-right}" y2="{sy(v):.1f}"/><text class="tick" x="{left-6}" y="{sy(v)+4:.1f}" text-anchor="end">{v} {ylabel}</text>')
    for t in range(0, 141, 20):
        out.append(f'<text class="tick" x="{sx(t):.1f}" y="{height-8}" text-anchor="middle">{t} s</text>')
    for name, (cls, pts) in series_map.items():
        pts = [(t, v) for t, v in pts if v is not None]
        if not pts: continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(t + 2.5):.1f},{sy(v):.1f}" for i, (t, v) in enumerate(pts))
        out.append(f'<path class="line {cls}" d="{d}"/>')
        t, v = pts[-1]; out.append(f'<circle class="dot {cls}" cx="{sx(t+2.5):.1f}" cy="{sy(v):.1f}" r="4"/>')
    out.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/></svg>')
    return "\n".join(out)

def legend(series_def):
    return '<div class="legend">' + "".join(f'<span><i class="sw {cls}"></i>{html.escape(n)}</span>' for _, n, cls in series_def) + '</div>'

sd = [(c, n, cls) for c, n, cls, _ in CANDS]
g = lambda c, k, key: rows[f"{c}_{k}"].get(key)
rows_ate = [(cl, {c: g(c, k, "ate_cm") for c, *_ in CANDS}) for k, cl in CONDS]
rows_rpe = [(cl, {c: g(c, k, "rpe_p95_cm") for c, *_ in CANDS}) for k, cl in CONDS]
rows_f20 = [(cl, {c: g(c, k, "first20_cm") for c, *_ in CANDS}) for k, cl in CONDS]
chart_ate = bars(rows_ate, sd, 30, [0, 10, 20, 30], lambda v: f"{v:.1f} cm", "ATE")
chart_rpe = bars(rows_rpe, sd, 25, [0, 5, 10, 15, 20, 25], lambda v: f"{v:.1f} cm", "RPE p95")
chart_f20 = bars(rows_f20, sd, 30, [0, 10, 20, 30], lambda v: f"{v:.1f} cm", "first 20 s")
day_lines = lines({n: (cls, series(c, "day") or []) for c, n, cls, _ in CANDS if series(c, "day")}, "day error over time")
night_lines = lines({n: (cls, series(c, "night") or []) for c, n, cls, _ in CANDS if series(c, "night")}, "night error over time")

def sweep_table():
    keep = ["base_day", "nomm_day", "clahe_day", "sat_day", "circle_day", "noimu_day", "base_night", "denoise_night", "nlmeans_night", "clahe_night", "gamma_night", "circle_night"]
    h = ['<div class="tablewrap"><table><thead><tr><th>cuVSLAM run</th><th>ATE</th><th>first 20 s</th><th>last 20 s</th><th>rot 20 s</th><th>RPE p95</th><th>obs/frame</th><th>coverage</th></tr></thead><tbody>']
    for k in keep:
        r = sweep.get(k)
        if not r or r.get("status") != "ok": continue
        f = lambda v, p=1: "—" if v is None else f"{v:.{p}f}"
        h.append(f'<tr><td><code>{k}</code></td><td>{f(r["ate_cm"])} cm</td><td>{f(r.get("first20_cm"))}</td><td>{f(r.get("last20_cm"))}</td><td>{f(r.get("rot_first20_deg"))}°</td><td>{f(r.get("rpe_p95_cm"))}</td><td>{f(r.get("obs0_mean"), 0)}</td><td>{r.get("coverage", 0)*100:.0f}%</td></tr>')
    h.append('</tbody></table></div>'); return "\n".join(h)

def main_table():
    h = ['<div class="tablewrap"><table><thead><tr><th>Candidate</th><th>Condition</th><th>ATE rmse</th><th>ATE max</th><th>first 20 s</th><th>last 20 s</th><th>RPE med</th><th>RPE p95</th><th>Coverage</th></tr></thead><tbody>']
    for c, n, cls, lic in CANDS:
        for k, cl in CONDS:
            r = rows[f"{c}_{k}"]
            if r.get("status") != "ok":
                h.append(f'<tr><td><i class="sw {cls}"></i>{n}</td><td>{cl}</td><td colspan="7" class="muted">{html.escape(r.get("status",""))}</td></tr>'); continue
            f = lambda v, p=1: "—" if v is None else f"{v:.{p}f}"
            h.append(f'<tr><td><i class="sw {cls}"></i>{n}</td><td>{cl}</td><td>{f(r["ate_cm"])} cm</td><td>{f(r["ate_max_cm"],0)} cm</td><td>{f(r.get("first20_cm"))} cm</td><td>{f(r.get("last20_cm"))} cm</td><td>{f(r["rpe_med_cm"],2)} cm</td><td>{f(r["rpe_p95_cm"])} cm</td><td>{r["coverage"]*100:.0f}%</td></tr>')
    h.append('</tbody></table></div>'); return "\n".join(h)

pod = {k: rows[f"podslam_{k}"] for k, _ in CONDS}
cu = {k: rows[f"cuvslam_{k}"] for k, _ in CONDS}
ov = {k: rows[f"openvins_{k}"] for k, _ in CONDS}
fmt = lambda v, p=1: "—" if v is None else f"{v:.{p}f}"

page = f'''<title>podslam Review</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@112,600;112,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root {{ --bg:#F3F5F8; --surface:#FFFFFF; --ink:#17212B; --muted:#5A6774; --line:#D5DCE4; --line-strong:#B9C3CE; --head:#1E3550; --evidence:#EEF2F6; --grid:#E4E8EE; --axis:#B9C3CE;
           --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#4a3aa7; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0E1318; --surface:#161D25; --ink:#E4EAF0; --muted:#97A3AF; --line:#2A343F; --line-strong:#3B4753; --head:#C9D8EA; --evidence:#1B242D; --grid:#243039; --axis:#3B4753; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#9085e9; }} }}
  :root[data-theme="dark"] {{ --bg:#0E1318; --surface:#161D25; --ink:#E4EAF0; --muted:#97A3AF; --line:#2A343F; --line-strong:#3B4753; --head:#C9D8EA; --evidence:#1B242D; --grid:#243039; --axis:#3B4753; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#9085e9; }}
  * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Source Sans 3","Segoe UI",Roboto,system-ui,sans-serif; font-size:17px; line-height:1.55; }}
  main {{ max-width:52rem; margin:0 auto; padding:1.5rem 1.1rem 4rem; display:flex; flex-direction:column; gap:2.3rem; }}
  h1,h2,h3 {{ font-family:"Archivo","Arial Narrow",Arial,sans-serif; font-stretch:112%; color:var(--head); text-wrap:balance; line-height:1.15; margin:0; }}
  h1 {{ font-size:1.9rem; font-weight:700; }} h2 {{ font-size:1.25rem; font-weight:700; }} h3 {{ font-size:1rem; font-weight:600; }}
  p {{ margin:0; max-width:64ch; }} p+p {{ margin-top:.8rem; }} .lede {{ font-size:1.12rem; }}
  .eyebrow,.tick,.rowlab,.val,.missing,.legend,th {{ font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace; }}
  .eyebrow {{ font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-bottom:.6rem; }}
  code {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.86em; background:var(--evidence); padding:.05em .35em; border-radius:3px; }}
  section {{ display:flex; flex-direction:column; gap:1rem; }} .sub {{ color:var(--muted); font-size:.95rem; }}
  .kpis {{ display:grid; grid-template-columns:repeat(2,1fr); gap:.8rem; }} @media (min-width:640px) {{ .kpis {{ grid-template-columns:repeat(4,1fr); }} }}
  .kpi {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:.85rem .95rem; display:flex; flex-direction:column; gap:.25rem; }}
  .kpi .label {{ font-size:.86rem; color:var(--muted); }} .kpi .value {{ font-size:1.9rem; font-weight:600; line-height:1.1; }} .kpi .delta {{ font-size:.86rem; color:var(--muted); }}
  .card {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:1rem 1rem .8rem; display:flex; flex-direction:column; gap:.6rem; }}
  .card h3 {{ display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; align-items:baseline; }} .card h3 small {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-weight:400; font-size:.72rem; color:var(--muted); letter-spacing:.06em; text-transform:uppercase; }}
  .chart {{ width:100%; height:auto; display:block; }} .chart .grid {{ stroke:var(--grid); stroke-width:1; }} .chart .axis {{ stroke:var(--axis); stroke-width:1; }}
  .chart .tick {{ fill:var(--muted); font-size:11px; }} .chart .rowlab {{ fill:var(--ink); font-size:12px; }} .chart .val {{ fill:var(--ink); font-size:11.5px; font-variant-numeric:tabular-nums; }} .chart .missing {{ fill:var(--muted); font-size:10.5px; font-style:italic; }}
  .chart .bar:hover path, .chart .bar:focus-visible path {{ opacity:.78; }} .chart .bar:focus-visible {{ outline:none; }} .chart .hit {{ fill:transparent; }}
  .chart .s1 path, .chart .dot.s1 {{ fill:var(--s1); }} .chart .s2 path, .chart .dot.s2 {{ fill:var(--s2); }} .chart .s3 path, .chart .dot.s3 {{ fill:var(--s3); }} .chart .s4 path, .chart .dot.s4 {{ fill:var(--s4); }}
  .chart .line {{ fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }} .chart .line.s1 {{ stroke:var(--s1); }} .chart .line.s2 {{ stroke:var(--s2); }} .chart .line.s3 {{ stroke:var(--s3); }} .chart .line.s4 {{ stroke:var(--s4); }}
  .chart .dot {{ stroke:var(--surface); stroke-width:2; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:.4rem 1.1rem; font-size:.74rem; color:var(--muted); letter-spacing:.04em; }} .legend span {{ display:inline-flex; align-items:center; gap:.4rem; }}
  .sw {{ display:inline-block; width:.8rem; height:.55rem; border-radius:2px; vertical-align:middle; margin-right:.45rem; }} .sw.s1 {{ background:var(--s1); }} .sw.s2 {{ background:var(--s2); }} .sw.s3 {{ background:var(--s3); }} .sw.s4 {{ background:var(--s4); }}
  .charts {{ display:grid; gap:.9rem; }}
  .tablewrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-size:.9rem; font-variant-numeric:tabular-nums; }}
  th {{ text-align:left; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); font-weight:500; padding:0 .6rem .5rem 0; border-bottom:1px solid var(--line-strong); white-space:nowrap; }}
  td {{ padding:.45rem .6rem .45rem 0; border-bottom:1px solid var(--line); white-space:nowrap; }} td.muted {{ color:var(--muted); font-style:italic; }}
  .evidence {{ background:var(--evidence); border-left:3px solid var(--line-strong); padding:.8rem 1rem; border-radius:0 6px 6px 0; font-size:.94rem; }}
  .split {{ display:grid; gap:.9rem; grid-template-columns:1fr; }} @media (min-width:720px) {{ .split {{ grid-template-columns:1fr 1fr; }} }}
  .split article {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:.95rem 1rem 1rem; display:flex; flex-direction:column; gap:.5rem; }}
  .split article ul {{ margin:0; padding-left:1.1rem; font-size:.94rem; }} .split li+li {{ margin-top:.25rem; }}
  pre {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.78rem; line-height:1.45; background:var(--evidence); padding:.9rem 1rem; border-radius:6px; overflow-x:auto; margin:0; }}
  footer {{ font-size:.84rem; color:var(--muted); border-top:1px solid var(--line); padding-top:1rem; }}
  #tip {{ position:fixed; pointer-events:none; background:var(--ink); color:var(--bg); font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.74rem; padding:.3rem .5rem; border-radius:4px; transform:translate(-50%,-130%); white-space:nowrap; z-index:10; }}
</style>
<main>
  <header>
    <div class="eyebrow">fisheye_slam · review · 2026-08-30</div>
    <h1>From cuVSLAM's failures to a system we own</h1>
    <p class="lede">Three real fisheye + IMU flights (TUM-VI <code>room1</code>: day, night-IR proxy, day→night), the same ground truth for everyone. First: what exactly goes wrong inside cuVSLAM, measured one knob at a time. Then: <strong>podslam</strong>, the permissively licensed replacement built today — its first results sit in the same charts.</p>
    <div class="kpis">
      <div class="kpi"><span class="label">cuVSLAM day, first 20 s</span><span class="value">{fmt(cu["day"].get("first20_cm"))} cm</span><span class="delta">three tracking losses, then a frame reset</span></div>
      <div class="kpi"><span class="label">podslam day, first 20 s</span><span class="value">{fmt(pod["day"].get("first20_cm"))} cm</span><span class="delta">same frames, world frame kept</span></div>
      <div class="kpi"><span class="label">podslam day ({pod["day"].get("coverage", 0)*100:.0f}% of the flight)</span><span class="value">{fmt(pod["day"].get("ate_cm"))} cm</span><span class="delta">OpenVINS {fmt(ov["day"].get("ate_cm"))} · cuVSLAM {fmt(cu["day"].get("ate_cm"))} (full flights)</span></div>
      <div class="kpi"><span class="label">podslam night ({pod["night"].get("coverage", 0)*100:.0f}% of the flight)</span><span class="value">{fmt(pod["night"].get("ate_cm"))} cm</span><span class="delta">OpenVINS {fmt(ov["night"].get("ate_cm"))} · cuVSLAM {fmt(cu["night"].get("ate_cm"))} (full flights)</span></div>
    </div>
  </header>

  <section>
    <h2>1 · What is wrong with cuVSLAM, systematically</h2>
    <p>Per-frame instrumentation (tracked flag, observations per camera) and per-window error after one global alignment turn "ATE 11 cm" into a mechanism:</p>
    <div class="split">
      <article><h3>Day: one event, not drift</h3><ul>
        <li>Tracking is lost at t = 11.6, 23.3 and 25.0 s — two of them during the fastest rotations of the flight (2.5–3 rad/s), one on a 60 % auto-exposure step.</li>
        <li>After the last loss the closed core <strong>re-initialises its coordinate frame</strong>: the first 25 s sit 19° rotated against the rest. That single reset is the day error.</li>
        <li>At night, same motion, no loss: the failure is image-driven (bright lamps + exposure).</li>
      </ul></article>
      <article><h3>Night: drift in the darkest 40 s</h3><ul>
        <li>Error grows from 10 cm to 19 cm in the last 20 s; <em>more</em> features than by day (426 vs 282 per frame) — the sensor noise is texture.</li>
        <li>Nothing exposed fixes it: denoising, multicam mode, SLAM mode and IMU noise scaling change nothing; the image-circle mask makes it worse.</li>
      </ul></article>
    </div>
    <div class="card"><h3>The sweep, one knob at a time <small>TUM-VI room1 · bench/sweep_cuvslam.sh</small></h3>{sweep_table()}</div>
    <div class="evidence">Two knobs cure the day failure: <code>--no-motion-model</code> (the internal constant-velocity prediction fights the fast turns; off → 8.0 cm, no losses, no reset) and photometric normalisation (<code>clahe</code> → 6.7 cm). Neither is a product answer: the detector, the outlier model and the reset policy stay closed, and the commercial license is NVIDIA's. Those became the acceptance criteria of the replacement.</div>
  </section>

  <section>
    <h2>2 · podslam — the replacement, and what it is made of</h2>
    <p>Permissive components only (OpenCV Apache-2.0, GTSAM BSD, PyTorch BSD; our code Apache-2.0). Same rig yaml, same bag contract, same outputs as every other candidate — it dropped straight into the benchmark.</p>
<pre>images ─► conditioning ─► Frontend ─► Tracker ─► Backend ─► pose (IMU frame)
              │              │           │           └ GTSAM fixed-lag smoother: pose / velocity / bias per keyframe,
              │              │           │             combined IMU factors, projection factors on normalized coords, Huber
              │              │           └ keyframes, stereo triangulation, landmark lifecycle (≥ 0.5° parallax or no landmark)
              │              └ klt (GFTT + pyramidal LK)  |  xfeat (learned, Apache-2.0)  |  yours: unit bearings + ids
              └ norm | clahe | gamma | nlmeans | enhance:&lt;torchscript&gt;   masks: circle | sat | learned:&lt;torchscript&gt;
imu ─► static init (gravity, gyro bias) ─► preintegration ─► gyro rotation prediction handed to the front-end</pre>
    <div class="split">
      <article><h3>Built-in answers to the cuVSLAM findings</h3><ul>
        <li>The world frame is never re-created; a visual loss is bridged by IMU propagation, a solver failure rebuilds the graph around the last estimate.</li>
        <li>Motion prediction is the gyro, not a velocity model — 3 rad/s turns stay inside the tracker's window.</li>
        <li>Photometric normalisation on by default; masks (image circle, saturation, learned) reach detection, tracking and stereo.</li>
      </ul></article>
      <article><h3>ML extension seams</h3><ul>
        <li><code>Frontend</code>: any detector/matcher that returns bearings with persistent ids — XFeat ships as the second implementation.</li>
        <li><code>conditioning</code>: per-frame image models (a night→day U-Net trainer is included, trained on a held-out sequence).</li>
        <li><code>masks</code>: learned reliability masks; next seams: learned depth prior for landmark init, per-observation learned noise.</li>
      </ul></article>
    </div>
    <p class="sub">Validated first on a synthetic world (KB4 stereo rig, IMU derived from the trajectory, perfect correspondences): 0.7 cm over 14 s with zero solver resets — which is how the two real bugs were found (a factor referencing a dropped landmark, and near-parallel stereo rays with no depth information).</p>
  </section>

  <section>
    <h2>3 · First results, side by side</h2>
    <p class="sub">Same metrics for all four (recomputed here from the trajectories): ATE after SE3 alignment; RPE over 1 s windows (p95 = the bad moments); the first-20-s window where cuVSLAM's reset lives. podslam is the KLT front-end with default settings, first run, untuned. <strong>All three podslam rows are partial flights</strong> (85 %, 72 % and 65 % coverage — see the table; the metrics are computed over the covered part): the dev laptop's CPU sits at 97 °C under an unrelated 24-hour Chrome process, and the thermal guard stops any run that reaches 100 °C. One earlier complete day flight scored 19.6 cm with 11 solver resets; the retry-instead-of-reset fix that followed brought the partial rerun to 15.2 cm.</p>
    <div class="charts">
      <div class="card"><h3>Absolute trajectory error <small>rmse · lower is better</small></h3>{legend(sd)}{chart_ate}</div>
      <div class="card"><h3>Worst moments <small>RPE@1 s, 95th percentile</small></h3>{legend(sd)}{chart_rpe}</div>
      <div class="card"><h3>The first 20 seconds <small>where cuVSLAM re-initialises</small></h3>{legend(sd)}{chart_f20}</div>
      <div class="card"><h3>Day: error over time <small>5 s windows after one global alignment</small></h3>{legend(sd)}{day_lines}</div>
      <div class="card"><h3>Night: error over time <small>the late drift</small></h3>{legend(sd)}{night_lines}</div>
    </div>
    {main_table()}
  </section>

  <section>
    <h2>4 · Honest state and next steps</h2>
    <p>podslam is one day old. What it already does: hold the frame through the fast turns and the exposure step that break cuVSLAM — on the first 37 s of the day flight, evaluated alone, it scores 2.9 cm where cuVSLAM scores 23 cm. What it does not do yet: keep that up for the whole flight. In all three conditions the error is flat for ~100 s and then grows in the last 40 s (day 8 → 38 cm per 10-s window, night 7 → 37, transition 8 → 51), which drags the whole-flight ATE above cuVSLAM's (19.6 / 16.5 / 22.6 vs 11.4 / 13.1 / 17.2 cm). Scale is right (path length ratio 0.99–1.00), so this is a heading/position drift; it coincides with the solver's soft resets (11 / 26 / 14 per flight — each one drops every landmark and lets the IMU carry the frame for a few keyframes) and with the end of the flight where the room offers the least texture. That reset path is the first thing to remove.</p>
    <p>Order of work, by expected gain: (1) no more graph resets — keep the healthy landmarks through a solver failure (rebuild with them) and initialise landmarks from motion parallax so the smoother never starves; (2) rim factors beyond 80° off-axis — more features under the pod's own light; (3) the learned enhancer and saturation/learned masks on the night rows; (4) the 3-camera pod on the render-offline sim recordings; (5) Orin timing. Not done yet either: loop closure, a ROS 2 node.</p>
  </section>
  <footer>Sources: <code>bench/results/room1/</code> (OpenVINS/Basalt 2026-07-06, cuVSLAM 2026-08-30), <code>bench/results/room1_sweep/</code> (cuVSLAM sweep, podslam), <code>docs/CUVSLAM_DIAGNOSIS_2026-08-30.md</code>, <code>docs/podslam.md</code>. All runs on the dev laptop (32 threads, RTX 4070).</footer>
</main>
<div id="tip" hidden></div>
<script>
(function(){{ var tip=document.getElementById('tip'); function show(el,x,y){{ tip.textContent=el.getAttribute('data-tip'); tip.hidden=false; tip.style.left=x+'px'; tip.style.top=y+'px'; }}
document.querySelectorAll('.bar').forEach(function(el){{ el.addEventListener('pointermove',function(e){{ show(el,e.clientX,e.clientY); }}); el.addEventListener('pointerleave',function(){{ tip.hidden=true; }}); el.addEventListener('focus',function(){{ var r=el.getBoundingClientRect(); show(el,r.left+r.width/2,r.top); }}); el.addEventListener('blur',function(){{ tip.hidden=true; }}); }}); }})();
</script>
'''
OUT.write_text(page); print("wrote", OUT, OUT.stat().st_size // 1024, "KB")
