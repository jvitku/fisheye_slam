"""Hourly results page: every podslam experiment so far, best-so-far vs the baselines, as HTML.

Reads bench/results/room1 (baselines), bench/results/room1_sweep (cuVSLAM sweep + podslam runs)
and bench/results/campaign.json (the campaign log: one entry per experiment, appended by
bench/campaign_log.py). Renders inline-SVG charts, no external assets.

Usage: PYTHONPATH=. uv run python bench/reports/progress.py . out.html
"""
import html, json, sys, datetime
from pathlib import Path
import numpy as np
from bench.evaluate import load_tum
from bench.sweep_summary import analyze
from bench.collect_results import find_trajectory

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
gt = load_tum(str(ROOT / "bench/results/room1/gt.txt"))
CONDS = [("day", "Day"), ("night", "Night + IR"), ("transition", "Day → night")]
BASE = [("openvins", "OpenVINS", "s1"), ("basalt", "Basalt", "s2"), ("cuvslam", "cuVSLAM 17", "s3")]

def score(p):
    return analyze(gt, p) if p and p.exists() and p.stat().st_size > 0 else {"status": "not run"}

baseline = {f"{c}_{k}": score(find_trajectory(ROOT / f"bench/results/room1/{c}_{k}")) for c, *_ in BASE for k, _ in CONDS}
log = json.loads((ROOT / "bench/results/campaign.json").read_text()) if (ROOT / "bench/results/campaign.json").exists() else []
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
f = lambda v, p=1: "—" if v is None else f"{v:.{p}f}"

def bar_chart(rows, vmax=30):
    """rows: [(label, cls, value)] single series horizontal bars with direct labels."""
    bar_h, gap, left, width = 12, 3, 190, 720; plot_w = width - left - 70
    height = 22 + len(rows) * (bar_h + gap) + 26
    sx = lambda v: left + plot_w * min(v, vmax) / vmax
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img"><title>ATE</title>']
    for tk in range(0, vmax + 1, 10):
        out.append(f'<line class="grid" x1="{sx(tk):.1f}" y1="12" x2="{sx(tk):.1f}" y2="{height-24}"/><text class="tick" x="{sx(tk):.1f}" y="{height-8}" text-anchor="middle">{tk} cm</text>')
    y = 18
    for label, cls, v in rows:
        out.append(f'<text class="rowlab" x="{left-8}" y="{y+bar_h-2}" text-anchor="end">{html.escape(label)}</text>')
        if v is None:
            out.append(f'<text class="missing" x="{left+6}" y="{y+bar_h-2}">not run</text>')
        else:
            x1 = sx(v); r = 4
            out.append(f'<g class="bar {cls}"><path d="M{left},{y} H{x1-r:.1f} a{r},{r} 0 0 1 {r},{r} V{y+bar_h-r} a{r},{r} 0 0 1 -{r},{r} H{left} Z"/><text class="val" x="{x1+6:.1f}" y="{y+bar_h-2}">{v:.1f}{"+" if v > vmax else ""}</text></g>')
        y += bar_h + gap
    out.append(f'<line class="axis" x1="{left}" y1="12" x2="{left}" y2="{height-24}"/></svg>')
    return "\n".join(out)

def timeline_chart(entries):
    """ATE per condition over the campaign (x = experiment index)."""
    if not entries: return ""
    width, height, left, top, right, bottom = 720, 220, 44, 14, 16, 30
    pw, ph = width - left - right, height - top - bottom
    n = max(len(entries), 2); ymax = 40
    sx = lambda i: left + pw * i / (n - 1); sy = lambda v: top + ph * (1 - min(v, ymax) / ymax)
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img"><title>campaign</title>']
    for v in range(0, ymax + 1, 10):
        out.append(f'<line class="grid" x1="{left}" y1="{sy(v):.1f}" x2="{width-right}" y2="{sy(v):.1f}"/><text class="tick" x="{left-6}" y="{sy(v)+4:.1f}" text-anchor="end">{v}</text>')
    for k, cls in (("day", "s4"), ("night", "s5"), ("transition", "s6")):
        pts = [(i, e["ate"].get(k)) for i, e in enumerate(entries) if e.get("ate", {}).get(k) is not None]
        if len(pts) >= 1:
            d = " ".join(f"{'M' if j == 0 else 'L'}{sx(i):.1f},{sy(v):.1f}" for j, (i, v) in enumerate(pts))
            out.append(f'<path class="line {cls}" d="{d}"/>')
            for i, v in pts: out.append(f'<circle class="dot {cls}" cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="4"><title>{html.escape(entries[i]["change"])}: {v:.1f} cm</title></circle>')
    for i, e in enumerate(entries):
        out.append(f'<text class="tick" x="{sx(i):.1f}" y="{height-8}" text-anchor="middle">#{i+1}</text>')
    out.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/></svg>')
    return "\n".join(out)

best = {k: min((e["ate"][k] for e in log if e.get("ate", {}).get(k) is not None), default=None) for k, _ in CONDS}
latest = log[-1] if log else None
rows = []
for k, cl in CONDS:
    for c, n, cls in BASE:
        rows.append((f"{n} · {cl}", cls, baseline[f"{c}_{k}"].get("ate_cm")))
    rows.append((f"podslam best · {cl}", "s4", best[k]))
chart = bar_chart(rows)

def log_table():
    h = ['<div class="tablewrap"><table><thead><tr><th>#</th><th>when</th><th>change</th><th>day</th><th>night</th><th>transition</th><th>resets d/n/t</th><th>commit</th></tr></thead><tbody>']
    for i, e in enumerate(log):
        a = e.get("ate", {}); r = e.get("resets", {})
        h.append(f'<tr><td>{i+1}</td><td>{html.escape(e.get("when",""))}</td><td>{html.escape(e.get("change",""))}</td>'
                 f'<td>{f(a.get("day"))}</td><td>{f(a.get("night"))}</td><td>{f(a.get("transition"))}</td>'
                 f'<td>{r.get("day","—")}/{r.get("night","—")}/{r.get("transition","—")}</td><td><code>{html.escape(str(e.get("commit",""))[:7])}</code></td></tr>')
    h.append('</tbody></table></div>'); return "\n".join(h)

hilti_rows = []

hp = Path("bench/results/hilti2022/summary.json")

if hp.exists():

    for r in json.loads(hp.read_text()):

        hilti_rows.append(f"<tr><td>{html.escape(r['run'])}</td><td>{html.escape(r['seq'].split('_')[0])}</td><td>{r['ate_se3_cm']:.1f}</td><td>{r['ate_sim3_cm']:.1f}</td><td>{r['scale']:.3f}</td><td>{r['coverage']:.2f}</td><td>{(str(round(r['ms_per_frame'])) if r.get('ms_per_frame') else '—')}</td></tr>")

hilti_table = ('<div class="tablewrap"><table><thead><tr><th>run</th><th>seq</th><th>ATE SE3 cm</th><th>ATE Sim3 cm</th><th>scale</th><th>coverage</th><th>ms/frame</th></tr></thead><tbody>' + "".join(hilti_rows) + "</tbody></table></div>") if hilti_rows else "<p class=\"sub\">no runs yet</p>"

page = f'''<title>podslam Campaign</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@112,600;112,700&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root {{ --bg:#F3F5F8; --surface:#FFFFFF; --ink:#17212B; --muted:#5A6774; --line:#D5DCE4; --line-strong:#B9C3CE; --head:#1E3550; --evidence:#EEF2F6; --grid:#E4E8EE; --axis:#B9C3CE; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#4a3aa7; --s5:#eda100; --s6:#e87ba4; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0E1318; --surface:#161D25; --ink:#E4EAF0; --muted:#97A3AF; --line:#2A343F; --line-strong:#3B4753; --head:#C9D8EA; --evidence:#1B242D; --grid:#243039; --axis:#3B4753; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#9085e9; --s5:#c98500; --s6:#d55181; }} }}
  :root[data-theme="dark"] {{ --bg:#0E1318; --surface:#161D25; --ink:#E4EAF0; --muted:#97A3AF; --line:#2A343F; --line-strong:#3B4753; --head:#C9D8EA; --evidence:#1B242D; --grid:#243039; --axis:#3B4753; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#9085e9; --s5:#c98500; --s6:#d55181; }}
  * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Source Sans 3","Segoe UI",Roboto,system-ui,sans-serif; font-size:17px; line-height:1.55; }}
  main {{ max-width:52rem; margin:0 auto; padding:1.5rem 1.1rem 4rem; display:flex; flex-direction:column; gap:2rem; }}
  h1,h2,h3 {{ font-family:"Archivo","Arial Narrow",Arial,sans-serif; font-stretch:112%; color:var(--head); text-wrap:balance; line-height:1.15; margin:0; }} h1 {{ font-size:1.8rem; font-weight:700; }} h2 {{ font-size:1.2rem; font-weight:700; }}
  p {{ margin:0; max-width:64ch; }} .sub {{ color:var(--muted); font-size:.95rem; }}
  .eyebrow,.tick,.rowlab,.val,.missing,th,.legend {{ font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace; }} .eyebrow {{ font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-bottom:.5rem; }}
  code {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.86em; background:var(--evidence); padding:.05em .35em; border-radius:3px; }}
  section {{ display:flex; flex-direction:column; gap:.9rem; }}
  .kpis {{ display:grid; grid-template-columns:repeat(2,1fr); gap:.8rem; }} @media (min-width:640px) {{ .kpis {{ grid-template-columns:repeat(4,1fr); }} }}
  .kpi {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:.85rem .95rem; display:flex; flex-direction:column; gap:.25rem; }} .kpi .label {{ font-size:.86rem; color:var(--muted); }} .kpi .value {{ font-size:1.9rem; font-weight:600; line-height:1.1; }} .kpi .delta {{ font-size:.86rem; color:var(--muted); }}
  .card {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:1rem 1rem .8rem; display:flex; flex-direction:column; gap:.6rem; }}
  .chart {{ width:100%; height:auto; display:block; }} .chart .grid {{ stroke:var(--grid); stroke-width:1; }} .chart .axis {{ stroke:var(--axis); stroke-width:1; }} .chart .tick {{ fill:var(--muted); font-size:11px; }} .chart .rowlab {{ fill:var(--ink); font-size:11.5px; }} .chart .val {{ fill:var(--ink); font-size:11.5px; }} .chart .missing {{ fill:var(--muted); font-size:10.5px; font-style:italic; }}
  .chart .s1 path,.chart .dot.s1 {{ fill:var(--s1); }} .chart .s2 path,.chart .dot.s2 {{ fill:var(--s2); }} .chart .s3 path,.chart .dot.s3 {{ fill:var(--s3); }} .chart .s4 path,.chart .dot.s4 {{ fill:var(--s4); }} .chart .dot.s5 {{ fill:var(--s5); }} .chart .dot.s6 {{ fill:var(--s6); }}
  .chart .line {{ fill:none; stroke-width:2; stroke-linejoin:round; }} .chart .line.s4 {{ stroke:var(--s4); }} .chart .line.s5 {{ stroke:var(--s5); }} .chart .line.s6 {{ stroke:var(--s6); }} .chart .dot {{ stroke:var(--surface); stroke-width:2; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:.4rem 1.1rem; font-size:.74rem; color:var(--muted); }} .legend span {{ display:inline-flex; align-items:center; gap:.4rem; }} .sw {{ display:inline-block; width:.8rem; height:.55rem; border-radius:2px; }} .sw.s4 {{ background:var(--s4); }} .sw.s5 {{ background:var(--s5); }} .sw.s6 {{ background:var(--s6); }}
  .tablewrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-size:.88rem; font-variant-numeric:tabular-nums; }} th {{ text-align:left; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); font-weight:500; padding:0 .6rem .5rem 0; border-bottom:1px solid var(--line-strong); white-space:nowrap; }} td {{ padding:.4rem .6rem .4rem 0; border-bottom:1px solid var(--line); vertical-align:top; }}
  .evidence {{ background:var(--evidence); border-left:3px solid var(--line-strong); padding:.8rem 1rem; border-radius:0 6px 6px 0; font-size:.94rem; }}
  footer {{ font-size:.84rem; color:var(--muted); border-top:1px solid var(--line); padding-top:1rem; }}
</style>
<main>
  <header>
    <div class="eyebrow">fisheye_slam · podslam campaign · updated {now}</div>
    <h1>Toward state-of-the-art multi-camera VIO</h1>
    <p class="sub">One row per experiment, always the same three real flights (TUM-VI room1: day, night-IR proxy, day→night; ATE rmse after SE3 alignment, cm) until the multi-camera datasets come online. Baselines are the other candidates on identical data.</p>
    <div class="kpis">
      <div class="kpi"><span class="label">experiments logged</span><span class="value">{len(log)}</span><span class="delta">{html.escape(latest["change"]) if latest else "—"}</span></div>
      <div class="kpi"><span class="label">best day</span><span class="value">{f(best["day"])} cm</span><span class="delta">OpenVINS {f(baseline["openvins_day"].get("ate_cm"))} · cuVSLAM {f(baseline["cuvslam_day"].get("ate_cm"))}</span></div>
      <div class="kpi"><span class="label">best night</span><span class="value">{f(best["night"])} cm</span><span class="delta">OpenVINS {f(baseline["openvins_night"].get("ate_cm"))} · cuVSLAM {f(baseline["cuvslam_night"].get("ate_cm"))}</span></div>
      <div class="kpi"><span class="label">best transition</span><span class="value">{f(best["transition"])} cm</span><span class="delta">OpenVINS {f(baseline["openvins_transition"].get("ate_cm"))} · cuVSLAM {f(baseline["cuvslam_transition"].get("ate_cm"))}</span></div>
    </div>
  </header>
  <section>
    <h2>Campaign so far</h2>
    <div class="card"><div class="legend"><span><i class="sw s4"></i>day</span><span><i class="sw s5"></i>night</span><span><i class="sw s6"></i>transition</span></div>{timeline_chart(log)}</div>
    {log_table()}
  </section>
  <section>
    <h2>Best podslam vs baselines</h2>
    <div class="card">{chart}</div>
  </section>
  <section>
    <h2>Multi-camera: Hilti-Oxford 2022</h2>
    <p class="sub">Real handheld rig with five synchronized cameras (forward stereo pair + backward/left/right) and mm-accurate 6-DoF ground truth; sequence exp14 (basement, 74 s, 38 m). Same calibration for every method. "Sim3" removes a global scale to show what is scale error.</p>
    {hilti_table}
  </section>
  <section>
    <h2>Now working on</h2>
    <div class="evidence">{html.escape(latest.get("next", "")) if latest else "—"}</div>
  </section>
  <footer>Plan: the podslam campaign plan artifact · Sources: <code>bench/results/campaign.json</code>, <code>bench/results/room1</code>, <code>bench/results/room1_sweep</code>. Regenerate: <code>PYTHONPATH=. uv run python bench/reports/progress.py . out.html</code></footer>
</main>
'''
OUT.write_text(page); print("wrote", OUT)
