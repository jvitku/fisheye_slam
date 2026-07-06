"""Collect candidate runs into the benchmark comparison table.

Walks bench/results/<seq>/<candidate>_<condition>/, converts each run's
trajectory to TUM if needed, evaluates against <seq>/gt.txt, scrapes
/usr/bin/time -v stats from run.log, and emits a json + markdown table.

Usage: python -m bench.collect_results bench/results/room1 [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from bench.evaluate import evaluate
from bench.ovstate2tum import convert as ov_convert


def find_trajectory(run_dir: Path) -> Path | None:
    """Locate/normalize the run's TUM trajectory."""
    tum = run_dir / "traj_tum.txt"
    if tum.exists():
        return tum
    ov_state = run_dir / "state_estimate.txt"
    if ov_state.exists():
        ov_convert(str(ov_state), str(tum))
        return tum
    for name in ("trajectory.txt", "stamped_traj_estimate.txt"):
        cand = run_dir / name
        if cand.exists():
            return cand
    return None


def scrape_time_v(log: Path) -> dict:
    """Pull wall / CPU / peak-RSS out of a /usr/bin/time -v capture."""
    out = {}
    if not log.exists():
        return out
    text = log.read_text(errors="replace")
    m = re.search(r"Elapsed \(wall clock\).*?(\d+):([\d.]+)(?::([\d.]+))?", text)
    if m:
        a, b, c = m.groups()
        out["wall_s"] = (int(a) * 3600 + float(b) * 60 + float(c)) if c else (int(a) * 60 + float(b))
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    if m:
        out["peak_rss_mb"] = round(int(m.group(1)) / 1024, 1)
    m = re.search(r"Percent of CPU this job got: (\d+)%", text)
    if m:
        out["cpu_percent"] = int(m.group(1))
    return out


def collect(results_dir: Path) -> dict:
    gt = results_dir / "gt.txt"
    if not gt.exists():
        raise SystemExit(f"missing {gt}")
    rows = {}
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        traj = find_trajectory(run_dir)
        entry = scrape_time_v(run_dir / "run.log")
        if traj is None:
            entry["status"] = "no trajectory (run failed?)"
        else:
            try:
                entry.update(evaluate(str(gt), str(traj)))
                entry["status"] = "ok"
            except Exception as e:  # diverged/too-few-poses runs still get a row
                entry["status"] = f"eval failed: {e}"
        rows[run_dir.name] = entry
    return rows


def to_markdown(rows: dict) -> str:
    hdr = ("| run | ATE rmse [m] | ATE max [m] | RPE@1s rmse [m] | coverage | gaps | "
           "poses | wall [s] | cpu | peak RSS [MB] | status |")
    sep = "|" + "---|" * 11
    lines = [hdr, sep]
    for name, r in sorted(rows.items()):
        ate = r.get("ate", {})
        rpe = r.get("rpe", {})
        fmt = lambda v, p=3: (f"{v:.{p}f}" if isinstance(v, (int, float)) else "—")
        lines.append(
            f"| {name} | {fmt(ate.get('rmse'))} | {fmt(ate.get('max'))} "
            f"| {fmt(rpe.get('rmse'))} | {fmt(r.get('coverage'), 2)} "
            f"| {r.get('gaps', '—')} | {r.get('est_poses', '—')} "
            f"| {fmt(r.get('wall_s'), 0)} | {r.get('cpu_percent', '—')}% "
            f"| {r.get('peak_rss_mb', '—')} | {r.get('status', '?')} |"
        )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    rows = collect(Path(args.results_dir))
    print(to_markdown(rows))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
