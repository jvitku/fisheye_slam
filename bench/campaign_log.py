"""Append one experiment to bench/results/campaign.json from the current sweep rows.

Usage: uv run python -m bench.campaign_log "what changed" "what's next" [--runs podslam_klt]
Scores bench/results/room1_sweep/<runs>_{day,night,transition} and records ATE + resets + commit.
"""
import argparse, datetime, json, re, subprocess, sys
from pathlib import Path
from bench.evaluate import load_tum
from bench.sweep_summary import analyze

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("change"); ap.add_argument("next")
    ap.add_argument("--runs", default="podslam_klt")
    ap.add_argument("--dir", default="bench/results/room1_sweep")
    a = ap.parse_args(argv)
    gt = load_tum("bench/results/room1/gt.txt")
    ate, resets = {}, {}
    for cond in ("day", "night", "transition"):
        d = Path(a.dir) / f"{a.runs}_{cond}"
        if (d / "est.tum").exists() and (d / "est.tum").stat().st_size > 0:
            r = analyze(gt, d / "est.tum"); ate[cond] = round(r.get("ate_cm", 0), 2) if r.get("status") == "ok" else None
            m = re.search(r"(\d+) soft resets", (d / "run.log").read_text(errors="replace")) if (d / "run.log").exists() else None
            resets[cond] = int(m.group(1)) if m else None
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    p = Path("bench/results/campaign.json")
    log = json.loads(p.read_text()) if p.exists() else []
    log.append({"when": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "change": a.change, "next": a.next,
                "ate": ate, "resets": resets, "commit": commit})
    p.write_text(json.dumps(log, indent=1))
    print(f"logged #{len(log)}: {ate} resets {resets}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
