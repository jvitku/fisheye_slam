"""Print the topics bench/record.sh records for a rig, one per line.

Cameras (color + depth where `depth: true`), every IMU of the rig (one per
member pod for composites) and the ground truth.

Usage: python -m bench.rig_topics rigs/pod3_oakdpro.yaml [--no-images]
  --no-images   only the IMU stream(s) + ground truth — the "flight pass" of
                bench/README.md "Physics once, render offline"
"""

import sys

from bench.rigdef import load_rig, recorded_topics


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    no_images = "--no-images" in argv
    argv = [a for a in argv if a != "--no-images"]
    if len(argv) != 1:
        sys.exit(__doc__)
    topics = recorded_topics(load_rig(argv[0]))
    if no_images:      # flight pass: IMU streams + ground truth only
        topics = [t for t in topics if "/image_raw" not in t]
    print("\n".join(topics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
