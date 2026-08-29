"""Print the topics bench/record.sh records for a rig, one per line.

Cameras (color + depth where `depth: true`), every IMU of the rig (one per
member pod for composites) and the ground truth.

Usage: python -m bench.rig_topics rigs/pod3_oakdpro.yaml
"""

import sys

from bench.rigdef import load_rig, recorded_topics


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit(__doc__)
    print("\n".join(recorded_topics(load_rig(argv[0]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
