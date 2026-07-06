"""Resource-usage wrapper (stand-in for /usr/bin/time -v, absent in images).

Runs the given command, then prints a stats line parsed by collect_results:
    BENCH_STATS wall_s=12.3 cpu_s=45.6 cpu_percent=370 maxrss_mb=812.4

Usage: python3 timer_wrap.py <cmd> [args...]
"""

import resource
import subprocess
import sys
import time


def main():
    t0 = time.time()
    rc = subprocess.run(sys.argv[1:]).returncode
    wall = time.time() - t0
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = ru.ru_utime + ru.ru_stime
    print(f"BENCH_STATS wall_s={wall:.1f} cpu_s={cpu:.1f} "
          f"cpu_percent={100 * cpu / max(wall, 1e-9):.0f} "
          f"maxrss_mb={ru.ru_maxrss / 1024:.1f}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
