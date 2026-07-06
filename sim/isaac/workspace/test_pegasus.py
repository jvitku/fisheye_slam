"""Basic import test for Pegasus Simulator inside Isaac Sim container.

Usage:
    docker compose run --rm isaac-sim python3 /workspace/test_pegasus.py
"""

import sys


def main():
    print("Testing Pegasus Simulator import...")

    try:
        from pegasus.simulator import Pegasus
        print(f"SUCCESS: Pegasus Simulator loaded: {Pegasus}")
    except ImportError as e:
        print(f"FAILED: Could not import Pegasus Simulator: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        import pegasus.simulator
        version = getattr(pegasus.simulator, "__version__", "unknown")
        print(f"Pegasus Simulator version: {version}")
    except Exception as e:
        print(f"WARNING: Could not get version info: {e}")

    print("All import tests passed.")


if __name__ == "__main__":
    main()
