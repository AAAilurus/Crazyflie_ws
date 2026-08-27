#!/usr/bin/env python3

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAINING_DIR = ROOT / "training_data"
OUT_DIR = ROOT / "lqr_validation_results"
OUT_DIR.mkdir(exist_ok=True)

LEADER_SCRIPT = ROOT / "lqr_expert_flight.py"
FOLLOWER_SCRIPT = ROOT / "lqr_learned_flight.py"


def find_new_csv(before, start_time):
    """Find CSV created by the flight that just finished."""
    after = set(TRAINING_DIR.glob("forward_lqr_*.csv"))

    new_files = list(after - before)

    if new_files:
        return max(new_files, key=lambda p: p.stat().st_mtime)

    # Fallback: newest CSV modified since flight began
    candidates = [
        p for p in after
        if p.stat().st_mtime >= start_time - 2.0
    ]

    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return None


def run_and_tee(cmd, logfile):
    print()
    print("=" * 60)
    print("RUNNING:")
    print(" ".join(str(x) for x in cmd))
    print("=" * 60)
    print()

    with logfile.open("w") as f:
        p = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            for line in p.stdout:
                print(line, end="")
                f.write(line)
                f.flush()

            rc = p.wait()

        except KeyboardInterrupt:
            print("\nCTRL+C received -> stopping active Crazyflie")
            try:
                p.send_signal(signal.SIGINT)
                p.wait(timeout=5)
            except Exception:
                p.terminate()

            raise

    return rc


def build_command(
    script,
    uri,
    z,
    takeoff_seconds,
    run_seconds,
    land_z,
    land_seconds,
    loop_hz,
):
    return [
        sys.executable,
        str(script),

        "--uri", uri,

        "--x", "0.0",
        "--y", "0.0",
        "--z", str(z),
        "--yaw", "0.0",

        "--takeoff-seconds", str(takeoff_seconds),
        "--run-seconds", str(run_seconds),

        "--land-z", str(land_z),
        "--land-seconds", str(land_seconds),

        "--loop-hz", str(loop_hz),
    ]


def main():

    parser = argparse.ArgumentParser(
        description="Sequential expert-vs-learned Crazyflie demo"
    )

    parser.add_argument(
        "--leader-uri",
        default="radio://0/80/2M/E7E7E7E7E7",
    )

    parser.add_argument(
        "--follower-uri",
        default="radio://0/80/2M/E7E7E7E702",
    )

    parser.add_argument("--z", type=float, default=1.0)

    parser.add_argument(
        "--takeoff-seconds",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--run-seconds",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--land-z",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--land-seconds",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--loop-hz",
        type=float,
        default=200.0,
    )

    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=3.0,
    )

    args = parser.parse_args()

    if not LEADER_SCRIPT.exists():
        raise RuntimeError(
            f"Missing leader script: {LEADER_SCRIPT}"
        )

    if not FOLLOWER_SCRIPT.exists():
        raise RuntimeError(
            f"Missing follower script: {FOLLOWER_SCRIPT}"
        )


    print()
    print("=" * 60)
    print("SEQUENTIAL EXPERT vs LEARNED DEMO")
    print("=" * 60)

    print()
    print("LEADER")
    print("  controller : EXPERT")
    print(f"  URI        : {args.leader_uri}")

    print()
    print("FOLLOWER")
    print("  controller : 182749 LEARNED")
    print(f"  URI        : {args.follower_uri}")

    print()
    print(f"Target z     : {args.z:.2f} m")
    print(f"Hover time   : {args.run_seconds:.1f} s")

    print()
    print("NO reference perturbations")
    print("NO eta excitation")

    print()
    print("Sequence:")
    print("LEADER takeoff -> hover -> land")
    print("then")
    print("FOLLOWER takeoff -> hover -> land")

    print("=" * 60)


    # ========================================================
    # LEADER / EXPERT
    # ========================================================

    print()
    print("\n" + "#" * 60)
    print("# 1/2  LEADER / EXPERT")
    print("#" * 60)

    before = set(TRAINING_DIR.glob("forward_lqr_*.csv"))
    start_time = time.time()

    leader_cmd = build_command(
        LEADER_SCRIPT,
        args.leader_uri,
        args.z,
        args.takeoff_seconds,
        args.run_seconds,
        args.land_z,
        args.land_seconds,
        args.loop_hz,
    )

    rc = run_and_tee(
        leader_cmd,
        OUT_DIR / "leader_expert.txt",
    )

    if rc != 0:
        print()
        print("LEADER DID NOT FINISH NORMALLY.")
        print("FOLLOWER WILL NOT START.")
        sys.exit(rc)

    leader_csv = find_new_csv(before, start_time)

    if leader_csv is not None:
        destination = OUT_DIR / "leader_expert.csv"
        shutil.copy2(leader_csv, destination)

        print()
        print("Leader CSV:")
        print(destination)

    else:
        print()
        print("WARNING: Could not find leader CSV.")


    # ========================================================
    # GAP
    # ========================================================

    print()
    print("=" * 60)
    print("LEADER FINISHED AND LANDED")
    print("=" * 60)

    print(
        f"Waiting {args.gap_seconds:.1f} seconds "
        "before follower..."
    )

    time.sleep(args.gap_seconds)


    # ========================================================
    # FOLLOWER / LEARNED
    # ========================================================

    print()
    print("\n" + "#" * 60)
    print("# 2/2  FOLLOWER / 182749 LEARNED")
    print("#" * 60)

    before = set(TRAINING_DIR.glob("forward_lqr_*.csv"))
    start_time = time.time()

    follower_cmd = build_command(
        FOLLOWER_SCRIPT,
        args.follower_uri,
        args.z,
        args.takeoff_seconds,
        args.run_seconds,
        args.land_z,
        args.land_seconds,
        args.loop_hz,
    )

    rc = run_and_tee(
        follower_cmd,
        OUT_DIR / "follower_learned_182749.txt",
    )

    if rc != 0:
        print()
        print("FOLLOWER DID NOT FINISH NORMALLY.")
        sys.exit(rc)

    follower_csv = find_new_csv(before, start_time)

    if follower_csv is not None:
        destination = OUT_DIR / "follower_learned_182749.csv"
        shutil.copy2(follower_csv, destination)

        print()
        print("Follower CSV:")
        print(destination)

    else:
        print()
        print("WARNING: Could not find follower CSV.")


    print()
    print("=" * 60)
    print("SEQUENTIAL DEMO FINISHED")
    print("=" * 60)

    print()
    print("Leader:")
    print("  expert Q")
    print("  expert K")

    print()
    print("Follower:")
    print("  learned Q from 182749")
    print("  learned K from 182749")

    print()
    print("NO eta")
    print("NO perturbations")

    print()
    print("Data:")
    print(OUT_DIR)

    print("=" * 60)


if __name__ == "__main__":
    main()
