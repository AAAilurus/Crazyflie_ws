#!/usr/bin/env python3

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent

# ============================================================
# CURRENT WORKING FILES — DO NOT CHANGE
# ============================================================

COLLECTOR = ROOT / "UTA_lighthouse_irl_hover21.py"

SPSA = (
    ROOT
    / "IRL_real"
    / "learning"
    / "spsa_matlab_version.py"
)

LEADER = ROOT / "UTA_lighthouse_leader_dual_exact.py"

FOLLOWER = ROOT / "UTA_lighthouse_follower_dual_182749.py"

DUAL_RUNNER = ROOT / "run_two_cf_182749_SEQUENTIAL.py"

DUAL_PLOTTER = ROOT / "plot_dual_demo_182749.py"

TRAINING_DIR = ROOT / "training_data"

FOLLOWER_DATA_DIR = (
    ROOT
    / "IRL_real"
    / "data"
    / "forward_excited_182749"
)

RESULT_ROOT = (
    ROOT
    / "IRL_real"
    / "results"
)

DUAL_DIR = ROOT / "dual_demo_182749"

PLOT_PYTHON = ROOT / "plot_env" / "bin" / "python3"

MASS_KG = 0.035


# ============================================================
# UTILITY
# ============================================================

def run_stream(cmd, logfile=None):

    print()
    print("=" * 72)
    print("RUNNING")
    print(" ".join(str(x) for x in cmd))
    print("=" * 72)
    print()

    log_handle = None

    if logfile is not None:
        logfile.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        log_handle = logfile.open("w")

    p = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:

        for line in p.stdout:

            print(line, end="")

            if log_handle is not None:
                log_handle.write(line)
                log_handle.flush()

        rc = p.wait()

    finally:

        if log_handle is not None:
            log_handle.close()

    if rc != 0:
        raise RuntimeError(
            f"Command failed with return code {rc}"
        )


# ============================================================
# FIND NEW TRAINING CSV
# ============================================================

def newest_training_csv(after_time):

    files = sorted(
        TRAINING_DIR.glob(
            "forward_lqr_irl_excited_*.csv"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Do not accidentally select an old _PROCESSED file.
    files = [
        p for p in files
        if "_PROCESSED" not in p.name
    ]

    for p in files:

        if p.stat().st_mtime >= after_time - 2.0:
            return p

    raise RuntimeError(
        "Could not find newly-created training CSV."
    )


# ============================================================
# RAW COLLECTOR CSV -> SPSA CSV
#
# Current collector saves:
#
# time
# x y z vx vy vz
# references
# phi_des theta_des delta_T
#
# SPSA expects:
#
# ex ey ez evx evy evz
# roll_lqr_raw pitch_lqr_raw az_lqr_raw
# ============================================================

def make_processed_csv(raw_path):

    processed_path = raw_path.with_name(
        raw_path.stem + "_PROCESSED.csv"
    )

    with raw_path.open("r") as f:

        reader = csv.DictReader(f)
        rows = list(reader)

    required = [
        "time",
        "x", "y", "z",
        "vx", "vy", "vz",
        "x_ref", "y_ref", "z_ref",
        "vx_ref", "vy_ref", "vz_ref",
        "phi_des",
        "theta_des",
        "delta_T",
    ]

    if not rows:
        raise RuntimeError(
            f"No data in {raw_path}"
        )

    missing = [
        c for c in required
        if c not in rows[0]
    ]

    if missing:
        raise RuntimeError(
            f"Training CSV missing: {missing}"
        )

    fieldnames = [
        "time",

        "x_lqr",
        "y_lqr",
        "z_lqr",

        "vx_lqr",
        "vy_lqr",
        "vz_lqr",

        "x_ref",
        "y_ref",
        "z_ref",

        "vx_ref",
        "vy_ref",
        "vz_ref",

        "ex",
        "ey",
        "ez",

        "evx",
        "evy",
        "evz",

        "roll_lqr_raw",
        "pitch_lqr_raw",
        "az_lqr_raw",
    ]

    with processed_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for r in rows:

            x = float(r["x"])
            y = float(r["y"])
            z = float(r["z"])

            vx = float(r["vx"])
            vy = float(r["vy"])
            vz = float(r["vz"])

            xr = float(r["x_ref"])
            yr = float(r["y_ref"])
            zr = float(r["z_ref"])

            vxr = float(r["vx_ref"])
            vyr = float(r["vy_ref"])
            vzr = float(r["vz_ref"])

            phi = float(r["phi_des"])
            theta = float(r["theta_des"])

            delta_T = float(r["delta_T"])

            az = delta_T / MASS_KG

            writer.writerow({

                "time": float(r["time"]),

                "x_lqr": x,
                "y_lqr": y,
                "z_lqr": z,

                "vx_lqr": vx,
                "vy_lqr": vy,
                "vz_lqr": vz,

                "x_ref": xr,
                "y_ref": yr,
                "z_ref": zr,

                "vx_ref": vxr,
                "vy_ref": vyr,
                "vz_ref": vzr,

                "ex": x - xr,
                "ey": y - yr,
                "ez": z - zr,

                "evx": vx - vxr,
                "evy": vy - vyr,
                "evz": vz - vzr,

                # These already contain the action excitation
                # applied by the collector.
                "roll_lqr_raw": phi,
                "pitch_lqr_raw": theta,
                "az_lqr_raw": az,
            })

    print()
    print("============================================")
    print("PROCESSED SPSA DATA CREATED")
    print("============================================")
    print("Raw       :", raw_path)
    print("Processed :", processed_path)
    print("Rows      :", len(rows))
    print("============================================")

    return processed_path


# ============================================================
# READ SPSA SUMMARY
# ============================================================

def read_summary(path):

    result = {}

    with path.open("r") as f:

        reader = csv.reader(f)

        next(reader, None)

        for row in reader:

            if len(row) >= 2:
                result[row[0]] = row[1]

    return result


# ============================================================
# INSTALL SPSA RESULT INTO CURRENT WORKING FOLLOWER
# ============================================================

def install_follower_result(
    result_dir,
    tag,
):

    FOLLOWER_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    K = np.loadtxt(
        result_dir / "K_learned.csv",
        delimiter=",",
    )

    Q = np.loadtxt(
        result_dir / "Q_learned.csv",
        delimiter=",",
    )

    if K.shape != (3, 6):

        raise RuntimeError(
            f"Bad K shape: {K.shape}"
        )

    if Q.shape != (6, 6):

        raise RuntimeError(
            f"Bad Q shape: {Q.shape}"
        )

    if not np.all(np.isfinite(K)):

        raise RuntimeError(
            "K contains NaN/Inf."
        )

    if not np.all(np.isfinite(Q)):

        raise RuntimeError(
            "Q contains NaN/Inf."
        )


    # --------------------------------------------------------
    # BACK UP OLD CONTROLLER DATA
    # --------------------------------------------------------

    backup = (
        FOLLOWER_DATA_DIR
        / "backup"
        / tag
    )

    backup.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in [
        "K_learned_real.csv",
        "Q_learned.csv",
        "Q_learned_full.csv",
    ]:

        src = FOLLOWER_DATA_DIR / name

        if src.exists():
            shutil.copy2(
                src,
                backup / name,
            )


    # --------------------------------------------------------
    # FOLLOWER EXPECTS HEADER ON K FILE
    # because its loader uses skiprows=1.
    # --------------------------------------------------------

    k_path = (
        FOLLOWER_DATA_DIR
        / "K_learned_real.csv"
    )

    with k_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "k1",
            "k2",
            "k3",
            "k4",
            "k5",
            "k6",
        ])

        for row in K:
            writer.writerow(
                [float(v) for v in row]
            )


    # --------------------------------------------------------
    # CURRENT FOLLOWER Q FORMAT:
    #
    # index,value
    #
    # It displays/stores diagonal Q metadata.
    #
    # Full learned Q is ALSO preserved separately.
    # --------------------------------------------------------

    q_meta_path = (
        FOLLOWER_DATA_DIR
        / "Q_learned.csv"
    )

    with q_meta_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "index",
            "value",
        ])

        for i, value in enumerate(
            np.diag(Q)
        ):

            writer.writerow([
                i,
                float(value),
            ])


    np.savetxt(
        FOLLOWER_DATA_DIR
        / "Q_learned_full.csv",
        Q,
        delimiter=",",
        fmt="%.12e",
    )


    print()
    print("============================================")
    print("NEW SPSA CONTROLLER INSTALLED")
    print("============================================")

    print()
    print("K_learned =")
    print(K)

    print()
    print("Full Q_learned =")
    print(Q)

    print()
    print(
        "Follower K file:",
        k_path,
    )

    print(
        "Follower Q metadata:",
        q_meta_path,
    )

    print(
        "Full Q archive:",
        FOLLOWER_DATA_DIR
        / "Q_learned_full.csv",
    )

    print("============================================")


# ============================================================
# K CONVERGENCE PLOT USING plot_env
# ============================================================

def plot_k_convergence(
    result_dir,
):

    if not PLOT_PYTHON.exists():

        raise RuntimeError(
            f"Plot Python not found: "
            f"{PLOT_PYTHON}"
        )

    history = result_dir / "history.csv"

    output_png = (
        result_dir
        / "K_convergence.png"
    )

    output_pdf = (
        result_dir
        / "K_convergence.pdf"
    )

    code = r'''
import sys
import numpy as np
import matplotlib.pyplot as plt

history = sys.argv[1]
png = sys.argv[2]
pdf = sys.argv[3]

d = np.genfromtxt(
    history,
    delimiter=",",
    names=True,
)

iteration = d["iter"]
error = d["K_error"]

fig, ax = plt.subplots(
    figsize=(8,5)
)

ax.semilogy(
    iteration,
    error,
    linewidth=2.0,
)

ax.set_xlabel(
    "SPSA Iteration",
    fontsize=13,
)

ax.set_ylabel(
    r"$\|K_i-K^*\|_F$",
    fontsize=13,
)

ax.set_title(
    "SPSA Gain Convergence",
    fontsize=14,
)

ax.grid(
    True,
    which="both",
    alpha=0.3,
)

fig.tight_layout()

fig.savefig(
    png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf,
    bbox_inches="tight",
)

plt.close(fig)
'''

    subprocess.run(
        [
            str(PLOT_PYTHON),
            "-c",
            code,
            str(history),
            str(output_png),
            str(output_pdf),
        ],
        cwd=ROOT,
        check=True,
    )

    print()
    print("K convergence:")
    print(output_png)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--leader-uri",
        default=(
            "radio://0/80/2M/"
            "E7E7E7E7E7"
        ),
    )

    parser.add_argument(
        "--follower-uri",
        default=(
            "radio://0/80/2M/"
            "E7E7E7E702"
        ),
    )

    parser.add_argument(
        "--z",
        type=float,
        default=1.0,
    )

    # Prevent an obviously bad learned K
    # from automatically reaching hardware.
    parser.add_argument(
        "--max-k-error",
        type=float,
        default=1.5,
    )

    args = parser.parse_args()


    # ========================================================
    # CHECK CURRENT FILES
    # ========================================================

    required_files = [
        COLLECTOR,
        SPSA,
        LEADER,
        FOLLOWER,
        DUAL_RUNNER,
        DUAL_PLOTTER,
    ]

    for p in required_files:

        if not p.exists():
            raise RuntimeError(
                f"Missing working file: {p}"
            )


    # ========================================================
    # 1. TRAIN LEADER
    # ========================================================

    print()
    print("#" * 72)
    print("# STEP 1 — COLLECT NEW EXPERT DATA")
    print("#" * 72)

    collection_start = time.time()

    run_stream(
        [
            sys.executable,
            COLLECTOR,

            "--uri",
            args.leader_uri,

            "--x",
            "0.0",

            "--y",
            "0.0",

            "--z",
            str(args.z),

            "--yaw",
            "0.0",

            "--takeoff-seconds",
            "4",

            "--run-seconds",
            "21",

            "--land-z",
            "0.05",

            "--land-seconds",
            "6",

            "--loop-hz",
            "200",
        ],

        ROOT
        / "pipeline_training_flight.txt",
    )


    raw_csv = newest_training_csv(
        collection_start
    )

    print()
    print("NEW TRAINING CSV:")
    print(raw_csv)


    # ========================================================
    # 2. AUTOMATICALLY CONVERT FOR SPSA
    # ========================================================

    print()
    print("#" * 72)
    print("# STEP 2 — PREPARE SPSA DATA")
    print("#" * 72)

    processed_csv = make_processed_csv(
        raw_csv
    )


    # Use the training timestamp as experiment ID.
    tag = raw_csv.stem.replace(
        "forward_lqr_irl_excited_",
        "",
    )

    result_dir = (
        RESULT_ROOT
        / f"spsa_{tag}"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # 3. SPSA
    # ========================================================

    print()
    print("#" * 72)
    print("# STEP 3 — RUN SPSA")
    print("#" * 72)

    run_stream(
        [
            sys.executable,
            SPSA,

            "--csv",
            processed_csv,

            "--alphaQ",
            "1e-2",

            "--c_spsa",
            "1e-4",

            "--tol_K",
            "1e-3",

            "--maxIter",
            "10000",

            "--pd_floor",
            "1e-6",

            "--reg_huu",
            "1e-8",

            "--q0",
            "10.0",

            "--seed",
            "42",

            "--result-dir",
            result_dir,
        ],

        ROOT
        / f"spsa_{tag}.txt",
    )


    # ========================================================
    # 4. CHECK SPSA BEFORE HARDWARE
    # ========================================================

    summary_path = (
        result_dir
        / "summary.csv"
    )

    summary = read_summary(
        summary_path
    )

    rank_phi = int(
        float(
            summary[
                "rank_Phi_raw"
            ]
        )
    )

    k_error = float(
        summary[
            "K_learned_error"
        ]
    )

    print()
    print("============================================")
    print("SPSA SAFETY CHECK")
    print("============================================")
    print(
        f"rank(Phi) = "
        f"{rank_phi}/45"
    )
    print(
        f"||K-K*||_F = "
        f"{k_error:.6f}"
    )
    print(
        f"Allowed maximum K error = "
        f"{args.max_k_error:.3f}"
    )
    print("============================================")


    if rank_phi != 45:

        raise RuntimeError(
            "STOPPING: rank(Phi) is not 45/45. "
            "Follower will NOT fly."
        )


    if not np.isfinite(k_error):

        raise RuntimeError(
            "STOPPING: invalid K error."
        )


    if k_error > args.max_k_error:

        raise RuntimeError(
            f"STOPPING: K error {k_error:.4f} "
            f"is larger than "
            f"{args.max_k_error:.4f}. "
            "Follower will NOT fly."
        )


    # ========================================================
    # 5. INSTALL Q/K
    # ========================================================

    print()
    print("#" * 72)
    print("# STEP 4 — INSTALL NEW LEARNED Q/K")
    print("#" * 72)

    install_follower_result(
        result_dir,
        tag,
    )


    # ========================================================
    # 6. K CONVERGENCE
    # ========================================================

    print()
    print("#" * 72)
    print("# STEP 5 — PLOT K CONVERGENCE")
    print("#" * 72)

    plot_k_convergence(
        result_dir
    )


    # ========================================================
    # COMPILE CURRENT WORKING FLIGHT CODE
    # ========================================================

    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",

            str(LEADER),
            str(FOLLOWER),
            str(DUAL_RUNNER),
        ],
        cwd=ROOT,
        check=True,
    )


    # ========================================================
    # PHYSICAL SAFETY CONFIRMATION
    # ========================================================

    print()
    print("=" * 72)
    print("READY FOR VALIDATION FLIGHT")
    print("=" * 72)

    print(
        "Leader   = EXPERT K*"
    )

    print(
        "Follower = NEW SPSA K"
    )

    print(
        "No perturbations."
    )

    print(
        "No eta excitation."
    )

    print()
    print(
        "Leader will takeoff -> hover -> land FIRST."
    )

    print(
        "Follower starts only after leader finishes."
    )

    print()
    print(
        "Press ENTER to start the dual flight."
    )

    print(
        "Press Ctrl+C now to stop."
    )

    input()


    # ========================================================
    # 7. DUAL VALIDATION
    # ========================================================

    print()
    print("#" * 72)
    print("# STEP 6 — LEADER / FOLLOWER VALIDATION")
    print("#" * 72)

    DUAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dual_log = (
        DUAL_DIR
        / "sequential_full.txt"
    )

    run_stream(
        [
            sys.executable,
            DUAL_RUNNER,

            "--leader-uri",
            args.leader_uri,

            "--follower-uri",
            args.follower_uri,

            "--z",
            str(args.z),

            "--takeoff-seconds",
            "4",

            "--run-seconds",
            "5",

            "--land-z",
            "0.05",

            "--land-seconds",
            "6",

            "--loop-hz",
            "200",

            "--gap-seconds",
            "3",
        ],

        dual_log,
    )


    # ========================================================
    # 8. DUAL PLOTS
    # ========================================================

    print()
    print("#" * 72)
    print("# STEP 7 — PLOT DUAL FLIGHT")
    print("#" * 72)

    subprocess.run(
        [
            str(PLOT_PYTHON),
            str(DUAL_PLOTTER),
        ],
        cwd=ROOT,
        check=True,
    )


    # ========================================================
    # 9. ARCHIVE THIS COMPLETE EXPERIMENT
    # ========================================================

    archive = (
        DUAL_DIR
        / "runs"
        / tag
    )

    archive.mkdir(
        parents=True,
        exist_ok=True,
    )

    files_to_archive = [

        DUAL_DIR
        / "leader_expert.csv",

        DUAL_DIR
        / "follower_learned_182749.csv",

        DUAL_DIR
        / "Z_Leader_vs_Follower.png",

        DUAL_DIR
        / "Z_Leader_vs_Follower.pdf",

        DUAL_DIR
        / "XY_Leader_vs_Follower.png",

        DUAL_DIR
        / "XY_Leader_vs_Follower.pdf",

        DUAL_DIR
        / "sequential_full.txt",

        result_dir
        / "K_convergence.png",

        result_dir
        / "K_convergence.pdf",

        result_dir
        / "K_learned.csv",

        result_dir
        / "Q_learned.csv",

        result_dir
        / "history.csv",

        result_dir
        / "summary.csv",

        raw_csv,

        processed_csv,
    ]

    for src in files_to_archive:

        if src.exists():

            shutil.copy2(
                src,
                archive / src.name,
            )


    print()
    print("=" * 72)
    print("COMPLETE IRL EXPERIMENT FINISHED")
    print("=" * 72)

    print()
    print("Training data:")
    print(raw_csv)

    print()
    print("SPSA result:")
    print(result_dir)

    print()
    print(
        "Final K error:",
        k_error,
    )

    print()
    print("Experiment archive:")
    print(archive)

    print()
    print("Final figures:")
    print(
        archive
        / "K_convergence.png"
    )

    print(
        archive
        / "Z_Leader_vs_Follower.png"
    )

    print(
        archive
        / "XY_Leader_vs_Follower.png"
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
