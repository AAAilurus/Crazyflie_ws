#!/usr/bin/env python3

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = Path("dual_demo_182749")

LEADER_FILE = DATA_DIR / "leader_expert.csv"
FOLLOWER_FILE = DATA_DIR / "follower_learned_182749.csv"


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path):

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        rows = list(reader)
        fields = reader.fieldnames

    print()
    print("FILE:", path)
    print("Columns:")
    print(fields)
    print("Rows:", len(rows))

    if not rows:
        raise RuntimeError(f"No data in {path}")

    return rows, fields


leader_rows, leader_fields = load_csv(LEADER_FILE)
follower_rows, follower_fields = load_csv(FOLLOWER_FILE)


# ============================================================
# COLUMN HELPER
# ============================================================

def get_column(rows, fields, candidates):

    for name in candidates:
        if name in fields:
            return np.array(
                [float(r[name]) for r in rows],
                dtype=float,
            )

    return None


def extract_flight(rows, fields):

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    t = get_column(
        rows,
        fields,
        [
            "time",
            "t",
            "time_s",
            "timestamp",
            "elapsed_time",
        ],
    )

    if t is None:
        # Outer LQR data is approximately 100 Hz
        t = np.arange(len(rows), dtype=float) * 0.01

    t = t - t[0]


    # --------------------------------------------------------
    # REFERENCE
    # --------------------------------------------------------

    xr = get_column(
        rows, fields,
        ["x_ref", "px_ref", "ref_x"]
    )

    yr = get_column(
        rows, fields,
        ["y_ref", "py_ref", "ref_y"]
    )

    zr = get_column(
        rows, fields,
        ["z_ref", "pz_ref", "ref_z"]
    )


    # --------------------------------------------------------
    # ACTUAL POSITION
    # --------------------------------------------------------

    x = get_column(
        rows, fields,
        ["x", "px", "position_x"]
    )

    y = get_column(
        rows, fields,
        ["y", "py", "position_y"]
    )

    z = get_column(
        rows, fields,
        ["z", "pz", "position_z"]
    )


    # --------------------------------------------------------
    # If CSV stores error e = position - reference
    # reconstruct position
    # --------------------------------------------------------

    ex = get_column(
        rows, fields,
        ["ex", "e_x", "error_x"]
    )

    ey = get_column(
        rows, fields,
        ["ey", "e_y", "error_y"]
    )

    ez = get_column(
        rows, fields,
        ["ez", "e_z", "error_z"]
    )


    if xr is None:
        xr = np.zeros(len(rows))

    if yr is None:
        yr = np.zeros(len(rows))


    if x is None and ex is not None:
        x = ex + xr

    if y is None and ey is not None:
        y = ey + yr

    if z is None and ez is not None and zr is not None:
        z = ez + zr


    if x is None:
        raise RuntimeError(
            "Could not find/reconstruct X position"
        )

    if y is None:
        raise RuntimeError(
            "Could not find/reconstruct Y position"
        )

    if z is None:
        raise RuntimeError(
            "Could not find/reconstruct Z position"
        )


    # If there is no z_ref column, use 1 m
    if zr is None:
        zr = np.ones(len(rows))


    return t, x, y, z, xr, yr, zr


tL, xL, yL, zL, xrL, yrL, zrL = extract_flight(
    leader_rows,
    leader_fields,
)

tF, xF, yF, zF, xrF, yrF, zrF = extract_flight(
    follower_rows,
    follower_fields,
)


print()
print("============================================")
print("DATA SUMMARY")
print("============================================")
print(
    f"Leader   : {len(tL)} samples, "
    f"{tL[-1]:.2f} s"
)
print(
    f"Follower : {len(tF)} samples, "
    f"{tF[-1]:.2f} s"
)
print()
print(
    f"Leader Z   range: "
    f"{zL.min():.3f} -> {zL.max():.3f} m"
)
print(
    f"Follower Z range: "
    f"{zF.min():.3f} -> {zF.max():.3f} m"
)
print("============================================")


# ============================================================
# Z FIGURE
#
# Leader   = BLUE SOLID
# Follower = ORANGE DOTTED
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(
    tL,
    zL,
    color="tab:blue",
    linestyle="-",
    linewidth=2.2,
    label="Leader (Expert)",
)

ax.plot(
    tF,
    zF,
    color="tab:orange",
    linestyle=":",
    linewidth=2.8,
    label="Follower (Learned)",
)

# Reference
ax.plot(
    tL,
    zrL,
    color="black",
    linestyle="--",
    linewidth=1.4,
    label="Reference",
)

ax.set_xlabel("Time (s)", fontsize=13)
ax.set_ylabel("Z Position (m)", fontsize=13)

ax.set_title(
    "Vertical Position Tracking",
    fontsize=14,
)

ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

fig.tight_layout()

fig.savefig(
    DATA_DIR / "Z_Leader_vs_Follower.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    DATA_DIR / "Z_Leader_vs_Follower.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# XY FIGURE
#
# X and Y in same image
#
# Leader   = BLUE SOLID
# Follower = ORANGE DOTTED
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(9, 7),
    sharex=True,
)


# -------------------- X --------------------

ax1.plot(
    tL,
    xL,
    color="tab:blue",
    linestyle="-",
    linewidth=2.2,
    label="Leader (Expert)",
)

ax1.plot(
    tF,
    xF,
    color="tab:orange",
    linestyle=":",
    linewidth=2.8,
    label="Follower (Learned)",
)

ax1.axhline(
    0.0,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Reference",
)

ax1.set_ylabel("X Position (m)", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)


# -------------------- Y --------------------

ax2.plot(
    tL,
    yL,
    color="tab:blue",
    linestyle="-",
    linewidth=2.2,
    label="Leader (Expert)",
)

ax2.plot(
    tF,
    yF,
    color="tab:orange",
    linestyle=":",
    linewidth=2.8,
    label="Follower (Learned)",
)

ax2.axhline(
    0.0,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Reference",
)

ax2.set_xlabel("Time (s)", fontsize=13)
ax2.set_ylabel("Y Position (m)", fontsize=12)

ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)


fig.suptitle(
    "Horizontal Position Tracking",
    fontsize=14,
)

fig.tight_layout()

fig.savefig(
    DATA_DIR / "XY_Leader_vs_Follower.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    DATA_DIR / "XY_Leader_vs_Follower.pdf",
    bbox_inches="tight",
)

plt.close(fig)


print()
print("============================================")
print("PLOTS CREATED")
print("============================================")
print(
    DATA_DIR / "Z_Leader_vs_Follower.png"
)
print(
    DATA_DIR / "XY_Leader_vs_Follower.png"
)
print("============================================")
