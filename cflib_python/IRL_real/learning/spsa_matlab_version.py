#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np


# ============================================================
# Expert K and true Q
# ============================================================

K_STAR = np.array([
    [
        -3.80477150e-15,
        -1.11792836e+00,
         0.0,
        -6.89649404e-16,
        -6.44500310e-01,
         0.0,
    ],
    [
         1.11792836e+00,
         2.62979829e-16,
         0.0,
         6.44500310e-01,
         6.71940445e-16,
         0.0,
    ],
    [
         0.0,
         0.0,
         2.41125823e+00,
         0.0,
         0.0,
         3.10928529e+00,
    ],
], dtype=float)

Q_STAR_DIAG = np.array(
    [8.0, 8.0, 12.0, 1.2, 1.2, 10.0],
    dtype=float,
)


# ============================================================
# Utility functions
# ============================================================

def build_pairs(d):
    pairs = []
    for i in range(d):
        for j in range(i, d):
            pairs.append((i, j))
    return pairs


def quadratic_features(Z, pairs):
    N = Z.shape[0]
    Phi = np.zeros((N, len(pairs)))

    for c, (i, j) in enumerate(pairs):

        if i == j:
            Phi[:, c] = Z[:, i] ** 2

        else:
            Phi[:, c] = (
                2.0 * Z[:, i] * Z[:, j]
            )

    return Phi


def build_phi_compact(
    Ek,
    Uk,
    Ek1,
    Uk1,
):
    Z = np.hstack((Ek, Uk))
    Z1 = np.hstack((Ek1, Uk1))

    d = Z.shape[1]

    pairs = build_pairs(d)

    Phi_k = quadratic_features(
        Z,
        pairs,
    )

    Phi_k1 = quadratic_features(
        Z1,
        pairs,
    )

    Phi = Phi_k - Phi_k1

    return Phi, pairs


def symmat_to_theta(Q):

    n = Q.shape[0]

    theta = []

    for i in range(n):
        for j in range(i, n):
            theta.append(Q[i, j])

    return np.asarray(theta)


def theta_to_symmat(theta, n):

    Q = np.zeros((n, n))

    k = 0

    for i in range(n):

        for j in range(i, n):

            Q[i, j] = theta[k]
            Q[j, i] = theta[k]

            k += 1

    return Q


def proj_psd_floor(Q, floor_eig):

    Q = 0.5 * (Q + Q.T)

    eigval, eigvec = np.linalg.eigh(Q)

    eigval = np.maximum(
        eigval,
        floor_eig,
    )

    Qp = (
        eigvec
        @ np.diag(eigval)
        @ eigvec.T
    )

    return 0.5 * (Qp + Qp.T)


# ============================================================
# Bellman evaluator
# ============================================================

class BellmanEvaluator:

    def __init__(
        self,
        Phi_scaled,
        col_scale,
        pairs,
        Ek,
        Uk,
        R,
        n,
        m,
        reg_huu,
    ):

        self.Phi_scaled = Phi_scaled
        self.col_scale = col_scale

        self.pairs = pairs

        self.Ek = Ek
        self.Uk = Uk

        self.R = R

        self.n = n
        self.m = m

        self.d = n + m

        self.reg_huu = reg_huu

        # Phi is fixed throughout SPSA.
        #
        # MATLAB:
        #
        #     h_scaled = Phi \ theta
        #
        # Here we precompute pseudoinverse once
        # to make 10000 SPSA iterations practical.

        self.Phi_pinv = np.linalg.pinv(
            Phi_scaled
        )

        # Constant input-cost term.

        self.uRu = np.einsum(
            "bi,ij,bj->b",
            Uk,
            R,
            Uk,
        )

    def eval_K(self, Q):

        # ----------------------------------------------------
        # Bellman RHS:
        #
        # theta_k =
        # e_k' Q e_k + u_k' R u_k
        # ----------------------------------------------------

        eQe = np.einsum(
            "bi,ij,bj->b",
            self.Ek,
            Q,
            self.Ek,
        )

        theta = eQe + self.uRu

        # ----------------------------------------------------
        # Estimate compact H parameters
        # ----------------------------------------------------

        h_scaled = (
            self.Phi_pinv @ theta
        )

        h_unique = (
            h_scaled / self.col_scale
        )

        # ----------------------------------------------------
        # Reconstruct symmetric H
        # ----------------------------------------------------

        H = np.zeros(
            (self.d, self.d)
        )

        for value, (i, j) in zip(
            h_unique,
            self.pairs,
        ):

            H[i, j] = value
            H[j, i] = value

        H = 0.5 * (H + H.T)

        # ----------------------------------------------------
        # Extract Hux and Huu
        # ----------------------------------------------------

        Hux = H[
            self.n:,
            :self.n,
        ]

        Huu = H[
            self.n:,
            self.n:,
        ]

        Huu = 0.5 * (
            Huu + Huu.T
        )

        Huu = (
            Huu
            + self.reg_huu
            * np.eye(self.m)
        )

        # ----------------------------------------------------
        # K = Huu^{-1} Hux
        # ----------------------------------------------------

        try:

            K = np.linalg.solve(
                Huu,
                Hux,
            )

        except np.linalg.LinAlgError:

            K = np.linalg.lstsq(
                Huu,
                Hux,
                rcond=None,
            )[0]

        return K


# ============================================================
# Create Bellman evaluator
# ============================================================

def make_evaluator(
    Ek,
    Uk,
    Ek1,
    Uk1,
    R,
    reg_huu,
    label,
):

    n = Ek.shape[1]
    m = Uk.shape[1]

    Phi_raw, pairs = build_phi_compact(
        Ek,
        Uk,
        Ek1,
        Uk1,
    )

    # --------------------------------------------------------
    # Same column scaling as MATLAB version
    # --------------------------------------------------------

    col_scale = np.linalg.norm(
        Phi_raw,
        axis=0,
    )

    col_scale[
        col_scale < 1e-12
    ] = 1.0

    Phi_scaled = (
        Phi_raw / col_scale
    )

    rank_raw = np.linalg.matrix_rank(
        Phi_raw
    )

    rank_scaled = np.linalg.matrix_rank(
        Phi_scaled
    )

    cond_raw = np.linalg.cond(
        Phi_raw
    )

    cond_scaled = np.linalg.cond(
        Phi_scaled
    )

    print()
    print(
        f"==== {label} PHI CHECK ===="
    )

    print(
        f"rank(Phi) raw      = "
        f"{rank_raw}/{Phi_raw.shape[1]}"
    )

    print(
        f"cond(Phi) raw      = "
        f"{cond_raw:.6e}"
    )

    print(
        f"rank(Phi) scaled   = "
        f"{rank_scaled}/{Phi_scaled.shape[1]}"
    )

    print(
        f"cond(Phi) scaled   = "
        f"{cond_scaled:.6e}"
    )

    evaluator = BellmanEvaluator(
        Phi_scaled,
        col_scale,
        pairs,
        Ek,
        Uk,
        R,
        n,
        m,
        reg_huu,
    )

    return evaluator


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default=(
            "training_data/"
            "forward_lqr_irl_excited_"
            "20260825_193640.csv"
        ),
    )

    parser.add_argument(
        "--trim-head-s",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--trim-tail-s",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--alphaQ",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--c_spsa",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--tol_K",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--maxIter",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--pd_floor",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--reg_huu",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--q0",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--result-dir",
        default="IRL_real/results/spsa_current",
        help="Directory for SPSA outputs.",
    )

    args = parser.parse_args()

    # ========================================================
    # Load current processed flight CSV
    # ========================================================

    csv_path = Path(args.csv)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)


    print()
    print("=" * 70)
    print("SPSA MATLAB VERSION - PYTHON")
    print("=" * 70)

    print(
        "CSV:",
        csv_path,
    )

    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=float,
        encoding=None,
    )

    if data.ndim == 0:
        data = np.array(
            [data],
            dtype=data.dtype,
        )

    names = data.dtype.names

    required = [
        "time",

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

    missing = [
        x
        for x in required
        if x not in names
    ]

    if missing:

        raise RuntimeError(
            "Missing columns: "
            + str(missing)
        )

    print(
        f"Loaded rows = {len(data)}"
    )

    print(
        f"time = "
        f"{data['time'][0]:.6f}"
        f" -> "
        f"{data['time'][-1]:.6f}"
    )

    # ========================================================
    # Trim beginning/end just like MATLAB
    # ========================================================

    t_start = (
        data["time"][0]
        + args.trim_head_s
    )

    t_end = (
        data["time"][-1]
        - args.trim_tail_s
    )

    keep = (
        (data["time"] >= t_start)
        &
        (data["time"] <= t_end)
    )

    data = data[keep]

    print(
        "After trim:",
        len(data),
        "rows",
    )

    # ========================================================
    # Exact outer-loop INPUT
    # ========================================================

    E = np.column_stack([
        data["ex"],
        data["ey"],
        data["ez"],

        data["evx"],
        data["evy"],
        data["evz"],
    ])

    # ========================================================
    # Exact raw outer-loop OUTPUT
    # ========================================================

    U = np.column_stack([
        data["roll_lqr_raw"],
        data["pitch_lqr_raw"],
        data["az_lqr_raw"],
    ])

    n = E.shape[1]
    m = U.shape[1]

    print(
        f"n={n}, m={m}"
    )

    # ========================================================
    # Check raw / safe / applied
    # ========================================================

    if all(
        x in names
        for x in [
            "roll_lqr_safe",
            "pitch_lqr_safe",
            "az_lqr_safe",
        ]
    ):

        U_safe = np.column_stack([
            data["roll_lqr_safe"],
            data["pitch_lqr_safe"],
            data["az_lqr_safe"],
        ])

        print(
            "max |safe-raw| =",
            np.max(
                np.abs(
                    U_safe - U
                )
            ),
        )

    if all(
        x in names
        for x in [
            "roll_applied",
            "pitch_applied",
            "az_applied",
        ]
    ):

        U_applied = np.column_stack([
            data["roll_applied"],
            data["pitch_applied"],
            data["az_applied"],
        ])

        print(
            "max |applied-raw| =",
            np.max(
                np.abs(
                    U_applied - U
                )
            ),
        )

    # ========================================================
    # Consecutive transitions
    #
    # Exactly like MATLAB version:
    #
    # Ek  = E(1:end-1)
    # Ek1 = E(2:end)
    # ========================================================

    Ek = E[:-1]
    Uk = U[:-1]

    Ek1 = E[1:]
    Uk1 = U[1:]

    N = len(Ek)

    print()
    print(
        f"N transitions = {N}"
    )

    print(
        "rank(Ek) = "
        f"{np.linalg.matrix_rank(Ek)}/6"
    )

    # ========================================================
    # R
    # ========================================================

    R = np.diag([
        6.0,
        6.0,
        2.0,
    ])

    # ========================================================
    # Build Bellman evaluator
    # ========================================================

    evaluator = make_evaluator(
        Ek,
        Uk,
        Ek1,
        Uk1,
        R,
        args.reg_huu,
        "FULL",
    )

    # ========================================================
    # SPSA
    #
    # FULL symmetric Q
    #
    # 6x6 symmetric Q:
    #
    # p = 6*7/2 = 21 parameters
    # ========================================================

    p = (
        n * (n + 1) // 2
    )

    Q_curr = (
        args.q0 * np.eye(n)
    )

    rng = np.random.RandomState(
        args.seed
    )

    print()
    print("=" * 70)
    print("SPSA SETTINGS")
    print("=" * 70)

    print(
        "Q parameters =",
        p,
    )

    print(
        "Bellman H features =",
        (n + m)
        * (n + m + 1)
        // 2,
    )

    print(
        "alphaQ   =",
        args.alphaQ,
    )

    print(
        "c_spsa   =",
        args.c_spsa,
    )

    print(
        "tol_K    =",
        args.tol_K,
    )

    print(
        "maxIter  =",
        args.maxIter,
    )

    print(
        "pd_floor =",
        args.pd_floor,
    )

    print(
        "reg_huu  =",
        args.reg_huu,
    )

    print(
        "q0       =",
        args.q0,
    )

    print(
        "seed     =",
        args.seed,
    )

    print("=" * 70)

    converged = False

    final_iter = (
        args.maxIter
    )

    # Store:
    # iteration, ||K-K*||_F, and all symmetric-Q parameters.
    hist = np.full(
        (args.maxIter, 2 + p),
        np.nan,
        dtype=float,
    )

    for it in range(
        1,
        args.maxIter + 1,
    ):

        # ----------------------------------------------------
        # Evaluate current Q
        # ----------------------------------------------------

        K_curr = evaluator.eval_K(
            Q_curr
        )

        K_error = np.linalg.norm(
            K_curr - K_STAR,
            ord="fro",
        )

        theta_hist = symmat_to_theta(
            Q_curr
        )

        hist[it - 1, 0] = float(it)
        hist[it - 1, 1] = float(K_error)
        hist[it - 1, 2:] = theta_hist

        if (
            it <= 10
            or it % 50 == 0
        ):

            print(
                f"iter {it:5d}: "
                f"||K-K*||="
                f"{K_error:.6e}   "
                f"diag(Q)="
                f"{np.round(np.diag(Q_curr),4)}"
            )

        if K_error <= args.tol_K:

            converged = True
            final_iter = it

            print()
            print(
                "CONVERGED at iter",
                it,
            )

            print(
                "||K-K*|| =",
                K_error,
            )

            break

        # ----------------------------------------------------
        # SPSA perturbation
        # ----------------------------------------------------

        theta_curr = symmat_to_theta(
            Q_curr
        )

        Delta = np.where(
            rng.rand(p) > 0.5,
            1.0,
            -1.0,
        )

        theta_plus = (
            theta_curr
            + args.c_spsa
            * Delta
        )

        theta_minus = (
            theta_curr
            - args.c_spsa
            * Delta
        )

        Q_plus = proj_psd_floor(
            theta_to_symmat(
                theta_plus,
                n,
            ),
            args.pd_floor,
        )

        Q_minus = proj_psd_floor(
            theta_to_symmat(
                theta_minus,
                n,
            ),
            args.pd_floor,
        )

        K_plus = evaluator.eval_K(
            Q_plus
        )

        K_minus = evaluator.eval_K(
            Q_minus
        )

        E_plus = (
            np.linalg.norm(
                K_plus - K_STAR,
                ord="fro",
            ) ** 2
        )

        E_minus = (
            np.linalg.norm(
                K_minus - K_STAR,
                ord="fro",
            ) ** 2
        )

        g_theta = (
            (E_plus - E_minus)
            /
            (
                2.0
                * args.c_spsa
                * Delta
            )
        )

        theta_new = (
            theta_curr
            - args.alphaQ
            * g_theta
        )

        Q_curr = proj_psd_floor(
            theta_to_symmat(
                theta_new,
                n,
            ),
            args.pd_floor,
        )

    # ========================================================
    # Final learned Q and K
    # ========================================================

    Q_learned = Q_curr

    K_learned = evaluator.eval_K(
        Q_learned
    )

    learned_error = np.linalg.norm(
        K_learned - K_STAR,
        ord="fro",
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        "converged =",
        converged,
    )

    print(
        "final_iter =",
        final_iter,
    )

    print()
    print(
        "Q_learned ="
    )

    print(
        Q_learned
    )

    print()
    print(
        "K_learned ="
    )

    print(
        K_learned
    )

    print()
    print(
        "K_star ="
    )

    print(
        K_STAR
    )

    print()
    print(
        "||K_learned-K_star||_F =",
        learned_error,
    )

    # ========================================================
    # ORACLE CHECK
    #
    # Use TRUE Q directly.
    # ========================================================

    Q_true = np.diag(
        Q_STAR_DIAG
    )

    K_oracle = evaluator.eval_K(
        Q_true
    )

    oracle_error = np.linalg.norm(
        K_oracle - K_STAR,
        ord="fro",
    )

    print()
    print("=" * 70)
    print("ORACLE CHECK — TRUE Q")
    print("=" * 70)

    print(
        "Q_true ="
    )

    print(
        Q_true
    )

    print()
    print(
        "K_oracle ="
    )

    print(
        K_oracle
    )

    print()
    print(
        "K_star ="
    )

    print(
        K_STAR
    )

    print()
    print(
        "||K_oracle-K_star||_F =",
        oracle_error,
    )

    # ========================================================
    # XY subsystem oracle
    # ========================================================

    idx_e_xy = [
        0,
        1,
        3,
        4,
    ]

    idx_u_xy = [
        0,
        1,
    ]

    Ek_xy = Ek[
        :,
        idx_e_xy,
    ]

    Ek1_xy = Ek1[
        :,
        idx_e_xy,
    ]

    Uk_xy = Uk[
        :,
        idx_u_xy,
    ]

    Uk1_xy = Uk1[
        :,
        idx_u_xy,
    ]

    R_xy = np.diag([
        6.0,
        6.0,
    ])

    K_star_xy = K_STAR[
        np.ix_(
            idx_u_xy,
            idx_e_xy,
        )
    ]

    evaluator_xy = make_evaluator(
        Ek_xy,
        Uk_xy,
        Ek1_xy,
        Uk1_xy,
        R_xy,
        args.reg_huu,
        "XY-ONLY",
    )

    Q_xy = np.diag([
        8.0,
        8.0,
        1.2,
        1.2,
    ])

    K_oracle_xy = (
        evaluator_xy.eval_K(
            Q_xy
        )
    )

    err_xy = np.linalg.norm(
        K_oracle_xy
        - K_star_xy,
        ord="fro",
    )

    print()
    print("=" * 70)
    print("XY-ONLY ORACLE")
    print("=" * 70)

    print(
        "K_oracle_xy ="
    )

    print(
        K_oracle_xy
    )

    print()
    print(
        "K_star_xy ="
    )

    print(
        K_star_xy
    )

    print()
    print(
        "XY oracle error =",
        err_xy,
    )

    # ========================================================
    # Z subsystem oracle
    # ========================================================

    idx_e_z = [
        2,
        5,
    ]

    idx_u_z = [
        2,
    ]

    Ek_z = Ek[
        :,
        idx_e_z,
    ]

    Ek1_z = Ek1[
        :,
        idx_e_z,
    ]

    Uk_z = Uk[
        :,
        idx_u_z,
    ]

    Uk1_z = Uk1[
        :,
        idx_u_z,
    ]

    R_z = np.array([
        [2.0],
    ])

    K_star_z = K_STAR[
        np.ix_(
            idx_u_z,
            idx_e_z,
        )
    ]

    evaluator_z = make_evaluator(
        Ek_z,
        Uk_z,
        Ek1_z,
        Uk1_z,
        R_z,
        args.reg_huu,
        "Z-ONLY",
    )

    Q_z = np.diag([
        12.0,
        10.0,
    ])

    K_oracle_z = (
        evaluator_z.eval_K(
            Q_z
        )
    )

    err_z = np.linalg.norm(
        K_oracle_z
        - K_star_z,
        ord="fro",
    )

    print()
    print("=" * 70)
    print("Z-ONLY ORACLE")
    print("=" * 70)

    print(
        "K_oracle_z ="
    )

    print(
        K_oracle_z
    )

    print()
    print(
        "K_star_z ="
    )

    print(
        K_star_z
    )

    print()
    print(
        "Z oracle error =",
        err_z,
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        "Full SPSA K error :",
        learned_error,
    )

    print(
        "Full oracle error :",
        oracle_error,
    )

    print(
        "XY oracle error   :",
        err_xy,
    )

    print(
        "Z oracle error    :",
        err_z,
    )

    print("=" * 70)

    # ========================================================
    # SAVE SPSA OUTPUTS
    # ========================================================

    import csv as _csv

    # Current script uses learned_error/oracle_error.
    # SPSA history is already stored in hist.
    hist_used = hist[:final_iter]

    np.savetxt(
        result_dir / "Q_learned.csv",
        Q_learned,
        delimiter=",",
        fmt="%.12e",
    )

    np.savetxt(
        result_dir / "K_learned.csv",
        K_learned,
        delimiter=",",
        fmt="%.12e",
    )

    np.savetxt(
        result_dir / "K_star.csv",
        K_STAR,
        delimiter=",",
        fmt="%.12e",
    )

    np.savetxt(
        result_dir / "K_oracle.csv",
        K_oracle,
        delimiter=",",
        fmt="%.12e",
    )

    np.savetxt(
        result_dir / "history.csv",
        hist_used,
        delimiter=",",
        header=(
            "iter,K_error,"
            + ",".join(
                f"Qtheta_{i+1}"
                for i in range(hist_used.shape[1] - 2)
            )
        ),
        comments="",
    )

    # Recompute full Phi diagnostics from variables already
    # available in this script.
    rank_Phi_raw = int(rank_raw)
    rank_Phi_scaled = int(rank_scaled)

    cond_Phi_raw = float(cond_raw)
    cond_Phi_scaled = float(cond_scaled)

    with open(
        result_dir / "summary.csv",
        "w",
        newline="",
    ) as f:

        w = _csv.writer(f)

        w.writerow(["metric", "value"])
        w.writerow(["csv", str(csv_path)])
        w.writerow(["N_transitions", len(Ek)])
        w.writerow([
            "rank_Ek",
            int(np.linalg.matrix_rank(Ek)),
        ])
        w.writerow([
            "rank_Phi_raw",
            rank_Phi_raw,
        ])
        w.writerow([
            "rank_Phi_scaled",
            rank_Phi_scaled,
        ])
        w.writerow([
            "cond_Phi_raw",
            cond_Phi_raw,
        ])
        w.writerow([
            "cond_Phi_scaled",
            cond_Phi_scaled,
        ])
        w.writerow([
            "converged",
            int(converged),
        ])
        w.writerow([
            "final_iter",
            int(final_iter),
        ])
        w.writerow([
            "K_learned_error",
            float(learned_error),
        ])
        w.writerow([
            "K_oracle_error",
            float(oracle_error),
        ])
        w.writerow([
            "K_oracle_xy_error",
            float(err_xy),
        ])
        w.writerow([
            "K_oracle_z_error",
            float(err_z),
        ])

    print()
    print("============================================")
    print("SPSA OUTPUTS SAVED")
    print("============================================")
    print("Result directory:", result_dir)
    print("Q :", result_dir / "Q_learned.csv")
    print("K :", result_dir / "K_learned.csv")
    print("History :", result_dir / "history.csv")
    print("Summary :", result_dir / "summary.csv")
    print("============================================")


if __name__ == "__main__":
    main()
