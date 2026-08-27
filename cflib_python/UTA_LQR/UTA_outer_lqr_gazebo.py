"""
Gazebo-matched 6-state outer-loop LQR for real Crazyflie Lighthouse flight.

State error:
    e = [
        x - x_des,
        y - y_des,
        z - z_des,
        vx - vx_des,
        vy - vy_des,
        vz - vz_des
    ]^T

Input:
    u = [
        roll_cmd,
        pitch_cmd,
        az_cmd
    ]^T

Hover model:
    x_dot  = vx
    y_dot  = vy
    z_dot  = vz

    vx_dot =  g * pitch
    vy_dot = -g * roll
    vz_dot = az

This is the same outer-loop dynamics, Q, R and discrete-LQR
design used by the successful Gazebo LQR + PID controller.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_discrete_are


def _col(x, n=None):
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if n is not None and x.shape != (n, 1):
        raise ValueError(
            f"Expected shape {(n, 1)}, got {x.shape}"
        )

    return x


@dataclass
class GazeboOuterLQROutput:
    phi_des_rad: float
    theta_des_rad: float
    az_cmd_mps2: float

    error: np.ndarray
    raw_u: np.ndarray


class GazeboOuterLQR:

    def __init__(
        self,
        dt=0.01,
        gravity=9.81,
        max_angle_deg=10.0,
        max_az_mps2=15.0,
    ):

        self.dt = float(dt)
        self.g = float(gravity)

        self.max_angle_rad = np.deg2rad(
            max_angle_deg
        )

        self.max_az_mps2 = float(
            max_az_mps2
        )

        # ====================================================
        # SAME CONTINUOUS MODEL AS WORKING GAZEBO
        # ====================================================

        self.Ac = np.zeros(
            (6, 6),
            dtype=float
        )

        self.Bc = np.zeros(
            (6, 3),
            dtype=float
        )

        # xdot = vx
        # ydot = vy
        # zdot = vz
        self.Ac[0, 3] = 1.0
        self.Ac[1, 4] = 1.0
        self.Ac[2, 5] = 1.0

        # vx_dot = g * pitch
        # vy_dot = -g * roll
        # vz_dot = az
        self.Bc[3, 1] = self.g
        self.Bc[4, 0] = -self.g
        self.Bc[5, 2] = 1.0

        # ====================================================
        # EXACT ZOH DISCRETIZATION
        # ====================================================

        n = 6
        m = 3

        block = np.zeros(
            (n + m, n + m)
        )

        block[:n, :n] = self.Ac
        block[:n, n:] = self.Bc

        discrete_block = expm(
            block * self.dt
        )

        self.Ad = discrete_block[
            :n,
            :n
        ]

        self.Bd = discrete_block[
            :n,
            n:
        ]

        # ====================================================
        # EXACT Q AND R FROM WORKING GAZEBO
        # ====================================================

        self.Q = np.diag(
            [
                8.0,
                8.0,
                12.0,

                1.2,
                1.2,
                10.0,
            ]
        )

        self.R = np.diag(
            [
                6.0,
                6.0,
                2.0,
            ]
        )

        self.P = solve_discrete_are(
            self.Ad,
            self.Bd,
            self.Q,
            self.R,
        )

        self.K = np.linalg.solve(
            self.R
            + self.Bd.T
            @ self.P
            @ self.Bd,

            self.Bd.T
            @ self.P
            @ self.Ad,
        )

        print()
        print("============================================")
        print(" LIGHTHOUSE OUTER LQR - GAZEBO DYNAMICS")
        print("============================================")
        print(
            "state = [ex ey ez evx evy evz]"
        )
        print(
            "input = [roll pitch az]"
        )
        print(
            f"dt = {self.dt:.4f} s"
        )

        print()
        print("Ac =")
        print(self.Ac)

        print()
        print("Bc =")
        print(self.Bc)

        print()
        print("Q =")
        print(self.Q)

        print()
        print("R =")
        print(self.R)

        print()
        print("K =")
        print(self.K)

        print("============================================")
        print()


    def compute(
        self,
        state,
        setpoint,
        dt=None,
    ):

        error = _col(
            [
                state.position.x
                - setpoint.position.x,

                state.position.y
                - setpoint.position.y,

                state.position.z
                - setpoint.position.z,

                state.velocity.x
                - setpoint.velocity.x,

                state.velocity.y
                - setpoint.velocity.y,

                state.velocity.z
                - setpoint.velocity.z,
            ],
            6,
        )

        # LQR law
        raw_u = -self.K @ error

        phi_des = float(
            raw_u[0, 0]
        )

        theta_des = float(
            raw_u[1, 0]
        )

        az_cmd = float(
            raw_u[2, 0]
        )

        # Real-flight safety limits.
        # These DO NOT change A, B, Q, R or K.
        phi_des = float(
            np.clip(
                phi_des,
                -self.max_angle_rad,
                self.max_angle_rad,
            )
        )

        theta_des = float(
            np.clip(
                theta_des,
                -self.max_angle_rad,
                self.max_angle_rad,
            )
        )

        az_cmd = float(
            np.clip(
                az_cmd,
                -self.max_az_mps2,
                self.max_az_mps2,
            )
        )

        return GazeboOuterLQROutput(
            phi_des_rad=phi_des,
            theta_des_rad=theta_des,
            az_cmd_mps2=az_cmd,
            error=error,
            raw_u=raw_u,
        )
