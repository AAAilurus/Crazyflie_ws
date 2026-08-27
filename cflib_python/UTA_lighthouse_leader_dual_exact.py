#!/usr/bin/env python3
"""
Run the Python cascaded PID on the host and send raw PWM to all four motors.

This bypasses the firmware flight controller by enabling `motorPowerSet.enable`
and publishing motor commands on the raw motor CRTP port.
"""

HOVER_HEIGHT = 0.15
# User-editable trajectory block.
# Edit this list directly to define the desired path.
# Each tuple is: (time_s, x_m, y_m, z_m, yaw_deg)
# time_s is relative to the start of the main control phase.
USE_SCRIPT_TRAJECTORY = False
USER_DEFINED_TRAJECTORY = [
    # TRUE original ERAU PID validation
    # time(s), x(m), y(m), z(m), yaw(deg)
    (0.0,  0.0, 0.0, 0.00, 0.0),
    (4.0,  0.0, 0.0, 0.15, 0.0),
    (6.0,  0.0, 0.0, 0.15, 0.0),
    (9.0,  0.0, 0.0, 0.05, 0.0),
]
# after the last waypoint, there will be a default landing to z=0.05m in three seconds if --no-land is not specified, regardless of the last waypoint's z value

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from pynput import keyboard


# Ensure local imports work no matter where the script is launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WAYPOINT_DATA_PATH = os.path.join(SCRIPT_DIR, "waypoint_data.json")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.utils import uri_helper
except ModuleNotFoundError as exc:
    if exc.name == "cflib":
        raise SystemExit(
            "Missing dependency: cflib. Install it in this Python environment "
            f"(for example: `pip install cflib`) and run again.\n"
            f"Current interpreter: {sys.executable}"
        ) from exc
    raise

from UTA_LQR.UTA_outer_lqr_gazebo import GazeboOuterLQR
from controller_lqr_real_200hz_FINAL_STABLE_1M.controller_pid import ControllerPID
from controller_lqr_real_200hz_FINAL_STABLE_1M.controller_types import (
    AccData,
    Attitude,
    AttitudeRate,
    Axis3f,
    Control,
    GyroData,
    Position,
    Quaternion,
    SensorData,
    Setpoint,
    SetpointMode,
    StabMode,
    State,
    Velocity,
    quat2rpy,
)
from motorRaw_dual_100hz import MotorRaw

logging.basicConfig(level=logging.ERROR)

# Taken from Crazyflie platform defaults / current local motor_control.py.
THRUST_MIN = 0.02136263065537499
THRUST_MAX = 0.2
VMOTOR2THRUST0 = -0.014058926705279723
VMOTOR2THRUST1 = 0.04265273261724981
VMOTOR2THRUST2 = 0.0018327760144017432
VMOTOR2THRUST3 = 0.0020576974784587178

IDLE_THRUST = 7000
UINT16_MAX = 65535
REDUCE_MULTIPLIER = 0.65
EKF_WARMUP_S = 1.5

# ============================================================
# FIXED LIGHTHOUSE ORIGIN
# ============================================================
# Physical point chosen as controller (0,0,0).
#
# Raw Lighthouse/EKF coordinates at that point:
#   x = +0.07 m
#   y = -0.12 m
#   z = -0.49 m
#
LIGHTHOUSE_ORIGIN_X = 0.07
LIGHTHOUSE_ORIGIN_Y = -0.12
LIGHTHOUSE_ORIGIN_Z = -0.49

@dataclass(frozen=True)
class TrajectoryWaypoint:
    time_s: float
    x: float
    y: float
    z: float
    yaw_deg: float = 0.0


SCRIPT_TRAJECTORY = [
    TrajectoryWaypoint(time_s=t, x=x, y=y, z=z, yaw_deg=yaw)
    for (t, x, y, z, yaw) in USER_DEFINED_TRAJECTORY
]


@dataclass
class MotorThrust:
    motor_1: float = 0.0
    motor_2: float = 0.0
    motor_3: float = 0.0
    motor_4: float = 0.0


def _validate_trajectory() -> None:
    if not SCRIPT_TRAJECTORY:
        raise ValueError("SCRIPT_TRAJECTORY is empty")

    last_t = -1.0
    for waypoint in SCRIPT_TRAJECTORY:
        if waypoint.time_s < 0.0:
            raise ValueError("Trajectory times must be >= 0")
        if waypoint.time_s < last_t:
            raise ValueError("Trajectory times must be non-decreasing")
        last_t = waypoint.time_s


def _sample_trajectory(t_s: float) -> Tuple[float, float, float, float]:
    if len(SCRIPT_TRAJECTORY) == 1 or t_s <= SCRIPT_TRAJECTORY[0].time_s:
        p = SCRIPT_TRAJECTORY[0]
        return p.x, p.y, p.z, p.yaw_deg

    if t_s >= SCRIPT_TRAJECTORY[-1].time_s:
        p = SCRIPT_TRAJECTORY[-1]
        return p.x, p.y, p.z, p.yaw_deg

    for idx in range(len(SCRIPT_TRAJECTORY) - 1):
        p0 = SCRIPT_TRAJECTORY[idx]
        p1 = SCRIPT_TRAJECTORY[idx + 1]
        if p0.time_s <= t_s <= p1.time_s:
            dt = p1.time_s - p0.time_s
            alpha = 0.0 if dt <= 0.0 else (t_s - p0.time_s) / dt
            x = p0.x + (p1.x - p0.x) * alpha
            y = p0.y + (p1.y - p0.y) * alpha
            z = p0.z + (p1.z - p0.z) * alpha
            yaw = p0.yaw_deg + (p1.yaw_deg - p0.yaw_deg) * alpha
            return x, y, z, yaw

    p = SCRIPT_TRAJECTORY[-1]
    return p.x, p.y, p.z, p.yaw_deg


json_data = {
    "time_s": [],
    "loop_hz": 0.0,
    "sample_period_s": 0.0,
    "position_x": [],
    "setpoint_x": [],
    "position_y": [],
    "setpoint_y": [],
    "position_z": [],
    "setpoint_z": [],
    #"velocity_x": [],
    #"velocity_y": [],
    #"velocity_z": [],
    #"roll": [],
    #"pitch": [],
    #"yaw": [],
    #"gyro_x": [],
    #"gyro_y": [],
    #"gyro_z": [],
    "vbat": [],
}


class HostPIDPWMPositionController:
    def __init__(
        self,
        uri: str,
        target_x: float,
        target_y: float,
        target_z: float,
        target_yaw: float,
        loop_hz: float,
        run_seconds: float,
        takeoff_seconds: float,
        land_z: float,
        land_seconds: float,
        do_land: bool,
    ):
        self.uri = uri
        self.loop_hz = loop_hz
        self.loop_period = 1.0 / loop_hz
        self.run_seconds = run_seconds
        self.takeoff_seconds = takeoff_seconds
        self.land_z = land_z
        self.land_seconds = land_seconds
        self.do_land = do_land
        self._log_time_origin = None

        # ====================================================
        # FORWARD LQR TRAINING DATA
        # ====================================================
        #
        # One row is recorded per OUTER-LQR update.
        #
        # Columns:
        #   time
        #   x y z
        #   vx vy vz
        #   x_ref y_ref z_ref
        #   vx_ref vy_ref vz_ref
        #   phi_des theta_des delta_T
        #
        # phi/theta are in radians.
        # delta_T is in Newtons.
        #
        self.forward_lqr_rows = []

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        self.forward_lqr_save_dir = os.path.join(
            SCRIPT_DIR,
            "training_data",
        )

        self.forward_lqr_csv_path = os.path.join(
            self.forward_lqr_save_dir,
            f"forward_lqr_{timestamp}.csv",
        )

        self.forward_lqr_mass_kg = 0.035

        json_data["loop_hz"] = self.loop_hz
        json_data["sample_period_s"] = self.loop_period

        # ====================================================
        # UTA OUTER LQR + ORIGINAL ERAU INNER PID
        # ====================================================

        self.controller = ControllerPID()
        self.controller.init()

        # ====================================================
        # GAZEBO-MATCHED OUTER LQR
        # ====================================================
        #
        # Same:
        #   A, B, Q, R, dt and LQR design
        #
        # as the successful Gazebo controller.
        #
        # Real-flight angle saturation is deliberately smaller
        # for the first Lighthouse XY test.
        #
        self.outer_lqr = GazeboOuterLQR(
            dt=0.01,
            gravity=9.81,
            max_angle_deg=1.0,
            max_az_mps2=15.0,
        )

        # Outer LQR updates at 100 Hz.
        # Host/inner controller runs at 500 Hz.
        self.outer_update_divider = 2
        self.outer_counter = 0

        # Held outer-loop commands.
        self.lqr_roll_des_deg = 0.0
        self.lqr_pitch_des_deg = 0.0
        self.lqr_az_cmd = 0.0

        # Real Crazyflie actuator calibration.
        #
        # Outer LQR still outputs az [m/s^2].
        # This maps az to the real Crazyflie thrust-command scale.
        # Real Crazyflie hover feedforward.
        # Use the previously validated real-flight value.
        # Fallback/reference value only.
        self.lqr_thrust_base = 35950.0

        # LQR az -> Crazyflie thrust-command scaling.
        self.lqr_az_to_thrust = 3500.0

        # ====================================================
        # BATTERY-AWARE HOVER FEEDFORWARD
        #
        # T_hover(V) =
        #     35000 + 11500 * (4.0 - Vbat)
        #
        # Based on the real-flight hover data.
        # ====================================================
        self.lqr_hover_base_4v = 35000.0
        self.lqr_hover_base_slope = 11000.0

        # Use only this battery range for feedforward.
        self.lqr_vbat_ff_min = 3.50
        self.lqr_vbat_ff_max = 4.00

        # Battery voltage used for hover feedforward.
        #
        # Capture ONCE at the first valid outer-LQR update,
        # then keep it fixed for the entire flight.
        self.lqr_vbat_frozen = None

        # Freeze battery only after the vehicle is airborne
        # and the battery is carrying motor load.
        self.lqr_vbat_freeze_height_m = 0.30

        # Partial compensation for battery sag AFTER the
        # airborne voltage has been frozen.
        #
        # 0     = fully frozen
        # 11500 = full live compensation
        # 5500  = intermediate compensation from flight data
        self.lqr_vbat_sag_slope = 5500.0

        # Current battery-compensated hover feedforward.
        self.lqr_thrust_base_dynamic = self.lqr_thrust_base

        self.lqr_thrust_cmd = self.lqr_thrust_base

        # Same outer-LQR -> thrust interface style
        # used in the successful Gazebo controller.


        # XY stays level close to the ground.
        self.xy_enable_height_m = 0.05

        # Fixed Lighthouse launch frame.
        self.frame_yaw_rad = 0.0

        # Attitude used during the Z-only startup test.
        self.launch_roll_deg = 0.0
        self.launch_pitch_deg = 0.0
        self.controller.init()
        self.stabilizer_step = 1
        self.control = Control()

        self.cf_state = State(
            attitude=Attitude(),
            position=Position(),
            velocity=Velocity(),
            acc=Axis3f(),
        )
        self.cf_sensors = SensorData(gyro=GyroData(), acc=AccData())
        self.cf_vbat = 4.2
        self._have_state = False
        self._have_sensor = False
        self._have_position = False

        self.ref_x0 = 0.0
        self.ref_y0 = 0.0
        self.ref_yaw0 = 0.0

        # Raw Lighthouse world coordinates.
        # The controller itself will use launch-relative x/y,
        # matching the Gazebo coordinate convention.
        self.lighthouse_x_world = 0.0
        self.lighthouse_y_world = 0.0
        self.lighthouse_z_world = 0.0
        self.local_xy_initialized = False

        if USE_SCRIPT_TRAJECTORY:
            _validate_trajectory()
            init_x, init_y, init_z, init_yaw = _sample_trajectory(0.0)
        else:
            init_x, init_y, init_z, init_yaw = target_x, target_y, target_z, target_yaw

        self.cf_setpoint = Setpoint()
        self.cf_setpoint.position = Position(x=init_x, y=init_y, z=init_z)
        self.cf_setpoint.velocity = Velocity(x=0.0, y=0.0, z=0.0)
        self.cf_setpoint.attitude = Attitude(roll=0.0, pitch=0.0, yaw=init_yaw)
        self.cf_setpoint.attitude_rate = AttitudeRate(roll=0.0, pitch=0.0, yaw=0.0)
        self.cf_setpoint.velocity_body = False
        self.cf_setpoint.mode = SetpointMode()

        # ====================================================
        # LQR OUTER / PID INNER MODES
        # ====================================================
        #
        # Same structure as ERAU/Vicon LQR:
        #
        # Outer LQR supplies:
        #   roll
        #   pitch
        #   thrust
        #
        # Original ControllerPID supplies:
        #   attitude PID
        #   rate PID
        #
        self.cf_setpoint.mode.x = StabMode.MODE_DISABLE
        self.cf_setpoint.mode.y = StabMode.MODE_DISABLE
        self.cf_setpoint.mode.z = StabMode.MODE_DISABLE

        self.cf_setpoint.mode.roll = StabMode.MODE_ABS
        self.cf_setpoint.mode.pitch = StabMode.MODE_ABS
        self.cf_setpoint.mode.yaw = StabMode.MODE_ABS

        self.cf = Crazyflie(rw_cache="./cache")
        self.motor_raw = MotorRaw(crazyflie=self.cf)

        # Lighthouse onboard-estimator XYZ
        self.log_position = LogConfig(
            name="LighthouseXYZ", period_in_ms=20
        )
        self.log_position.add_variable("stateEstimate.x", "FP16")
        self.log_position.add_variable("stateEstimate.y", "FP16")
        self.log_position.add_variable("stateEstimate.z", "FP16")

        self.log_state = LogConfig(name="HostPIDState", period_in_ms=10)
        self.log_state.add_variable("stateEstimate.vx", "FP16")
        self.log_state.add_variable("stateEstimate.vy", "FP16")
        self.log_state.add_variable("stateEstimate.vz", "FP16")
        #self.log_state.add_variable("stateEstimate.ax", "FP16")
        #self.log_state.add_variable("stateEstimate.ay", "FP16")
        #self.log_state.add_variable("stateEstimate.az", "FP16")
        #self.log_state.add_variable("stateEstimate.x", "FP16")
        #self.log_state.add_variable("stateEstimate.y", "FP16")
        #self.log_state.add_variable("stateEstimate.z", "FP16")
        self.log_state.add_variable("stateEstimate.pitch", "FP16")
        self.log_state.add_variable("stateEstimate.roll", "FP16")
        self.log_state.add_variable("stateEstimate.yaw", "FP16")
        #self.log_state.add_variable("stateEstimate.qx", "FP16")
        #self.log_state.add_variable("stateEstimate.qy", "FP16")
        #self.log_state.add_variable("stateEstimate.qz", "FP16")
        #self.log_state.add_variable("stateEstimate.qw", "FP16")
        self.log_state.add_variable("gyro.x", "FP16")
        self.log_state.add_variable("gyro.y", "FP16")
        self.log_state.add_variable("gyro.z", "FP16")
        self.log_state.add_variable("pm.vbat", "FP16")
        #self.log_sensor = LogConfig(name="HostPIDSensor", period_in_ms=10)
        #self.log_sensor.add_variable("gyro.x", "float")
        #self.log_sensor.add_variable("gyro.y", "float")
        #self.log_sensor.add_variable("gyro.z", "float")
        #self.log_sensor.add_variable("acc.x", "float")
        #self.log_sensor.add_variable("acc.y", "float")
        #self.log_sensor.add_variable("acc.z", "float")
        #self.log_sensor.add_variable("pm.vbat", "FP16")
        self.killed = False
        self.m1_multiplier = 1.0
        self.m2_multiplier = 1.0
        self.m3_multiplier = 1.0
        self.m4_multiplier = 1.0

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def kill(self):
        print("[KILLING DRONE]")
        self.killed = True
        self._stop_motors()

    def on_press(self, key):
        if key == keyboard.Key.space:
            self.kill()
            return
        ch = getattr(key, "char", None)
        if ch == "1":
            self.m1_multiplier = REDUCE_MULTIPLIER
        elif ch == "2":
            self.m2_multiplier = REDUCE_MULTIPLIER
        elif ch == "3":
            self.m3_multiplier = REDUCE_MULTIPLIER
        elif ch == "4":
            self.m4_multiplier = REDUCE_MULTIPLIER

    def run(self):
        cflib.crtp.init_drivers()
        print(f"Connecting to {self.uri}")

        with SyncCrazyflie(self.uri, cf=self.cf) as scf:
            logs_started = False
            try:
                self.cf.log.add_config(self.log_position)
                self.log_position.data_received_cb.add_callback(
                    self._log_position_callback
                )
                self.cf.log.add_config(self.log_state)
                self.log_state.data_received_cb.add_callback(self._log_state_callback)
                #self.cf.log.add_config(self.log_sensor)
                #self.log_sensor.data_received_cb.add_callback(self._log_sensor_callback)
                self.log_position.start()
                self.log_state.start()
                #self.log_sensor.start()
                logs_started = True
                scf.cf.param.set_value("motorPowerSet.enable", 1)
                self._wait_for_logs(timeout_s=5.0)

                # Use the fixed calibrated Lighthouse origin.
                #
                # Controller coordinates:
                #   x = x_raw - 0.07
                #   y = y_raw - (-0.12)
                #   z = z_raw - (-0.49)
                #
                # ====================================================
                # FIXED LAUNCH FRAME -- MATCH GAZEBO
                # ====================================================
                #
                # Gazebo successful test starts at:
                #   x = 0
                #   y = 0
                #   z = 0
                #   yaw = 0 in controller coordinates
                #
                # Use the CURRENT Lighthouse position as the
                # controller origin.
                #
                self.ref_x0 = self.lighthouse_x_world
                self.ref_y0 = self.lighthouse_y_world
                self.ref_z0 = self.lighthouse_z_world

                # Physical heading at launch.
                self.ref_yaw0 = self.cf_state.attitude.yaw

                # Rotate Lighthouse world X/Y into a FIXED frame
                # aligned with the Crazyflie heading at launch.
                #
                # Important:
                # this is NOT a continuously rotating body frame.
                self.frame_yaw_rad = math.radians(
                    self.ref_yaw0
                )

                self.local_xy_initialized = True

                # Match Gazebo yaw condition:
                # hold the physical heading present at startup.
                self.cf_setpoint.attitude.yaw = self.ref_yaw0

                # Controller coordinates now match Gazebo:
                # launch position = x=0, y=0.
                # Do NOT artificially force controller position to zero.
                #
                # Use the actual Lighthouse measurement relative to the
                # fixed calibrated origin.
                self.cf_state.position.x = (
                    self.lighthouse_x_world - self.ref_x0
                )
                self.cf_state.position.y = (
                    self.lighthouse_y_world - self.ref_y0
                )
                self.cf_state.position.z = (
                    self.lighthouse_z_world - self.ref_z0
                )

                print(
                    "[UTA ORIGIN CHECK] "
                    f"controller xyz=("
                    f"{self.cf_state.position.x:+.3f},"
                    f"{self.cf_state.position.y:+.3f},"
                    f"{self.cf_state.position.z:+.3f})"
                )

                print(
                    "[Lighthouse] Ready: "
                    f"x0_world={self.ref_x0:.3f} "
                    f"y0_world={self.ref_y0:.3f} "
                    f"z0_world={self.ref_z0:.3f} "
                    f"yaw0={self.ref_yaw0:.1f} deg"
                )
                self._spinup(duration_s=1.0)

                # ====================================================
                # WAIT FOR LIGHTHOUSE / EKF TO ACTUALLY SETTLE
                # ====================================================
                #
                # Do not start LQR while stateEstimate.vx/vy/vz still
                # contains the large convergence transient.
                print("[ESTIMATOR SETTLE] waiting for stable position/velocity...")

                settle_deadline = time.monotonic() + 6.0
                stable_count = 0
                last_pos = None

                while time.monotonic() < settle_deadline:
                    # Keep motors only at idle during estimator settling.
                    self.motor_raw.send_motor_raw(
                        IDLE_THRUST,
                        IDLE_THRUST,
                        IDLE_THRUST,
                        IDLE_THRUST,
                    )

                    pos = (
                        self.lighthouse_x_world,
                        self.lighthouse_y_world,
                        self.lighthouse_z_world,
                    )

                    vx = float(self.cf_state.velocity.x)
                    vy = float(self.cf_state.velocity.y)
                    vz = float(self.cf_state.velocity.z)

                    if last_pos is None:
                        position_stable = False
                    else:
                        position_step = max(
                            abs(pos[0] - last_pos[0]),
                            abs(pos[1] - last_pos[1]),
                            abs(pos[2] - last_pos[2]),
                        )
                        position_stable = position_step < 0.010

                    velocity_stable = (
                        abs(vx) < 0.08
                        and abs(vy) < 0.08
                        and abs(vz) < 0.05
                    )

                    if position_stable and velocity_stable:
                        stable_count += 1
                    else:
                        stable_count = 0

                    # Diagnostic only. No control behavior changed.
                    if last_pos is None:
                        position_step_dbg = float("nan")
                    else:
                        position_step_dbg = max(
                            abs(pos[0] - last_pos[0]),
                            abs(pos[1] - last_pos[1]),
                            abs(pos[2] - last_pos[2]),
                        )

                    print(
                        "[SETTLE DEBUG] "
                        f"world=({pos[0]:+.3f},"
                        f"{pos[1]:+.3f},"
                        f"{pos[2]:+.3f}) | "
                        f"vel=({vx:+.3f},"
                        f"{vy:+.3f},"
                        f"{vz:+.3f}) | "
                        f"step={position_step_dbg:.4f} | "
                        f"pos_ok={position_stable} "
                        f"vel_ok={velocity_stable} "
                        f"count={stable_count}/10"
                    )

                    if stable_count >= 10:
                        print(
                            "[ESTIMATOR SETTLED] "
                            f"world=({pos[0]:+.3f},"
                            f"{pos[1]:+.3f},"
                            f"{pos[2]:+.3f}) "
                            f"vel=({vx:+.3f},"
                            f"{vy:+.3f},"
                            f"{vz:+.3f})"
                        )
                        break

                    last_pos = pos
                    time.sleep(0.05)

                else:
                    raise RuntimeError(
                        "Lighthouse/EKF did not settle. "
                        "Flight aborted before takeoff."
                    )

                # ====================================================
                # RE-ZERO AFTER LIGHTHOUSE / EKF HAS SETTLED
                # ====================================================
                #
                # The first real test showed ~6-8 cm estimator motion
                # between initial log acquisition and controller start.
                #
                # Capture the launch frame NOW, immediately before
                # closed-loop control.
                self.ref_x0 = self.lighthouse_x_world
                self.ref_y0 = self.lighthouse_y_world
                self.ref_z0 = self.lighthouse_z_world

                self.ref_yaw0 = self.cf_state.attitude.yaw
                self.frame_yaw_rad = math.radians(
                    self.ref_yaw0
                )

                # Force an immediate zero-state update for the new frame.
                self.cf_state.position.x = 0.0
                self.cf_state.position.y = 0.0
                self.cf_state.position.z = 0.0

                self.cf_setpoint.attitude.yaw = self.ref_yaw0

                print(
                    "[FINAL LAUNCH FRAME] "
                    f"world=("
                    f"{self.ref_x0:+.3f},"
                    f"{self.ref_y0:+.3f},"
                    f"{self.ref_z0:+.3f}) "
                    f"yaw0={self.ref_yaw0:+.1f} deg"
                )

                # ====================================================
                # INNER PID STARTUP INITIALIZATION
                # ====================================================
                #
                # For the Z-only test, do NOT suddenly command
                # roll=pitch=0 while the vehicle is still on the floor.
                #
                # Hold the measured launch attitude and reset all
                # attitude/rate PID histories to the current state.
                self.launch_roll_deg = self.cf_state.attitude.roll
                self.launch_pitch_deg = self.cf_state.attitude.pitch

                self.cf_setpoint.attitude.roll = self.launch_roll_deg
                self.cf_setpoint.attitude.pitch = self.launch_pitch_deg
                self.cf_setpoint.attitude.yaw = self.ref_yaw0

                self.controller.attitude_controller.reset_all_pid(
                    self.cf_state.attitude.roll,
                    self.cf_state.attitude.pitch,
                    self.cf_state.attitude.yaw,
                )

                print(
                    "[PID RESET] "
                    f"hold roll={self.launch_roll_deg:+.2f} "
                    f"pitch={self.launch_pitch_deg:+.2f} "
                    f"yaw={self.ref_yaw0:+.2f}"
                )

                # Match the stable Gazebo startup:
                # do not manually reset attitude/rate PID states.
                self.stabilizer_step = 0

                print(
                    "[CONTROLLER START] "
                    f"roll={self.cf_state.attitude.roll:.2f} "
                    f"pitch={self.cf_state.attitude.pitch:.2f} "
                    f"yaw={self.cf_state.attitude.yaw:.2f} "
                    f"yaw_ref={self.ref_yaw0:.2f}"
                )

                self._log_time_origin = time.monotonic()
                
                # ====================================================
                # UTA FLIGHT SEQUENCE
                #
                #   smooth takeoff
                #       ->
                #   hover
                #       ->
                #   smooth landing
                #
                # Horizontal reference is ALWAYS:
                #
                #   x_ref = 0
                #   y_ref = 0
                #
                # ====================================================

                target_z = self.cf_setpoint.position.z

                # ----------------------------------------------------
                # TAKEOFF
                # ----------------------------------------------------

                print()
                print("============================================")
                print("UTA SMOOTH TAKEOFF")
                print("============================================")

                # Start the reference from the ACTUAL measured
                # Lighthouse-relative altitude. This prevents an
                # artificial vertical step at controller startup.
                takeoff_z_start = self.cf_state.position.z

                print(
                    "[TAKEOFF START] "
                    f"measured z_start={takeoff_z_start:+.3f} m"
                )

                self._control_z_ramp(
                    z_start=takeoff_z_start,
                    z_final=target_z,
                    duration_s=self.takeoff_seconds,
                )

                if not self.killed:

                    # ------------------------------------------------
                    # HOVER
                    # ------------------------------------------------

                    print()
                    print("============================================")
                    print("UTA HOVER")
                    print("============================================")

                    self.cf_setpoint.position.x = 0.0
                    self.cf_setpoint.position.y = 0.0
                    self.cf_setpoint.position.z = target_z

                    self.cf_setpoint.velocity.x = 0.0
                    self.cf_setpoint.velocity.y = 0.0
                    self.cf_setpoint.velocity.z = 0.0

                    self.cf_setpoint.attitude.yaw = self.ref_yaw0

                    self._control_for(
                        duration_s=self.run_seconds,
                        follow_script_trajectory=False,
                    )

                if self.do_land and not self.killed:

                    # ------------------------------------------------
                    # LAND
                    # ------------------------------------------------

                    print()
                    print("============================================")
                    print("UTA SMOOTH LANDING")
                    print("============================================")

                    self._land_until_ground(
                        duration_s=self.land_seconds,
                    )
            finally:
                self._stop_motors()

                # Always save collected forward-LQR data,
                # including after normal landing, Ctrl+C,
                # or emergency termination that reaches finally.
                self._save_forward_lqr_csv()
                if logs_started:
                    self.log_position.stop()
                    self.log_state.stop()
                    #self.log_sensor.stop()
                with open(WAYPOINT_DATA_PATH, "w", encoding="utf-8") as json_file:
                    json.dump(json_data, json_file, indent=4)

    def _wait_for_logs(self, timeout_s: float):
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if self._have_state and self._have_sensor and self._have_position:
                return
            time.sleep(0.01)
        raise RuntimeError("Timed out waiting for state/sensor logs")

    def _spinup(self, duration_s: float):
        print(f"Spinup for {duration_s:.1f}s")
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            self.motor_raw.send_motor_raw(
                IDLE_THRUST, IDLE_THRUST, IDLE_THRUST, IDLE_THRUST
            )
            time.sleep(0.03)

    def _control_z_ramp(
        self,
        z_start: float,
        z_final: float,
        duration_s: float,
    ):
        """
        Smooth vertical reference using cubic smoothstep.

        x_ref = 0
        y_ref = 0

        z_ref = z_start + (z_final-z_start)*(3*s^2 - 2*s^3)

        vz_ref = (z_final-z_start)*(6*s - 6*s^2)/T

        where s = t/T.

        This gives zero vertical reference velocity at the
        beginning and end of the maneuver.
        """

        print(
            f"[Z RAMP] {z_start:.3f} -> {z_final:.3f} m "
            f"in {duration_s:.1f} s"
        )

        phase_start = time.monotonic()
        end = phase_start + duration_s

        next_tick = time.monotonic()
        next_print = time.monotonic()

        while time.monotonic() < end:

            if self.killed:
                return

            now = time.monotonic()

            elapsed = now - phase_start
            s = min(max(elapsed / duration_s, 0.0), 1.0)

            smooth = 3.0 * s * s - 2.0 * s * s * s

            z_ref = (
                z_start
                + (z_final - z_start) * smooth
            )

            vz_ref = (
                (z_final - z_start)
                * (6.0 * s - 6.0 * s * s)
                / duration_s
            )

            # ================================================
            # NO HORIZONTAL TRAJECTORY
            # ================================================

            self.cf_setpoint.position.x = 0.0
            self.cf_setpoint.position.y = 0.0
            self.cf_setpoint.position.z = z_ref

            self.cf_setpoint.velocity.x = 0.0
            self.cf_setpoint.velocity.y = 0.0
            self.cf_setpoint.velocity.z = vz_ref

            # Keep startup heading.
            self.cf_setpoint.attitude.yaw = self.ref_yaw0

            if self._log_time_origin is None:
                self._log_time_origin = now

            json_data["time_s"].append(
                now - self._log_time_origin
            )

            json_data["position_x"].append(
                self.cf_state.position.x
            )
            json_data["setpoint_x"].append(
                self.cf_setpoint.position.x
            )

            json_data["position_y"].append(
                self.cf_state.position.y
            )
            json_data["setpoint_y"].append(
                self.cf_setpoint.position.y
            )

            json_data["position_z"].append(
                self.cf_state.position.z
            )
            json_data["setpoint_z"].append(
                self.cf_setpoint.position.z
            )

            self._control_step()

            if now >= next_print:
                print(
                    "[Z RAMP] "
                    f"z={self.cf_state.position.z:+.3f} "
                    f"z_ref={z_ref:+.3f} "
                    f"vz={self.cf_state.velocity.z:+.3f} "
                    f"vz_ref={vz_ref:+.3f} | "
                    f"x={self.cf_state.position.x:+.3f} "
                    f"y={self.cf_state.position.y:+.3f}"
                )
                next_print = now + 0.25

            next_tick += self.loop_period

            sleep_time = next_tick - time.monotonic()

            if sleep_time > 0.0:
                time.sleep(sleep_time)

        # Force exact final reference.
        self.cf_setpoint.position.x = 0.0
        self.cf_setpoint.position.y = 0.0
        self.cf_setpoint.position.z = z_final

        self.cf_setpoint.velocity.x = 0.0
        self.cf_setpoint.velocity.y = 0.0
        self.cf_setpoint.velocity.z = 0.0

    def _land_until_ground(self, duration_s: float):
        """
        Controlled landing.

        Hover time is handled separately.

        During landing:
            z > 0.005 m  -> keep landing
            z <= 0.005 m -> motors OFF immediately

        There is NO confirmation timer.
        """

        MOTOR_OFF_HEIGHT_M = 0.050
        FINAL_DESCENT_RATE_MPS = 0.05

        actual_start_z = float(
            self.cf_state.position.z
        )

        print()
        print("============================================")
        print("UTA CONTROLLED LANDING")
        print("============================================")
        print(
            f"[LAND START] measured z_start="
            f"{actual_start_z:+.4f} m"
        )
        print(
            f"[MOTOR OFF THRESHOLD] "
            f"measured z <= "
            f"{MOTOR_OFF_HEIGHT_M:.4f} m"
        )

        # ====================================================
        # STAGE 1
        #
        # Smooth descent from ACTUAL current measured height
        # to the approach height, normally 0.05 m.
        #
        # Check measured height continuously.
        # ====================================================

        z_start = actual_start_z
        z_final = float(self.land_z)

        start_time = time.monotonic()
        end_time = start_time + duration_s

        next_tick = start_time
        next_print = start_time

        while (
            time.monotonic() < end_time
            and not self.killed
        ):

            now = time.monotonic()

            measured_z = float(
                self.cf_state.position.z
            )

            measured_vz = float(
                self.cf_state.velocity.z
            )

            # ================================================
            # NORMAL MOTOR-OFF CONDITION
            # ================================================
            if measured_z <= MOTOR_OFF_HEIGHT_M:

                print()
                print(
                    f"[GROUND REACHED] "
                    f"measured z={measured_z:+.4f} m"
                )
                print("[MOTORS OFF]")

                self._stop_motors()
                return

            elapsed = now - start_time

            alpha = min(
                max(
                    elapsed / max(duration_s, 1e-6),
                    0.0,
                ),
                1.0,
            )

            # Cubic smoothstep
            smooth = (
                3.0 * alpha**2
                - 2.0 * alpha**3
            )

            smooth_dot = (
                6.0 * alpha
                - 6.0 * alpha**2
            ) / max(duration_s, 1e-6)

            z_ref = (
                z_start
                + (z_final - z_start) * smooth
            )

            vz_ref = (
                (z_final - z_start)
                * smooth_dot
            )

            self.cf_setpoint.position.x = 0.0
            self.cf_setpoint.position.y = 0.0
            self.cf_setpoint.position.z = z_ref

            self.cf_setpoint.velocity.x = 0.0
            self.cf_setpoint.velocity.y = 0.0
            self.cf_setpoint.velocity.z = vz_ref

            self.cf_setpoint.attitude.yaw = (
                self.ref_yaw0
            )

            self._control_step()

            if now >= next_print:

                print(
                    f"[LANDING] "
                    f"z={measured_z:+.4f} "
                    f"z_ref={z_ref:+.4f} "
                    f"vz={measured_vz:+.3f} "
                    f"vz_ref={vz_ref:+.3f}"
                )

                next_print = now + 0.25

            next_tick += self.loop_period

            sleep_time = (
                next_tick - time.monotonic()
            )

            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

        if self.killed:
            return

        # ====================================================
        # STAGE 2
        #
        # We reached the 5-cm approach reference.
        #
        # There is NO fixed landing timer from here.
        # Continue lowering the reference until the ACTUAL
        # measured height reaches <= 5 mm.
        # ====================================================

        print()
        print("============================================")
        print("UTA FINAL APPROACH")
        print("============================================")
        print(
            "[FINAL APPROACH] "
            "continuously checking measured z"
        )

        z_ref = float(self.land_z)

        previous_time = time.monotonic()
        next_tick = previous_time
        next_print = previous_time

        while not self.killed:

            now = time.monotonic()

            dt = now - previous_time
            previous_time = now

            dt = min(
                max(dt, 0.0),
                0.05,
            )

            measured_z = float(
                self.cf_state.position.z
            )

            measured_vz = float(
                self.cf_state.velocity.z
            )

            # ================================================
            # THE NORMAL MOTOR-OFF CONDITION
            #
            # NO 0.10 s timer.
            # NO extra 10 s timer.
            # ================================================
            if measured_z <= MOTOR_OFF_HEIGHT_M:

                print()
                print(
                    f"[GROUND REACHED] "
                    f"measured z={measured_z:+.4f} m"
                )

                print(
                    "[LANDING COMPLETE] "
                    "measured height <= 5 mm"
                )

                print("[MOTORS OFF]")

                self._stop_motors()
                return

            # Slowly move the reference downward.
            z_ref -= (
                FINAL_DESCENT_RATE_MPS * dt
            )

            # This is only a virtual-reference limit.
            z_ref = max(z_ref, -0.20)

            self.cf_setpoint.position.x = 0.0
            self.cf_setpoint.position.y = 0.0
            self.cf_setpoint.position.z = z_ref

            self.cf_setpoint.velocity.x = 0.0
            self.cf_setpoint.velocity.y = 0.0
            self.cf_setpoint.velocity.z = (
                -FINAL_DESCENT_RATE_MPS
            )

            self.cf_setpoint.attitude.yaw = (
                self.ref_yaw0
            )

            self._control_step()

            if now >= next_print:

                print(
                    f"[FINAL APPROACH] "
                    f"z={measured_z:+.4f} "
                    f"z_ref={z_ref:+.4f} "
                    f"vz={measured_vz:+.3f}"
                )

                next_print = now + 0.25

            next_tick += self.loop_period

            sleep_time = (
                next_tick - time.monotonic()
            )

            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()


    def _control_for(self, duration_s: float, follow_script_trajectory: bool = False):
        print(
            "Control target "
            f"x={self.cf_setpoint.position.x:.2f}, "
            f"y={self.cf_setpoint.position.y:.2f}, "
            f"z={self.cf_setpoint.position.z:.2f}, "
            f"yaw={self.cf_setpoint.attitude.yaw:.1f} deg"
        )
        phase_start = time.monotonic()
        end = phase_start + duration_s
        next_tick = time.monotonic()
        next_print = time.monotonic()

        # Battery samples belonging to this control phase
        hover_vbat_start = len(json_data["vbat"])

        # ----------------------------------------------------
        # HEIGHT STATISTICS
        #
        # position_z is already logged every controller cycle.
        # Remember the first index belonging to this hover.
        # ----------------------------------------------------
        hover_log_start = len(json_data["position_z"])

        while time.monotonic() < end:
            if self.killed:
                return
            now = time.monotonic()
            if follow_script_trajectory:
                elapsed = now - phase_start
                x, y, z, yaw = _sample_trajectory(elapsed)
                self.cf_setpoint.position.x = x
                self.cf_setpoint.position.y = y
                self.cf_setpoint.position.z = z
                self.cf_setpoint.attitude.yaw = self.ref_yaw0 + yaw

            if self._log_time_origin is None:
                self._log_time_origin = now
            json_data["time_s"].append(now - self._log_time_origin)
            json_data["position_x"].append(self.cf_state.position.x)
            json_data["setpoint_x"].append(self.cf_setpoint.position.x)
            json_data["position_y"].append(self.cf_state.position.y)
            json_data["setpoint_y"].append(self.cf_setpoint.position.y)
            json_data["position_z"].append(self.cf_state.position.z)
            json_data["setpoint_z"].append(self.cf_setpoint.position.z)
            json_data["vbat"].append(float(self.cf_vbat))
            #json_data["velocity_x"].append(self.cf_state.velocity.x)
            #json_data["velocity_y"].append(self.cf_state.velocity.y)
            #json_data["velocity_z"].append(self.cf_state.velocity.z)
            #json_data["roll"].append(self.cf_state.attitude.roll)
            #json_data["pitch"].append(self.cf_state.attitude.pitch)
            #json_data["yaw"].append(self.cf_state.attitude.yaw)
            #json_data["gyro_x"].append(self.cf_sensors.gyro.x)
            #json_data["gyro_y"].append(self.cf_sensors.gyro.y)
            #json_data["gyro_z"].append(self.cf_sensors.gyro.z)
            #json_data["vbat"].append(self.cf_vbat)

            self._control_step()
            next_tick += self.loop_period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

            if time.monotonic() >= next_print:
                print(
                    "state "
                    f"x={self.cf_state.position.x:+.3f} "
                    f"y={self.cf_state.position.y:+.3f} "
                    f"z={self.cf_state.position.z:+.3f} | "
                    f"sp=({self.cf_setpoint.position.x:+.3f},"
                    f"{self.cf_setpoint.position.y:+.3f},"
                    f"{self.cf_setpoint.position.z:+.3f}) | "
                    f"v=({self.cf_state.velocity.x:+.3f},"
                    f"{self.cf_state.velocity.y:+.3f},"
                    f"{self.cf_state.velocity.z:+.3f}) | "
                    f"thrust={self.control.thrust:.1f} "
                    f"vbat={self.cf_vbat:.3f} V "
                    f"vbat_frozen={(self.lqr_vbat_frozen if self.lqr_vbat_frozen is not None else float('nan')):.3f} V "
                    f"hover_base={self.lqr_thrust_base_dynamic:.1f} "
                    f"inner_out=({self.control.roll:+.0f},"
                    f"{self.control.pitch:+.0f},"
                    f"{self.control.yaw:+.0f})"
                )

                try:
                    lqr = self.controller.outer_lqr

                    err = lqr.x_prev

                    if err is not None:
                        print(
                            "[UTA XY DEBUG] "
                            f"err_xy=({float(err[0,0]):+.3f},"
                            f"{float(err[1,0]):+.3f}) "
                            f"err_vxy=({float(err[3,0]):+.3f},"
                            f"{float(err[4,0]):+.3f}) | "
                            f"LQR_phi={self.controller.last_phi_des_deg:+.3f}deg "
                            f"LQR_theta_interface="
                            f"{self.controller.last_theta_des_deg:+.3f}deg | "
                            f"att_actual=({self.cf_state.attitude.roll:+.3f},"
                            f"{self.cf_state.attitude.pitch:+.3f})deg | "
                            f"rate_des=({self.controller.rate_des_roll:+.2f},"
                            f"{self.controller.rate_des_pitch:+.2f})deg/s | "
                            f"gyro=({self.cf_sensors.gyro.x:+.2f},"
                            f"{-self.cf_sensors.gyro.y:+.2f})deg/s"
                        )
                except Exception as exc:
                    print(f"[UTA XY DEBUG ERROR] {exc}")

                print(
                    "[FRAME] "
                    f"world=({self.lighthouse_x_world:+.3f},"
                    f"{self.lighthouse_y_world:+.3f}) | "
                    f"local=({self.cf_state.position.x:+.3f},"
                    f"{self.cf_state.position.y:+.3f}) | "
                    f"yaw0={self.ref_yaw0:+.1f} deg"
                )

    
                next_print = time.monotonic() + 0.25


        # ====================================================
        # UTA HEIGHT SUMMARY
        # ====================================================
        #
        # Only generate this summary for the normal fixed
        # hover test, not a scripted trajectory.
        #
        if not follow_script_trajectory:

            hover_values = json_data["position_z"][
                hover_log_start:
            ]

            if hover_values:

                hover_z = np.asarray(
                    hover_values,
                    dtype=float,
                )

                # All samples collected so far:
                # takeoff + hover.
                #
                # Landing has NOT started yet, so this gives
                # the maximum height reached before landing.
                flight_z = np.asarray(
                    json_data["position_z"],
                    dtype=float,
                )

                target_z = float(
                    self.cf_setpoint.position.z
                )

                max_height = float(
                    np.max(flight_z)
                )

                hover_mean = float(
                    np.mean(hover_z)
                )

                hover_min = float(
                    np.min(hover_z)
                )

                hover_max = float(
                    np.max(hover_z)
                )

                final_height = float(
                    hover_z[-1]
                )

                # --------------------------------------------
                # Define "stable height" as the final 3 s of
                # the hover period.
                # --------------------------------------------
                stable_window_s = min(
                    3.0,
                    float(duration_s),
                )

                stable_samples = max(
                    1,
                    int(
                        round(
                            stable_window_s
                            * self.loop_hz
                        )
                    ),
                )

                stable_samples = min(
                    stable_samples,
                    len(hover_z),
                )

                stable_z = hover_z[
                    -stable_samples:
                ]

                stable_mean = float(
                    np.mean(stable_z)
                )

                stable_min = float(
                    np.min(stable_z)
                )

                stable_max = float(
                    np.max(stable_z)
                )

                stable_std = float(
                    np.std(stable_z)
                )

                overshoot = (
                    max_height
                    - target_z
                )

                stable_error = (
                    stable_mean
                    - target_z
                )

                print()
                print(
                    "============================================"
                )
                print(
                    "UTA HEIGHT SUMMARY"
                )
                print(
                    "============================================"
                )

                print(
                    f"Target height                 : "
                    f"{target_z:.4f} m"
                )

                print(
                    f"Maximum height reached        : "
                    f"{max_height:.4f} m"
                )

                print(
                    f"Maximum overshoot             : "
                    f"{overshoot:+.4f} m"
                )

                print(
                    f"Average hover height          : "
                    f"{hover_mean:.4f} m"
                )

                print(
                    f"Hover range                   : "
                    f"{hover_min:.4f} -> "
                    f"{hover_max:.4f} m"
                )

                print(
                    f"Stable height (last "
                    f"{stable_window_s:.1f} sec)    : "
                    f"{stable_mean:.4f} m"
                )

                print(
                    f"Stable height range           : "
                    f"{stable_min:.4f} -> "
                    f"{stable_max:.4f} m"
                )

                print(
                    f"Stable std deviation          : "
                    f"{stable_std:.4f} m"
                )

                print(
                    f"Final hover height            : "
                    f"{final_height:.4f} m"
                )

                print(
                    f"Stable-state Z error          : "
                    f"{stable_error:+.4f} m"
                )

                print(
                    "============================================"
                )
                print()


        # ====================================================
        # UTA BATTERY SUMMARY
        # ====================================================
        if not follow_script_trajectory:
            hover_vbat = json_data["vbat"][hover_vbat_start:]

            if hover_vbat:
                vb = np.asarray(hover_vbat, dtype=float)

                print()
                print("============================================")
                print("UTA BATTERY SUMMARY")
                print("============================================")
                print(
                    f"Battery at hover start        : "
                    f"{vb[0]:.3f} V"
                )
                print(
                    f"Average battery during hover  : "
                    f"{np.mean(vb):.3f} V"
                )
                print(
                    f"Minimum battery during hover  : "
                    f"{np.min(vb):.3f} V"
                )
                print(
                    f"Maximum battery during hover  : "
                    f"{np.max(vb):.3f} V"
                )
                print(
                    f"Battery at hover end          : "
                    f"{vb[-1]:.3f} V"
                )
                print(
                    f"Voltage drop during hover     : "
                    f"{vb[0] - vb[-1]:.3f} V"
                )
                print("============================================")
                print()


    def _record_forward_lqr_sample(
        self,
        lqr_out,
    ):
        """
        Save one sample at one OUTER-LQR timestep.

        State:
            x y z vx vy vz

        Reference:
            x_ref y_ref z_ref
            vx_ref vy_ref vz_ref

        Control:
            phi_des   [rad]
            theta_des [rad]
            delta_T   [N]
        """

        now = time.monotonic()

        if self._log_time_origin is None:
            self._log_time_origin = now

        t = now - self._log_time_origin

        phi_des = float(
            lqr_out.phi_des_rad
        )

        theta_des = float(
            lqr_out.theta_des_rad
        )

        # Current Gazebo-matched outer model produces
        # vertical acceleration command:
        #
        #       az = delta_T / m
        #
        # therefore:
        #
        #       delta_T = m * az
        #
        az_cmd = float(
            lqr_out.az_cmd_mps2
        )

        delta_T = (
            self.forward_lqr_mass_kg
            * az_cmd
        )

        row = [
            float(t),

            float(self.cf_state.position.x),
            float(self.cf_state.position.y),
            float(self.cf_state.position.z),

            float(self.cf_state.velocity.x),
            float(self.cf_state.velocity.y),
            float(self.cf_state.velocity.z),

            float(self.cf_setpoint.position.x),
            float(self.cf_setpoint.position.y),
            float(self.cf_setpoint.position.z),

            float(self.cf_setpoint.velocity.x),
            float(self.cf_setpoint.velocity.y),
            float(self.cf_setpoint.velocity.z),

            phi_des,
            theta_des,
            float(delta_T),
        ]

        self.forward_lqr_rows.append(row)


    def _save_forward_lqr_csv(self):
        """
        Automatically save all forward-LQR samples.
        """

        if not self.forward_lqr_rows:
            print(
                "[TRAINING CSV] "
                "No forward-LQR samples to save."
            )
            return

        os.makedirs(
            self.forward_lqr_save_dir,
            exist_ok=True,
        )

        header = [
            "time",

            "x",
            "y",
            "z",

            "vx",
            "vy",
            "vz",

            "x_ref",
            "y_ref",
            "z_ref",

            "vx_ref",
            "vy_ref",
            "vz_ref",

            "phi_des",
            "theta_des",
            "delta_T",
        ]

        with open(
            self.forward_lqr_csv_path,
            "w",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(header)

            writer.writerows(
                self.forward_lqr_rows
            )

        print()
        print(
            "============================================"
        )
        print(
            "FORWARD LQR TRAINING DATA SAVED"
        )
        print(
            "============================================"
        )
        print(
            f"CSV: {self.forward_lqr_csv_path}"
        )
        print(
            f"Samples: "
            f"{len(self.forward_lqr_rows)}"
        )
        print(
            "Sampling: one row per outer-LQR update"
        )
        print(
            "phi_des, theta_des : rad"
        )
        print(
            "delta_T            : N"
        )
        print(
            "============================================"
        )
        print()


    def _control_step(self):
        if self.killed:
            return

        # Automatic emergency cutoff.
        if (
            abs(self.cf_state.attitude.roll) > 20.0
            or abs(self.cf_state.attitude.pitch) > 20.0
        ):
            print(
                "[AUTO KILL] "
                f"roll={self.cf_state.attitude.roll:.1f} "
                f"pitch={self.cf_state.attitude.pitch:.1f}"
            )
            self.kill()
            return
        # ====================================================
        # UTA OUTER LQR @ 100 Hz
        # ORIGINAL ERAU ControllerPID INNER LOOP @ 500 Hz
        # ====================================================

        if (self.outer_counter % self.outer_update_divider) == 0:

            lqr_out = self.outer_lqr.compute(
                state=self.cf_state,
                setpoint=self.cf_setpoint,
                dt=0.01,
            )

            # ------------------------------------------------
            # FORWARD LQR TRAINING SAMPLE
            #
            # This is inside the outer-LQR update block,
            # therefore one CSV row = one LQR timestep.
            # ------------------------------------------------
            self._record_forward_lqr_sample(
                lqr_out
            )

            # Same LQR -> inner PID interface as the Vicon code.
            self.lqr_roll_des_deg = float(
                np.degrees(lqr_out.phi_des_rad)
            )

            self.lqr_pitch_des_deg = float(
                -np.degrees(lqr_out.theta_des_rad)
            )

            # Gazebo LQR third input:
            #     az_cmd [m/s^2]
            #
            # Keep the SAME outer dynamics:
            #
            #     z_ddot = az_cmd
            #
            # For the real Crazyflie:
            #
            #     delta_T = m * az_cmd
            #
            # therefore:
            #
            #     total_T = m * (g + az_cmd)
            #
            self.lqr_az_cmd = float(
                lqr_out.az_cmd_mps2
            )

            # =================================================
            # BATTERY-AWARE HOVER FEEDFORWARD
            # =================================================
            #
            # Keep the LQR itself unchanged.
            #
            #       T_cmd = T_hover(Vbat) + 3500 * az_cmd
            #
            # where
            #
            #       T_hover(Vbat)
            #       = 35000 + 11500 * (4.0 - Vbat_filtered)
            #
            vbat_now = float(self.cf_vbat)

            # =================================================
            # FREEZE BATTERY VOLTAGE ONCE PER FLIGHT
            # =================================================
            #
            # The first valid battery measurement seen by the
            # outer LQR becomes the feedforward voltage for the
            # entire flight.
            #
            # It does NOT continue increasing hover thrust as
            # loaded battery voltage sags during hover.
            #
            if self.lqr_vbat_frozen is None:
                if (
                    self.cf_state.position.z
                    >= self.lqr_vbat_freeze_height_m
                    and np.isfinite(vbat_now)
                    and 3.0 <= vbat_now <= 4.5
                ):
                    self.lqr_vbat_frozen = vbat_now

                    print()
                    print(
                        "[VBAT FROZEN @ AIRBORNE] "
                        f"z={self.cf_state.position.z:.3f} m "
                        f"vbat={self.lqr_vbat_frozen:.3f} V"
                    )

            # Safe fallback only if the first measurement
            # has not arrived yet.
            if self.lqr_vbat_frozen is None:
                vbat_for_ff = 3.90
            else:
                vbat_for_ff = float(
                    np.clip(
                        self.lqr_vbat_frozen,
                        self.lqr_vbat_ff_min,
                        self.lqr_vbat_ff_max,
                    )
                )

            # Battery-dependent hover feedforward.
            # Base calculated from the battery voltage measured
            # once at z >= 0.30 m.
            frozen_hover_base = float(
                self.lqr_hover_base_4v
                + self.lqr_hover_base_slope
                * (4.0 - vbat_for_ff)
            )

            # Partial compensation for voltage sag that occurs
            # after the airborne voltage was frozen.
            #
            # Do not compensate voltage increases.
            sag_v = max(
                0.0,
                float(vbat_for_ff) - float(vbat_now)
            )

            sag_comp = (
                self.lqr_vbat_sag_slope * sag_v
            )

            self.lqr_thrust_base_dynamic = float(
                frozen_hover_base + sag_comp
            )

            # LQR vertical correction is UNCHANGED.
            self.lqr_thrust_cmd = float(
                np.clip(
                    self.lqr_thrust_base_dynamic
                    + self.lqr_az_cmd
                    * self.lqr_az_to_thrust,
                    20000.0,
                    60000.0,
                )
            )

        self.outer_counter += 1

        # ====================================================
        # GAZEBO-MATCHED XY OUTER LQR
        # ====================================================
        #
        # Near the floor keep the vehicle level.
        # Once safely airborne, use the LQR roll/pitch commands.
        #
        if (
            self.cf_state.position.z
            >= self.xy_enable_height_m
        ):
            self.cf_setpoint.attitude.roll = (
                self.lqr_roll_des_deg
            )

            self.cf_setpoint.attitude.pitch = (
                self.lqr_pitch_des_deg
            )

        else:
            # Z-only validation:
            # command level attitude so the vehicle does not
            # continuously accelerate horizontally.
            self.cf_setpoint.attitude.roll = 0.0
            self.cf_setpoint.attitude.pitch = 0.0

        # Hold startup yaw.
        self.cf_setpoint.attitude.yaw = self.ref_yaw0

        # Vertical LQR remains active.
        self.cf_setpoint.thrust = self.lqr_thrust_cmd

        # Ensure position PID outputs are overridden by the
        # manually supplied attitude/thrust commands.
        self.cf_setpoint.mode.x = StabMode.MODE_DISABLE
        self.cf_setpoint.mode.y = StabMode.MODE_DISABLE
        self.cf_setpoint.mode.z = StabMode.MODE_DISABLE
        self.cf_setpoint.mode.roll = StabMode.MODE_ABS
        self.cf_setpoint.mode.pitch = StabMode.MODE_ABS
        self.cf_setpoint.mode.yaw = StabMode.MODE_ABS

        # ORIGINAL ERAU inner attitude/rate controller.
        self.controller.controller_pid(
            self.control,
            self.cf_setpoint,
            self.cf_sensors,
            self.cf_state,
            self.stabilizer_step,
        )

        # ====================================================
        # INNER PID / MOTOR-MIX DEBUG
        # ====================================================
        # Print the active ERAU inner PID outputs and the
        # corresponding virtual motor mix. These PID outputs
        # are ACTIVE and continue to the real motor mixer.

        if self.stabilizer_step % 50 == 0:
            _dbg_r = float(self.control.roll)
            _dbg_p = float(self.control.pitch)
            _dbg_y = float(self.control.yaw)
            _dbg_T = float(self.control.thrust)

            _dbg_m1 = _dbg_T - _dbg_r / 2.0 + _dbg_p / 2.0 + _dbg_y
            _dbg_m2 = _dbg_T - _dbg_r / 2.0 - _dbg_p / 2.0 - _dbg_y
            _dbg_m3 = _dbg_T + _dbg_r / 2.0 - _dbg_p / 2.0 + _dbg_y
            _dbg_m4 = _dbg_T + _dbg_r / 2.0 + _dbg_p / 2.0 - _dbg_y

            _yaw_actual = float(self.cf_state.attitude.yaw)
            _yaw_ref = float(self.cf_setpoint.attitude.yaw)
            _yaw_des = float(self.controller.attitude_desired.yaw)
            _yaw_rate_des = float(self.controller.rate_desired.yaw)
            _gyro_z = float(self.cf_sensors.gyro.z)
            _yaw_applied = float(np.clip(_dbg_y, -2000.0, 2000.0))

            print(
                "PID_SCALE | "
                f"att=({self.cf_state.attitude.roll:+7.2f},"
                f"{self.cf_state.attitude.pitch:+7.2f}) deg | "
                f"des=({self.cf_setpoint.attitude.roll:+6.2f},"
                f"{self.cf_setpoint.attitude.pitch:+6.2f}) deg | "
                f"PID=({_dbg_r:+8.1f},{_dbg_p:+8.1f},{_dbg_y:+8.1f}) | "
                f"T={_dbg_T:8.1f} | "
                f"YAW actual={_yaw_actual:+7.2f} "
                f"ref={_yaw_ref:+7.2f} "
                f"des={_yaw_des:+7.2f} "
                f"rate_des={_yaw_rate_des:+8.2f} "
                f"gyro_z={_gyro_z:+8.2f} "
                f"raw={_dbg_y:+8.0f} "
                f"applied={_yaw_applied:+7.0f}"
            )

        # Keep the ORIGINAL ERAU inner attitude/rate PID outputs.
        # Roll, pitch, and yaw corrections now reach the motor mixer.

        self.stabilizer_step += 1

        # ====================================================
        # VICON/ERAU MOTOR PATH
        # ====================================================
        #
        # Do NOT apply the old Lighthouse battery compensator here.
        #
        # LQR total force -> control.thrust
        # inner PID       -> roll/pitch/yaw
        # mixer           -> four motor commands
        # cap             -> PWM
        #
        raw = MotorThrust()
        self._power_distributor(self.control, raw)

        pwm = MotorThrust()
        self._power_distribution_cap(raw, pwm)

        self.motor_raw.send_motor_raw(
            int(pwm.motor_1 * self.m1_multiplier),
            int(pwm.motor_2 * self.m2_multiplier),
            int(pwm.motor_3 * self.m3_multiplier),
            int(pwm.motor_4 * self.m4_multiplier),
        )

    def _force_total_to_thrust_cmd(self, total_force_n: float) -> float:
        """
        Convert TOTAL Crazyflie thrust [N] into the thrust-command
        scale used by the existing mixer.

        THRUST_MAX is the maximum thrust per motor.
        """

        total_force_n = float(
            np.clip(
                total_force_n,
                0.0,
                4.0 * THRUST_MAX,
            )
        )

        force_per_motor = total_force_n / 4.0

        thrust_cmd = (
            force_per_motor / THRUST_MAX
        ) * UINT16_MAX

        return float(
            np.clip(
                thrust_cmd,
                0.0,
                UINT16_MAX,
            )
        )

    def _power_distributor(self, control: Control, motor_thrust: MotorThrust):
        # REAL Crazyflie mixer.
        #
        # Do NOT use the Gazebo +/-600 limit for roll/pitch.
        # The real ERAU PID outputs are naturally much larger.
        #
        # For this Z-only diagnostic, keep yaw limited so that
        # the large yaw transient cannot dominate the mixer.

        roll = float(control.roll)
        pitch = float(control.pitch)

        yaw_limit = 2000.0
        yaw = float(
            np.clip(
                control.yaw,
                -yaw_limit,
                yaw_limit,
            )
        )

        r = roll / 2.0
        p = pitch / 2.0

        motor_thrust.motor_1 = control.thrust - r + p + yaw
        motor_thrust.motor_2 = control.thrust - r - p - yaw
        motor_thrust.motor_3 = control.thrust + r - p + yaw
        motor_thrust.motor_4 = control.thrust + r + p - yaw

    def _battery_compensator(
        self, motor_thrust_uncapped: MotorThrust, motor_thrust_bat_comp: MotorThrust
    ):
        supply_voltage = self.cf_vbat
        motor_thrust_bat_comp.motor_1 = self._compensate_voltage(
            i_thrust=motor_thrust_uncapped.motor_1, supply_voltage=supply_voltage
        )
        motor_thrust_bat_comp.motor_2 = self._compensate_voltage(
            i_thrust=motor_thrust_uncapped.motor_2, supply_voltage=supply_voltage
        )
        motor_thrust_bat_comp.motor_3 = self._compensate_voltage(
            i_thrust=motor_thrust_uncapped.motor_3, supply_voltage=supply_voltage
        )
        motor_thrust_bat_comp.motor_4 = self._compensate_voltage(
            i_thrust=motor_thrust_uncapped.motor_4, supply_voltage=supply_voltage
        )

    def _compensate_voltage(self, i_thrust: float, supply_voltage: float) -> float:
        if supply_voltage < 2.0:
            return 0.0

        thrust = (i_thrust / UINT16_MAX) * THRUST_MAX
        if thrust < THRUST_MIN:
            return 0.0

        p = -VMOTOR2THRUST2 / (3.0 * VMOTOR2THRUST3)
        q = p * p * p + (
            VMOTOR2THRUST2 * VMOTOR2THRUST1
            - 3.0 * VMOTOR2THRUST3 * (VMOTOR2THRUST0 - thrust)
        ) / (6.0 * VMOTOR2THRUST3 * VMOTOR2THRUST3)
        r = VMOTOR2THRUST1 / (3.0 * VMOTOR2THRUST3)
        qrp = math.sqrt(q * q + (r - p * p) * (r - p * p) * (r - p * p))

        motor_voltage = self._cbrt(q + qrp) + self._cbrt(q - qrp) + p
        ratio = motor_voltage / supply_voltage
        return UINT16_MAX * ratio

    @staticmethod
    def _cbrt(x: float) -> float:
        return math.copysign(abs(x) ** (1.0 / 3.0), x)

    def _power_distribution_cap(
        self, motor_thrust_bat_comp: MotorThrust, motor_thrust_pwm: MotorThrust
    ):
        thrusts = [
            motor_thrust_bat_comp.motor_1,
            motor_thrust_bat_comp.motor_2,
            motor_thrust_bat_comp.motor_3,
            motor_thrust_bat_comp.motor_4,
        ]
        reduction = max(0.0, max(thrusts) - UINT16_MAX)
        motor_thrust_pwm.motor_1 = max(
            IDLE_THRUST, motor_thrust_bat_comp.motor_1 - reduction
        )
        motor_thrust_pwm.motor_2 = max(
            IDLE_THRUST, motor_thrust_bat_comp.motor_2 - reduction
        )
        motor_thrust_pwm.motor_3 = max(
            IDLE_THRUST, motor_thrust_bat_comp.motor_3 - reduction
        )
        motor_thrust_pwm.motor_4 = max(
            IDLE_THRUST, motor_thrust_bat_comp.motor_4 - reduction
        )

    def _stop_motors(self):
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            self.motor_raw.send_motor_raw(0, 0, 0, 0)
            time.sleep(0.01)

    def _log_position_callback(
        self,
        _timestamp,
        data,
        _logconf,
    ):
        self.lighthouse_x_world = data["stateEstimate.x"]
        self.lighthouse_y_world = data["stateEstimate.y"]
        self.lighthouse_z_world = data["stateEstimate.z"]

        if self.local_xy_initialized:
            # ====================================================
            # ERAU-STYLE GLOBAL XY FRAME
            # ====================================================
            # Use translated Lighthouse/world coordinates directly.
            #
            # NO current-yaw rotation.
            #
            # x_controller = x_lighthouse - x_origin
            # y_controller = y_lighthouse - y_origin
            #
            dx_world = self.lighthouse_x_world - self.ref_x0
            dy_world = self.lighthouse_y_world - self.ref_y0

            # Rotate Lighthouse/world displacement into the
            # FIXED launch-heading frame.
            #
            # This makes the real controller coordinates match
            # the yaw=0 coordinate convention used in Gazebo.
            c = math.cos(self.frame_yaw_rad)
            s = math.sin(self.frame_yaw_rad)

            self.cf_state.position.x = (
                c * dx_world
                + s * dy_world
            )

            self.cf_state.position.y = (
                -s * dx_world
                + c * dy_world
            )
        else:
            self.cf_state.position.x = self.lighthouse_x_world
            self.cf_state.position.y = self.lighthouse_y_world

        # Launch-relative Z, matching launch-relative X/Y.
        if self.local_xy_initialized:
            self.cf_state.position.z = (
                self.lighthouse_z_world - self.ref_z0
            )
        else:
            self.cf_state.position.z = self.lighthouse_z_world

        self._have_position = True

    def _log_state_callback(self, _timestamp, data, _logconf):
        #self.cf_state.position.x = data["stateEstimate.x"]
        #self.cf_state.position.y = data["stateEstimate.y"]
        #self.cf_state.position.z = data["stateEstimate.z"]
        vx_world = data["stateEstimate.vx"]
        vy_world = data["stateEstimate.vy"]

        # ====================================================
        # ERAU-STYLE GLOBAL XY VELOCITY
        # ====================================================
        # stateEstimate.vx/vy are used directly.
        # NO current-yaw rotation.
        #
        if self.local_xy_initialized:

            c = math.cos(self.frame_yaw_rad)
            s = math.sin(self.frame_yaw_rad)

            # Same FIXED launch-heading frame as position.
            self.cf_state.velocity.x = (
                c * vx_world
                + s * vy_world
            )

            self.cf_state.velocity.y = (
                -s * vx_world
                + c * vy_world
            )

        else:
            self.cf_state.velocity.x = vx_world
            self.cf_state.velocity.y = vy_world

        self.cf_state.velocity.z = data["stateEstimate.vz"]
        #self.cf_state.acc.x = data["stateEstimate.ax"]
        #self.cf_state.acc.y = data["stateEstimate.ay"]
        #self.cf_state.acc.z = data["stateEstimate.az"]
        # Real Crazyflie stateEstimate attitude is already in CF convention.
        # Keep attitude in its native/world Crazyflie coordinates.
        self.cf_state.attitude.roll = data["stateEstimate.roll"]
        self.cf_state.attitude.pitch = data["stateEstimate.pitch"]
        self.cf_state.attitude.yaw = data["stateEstimate.yaw"]

        self.cf_sensors.gyro.x = data["gyro.x"]
        self.cf_sensors.gyro.y = data["gyro.y"]
        self.cf_sensors.gyro.z = data["gyro.z"]
        self.cf_vbat = data["pm.vbat"]

        # qx = data["stateEstimate.qx"]
        # qy = data["stateEstimate.qy"]
        # qz = data["stateEstimate.qz"]
        # qw = data["stateEstimate.qw"]
        # q = Quaternion(w=qw, x=qx, y=qy, z=qz)
        # rpy = quat2rpy(q)
        # self.cf_state.attitude_quaternion.x = qx
        # self.cf_state.attitude_quaternion.y = qy
        # self.cf_state.attitude_quaternion.z = qz
        # self.cf_state.attitude_quaternion.w = qw

        # self.cf_state.attitude.roll = np.degrees(rpy.x)
        # self.cf_state.attitude.pitch = np.degrees(rpy.y)
        # self.cf_state.attitude.yaw = np.degrees(rpy.z)

        self._have_state = True
        self._have_sensor = True

    #def _log_sensor_callback(self, _timestamp, data, _logconf):
    #    self.cf_sensors.gyro.x = data["gyro.x"]
    #    self.cf_sensors.gyro.y = data["gyro.y"]
    #    self.cf_sensors.gyro.z = data["gyro.z"]
    #    self.cf_sensors.acc.x = data["acc.x"]
    #    self.cf_sensors.acc.y = data["acc.y"]
    #    self.cf_sensors.acc.z = data["acc.z"]
    #    self.cf_vbat = data["pm.vbat"]
    #    self._have_sensor = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="UTA discrete outer-LQR Lighthouse controller with ERAU inner attitude/rate PID."
    )
    parser.add_argument(
        "--uri",
        default=uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7"),
        help="Crazyflie URI",
    )
    parser.add_argument(
        "--x",
        type=float,
        default=0.0,
        help="Desired X position (m), used when USE_SCRIPT_TRAJECTORY=False",
    )
    parser.add_argument(
        "--y",
        type=float,
        default=0.0,
        help="Desired Y position (m), used when USE_SCRIPT_TRAJECTORY=False",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=0.6,
        help="Desired Z position (m), used when USE_SCRIPT_TRAJECTORY=False",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=0.0,
        help="Desired yaw (deg), used when USE_SCRIPT_TRAJECTORY=False",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=5.0,
        help="Main control duration before landing",
    )
    parser.add_argument(
        "--loop-hz", type=float, default=500.0, help="Host control loop rate"
    )
    parser.add_argument(
        "--takeoff-seconds",
        type=float,
        default=8.0,
        help="Smooth takeoff duration",
    )
    parser.add_argument(
        "--land-z", type=float, default=0.05, help="Landing target z (m)"
    )
    parser.add_argument(
        "--land-seconds", type=float, default=3.0, help="Landing duration"
    )
    parser.add_argument(
        "--no-land",
        action="store_true",
        help="Skip landing phase and stop motors directly after run phase",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    controller = HostPIDPWMPositionController(
        uri=args.uri,
        target_x=args.x,
        target_y=args.y,
        target_z=args.z,
        target_yaw=args.yaw,
        loop_hz=args.loop_hz,
        run_seconds=args.run_seconds,
        takeoff_seconds=args.takeoff_seconds,
        land_z=args.land_z,
        land_seconds=args.land_seconds,
        do_land=not args.no_land,
    )
    try:
        controller.run()
    except KeyboardInterrupt:
        print("Interrupted, stopping motors")
        controller._stop_motors()
        sys.exit(130)
    except Exception as exc:
        msg = str(exc)
        if "Resource busy" in msg or "Couldn't load link driver" in msg:
            print("Fatal error: Crazyradio is busy.")
            print(
                "Close other tools using the radio (for example `cfclient`) and retry."
            )
            print(f"URI: {args.uri}")
            controller._stop_motors()
            sys.exit(1)
        print(f"Fatal error: {exc}")
        controller._stop_motors()
        raise


if __name__ == "__main__":
    main()
