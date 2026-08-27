#!/usr/bin/env python3

"""
Dual Crazyflie motor sender.

Controller still runs at 200 Hz.
Normal motor CRTP transmission is every second call = 100 Hz.

ZERO motor commands bypass the divider and are sent immediately.
"""

import struct

from cflib.crtp.crtpstack import CRTPPacket


MOTOR_RAW_PORT = 0x09


class MotorRaw:

    SETPOINT_CH = 0

    def __init__(
        self,
        crazyflie=None,
        test=False,
        using_json_log=False,
    ):
        self._cf = crazyflie
        self.test = test

        # Controller calls send_motor_raw at ~200 Hz.
        # Send every second normal command.
        self._tx_counter = 0


    def _send_packet(
        self,
        m1,
        m2,
        m3,
        m4,
    ):

        pk = CRTPPacket()

        pk.port = MOTOR_RAW_PORT
        pk.channel = self.SETPOINT_CH

        pk.data = struct.pack(
            "<HHHH",
            int(m1),
            int(m2),
            int(m3),
            int(m4),
        )

        self._cf.send_packet(pk)


    def send_motor_raw(
        self,
        m1,
        m2,
        m3,
        m4,
    ):

        if self.test:
            return

        m1 = int(m1)
        m2 = int(m2)
        m3 = int(m3)
        m4 = int(m4)

        # ==============================================
        # SAFETY:
        # motor-off is ALWAYS transmitted immediately.
        # ==============================================
        if (
            m1 == 0
            and m2 == 0
            and m3 == 0
            and m4 == 0
        ):
            self._send_packet(
                0, 0, 0, 0
            )

            # Start clean after motor-off.
            self._tx_counter = 0
            return

        # ==============================================
        # 200 Hz controller -> 100 Hz radio TX
        # ==============================================
        self._tx_counter += 1

        if self._tx_counter < 2:
            return

        self._tx_counter = 0

        self._send_packet(
            m1,
            m2,
            m3,
            m4,
        )
