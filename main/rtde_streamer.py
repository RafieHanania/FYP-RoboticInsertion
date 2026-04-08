import time 
import threading
from typing import Tuple
import sys
import numpy as np

sys.path.append("..")

from rtde import rtde_config, rtde
from utils import vel_tcp_to_base

cmd6 = Tuple[float, float, float, float, float, float]

class RTDEStreamer(threading.Thread):
    """
    Write the velocity commands into RTDE input register
    and Toggles watchdog
    """

    def __init__(self,
                 cmd_in,
                 stop_event: threading.Event,
                 robot_ip: str,
                 recipe_path: str,
                 rate_hz: float = 100.0,
                 rtde_port: int = 30004
                 ):
        threading.Thread.__init__(self, daemon=True, name="StreamerThread")
        self.cmd_in = cmd_in
        self.stop_event = stop_event
        self.robot_ip = robot_ip
        self.recipe_path = recipe_path
        self.rate_hz = rate_hz
        self.rtde_port = rtde_port
        self._last_tcp_pose: list = None          # populated during run()
        self._motion_start_t: float = None        # time of first non-zero cmd
        self._motion_end_t: float = None          # time of last non-zero cmd

    def run(self):
        dt = 1.0 / self.rate_hz
        
        conf = rtde_config.ConfigFile(self.recipe_path)
        state_names, state_types = conf.get_recipe("state") # unused
        setp_names, setp_types = conf.get_recipe("setp")
        watchdog_names, watchdog_types = conf.get_recipe("watchdog") 

        con = rtde.RTDE(self.robot_ip, self.rtde_port)
        con.connect()

        con.get_controller_version()

        con.send_output_setup(state_names, state_types, frequency=self.rate_hz)
        setp = con.send_input_setup(setp_names, setp_types)
        watchdog = con.send_input_setup(watchdog_names, watchdog_types)

        if setp is None or watchdog is None:
            raise RuntimeError("RTDE input setup failed")
        
        if not con.send_start():
            raise RuntimeError("RTDE start failed.")
        
        try:
            # send initial zeros
            self._write_cmd(setp, (0,0,0,0,0,0))
            con.send(setp)

            heartbeat = 0
            watchdog.input_int_register_0 = 0
            watchdog.input_int_register_1 = heartbeat
            con.send(watchdog)

            while not self.stop_event.is_set():
                state = con.receive()
                if state is None:
                    # Error --> send zero and restart next cycle
                    self._write_cmd(setp, (0,0,0,0,0,0))
                    con.send(setp)
                    time.sleep(dt)
                    continue

                # Extract rotation-vector part of actual_tcp_pose
                tcp_pose = state.actual_TCP_pose
                self._last_tcp_pose = list(tcp_pose)          # store for final print
                rx, ry, rz = tcp_pose[3], tcp_pose[4], tcp_pose[5]

                # Read TCP-frame command from controller
                cmd_tcp: cmd6 = self.cmd_in.get() or (0,0,0,0,0,0)

                # Rotate TCP-frame velocity
                v_tcp = np.array(cmd_tcp, dtype=float)
                v_base = vel_tcp_to_base(v_tcp, rx, ry, rz)

                cmd_base = (
                    v_base[0], v_base[1], v_base[2], 
                    v_base[3], v_base[4], v_base[5]
                ) 

                self._write_cmd(setp, cmd_base)

                # ---- Convergence timer ----
                is_moving = any(abs(v) > 1e-9 for v in cmd_base)
                if is_moving:
                    if self._motion_start_t is None:
                        self._motion_start_t = time.time()
                        print("[RTDEStreamer] Motion started — convergence timer running")
                    self._motion_end_t = time.time()

                # print(dt)

                heartbeat += 1
                watchdog.input_int_register_0 = 1
                watchdog.input_int_register_1 = heartbeat

                con.send(setp)
                # watchdog = 1
                con.send(watchdog)

                time.sleep(dt)

        except Exception as e:
            print("RTDE send failed")
            print("cmd_tcp =", cmd_tcp)
            print("cmd_base =", cmd_base)
            print("heartbeat =", heartbeat)
            print("cmd_valid =", watchdog.input_int_register_0)
            raise

        finally:
            if self._last_tcp_pose is not None:
                x, y, z, rx, ry, rz = self._last_tcp_pose
                print("\n[RTDEStreamer] ── Final TCP pose ──")
                print(f"  Position  : x={x*1000:.2f} mm  y={y*1000:.2f} mm  z={z*1000:.2f} mm")
                print(f"  Rotation  : rx={rx:.4f} rad  ry={ry:.4f} rad  rz={rz:.4f} rad")
            if self._motion_start_t is not None and self._motion_end_t is not None:
                elapsed = self._motion_end_t - self._motion_start_t
                print(f"  Convergence time : {elapsed:.3f} s")
            try:
                con.send_pause()
            except Exception:
                pass
            try:
                con.disconnect()
            except Exception:
                pass
            

    @staticmethod
    def _write_cmd(setp, cmd: cmd6):
        vx, vy, vz, wx, wy, wz = cmd
        setp.input_double_register_0 = float(vx)
        setp.input_double_register_1 = float(vy)
        setp.input_double_register_2 = float(vz)
        setp.input_double_register_3 = float(wx)
        setp.input_double_register_4 = float(wy)
        setp.input_double_register_5 = float(wz)
        


    @staticmethod
    def _setp_to_list(sp):
        """Read input_double_register from the DataObject and turn it into a python list"""
        return [getattr(sp, f"input_double_register_{i}") for i in range(0,6)]

    @staticmethod
    def _list_to_setp(sp, list):
        """Write a 6 element lsit into input_double_register_0..5 into the DataObject """
        for i, v in enumerate(list):
            setattr(sp, f"input_double_register_{i}", v)
        return sp