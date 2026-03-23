import threading

from buffers import LatestValue
from camera_config import get_intrinsics, scale_factor, max_fps
from controller import VisualServoController
from rtde_streamer import RTDEStreamer
from vision import VisionProducer


class VisualServoApp:
    def __init__(self, robot_ip: int,
                 recipe_path: str,
                 rate_hz,
                 img_w: int = 640,
                 img_h: int = 480,
                 rtde_port: int = 30004):
        self.stop_event = threading.Event()

        self.latest_det = LatestValue()
        self.latest_cmd = LatestValue()

        # Look up intrinsics for the chosen resolution
        intr    = get_intrinsics(img_w, img_h)
        scale   = scale_factor(img_w, img_h)
        cam_fps = max_fps(img_w, img_h)

        self.vision = VisionProducer(
            self.latest_det, self.stop_event, img_w, img_h,
        )
        self.controller = VisualServoController(
            self.latest_det, self.latest_cmd, self.stop_event,
            img_w, img_h,
            fx=intr.fx, fy=intr.fy,
            cx=intr.cx, cy=intr.cy,
            rate_hz=rate_hz,
            resolution_scale=scale,
            camera_fps=cam_fps,
        )
        self.streamer = RTDEStreamer(
            self.latest_cmd, self.stop_event,
            robot_ip, recipe_path, rate_hz, rtde_port,
        )

    def start(self):
        self.vision.start()
        self.controller.start()
        self.streamer.start()
        print("EVENT_STARTED")

    def stop(self):
        self.stop_event.set()
        print("EVENT STOPPED")