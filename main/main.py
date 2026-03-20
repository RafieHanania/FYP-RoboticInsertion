import time
from app import VisualServoApp
import cv2

RTDE_PORT = 30004
ROBOT_IP = "169.254.149.88"
RECIPE_PATH = "control_loop_configuration.xml" 
RATE_HZ = 100
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def main(robot_ip=ROBOT_IP, recipe_path=RECIPE_PATH, rate_hz=RATE_HZ, img_w=IMAGE_WIDTH, img_h=IMAGE_HEIGHT, rtde_port=RTDE_PORT):
    app = VisualServoApp(robot_ip, 
                         recipe_path, 
                         rate_hz,
                         img_w,
                         img_h,
                         rtde_port
                         )
    print("hello")
    app.start()
    # print(f"Vision thread alive: {app.vision.is_alive()}")  # add this
    # print(f"Vision thread started: {app.vision.ident}")     # add this


    try:
        while True:
            # cv2.imshow MUST run on the main thread
            frame = app.vision.latest_frame.get()

            if frame is not None:
                cv2.imshow("OBB-Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            time.sleep(1.0 / 60.0)

    except KeyboardInterrupt:
        pass
            # time.sleep(1.0)
    finally:
        app.stop()
        cv2.destroyAllWindows()
        time.sleep(0.2)

if __name__ == '__main__':
    main()


