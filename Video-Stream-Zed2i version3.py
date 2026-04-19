import pyzed.sl as sl
import cv2
import numpy as np

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Kamera Fehler!")
        exit(1)

    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.enable_area_memory = True
    zed.enable_positional_tracking(tracking_params)

    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking     = True
    body_params.detection_model     = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_params.body_format         = sl.BODY_FORMAT.BODY_34
    body_params.enable_body_fitting = True
    zed.enable_body_tracking(body_params)

    body_runtime = sl.BodyTrackingRuntimeParameters()
    body_runtime.detection_confidence_threshold = 40

    image   = sl.Mat()
    bodies  = sl.Bodies()
    runtime = sl.RuntimeParameters()

    print("🎥 Diagnose-Modus – alle Punkte mit Nummern")
    print("Drücke 'q' zum Beenden")

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
            zed.retrieve_bodies(bodies, body_runtime)

            if bodies.is_new:
                for body in bodies.body_list:
                    if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                        kps = body.keypoint_2d

                        # JEDEN Punkt mit seiner Nummer anzeigen
                        for idx, kp in enumerate(kps):
                            x, y = int(kp[0]), int(kp[1])
                            h, w = frame.shape[:2]
                            if 0 < x < w and 0 < y < h:
                                # Punkt zeichnen
                                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                                # Nummer daneben schreiben
                                cv2.putText(frame,
                                            str(idx),
                                            (x + 5, y - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.45, (255, 255, 255), 1)

            cv2.imshow("DIAGNOSE – Keypoint Nummern", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()