import pyzed.sl as sl
import cv2
import numpy as np

COLOR_HAND = (0, 255, 0)
COLOR_ARM  = (255, 165, 0)
COLOR_BODY = (0, 120, 255)
COLOR_LINE = (200, 200, 200)

BODY_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (3, 4), (4, 5), (5, 6), (6, 7),
    (3, 8), (8, 9), (9, 10), (10, 11),
    (0, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17),
    (7, 18), (18, 19), (19, 20),
    (7, 21), (21, 22), (22, 23),
    (11, 24), (24, 25), (25, 26),
    (11, 27), (27, 28), (28, 29),
]

HAND_KEYPOINTS = {7, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}

def get_color(kp_index):
    if kp_index in HAND_KEYPOINTS:
        return COLOR_HAND
    elif kp_index in {5, 6, 7, 9, 10, 11}:
        return COLOR_ARM
    else:
        return COLOR_BODY

def draw_skeleton(frame, keypoints_2d):
    h, w = frame.shape[:2]
    for (i, j) in BODY_CONNECTIONS:
        if i < len(keypoints_2d) and j < len(keypoints_2d):
            kp1 = keypoints_2d[i]
            kp2 = keypoints_2d[j]
            if (kp1[0] > 0 and kp1[1] > 0 and
                kp2[0] > 0 and kp2[1] > 0 and
                kp1[0] < w and kp1[1] < h and
                kp2[0] < w and kp2[1] < h):
                cv2.line(frame,
                         (int(kp1[0]), int(kp1[1])),
                         (int(kp2[0]), int(kp2[1])),
                         COLOR_LINE, 1)
    for idx, kp in enumerate(keypoints_2d):
        x, y = int(kp[0]), int(kp[1])
        if 0 < x < w and 0 < y < h:
            color = get_color(idx)
            radius = 6 if idx in HAND_KEYPOINTS else 4
            cv2.circle(frame, (x, y), radius, color, -1)

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # ULTRA war deprecated

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Kamera Fehler: {status}")
        exit(1)

    # ── 1. Positional Tracking ZUERST aktivieren ──────────────
    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.enable_area_memory = True
    status = zed.enable_positional_tracking(tracking_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Positional Tracking Fehler: {status}")
        zed.close()
        exit(1)
    print("✅ Positional Tracking aktiviert!")

    # ── 2. Body Tracking danach aktivieren ───────────────────
    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking       = True
    body_params.detection_model       = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_params.body_format           = sl.BODY_FORMAT.BODY_34
    body_params.enable_body_fitting   = True

    status = zed.enable_body_tracking(body_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Body Tracking Fehler: {status}")
        zed.close()
        exit(1)
    print("✅ Body Tracking aktiviert!")

    body_runtime = sl.BodyTrackingRuntimeParameters()
    body_runtime.detection_confidence_threshold = 40

    image   = sl.Mat()
    bodies  = sl.Bodies()
    runtime = sl.RuntimeParameters()

    print("🎥 Stream läuft – drücke 'q' zum Beenden")

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)

            zed.retrieve_bodies(bodies, body_runtime)

            person_count = 0
            for body in bodies.body_list:
                if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                    person_count += 1
                    draw_skeleton(frame, body.keypoint_2d)

                    head = body.keypoint_2d[0]
                    if head[0] > 0 and head[1] > 0:
                        cv2.putText(frame,
                                    f"Person {body.id}",
                                    (int(head[0]) - 30, int(head[1]) - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 0), 2)

            cv2.putText(frame,
                        f"Personen erkannt: {person_count}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

            cv2.imshow("ZED 2i – Body & Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()