import pyzed.sl as sl
import cv2
import numpy as np

# Echte Keypoints basierend auf Diagnose-Screenshot
WICHTIGE_PUNKTE = {
    11,          # Hals/Brust oben
    2,          # Brust Mitte
    12,         # Linke Schulter
    5,          # Rechte Schulter
    13,         # Linker Ellbogen
    6,          # Rechter Ellbogen
    15,         # Linkes Handgelenk
    8,          # Rechtes Handgelenk
    27,         # Kopf Mitte
}

# Farben
COLOR_HAND = (0, 255, 0)      # Grün  → Hände
COLOR_ARM  = (255, 165, 0)    # Orange → Arme/Schultern
COLOR_BODY = (0, 180, 255)    # Blau  → Körper
COLOR_HEAD = (255, 255, 0)    # Gelb  → Kopf
COLOR_LINE = (200, 200, 200)  # Grau  → Linien

HAND_KP   = {8, 15}
ARM_KP    = {13,  6}
SHOULDER_KP = {5, 12}

# Verbindungen basierend auf DEINEN angepassten Punkten
BODY_CONNECTIONS = [
    (27, 11),       # Kopf → Hals
    (11, 2),        # Hals → Brust
    (11, 12), (11, 5),   # Hals → Linke Schulter & Rechte Schulter
    (12, 13),       # Linke Schulter → Linker Ellbogen
    (5, 6),         # Rechte Schulter → Rechter Ellbogen
    (6, 8),         # Rechter Ellbogen → Rechtes Handgelenk
    (13, 15),       # Linker Ellbogen → Linke Handgelenk

]

def get_color(idx):
    if idx in HAND_KP:     return COLOR_HAND
    if idx in ARM_KP:      return COLOR_ARM
    if idx in SHOULDER_KP: return COLOR_ARM
    if idx == 20:          return COLOR_HEAD
    return COLOR_BODY

def draw_skeleton(frame, kps):
    h, w = frame.shape[:2]

    for (i, j) in BODY_CONNECTIONS:
        if i >= len(kps) or j >= len(kps):
            continue
        x1, y1 = int(kps[i][0]), int(kps[i][1])
        x2, y2 = int(kps[j][0]), int(kps[j][1])
        if 0 < x1 < w and 0 < y1 < h and 0 < x2 < w and 0 < y2 < h:
            cv2.line(frame, (x1, y1), (x2, y2), COLOR_LINE, 2)

    for idx in WICHTIGE_PUNKTE:
        if idx >= len(kps):
            continue
        x, y = int(kps[idx][0]), int(kps[idx][1])
        if 0 < x < w and 0 < y < h:
            color  = get_color(idx)
            radius = 8 if idx in HAND_KP else 5
            cv2.circle(frame, (x, y), radius, color, -1)

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
    print("✅ Positional Tracking aktiviert!")

    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking     = True
    body_params.detection_model     = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_params.body_format         = sl.BODY_FORMAT.BODY_34
    body_params.enable_body_fitting = True

    if zed.enable_body_tracking(body_params) != sl.ERROR_CODE.SUCCESS:
        print("Body Tracking Fehler!")
        zed.close()
        exit(1)
    print("✅ Body Tracking aktiviert!")

    body_runtime = sl.BodyTrackingRuntimeParameters()
    body_runtime.detection_confidence_threshold = 40

    image   = sl.Mat()
    bodies  = sl.Bodies()
    runtime = sl.RuntimeParameters()

    print("🎥 Drücke 'q' zum Beenden")

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
            zed.retrieve_bodies(bodies, body_runtime)

            if bodies.is_new:
                person_count = 0
                for body in bodies.body_list:
                    if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                        person_count += 1
                        draw_skeleton(frame, body.keypoint_2d)

                        # Name über Kopf (Punkt 20)
                        head = body.keypoint_2d[20]
                        hx, hy = int(head[0]), int(head[1])
                        if 0 < hx < frame.shape[1] and 0 < hy < frame.shape[0]:
                            cv2.putText(frame,
                                        f"Person {body.id}",
                                        (hx - 40, hy - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (255, 255, 0), 2)

                cv2.putText(frame,
                            f"Personen: {person_count}",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)

            cv2.imshow("ZED 2i – Rehabilitation Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()