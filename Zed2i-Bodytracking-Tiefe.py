import pyzed.sl as sl
import cv2
import numpy as np

WICHTIGE_PUNKTE = {
    11, 2,
    12, 5,
    13, 6,
    15, 8,
    27,
}

COLOR_HAND    = (0, 255, 0)
COLOR_ARM     = (255, 165, 0)
COLOR_BODY    = (0, 180, 255)
COLOR_HEAD    = (255, 255, 0)
COLOR_LINE    = (200, 200, 200)
COLOR_GOOD    = (0, 255, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_BAD     = (0, 0, 255)

HAND_KP     = {8, 15}
ARM_KP      = {13, 6}
SHOULDER_KP = {5, 12}

BODY_CONNECTIONS = [
    (27, 11),
    (11, 2),
    (11, 12), (11, 5),
    (12, 13),
    (5, 6),
    (6, 8),
    (13, 15),
]

def berechne_winkel_3d(p1, p2, p3):
    """3D Winkel mit X, Y, Z – präziser als 2D"""
    a = np.array([p1[0], p1[1], p1[2]])
    b = np.array([p2[0], p2[1], p2[2]])
    c = np.array([p3[0], p3[1], p3[2]])

    # Ungültige Punkte abfangen
    if np.allclose(a, 0) or np.allclose(b, 0) or np.allclose(c, 0):
        return None

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return round(np.degrees(np.arccos(cosine)), 1)

def berechne_abstand(p):
    """Abstand der Person zur Kamera in Metern (Z-Koordinate)"""
    if p[2] > 0:
        return round(p[2], 2)
    return None

def winkel_farbe(winkel):
    if winkel >= 150:  return COLOR_GOOD
    elif winkel >= 90: return COLOR_WARNING
    else:              return COLOR_BAD

def winkel_text(winkel):
    if winkel >= 150:  return "Gut gestreckt!"
    elif winkel >= 90: return "Weiter strecken..."
    else:              return "Mehr strecken!"

def get_color(idx):
    if idx in HAND_KP:     return COLOR_HAND
    if idx in ARM_KP:      return COLOR_ARM
    if idx in SHOULDER_KP: return COLOR_ARM
    if idx == 27:          return COLOR_HEAD
    return COLOR_BODY

def draw_skeleton(frame, kps_2d):
    h, w = frame.shape[:2]
    for (i, j) in BODY_CONNECTIONS:
        if i >= len(kps_2d) or j >= len(kps_2d): continue
        x1, y1 = int(kps_2d[i][0]), int(kps_2d[i][1])
        x2, y2 = int(kps_2d[j][0]), int(kps_2d[j][1])
        if 0 < x1 < w and 0 < y1 < h and 0 < x2 < w and 0 < y2 < h:
            cv2.line(frame, (x1, y1), (x2, y2), COLOR_LINE, 2)
    for idx in WICHTIGE_PUNKTE:
        if idx >= len(kps_2d): continue
        x, y = int(kps_2d[idx][0]), int(kps_2d[idx][1])
        if 0 < x < w and 0 < y < h:
            color  = get_color(idx)
            radius = 8 if idx in HAND_KP else 5
            cv2.circle(frame, (x, y), radius, color, -1)

def draw_winkel(frame, kps_2d, kps_3d, schulter_idx, ellbogen_idx, handgelenk_idx, seite):
    h, w = frame.shape[:2]

    # 2D Koordinaten für die Anzeige
    e2d = kps_2d[ellbogen_idx]
    ex, ey = int(e2d[0]), int(e2d[1])

    # Sichtbarkeitscheck
    if not (0 < ex < w and 0 < ey < h):
        return

    # 3D Koordinaten für die Berechnung
    s3d = kps_3d[schulter_idx]
    e3d = kps_3d[ellbogen_idx]
    g3d = kps_3d[handgelenk_idx]

    winkel = berechne_winkel_3d(s3d, e3d, g3d)

    # Wenn Punkte nicht sichtbar
    if winkel is None:
        cv2.putText(frame,
                    f"{seite}: Nicht sichtbar",
                    (20, 80 if seite == "Links" else 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (128, 128, 128), 2)
        return

    farbe = winkel_farbe(winkel)
    text  = winkel_text(winkel)

    # Kreis am Ellbogen
    cv2.circle(frame, (ex, ey), 14, farbe, -1)
    cv2.circle(frame, (ex, ey), 14, (255, 255, 255), 2)

    # Winkel am Ellbogen
    cv2.putText(frame,
                f"{winkel}",
                (ex - 20, ey - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2)

    # Feedback oben links
    y_pos = 80 if seite == "Links" else 120
    cv2.putText(frame,
                f"{seite}: {winkel} Grad _ {text}",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, farbe, 2)

def draw_abstand(frame, kps_3d):
    """Abstand zur Kamera oben rechts anzeigen"""
    # Brust (Punkt 2) als Referenzpunkt
    abstand = berechne_abstand(kps_3d[2])
    if abstand is None:
        return

    h, w = frame.shape[:2]

    # Farbe je nach Abstand
    if abstand < 1.0:
        farbe = (0, 0, 255)      # Rot – zu nah
        hinweis = "Zu nah!"
    elif abstand > 3.5:
        farbe = (0, 165, 255)    # Orange – zu weit
        hinweis = "Zu weit!"
    else:
        farbe = (0, 255, 0)      # Grün – optimaler Abstand
        hinweis = "Abstand OK"

    cv2.putText(frame,
                f"Abstand: {abstand}m _ {hinweis}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, farbe, 2)

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
    print("Positional Tracking aktiviert!")

    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking     = True
    body_params.detection_model     = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_params.body_format         = sl.BODY_FORMAT.BODY_34
    body_params.enable_body_fitting = True

    if zed.enable_body_tracking(body_params) != sl.ERROR_CODE.SUCCESS:
        print("Body Tracking Fehler!")
        zed.close()
        exit(1)
    print("Body Tracking aktiviert!")

    body_runtime = sl.BodyTrackingRuntimeParameters()
    body_runtime.detection_confidence_threshold = 40

    image   = sl.Mat()
    bodies  = sl.Bodies()
    runtime = sl.RuntimeParameters()

    print("Druecke q zum Beenden")

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
                        kps_2d = body.keypoint_2d  # Für Anzeige
                        kps_3d = body.keypoint     # Für Berechnung (NEU!)

                        draw_skeleton(frame, kps_2d)

                        # Abstand anzeigen
                        draw_abstand(frame, kps_3d)

                        # 3D Winkel für beide Arme
                        draw_winkel(frame, kps_2d, kps_3d,
                                    schulter_idx=12,
                                    ellbogen_idx=13,
                                    handgelenk_idx=15,
                                    seite="Links")

                        draw_winkel(frame, kps_2d, kps_3d,
                                    schulter_idx=5,
                                    ellbogen_idx=6,
                                    handgelenk_idx=8,
                                    seite="Rechts")

                        # Name über Kopf
                        head = kps_2d[27]
                        hx, hy = int(head[0]), int(head[1])
                        if 0 < hx < frame.shape[1]:
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

            cv2.imshow("ZED 2i _ 3D Reha Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()