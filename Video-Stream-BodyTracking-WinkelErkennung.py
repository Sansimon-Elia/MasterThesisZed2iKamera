import pyzed.sl as sl
import cv2
import numpy as np

# Keypoints (deine bewährten Nummern)
WICHTIGE_PUNKTE = {
    11, 2,          # Hals, Brust
    12, 5,          # Schultern
    13, 6,          # Ellbogen
    15, 8,          # Handgelenke
    27,             # Kopf
}

COLOR_HAND    = (0, 255, 0)
COLOR_ARM     = (255, 165, 0)
COLOR_BODY    = (0, 180, 255)
COLOR_HEAD    = (255, 255, 0)
COLOR_LINE    = (200, 200, 200)
COLOR_GOOD    = (0, 255, 0)      # Grün  → Winkel OK
COLOR_WARNING = (0, 165, 255)    # Orange → Mittlerer Winkel
COLOR_BAD     = (0, 0, 255)      # Rot   → Zu wenig gestreckt

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

def berechne_winkel(p1, p2, p3):
    """
    Berechnet den Winkel an Punkt p2
    p1 = Schulter, p2 = Ellbogen, p3 = Handgelenk
    """
    a = np.array([p1[0], p1[1]])  # Schulter
    b = np.array([p2[0], p2[1]])  # Ellbogen (Mittelpunkt)
    c = np.array([p3[0], p3[1]])  # Handgelenk

    # Vektoren vom Ellbogen aus
    ba = a - b
    bc = c - b

    # Winkel berechnen mit arccos
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    winkel = np.degrees(np.arccos(cosine))
    return round(winkel, 1)

def winkel_farbe(winkel):
    """Farbe basierend auf Streckungsgrad"""
    if winkel >= 150:   return COLOR_GOOD      # Fast gestreckt ✅
    elif winkel >= 90:  return COLOR_WARNING   # Halb gebeugt
    else:               return COLOR_BAD       # Stark gebeugt

def winkel_text(winkel):
    """Feedback-Text für den Patienten"""
    if winkel >= 150:   return "Gut gestreckt!"
    elif winkel >= 90:  return "Weiter strecken..."
    else:               return "Mehr strecken!"

def get_color(idx):
    if idx in HAND_KP:     return COLOR_HAND
    if idx in ARM_KP:      return COLOR_ARM
    if idx in SHOULDER_KP: return COLOR_ARM
    if idx == 27:          return COLOR_HEAD
    return COLOR_BODY

def draw_skeleton(frame, kps):
    h, w = frame.shape[:2]
    for (i, j) in BODY_CONNECTIONS:
        if i >= len(kps) or j >= len(kps): continue
        x1, y1 = int(kps[i][0]), int(kps[i][1])
        x2, y2 = int(kps[j][0]), int(kps[j][1])
        if 0 < x1 < w and 0 < y1 < h and 0 < x2 < w and 0 < y2 < h:
            cv2.line(frame, (x1, y1), (x2, y2), COLOR_LINE, 2)
    for idx in WICHTIGE_PUNKTE:
        if idx >= len(kps): continue
        x, y = int(kps[idx][0]), int(kps[idx][1])
        if 0 < x < w and 0 < y < h:
            color  = get_color(idx)
            radius = 8 if idx in HAND_KP else 5
            cv2.circle(frame, (x, y), radius, color, -1)

def draw_winkel(frame, kps, schulter_idx, ellbogen_idx, handgelenk_idx, seite):
    """Winkel berechnen und auf dem Bild anzeigen"""
    h, w = frame.shape[:2]

    s = kps[schulter_idx]
    e = kps[ellbogen_idx]
    g = kps[handgelenk_idx]

    # Nur wenn alle 3 Punkte sichtbar sind
    if not all(0 < p[0] < w and 0 < p[1] < h for p in [s, e, g]):
        return

    winkel = berechne_winkel(s, e, g)
    farbe  = winkel_farbe(winkel)
    text   = winkel_text(winkel)

    ex, ey = int(e[0]), int(e[1])

    # Kreis am Ellbogen – Farbe zeigt Streckungsgrad
    cv2.circle(frame, (ex, ey), 14, farbe, -1)
    cv2.circle(frame, (ex, ey), 14, (255,255,255), 2)  # weißer Rand

    # Winkel-Zahl direkt am Ellbogen
    cv2.putText(frame,
                f"{winkel}",
                (ex - 20, ey - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2)

    # Feedback-Text links oben (je nach Seite)
    y_pos = 80 if seite == "Links" else 120
    cv2.putText(frame,
                f"{seite}: {winkel} Grad _ {text}",
                (20, y_pos),
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

    print("🎥 Ellbogen-Winkel Messung aktiv – drücke 'q' zum Beenden")

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
                        kps = body.keypoint_2d
                        draw_skeleton(frame, kps)

                        # Winkel für beide Arme messen
                        draw_winkel(frame, kps,
                                    schulter_idx=12,
                                    ellbogen_idx=13,
                                    handgelenk_idx=15,
                                    seite="Links")

                        draw_winkel(frame, kps,
                                    schulter_idx=5,
                                    ellbogen_idx=6,
                                    handgelenk_idx=8,
                                    seite="Rechts")

                        # Name über Kopf
                        head = kps[27]
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

            cv2.imshow("ZED 2i – Ellbogen Reha Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()