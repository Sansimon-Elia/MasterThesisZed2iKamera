from flask import Flask, request
import logging
import time

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

start_time = None
last_5s_marker = 0


@app.route('/sensorlog', methods=['POST'])
def sensorlog():
    global start_time, last_5s_marker

    data = request.json
    if data is None:
        return "No data", 400

    # Timer beim ersten empfangenen Paket starten
    if start_time is None:
        start_time = time.time()
        last_5s_marker = 0
        print("\n🟢 Timer gestartet. Erste Bewegung halten...\n")

    elapsed = time.time() - start_time

    # Alle 5 Sekunden Hinweis ausgeben
    current_marker = int(elapsed // 5)

    if current_marker > last_5s_marker:
        last_5s_marker = current_marker
        print("\n" + "=" * 70)
        print(f"✅ {current_marker * 5} Sekunden gelaufen — jetzt Bewegung ändern!")
        print("=" * 70 + "\n")

    # Alle Parameter auslesen
    gx    = float(data.get("gravityX", 0))
    gy    = float(data.get("gravityY", 0))
    gz    = float(data.get("gravityZ", 0))
    pitch = float(data.get("motionPitch", 0))
    roll  = float(data.get("motionRoll",  0))
    yaw   = float(data.get("motionYaw",   0))
    ax    = float(data.get("motionUserAccelerationX", 0))
    ay    = float(data.get("motionUserAccelerationY", 0))
    az    = float(data.get("motionUserAccelerationZ", 0))

    print(
        f"t={elapsed:05.2f}s | "
        f"gX={gx:+.2f} gY={gy:+.2f} gZ={gz:+.2f} | "
        f"pitch={pitch:+.2f} roll={roll:+.2f} yaw={yaw:+.2f} | "
        f"aX={ax:+.2f} aY={ay:+.2f} aZ={az:+.2f}"
    )

    return "OK", 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=56671, debug=False)