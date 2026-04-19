from flask import Flask, request
import json

app = Flask(__name__)

@app.route('/sensorlog', methods=['POST'])
def sensorlog():
    data = request.json

    print("\n--- Neue Sensordaten empfangen ---")

    if data is not None:
        if "motionUserAccelerationX" in data:
            print("Accel X:", data["motionUserAccelerationX"])

        if "motionUserAccelerationY" in data:
            print("Accel Y:", data["motionUserAccelerationY"])

        if "motionUserAccelerationZ" in data:
            print("Accel Z:", data["motionUserAccelerationZ"])

        if "motionYaw" in data:
            print("Yaw:", data["motionYaw"])

        if "motionPitch" in data:
            print("Pitch:", data["motionPitch"])

        if "motionRoll" in data:
            print("Roll:", data["motionRoll"])

        if "rotationRateZ" in data:
            print("Rotation Z:", data["rotationRateZ"])

        if "heartRate" in data:
            print("Heart Rate:", data["heartRate"])

        if "averageHeartRate" in data:
            print("Average Heart Rate:", data["averageHeartRate"])

        print("Raw Data:", json.dumps(data))

    return "OK", 200


if __name__ == "__main__":
    print("Flask Server gestartet...")
    print("Warte auf Sensordaten von Apple Watch / iPhone...\n")

    app.run(host="0.0.0.0", port=56671, debug=False)