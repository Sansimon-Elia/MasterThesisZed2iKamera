#!/usr/bin/env python
# encoding: utf-8

"""
Kleiner Test fuer den feelSpace naviBelt:
Laesst 5 Sekunden lang die rechte Seite vibrieren, dann 5 Sekunden die linke Seite.

Vorher installieren:
    pip install pybelt
"""

import time

from pybelt.examples_utility import belt_controller_log_to_stdout, interactive_belt_connection
from pybelt.belt_controller import (
    BeltController,
    BeltControllerDelegate,
    BeltConnectionState,
    BeltMode,
    BeltOrientationType,
    BeltVibrationPattern,
)

VIBRATION_DURATION_S = 5

# Winkel-Konvention der ANGLE-Orientierung: 0 Grad = vorne, im Uhrzeigersinn positiv.
ANGLE_RIGHT = 90
ANGLE_LEFT = 270


class Delegate(BeltControllerDelegate):
    # Keine Callbacks noetig fuer diesen einfachen Test
    pass


def vibrate_side(belt_controller: BeltController, angle: int, duration_s: float):
    belt_controller.send_vibration_command(
        channel_index=1,
        pattern=BeltVibrationPattern.CONTINUOUS,
        intensity=None,
        orientation_type=BeltOrientationType.ANGLE,
        orientation=angle,
        pattern_iterations=None,
        pattern_period=500,
        pattern_start_time=0,
        exclusive_channel=True,
        clear_other_channels=True,
    )
    time.sleep(duration_s)
    belt_controller.stop_vibration(channel_index=1)


def main():
    belt_controller_log_to_stdout()

    delegate = Delegate()
    belt_controller = BeltController(delegate)

    # Interaktiver Verbindungsaufbau (Bluetooth-Scan / USB-Auswahl per Konsole)
    interactive_belt_connection(belt_controller)

    if belt_controller.get_connection_state() != BeltConnectionState.CONNECTED:
        print("Verbindung fehlgeschlagen.")
        return 1

    belt_controller.set_belt_mode(BeltMode.APP_MODE)

    print("Rechte Seite vibriert...")
    vibrate_side(belt_controller, ANGLE_RIGHT, VIBRATION_DURATION_S)

    print("Linke Seite vibriert...")
    vibrate_side(belt_controller, ANGLE_LEFT, VIBRATION_DURATION_S)

    print("Fertig. Trenne Verbindung.")
    belt_controller.disconnect_belt()

    return 0


if __name__ == "__main__":
    main()
