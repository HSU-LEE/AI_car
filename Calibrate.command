#!/bin/bash
# Venue calibration. Place the car in the center of a straight lane, pointing along it.
# Measures track color, focus, exposure, lane width, horizon, and camera center, then saves them on the Pi.
source "$(dirname "$0")/Commands/_common.sh"

echo "Place the car in the middle of a straight lane, facing along it. Keep the car and the camera view clear while it measures (about 15 seconds)."
read -r -p "Press Enter when ready (Ctrl+C to cancel) " _
$A calibrate
finish
