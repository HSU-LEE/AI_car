#!/bin/bash
# Venue calibration. Place the car in the center of a straight lane, pointing along it.
# Measures track color, focus, exposure, lane width, horizon, and camera center, then saves them on the Pi.
source "$(dirname "$0")/../_common.sh"

echo "차를 직선 차로 가운데에, 차로 방향으로 똑바로 놓으세요. 측정하는 동안(약 15초) 차와 카메라 앞을 가리지 마세요."
read -r -p "준비되면 엔터 (취소: Ctrl+C) " _
$A calibrate
finish
