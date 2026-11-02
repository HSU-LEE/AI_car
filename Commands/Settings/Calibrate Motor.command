#!/bin/bash
# Measure the motor floor. Place the car on the track with 30 cm clear ahead. It creeps forward and back.
source "$(dirname "$0")/../_common.sh"

read -r -p "차를 트랙에 놓고 앞을 30cm 비워 두세요. 준비되면 엔터 (취소: Ctrl+C) " _
$A calibrate-motor
finish
