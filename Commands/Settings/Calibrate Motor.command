#!/bin/bash
# 모터 최소값 측정: 차를 트랙에 놓고 (앞에 30cm 공간) 더블클릭. 앞으로 몇 cm 움직였다 돌아옴
source "$(dirname "$0")/../_common.sh"

read -r -p "차를 트랙에 놓고 앞을 30cm 비워 두세요. 준비되면 엔터 (취소: Ctrl+C) " _
$A calibrate-motor
finish
