#!/bin/bash
# 조향 각도 테스트 (1초 뒤 중앙 복귀)
source "$(dirname "$0")/../_common.sh"

a=$(ask "각도 (-26 ~ 26, 음수=왼쪽): ")
$A steer "$a"
finish
