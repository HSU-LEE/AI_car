#!/bin/bash
# 신호등 인식 기준 맞추기 (브라우저 보며 w/s 채도, e/d 밝기, r/f 크기, Enter 저장)
source "$(dirname "$0")/../_common.sh"

$A tune-light
finish
