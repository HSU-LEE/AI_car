#!/bin/bash
# 최근 녹화를 맥에서 재분석해 영상 만들기
source "$(dirname "$0")/../_common.sh"

$A replay latest
finish
