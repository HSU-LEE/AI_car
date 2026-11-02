#!/bin/bash
# 자율주행 시작 (녹화 안 함, Ctrl+C로 정지)
source "$(dirname "$0")/../_common.sh"

s=$(ask "속도 (엔터 = 기본값): ")
$A go $s
finish
