#!/bin/bash
# 설정을 바꿔 한 번만 자율주행 (예: KP=1.8 KH=1.0)
source "$(dirname "$0")/../_common.sh"

v=$(ask "KEY=값 (여러 개는 띄어쓰기): ")
s=$(ask "속도 (엔터 = 기본값): ")
$A go $s --set $v
finish
