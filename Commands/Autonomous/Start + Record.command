#!/bin/bash
# 자율주행 + 녹화
source "$(dirname "$0")/../_common.sh"

s=$(ask "속도 (엔터 = 기본값): ")
$A go $s --record
finish
