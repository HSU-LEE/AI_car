#!/bin/bash
# 서보 중앙을 숫자만큼 옮겨 저장
source "$(dirname "$0")/../_common.sh"

d=$(ask "옮길 각도 (예: -0.5): ")
$A trim "$d"
finish
