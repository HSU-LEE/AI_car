#!/bin/bash
# 파이의 오래된 녹화 정리 (삭제 전에 확인)
source "$(dirname "$0")/../_common.sh"

k=$(ask "최근 몇 개를 남길까요 (엔터 = 5): ")
$A clean-runs $k
finish
