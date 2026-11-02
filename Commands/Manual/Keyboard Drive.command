#!/bin/bash
# 키보드로 운전 (w/s 속도, a/d 조향, 스페이스 정지, q 종료)
source "$(dirname "$0")/../_common.sh"

s=$(ask "최대 속도 (엔터 = 0.6): ")
$A drive-keys $s
finish
