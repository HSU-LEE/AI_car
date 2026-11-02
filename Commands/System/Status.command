#!/bin/bash
# 파이 상태 (주소, 와이파이, 온도, 디스크, 실행 중인 프로그램)
source "$(dirname "$0")/../_common.sh"

$A status
finish
