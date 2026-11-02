#!/bin/bash
# 직진 주행 (속도, 시간 지정)
source "$(dirname "$0")/../_common.sh"

s=$(ask "속도 (-1 ~ 1, 음수=후진): ")
t=$(ask "시간(초, 엔터 = 1): ")
$A drive "$s" $t
finish
