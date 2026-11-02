#!/bin/bash
# Drive straight for a chosen speed and time.
source "$(dirname "$0")/../_common.sh"

s=$(ask "속도 (-1 ~ 1, 음수=후진): ")
t=$(ask "시간(초, 엔터 = 1): ")
$A drive "$s" $t
finish
