#!/bin/bash
# Keyboard drive: w/s speed, a/d steer, space stops, q quits.
source "$(dirname "$0")/../_common.sh"

s=$(ask "최대 속도 (엔터 = 0.6): ")
$A drive-keys $s
finish
