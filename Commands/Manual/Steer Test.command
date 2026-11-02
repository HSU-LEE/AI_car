#!/bin/bash
# Test a steer angle, then return to center after one second.
source "$(dirname "$0")/../_common.sh"

a=$(ask "각도 (-26 ~ 26, 음수=왼쪽): ")
$A steer "$a"
finish
