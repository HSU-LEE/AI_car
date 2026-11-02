#!/bin/bash
# course.py N번 동작부터 자율주행
source "$(dirname "$0")/../_common.sh"

k=$(ask "몇 번 동작부터: ")
s=$(ask "속도 (엔터 = 기본값): ")
$A go $s --start "$k"
finish
