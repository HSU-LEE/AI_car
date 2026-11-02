#!/bin/bash
# 저장된 와이파이 삭제
source "$(dirname "$0")/../_common.sh"

$A wifi
echo
w=$(ask "삭제할 와이파이 이름: ")
$A wifi-del "$w"
finish
