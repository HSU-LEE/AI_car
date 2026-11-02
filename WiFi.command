#!/bin/bash
# 파이 USB 랜카드를 새 와이파이에 등록
source "$(dirname "$0")/Commands/_common.sh"

$A wifi
echo
$A wifi-add
finish
