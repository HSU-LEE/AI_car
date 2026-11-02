#!/bin/bash
# 현재 설정 보기 (보정값, 흰 선, 주행값)
source "$(dirname "$0")/../_common.sh"

$A config
finish
