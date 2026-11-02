#!/bin/bash
# 최근 녹화를 맥 runs/ 폴더로 받기
source "$(dirname "$0")/../_common.sh"

$A fetch latest
finish
