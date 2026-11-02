#!/bin/bash
# 즉시 정지: 실행 중인 주행을 멈추고 조향을 중앙으로
cd "$(dirname "$0")" || exit 1

./aicar stop
echo
read -r -p "엔터를 누르면 창을 닫습니다..." _
