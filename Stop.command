#!/bin/bash
# Stop at once: halt the running drive and center the steering.
cd "$(dirname "$0")" || exit 1

./aicar stop
echo
read -r -p "엔터를 누르면 창을 닫습니다..." _
