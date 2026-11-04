#!/bin/bash
# Stop every running drive or manual-control program and turn the motor off.
# Used by `aicar stop` and by Ctrl+C during autonomous driving.
cd ~/aicar || exit 1

find_pids() {
    # Real python processes only. `pgrep -f` also matches the shell that launched this script.
    ps -eo pid=,comm=,args= |
        awk '$2 ~ /^python/ && /(lane_follow|drive_keys|tune_center|tune_white|tune_light|tune_person|camera_view|calibrate|gap_probe|car)\.py/ {print $1}'
}

pids=$(find_pids)
if [ -z "$pids" ]; then
    echo "실행 중인 주행 없음"
else
    kill -TERM $pids 2>/dev/null
    for _ in $(seq 25); do      # Wait up to 5s for a clean exit (motor off, recording saved).
        [ -z "$(find_pids)" ] && break
        sleep 0.2
    done
    left=$(find_pids)
    if [ -n "$left" ]; then
        kill -KILL $left 2>/dev/null
        sleep 0.3
        echo "응답이 없어 강제 종료"
    fi
    echo "주행 프로그램 종료"
fi

# If the process was killed, grab the motor pins again and stop them.
if venv/bin/python car.py center >/dev/null 2>&1; then
    echo "모터 정지, 조향 중앙"
else
    echo "모터 정지 확인 실패. 차 전원을 직접 끄세요."
    exit 1
fi
