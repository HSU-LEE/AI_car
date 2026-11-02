#!/bin/bash
# 실행 중인 주행/조작 프로그램을 모두 멈추고 모터를 끈다 (aicar stop, 자율주행 중 Ctrl+C에서 사용)
cd ~/aicar || exit 1

find_pids() {
    # 실제 python 프로세스만 고름. pgrep -f는 이 스크립트를 부른 셸의 명령줄까지 잡아서 쓰지 않음
    ps -eo pid=,comm=,args= |
        awk '$2 ~ /^python/ && /(lane_follow|drive_keys|tune_center|tune_white|tune_light|camera_view|calibrate|calibrate_motor|gap_probe|car)\.py/ {print $1}'
}

pids=$(find_pids)
if [ -z "$pids" ]; then
    echo "실행 중인 주행 없음"
else
    kill -TERM $pids 2>/dev/null
    for _ in $(seq 25); do      # 최대 5초 동안 정상 종료(모터 정지, 녹화 저장)를 기다림
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

# 강제 종료됐을 때도 모터가 확실히 꺼지도록 모터 핀을 다시 잡아 정지시킴
if venv/bin/python car.py center >/dev/null 2>&1; then
    echo "모터 정지, 조향 중앙"
else
    echo "모터 정지 확인 실패. 차 전원을 직접 끄세요."
    exit 1
fi
