#!/bin/bash
# Double-click menu. Every action is the `aicar` command in this folder.
cd "$(dirname "$0")" || exit 1
A=./aicar
STATE=.aicar

B=$'\033[1m'; D=$'\033[2m'; G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; C=$'\033[36m'; N=$'\033[0m'

ask() { read -r -p "  $1" REPLY || return 1; echo "$REPLY"; }   # Fail when input ends because the window closed.

# --- Status line ---
pi_status() {
    local h state
    h=$(cat "$STATE/host" 2>/dev/null)
    if [ -z "$h" ] || ! nc -z -G 1 "$h" 22 2>/dev/null; then
        echo "  ${R}●${N} 파이 연결 안 됨 ${D}(명령을 실행하면 자동으로 다시 찾음)${N}"
        return
    fi
    state=$(ssh -n -o BatchMode=yes -o ConnectTimeout=2 -o ControlMaster=auto \
                -o "ControlPath=$STATE/cm-%r@%h" -o ControlPersist=120 "pi@$h" \
                'p=$(cat ~/aicar/drive.pid 2>/dev/null) && ps -p "$p" -o args= | grep -q lane_follow && echo drive' 2>/dev/null)
    if [ "$state" = drive ]; then
        echo "  ${G}●${N} 파이 연결됨 ${D}$h${N}   ${Y}▶ 자율주행 중${N} ${D}(s: 정지)${N}"
    else
        echo "  ${G}●${N} 파이 연결됨 ${D}$h${N}   ${D}○ 대기 중${N}"
    fi
}

header() {
    clear
    echo
    echo "  ${B}AICAR${N}${1:+ ${D}›${N} ${B}$1${N}}"
    echo "  ${D}──────────────────────────────────────────${N}"
    pi_status
    echo
}

item() { printf "   ${C}%-3s${N} %s\n" "$1" "$2"; }
section() { echo "  ${D}$1${N}"; }

pause() { echo; read -r -p "  ${D}엔터를 누르면 돌아갑니다...${N}" _; }

# --- Submenus ---
# Form: "key|label|command". Key s stops from anywhere.
submenu() {
    local title="$1"; shift
    local entries=("$@") e key label cmd n
    while true; do
        header "$title"
        for e in "${entries[@]}"; do
            if [ "$e" = "-" ]; then echo; continue; fi
            key=${e%%|*}; label=${e#*|}; label=${label%%|*}
            item "$key" "$label"
        done
        echo
        item s "${R}■ 정지${N}"
        item b "뒤로"
        echo
        n=$(ask "선택 › ") || exit 0
        case "$n" in
            b|B|"") return ;;
            s|S) echo; $A stop; pause; continue ;;
        esac
        cmd=""
        for e in "${entries[@]}"; do
            [ "${e%%|*}" = "$n" ] && cmd=${e##*|}
        done
        if [ -z "$cmd" ]; then echo "  ${R}없는 번호입니다${N}"; sleep 1; continue; fi
        echo
        eval "$cmd"
        pause
    done
}

menu_manual() {
    submenu "수동 운전" \
        '1|키보드로 운전 (w/s 속도, a/d 조향, q 종료)|s=$(ask "최대 속도 (엔터 = 0.6): "); $A drive-keys $s' \
        '2|직진 주행 (속도, 시간 지정)|s=$(ask "속도 (-1 ~ 1, 음수=후진): "); t=$(ask "시간(초, 엔터 = 1): "); $A drive "$s" $t' \
        '-' \
        '3|조향 중앙|$A center' \
        '4|조향 각도 테스트|a=$(ask "각도 (-26 ~ 26, 음수=왼쪽): "); $A steer "$a"' \
        '5|조향 + 전진/후진 한 번에 테스트|$A test'
}

menu_auto() {
    submenu "자율주행" \
        '1|자율주행 시작|$A go' \
        '2|자율주행 + 녹화|$A go --record' \
        '3|코스 N번 동작부터 시작|k=$(ask "몇 번 동작부터: "); $A go --start "$k"' \
        '4|설정 바꿔서 한 번만 주행 (예: KP=1.8)|v=$(ask "KEY=값 (여러 개는 띄어쓰기): "); $A go --set $v' \
        '-' \
        '5|인식만 보기 (모터 꺼짐, 손으로 밀며 확인)|$A view' \
        '6|주행 화면 다시 보기 (연결 끊긴 뒤)|$A log' \
        '-' \
        '7|코스 편집 (course.py)|$A course'
}

menu_camera() {
    submenu "카메라 / 녹화" \
        '1|카메라 + 인식 화면 보기 (모터 안 움직임)|$A camera' \
        '2|사진 찍기|$A snap' \
        '-' \
        '3|녹화 목록|$A runs' \
        '4|최근 녹화 맥으로 받기|$A fetch latest' \
        '5|최근 녹화 재분석 (영상 만들기)|$A replay latest' \
        '6|오래된 녹화 정리|k=$(ask "최근 몇 개를 남길까요 (엔터 = 5): "); $A clean-runs $k'
}

menu_settings() {
    submenu "설정 / 보정" \
        '1|현재 설정 보기|$A config' \
        '0|자동 캘리브레이션 (차를 직선 차로 가운데에 놓고)|echo "  차를 직선 차로 가운데에, 차로 방향으로 똑바로 놓으세요. 측정하는 동안(약 15초) 차와 카메라 앞을 가리지 마세요."; y=$(ask "준비되면 엔터 (취소: n) "); [ "$y" != n ] && $A calibrate' \
        'f|초점만 다시 맞추기 (차를 트랙 위에)|$A focus' \
        '-' \
        '2|서보 중앙 맞추기 (키보드)|$A tune-center' \
        '3|서보 중앙 숫자로 옮기기 (trim)|d=$(ask "옮길 각도 (예: -0.5): "); $A trim "$d"' \
        '4|트랙 색 (검은 바닥/흰 선 ↔ 흰 바닥/검은 선)|$A track' \
        '5|선 인식 맞추기 (브라우저 보며)|$A tune-white' \
        '6|신호등 인식 맞추기 (브라우저 보며)|$A tune-light' \
        'p|레고 사람 등록 / 인식 확인 (브라우저 보며)|$A person' \
        '-' \
        '7|주행 설정 편집 (lane_follow.py)|$A set'
}

menu_system() {
    submenu "파이 관리" \
        '1|상태 자세히 보기|$A status' \
        '2|코드 업데이트 (맥 → 파이)|$A push' \
        '3|파이 코드 가져오기 (파이 → 맥)|$A pull-code' \
        '4|파이 터미널 열기|$A ssh' \
        '5|파이썬 환경 점검|$A check' \
        '-' \
        '6|와이파이 상태 / 주변 와이파이|$A wifi' \
        '7|새 와이파이 등록|$A wifi; echo; $A wifi-add' \
        '8|저장된 와이파이 삭제|$A wifi; echo; w=$(ask "삭제할 와이파이 이름: "); $A wifi-del "$w"' \
        '-' \
        '9|파이 재부팅|$A reboot' \
        '0|파이 끄기 (전원 뽑기 전에)|y=$(ask "파이를 끌까요? [y/N] "); [ "$y" = y ] && $A shutdown'
}

# --- Main screen ---
while true; do
    header
    section "빠른 실행"
    item g "자율주행 시작"
    item s "${R}■ 정지${N}"
    item c "카메라 + 인식 화면"
    item u "코드 업데이트"
    echo
    section "메뉴"
    item 1 "수동 운전"
    item 2 "자율주행"
    item 3 "카메라 / 녹화"
    item 4 "설정 / 보정"
    item 5 "파이 관리  ${D}연결 · 와이파이 · 전원${N}"
    echo
    item r "${D}상태 새로고침${N}"
    item q "${D}종료${N}"
    echo
    n=$(ask "선택 › ") || exit 0
    case "$n" in
        g|G) echo; $A go; pause ;;
        s|S) echo; $A stop; pause ;;
        c|C) echo; $A camera; pause ;;
        u|U) echo; $A push; pause ;;
        1) menu_manual ;;
        2) menu_auto ;;
        3) menu_camera ;;
        4) menu_settings ;;
        5) menu_system ;;
        r|R|"") ;;
        q|Q|0) clear; exit 0 ;;
        *) echo "  ${R}없는 번호입니다${N}"; sleep 1 ;;
    esac
done
