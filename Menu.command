#!/bin/bash
# Double-click menu. Every action is the `aicar` command in this folder.
cd "$(dirname "$0")" || exit 1
A=./aicar
STATE=.aicar

B=$'\033[1m'; D=$'\033[2m'; G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; C=$'\033[36m'; N=$'\033[0m'

ask() { read -r -p "  $1" REPLY || return 1; echo "$REPLY"; }   # Fail when input ends because the window closed.
say() { echo "  $*"; }
warn() { echo "  ${Y}!${N} $*"; }
cancel() { echo "  ${D}취소했습니다.${N}"; return 1; }

# Yes/no. Only y (or Y) is yes.
confirm() { local y; y=$(ask "$1 [y/N] ") || return 1; [ "$y" = y ] || [ "$y" = Y ] || cancel; }

# Before anything that moves the car: what moves, then a yes.
moves() { warn "$1"; confirm "준비됐나요?"; }

# A number within a range, or cancel. ask_num "prompt" lo hi [default]
ask_num() {
    local v
    v=$(ask "$1") || return 1
    [ -z "$v" ] && [ -n "${4:-}" ] && v=$4
    [ -z "$v" ] && { cancel; return 1; }
    if ! [[ "$v" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || ! awk -v v="$v" -v lo="$2" -v hi="$3" 'BEGIN{exit !(v>=lo && v<=hi)}'; then
        echo "  ${R}$2 ~ $3 사이의 숫자를 넣어 주세요${N}" >&2
        return 1
    fi
    echo "$v"
}

# One of the given words, or cancel. ask_pick "prompt" default word...
ask_pick() {
    local prompt=$1 def=$2 v w; shift 2
    v=$(ask "$prompt") || return 1
    v=${v:-$def}
    for w in "$@"; do [ "$v" = "$w" ] && { echo "$v"; return 0; }; done
    echo "  ${R}$* 중에서 골라 주세요${N}" >&2
    return 1
}

# --- Status line ---
pi_status() {
    local h state
    h=$(cat "$STATE/host" 2>/dev/null)
    if [ -z "$h" ] || ! nc -z -G 1 "$h" 22 2>/dev/null; then
        echo "  ${R}●${N} 파이에 연결되지 않았습니다 ${D}(명령을 실행하면 자동으로 다시 찾습니다. 차 전원과 와이파이를 확인하세요)${N}"
        return
    fi
    state=$(ssh -n -o BatchMode=yes -o ConnectTimeout=2 -o ControlMaster=auto \
                -o "ControlPath=$STATE/cm-%r@%h" -o ControlPersist=120 "pi@$h" \
                'p=$(cat ~/aicar/drive.pid 2>/dev/null) && ps -p "$p" -o args= | grep -q lane_follow && echo drive
                 grep -q "\"enabled\": *false" ~/aicar/button.json 2>/dev/null && echo button-off' 2>/dev/null)
    local btn="${D}버튼 출발 켜짐${N}"
    [[ "$state" == *button-off* ]] && btn="${D}버튼 출발 꺼짐 (바로 출발)${N}"
    if [[ "$state" == *drive* ]]; then
        echo "  ${G}●${N} 파이 연결됨 ${D}$h${N}   ${Y}▶ 자율주행 중${N} ${D}(s: 정지)${N}   $btn"
    else
        echo "  ${G}●${N} 파이 연결됨 ${D}$h${N}   ${D}○ 대기 중${N}   $btn"
    fi
}

header() {
    clear
    echo
    echo "  ${B}AICAR${N}${1:+ ${D}›${N} ${B}$1${N}}"
    echo "  ${D}──────────────────────────────────────────────────────${N}"
    pi_status
    echo
}

# item key label. A hint after it (the menus still pass one) is not shown.
item() { printf "   ${C}%-3s${N} %s\n" "$1" "$2"; }
section() { echo "  ${B}$1${N}"; }

pause() { echo; read -r -p "  ${D}엔터를 누르면 메뉴로 돌아갑니다...${N}" _; }

# --- Submenus ---
# submenu "title" "one line about this screen (kept for reference, not shown)" entries...
# Entry: "key|label|hint|command" (hint may be empty; the command may contain |). "-" is a gap, "#text" a section title.
# Keys s (stop), b (back) and h (help) work everywhere.
submenu() {
    local title="$1" about="$2"; shift 2
    local entries=("$@") e key rest label hint cmd n
    while true; do
        header "$title" "$about"
        for e in "${entries[@]}"; do
            case "$e" in
                -) echo; continue ;;
                \#*) section "${e#\#}"; continue ;;
            esac
            key=${e%%|*}; rest=${e#*|}; label=${rest%%|*}; rest=${rest#*|}; hint=${rest%%|*}
            item "$key" "$label" "$hint"
        done
        echo
        item s "${R}■ 정지${N}" "달리는 차를 바로 멈춤"
        item b "뒤로"
        echo
        n=$(ask "선택 › ") || exit 0
        case "$n" in
            b|B|"") return ;;
            s|S) echo; $A stop; pause; continue ;;
            h|H) help_screen; continue ;;
        esac
        cmd=""
        for e in "${entries[@]}"; do
            [ "${e%%|*}" = "$n" ] && { rest=${e#*|}; rest=${rest#*|}; cmd=${rest#*|}; }
        done
        if [ -z "$cmd" ]; then echo "  ${R}없는 번호입니다. 다시 선택하세요${N}"; sleep 1; continue; fi
        echo
        eval "$cmd"
        pause
    done
}

menu_manual() {
    submenu "수동 운전" "차를 직접 움직여 봅니다. 바퀴가 돌 수 있으니 차를 들거나 넓은 곳에 두세요." \
        '#운전' \
        '1|키보드로 운전|w/s 속도, a/d 조향, 스페이스 정지, q 끝|moves "키보드로 차가 움직입니다." && { s=$(ask_num "최대 속도 0.1~1 (엔터 = 0.6): " 0.1 1 0.6) && $A drive-keys "$s"; }' \
        '2|직진 주행|속도와 시간을 정해 똑바로|s=$(ask_num "속도 -1~1 (음수 = 후진): " -1 1) && t=$(ask_num "시간 0.1~10초 (엔터 = 1): " 0.1 10 1) && moves "속도 $s로 ${t}초 동안 움직입니다." && $A drive "$s" "$t"' \
        '-' \
        '#조향' \
        '3|조향 중앙으로|앞바퀴를 똑바로|$A center' \
        '4|조향 각도 시험|정한 각도로 1초 뒤 중앙|a=$(ask_num "각도 -26~26 (음수 = 왼쪽): " -26 26) && $A steer "$a"' \
        '5|조향 + 전진/후진 시험|한 번에 전체 동작 확인|moves "앞바퀴가 움직이고 차가 앞뒤로 조금 움직입니다." && $A test'
}

menu_auto() {
    submenu "자율주행" "course.py 순서대로 달립니다. 버튼 출발이 켜져 있으면 차의 버튼을 눌러야 출발합니다." \
        '#달리기' \
        '1|자율주행 시작|코드를 올리고 바로 시작|$A go' \
        '2|자율주행 + 녹화|나중에 다시 분석할 수 있게 영상 저장|$A go --record' \
        '3|코스 중간부터 시작|몇 번째 동작부터 할지 고름|k=$(ask_num "몇 번 동작부터 (1부터): " 1 999) && $A go --start "$k"' \
        '4|설정 바꿔서 한 번만|예: KP=1.8 (이번 주행에만 적용)|v=$(ask "KEY=값 (여러 개는 띄어쓰기): ") && if [ -n "$v" ]; then $A go --set $v; else cancel; fi' \
        '5|버튼 없이 바로 출발|이번 한 번만 버튼을 건너뜀|moves "버튼 없이 바로 달립니다." && $A go --button off' \
        '-' \
        '#보기' \
        '6|인식만 보기|모터 꺼짐. 손으로 밀며 인식 확인|$A view' \
        '7|주행 화면 다시 보기|연결이 끊긴 뒤 다시 붙기|$A log' \
        '-' \
        '#준비' \
        '8|코스 편집|course.py 열기, 저장 후 u로 올리기|$A course' \
        '9|버튼 · LED 설정|버튼 출발, 꾹 누르기, 조향 확인, 초록 LED|menu_button'
}

menu_button() {
    submenu "버튼 · LED 설정" "GPIO21 버튼: 짧게 = 출발 / 일시정지 / 다시 출발, 꾹 = 대기로. 설정은 파이에 저장됩니다." \
        '0|지금 설정 보기||$A button' \
        '-' \
        '#버튼 출발' \
        '1|버튼 출발 켜기/끄기|끄면 aicar go가 바로 출발|v=$(ask_pick "on 또는 off: " "" on off) && $A button "$v"' \
        '2|주행 중 누르면 바로 멈춤|off면 손을 뗄 때 멈춤 (기본 on)|v=$(ask_pick "on 또는 off: " "" on off) && $A button press-stop "$v"' \
        '-' \
        '#꾹 누르기' \
        '3|꾹 누른 뒤 다시 출발하면|restart = 1단계부터, keep = 멈춘 곳부터|v=$(ask_pick "restart 또는 keep: " "" restart keep) && $A button hold "$v"' \
        '4|꾹 누르는 시간|1~10초 (기본 3초)|v=$(ask_num "몇 초 1~10: " 1 10) && $A button hold-sec "$v"' \
        '-' \
        '#출발 전 조향 확인' \
        '5|조향 확인 켜기/끄기|처음 출발할 때 앞바퀴 왼쪽→오른쪽→중앙|v=$(ask_pick "on 또는 off: " "" on off) && $A button sweep "$v"' \
        '6|조향 확인 빠르기|slow 3초 / normal 1.8초 / fast 1.2초|v=$(ask_pick "slow, normal, fast (엔터 = normal): " normal slow normal fast) && $A button sweep-speed "$v"' \
        '-' \
        '#초록 LED' \
        '7|LED 표시 켜기/끄기|주행 중 켜짐, 조향 확인·일시정지·누름은 깜빡임|v=$(ask_pick "on 또는 off: " "" on off) && $A button led "$v"' \
        '8|LED 사용 준비|파이 설정 한 번 (처음 한 번만)|confirm "파이에 초록 LED 권한 규칙을 추가할까요?" && $A led setup' \
        '9|LED 사용 되돌리기|권한 규칙 지우기|confirm "초록 LED 권한 규칙을 지울까요?" && $A led undo' \
        '-' \
        '#시험 (모터는 돌지 않음)' \
        't|버튼 누름 시험|30초 동안 누르면 무엇이 되는지 표시|$A button test' \
        'w|조향 확인 해 보기|앞바퀴만 좌우로 움직임|moves "앞바퀴가 좌우로 움직입니다 (모터는 꺼짐)." && $A button sweep-test' \
        'l|LED 패턴 보기|모든 패턴을 이름과 함께 차례로|$A button led-test'
}

menu_camera() {
    submenu "카메라 / 기록" "녹화는 --record 주행만, 주행 기록은 매 주행 자동으로 파이에 남습니다 (영상 없음)." \
        '#카메라' \
        '1|카메라 + 인식 화면|모터 안 움직임, 브라우저로 봄|$A camera' \
        '2|사진 찍기|맥으로 받아서 열기|$A snap' \
        '-' \
        '#녹화' \
        '3|녹화 목록|파이에 있는 녹화|$A runs' \
        '4|최근 녹화 받기|맥의 runs/ 폴더로|$A fetch latest' \
        '5|최근 녹화 다시 분석|맥에서 영상으로 만들어 열기|$A replay latest' \
        '6|오래된 녹화 지우기|최근 몇 개만 남김|k=$(ask_num "최근 몇 개를 남길까요 (엔터 = 5): " 1 100 5) && confirm "파이에서 나머지 녹화를 지울까요? 되돌릴 수 없습니다." && $A clean-runs "$k"' \
        '-' \
        '#주행 기록' \
        '7|주행 기록 목록|최근 30개|$A logs' \
        '8|최근 주행 기록 보기|동작별 시작·끝, 경고, 오류|$A fetch-log latest' \
        '9|cm= 종료 위치 보고|목표 cm, 끝날 때 cm, 오차, 속도|n=$(ask_num "최근 몇 번 주행 (엔터 = 5): " 1 50 5) && $A end-report "$n"' \
        '0|주행 기록 자동 저장 켜기/끄기|기본 켜짐|$A drive-log; confirm "켜짐/꺼짐을 바꿀까요?" && $A drive-log toggle'
}

menu_register() {
    submenu "인식 등록" "브라우저 화면에서 상자를 그려 등록합니다. 경기장 조명이 바뀌면 다시 등록하세요." \
        '#레고 사람' \
        '1|레고 사람 새로 등록|상자 안에서 돌리면 자동으로 찍힘|$A person new' \
        '2|레고 사람 더 등록|이미 등록한 것은 유지|$A person add' \
        '-' \
        '#표지' \
        '3|신호등 등록|빨간불, 초록불에 상자|v=$(ask_pick "new 또는 add (엔터 = new): " new new add) && $A light "$v"' \
        '4|횡단보도 등록|줄무늬에 상자, 2~3가지 거리에서|v=$(ask_pick "new 또는 add (엔터 = new): " new new add) && $A crosswalk "$v"' \
        '5|차선 등록|선 조각에 상자, 반사광 기준 자동|v=$(ask_pick "new 또는 add (엔터 = new): " new new add) && $A line "$v"' \
        '6|정지 지점 등록|차를 멈출 자리에 놓고 상자 (go_to용)|v=$(ask_pick "new 또는 add (엔터 = new): " new new add) && $A stoppoint "$v"'
}

menu_settings() {
    submenu "설정 / 보정" "경기장에 처음 가면 0번(자동 보정)부터 하세요." \
        '1|지금 설정 보기|보정, 선, 주요 주행 값|$A config' \
        '-' \
        '#보정' \
        '0|자동 보정|직선 차로 가운데에 놓고 약 15초|say "차를 직선 차로 가운데에, 차로 방향으로 똑바로 놓으세요."; say "측정하는 동안 차와 카메라 앞을 가리지 마세요."; confirm "시작할까요?" && $A calibrate' \
        'e|정지 거리 보정|차 앞 10cm(줄자)에 차선 끝을 두고|say "차선 끝이 차 앞에서 정확히 10cm(줄자)가 되게 놓으세요."; confirm "시작할까요?" && $A calibrate --end 10' \
        'f|초점 측정만|저장 안 함, 주행은 15cm 고정|$A focus' \
        'g|차선 끊김 확인|모터 꺼짐, 손으로 밀며 기록|s=$(ask_num "몇 초 (엔터 = 20): " 5 120 20) && $A gap-probe "$s"' \
        '-' \
        '#조향 중앙' \
        '2|키보드로 맞추기|a/d로 움직이고 s로 저장|$A tune-center' \
        '3|숫자로 옮기기|예: -0.5 (왼쪽으로 0.5도)|d=$(ask_num "옮길 각도 -5~5: " -5 5) && $A trim "$d"' \
        '-' \
        '#인식' \
        '4|트랙 색 바꾸기|검은 바닥·흰 선 ↔ 흰 바닥·검은 선|v=$(ask_pick "white 또는 black: " "" white black) && $A track "$v"' \
        '5|선 인식 맞추기|브라우저 보며 w/s, 엔터 저장|$A tune-white' \
        '6|신호등 색 맞추기|브라우저 보며, 엔터 저장|$A tune-light' \
        '7|인식 등록|레고 사람, 신호등, 횡단보도, 차선, 정지 지점|menu_register' \
        '-' \
        '8|주행 설정 편집|lane_follow.py 열기, 저장 후 u로 올리기|$A set'
}

menu_system() {
    submenu "파이 관리" "연결, 코드, 와이파이, 전원. 전원을 뽑기 전에는 0번으로 끄세요." \
        '#연결과 코드' \
        '1|상태 자세히 보기|주소, 와이파이, 온도, 저장 공간, 실행 중|$A status' \
        '2|코드 올리기 (맥 → 파이)|go와 view는 저절로 올림|$A push' \
        '3|코드 가져오기 (파이 → 맥)|파이에서 직접 고친 경우|confirm "맥의 pi/ 폴더를 파이 코드로 덮어쓸까요?" && $A pull-code' \
        '4|파이 터미널 열기||$A ssh' \
        '5|파이썬 환경 점검||$A check' \
        '6|맥에서 인식 시험|파이 없이 합성 영상으로 전체 시험|$A test-vision' \
        '-' \
        '#와이파이' \
        '7|와이파이 상태 / 주변 와이파이||$A wifi' \
        '8|새 와이파이 등록|이름과 비밀번호를 물어봄|$A wifi; echo; $A wifi-add' \
        '9|저장된 와이파이 삭제||$A wifi; echo; w=$(ask "삭제할 와이파이 이름: ") && [ -n "$w" ] && confirm "$w 를 지울까요?" && $A wifi-del "$w"' \
        '-' \
        '#전원' \
        'r|파이 재부팅|약 1분 걸림|confirm "파이를 재부팅할까요?" && $A reboot' \
        'f|부팅 빠르게|안 쓰는 서비스 끄기 (되돌리기: aicar fast-boot undo)|confirm "파이 설정을 바꿀까요?" && $A fast-boot' \
        '0|파이 끄기|전원 뽑기 전에|confirm "파이를 종료할까요?" && $A shutdown'
}

# Before a race: everything that can be checked without moving the car, in one go.
race_check() {
    header "경기 전 점검" "차를 움직이지 않고 확인할 수 있는 것을 차례로 봅니다."
    section "1. 파이 상태"
    $A status || { warn "파이에 연결되지 않습니다. 차 전원과 와이파이를 확인하세요."; pause; return; }
    echo; section "2. 설정과 보정"
    $A config
    echo; section "3. 버튼 · LED"
    $A button
    echo; section "4. 코드 올리기"
    $A push
    echo
    say "${B}남은 것:${N} 경기장이 새 곳이면 설정 → 0(자동 보정), 그다음 자율주행 → 6(인식만 보기)으로 확인하세요."
    pause
}

help_screen() {
    header "도움말"
    section "처음 쓸 때"
    say "1. 차 전원을 켜고 맥을 같은 와이파이(또는 차 핫스팟 AICAR-AP)에 연결합니다."
    say "2. 4 설정 → 0 자동 보정 (경기장마다 한 번)."
    say "3. 2 자율주행 → 6 인식만 보기로 선이 잘 잡히는지 봅니다."
    say "4. g 자율주행 시작. 버튼 출발이 켜져 있으면 차의 버튼을 누릅니다."
    echo
    section "멈추는 방법"
    say "어디서든 s (정지) · 주행 화면에서 Ctrl+C · 차의 버튼 한 번 (누르는 순간 멈춤)"
    say "맥 연결이 끊겨도 차는 계속 달립니다. 버튼으로 멈추거나, 다시 연결해 s를 누르세요."
    echo
    section "초록 LED (버튼 출발일 때)"
    say "꺼짐 = 대기 · 빠른 깜빡임 = 조향 확인 · 켜짐 = 주행 중 · 두 번 깜빡 = 일시정지"
    say "꾹 누르는 동안: 1초에 1번, 2초에 2번, 3초에 떨림 → 손을 떼면 대기"
    echo
    section "더 알아보기"
    say "터미널에서 ./aicar help 로 모든 명령을 볼 수 있습니다."
    pause
}

# --- Main screen ---
while true; do
    header
    section "빠른 실행"
    item g "자율주행 시작" "course.py 순서대로"
    item s "${R}■ 정지${N}" "달리는 차를 바로 멈춤"
    item v "인식만 보기" "모터 꺼짐, 손으로 밀며 확인"
    item c "카메라 + 인식 화면" "모터 꺼짐"
    item u "코드 올리기" "맥 → 파이"
    item k "경기 전 점검" "상태, 보정, 버튼, 코드를 한 번에"
    echo
    section "메뉴"
    item 1 "수동 운전" "키보드 운전, 직진, 조향 시험"
    item 2 "자율주행" "출발 방식, 코스 편집, 버튼 · LED"
    item 3 "카메라 / 기록" "녹화, 주행 기록, 종료 위치 보고"
    item 4 "설정 / 보정" "자동 보정, 조향 중앙, 인식 등록"
    item 5 "파이 관리" "연결, 와이파이, 전원"
    echo
    item h "도움말"
    item r "상태 새로고침"
    item q "끝내기"
    echo
    n=$(ask "선택 › ") || exit 0
    case "$n" in
        g|G) echo; $A go; pause ;;
        s|S) echo; $A stop; pause ;;
        v|V) echo; $A view; pause ;;
        c|C) echo; $A camera; pause ;;
        u|U) echo; $A push; pause ;;
        k|K) race_check ;;
        1) menu_manual ;;
        2) menu_auto ;;
        3) menu_camera ;;
        4) menu_settings ;;
        5) menu_system ;;
        h|H) help_screen ;;
        r|R|"") ;;
        q|Q|0) clear; exit 0 ;;
        *) echo "  ${R}없는 번호입니다. 다시 선택하세요${N}"; sleep 1 ;;
    esac
done
