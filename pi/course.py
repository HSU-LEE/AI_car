from course_lib import *    
from course_lib import _flat, setup

# --- 주행 ---
SPEED = 0.45                 # 기본 주행 속도. 코스 중간에 바꾸려면 set_speed(0.4)

# --- 신호등 ---
RED_BACK_ON = True          # wait_green에서 빨간불을 못 찾으면 후진해서 다시 찾을지 (True / False)

# --- 레고 사람 (등록: Tune Person.command) ---
PERSON_ON = True            # 사람을 보고 멈출지
PERSON_ANY = False          # 등록하지 않은 레고에도 멈출지 (True / False). False면 등록한 레고만 찾음. 여러 레고는 aicar person add로 추가 등록
PERSON_IN_LANE = True       # 차선 사이에 있는 사람만 볼지 (True / False). 차선이 하나도 안 보일 때는 구분할 수 없어 사람으로 봄
PERSON_AT = 0.55            # 이 선(화면 높이 0.4~0.95)보다 발이 아래로 와야 사람으로 봄. 0.55 = 약 40 cm, 0.5 = 약 46 cm. 위쪽(먼 곳)은 무시. 아래쪽은 색만 맞으면 크기, 윤곽, 위치와 상관없이 바로 정지. None = 등록한 위치
PERSON_STOP_DELAY = 0.0     # 지금은 안 씀 (감지되면 바로 멈춤)
PERSON_RESUME_DELAY = 1.5   # 사람이 전혀 안 보인 뒤 출발까지 기다리는 시간 (초). 그 사이 한 번이라도 보이면 다시 기다림
PERSON_MIN_STOP = 2.0       # 최소 정지 시간 (초)
PERSON_MAX_WAIT = None      # 사람이 있어도 출발하는 시간 (초). None = 없어질 때까지 대기
PERSON_X = (0.0, 1.0)       # 사람을 찾을 가로 범위: 왼쪽 0 ~ 오른쪽 1. 전체 = 화면에 잡히면 어디든

setup(globals())    


# --- 동작 함수 한눈에 보기 ---
#
# [기본 동작]
# follow(sec, until)        차선을 따라가다 sec초가 지나거나 until 조건이 되면 끝 (기본 until="junction")
# straight(sec, angle)      조향을 유지(angle="hold")하거나 지정 각도로 직진, angle=0이면 정면
# right(sec, angle)         새 차선이 잡힐 때까지 우회전 (0.5~3초)
# left(sec, angle)          새 차선이 잡힐 때까지 좌회전 (0.5~3초)
# turn(direction)           "left"/"right"를 변수로 받아 left()/right() 호출
# curve(angle, sec)         지정 각도로 돌면서 차선 추종을 섞음
# stop(sec)                 그 자리에서 sec초 정지
# wait(until="clear")       선 채로 앞 장애물이 없어질 때까지 대기
# wait_green(max_sec)       선 채로 초록불이 될 때까지 대기 (빨간불을 봤으면 무제한. 못 찾으면 1초씩 최대 2번 후진해 찾고, 그래도 없으면 max_sec초 뒤 출발)
# start_gate(hold_sec)      출발선 앞 장애물이 치워지고 hold_sec초 지나면 출발
# finish()                  코스 끝, 정지
# set_speed(v)              이 뒤 주행 속도를 v로 (턴, straight, 직접 준 speed= 제외). set_speed()는 기본값으로 되돌림. tr, tl, corner가 끝나면 다시 쓰지 않아도 set_speed 값(없으면 SPEED)으로 돌아옴
#
# [목표 지점까지 가기]
# to_junction(at, nth)      다음 교차로 선 앞까지 감속하며 가서 끝 (nth=2: 첫 선은 지나치고 두 번째에서)
# to_crosswalk(at)          횡단보도 앞까지 감속하며 가서 끝
# to_gap(at)                차선이 끊기는 곳 앞까지 감속하며 가서 끝
# to_end(at)                바로 앞 끝에서 끝: 교차로선(가로선)이든 차선 끊김이든 먼저 오는 쪽
#                           to_junction, to_end의 cm=10: 선을 자로 재서 기준 위치에서 10cm 더 간 뒤 끝 (시간과 달리 배터리·속도 영향이 적음)
#                           cm=0은 선이 카메라 앞 38cm일 때. 약 17cm까지는 선으로 재고, 선이 화면을 벗어나면 잰 속도로 추정 (최대 30)
#
# [묶음 동작 (여러 단계)]
# cross_gap(at, angle)      끊긴 곳까지 간 뒤 새 차선이 잡힐 때까지 직진
# traffic(at, max_sec)      횡단보도 앞에 서서 초록불에 출발
# cross(at, line, approach) 교차로선까지 가서 정면으로 건넘 / line="crosswalk": 차선 따라 횡단보도 건넘 (approach=False: 이미 앞에 서 있을 때)
# corner(direction, wide)   교차로 선까지 가서 go_sec초 더 간 뒤 꺾고 정렬·천천히 주행 (to_junction + straight + tr/tl)
# uturn(direction)          교차로 선 앞에서 최대 조향으로 돌아 정렬·천천히 주행 (blind_sec 동안 새 차선 안 봄)
# tr(min_sec, sec, deg)     우회전 → 차로에 맞춰 정렬 → sec초 천천히(0.3) 주행 (deg: 이만큼 돌면 끝, min_sec: 최소 이 시간은 돎)
# tl(min_sec, sec, deg)     좌회전 → 차로에 맞춰 정렬 → sec초 천천히(0.3) 주행 (deg: 이만큼 돌면 끝, min_sec: 최소 이 시간은 돎)
#
# [공통 옵션]
# at=        감지 위치 (0.4~1.0, 클수록 가까이)     slow=      선에 가까워질수록 이 속도까지 감속
# speed=     주행 속도                              blind_sec= 시작 후 이 시간 동안 감지 안 함
# min_sec=   최소 진행 시간                         max_sec=   최대 진행 시간
# side=      기준 선 ("left"/"right"/"center")      shift=     차로 안에서 치우침 (cm, +오른쪽, ±8)
# style=     조향 성향 ("soft"/"normal"/"sharp")    steer=     최대 조향각
# hold_sec=  조건이 이 시간 동안 유지돼야 끝
# until=     "junction" "crosswalk" "lane" "gap" "end" "crossed" "green" "obstacle" "clear" "lost" "aligned"

# --- 예시 코스 ---

    # start_gate(2),
    # to_junction(),
    # straight(),
    # tr(),
    # follow(4),
    # set_speed(0.35),
    # to_crosswalk(0.8),
    # follow(2),
    # wait_green(max_sec=20),
    # stop(1.5),
    # pass_crosswalk(),
    # set_speed(0.45),
    # to_junction(),
    # straight(),
    # tr(),
    # set_speed(0.35),
    # to_junction(),
    # straight(2.5),
    # follow(4),
    # set_speed(0.45),
    # to_junction(),
    # follow(2.5),    
    # straight(2),
    # tr(),
    # set_speed(0.45),
    # to_junction(),
    # to_crosswalk(0.85),
    # straight(4),
    # finish(),


# -------------



# --- 코스 ---

COURSE = _flat([
    start_gate(2), 
    to_junction(),
    straight(),
    tr(),
    follow(4),
    set_speed(0.35),
    to_crosswalk(0.8),
    follow(2),
    wait_green(max_sec=20),
    stop(1.5),
    pass_crosswalk(),
    set_speed(0.45),
    to_junction(),
    straight(),
    tr(),
    set_speed(0.35),
    to_junction(),
    straight(2.5),
    follow(4),
    set_speed(0.45),
    to_junction(),
    follow(2.5),    
    straight(2),
    tr(),
    set_speed(0.45),
    to_junction(),
    to_crosswalk(0.85),
    straight(4),
    finish(),

 


])

if __name__ == "__main__":
    for i, step in enumerate(_flat(COURSE), 1):
        print(f"{i:2d}. {describe(step)}")



