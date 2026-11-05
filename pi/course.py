


# --- 주행 ---
# 전진 최소 속도는 0.3 (lane_follow.MOTOR_MIN). 그보다 낮은 값은 0.3으로 달린다.
SPEED = 0.5                 # 기본 주행 속도. 코스 중간에 바꾸려면 set_speed(0.4)

# --- 레고 사람 (등록: Tune Person.command) ---
PERSON_ON = True            # 사람을 보고 멈출지
PERSON_AT = None            # 정지 위치: 화면 높이 0.4~0.95, 클수록 가까이. None = 등록한 위치
PERSON_SLOW = 0.3           # 사람이 보일 때 줄일 속도. None = 안 줄임
PERSON_STOP_DELAY = 0.0     # 정지 위치에 닿고 더 가는 시간 (초)
PERSON_RESUME_DELAY = 1.5   # 사람이 없어지고 출발까지 기다리는 시간 (초)
PERSON_MIN_STOP = 2.0       # 최소 정지 시간 (초)
PERSON_MAX_WAIT = None      # 사람이 있어도 출발하는 시간 (초). None = 없어질 때까지 대기
PERSON_X = (0.1, 0.9)       # 사람을 찾을 가로 범위: 왼쪽 0 ~ 오른쪽 1


EVENTS = ("junction", "crosswalk", "lane", "gap", "end", "crossed", "green", "obstacle", "clear", "lost", "aligned", None)
SIDES = ("left", "right", "center", None)
STYLES = ("soft", "normal", "sharp", None)
OPTIONS = ("min_sec", "max_sec", "speed", "at", "side", "shift", "steer", "style", "hold_sec", "slow", "blind_sec", "deg", "nth", "cm")


def _hint(word, choices):
    import difflib
    near = difflib.get_close_matches(str(word), [str(c) for c in choices if c is not None], 1)
    return f' → "{near[0]}" 말인가요?' if near else ""


def _check_config():
    if PERSON_AT is not None and not 0.4 <= PERSON_AT <= 0.95:
        raise ValueError("PERSON_AT은 0.4 ~ 0.95 (클수록 가까이) 또는 None")
    if not 0 <= PERSON_X[0] < PERSON_X[1] <= 1:
        raise ValueError("PERSON_X는 (왼쪽, 오른쪽), 0 ~ 1")
    for name, value in (("PERSON_STOP_DELAY", PERSON_STOP_DELAY), ("PERSON_RESUME_DELAY", PERSON_RESUME_DELAY),
                        ("PERSON_MIN_STOP", PERSON_MIN_STOP), ("PERSON_MAX_WAIT", PERSON_MAX_WAIT or 0)):
        if value < 0:
            raise ValueError(f"{name}는 0 이상의 초")


_check_config()


def _step(do, angle, sec, until, **opt):
    for k in opt:
        if k not in OPTIONS:
            raise ValueError(f"{do}(): 옵션 {k}= 는 없음{_hint(k, OPTIONS)} (쓸 수 있는 옵션: {', '.join(OPTIONS)})")
    if sec is not None and until is not None and do != "wait":
        for k in ("at", "slow", "cm"):
            if opt.get(k) is not None:
                raise ValueError(f"{do}(): {k}=는 until과 함께 쓰는 옵션이라 sec과 같이 못 씀")
        until = None
    if until not in EVENTS:
        raise ValueError(f'{do}(): until="{until}" 는 없음{_hint(until, EVENTS)}')
    if opt.get("side") not in SIDES:
        raise ValueError(f'{do}(): side는 "left" / "right" / "center"{_hint(opt.get("side"), SIDES)}')
    if opt.get("style") not in STYLES:
        raise ValueError(f'{do}(): style은 "soft" / "normal" / "sharp"{_hint(opt.get("style"), STYLES)}')
    at = opt.get("at")
    if at is not None and not 0.4 <= at <= 1.0:
        raise ValueError(f"{do}(): at은 0.4 ~ 1.0 (클수록 가까이)")
    if at is not None and at > 0.85 and until == "crosswalk":
        print(f"[주의] {do}(until=\"crosswalk\", at={at}): 0.85보다 크면 횡단보도가 안 잡혀 안 끝날 수 있음 (0.67 권장)")
    if opt.get("slow") is not None and until not in ("junction", "crosswalk", "gap", "end"):
        raise ValueError(f'{do}(): slow=는 until="junction" / "crosswalk" / "gap" / "end"일 때만 씀 (그 선에 가까워질수록 감속)')
    nth = opt.get("nth")
    if nth is not None and (not isinstance(nth, int) or nth < 1):
        raise ValueError(f"{do}(): nth는 1 이상의 정수 (몇 번째 선에서 끝낼지)")
    cm = opt.get("cm")
    if cm is not None and until not in ("junction", "end"):
        raise ValueError(f'{do}(): cm=는 until="junction" / "end"일 때만 씀')
    if cm is not None and not 0 <= cm <= 30:
        raise ValueError(f"{do}(): cm는 0 ~ 30 (약 17cm까지는 선으로 재고, 그 뒤는 속도로 추정)")
    shift = opt.get("shift")
    if shift is not None and abs(shift) > 8:
        raise ValueError(f"{do}(): shift는 -8 ~ 8 (cm, +는 오른쪽). 차로 폭이 20cm라 그 이상은 선을 밟음")
    if sec is None and until is None and opt.get("max_sec") is None and do != "finish":
        raise ValueError(f"{do}(): sec, until, max_sec 중 하나는 필요")
    return {"do": do, "angle": angle, "sec": sec, "until": until,
            "min": opt.get("min_sec", 0.0), "max": opt.get("max_sec"), "speed": opt.get("speed"),
            "at": at, "hold": opt.get("hold_sec", 0.0), "side": opt.get("side"), "shift": shift,
            "steer": opt.get("steer"), "style": opt.get("style"), "slow": opt.get("slow"),
            "blind": opt.get("blind_sec") or 0.0, "deg": opt.get("deg"), "nth": opt.get("nth"), "cm": cm}


def _auto(step, speed, default):
    """speed=를 안 줬으면 default로 달리고, 코스의 set_speed()를 따른다. speed=를 직접 주면 그 값 그대로."""
    return dict(step, speed=default, auto=True) if speed is None else step


def set_speed(speed=None):
    """이 뒤 동작들의 주행 속도를 바꾼다. 턴(tr, tl, left, right, curve)과 straight, 직접 준 speed=는 그대로. set_speed()는 기본값으로 되돌린다."""
    if speed is not None and not 0 < speed <= 1:
        raise ValueError(f"set_speed(): 속도는 0보다 크고 1 이하 ({speed})")
    return {"do": "set_speed", "speed": speed}


def follow(sec=None, until="junction", speed=None, **opt):
    """차선을 따라가다가 until 조건이 되면 끝낸다."""
    return _auto(_step("follow", 0.0, sec, until, speed=speed, **opt), speed, 0.425)


def straight(sec=None, until=None, angle="hold", speed=None, **opt):
    """시작 순간의 조향각을 유지한 채 직진하고, 0이면 정면으로 간다."""
    if not (angle == "hold" or isinstance(angle, (int, float))):
        raise ValueError('straight(): angle은 숫자 또는 "hold"')
    return _step("straight", angle, sec, until, speed=speed, **opt)


def right(sec=None, angle=None, until="lane", speed=None, **opt):
    """새 차선이 잡힐 때까지 오른쪽으로 꺾는다."""
    opt.setdefault("min_sec", 0.5)
    opt.setdefault("max_sec", 3.0)
    return _step("right", None if angle is None else abs(angle), sec, until, speed=speed, **opt)


def left(sec=None, angle=None, until="lane", speed=None, **opt):
    """새 차선이 잡힐 때까지 왼쪽으로 꺾는다."""
    opt.setdefault("min_sec", 0.5)
    opt.setdefault("max_sec", 3.0)
    return _step("left", None if angle is None else -abs(angle), sec, until, speed=speed, **opt)


def turn(direction, **opt):
    """방향을 변수로 받아 좌회전이나 우회전을 만든다."""
    if direction not in ("left", "right"):
        raise ValueError(f'turn(): 방향은 "left" / "right"{_hint(direction, ("left", "right"))}')
    return (right if direction == "right" else left)(**opt)


def curve(angle, sec=None, until=None, speed=None, **opt):
    """지정 각도로 돌면서 차선 추종을 섞는다."""
    return _step("curve", angle, sec, until, speed=speed, **opt)


def stop(sec, speed=0.0):
    """그 자리에서 지정한 시간만큼 선다."""
    return _step("stop", 0.0, sec, None, speed=speed)


def wait_green(max_sec=10, min_sec=0.0, speed=0.0):
    """선 채로 신호등이 빨강에서 초록으로 바뀌면 출발한다."""
    return _step("wait", 0.0, None, "green", min_sec=min_sec, max_sec=max_sec, speed=speed)


def wait(sec=None, until="clear", min_sec=0.0, max_sec=None, hold_sec=0.0, speed=0.0):
    """선 채로 앞 장애물이 없어질 때까지 기다린다."""
    return _step("wait", 0.0, sec, until, min_sec=min_sec, max_sec=max_sec, speed=speed, hold_sec=hold_sec)


def start_gate(hold_sec=1.0, speed=0.0):
    """앞을 막던 장애물이 치워지면 출발한다."""
    return wait(until="clear", hold_sec=hold_sec, speed=speed)


def finish():
    """코스를 끝내고 정지한다."""
    return _step("finish", 0.0, 0, None)


def to_junction(at=None, slow=0.3, blind_sec=0.5, speed=None, nth=1, cm=None):
    """다음 교차로 선 앞까지 가서 끝낸다. nth=2면 첫 번째 선(교차로처럼 보이는 굽은 길 등)은 지나치고 두 번째에서 끝낸다. cm=면 선을 자로 재서 기준 위치에서 그만큼 더 간 뒤 끝낸다."""
    return _auto(follow(until="junction", speed=speed, at=at, slow=slow, blind_sec=blind_sec, nth=nth if nth > 1 else None, cm=cm), speed, SPEED)


def to_crosswalk(at=0.67, slow=0.3, speed=None):
    """횡단보도 앞까지 가서 끝낸다."""
    return _auto(follow(until="crosswalk", speed=speed, at=at, slow=slow), speed, SPEED)


def pass_crosswalk():
    """횡단보도 앞에 선 상태에서 바로 차선 따라 건너고, 지나가면 끝낸다."""
    return cross(line="crosswalk", approach=False)

def to_gap(at=0.85, slow=0.3, speed=None):
    """차선이 끊기는 곳 앞까지 가서 끝낸다."""
    return _auto(follow(until="gap", speed=speed, at=at, slow=slow), speed, SPEED)


def to_end(at=0.5, slow=0.3, blind_sec=0.0, speed=None, shift=None, cm=None, max_sec=None):
    """바로 앞 끝(교차로선이든 차선 끊김이든 먼저 오는 쪽)에서 끝낸다. shift=치우쳐 달릴 cm(+는 오른쪽). cm=면 선을 자로 재서 기준 위치에서 그만큼 더 간 뒤 끝낸다."""
    return _auto(follow(until="end", speed=speed, at=at, slow=slow, blind_sec=blind_sec, shift=shift, cm=cm, max_sec=max_sec), speed, 0.3)


def _turn_fn(direction):
    """방향 이름을 right / left 동작 함수로 바꾼다."""
    if direction not in ("left", "right"):
        raise ValueError(f'방향은 "left" / "right"{_hint(direction, ("left", "right"))}')
    return right if direction == "right" else left


def _straight_speed(speed):
    """묶음 동작 안의 straight는 set_speed를 따르지 않는다."""
    return SPEED if speed is None else speed


def cross_gap(at=0.85, angle=0, speed=None):
    """끊긴 구간을 지나 다시 차선을 따라간다."""
    return [to_gap(at, speed=speed), straight(until="lane", angle=angle, max_sec=3, speed=_straight_speed(speed))]


def traffic(at=0.73, max_sec=10, speed=None):
    """횡단보도 앞에서 서서 초록불에 출발한다."""
    return [to_crosswalk(at, speed=speed), wait_green(max_sec=max_sec)]


def cross(at=None, line="junction", speed=None, approach=True):
    """교차로선은 정면으로, line="crosswalk"면 횡단보도를 차선 따라 건너고 끝낸다. approach=False면 다가가기 생략."""
    if line not in ("junction", "crosswalk"):
        raise ValueError(f'cross(): line은 "junction" / "crosswalk"{_hint(line, ("junction", "crosswalk"))}')
    if line == "junction":
        steps = [to_junction(at, speed=speed)] if approach else []
        return steps + [straight(until="lane", angle=0, min_sec=0.8, max_sec=4, speed=_straight_speed(speed))]
    steps = [to_crosswalk(at or 0.67, speed=speed)] if approach else []
    return steps + [_auto(follow(until="crossed", hold_sec=0.3, min_sec=0.5, max_sec=5, speed=speed), speed, SPEED)]


def corner(direction, at=None, go_sec=0.5, wide=0, stop_sec=0.20, speed=None):
    """교차로 선까지 가서 go_sec초 더 간 뒤 꺾어 정렬하고 stop_sec초 천천히 간다 (wide=바깥으로 치우칠 cm)."""
    turn_fn = _turn_fn(direction)
    shift = wide if direction == "left" else -wide  # 좌회전의 바깥은 오른쪽.
    steps = [_auto(follow(until="junction", at=at, slow=0.3, blind_sec=1.5, speed=speed, shift=shift or None), speed, SPEED)]
    if go_sec > 0:
        steps.append(straight(go_sec))
    return steps + _turn_align_stop(turn_fn, stop_sec)


def uturn(direction, at=None, blind_sec=1.5, stop_sec=0.20, speed=None):
    """교차로 선 앞에서 최대 조향으로 180도 돌아 정렬하고 stop_sec초 천천히 간다 (blind_sec 동안 새 차선 안 봄)."""
    turn_speed = 0.4 if speed is None else speed
    return [_auto(follow(until="junction", at=at, slow=turn_speed, speed=speed), speed, turn_speed),
            _turn_align_stop(_turn_fn(direction), stop_sec, turn_speed, angle=None, blind_sec=blind_sec, max_sec=5, deg=180)]


def tr(min_sec=0.5, sec=0.50, speed=None, deg=90):
    """우회전(최소 min_sec초, deg도 돌거나 새 차선이 보이면 끝) → 정렬 → sec초 천천히 차선을 따라간다."""
    return _turn_align_stop(right, sec, speed, angle=TR_ANGLE, deg=deg, min_sec=min_sec)


def tl(min_sec=0.5, sec=0.50, speed=None, deg=90):
    """좌회전(최소 min_sec초, deg도 돌거나 새 차선이 보이면 끝) → 정렬 → sec초 천천히 차선을 따라간다."""
    return _turn_align_stop(left, sec, speed, angle=TL_ANGLE, deg=deg, min_sec=min_sec)


TURN_SPEED = 0.35
TR_ANGLE = 21           
TL_ANGLE = 26          
ALIGN_SPEED = 0.4
SETTLE_SPEED = 0.3     
TURN_BLIND = 1.0        
TURN_BLIND_SPEED = 0.35  


def _turn_align_stop(turn_fn, stop_sec, speed=None, angle=21, blind_sec=None, max_sec=10, deg=90, min_sec=0.5):
    turn_speed = TURN_SPEED if speed is None else speed
    align_speed = ALIGN_SPEED if speed is None else speed
    if blind_sec is None:
        blind_sec = round(TURN_BLIND * TURN_BLIND_SPEED / max(turn_speed, TURN_BLIND_SPEED), 2)
    # 새 차선이 잡히거나, 카메라로 잰 회전각이 deg에 닿으면(lane_follow.TURN_YAW_PX) 회전을 끝낸다.
    max_sec = max(max_sec, min_sec + 1.0)     # min_sec이 길면 최대 시간도 늘린다 (최소 시간 뒤 1초는 새 차선/deg를 기다림).
    steps = [turn_fn(until="lane", angle=angle, min_sec=min_sec, max_sec=max_sec, speed=turn_speed, blind_sec=blind_sec, deg=deg),
             follow(until="aligned", max_sec=3, speed=align_speed, style="soft")]
    return steps + [follow(stop_sec, speed=SETTLE_SPEED)] if stop_sec > 0 else steps

def _expand(steps):
    out = []
    for s in steps:
        out.extend(_expand(s) if isinstance(s, list) else [s])
    return out


def _flat(steps):
    """묶음(list)을 펼치고, set_speed를 그 뒤 동작들의 속도에 반영한다."""
    out, cruise = [], None
    for s in _expand(steps):
        if s["do"] == "set_speed":
            cruise = s["speed"]
        else:
            out.append(dict(s, speed=cruise) if cruise is not None and s.get("auto") else s)
    return out


def describe(step):
    names = {"junction": "교차로", "crosswalk": "횡단보도", "lane": "새 차선", "gap": "차선 끊김", "end": "끝(교차로선/끊김)", "crossed": "횡단보도 지나감",
             "green": "초록불", "lost": "차선 놓침", "obstacle": "장애물", "clear": "장애물 없어짐",
             "aligned": "정렬"}
    do = {"follow": "차선 따라", "straight": "직진", "right": "우회전", "left": "좌회전", "curve": "곡선",
          "stop": "정지", "wait": "대기", "finish": "끝"}[step["do"]]
    parts = [do]
    if step["sec"]:
        parts.append(f"{step['sec']}초")
    elif step["until"]:
        parts.append(f"{names[step['until']]}까지" + (f"(화면 {step['at']})" if step["at"] else ""))
    for key, fmt in (("side", "{}쪽 선 기준"), ("shift", "{:+}cm 치우쳐"), ("steer", "최대 {}도"),
                     ("style", "{}"), ("speed", "속도 {}"), ("slow", "→ {}까지 감속"),
                     ("blind", "시작 {}초 감지 안 함"), ("min", "최소 {}초"), ("deg", "{}도 돌면 끝"), ("nth", "{}번째 선에서"), ("cm", "+{}cm")):
        if step.get(key) is not None and not (key == "speed" and step["do"] in ("stop", "wait")) \
                and not (key in ("blind", "min") and not step[key]):
            parts.append(fmt.format({"left": "왼", "right": "오른"}.get(step[key], step[key])))
    return " ".join(parts)













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
# wait_green(max_sec)       선 채로 초록불이 될 때까지 대기 (최대 max_sec초)
# start_gate(hold_sec)      출발선 앞 장애물이 치워지고 hold_sec초 지나면 출발
# finish()                  코스 끝, 정지
# set_speed(v)              이 뒤 주행 속도를 v로 (턴, straight, 직접 준 speed= 제외). set_speed()는 기본값으로 되돌림
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

    # set_speed(0.4),        
    # cross_gap(),
    # set_speed(0.6),        
    # to_crosswalk(0.8),
    # stop(3),
    # pass_crosswalk(),
    # to_junction(),
    # tr(),
    # to_junction(),
    # straight(3),
    # tl(),
    # follow(7),
#------------------------------------------------

    # to_end(0.6),
    # straight(0.5),
    # tr(2.5),
    # to_junction(),
    # straight(1.5),
    # tl(),
    # follow(3),



# --- 코스 ---

COURSE = _flat([
    start_gate(2),
    
    set_speed(0.4),    
    cross_gap(),
    set_speed(0.6),        
    to_junction(cm=7),
    tr(),
    to_junction(cm=7),
    tl(),
    to_end(max_sec=8),



    finish(),

 


])

if __name__ == "__main__":
    for i, step in enumerate(_flat(COURSE), 1):
        print(f"{i:2d}. {describe(step)}")



