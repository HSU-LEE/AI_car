
SPEED = 0.375

EVENTS = ("junction", "crosswalk", "lane", "gap", "green", "obstacle", "clear", "lost", "aligned", None)
SIDES = ("left", "right", "center", None)
STYLES = ("soft", "normal", "sharp", None)
OPTIONS = ("min_sec", "max_sec", "speed", "at", "side", "shift", "steer", "style", "hold_sec", "slow", "blind_sec")


def _hint(word, choices):
    import difflib
    near = difflib.get_close_matches(str(word), [str(c) for c in choices if c is not None], 1)
    return f' → "{near[0]}" 말인가요?' if near else ""


def _step(do, angle, sec, until, **opt):
    for k in opt:
        if k not in OPTIONS:
            raise ValueError(f"{do}(): 옵션 {k}= 는 없음{_hint(k, OPTIONS)} (쓸 수 있는 옵션: {', '.join(OPTIONS)})")
    if sec is not None and until is not None and do != "wait":
        for k in ("at", "slow"):
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
    if opt.get("slow") is not None and until not in ("junction", "crosswalk", "gap"):
        raise ValueError(f'{do}(): slow=는 until="junction" / "crosswalk" / "gap"일 때만 씀 (그 선에 가까워질수록 감속)')
    shift = opt.get("shift")
    if shift is not None and abs(shift) > 8:
        raise ValueError(f"{do}(): shift는 -8 ~ 8 (cm, +는 오른쪽). 차로 폭이 20cm라 그 이상은 선을 밟음")
    if sec is None and until is None and opt.get("max_sec") is None and do != "finish":
        raise ValueError(f"{do}(): sec, until, max_sec 중 하나는 필요")
    return {"do": do, "angle": angle, "sec": sec, "until": until,
            "min": opt.get("min_sec", 0.0), "max": opt.get("max_sec"), "speed": opt.get("speed"),
            "at": at, "hold": opt.get("hold_sec", 0.0), "side": opt.get("side"), "shift": shift,
            "steer": opt.get("steer"), "style": opt.get("style"), "slow": opt.get("slow"),
            "blind": opt.get("blind_sec") or 0.0}


def follow(sec=None, until="junction", speed=0.225, **opt):
    """차선을 따라가다가 until 조건이 되면 끝낸다."""
    return _step("follow", 0.0, sec, until, speed=speed, **opt)


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


def to_junction(at=None, slow=0.2, blind_sec=1.5, speed=SPEED):
    """다음 교차로 선 앞까지 가서 끝낸다."""
    return follow(until="junction", speed=speed, at=at, slow=slow, blind_sec=blind_sec)


def to_crosswalk(at=0.67, slow=0.2, speed=SPEED):
    """횡단보도 앞까지 가서 끝낸다."""
    return follow(until="crosswalk", speed=speed, at=at, slow=slow)


def to_gap(at=0.85, slow=0.2, speed=SPEED):
    """차선이 끊기는 곳 앞까지 가서 끝낸다."""
    return follow(until="gap", speed=speed, at=at, slow=slow)


def to_end(at=0.5, slow=0.18, blind_sec=0.0, speed=SPEED):
    """교차로가 아닌 바로 앞 끝 선 앞에서 끝낸다."""
    return follow(until="gap", speed=speed, at=at, slow=slow, blind_sec=blind_sec)


def _turn_fn(direction):
    """방향 이름을 right / left 동작 함수로 바꾼다."""
    if direction not in ("left", "right"):
        raise ValueError(f'방향은 "left" / "right"{_hint(direction, ("left", "right"))}')
    return right if direction == "right" else left


def cross_gap(at=0.85, angle=0, speed=SPEED):
    """끊긴 구간을 지나 다시 차선을 따라간다."""
    return [to_gap(at, speed=speed), straight(until="lane", angle=angle, max_sec=3, speed=speed)]


def traffic(at=0.73, max_sec=10, speed=SPEED):
    """횡단보도 앞에서 서서 초록불에 출발한다."""
    return [to_crosswalk(at, speed=speed), wait_green(max_sec=max_sec)]


def cross(at=None, line="junction", speed=SPEED):
    """교차로 선 앞까지 가서 정면으로 건너고, 반대편 차선이 잡히면 끝낸다. line="crosswalk"면 횡단보도를 건넌다."""
    if line not in ("junction", "crosswalk"):
        raise ValueError(f'cross(): line은 "junction" / "crosswalk"{_hint(line, ("junction", "crosswalk"))}')
    approach = to_junction(at, speed=speed) if line == "junction" else to_crosswalk(at or 0.67, speed=speed)
    return [approach, straight(until="lane", angle=0, min_sec=0.8, max_sec=4, speed=speed)]


def corner(direction, at=None, go_sec=0.5, wide=0, stop_sec=0.20, speed=SPEED):
    """교차로 선까지 가서 go_sec초 더 나간 뒤 직각으로 꺾고, 차로에 맞춰 정렬하고 선다.

    to_junction() → straight(go_sec) → tr()/tl() 과 같다. wide는 다가갈 때 회전 바깥쪽으로 치우칠 cm.
    """
    turn_fn = _turn_fn(direction)
    shift = wide if direction == "left" else -wide  # 좌회전의 바깥은 오른쪽.
    steps = [follow(until="junction", at=at, slow=0.2, blind_sec=1.5, speed=speed, shift=shift or None)]
    if go_sec > 0:
        steps.append(straight(go_sec))
    return steps + _turn_align_stop(turn_fn, stop_sec)


def uturn(direction, at=None, blind_sec=1.5, stop_sec=0.20, speed=0.2):
    """교차로 선 앞에서 최대 조향으로 돌아 반대 방향 차로로 들어가고, 정렬하고 선다.

    blind_sec 동안은 새 차선을 보지 않는다. 90° 쯤에서 보이는 가로선을 새 차선으로 착각하지 않게,
    차가 135° 이상 돌 시간으로 맞춘다 (실제 차에서 조절).
    """
    return [follow(until="junction", at=at, slow=speed, speed=speed),
            _turn_align_stop(_turn_fn(direction), stop_sec, speed, angle=None, blind_sec=blind_sec, max_sec=5)]


def tr(stop_sec=0.20, speed=None):
    """우회전한 뒤 차로에 맞춰 정렬하고 선다."""
    return _turn_align_stop(right, stop_sec, speed)


def tl(stop_sec=0.20, speed=None):
    """좌회전한 뒤 차로에 맞춰 정렬하고 선다."""
    return _turn_align_stop(left, stop_sec, speed)


TURN_SPEED = 0.35 
ALIGN_SPEED = 0.2  


def _turn_align_stop(turn_fn, stop_sec, speed=None, angle=21, blind_sec=1.0, max_sec=3):
    """회전 다음 정렬, 그다음 정지로 이어 붙인다. angle=None이면 최대 조향."""
    turn_speed = TURN_SPEED if speed is None else speed
    align_speed = ALIGN_SPEED if speed is None else speed
    steps = [turn_fn(until="lane", angle=angle, min_sec=0.5, max_sec=max_sec, speed=turn_speed, blind_sec=blind_sec),
             follow(until="aligned", max_sec=3, speed=align_speed, style="soft")]
    return steps + [stop(stop_sec)] if stop_sec > 0 else steps

def _flat(steps):
    out = []
    for s in steps:
        out.extend(_flat(s) if isinstance(s, list) else [s])
    return out


def describe(step):
    names = {"junction": "교차로", "crosswalk": "횡단보도", "lane": "새 차선", "gap": "차선 끊김",
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
                     ("blind", "시작 {}초 감지 안 함")):
        if step.get(key) is not None and not (key == "speed" and step["do"] in ("stop", "wait")) \
                and not (key == "blind" and not step[key]):
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
#
# [목표 지점까지 가기]
# to_junction()             다음 교차로 선 앞까지 감속하며 가서 끝 (시작 1.5초는 감지 안 함)
# to_crosswalk(at)          횡단보도 앞까지 감속하며 가서 끝
# to_gap(at)                차선이 끊기는 곳 앞까지 감속하며 가서 끝
# to_end(at)                교차로가 아닌 바로 앞 끝 선 앞에서 끝
#
# [묶음 동작 (여러 단계)]
# cross_gap(at, angle)      끊긴 곳까지 간 뒤 새 차선이 잡힐 때까지 직진
# traffic(at, max_sec)      횡단보도 앞에 서서 초록불에 출발
# cross(at, line)           교차로 선까지 가서 정면으로 건너 반대편 차선에서 끝 (line="crosswalk"면 횡단보도)
# corner(direction, wide)   교차로 선까지 가서 go_sec초 더 간 뒤 꺾고 정렬·정지 (to_junction + straight + tr/tl)
# uturn(direction)          교차로 선 앞에서 최대 조향으로 돌아 정렬·정지 (blind_sec 동안 새 차선 안 봄)
# tr(stop_sec)              우회전 → 차로에 맞춰 정렬 → 잠깐 정지
# tl(stop_sec)              좌회전 → 차로에 맞춰 정렬 → 잠깐 정지
#
# [공통 옵션]
# at=        감지 위치 (0.4~1.0, 클수록 가까이)     slow=      선에 가까워질수록 이 속도까지 감속
# speed=     주행 속도                              blind_sec= 시작 후 이 시간 동안 감지 안 함
# min_sec=   최소 진행 시간                         max_sec=   최대 진행 시간
# side=      기준 선 ("left"/"right"/"center")      shift=     차로 안에서 치우침 (cm, +오른쪽, ±8)
# style=     조향 성향 ("soft"/"normal"/"sharp")    steer=     최대 조향각
# hold_sec=  조건이 이 시간 동안 유지돼야 끝
# until=     "junction" "crosswalk" "lane" "gap" "green" "obstacle" "clear" "lost" "aligned"



# --- 코스 ---

COURSE = _flat([
    start_gate(2),

    cross_gap(0.85, speed=0.2),    
    to_junction(),
    straight(0.5),
    tr(),
    to_junction(),
    straight(1),
    tl(),
    follow(3),
    
    finish(),

 


])

if __name__ == "__main__":
    for i, step in enumerate(_flat(COURSE), 1):
        print(f"{i:2d}. {describe(step)}")



# --- 예시 코스 ---
#
# to_end(speed=0.24),
# straight(0.5),
# tr(),
# to_junction(),
# straight(0.5),
# tl(),
# follow(3),
# cross_gap(0.85, speed=0.2),
# follow(2),
# finish(),
#------------------------------------------------
# cross_gap(0.85, speed=0.2),
# to_junction(),
# straight(0.5),
# tr(),
# to_junction(),
# straight(1),
# tl(),
# follow(3),
# finish(),