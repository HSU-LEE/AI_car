
# --- 주행 ---
SPEED = 0.75                 # 기본 주행 속도. 코스 중간에 바꾸려면 set_speed(0.4)
STEER_MAX = None            # 차선 주행 최대 조향각 (도, 1~26). None = lane_follow.py AUTO_STEER_MAX(26). 코스 중간에 바꾸려면 set_steer(15)

# --- 신호등 ---
RED_BACK_ON = True          # wait_green에서 빨간불을 못 찾으면 후진해서 다시 찾을지 (True / False)

# --- 레고 사람 (등록: Tune Person.command) ---
PERSON_ON = True            # 사람을 보고 멈출지
PERSON_AT = 0.55            # 이 선(화면 높이 0.4~0.95)보다 발이 아래로 와야 사람으로 봄. 0.55 = 약 40 cm, 0.5 = 약 46 cm. 위쪽(먼 곳)은 무시. 아래쪽은 색만 맞으면 크기, 윤곽, 위치와 상관없이 바로 정지. None = 등록한 위치
PERSON_RESUME_DELAY = 3.5   # 사람이 전혀 안 보인 뒤 출발까지 기다리는 시간 (초). 그 사이 한 번이라도 보이면 다시 기다림
PERSON_MIN_STOP = 2.0       # 최소 정지 시간 (초)
PERSON_MAX_WAIT = None      # 사람이 있어도 출발하는 시간 (초). None = 없어질 때까지 대기
PERSON_X = (0.0, 1.0)       # 사람을 찾을 가로 범위: 왼쪽 0 ~ 오른쪽 1. 전체 = 화면에 잡히면 어디든
PERSON_IN_LANE = True       # True = 발이 양쪽 차선 사이에 있어야 사람으로 봄. 차선이 하나도 안 보이면 구분할 수 없어 어디든 인정
PERSON_ANY = False          # True = 등록한 레고 말고 다른 레고 사람도 찾음


EVENTS = ("junction", "crosswalk", "lane", "gap", "end", "crossed", "green", "obstacle", "clear", "lost", "aligned", "enter",
          "_crosswalk", "_crossed", "_stop", None)
SIDES = ("left", "right", "center", None)
STYLES = ("soft", "normal", "sharp", None)
OPTIONS = ("min_sec", "max_sec", "speed", "at", "side", "shift", "steer", "style", "hold_sec", "slow", "blind_sec", "deg", "nth", "cm", "min_speed")


def _hint(word, choices):
    import difflib
    near = difflib.get_close_matches(str(word), [str(c) for c in choices if c is not None], 1)
    return f' -> did you mean "{near[0]}"?' if near else ""


def _check_config():
    if STEER_MAX is not None and (isinstance(STEER_MAX, bool) or not isinstance(STEER_MAX, (int, float)) or not 0 < STEER_MAX <= 90):
        raise ValueError("STEER_MAX must be more than 0 and at most 90 (degrees), or None")
    if PERSON_AT is not None and not 0.4 <= PERSON_AT <= 0.95:
        raise ValueError("PERSON_AT must be 0.4 to 0.95 (larger is nearer) or None")
    if not 0 <= PERSON_X[0] < PERSON_X[1] <= 1:
        raise ValueError("PERSON_X must be (left, right), 0 to 1")
    for name, value in (("PERSON_RESUME_DELAY", PERSON_RESUME_DELAY),
                        ("PERSON_MIN_STOP", PERSON_MIN_STOP), ("PERSON_MAX_WAIT", PERSON_MAX_WAIT or 0)):
        if value < 0:
            raise ValueError(f"{name} must be 0 or more seconds")


_check_config()


def _bad_num(value, lo, hi):
    return isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi


def _step(do, angle, sec, until, **opt):
    for k in opt:
        if k not in OPTIONS:
            raise ValueError(f"{do}(): no option {k}={_hint(k, OPTIONS)} (options: {', '.join(OPTIONS)})")
    if sec is not None and until is not None and do != "wait":
        for k in ("at", "slow", "cm"):
            if opt.get(k) is not None:
                raise ValueError(f"{do}(): {k}= goes with until, so it cannot be used with sec")
        raise ValueError(f"{do}(): sec and until cannot be used together (to end on time, leave out until)")
    if until not in EVENTS:
        raise ValueError(f'{do}(): no until="{until}"{_hint(until, EVENTS)}')
    if opt.get("side") not in SIDES:
        raise ValueError(f'{do}(): side must be "left" / "right" / "center"{_hint(opt.get("side"), SIDES)}')
    if opt.get("style") not in STYLES:
        raise ValueError(f'{do}(): style must be "soft" / "normal" / "sharp"{_hint(opt.get("style"), STYLES)}')
    at = opt.get("at")
    if at is not None and _bad_num(at, 0.4, 1.0):
        raise ValueError(f"{do}(): at must be 0.4 to 1.0 (larger is nearer)")
    if at is not None and at > 0.85 and until in ("crosswalk", "_crosswalk"):
        print(f"[warning] {do}(until=\"crosswalk\", at={at}): above 0.85 the crosswalk may not be seen and the action may not end (0.67 recommended)")
    if opt.get("slow") is not None and (_bad_num(opt["slow"], 0.0, 1.0) or until not in ("junction", "crosswalk", "gap", "end", "_crosswalk", "_stop")):
        if until not in ("junction", "crosswalk", "gap", "end", "_crosswalk", "_stop"):
            raise ValueError(f'{do}(): slow= is only for until="junction" / "crosswalk" / "gap" / "end"')
        raise ValueError(f"{do}(): slow must be 0 to 1")
    nth = opt.get("nth")
    if nth is not None and (isinstance(nth, bool) or not isinstance(nth, int) or nth < 1):
        raise ValueError(f"{do}(): nth must be an integer of 1 or more")
    cm = opt.get("cm")
    if cm is not None and until not in ("junction", "end"):
        raise ValueError(f'{do}(): cm= is only for until="junction" / "end"')
    if cm is not None and _bad_num(cm, 0, 60):
        raise ValueError(f"{do}(): cm must be 0 to 60 (to_junction: the LANE END distance on screen, to_end: distance past the base)")
    shift = opt.get("shift")
    if shift is not None and _bad_num(shift, -8, 8):
        raise ValueError(f"{do}(): shift must be -8 to 8 (cm, + is right)")
    if sec is not None and _bad_num(sec, 0, 600):
        raise ValueError(f"{do}(): sec must be 0 to 600")
    speed = opt.get("speed")
    if speed is not None and _bad_num(speed, 0, 1):
        raise ValueError(f"{do}(): speed must be 0 to 1")
    min_speed = opt.get("min_speed")
    if min_speed is not None and _bad_num(min_speed, 0, 1):
        raise ValueError(f"{do}(): min_speed must be 0 to 1")
    min_sec = 0.0 if opt.get("min_sec") is None else opt.get("min_sec")
    max_sec = opt.get("max_sec")
    hold = 0.0 if opt.get("hold_sec") is None else opt.get("hold_sec")
    blind = 0.0 if opt.get("blind_sec") is None else opt.get("blind_sec")
    deg = opt.get("deg")
    steer = opt.get("steer")
    slow = opt.get("slow")
    for name, value, lo, hi in (("min_sec", min_sec, 0, 600), ("max_sec", max_sec, 0, 600),
                                ("hold_sec", hold, 0, 600), ("blind_sec", blind, 0, 600),
                                ("deg", deg, 0, 360), ("steer", steer, 0, 90)):
        if value is not None and _bad_num(value, lo, hi):
            raise ValueError(f"{do}(): {name} must be {lo:g} to {hi:g}")
    if deg is not None and deg <= 0:
        raise ValueError(f"{do}(): deg must be more than 0")
    if max_sec is not None and min_sec > max_sec:
        raise ValueError(f"{do}(): min_sec ({min_sec:g}) is larger than max_sec ({max_sec:g})")
    if sec is None and until is None and max_sec is None and do != "finish":
        raise ValueError(f"{do}(): needs one of sec, until, max_sec")
    return {"do": do, "angle": angle, "sec": sec, "until": until,
            "min": min_sec, "max": max_sec, "speed": speed,
            "at": at, "hold": hold, "side": opt.get("side"), "shift": shift,
            "steer": steer, "style": opt.get("style"), "slow": slow,
            "blind": blind, "deg": deg, "nth": nth, "cm": cm, "min_speed": min_speed}


def _auto(step, speed, default):
    return dict(step, speed=default, auto=True) if speed is None else step


def set_steer(deg=None):
    if deg is not None and (isinstance(deg, bool) or not isinstance(deg, (int, float)) or not 0 < deg <= 90):
        raise ValueError(f"set_steer(): deg must be more than 0 and at most 90 ({deg})")
    return {"do": "set_steer", "steer": deg}


def set_speed(speed=None):
    if speed is not None and (isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0 < speed <= 1):
        raise ValueError(f"set_speed(): speed must be more than 0 and at most 1 ({speed})")
    return {"do": "set_speed", "speed": speed}


_MISSING = object()


def _timed_until(sec, until, default):
    if until is _MISSING:
        return None if sec is not None else default
    return until


def _steer_angle(do, angle):
    if angle is None:
        return None
    if _bad_num(angle, -90, 90):
        raise ValueError(f"{do}(): angle must be -90 to 90")
    return angle


def follow(sec=None, until=_MISSING, speed=None, **opt):
    return _auto(_step("follow", 0.0, sec, _timed_until(sec, until, "junction"), speed=speed, **opt), speed, SPEED)


def straight(sec=None, until=None, angle="hold", speed=None, **opt):
    if angle != "hold":
        angle = _steer_angle("straight", angle)
    return _step("straight", angle, sec, until, speed=speed, **opt)

def push(sec=None, until=None, angle="hold", speed=None, **opt):
    # 속도: speed= > set_speed() > SPEED. 주행 내내 그 속도를 유지
    if angle != "hold":
        angle = _steer_angle("straight", angle)
    return _auto(_step("straight", angle, sec, until, speed=speed, **opt), speed, SPEED)

def right(sec=None, angle=None, until=_MISSING, speed=None, **opt):
    if opt.get("min_sec") is None:
        opt["min_sec"] = 0.5
    if opt.get("max_sec") is None:
        min_sec = opt["min_sec"]
        longer = min_sec + 1.0 if isinstance(min_sec, (int, float)) and not isinstance(min_sec, bool) else 0
        opt["max_sec"] = max(3.0, longer)
    angle = _steer_angle("right", angle)
    return _step("right", None if angle is None else abs(angle), sec, _timed_until(sec, until, "lane"), speed=speed, **opt)


def left(sec=None, angle=None, until=_MISSING, speed=None, **opt):
    if opt.get("min_sec") is None:
        opt["min_sec"] = 0.5
    if opt.get("max_sec") is None:
        min_sec = opt["min_sec"]
        longer = min_sec + 1.0 if isinstance(min_sec, (int, float)) and not isinstance(min_sec, bool) else 0
        opt["max_sec"] = max(3.0, longer)
    angle = _steer_angle("left", angle)
    return _step("left", None if angle is None else -abs(angle), sec, _timed_until(sec, until, "lane"), speed=speed, **opt)


def turn(direction, **opt):
    if direction not in ("left", "right"):
        raise ValueError(f'turn(): direction must be "left" / "right"{_hint(direction, ("left", "right"))}')
    return (right if direction == "right" else left)(**opt)


def curve(angle, sec=None, until=None, speed=None, **opt):
    if angle is None or _bad_num(angle, -90, 90):
        raise ValueError(f"curve(): angle must be -90 to 90 ({angle})")
    return _step("curve", angle, sec, until, speed=speed, **opt)


def stop(sec, speed=0.0):
    return _step("stop", 0.0, sec, None, speed=speed)


def wait_green(max_sec=10, min_sec=0.0, speed=0.0):
    return _step("wait", 0.0, None, "green", min_sec=min_sec, max_sec=max_sec, speed=speed)


def wait(sec=None, until="clear", min_sec=0.0, max_sec=None, hold_sec=0.0, speed=0.0):
    return _step("wait", 0.0, sec, until, min_sec=min_sec, max_sec=max_sec, speed=speed, hold_sec=hold_sec)


def next():
    return _step("wait", 0.0, None, "enter", speed=0.0)


def start_gate(hold_sec=1.0, speed=0.0):
    return wait(until="clear", hold_sec=hold_sec, speed=speed)


def finish():
    return _step("finish", 0.0, 0, None)


def to_junction(at=None, slow=0.3, blind_sec=0.5, speed=None, nth=1, cm=None, steer=None, sec=None, min_speed=None, shift=None):
    if nth is None:
        nth = 1
    if isinstance(nth, bool) or not isinstance(nth, int) or nth < 1:
        raise ValueError(f"to_junction(): nth must be an integer of 1 or more ({nth})")
    if sec is not None and _bad_num(sec, 0, 600):
        raise ValueError(f"to_junction(): sec must be 0 to 600 ({sec})")
    step = _auto(follow(until="junction", shift=shift, speed=speed, at=at, slow=slow, blind_sec=blind_sec, nth=nth if nth > 1 else None, cm=cm, steer=steer, max_sec=sec, min_speed=min_speed), speed, SPEED)
    # cm, at 둘 다 없으면 lane_follow.py LANEEND_BASE_CM(기본 10cm)을 주행 시작 때 채움 (여기서 lane_follow를 부르면 순환 import)
    return dict(step, cm_default=True) if cm is None and at is None else step

def go(at=None, slow=0.3, blind_sec=0.5, speed=None, nth=1, cm=None, steer=None, sec=None, min_speed=None, shift=None):
    if nth is None:
        nth = 1
    if isinstance(nth, bool) or not isinstance(nth, int) or nth < 1:
        raise ValueError(f"go(): nth must be an integer of 1 or more ({nth})")
    if sec is not None and _bad_num(sec, 0, 600):
        raise ValueError(f"go(): sec must be 0 to 600 ({sec})")
    step = _auto(follow(until="junction", shift=shift, speed=speed, at=at, slow=slow, blind_sec=blind_sec, nth=nth if nth > 1 else None, cm=cm, steer=steer, max_sec=sec, min_speed=min_speed), speed, SPEED)
    # cm, at 둘 다 없으면 lane_follow.py LANEEND_BASE_CM(기본 10cm)을 주행 시작 때 채움 (여기서 lane_follow를 부르면 순환 import)
    return dict(step, cm_default=True) if cm is None and at is None else step


def to_crosswalk(at=0.67, slow=0.3, speed=None, steer=None, sec=None, min_speed=None, shift=None):
    # sec=3: 횡단보도에서 멈추거나 3초가 지나면 끝 (먼저 오는 쪽). steer=15: 최대 조향각
    if sec is not None and _bad_num(sec, 0, 600):
        raise ValueError(f"to_crosswalk(): sec must be 0 to 600 ({sec})")
    return _auto(follow(until="crosswalk", shift=shift, speed=speed, at=at, slow=slow, steer=steer, max_sec=sec, min_speed=min_speed), speed, SPEED)


def pass_crosswalk():
    return cross(line="crosswalk", approach=False)


# 등록한 횡단보도(aicar crosswalk)로 판단하는 버전. 위의 to_crosswalk / pass_crosswalk는 그대로 두고 함께 쓸 수 있음
def _to_crosswalk(at=0.67, slow=0.3, speed=None, steer=None, sec=None, min_speed=None, shift=None):
    # 등록한 횡단보도가 at 높이까지 오면 끝. sec=3: 횡단보도에서 멈추거나 3초가 지나면 끝 (먼저 오는 쪽). steer=15: 최대 조향각
    if sec is not None and _bad_num(sec, 0, 600):
        raise ValueError(f"_to_crosswalk(): sec must be 0 to 600 ({sec})")
    return _auto(follow(until="_crosswalk", shift=shift, speed=speed, at=at, slow=slow, steer=steer, max_sec=sec, min_speed=min_speed), speed, SPEED)


def go_to(name, offset=0, nth=1, slow=None, speed=None, steer=None, sec=None, min_speed=None, blind_sec=0.5, shift=None):
    # 등록한 정지 지점(aicar stoppoint, 메뉴 t)까지 가서 정지. 차를 그 자리에 놓고 등록한 모양이 등록 때 위치에 오면 끝
    # offset=3: 등록 위치보다 3cm 더 가서 / -3: 덜 가서.  nth=2: 같은 모양을 하나 지나치고 두 번째에서
    # sec=10: 못 찾아도 10초가 지나면 끝.  감속 없이 지정 속도로 가서 정지 (slow=0.3을 주면 다가가며 감속). steer, speed, min_speed, blind_sec는 go()와 같음
    # 여러 지점 중 먼저 오는 곳에서 정지: go_to("A|B") 또는 go_to(["A", "B"]) (소문자도 됨: "a|b")
    names = name.split("|") if isinstance(name, str) else list(name) if isinstance(name, (list, tuple)) else None
    if not names or not all(isinstance(n, str) and len(n.strip()) == 1 and n.strip().upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for n in names):
        raise ValueError(f'go_to(): stop points are letters A-Z, such as "A" or "A|B" ({name!r})')
    names = [n.strip().upper() for n in names]
    name = names[0] if len(names) == 1 else names
    if _bad_num(offset, -30, 30):
        raise ValueError(f"go_to(): offset must be -30 to 30 cm ({offset})")
    if isinstance(nth, bool) or not isinstance(nth, int) or nth < 1:
        raise ValueError(f"go_to(): nth must be an integer of 1 or more ({nth})")
    if sec is not None and _bad_num(sec, 0, 600):
        raise ValueError(f"go_to(): sec must be 0 to 600 ({sec})")
    step = follow(until="_stop", shift=shift, speed=speed, slow=slow, steer=steer, max_sec=sec, min_speed=min_speed,
                  blind_sec=blind_sec, nth=nth if nth > 1 else None)
    return _auto(dict(step, stop=name, stop_cm=offset), speed, SPEED)


def _pass_crosswalk(speed=None, steer=None):
    # 등록한 횡단보도를 지나갈 때까지 차선 따라 주행 (화면에서 사라지면 끝, 최대 5초)
    return _auto(follow(until="_crossed", hold_sec=0.3, min_sec=0.5, max_sec=5, speed=speed, steer=steer), speed, SPEED)

def to_gap(at=0.85, slow=0.3, speed=None, steer=None, min_speed=None, shift=None):
    return _auto(follow(until="gap", shift=shift, speed=speed, at=at, slow=slow, steer=steer, min_speed=min_speed), speed, SPEED)


def to_end(at=0.5, slow=0.3, blind_sec=0.0, speed=None, shift=None, cm=None, max_sec=None):
    return _auto(follow(until="end", speed=speed, at=at, slow=slow, blind_sec=blind_sec, shift=shift, cm=cm, max_sec=max_sec), speed, 0.3)


def _turn_fn(direction):
    if direction not in ("left", "right"):
        raise ValueError(f'direction must be "left" / "right"{_hint(direction, ("left", "right"))}')
    return right if direction == "right" else left


def _straight_speed(speed):
    return SPEED if speed is None else speed


def cross_gap(at=0.85, angle=0, speed=None, steer=None, sec=None):
    # sec=None: 끊긴 곳을 새 차선이 잡힐 때까지 직진 (최대 3초). sec=1.5: 1.5초 동안 직진
    # steer=15: 끊긴 곳까지 가는 동안과 직진하는 동안 최대 조향각 15도
    # 속도: speed= > set_speed() > SPEED. 끊긴 곳까지 가는 동안과 직진하는 동안 모두
    cross = (straight(sec, angle=angle, speed=speed, steer=steer) if sec is not None
             else straight(until="lane", angle=angle, max_sec=3, speed=speed, steer=steer))
    return [to_gap(at, speed=speed, steer=steer), _auto(cross, speed, SPEED)]


def traffic(at=0.73, max_sec=10, speed=None):
    return [to_crosswalk(at, speed=speed), wait_green(max_sec=max_sec)]


def cross(at=None, line="junction", speed=None, approach=True):
    if line not in ("junction", "crosswalk"):
        raise ValueError(f'cross(): line must be "junction" / "crosswalk"{_hint(line, ("junction", "crosswalk"))}')
    if line == "junction":
        steps = [to_junction(at, speed=speed)] if approach else []
        return steps + [straight(until="lane", angle=0, min_sec=0.8, max_sec=4, speed=_straight_speed(speed))]
    steps = [to_crosswalk(at or 0.67, speed=speed)] if approach else []
    return steps + [_auto(follow(until="crossed", hold_sec=0.3, min_sec=0.5, max_sec=5, speed=speed), speed, SPEED)]


def corner(direction, at=None, go_sec=0.5, wide=0, stop_sec=0.20, speed=None):
    if _bad_num(go_sec, 0, 600):
        raise ValueError(f"corner(): go_sec must be 0 to 600 ({go_sec})")
    if _bad_num(wide, -8, 8):
        raise ValueError(f"corner(): wide must be -8 to 8 cm ({wide})")
    turn_fn = _turn_fn(direction)
    shift = wide if direction == "left" else -wide
    steps = [_auto(follow(until="junction", at=at, slow=0.3, blind_sec=1.5, speed=speed, shift=shift or None), speed, SPEED)]
    if go_sec > 0:
        steps.append(straight(go_sec))
    return steps + _turn_align_stop(turn_fn, stop_sec)


def uturn(direction, at=None, blind_sec=1.5, stop_sec=0.20, speed=None):
    turn_speed = 0.4 if speed is None else speed
    return [_auto(follow(until="junction", at=at, slow=turn_speed, speed=speed), speed, turn_speed),
            _turn_align_stop(_turn_fn(direction), stop_sec, turn_speed, angle=None, blind_sec=blind_sec, max_sec=5, deg=180)]


def tr(min_sec=2.5, sec=0.50, speed=None, deg=90, align=True):
    return _cruise(_turn_align_stop(right, sec, speed, angle=TR_ANGLE, deg=deg, min_sec=min_sec, align=align), speed)


def tl(min_sec=2.5, sec=0.50, speed=None, deg=90, align=True):
    return _cruise(_turn_align_stop(left, sec, speed, angle=TL_ANGLE, deg=deg, min_sec=min_sec, align=align), speed)


TURN_SPEED = 0.35
TR_ANGLE = 21
TL_ANGLE = 26
ALIGN_SPEED = 0.4
SETTLE_SPEED = 0.3
TURN_BLIND = 1.0
TURN_BLIND_SPEED = 0.35


def _cruise(steps, speed):
    # tr()/tl() 속도: speed= > set_speed() > 기본(TURN_SPEED, ALIGN_SPEED, SETTLE_SPEED). set_speed()가 있으면 회전, 정렬, 천천히 주행 모두 그 속도
    if speed is not None:
        return steps
    return [dict(s, auto=True, turn_part=True, turn_blind=s["do"] in ("left", "right")) for s in steps]


def _turn_align_stop(turn_fn, stop_sec, speed=None, angle=21, blind_sec=None, max_sec=10, deg=90, min_sec=0.5, align=True):
    # align=False: 회전이 끝나면 차로 가운데 맞추기(정렬) 없이 바로 sec초 주행
    if _bad_num(stop_sec, 0, 600):
        raise ValueError(f"settle time after aligning must be 0 to 600 ({stop_sec})")
    turn_speed = TURN_SPEED if speed is None else speed
    align_speed = ALIGN_SPEED if speed is None else speed
    if blind_sec is None:
        blind_sec = round(TURN_BLIND * TURN_BLIND_SPEED / max(turn_speed, TURN_BLIND_SPEED), 2)
    max_sec = max(max_sec, min_sec + 1.0)
    if not isinstance(align, bool):
        raise ValueError(f"align must be True or False ({align})")
    steps = [turn_fn(until="lane", angle=angle, min_sec=min_sec, max_sec=max_sec, speed=turn_speed, blind_sec=blind_sec, deg=deg)]
    if align:
        steps.append(follow(until="aligned", max_sec=3, speed=align_speed, style="soft"))
    return steps + [follow(stop_sec, speed=SETTLE_SPEED)] if stop_sec > 0 else steps

def _expand(steps):
    out = []
    for s in steps:
        out.extend(_expand(s) if isinstance(s, list) else [s])
    return out


def _flat(steps):
    out, cruise, steer = [], None, STEER_MAX
    for s in _expand(steps):
        if s["do"] == "set_speed":
            cruise = s["speed"]
        elif s["do"] == "set_steer":
            steer = STEER_MAX if s["steer"] is None else s["steer"]
        else:
            if cruise is not None and s.get("auto"):
                s = dict(s, speed=cruise)
                if s.get("turn_blind"):     # A faster turn sweeps past the old lane sooner, so it looks away for less time.
                    s["blind"] = round(TURN_BLIND * TURN_BLIND_SPEED / max(cruise, TURN_BLIND_SPEED), 2)
            if steer is not None and s.get("auto") and s["do"] == "follow" and s.get("steer") is None and not s.get("turn_part"):
                s = dict(s, steer=steer)
            out.append(s)
    return out


def describe(step):
    names = {"junction": "junction", "crosswalk": "crosswalk", "lane": "new lane", "gap": "lane gap", "end": "end (junction line / gap)", "crossed": "crosswalk passed", "_crosswalk": "registered crosswalk", "_crossed": "registered crosswalk passed", "_stop": "stop point",
             "green": "green light", "lost": "lane lost", "obstacle": "obstacle", "clear": "obstacle cleared",
             "aligned": "aligned", "enter": "Enter"}
    do = {"follow": "follow lane", "straight": "straight", "right": "turn right", "left": "turn left", "curve": "curve",
          "stop": "stop", "wait": "wait", "finish": "finish"}[step["do"]]
    parts = [do]
    where = f"until {names[step['until']]}" + (f" (screen {step['at']})" if step["at"] else "") if step["until"] else ""
    if step["sec"] and where and step["do"] == "wait":
        parts.append(f"{step['sec']}s or {where}")
    elif step["sec"]:
        parts.append(f"{step['sec']}s")
    elif where:
        parts.append(where)
    for key, fmt in (("side", "on the {} line"), ("shift", "{:+}cm off center"), ("steer", "max {} deg"),
                     ("style", "{}"), ("speed", "speed {}"), ("slow", "slow to {}"),
                     ("blind", "ignore first {}s"), ("min", "at least {}s"), ("deg", "end after {} deg"), ("nth", "at line #{}"), ("cm", "+{}cm")):
        if step.get(key) is not None and not (key == "speed" and step["do"] in ("stop", "wait")) \
                and not (key in ("blind", "min") and not step[key]):
            parts.append(fmt.format(step[key]))
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
# wait(until="clear")       선 채로 앞 장애물이 없어질 때까지 대기. sec=를 같이 주면 그 시간 또는 조건 중 먼저 온 쪽
# wait_green(max_sec)       선 채로 초록불이 될 때까지 대기 (빨간불을 봤으면 무제한. 못 찾으면 1초씩 최대 2번 후진해 찾고, 그래도 없으면 max_sec초 뒤 출발)
# start_gate(hold_sec)      출발선 앞 장애물이 치워지고 hold_sec초 지나면 출발
# finish()                  코스 끝, 정지
# set_speed(v)              이 뒤 주행 속도를 v로 (턴, straight, 직접 준 speed= 제외). set_speed()는 기본값으로 되돌림
# set_steer(deg)            이 뒤 차선 주행 최대 조향각을 deg도로 (턴, straight, 직접 준 steer= 제외). set_steer()는 STEER_MAX로 되돌림
# 
# [목표 지점까지 가기]
# to_junction(at, nth)      다음 교차로 선 앞까지 감속하며 가서 끝 (nth=2: 첫 선은 지나치고 두 번째에서). 아래까지 보였다가 사라져도 끝
# to_crosswalk(at)          횡단보도 앞까지 감속하며 가서 끝
# to_gap(at)                차선이 끊기는 곳 앞까지 감속하며 가서 끝
# to_end(at)                바로 앞 끝에서 끝: 교차로선(가로선)이든 차선 끊김이든 먼저 오는 쪽
#                           go(sec=3) / to_junction(sec=3): 교차로에서 멈추거나 3초가 지나면 끝 (먼저 오는 쪽)
#                           to_junction(cm=25): 화면의 LANE END 거리가 25cm가 되면 끝 (차로 선이 교차로에서 끝나는 곳까지 거리). cm 없으면 10
#                                            거리는 차 앞쪽 기준. 4.5보다 작게 주면 선이 화면에서 안 보여 4.5에서 끝냄. 화면 값과 줄자 값이 다르면 lane_follow.py LANEEND_OFFSET_CM
#                           to_end(cm=10): 선을 자로 재서 기준 위치에서 10cm 더 간 뒤 끝. cm=0은 선이 카메라 앞 38cm일 때
# pass_crosswalk            횡단보도 지나가기 
# _to_crosswalk(at) / _pass_crosswalk()   위와 같지만 등록한 횡단보도(aicar crosswalk, 메뉴 c)로 판단. 기존 것과 함께 쓸 수 있음
# go_to("A", offset)        등록한 정지 지점 A(aicar stoppoint, 메뉴 t)까지 가서 정지. offset=3: 3cm 더 가서
#                           go_to("A|B"): A 또는 B 중 먼저 오는 곳에서 정지 (소문자 "a|b"도 됨)
# [묶음 동작 (여러 단계)]
# cross_gap(at, angle)      끊긴 곳까지 간 뒤 새 차선이 잡힐 때까지 직진 (sec=1.5: 1.5초만 직진, steer=15: 최대 조향각)
# traffic(at, max_sec)      횡단보도 앞에 서서 초록불에 출발
# cross(at, line, approach) 교차로선까지 가서 정면으로 건넘 / line="crosswalk": 차선 따라 횡단보도 건넘 (approach=False: 이미 앞에 서 있을 때)
# corner(direction, wide)   교차로 선까지 가서 go_sec초 더 간 뒤 꺾고 정렬·천천히 주행 (to_junction + straight + tr/tl)
# uturn(direction)          교차로 선 앞에서 최대 조향으로 돌아 정렬·천천히 주행 (blind_sec 동안 새 차선 안 봄)
# tr(min_sec, sec, deg)     우회전 → 차로에 맞춰 정렬 → sec초 천천히(0.3) 주행 (deg: 이만큼 돌면 끝, min_sec: 최소 이 시간은 돎)
# tl(min_sec, sec, deg)     좌회전 → 차로에 맞춰 정렬 → sec초 천천히(0.3) 주행 (deg: 이만큼 돌면 끝, min_sec: 최소 이 시간은 돎)
#                           tr(align=False) / tl(align=False): 회전 뒤 차로 가운데 맞추기(정렬)를 건너뜀
#                           tr()/tl() 속도: set_speed()가 있으면 회전·정렬·주행 모두 그 속도. 없으면 회전 0.35, 정렬 0.4, 주행 0.3
#
# [공통 옵션]
# at=        감지 위치 (0.4~1.0, 클수록 가까이)     slow=      선에 가까워질수록 이 속도까지 감속
# speed=     주행 속도                              blind_sec= 시작 후 이 시간 동안 감지 안 함
# min_sec=   최소 진행 시간                         max_sec=   최대 진행 시간
# side=      기준 선 ("left"/"right"/"center")      shift=     차로 안에서 치우침 (cm, +오른쪽, ±8)
# style=     조향 성향 ("soft"/"normal"/"sharp")    steer=     최대 조향각
# min_speed= 감속(조향, slow=, 차선 놓침)해도 이 속도 밑으로 안 내려감 (예: min_speed=0.425)
# hold_sec=  조건이 이 시간 동안 유지돼야 끝
# until=     "junction" "crosswalk" "lane" "gap" "end" "crossed" "green" "obstacle" "clear" "lost" "aligned"

# --- 예시 코스 ---

    # to_junction(cm=7),
    # straight(2.5),
    # tr(),
    # follow(4),
    # to_crosswalk(0.8),
    # follow(1.5),
    # wait_green(max_sec=20),
    # stop(1.5),
    # pass_crosswalk(),
    # to_junction(cm=7),
    # straight(2.5),
    # tr(),
    # to_junction(cm=7),
    # straight(2.5),
    # follow(4),
    # set_speed(0.4),
    # to_junction(cm=7),
    # tr(),
    # to_junction(),
    # straight(4),
    # to_crosswalk(0.85),

# -------------






# --- 코스 ---

COURSE = _flat([
    start_gate(2),
    go(steer=8,shift=6),

    finish(),
 


])

if __name__ == "__main__":
    for i, step in enumerate(_flat(COURSE), 1):
        print(f"{i:2d}. {describe(step)}")