
REFLECT_MOTION_ON = False
STOP_POINTS_ON = False
SLOWDOWN_ON = False
SPEED = 0.95
STEER_MAX = 12
JUNCTION_TRACK_ASSIST = True
LANE_POINTS = "pairs" 

RED_BACK_ON = False

PERSON_ON = True
PERSON_AT = 0.55
PERSON_RESUME_DELAY = 2
PERSON_MIN_STOP = 2.0
PERSON_MAX_WAIT = None
PERSON_X = (0.0, 1.0)
PERSON_IN_LANE = True
PERSON_ANY = False


EVENTS = ("junction", "crosswalk", "lane", "gap", "end", "crossed", "green", "obstacle", "clear", "lost", "aligned", "enter",
          "_crosswalk", "_crossed", "_stop", None)
SIDES = ("left", "right", "center", None)
STYLES = ("soft", "normal", "sharp", None)
OPTIONS = ("min_sec", "max_sec", "speed", "at", "side", "shift", "steer", "style", "hold_sec", "slow", "blind_sec", "deg", "nth", "gap", "cm", "min_speed")

# --- VS Code 자동완성용 (Pylance가 읽음). 실행에는 영향 없음 ---
# until=, side=, style= 등에서 따옴표를 열면 가능한 값이 목록으로 나오고, 괄호 안에서 공통 옵션이 자동완성됨
from typing import TYPE_CHECKING, Literal  # noqa: E402

Until = Literal["junction", "crosswalk", "lane", "gap", "end", "crossed", "green", "obstacle", "clear", "lost", "aligned", "enter",
                "_crosswalk", "_crossed", "_stop"]
Side = Literal["left", "right", "center"]
Style = Literal["soft", "normal", "sharp"]
Direction = Literal["left", "right"]

if TYPE_CHECKING:
    from typing import TypedDict, Union
    from typing_extensions import Unpack

    class Opt(TypedDict, total=False):
        """공통 옵션 (follow, straight, push, right, left, curve, turn에서 자동완성)"""
        at: float           # 감지 위치 0.4~1.0, 클수록 가까이
        slow: float         # 선에 가까워질수록 이 속도까지 감속 (SLOWDOWN_ON = True일 때만)
        blind_sec: float    # 시작 후 이 시간 동안 감지 안 함
        min_sec: float      # 최소 진행 시간
        max_sec: float      # 최대 진행 시간
        side: Side          # 기준 선
        shift: float        # 차로 안에서 치우침 (cm, +오른쪽, ±8)
        style: Style        # 조향 성향
        steer: float        # 최대 조향각
        nth: int            # n번째에서 끝 (앞의 것은 지나침)
        gap: Union[float, list]     # 지나친 뒤 다음 인식까지 쉬는 초
        min_speed: float    # 감속해도 이 속도 밑으로 안 내려감
        hold_sec: float     # 조건이 이 시간 동안 유지돼야 끝
        cm: float           # 끝선/차선 끝까지 거리 (cm)
        deg: float          # 이만큼 돌면 끝 (회전)

    class TurnOpt(Opt, total=False):
        sec: float
        angle: float
        until: Until
        speed: float


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
    gap = _gap(opt.get("gap"), nth, do)
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
            "blind": blind, "deg": deg, "nth": nth, "gap": gap, "cm": cm, "min_speed": min_speed}


def _auto(step, speed, default):
    return dict(step, speed=default, auto=True) if speed is None else dict(step, own_speed=True)


def set_steer(deg=None):
    """이 뒤 차선 주행 최대 조향각을 deg도로 (턴, straight, 직접 준 steer= 제외).

    set_steer()는 STEER_MAX로 되돌림. speed=를 직접 준 차선 주행에도 적용 (tr/tl 안 제외)."""
    if deg is not None and (isinstance(deg, bool) or not isinstance(deg, (int, float)) or not 0 < deg <= 90):
        raise ValueError(f"set_steer(): deg must be more than 0 and at most 90 ({deg})")
    return {"do": "set_steer", "steer": deg}


def set_style(style: "Style | None" = None):
    """이 뒤 차선 주행 조향 성향을 "soft"/"normal"/"sharp"로 (턴, straight, 직접 준 style= 제외).

    set_style()은 기본으로 되돌림. speed=를 직접 준 차선 주행에도 적용 (tr/tl 안 제외)."""
    if style not in STYLES:
        raise ValueError(f'set_style(): style must be "soft" / "normal" / "sharp"{_hint(style, STYLES)}')
    return {"do": "set_style", "style": style}


def set_speed(speed=None):
    """이 뒤 주행 속도를 speed로 (턴, straight, 직접 준 speed= 제외). set_speed()는 기본값으로 되돌림.

    예: set_speed(0.5)"""
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


def follow(sec=None, until: "Until | None" = _MISSING, speed=None, **opt: "Unpack[Opt]"):
    """차선을 따라가다 sec초가 지나거나 until 조건이 되면 끝.

    sec만 주면 시간으로 끝, 둘 다 없으면 until="junction".
    예: follow(3), follow(until="crosswalk")"""
    return _auto(_step("follow", 0.0, sec, _timed_until(sec, until, "junction"), speed=speed, **opt), speed, SPEED)


def back(sec, speed=None, angle=0):
    """sec초 동안 후진 (차선은 안 봄).

    angle=10: 바퀴를 오른쪽으로 10도 꺾고 후진, speed=0.5: 후진 속도. 예: back(1.5)"""
    if sec is None or _bad_num(sec, 0, 600):
        raise ValueError(f"back(): sec must be 0 to 600 ({sec})")
    angle = _steer_angle("back", angle)
    return _step("back", 0 if angle is None else angle, sec, None, speed=speed)


def straight(sec=None, until: "Until | None" = None, angle="hold", speed=None, **opt: "Unpack[Opt]"):
    """조향을 유지(angle="hold")하거나 지정 각도로 직진. angle=0이면 정면. 차선은 보지 않음.

    예: straight(1.5, angle=0)"""
    if angle != "hold":
        angle = _steer_angle("straight", angle)
    return _step("straight", angle, sec, until, speed=speed, **opt)

def push(sec=None, until: "Until | None" = None, angle="hold", speed=None, **opt: "Unpack[Opt]"):
    """straight와 같지만 속도가 speed= > set_speed() > SPEED를 따르고 주행 내내 그 속도 유지.

    예: push(1.0)"""
    if angle != "hold":
        angle = _steer_angle("straight", angle)
    return _auto(_step("straight", angle, sec, until, speed=speed, **opt), speed, SPEED)

def right(sec=None, angle=None, until: "Until | None" = _MISSING, speed=None, **opt: "Unpack[Opt]"):
    """새 차선이 잡힐 때까지 우회전 (0.5~3초).

    angle=: 바퀴 각도. 예: right(), right(angle=20)"""
    if opt.get("min_sec") is None:
        opt["min_sec"] = 0.5
    if opt.get("max_sec") is None:
        min_sec = opt["min_sec"]
        longer = min_sec + 1.0 if isinstance(min_sec, (int, float)) and not isinstance(min_sec, bool) else 0
        opt["max_sec"] = max(3.0, longer)
    angle = _steer_angle("right", angle)
    return _step("right", None if angle is None else abs(angle), sec, _timed_until(sec, until, "lane"), speed=speed, **opt)


def left(sec=None, angle=None, until: "Until | None" = _MISSING, speed=None, **opt: "Unpack[Opt]"):
    """새 차선이 잡힐 때까지 좌회전 (0.5~3초).

    angle=: 바퀴 각도. 예: left(), left(angle=20)"""
    if opt.get("min_sec") is None:
        opt["min_sec"] = 0.5
    if opt.get("max_sec") is None:
        min_sec = opt["min_sec"]
        longer = min_sec + 1.0 if isinstance(min_sec, (int, float)) and not isinstance(min_sec, bool) else 0
        opt["max_sec"] = max(3.0, longer)
    angle = _steer_angle("left", angle)
    return _step("left", None if angle is None else -abs(angle), sec, _timed_until(sec, until, "lane"), speed=speed, **opt)


def turn(direction: "Direction", **opt: "Unpack[TurnOpt]"):
    """"left"/"right"를 변수로 받아 left()/right() 호출.

    예: turn("right")"""
    if direction not in ("left", "right"):
        raise ValueError(f'turn(): direction must be "left" / "right"{_hint(direction, ("left", "right"))}')
    return (right if direction == "right" else left)(**opt)


def curve(angle, sec=None, until: "Until | None" = None, speed=None, **opt: "Unpack[Opt]"):
    """지정 각도로 돌면서 차선 추종을 섞음.

    예: curve(12, 2)"""
    if angle is None or _bad_num(angle, -90, 90):
        raise ValueError(f"curve(): angle must be -90 to 90 ({angle})")
    return _step("curve", angle, sec, until, speed=speed, **opt)


def stop(sec, speed=0.0):
    """그 자리에서 sec초 정지.

    예: stop(1.5)"""
    return _step("stop", 0.0, sec, None, speed=speed)


def wait_green(max_sec=10, min_sec=0.0, speed=0.0):
    """선 채로 초록불이 될 때까지 대기.

    빨간불을 봤으면 무제한. 못 찾으면 max_sec초 뒤 출발.
    RED_BACK_ON = True면 1초씩 최대 2번 후진해 다시 찾음. 예: wait_green(max_sec=20)"""
    return _step("wait", 0.0, None, "green", min_sec=min_sec, max_sec=max_sec, speed=speed)


def wait(sec=None, until: "Until" = "clear", min_sec=0.0, max_sec=None, hold_sec=0.0, speed=0.0):
    """선 채로 앞 장애물이 없어질 때까지 대기 (until="clear").

    sec=를 같이 주면 그 시간 또는 조건 중 먼저 온 쪽. 예: wait(until="clear", hold_sec=1)"""
    return _step("wait", 0.0, sec, until, min_sec=min_sec, max_sec=max_sec, speed=speed, hold_sec=hold_sec)


def next():
    """웹 페이지에서 Enter를 누를 때까지 대기."""
    return _step("wait", 0.0, None, "enter", speed=0.0)


def start_gate(hold_sec=1.0, speed=0.0):
    """출발선 앞 장애물(차단바)이 치워지고 hold_sec초 지나면 출발.

    거리센서가 없으면 주행을 시작하지 않음. 예: start_gate(2)"""
    return wait(until="clear", hold_sec=hold_sec, speed=speed)


def finish():
    """코스 끝, 정지. finish()가 없어도 마지막 동작이 끝나면 차는 멈춤."""
    return _step("finish", 0.0, 0, None)


def _gap(gap, nth, name):
    if gap is None:
        return None
    gaps = list(gap) if isinstance(gap, (list, tuple)) else [gap]
    if not gaps or any(_bad_num(g, 0, 60) for g in gaps):
        raise ValueError(f"{name}(): gap must be 0 to 60 seconds, or a list of them ({gap!r})")
    n = nth or 1
    if n < 2:
        return None
    if isinstance(gap, (list, tuple)) and len(gaps) != n - 1:
        raise ValueError(f"{name}(): nth={n} has {n - 1} gaps, so gap needs {n - 1} numbers ({gap!r})")
    return [float(g) for g in gaps] if len(gaps) > 1 else [float(gaps[0])] * (n - 1)


def _nth(nth, name):
    if nth is None:
        return None
    if isinstance(nth, bool) or not isinstance(nth, int) or nth < 1:
        raise ValueError(f"{name}(): nth must be an integer of 1 or more ({nth})")
    return nth if nth > 1 else None


_LANE_CM = 20.0


def _lanes_cm(lanes, cm, who):
    """lanes=1.2: stop when the lane end is 1.2 lane widths ahead, the same as cm= that many lane widths."""
    if lanes is None:
        return cm
    if cm is not None:
        raise ValueError(f"{who}(): give lanes= or cm=, not both")
    if isinstance(lanes, bool) or not isinstance(lanes, (int, float)) or not 0 <= lanes <= 3:
        raise ValueError(f"{who}(): lanes must be 0 to 3 lane widths ({lanes})")
    return round(lanes * _LANE_CM, 1)


def to_junction(at=None, slow=None, blind_sec=0.5, speed=None, nth=1, cm=None, steer=None, sec=None, min_speed=None, shift=None, gap=None, lanes=None):
    """교차로 앞 차선 끝까지 가서 끝 (go()와 같음, cm 없으면 10cm).

    cm=25: LANE END 25cm에서 끝 (차 앞 기준). lanes=1.2: 차선 폭 1.2개 앞. sec=3: 3초가 지나도 끝"""
    cm = _lanes_cm(lanes, cm, "to_junction")
    if nth is None:
        nth = 1
    if isinstance(nth, bool) or not isinstance(nth, int) or nth < 1:
        raise ValueError(f"to_junction(): nth must be an integer of 1 or more ({nth})")
    if sec is not None and _bad_num(sec, 0, 600):
        raise ValueError(f"to_junction(): sec must be 0 to 600 ({sec})")
    step = _auto(follow(until="junction", shift=shift, speed=speed, at=at, slow=slow, blind_sec=blind_sec, nth=nth if nth > 1 else None, gap=gap, cm=cm, steer=steer, max_sec=sec, min_speed=min_speed), speed, SPEED)
    return dict(step, cm_default=True) if cm is None and at is None else step

def go(at=None, slow=None, blind_sec=0.5, speed=None, nth=1, cm=None, steer=None, sec=None, min_speed=None, shift=None, gap=None, lanes=None):
    """교차로 앞 차선 끝까지 가서 끝 (cm 없으면 10cm). 아래까지 보였다가 사라져도 끝.

    cm=25: 화면의 LANE END가 25cm가 되면 끝 (차 앞 기준, 4.5 미만은 4.5). LANE END는 차선 선이 꺾이는 곳
    (안 보이면 선이 끝나는 곳). 테이프 이음새는 끝으로 보지 않음.
    lanes=1.2: 차선 폭 1.2개 앞 (cm=와 같이 못 씀). sec=3: 교차로 또는 3초 중 먼저.
    nth=2, gap=1: 하나 지나치고 두 번째에서. 화면 값과 줄자 값이 다르면 aicar calibrate --end 10"""
    cm = _lanes_cm(lanes, cm, "go")
    if nth is None:
        nth = 1
    if isinstance(nth, bool) or not isinstance(nth, int) or nth < 1:
        raise ValueError(f"go(): nth must be an integer of 1 or more ({nth})")
    if sec is not None and _bad_num(sec, 0, 600):
        raise ValueError(f"go(): sec must be 0 to 600 ({sec})")
    step = _auto(follow(until="junction", shift=shift, speed=speed, at=at, slow=slow, blind_sec=blind_sec, nth=nth if nth > 1 else None, gap=gap, cm=cm, steer=steer, max_sec=sec, min_speed=min_speed), speed, SPEED)
    return dict(step, cm_default=True) if cm is None and at is None else step


def go_dis(speed=None, steer=None, sec=None, min_speed=None, shift=None, blind_sec=0.0):
    """차선을 따라가다 앞 거리센서에 차단바(물체)가 잡히면 끝.

    sec=5: 못 만나도 5초 뒤 끝. 바로 뒤에 start_gate()를 쓰면 차단바가 올라갈 때까지 대기.
    거리센서가 없으면 주행을 시작하지 않음. 예: go_dis(), start_gate(1)"""
    if sec is not None and _bad_num(sec, 0, 600):
        raise ValueError(f"go_dis(): sec must be 0 to 600 ({sec})")
    return _auto(follow(until="obstacle", blind_sec=blind_sec, shift=shift, speed=speed, steer=steer, max_sec=sec, min_speed=min_speed), speed, SPEED)


def to_crosswalk(at=0.67, slow=None, speed=None, steer=None, sec=None, min_speed=None, shift=None, nth=1, gap=None, blind_sec=0.0):
    """횡단보도 앞까지 가서 끝.

    at=: 횡단보도가 화면 이 높이까지 오면 끝 (클수록 가까이). 예: to_crosswalk(0.85)"""
    if sec is not None and _bad_num(sec, 0, 600):
        raise ValueError(f"to_crosswalk(): sec must be 0 to 600 ({sec})")
    return _auto(follow(until="crosswalk", nth=_nth(nth, "to_crosswalk"), gap=gap, blind_sec=blind_sec, shift=shift, speed=speed, at=at, slow=slow, steer=steer, max_sec=sec, min_speed=min_speed), speed, SPEED)


def pass_crosswalk(speed=None, steer=None, shift=None, min_speed=None):
    """횡단보도를 지나갈 때까지 차선 따라 주행 (최대 5초)."""
    return [_auto(follow(until="crossed", hold_sec=0.3, min_sec=0.5, max_sec=5, speed=speed, steer=steer, shift=shift, min_speed=min_speed), speed, SPEED)]


def _to_crosswalk(at=0.67, slow=None, speed=None, steer=None, sec=None, min_speed=None, shift=None, nth=1, gap=None, blind_sec=0.0):
    """to_crosswalk와 같지만 등록한 횡단보도(aicar crosswalk, 메뉴 c)로 판단.

    예: _to_crosswalk(0.85, nth=1, gap=1.5)"""
    if sec is not None and _bad_num(sec, 0, 600):
        raise ValueError(f"_to_crosswalk(): sec must be 0 to 600 ({sec})")
    return _auto(follow(until="_crosswalk", nth=_nth(nth, "_to_crosswalk"), gap=gap, blind_sec=blind_sec, shift=shift, speed=speed, at=at, slow=slow, steer=steer, max_sec=sec, min_speed=min_speed), speed, SPEED)


def go_to(name, offset=0, nth=1, slow=None, speed=None, steer=None, sec=None, min_speed=None, blind_sec=0.5, shift=None, gap=None, cm=None, lanes=None):
    """등록한 정지 지점(aicar stoppoint, 메뉴 t)까지 가서 정지.

    offset=3: 3cm 더 / -3: 덜. sec=10: 못 찾아도 10초 뒤 끝. go_to("A|B"): 먼저 오는 곳.
    STOP_POINTS_ON = False면 go()처럼 차선 끝에서 정지 (cm= 사용)"""
    cm = _lanes_cm(lanes, cm, "go_to")
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
    if not STOP_POINTS_ON:
        return go(cm=cm, slow=slow, blind_sec=blind_sec, speed=speed, nth=nth, gap=gap, steer=steer, sec=sec, min_speed=min_speed, shift=shift)
    if cm is not None:
        print(f'[warning] go_to("{"|".join(names)}", cm={cm}): cm= is only used when STOP_POINTS_ON = False. The stop point stops where it was registered (move it with offset=)')
    step = follow(until="_stop", shift=shift, speed=speed, slow=slow, steer=steer, max_sec=sec, min_speed=min_speed,
                  blind_sec=blind_sec, nth=nth if nth > 1 else None, gap=gap)
    return _auto(dict(step, stop=name, stop_cm=offset), speed, SPEED)


def _pass_crosswalk(speed=None, steer=None, shift=None, min_speed=None, sec=5):
    """등록한 횡단보도를 지나갈 때까지 차선 따라 주행 (화면에서 사라지면 끝, 최대 sec초)."""
    if _bad_num(sec, 0.5, 600):
        raise ValueError(f"_pass_crosswalk(): sec must be 0.5 to 600 ({sec})")
    return _auto(follow(until="_crossed", hold_sec=0.3, min_sec=0.5, max_sec=sec, speed=speed, steer=steer, shift=shift, min_speed=min_speed), speed, SPEED)

def to_gap(at=0.85, slow=None, speed=None, steer=None, min_speed=None, shift=None, sec=None, blind_sec=0.0, nth=1, gap=None):
    """차선이 끊기는 곳 앞까지 가서 끝.

    예: to_gap(0.85)"""
    return _auto(follow(until="gap", shift=shift, speed=speed, at=at, slow=slow, steer=steer, min_speed=min_speed, max_sec=sec,
                        blind_sec=blind_sec, nth=_nth(nth, "to_gap"), gap=gap), speed, SPEED)


def to_end(at=0.5, slow=None, blind_sec=0.0, speed=None, shift=None, cm=None, max_sec=None, steer=None, min_speed=None, sec=None, lanes=None):
    """바로 앞 끝에서 끝: 교차로선(가로선)이든 차선 끊김이든 먼저 오는 쪽 (sec=는 max_sec=와 같음).

    to_end(cm=10): 기준 위치에서 10cm 더 간 뒤 끝. cm=0은 선이 카메라 앞 38cm일 때 (차 앞 기준 아님).
    막다른 길에서 차 앞 몇 cm에 세우려면 go_end(cm=)"""
    cm = _lanes_cm(lanes, cm, "to_end")
    if sec is not None and max_sec is not None:
        raise ValueError("to_end(): give sec= or max_sec=, not both")
    return _auto(follow(until="end", speed=speed, at=at, slow=slow, blind_sec=blind_sec, shift=shift, cm=cm,
                        max_sec=max_sec if sec is None else sec, steer=steer, min_speed=min_speed), speed, 0.3 if SLOWDOWN_ON else SPEED)


def go_end(cm=10, at=0.5, slow=None, speed=None, steer=None, sec=None, min_speed=None, shift=None, blind_sec=0.0):
    """막다른 길: 차선을 가로막은 끝선이 차 앞 cm에 오면 정지 (0~40cm, go(cm=)와 같은 차 앞 기준).

    차선 끊김은 끝으로 보지 않고 지나감. 끝선이 차 밑으로 들어가 안 보여도 속도로 거리를 계산.
    예: go_end(10)"""
    # 막다른 길: 차선을 가로막은 끝선까지 가서, 차 앞에서 cm 앞에 정지 (go(cm=)와 같은 기준). 차선 끊김(틈)은 끝으로 보지 않고 지나감
    # 끝선이 차 밑으로 들어가 안 보여도 속도로 거리를 이어서 계산. 끝선 바로 뒤 매트 끝은 정지 사유로 보지 않음 (끝선을 못 쟀으면 그대로 정지)
    if _bad_num(cm, 0, 40):
        raise ValueError(f"go_end(): cm must be 0 to 40 (distance from the car front to the end line) ({cm})")
    if sec is not None and _bad_num(sec, 0, 600):
        raise ValueError(f"go_end(): sec must be 0 to 600 ({sec})")
    step = follow(until="end", cm=cm, at=at, slow=slow, speed=speed, steer=steer, max_sec=sec, min_speed=min_speed, shift=shift,
                  blind_sec=blind_sec)
    return dict(_auto(step, speed, SPEED), dead_end=True)


def _turn_fn(direction):
    if direction not in ("left", "right"):
        raise ValueError(f'direction must be "left" / "right"{_hint(direction, ("left", "right"))}')
    return right if direction == "right" else left


def _straight_speed(speed):
    return SPEED if speed is None else speed


def cross_gap(at=0.85, angle=0, speed=None, steer=None, sec=None):
    """끊긴 곳까지 간 뒤 새 차선이 잡힐 때까지 직진 (sec=1.5: 1.5초만 직진).

    보통의 차선 끊김은 follow()/go()만으로도 넘어감. 끊긴 곳에서 꼭 따로 서거나 직진할 때만 사용"""
    cross = (straight(sec, angle=angle, speed=speed, steer=steer) if sec is not None
             else straight(until="lane", angle=angle, max_sec=3, speed=speed, steer=steer))
    return [to_gap(at, speed=speed, steer=steer), _auto(cross, speed, SPEED)]


def traffic(at=0.73, max_sec=10, speed=None, steer=None, shift=None):
    """횡단보도 앞에 서서 초록불에 출발 (to_crosswalk + wait_green).

    예: traffic(0.73, max_sec=10)"""
    return [to_crosswalk(at, speed=speed, steer=steer, shift=shift), wait_green(max_sec=max_sec)]


def cross(at=None, line: 'Literal["junction", "crosswalk"]' = "junction", speed=None, approach=True):
    """교차로선까지 가서 정면으로 건넘. line="crosswalk": 차선 따라 횡단보도 건넘.

    approach=False: 이미 앞에 서 있을 때"""
    if line not in ("junction", "crosswalk"):
        raise ValueError(f'cross(): line must be "junction" / "crosswalk"{_hint(line, ("junction", "crosswalk"))}')
    if line == "junction":
        steps = [to_junction(at, speed=speed)] if approach else []
        return steps + [straight(until="lane", angle=0, min_sec=0.8, max_sec=4, speed=_straight_speed(speed))]
    steps = [to_crosswalk(at or 0.67, speed=speed)] if approach else []
    return steps + [_auto(follow(until="crossed", hold_sec=0.3, min_sec=0.5, max_sec=5, speed=speed), speed, SPEED)]


def corner(direction: "Direction", at=None, go_sec=0.5, wide=0, stop_sec=0.20, speed=None):
    """교차로 선까지 가서 go_sec초 더 간 뒤 꺾고 정렬·주행 (to_junction + straight + tr/tl).

    wide=: 꺾기 전 바깥쪽 치우침 (cm). 예: corner("right")"""
    if _bad_num(go_sec, 0, 600):
        raise ValueError(f"corner(): go_sec must be 0 to 600 ({go_sec})")
    if _bad_num(wide, -8, 8):
        raise ValueError(f"corner(): wide must be -8 to 8 cm ({wide})")
    turn_fn = _turn_fn(direction)
    shift = wide if direction == "left" else -wide
    steps = [_auto(follow(until="junction", at=at, blind_sec=1.5, speed=speed, shift=shift or None), speed, SPEED)]
    if go_sec > 0:
        steps.append(straight(go_sec))
    return steps + _turn_align_stop(turn_fn, stop_sec)


def uturn(direction: "Direction", at=None, blind_sec=1.5, stop_sec=0.20, speed=None):
    """교차로 선 앞에서 최대 조향으로 돌아 정렬·주행 (blind_sec 동안 새 차선 안 봄).

    예: uturn("left")"""
    turn_speed = (0.4 if SLOWDOWN_ON else SPEED) if speed is None else speed
    return [_auto(follow(until="junction", at=at, slow=turn_speed if SLOWDOWN_ON else None, speed=speed), speed, turn_speed),
            _turn_align_stop(_turn_fn(direction), stop_sec, turn_speed, angle=None, blind_sec=blind_sec, max_sec=5, deg=180)]


def tr(min_sec=2.5, sec=0.50, speed=None, deg=90, align=True):
    """우회전 → 차로에 맞춰 정렬 → sec초 주행.

    deg: 이만큼 돌면 끝, min_sec: 최소 이 시간은 돎. align=False: 정렬 건너뜀.
    속도: speed= > set_speed() > SPEED (감속 없음). 급한 U자 곡선은 차선 따라가기 대신 tr/tl로"""
    return _cruise(_turn_align_stop(right, sec, speed, angle=TR_ANGLE, deg=deg, min_sec=min_sec, align=align), speed)


def tl(min_sec=2.5, sec=0.50, speed=None, deg=90, align=True):
    """좌회전 → 차로에 맞춰 정렬 → sec초 주행.

    deg: 이만큼 돌면 끝, min_sec: 최소 이 시간은 돎. align=False: 정렬 건너뜀.
    속도: speed= > set_speed() > SPEED (감속 없음)"""
    return _cruise(_turn_align_stop(left, sec, speed, angle=TL_ANGLE, deg=deg, min_sec=min_sec, align=align), speed)


TURN_SPEED = 0.5
TR_ANGLE = 20
TL_ANGLE = 20
ALIGN_SPEED = 0.4
SETTLE_SPEED = 0.3
TURN_BLIND = 1.0
TURN_BLIND_SPEED = 0.35


def _cruise(steps, speed):
    if speed is None:
        steps = [dict(s, speed=SPEED, keep_speed=True) for s in steps]
    else:
        steps = [dict(s, keep_speed=True, turn_part=True) for s in steps]
    if speed is not None:
        return steps
    return [dict(s, auto=True, turn_part=True, turn_blind=s["do"] in ("left", "right")) for s in steps]


def _turn_align_stop(turn_fn, stop_sec, speed=None, angle=21, blind_sec=None, max_sec=10, deg=90, min_sec=0.5, align=True):
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
    out, cruise, steer, style = [], None, STEER_MAX, None
    for s in _expand(steps):
        if s["do"] == "set_speed":
            cruise = s["speed"]
        elif s["do"] == "set_style":
            style = s["style"]
        elif s["do"] == "set_steer":
            steer = STEER_MAX if s["steer"] is None else s["steer"]
        else:
            if cruise is not None and s.get("auto"):
                s = dict(s, speed=cruise)
                if s.get("turn_blind"):
                    s["blind"] = round(TURN_BLIND * TURN_BLIND_SPEED / max(cruise, TURN_BLIND_SPEED), 2)
            mine = s.get("auto") or s.get("own_speed")
            if steer is not None and mine and s["do"] == "follow" and s.get("steer") is None and not s.get("turn_part"):
                s = dict(s, steer=steer)
            if style is not None and mine and s["do"] == "follow" and s.get("style") is None and not s.get("turn_part"):
                s = dict(s, style=style)
            out.append(s)
    return out


COMMON = ("at", "slow", "speed", "blind_sec", "min_sec", "max_sec", "side", "shift", "style", "steer",
          "nth", "gap", "min_speed", "hold_sec", "until", "cm")
_EVERY_STEP = ("speed", "steer", "shift", "side", "style", "min_speed")
_STEP_KEY = {"blind_sec": "blind", "min_sec": "min", "max_sec": "max", "hold_sec": "hold"}


def _apply_common(name, out, extra):
    bad = [k for k in extra if k not in COMMON]
    if bad:
        raise ValueError(f"{name}(): no option {bad[0]}={_hint(bad[0], COMMON)} (common options: {', '.join(COMMON)})")
    steps = _expand(out) if isinstance(out, list) else [out]
    if not steps:
        return out
    main = ([st for st in steps if st.get("until") is not None] or steps)[0]
    until = extra.get("until", main.get("until"))
    if "until" in extra and main.get("sec"):
        raise ValueError(f"{name}(): until= cannot be used with a timed step (sec); leave out sec")
    nth = extra.get("nth", main.get("nth"))
    check = {k: v for k, v in extra.items() if k not in ("until", "gap")}
    check["nth"] = nth
    _step(name, 0.0, None, until if until is not None else "junction",
          **{k: v for k, v in check.items() if k in OPTIONS})
    if "nth" in extra:
        nth = None if nth is None or nth <= 1 else nth
    gap = _gap(extra["gap"], nth, name) if "gap" in extra else main.get("gap")
    new = []
    for st in steps:
        st = dict(st)
        is_main = _same(st, main)
        for k, v in extra.items():
            if not is_main and k not in _EVERY_STEP:
                continue
            if k == "nth":
                v = nth
            elif k == "gap":
                v = gap
            elif k in ("blind_sec", "min_sec", "hold_sec") and v is None:
                v = 0.0
            st[_STEP_KEY.get(k, k)] = v
            if k == "speed" and v is not None:
                st.pop("auto", None)
                st["own_speed"] = True
            if k == "blind_sec":
                st.pop("turn_blind", None)
        if is_main and (extra.get("until", "junction") != "junction" or "cm" in extra or "at" in extra):
            st.pop("cm_default", None)
        if is_main and "nth" in extra and "gap" not in extra and nth is None:
            st["gap"] = None
        new.append(st)
    return new if isinstance(out, list) else new[0]


def _same(st, main):
    return st.get("_id") == main.get("_id")


def _with_common(fn):
    import functools
    import inspect
    params = inspect.signature(fn).parameters
    own = set(params)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        own |= set(OPTIONS)

    @functools.wraps(fn)
    def run(*args, **kw):
        extra = {k: kw.pop(k) for k in list(kw) if k not in own}
        out = fn(*args, **kw)
        if not extra:
            return out
        steps = _expand(out) if isinstance(out, list) else [out]
        tagged = [dict(st, _id=i) for i, st in enumerate(steps)]
        res = _apply_common(fn.__name__, tagged if isinstance(out, list) else tagged[0], extra)
        res = res if isinstance(res, list) else [res]
        for st in res:
            st.pop("_id", None)
        return res if isinstance(out, list) else res[0]
    return run


for _name in ("follow", "straight", "push", "right", "left", "turn", "curve", "back", "stop", "wait", "wait_green", "start_gate",
              "to_junction", "go", "go_dis", "go_end", "to_crosswalk", "pass_crosswalk", "_to_crosswalk", "_pass_crosswalk", "go_to", "to_gap", "to_end",
              "cross_gap", "traffic", "cross", "corner", "uturn", "tr", "tl"):
    globals()[_name] = _with_common(globals()[_name])


def describe(step):
    names = {"junction": "junction", "crosswalk": "crosswalk", "lane": "new lane", "gap": "lane gap", "end": "end (junction line / gap)", "crossed": "crosswalk passed", "_crosswalk": "registered crosswalk", "_crossed": "registered crosswalk passed", "_stop": "stop point",
             "green": "green light", "lost": "lane lost", "obstacle": "obstacle", "clear": "obstacle cleared",
             "aligned": "aligned", "enter": "Enter"}
    do = {"follow": "follow lane", "straight": "straight", "back": "back up", "right": "turn right", "left": "turn left", "curve": "curve",
          "stop": "stop", "wait": "wait", "finish": "finish"}[step["do"]]
    parts = [do]
    until = "dead end line" if step.get("dead_end") else names.get(step["until"]) if step["until"] else None
    where = f"until {until}" + (f" (screen {step['at']})" if step["at"] else "") if step["until"] else ""
    if step["sec"] and where and step["do"] == "wait":
        parts.append(f"{step['sec']}s or {where}")
    elif step["sec"]:
        parts.append(f"{step['sec']}s")
    elif where:
        parts.append(where)
    for key, fmt in (("side", "on the {} line"), ("shift", "{:+}cm off center"), ("steer", "max {} deg"),
                     ("style", "{}"), ("speed", "speed {}"), ("slow", "slow to {}"),
                     ("blind", "ignore first {}s"), ("min", "at least {}s"), ("deg", "end after {} deg"), ("nth", "at line #{}"), ("gap", "gaps {}s"), ("cm", "+{}cm")):
        if step.get(key) is not None and not (key == "speed" and step["do"] in ("stop", "wait")) \
                and not (key in ("blind", "min") and not step[key]):
            parts.append(fmt.format(step[key]))
    return " ".join(parts)


# --- 코스 ---

COURSE = _flat([
    set_style("soft"),
    start_gate(2),

    # follow(4),
    # _to_crosswalk(0.85, nth=1, gap=1.5),
    # follow(0.25),
    # stop(6.5),
    # pass_crosswalk(),
    # follow(0.75),
    # stop(1),

    finish(),
])

if __name__ == "__main__":
    for i, step in enumerate(_flat(COURSE), 1):
        print(f"{i:2d}. {describe(step)}")