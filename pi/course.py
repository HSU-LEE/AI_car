

# < 속도 조정 >

SPEED = 0.375








# at= 값은 넓은 화각(센서 전체, 약 66도) 기준. 예전 좁은 화각 값 f는 0.5 + (f - 0.5) / 1.5 로 환산
#   (예: 0.85 → 0.73, 0.75 → 0.67, 0.7 → 0.63, 0.65 → 0.6). 0.8보다 크면 예전 화면보다 더 가까운 바닥을 봄
EVENTS = ("junction", "crosswalk", "lane", "gap", "green", "obstacle", "clear", "lost", "aligned", None)
SIDES = ("left", "right", "center", None)
STYLES = ("soft", "normal", "sharp", None)
OPTIONS = ("min_sec", "max_sec", "speed", "at", "side", "shift", "steer", "style", "hold_sec", "slow")


def _hint(word, choices):
    import difflib
    near = difflib.get_close_matches(str(word), [str(c) for c in choices if c is not None], 1)
    return f' → "{near[0]}" 말인가요?' if near else ""


def _step(do, angle, sec, until, **opt):
    for k in opt:
        if k not in OPTIONS:
            raise ValueError(f"{do}(): 옵션 {k}= 는 없음{_hint(k, OPTIONS)} (쓸 수 있는 옵션: {', '.join(OPTIONS)})")
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
            "steer": opt.get("steer"), "style": opt.get("style"), "slow": opt.get("slow")}


def follow(sec=None, until="junction", **opt):
    return _step("follow", 0.0, sec, until, **opt)


def straight(sec=None, until=None, angle=0, **opt):
    if not (angle == "hold" or isinstance(angle, (int, float))):
        raise ValueError('straight(): angle은 숫자 또는 "hold"')
    return _step("straight", angle, sec, until, **opt)


def right(sec=None, angle=None, until="lane", **opt):
    """우회전. angle을 안 쓰면 차가 꺾을 수 있는 최대 각도 (lane_follow.py AUTO_STEER_MAX)"""
    opt.setdefault("min_sec", 0.5)
    opt.setdefault("max_sec", 3.0)
    return _step("right", None if angle is None else abs(angle), sec, until, **opt)


def left(sec=None, angle=None, until="lane", **opt):
    """좌회전. angle을 안 쓰면 차가 꺾을 수 있는 최대 각도"""
    opt.setdefault("min_sec", 0.5)
    opt.setdefault("max_sec", 3.0)
    return _step("left", None if angle is None else -abs(angle), sec, until, **opt)


def turn(direction, **opt):
    """turn("right") = right(), turn("left") = left(). 묶음 안에서 방향을 변수로 받을 때 편함"""
    if direction not in ("left", "right"):
        raise ValueError(f'turn(): 방향은 "left" / "right"{_hint(direction, ("left", "right"))}')
    return (right if direction == "right" else left)(**opt)


def curve(angle, sec=None, until=None, **opt):
    return _step("curve", angle, sec, until, **opt)


def stop(sec):
    return _step("stop", 0.0, sec, None, speed=0.0)


def wait_green(max_sec=10, min_sec=0.0):
    """정지한 채로 신호등이 빨강 → 초록으로 바뀌면 끝 → 다음 동작으로 출발.
    빨간불을 3프레임 연속 본 뒤, 그 빨간 램프 근처에서 초록불이 3프레임 연속 보여야 함 (JUNCTION_FRAMES).
    처음부터 보이던 초록 물체나, 다른 곳에 있는 초록 물체로는 출발하지 않음.
    max_sec: 신호 인식이 실패해도 이 시간이 지나면 출발 (로봇 정지 판정 방지)"""
    return _step("wait", 0.0, None, "green", min_sec=min_sec, max_sec=max_sec, speed=0.0)


def wait(sec=None, until="clear", min_sec=0.0, max_sec=None, hold_sec=0.0):
    """정지한 채로 기다림. 기본은 앞 장애물이 없어질 때까지.
    until="obstacle"이면 물체가 나타날 때까지, sec를 주면 그 시간만큼.
    hold_sec: 그 상태가 이 시간(초) 동안 끊김 없이 계속되어야 끝 (중간에 바뀌면 처음부터 다시 셈)
    max_sec: 센서가 계속 같은 상태여도 이 시간이 지나면 다음 동작으로"""
    return _step("wait", 0.0, sec, until, min_sec=min_sec, max_sec=max_sec, speed=0.0, hold_sec=hold_sec)


def start_gate(hold_sec=1.0):
    """출발 대기: 앞을 막고 있던 장애물이 치워진 상태가 hold_sec초 계속되면 출발"""
    return wait(until="clear", hold_sec=hold_sec)


def finish():
    return _step("finish", 0.0, 0, None)


def to_junction(at=None, slow=0.2):
    """다음 교차로까지. at: 교차로 선 앞 얼마까지 (0.53 멀리 ~ 0.8 바로 앞)
    slow: 교차로 선이 보이면 점점 줄여서 이 속도로 도착 (None이면 감속 안 함)"""
    return follow(until="junction", speed=SPEED, at=at, slow=slow)


def to_crosswalk(at=None):  # 횡단보도까지. at: 횡단보도 앞 얼마까지 (0.53 멀리 ~ 0.8 바로 앞)
    return follow(until="crosswalk", speed=SPEED, at=at)


def to_gap(at=None, speed=SPEED):  # 차선이 끊기는 곳까지. at: 끊기는 곳 앞 얼마까지 (0.6 일찍 ~ 0.8 바로 앞)
    return follow(until="gap", speed=speed, at=at)


def cross_gap(at=0.7, angle=0, speed=SPEED):     # 끊긴 구간 통과 → 다시 차선 따라감
    # angle=0: 직선 위 끊김, angle="hold": 곡선 위 끊김 (들어갈 때 조향각 유지)
    # max_sec=3은 차선을 끝내 못 찾을 때만 쓰는 안전장치. 보통은 새 차선이 잡혀서 끝남
    return [to_gap(at, speed), straight(until="lane", angle=angle, max_sec=3, speed=speed)]


def traffic(at=0.73, max_sec=10):   # 횡단보도 앞 정지 → 초록불에 출발
    # signal이 아니라 traffic: lane_follow.py가 쓰는 signal 모듈과 헷갈리지 않게
    return [to_crosswalk(at), wait_green(max_sec=max_sec)]


def cross(at=0.67, speed=None):
    """십자 교차로 직진 통과: 횡단보도(교차로 앞)까지 → 직진 → 건너편 차선이 잡히면 끝"""
    return [follow(until="crosswalk", at=at, speed=speed),
            straight(until="lane", min_sec=0.8, max_sec=4, speed=speed)]


def corner(direction, at=0.6, speed=None, wide=0):
    """ㄱ자 직각 코너: 앞을 가로막는 선까지 → 최대로 꺾어 회전 → 새 차선이 잡히면 끝
    at: 클수록 늦게 꺾음 (0.6 기본, 차선 아래로 빠지면 줄이기 / 안쪽 모서리를 밟으면 키우기)
    wide=3: 코너 전에 바깥쪽으로 3cm 붙어서 들어감"""
    shift = wide if direction == "left" else -wide     # 좌회전이면 바깥 = 오른쪽(+)
    return [follow(until="junction", at=at, speed=speed, shift=shift or None),
            turn(direction, min_sec=0.5, max_sec=4, speed=speed)]


def uturn(direction, speed=0.2):
    """U자 급커브 (30cm 타일): 앞을 가로막는 바깥 선이 보이면 → 최대로 꺾어 돌기 → 새 차선이 잡히면 끝"""
    return [follow(until="junction", speed=speed),
            turn(direction, min_sec=1.0, max_sec=5, speed=speed)]


def tr(stop_sec=1.0):
    """우회전: 새 차선이 잡힐 때까지 돌고(최대 3초) → 차로에 맞춰 정렬될 때까지만 조금 가고 → 그 자리에서 정지.
    stop_sec: 정렬 뒤 멈춰 있는 시간(초). 0이면 멈추지 않고 바로 다음 동작"""
    return _turn_align_stop(right, stop_sec)


def tl(stop_sec=1.0):
    """좌회전: tr()과 같고 방향만 왼쪽"""
    return _turn_align_stop(left, stop_sec)


TURN_SPEED = 0.3      # tr()/tl() 회전 속도 (빠르면 반경이 커지고 과회전). 모터 최소값보다 낮으면 최소값으로 달림
ALIGN_SPEED = 0.2     # 회전 뒤 정렬할 때 속도 (천천히 맞추고 멈춤)


def _turn_align_stop(turn_fn, stop_sec):
    steps = [turn_fn(until="lane", angle=21, min_sec=0.5, max_sec=3, speed=TURN_SPEED),  # 3초는 차선을 못 찾을 때의 안전장치
             # 천천히, 부드러운 조향으로 정렬. 못 하면 3초 뒤 그냥 넘어감
             follow(until="aligned", max_sec=3, speed=ALIGN_SPEED, style="soft")]
    return steps + [stop(stop_sec)] if stop_sec > 0 else steps

def _flat(steps):
    """묶음(리스트)을 펼쳐서 동작 하나하나의 목록으로. 묶음 안의 묶음도 펼침"""
    out = []
    for s in steps:
        out.extend(_flat(s) if isinstance(s, list) else [s])
    return out


def describe(step):
    """동작 하나를 사람이 읽는 문장으로"""
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
                     ("style", "{}"), ("speed", "속도 {}"), ("slow", "→ {}까지 감속")):
        if step.get(key) is not None and not (key == "speed" and step["do"] in ("stop", "wait")):
            parts.append(fmt.format({"left": "왼", "right": "오른"}.get(step[key], step[key])))
    return " ".join(parts)


# ↓↓↓ 여기를 코스에 맞게 수정 ↓↓↓

COURSE = _flat([
    cross_gap(0.85, speed=0.2),    
    to_junction(),
    follow(1.5),
    tr(),
    to_junction(),
    follow(1.5),
    tl(),
    to_gap(0.75),
    finish(),
])

if __name__ == "__main__":
    for i, step in enumerate(_flat(COURSE), 1):
        print(f"{i:2d}. {describe(step)}")

"""
    start_gate(1.0),                   
    to_junction(),
    tr(),
    follow(5,speed=SPEED),
    follow(until="crosswalk", at=0.63, max_sec=8.5, speed=SPEED, slow=0.2), 
    wait_green(max_sec=10),    
    stop(1),
    straight(max_sec=4, speed=SPEED),  
    to_junction(),
    tr(),
    to_junction(),
    straight(max_sec=4, speed=SPEED),  
    follow(until="junction", speed=SPEED, slow=0.2),
    right(6, speed=0.35, angle=21),
    to_junction(),
    straight(max_sec=7, speed=SPEED),  
    finish(),
"""