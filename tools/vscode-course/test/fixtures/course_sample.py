EVENTS = ("junction", "end", "clear", None)
OPTIONS = ("min_sec", "max_sec", "speed", "at", "side")
COMMON = ("at", "speed", "side")


def follow(sec=None, until="junction", speed=None, **opt):
    return {"do": "follow", "sec": sec, "until": until, "speed": speed, **opt}


def go(at=None, speed=None, nth=1):
    return {"do": "follow", "until": "junction", "at": at, "speed": speed}


def turn(direction, **opt):
    return {"do": direction, "until": None}


def finish():
    return {"do": "finish", "until": None}


def undocumented_action(sec=1.0):
    return {"do": "stop", "sec": sec, "until": None}


def _helper():
    return None


for _name in ("follow", "turn"):
    pass

COURSE = [
    follow(until="end"),
    go(at=0.8),
    turn("left"),
    finish(),
]
