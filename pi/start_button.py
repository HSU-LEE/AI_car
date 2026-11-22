"""Start button on GPIO21 (pin 40, the other side to GND pin 39).

A short press starts the drive, pauses it, or resumes it. Holding it for hold_sec puts the car back to waiting for a start.
Settings live in button.json next to this file on the Pi, so they hold without the Mac. aicar push leaves *.json alone.

  python start_button.py show                  current settings
  python start_button.py on | off              use the button for aicar go, or start at once as before
  python start_button.py hold restart | keep   after a long hold: start again from step 1, or carry on from where it stopped
  python start_button.py hold-sec 3            seconds for a long hold (1 to 10)
  python start_button.py test [seconds]        print presses and the state they lead to, without the motor
  python start_button.py sweep on | off        before a fresh start, turn the front wheels slowly left, right, and back to center
  python start_button.py sweep-speed slow | normal | fast    how long that takes (3, 1.8, 1.2 s)
  python start_button.py sweep-test            run the steering check once now (servo only, the motor stays off)
"""
import math
import sys
import threading
import time
from pathlib import Path

from safe_io import read_json, write_json

PIN = 21
BOUNCE_SEC = 0.03           # A change shorter than this is contact bounce, not a press.
POLL_SEC = 0.005
SETTINGS_FILE = Path(__file__).with_name("button.json")
DEFAULTS = {"enabled": True, "hold": "restart", "hold_sec": 3.0, "sweep": True, "sweep_speed": "normal"}
HOLD_MIN, HOLD_MAX = 1.0, 10.0

SWEEP_DEG = 20.0            # How far the steering check turns each way. Under the steering limit, so the servo never hits its end.
SWEEP_SETTLE = 0.3          # Seconds at center at the end, for the servo to settle before driving.
SWEEP_RETURN = 2.0          # The last part, right back to center, takes this many times as long as center to left.
SWEEP_SEC = {"slow": 3.0, "normal": 1.8, "fast": 1.2}      # Whole check, settle included.

WAIT, RUN, PAUSE = "WAIT", "RUN", "PAUSE"
NAMES = {WAIT: "waiting for the start button", RUN: "driving", PAUSE: "paused"}


def clean_settings(raw):
    """Settings with every value checked. A missing or bad value takes its default."""
    s = dict(DEFAULTS)
    if not isinstance(raw, dict):
        return s
    if isinstance(raw.get("enabled"), bool):
        s["enabled"] = raw["enabled"]
    if raw.get("hold") in ("restart", "keep"):
        s["hold"] = raw["hold"]
    sec = raw.get("hold_sec")
    if isinstance(sec, (int, float)) and not isinstance(sec, bool) and HOLD_MIN <= sec <= HOLD_MAX:
        s["hold_sec"] = float(sec)
    if isinstance(raw.get("sweep"), bool):
        s["sweep"] = raw["sweep"]
    if raw.get("sweep_speed") in SWEEP_SEC:
        s["sweep_speed"] = raw["sweep_speed"]
    return s


def load_settings(path=SETTINGS_FILE):
    try:
        return clean_settings(read_json(path))
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save_settings(settings, path=SETTINGS_FILE):
    s = clean_settings(settings)
    write_json(path, s)     # Whole or not at all, even if the battery is pulled while saving.
    return s


def describe(s):
    if not s["enabled"]:
        return "button OFF (aicar go starts driving at once)"
    after = "start again from step 1" if s["hold"] == "restart" else "carry on from where it stopped"
    sweep = (f"steering check before a fresh start: ON ({s['sweep_speed']}, {SWEEP_SEC[s['sweep_speed']]:g} s)"
             if s["sweep"] else "steering check before a fresh start: OFF")
    return f"button ON: tap = start / pause / resume, hold {s['hold_sec']:g} s = wait for start ({after})\n{sweep}"


class Sweep:
    """The steering check: center -> left -> right (a sine), then slower from right back to center (eased to a stop),
    then a short rest at center.

    angle(now) gives the steering angle (negative = left), or None once the check is over. Time is passed in."""

    def __init__(self, start, speed="normal", deg=SWEEP_DEG):
        self.start, self.deg = start, deg
        self.total = SWEEP_SEC.get(speed, SWEEP_SEC["normal"])
        quarter = (self.total - SWEEP_SETTLE) / (3 + SWEEP_RETURN)     # Center to left.
        self.period = 4 * quarter
        self.left_at, self.right_at = quarter, 3 * quarter
        self.center_at = self.right_at + SWEEP_RETURN * quarter

    def angle(self, now):
        t = max(0.0, now - self.start)
        if t >= self.total:
            return None
        if t >= self.center_at:
            return 0.0
        if t >= self.right_at:      # Right to center: a half cosine, so it slows down into center.
            u = (t - self.right_at) / (self.center_at - self.right_at)
            return self.deg * (1 + math.cos(math.pi * u)) / 2
        return -self.deg * math.sin(2 * math.pi * t / self.period)


class Presses:
    """Raw button levels in, "tap" and "hold" events out. Time is passed in, so it runs the same without the button.

    A tap is a press released before hold_sec. A hold fires the moment hold_sec is reached, while still held,
    and its release is not a tap. A button already held when this starts counts only after it is released once."""

    def __init__(self, hold_sec=DEFAULTS["hold_sec"], bounce=BOUNCE_SEC):
        self.hold_sec, self.bounce = hold_sec, bounce
        self.level = None           # Debounced level: True pressed.
        self.raw = None
        self.raw_since = 0.0
        self.down_at = None
        self.held = False
        self.ready = False          # False until the button is seen released.

    def feed(self, pressed, now):
        events = []
        if pressed != self.raw:
            self.raw, self.raw_since = pressed, now
        if self.level is None:
            self.level = pressed
            self.ready = not pressed
            return events
        if self.raw != self.level and now - self.raw_since >= self.bounce:
            self.level = self.raw
            if not self.ready:
                self.ready = not self.level
            elif self.level:
                self.down_at, self.held = self.raw_since, False
            elif self.down_at is not None:
                if not self.held:
                    events.append("tap")
                self.down_at, self.held = None, False
        if self.ready and self.level and self.down_at is not None and not self.held and now - self.down_at >= self.hold_sec:
            self.held = True
            events.append("hold")
        return events


class StartControl:
    """WAIT -> tap -> RUN -> tap -> PAUSE -> tap -> RUN. A hold in RUN or PAUSE goes back to WAIT.

    handle() returns what the drive has to do: "start" (fresh course), "resume", "pause", "wait", or None."""

    def __init__(self, hold="restart"):
        self.hold = hold
        self.state = WAIT
        self.fresh = True           # The next start runs the course from the first step.

    def handle(self, event):
        if event == "tap":
            if self.state == WAIT:
                self.state = RUN
                if self.fresh:
                    self.fresh = False
                    return "start"
                return "resume"
            if self.state == RUN:
                self.state = PAUSE
                return "pause"
            if self.state == PAUSE:
                self.state = RUN
                return "resume"
        elif event == "hold" and self.state in (RUN, PAUSE):
            self.state = WAIT
            self.fresh = self.hold == "restart"
            return "wait"
        return None

    def cancel(self):
        """A press during the steering check: back to waiting, and the next start is still a fresh one."""
        self.state, self.fresh = WAIT, True

    def finished(self):
        """The course ran to its end: wait for a start, which runs it again from the first step."""
        self.state, self.fresh = WAIT, True


class ButtonReader:
    """Reads GPIO21 in a background thread and queues tap and hold events. take() returns those seen since the last call."""

    def __init__(self, hold_sec=DEFAULTS["hold_sec"], pin=PIN):
        from gpiozero import Button
        self.button = Button(pin, pull_up=True, bounce_time=None)
        self.presses = Presses(hold_sec)
        self.events = []
        self.lock = threading.Lock()
        self.error = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        fails = 0
        while not self._stop.is_set():
            try:
                pressed = self.button.is_pressed
                fails = 0
            except Exception as e:      # Kept failing (20 reads, 0.1 s): stop reading and say so once.
                fails += 1
                if fails >= 20:
                    self.error = e
                    print(f"\n[error] start button stopped working: {type(e).__name__}: {e}. "
                          f"Ctrl+C or aicar stop still stops the car.", flush=True)
                    return
                time.sleep(POLL_SEC)
                continue
            ev = self.presses.feed(pressed, time.monotonic())
            if ev:
                with self.lock:
                    self.events.extend(ev)
            time.sleep(POLL_SEC)

    def pressed(self):
        return bool(self.presses.level)

    def take(self):
        with self.lock:
            ev, self.events = self.events, []
        return ev

    def close(self):
        self._stop.set()
        self.thread.join(timeout=1)
        self.button.close()


def test(seconds=30):
    s = load_settings()
    print(describe(s))
    print(f"Press the button for {seconds:g} s. The motor does not move.")
    reader = ButtonReader(s["hold_sec"])
    ctl = StartControl(s["hold"])
    print(f"state: {NAMES[ctl.state]}", flush=True)
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            for ev in reader.take():
                act = ctl.handle(ev)
                print(f"{ev:4s} -> {act or 'ignored':6s}  state: {NAMES[ctl.state]}", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
    print("test done")


def sweep_test(s):
    """The steering check once, servo only. The motor gets no command but stop."""
    from car import Car
    car = Car()
    sw = Sweep(time.monotonic(), s["sweep_speed"])
    print(f"Steering check: left, right, center ({sw.total:g} s). The motor stays off.", flush=True)
    try:
        car.stop()
        while True:
            a = sw.angle(time.monotonic())
            if a is None:
                break
            car.steer(a)
            time.sleep(0.02)
        print("done: wheels at center")
    except KeyboardInterrupt:
        print("stopped")
    finally:
        car.close()     # Motor stopped, wheels centered.


def main(argv):
    s = load_settings()
    cmd = argv[0] if argv else "show"
    if cmd == "show":
        pass
    elif cmd in ("on", "off"):
        s["enabled"] = cmd == "on"
        s = save_settings(s)
    elif cmd == "hold" and len(argv) == 2 and argv[1] in ("restart", "keep"):
        s["hold"] = argv[1]
        s = save_settings(s)
    elif cmd == "hold-sec" and len(argv) == 2:
        try:
            sec = float(argv[1])
        except ValueError:
            sec = None
        if sec is None or not HOLD_MIN <= sec <= HOLD_MAX:
            raise SystemExit(f"hold-sec must be a number from {HOLD_MIN:g} to {HOLD_MAX:g}")
        s["hold_sec"] = sec
        s = save_settings(s)
    elif cmd == "test":
        try:
            sec = float(argv[1]) if len(argv) > 1 else 30.0
        except ValueError:
            sec = -1.0
        if not 0 < sec <= 3600:
            raise SystemExit("test seconds must be a number from 1 to 3600")
        test(sec)
        return
    elif cmd == "sweep" and len(argv) == 2 and argv[1] in ("on", "off"):
        s["sweep"] = argv[1] == "on"
        s = save_settings(s)
    elif cmd == "sweep-speed" and len(argv) == 2 and argv[1] in SWEEP_SEC:
        s["sweep_speed"] = argv[1]
        s = save_settings(s)
    elif cmd == "sweep-test":
        sweep_test(s)
        return
    else:
        raise SystemExit("usage: start_button.py [show | on | off | hold restart|keep | hold-sec N | test [seconds] | "
                         "sweep on|off | sweep-speed slow|normal|fast | sweep-test]")
    print(describe(s))


if __name__ == "__main__":
    main(sys.argv[1:])
