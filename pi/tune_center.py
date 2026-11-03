"""Tune the servo center.

Adjust with the keys until the wheels point straight, then press s.
The value is saved in calib.json and car.py loads it on the next run.

  z / c   0.1 degree left / right
  a / d   0.5 degree left / right
  A / D   2 degrees left / right
  r       Reset the trim to 0
  s       Save and quit
  q       Quit without saving
"""
import json
import sys
import termios
import tty

from car import CALIB_FILE, Car, load_calib, load_trim

MAX_TRIM = 20


def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    print(__doc__)
    car = Car()
    trim = load_trim()
    steps = {"z": -0.1, "c": 0.1, "a": -0.5, "d": 0.5, "A": -2, "D": 2}
    try:
        while True:
            car.servo.angle = trim
            trim = round(trim, 1)
            print(f"\rtrim = {trim:+.1f}°   ", end="", flush=True)
            key = getkey()
            if key in steps:
                trim = max(-MAX_TRIM, min(MAX_TRIM, trim + steps[key]))
            elif key == "r":
                trim = 0.0
            elif key == "s":
                # A new center makes the learned running bias stale, so clear it.
                data = load_calib()     # Keep other saved values, such as the motor minimums.
                data.update(steer_trim=trim, steer_bias=0.0)
                CALIB_FILE.write_text(json.dumps(data))
                print(f"\n저장됨: {CALIB_FILE} (steer_trim = {trim:+.1f})")
                break
            elif key in ("q", "\x03"):
                print("\n저장하지 않고 종료")
                break
    finally:
        car.close()


if __name__ == "__main__":
    main()
