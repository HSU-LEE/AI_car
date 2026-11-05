"""Keyboard drive.

  w / s     Speed +0.1 / -0.1 (negative = reverse)
  a / d     Steer 5 degrees left / right
  space     Stop (speed 0)
  c         Center the steering
  q         Quit

Steering uses car.py STEER_LIMIT and the center trim in calib.json.
"""
import argparse
import select
import sys
import termios
import tty

from car import STEER_LIMIT, Car



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-speed", type=float, default=0.6, help="최대 속도 (0 ~ 1)")
    args = ap.parse_args()

    print(__doc__)
    car = Car()
    car.center()
    speed, angle = 0.0, 0.0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            print(f"\r속도 {speed:+.1f}  조향 {angle:+3.0f}도   ", end="", flush=True)
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                continue
            key = sys.stdin.read(1)
            if key == "w":
                speed = min(args.max_speed, round(speed + 0.1, 1))
            elif key == "s":
                speed = max(-args.max_speed, round(speed - 0.1, 1))
            elif key == "a":
                angle = max(-STEER_LIMIT, angle - 5)
            elif key == "d":
                angle = min(STEER_LIMIT, angle + 5)
            elif key == "c":
                angle = 0.0
            elif key == " ":
                speed = 0.0
            elif key == "q":
                break
            car.steer(angle)
            car.drive(speed)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        car.close()
        print("\n정지")


if __name__ == "__main__":
    main()
