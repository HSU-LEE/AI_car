"""키보드로 직접 운전

  w / s     속도 +0.1 / -0.1 (음수 = 후진)
  a / d     조향 5도씩 왼쪽 / 오른쪽
  스페이스  정지 (속도 0)
  c         조향 중앙
  q         종료

조향은 car.py의 STEER_LIMIT, 중앙 보정(calib.json)이 그대로 적용된다.
"""
import argparse
import select
import sys
import termios
import time
import tty

from car import STEER_LIMIT, Car

KICK_SPEED, KICK_TIME = 0.6, 0.25   # lane_follow.py와 같은 출발 킥 (낮은 속도에서는 모터가 안 돌기 때문)


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
            prev = speed
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
            if prev == 0 and speed != 0:
                car.drive(KICK_SPEED if speed > 0 else -KICK_SPEED)
                time.sleep(KICK_TIME)
            car.drive(speed)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        car.close()
        print("\n정지")


if __name__ == "__main__":
    main()
