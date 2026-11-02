"""서보 중앙각 튜닝

바퀴가 똑바로 설 때까지 키로 조정한 뒤 s로 저장하면 calib.json에 기록되고,
car.py가 다음 실행부터 자동으로 반영합니다.

  z / c   0.1도씩 왼쪽 / 오른쪽
  a / d   0.5도씩 왼쪽 / 오른쪽
  A / D   2도씩 왼쪽 / 오른쪽
  r       보정값 0으로 초기화
  s       저장 후 종료
  q       저장하지 않고 종료
"""
import json
import sys
import termios
import tty

from car import CALIB_FILE, Car, load_trim

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
                # 중앙을 새로 맞췄으니 주행 중 자동으로 배운 보정값은 초기화
                CALIB_FILE.write_text(json.dumps({"steer_trim": trim, "steer_bias": 0.0}))
                print(f"\n저장됨: {CALIB_FILE} (steer_trim = {trim:+.1f})")
                break
            elif key in ("q", "\x03"):
                print("\n저장하지 않고 종료")
                break
    finally:
        car.close()


if __name__ == "__main__":
    main()
