"""모터 최소값 자동 측정 — 차를 트랙(바닥)에 놓고 실행. 앞으로 몇 cm 움직였다가 뒤로 돌아옴

  python calibrate_motor.py          직진 / 많이 꺾은 상태에서 각각 측정해 calib.json에 저장
  python calibrate_motor.py --dry    측정만

엔코더가 없으니 카메라로 판단: 모터값을 조금씩 올리며 잠깐 돌려 보고, 그 전후 화면이 바뀌었으면 '움직였다'.
처음 움직인 값 + 여유(MARGIN)를 MOTOR_MIN(직진) / MOTOR_MIN_TURN(꺾었을 때)으로 저장 →
lane_follow.py가 그보다 낮은 명령은 이 값까지 올려서 보냄 (바닥에서 차가 서 버리는 것 방지)
배터리가 줄면 값이 올라가니 경기 전에 다시 재면 좋음
"""
import argparse
import json
import time

import cv2
import numpy as np

import lane_follow as lf
from car import CALIB_FILE, STEER_LIMIT, Car, load_calib

STEPS = [round(0.2 + 0.04 * i, 2) for i in range(14)]     # 0.20 ~ 0.72
PULSE = 0.5         # 한 번 돌리는 시간(초)
SETTLE = 0.4        # 멈춘 뒤 화면이 안정될 때까지
MARGIN = 0.03       # 처음 움직인 값보다 이만큼 높게 저장 (경계값은 불안정)
TURN_ANGLE = round(STEER_LIMIT * 0.8)   # '많이 꺾은 상태' 측정 각도


def small_gray(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA), (5, 5), 0).astype(np.int16)


def grab(cam):
    cam.capture_array()             # 버퍼에 남아 있던 이전 프레임 버림
    return small_gray(cam.capture_array())


def frame_diff(a, b):
    return float(np.mean(np.abs(a - b)))


def moved_threshold(noise):
    """가만히 있을 때의 화면 변화(noise)보다 확실히 큰 변화만 '움직임'으로"""
    return max(3 * noise, noise + 4.0)


def measure(car, cam, angle, threshold):
    """angle로 꺾고 모터값을 올려 가며 처음 움직인 값. 못 움직이면 None"""
    car.steer(angle)
    time.sleep(0.5)
    for s in STEPS:
        before = grab(cam)
        car.drive(s)
        time.sleep(PULSE)
        car.stop()
        time.sleep(SETTLE)
        d = frame_diff(before, grab(cam))
        print(f"      모터 {s:.2f} → 화면 변화 {d:5.1f} {'움직임' if d > threshold else ''}", flush=True)
        if d > threshold:
            car.drive(-max(s, 0.5))     # 앞으로 간 만큼 뒤로 (후진 최소값은 다를 수 있어서 넉넉히)
            time.sleep(PULSE)
            car.stop()
            time.sleep(SETTLE)
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="측정만 하고 저장 안 함")
    args = ap.parse_args()

    car, cam = Car(), lf.open_camera()
    result = {}
    try:
        car.center()
        time.sleep(1.0)
        noise = max(frame_diff(grab(cam), grab(cam)) for _ in range(5))
        threshold = moved_threshold(noise)
        print(f"[모터 최소값] 가만히 있을 때 화면 변화 {noise:.1f} → 움직임 기준 {threshold:.1f}")
        for key, angle, name in (("motor_min", 0, "직진"), ("motor_min_turn", TURN_ANGLE, f"{TURN_ANGLE}도 꺾음")):
            print(f"  {name}:", flush=True)
            s = measure(car, cam, angle, threshold)
            if s is None:
                print(f"      {STEPS[-1]}까지 올려도 안 움직임 → 저장 안 함 "
                      "(바퀴가 떠 있거나, 모터 배터리/스위치/배선 확인)")
                continue
            result[key] = round(min(s + MARGIN, 1.0), 2)
            print(f"      → {name} 최소값 {result[key]}")
    finally:
        car.close()
        cam.stop()
        cam.close()

    if "motor_min" in result and "motor_min_turn" in result:
        result["motor_min_turn"] = max(result["motor_min_turn"], result["motor_min"])
    if args.dry or not result:
        print("저장 안 함" + (" (--dry)" if args.dry else ""))
    else:
        data = load_calib()
        data.update(result)
        CALIB_FILE.write_text(json.dumps(data))
        print(f"저장: {CALIB_FILE}  {result}  (이전 값: 직진 {lf.MOTOR_MIN}, 꺾음 {lf.MOTOR_MIN_TURN})")


if __name__ == "__main__":
    main()
