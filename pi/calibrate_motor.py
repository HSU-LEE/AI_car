"""Measure the lowest motor command that actually rolls the car. Place it on the track.
It creeps forward a few centimeters, then backs up.

  python calibrate_motor.py          Measure straight and while steered, then save calib.json.
  python calibrate_motor.py --dry    Measure only.

There is no encoder, so the camera decides: raise the motor command, pulse it,
and treat a changed frame as motion. The first value that moves the car, plus MARGIN,
is saved as MOTOR_MIN (straight) and MOTOR_MIN_TURN (steered). lane_follow.py
raises any lower command to that floor so the car does not stall on the floor.
Re-measure before a run if the battery is lower.
"""
import argparse
import json
import time

import cv2
import numpy as np

import lane_follow as lf
from car import CALIB_FILE, STEER_LIMIT, Car, load_calib

STEPS = [round(0.2 + 0.04 * i, 2) for i in range(14)]     # 0.20 .. 0.72
PULSE = 0.5         # How long each pulse runs, seconds.
SETTLE = 0.4        # Wait after stopping so the picture is still.
MARGIN = 0.03       # Save this much above the first value that moved. The edge value is unreliable.
TURN_ANGLE = round(STEER_LIMIT * 0.8)   # Steer angle used for the "hard turn" measurement.


def small_gray(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA), (5, 5), 0).astype(np.int16)


def grab(cam):
    cam.capture_array()             # Drop the frame still sitting in the buffer.
    return small_gray(cam.capture_array())


def frame_diff(a, b):
    return float(np.mean(np.abs(a - b)))


def moved_threshold(noise):
    """A change clearly larger than the still-frame noise. +6 avoids a borderline false trigger."""
    return max(3 * noise, noise + 6.0)


def measure(car, cam, angle, threshold):
    """Steer to angle and raise the motor command until the car moves. None if it never does."""
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
            car.drive(-max(s, 0.5))     # Back up about as far as it rolled. Reverse may need a higher floor.
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
