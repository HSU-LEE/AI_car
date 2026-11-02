"""흰 선 인식 기준값 튜닝

기본은 자동 캘리브레이션이라 따로 할 필요는 없다. 조명이 특이해서 선을 놓치거나
엉뚱한 곳을 선으로 잡을 때만 직접 정하면 된다.

맥 브라우저에서 http://10.42.0.1:8000 을 열고 주황색(선으로 인식된 부분)을 보면서 조정:
  w / s   기준값 +2 / -2  (올리면 덜 잡고, 내리면 더 잡음)
  W / S   기준값 +10 / -10
  a       자동으로 되돌리기
  Enter   저장 후 종료 (vision.json → lane_follow.py가 다음 실행부터 사용)
  q       저장하지 않고 종료

주황색이 흰 선 위에만 꽉 차게 칠해지고, 바닥/반사광에는 안 칠해지는 값이 좋다.
"""
import select
import sys
import termios
import tty

import cv2

import lane_follow as lf


def main():
    print(__doc__)
    cam = lf.open_camera()
    stream = lf.Stream(8000)
    detect = lf.LaneDetector()
    steps = {"w": 2, "s": -2, "W": 10, "S": -10}

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            _, _, dbg = detect(cam.capture_array())
            mode = "AUTO" if lf.WHITE_THR is None else "MANUAL"
            cv2.putText(dbg, f"{mode} white {detect.thr:.0f}  [w/s +-2  W/S +-10  a auto  Enter save]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            stream.update(dbg)
            print(f"\r{mode:6s} white = {detect.thr:5.1f}   ", end="", flush=True)

            if not select.select([sys.stdin], [], [], 0)[0]:
                continue
            key = sys.stdin.read(1)
            if key in steps:
                base = detect.thr if lf.WHITE_THR is None else lf.WHITE_THR
                lf.WHITE_THR = int(max(5, min(250, base + steps[key])))
            elif key == "a":
                lf.WHITE_THR = None
            elif key in ("\n", "\r"):
                lf.save_vision(white_thr=lf.WHITE_THR)
                print(f"\n저장됨: {lf.VISION_FILE} (white_thr = {lf.WHITE_THR or '자동'})")
                break
            elif key == "q":
                print("\n저장하지 않고 종료")
                break
    except KeyboardInterrupt:
        print("\n저장하지 않고 종료")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        cam.stop()


if __name__ == "__main__":
    main()
