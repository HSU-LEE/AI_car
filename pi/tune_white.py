"""Tune the white-line threshold.

Automatic calibration is the default. Set a value by hand only when the light
misses the line or paints the floor.

Open http://10.42.0.1:8000 and watch the orange pixels (pixels treated as the line):
  w / s   Threshold +2 / -2   (higher catches less, lower catches more)
  W / S   Threshold +10 / -10
  a       Return to automatic
  Enter   Save and quit (vision.json, used by lane_follow.py on the next run)
  q       Quit without saving

A good value paints the white line and nothing else.
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
