"""Register the LEGO figure of the day and check that it is found. Normally started by `aicar person`, which asks new or add first.

Stand the figure inside the yellow guide box on http://<pi-ip>:8000, where the car should stop. The car should be on the track, as while driving.
Auto capture is on from the start: turn the figure, take your hand away, and it takes a shot once the view is still. Enter saves.
Or drag a box around the figure on the page. That takes a shot at once with all of the box's figure, whatever its colors, and later shots use the same box.

  python tune_person.py          new figure (the saved one stays until Enter)
  python tune_person.py --add    add views to the saved figure, so several figures are all found

Left: camera. Green box = found. Grey box = right size, but wrong colors (c) or outline (s). Thin yellow box = what the last shot took as the figure.
Right: pixels whose color counts as the figure.
"""
import argparse
import os
import select
import sys
import termios
import time
import tty

import cv2
import numpy as np

import lane_follow as lf
from person import AutoViews, PersonDetector, guide_px

VIEW_FRAMES = 5
BOX_STEP = 0.03         # Guide box step per key press, as a fraction of the frame.
BOX_MIN = 0.15          # Smallest guide box side, as a fraction of the frame.
MOVES = {"w": (0, -1), "s": (0, 1), "a": (-1, 0), "d": (1, 0)}
# The same physical keys while the keyboard is in Korean mode.
HANGUL = dict(zip("ㅈㅁㄴㅇㅋㅌㅅㅕㅂ", "wasdzxtuq"))
HELP = """
  레고를 노란 네모 안에 세우세요. 레고 전체가 들어오게. 발 위치가 정지 위치가 됩니다.
  레고를 돌리고 손을 빼면 저절로 찍힙니다. 앞, 옆, 뒤를 찍고 엔터를 누르세요.
  잘 안 잡히면 웹페이지 카메라 화면에서 레고를 드래그해 감싸세요. 바로 한 장 찍고, 다음 사진도 그 네모로 찍습니다.

    엔터       저장하고 끝내기
    w a s d    네모 옮기기          z / x    네모 작게 / 크게
    스페이스   지금 한 장 찍기      t        자동 촬영 끄기 / 켜기
    u          마지막 한 장 취소    (인식 기준은 찍을 때마다 자동으로 정해집니다)
    q          저장하지 않고 끝내기
"""


def grab(cam, n):
    frames = [cam.capture_array() for _ in range(n)]
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def show_at(detector):
    """The at height driving will use, as set in course.py."""
    return lf.PERSON_AT if lf.PERSON_AT is not None else detector.stop_at(lf.FRAME_H)


def move_box(box, dx, dy):
    """Guide box moved by whole steps, staying in the frame."""
    x0, y0, x1, y1 = box
    dx = float(np.clip(dx * BOX_STEP, -x0, 1 - x1))
    dy = float(np.clip(dy * BOX_STEP, -y0, 1 - y1))
    return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)


def resize_box(box, steps):
    """Guide box grown or shrunk around its center, staying in the frame."""
    x0, y0, x1, y1 = box
    grow = steps * BOX_STEP / 2
    if x1 - x0 + 2 * grow < BOX_MIN or y1 - y0 + 2 * grow < BOX_MIN:
        return box
    return (max(x0 - grow, 0.0), max(y0 - grow, 0.0), min(x1 + grow, 1.0), min(y1 + grow, 1.0))


def read_keys(fd):
    """Keys waiting on the terminal, in English-mode letters. Read as bytes, because a Korean key is several bytes and the Pi's locale may not be UTF-8."""
    if not select.select([fd], [], [], 0)[0]:
        return ""
    text = os.read(fd, 64).decode("utf-8", "ignore")
    return "".join(HANGUL.get(c, c.lower()) for c in text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="store_true", help="저장된 사람에 시점을 더함")
    args = ap.parse_args()

    detector = PersonDetector.load(lf.HORIZON_Y) if args.add else None
    if args.add:
        print(f"\n  추가 등록: 저장된 시점 {len(detector.views)}개에 더합니다." if detector else "\n  저장된 사람이 없어 새로 등록합니다.")
    else:
        print("\n  재등록: 엔터로 저장할 때 기존 사람이 지워집니다. q로 나가면 그대로입니다.")
    detector = detector or PersonDetector(lf.HORIZON_Y)
    print(HELP)
    if lf.CAMERA_LOCK is None:
        print("  [주의] 노출이 고정되어 있지 않습니다. 먼저 aicar calibrate 를 하면 더 안정적입니다.\n")
    cam = lf.open_camera()
    stream = lf.Stream(8000, "AICAR 레고 사람", "드래그 레고 지정 · Enter 두 번 저장 · wasd 네모 이동 · z/x 크기 · Space 찍기 · t 자동 촬영 · u 취소 · q 두 번 저장 안 하고 끝", confirm="\nq", drag_share=0.5)
    auto = AutoViews(detector)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    done = False
    try:
        while not done:
            frame = cam.capture_array()
            found = detector(frame, lf.PERSON_X)
            view = frame.copy()
            lf.draw_person(view, detector, show_at(detector))
            x0, y0, x1, y1 = guide_px(frame.shape, detector.box)
            cv2.rectangle(view, (x0, y0), (x1, y1), lf.C_AT, 2)
            lf.chip(view, "FIGURE" if detector.marked else "STAND FIGURE IN HERE", x0, y0 - 22, lf.C_AT, 0.38)
            if detector.last_box:
                x, y, w, h = detector.last_box
                cv2.rectangle(view, (x, y), (x + w, y + h), lf.C_AT, 1)
                lf.chip(view, f"VIEW {len(detector.views)}", x, y + h + 2, lf.C_AT, 0.35)
            cv2.putText(view, f"{'AUTO  ' if auto else ''}views {len(detector.views)}  " + (
                f"FOUND c{found['score']:.2f} s{found['shape']:.2f} feet {found['feet'] / lf.FRAME_H:.2f}" if found else "-"),
                (10, lf.FRAME_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            mask = np.zeros_like(frame)
            if detector.fg is not None:
                mask[cv2.resize(detector.fg, (lf.FRAME_W, lf.FRAME_H), interpolation=cv2.INTER_NEAREST) > 0] = (60, 220, 60)
            stream.update(np.hstack([view, mask]))
            print(f"\r  찍은 장수 {len(detector.views)}  |  자동 촬영 {'켜짐' if auto else '꺼짐'}  |  "
                  f"{'인식됨' if found else '인식 안 됨'}      ", end="", flush=True)
            note = auto.update(frame, time.monotonic()) if auto else None

            dragged = stream.read_box()
            if dragged:
                detector.box, detector.marked = dragged, True
                note = detector.add_view(grab(cam, VIEW_FRAMES))[1]
                if auto:
                    auto.touch()
                    auto.ready = False      # Just taken. The next one waits until the figure has been turned.

            for key in read_keys(fd) + stream.read_keys(terminal=False).lower():
                if key in MOVES or key in ("z", "x"):
                    detector.box = move_box(detector.box, *MOVES[key]) if key in MOVES else resize_box(detector.box, 1 if key == "x" else -1)
                    if auto:
                        auto.touch()    # The shot waits until the box has stopped moving.
                elif key == " ":
                    note = detector.add_view(grab(cam, VIEW_FRAMES))[1]
                elif key == "t":
                    auto = None if auto else AutoViews(detector)
                elif key == "u":
                    detector.undo()
                    note = f"마지막 한 장을 취소했습니다 (남은 장수 {len(detector.views)})"
                elif key in ("\n", "\r"):
                    if detector.views:
                        detector.save()
                        print(f"\n\n  저장했습니다: {len(detector.views)}장. 다음 주행부터 적용됩니다.")
                        done = True
                    else:
                        note = "아직 찍은 사진이 없어 저장할 수 없습니다"
                elif key == "q":
                    print("\n\n  저장하지 않고 끝냅니다.")
                    done = True
                if done:
                    break
            if note and not done:
                print(f"\n  {note}")
    except KeyboardInterrupt:
        print("\n\n  저장하지 않고 끝냅니다.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        cam.stop()


if __name__ == "__main__":
    main()
