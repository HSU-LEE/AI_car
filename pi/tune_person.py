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
  Stand the LEGO figure inside the yellow box, all of it. Its feet set where the car stops.
  Turn the figure and take your hand away: a view is taken by itself. Take front, side and back, then press Enter.
  If it is not found, drag around the figure on the web page camera view. That takes a view at once, and later ones use that box.

    Enter      save and quit
    w a s d    move the box         z / x    box smaller / larger
    Space      take a view now      t        auto capture off / on
    u          undo the last view   (thresholds are set automatically with each view)
    q          quit without saving
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
    ap.add_argument("--add", action="store_true", help="add views to the saved figure")
    args = ap.parse_args()

    detector = PersonDetector.load(lf.HORIZON_Y) if args.add else None
    if args.add:
        print(f"\n  Adding: to the {len(detector.views)} saved views." if detector else "\n  No saved figure: registering a new one.")
    else:
        print("\n  New registration: saving with Enter replaces the saved figure. q leaves it as it is.")
    detector = detector or PersonDetector(lf.HORIZON_Y)
    print(HELP)
    if lf.CAMERA_LOCK is None:
        print("  [warning] Exposure is not locked. Run aicar calibrate first for steadier results.\n")
    cam = lf.open_camera()
    stream = lf.Stream(8000, "AICAR LEGO figure", "drag the figure · Enter twice save · wasd move box · z/x size · Space take · t auto · u undo · q twice quit without saving", confirm="\nq", drag_share=0.5)
    auto = AutoViews(detector)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    done = False
    try:
        while not done:
            frame = cam.capture_array()
            found = detector(frame, lf.PERSON_X, show_at(detector) * lf.FRAME_H if detector.views else None)
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
            print(f"\r  views {len(detector.views)}  |  auto capture {'on' if auto else 'off'}  |  "
                  f"{'found' if found else 'not found'}      ", end="", flush=True)
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
                    note = f"Last view undone ({len(detector.views)} left)"
                elif key in ("\n", "\r"):
                    if detector.views:
                        detector.save()
                        print(f"\n\n  Saved: {len(detector.views)} views. Used from the next drive.")
                        done = True
                    else:
                        note = "No views yet, nothing to save"
                elif key == "q":
                    print("\n\n  Quit without saving.")
                    done = True
                if done:
                    break
            if note and not done:
                print(f"\n  {note}")
    except KeyboardInterrupt:
        print("\n\n  Quit without saving.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        cam.stop()


if __name__ == "__main__":
    main()
