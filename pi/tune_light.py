"""Tune traffic-light thresholds for the lighting on site.

Open http://<pi-ip>:8000 with the light in front of the camera:
  Left: camera plus the lamp box. Right: pixels counted as red or green.
  w / s   Saturation +5 / -5   (higher ignores pale color)
  e / d   Brightness +5 / -5   (higher ignores dark color)
  r / f   Minimum size +5 / -5 (higher ignores small reflections)
  Enter   Save and quit (vision.json, used by lane_follow.py on the next run)
  q       Quit without saving

A good value boxes only the red and green lamps.

"lamp H S V" under the image is the median of bright, colored pixels in the lamp area.
  H (hue, 0..180): red is 0..10 or 170..180, green is 40..80.
    wait_green treats hue 35..100 as green, but only at the spot where red was seen.
    A green lamp at H 80..100 looks cyan. wait_green still accepts it; a full-frame search does not.
  If S or V is below the threshold shown (S>=, V>=), lower it with w/s or e/d.
"""
import sys
import termios
import tty

import cv2
import numpy as np

import lane_follow as lf


def lamp_hsv(frame):
    """Median H, S, V and count of bright colored pixels (V>=150, S>=40) in the lamp area."""
    vals = []
    for x0, y0, x1, y1 in lf.TrafficLight.regions():
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
        vals.append(hsv[(hsv[:, 2] >= 150) & (hsv[:, 1] >= 40)])
    px = np.concatenate(vals)
    if len(px) < 10:
        return "lamp -"
    h, s, v = np.median(px, axis=0).astype(int)
    return f"lamp H{h} S{s} V{v} ({len(px)}px)"


def main():
    print(__doc__)
    cam = lf.open_camera()
    stream = lf.Stream(8000, "AICAR traffic light", "w/s saturation · e/d brightness · r/f size · Enter twice save · q twice quit without saving", confirm="\nq")
    traffic = lf.TrafficLight()
    traffic.keep_mask = True
    steps = {"w": ("LIGHT_S_MIN", 5), "s": ("LIGHT_S_MIN", -5),
             "e": ("LIGHT_V_MIN", 5), "d": ("LIGHT_V_MIN", -5),
             "r": ("LIGHT_AREA_MIN", 5), "f": ("LIGHT_AREA_MIN", -5)}

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            frame = cam.capture_array()
            state = traffic(frame)
            view = frame.copy()
            traffic.draw(view)
            info = (f"S>={lf.LIGHT_S_MIN} V>={lf.LIGHT_V_MIN} area>={lf.LIGHT_AREA_MIN}  -> {state or '-'}  "
                    f"{lamp_hsv(frame)}")
            cv2.putText(view, info, (10, lf.FRAME_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(traffic.mask_vis, "[w/s sat  e/d bright  r/f size  Enter save]", (10, lf.FRAME_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            stream.update(np.hstack([view, traffic.mask_vis]))
            print(f"\r{info}      ", end="", flush=True)

            for key in stream.read_keys():
                if key in steps:
                    name, step = steps[key]
                    setattr(lf, name, int(max(0, min(255 if name != "LIGHT_AREA_MIN" else 2000,
                                                     getattr(lf, name) + step))))
                elif key in ("\n", "\r"):
                    lf.save_vision(light_s_min=lf.LIGHT_S_MIN, light_v_min=lf.LIGHT_V_MIN,
                                   light_area_min=lf.LIGHT_AREA_MIN)
                    print(f"\nSaved: {lf.VISION_FILE}")
                    return
                elif key == "q":
                    print("\nQuit without saving")
                    return
    except KeyboardInterrupt:
        print("\nQuit without saving")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        cam.stop()


if __name__ == "__main__":
    main()
