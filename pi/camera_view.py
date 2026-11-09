"""Show the raw camera and the lane view side by side. Motors and servo stay off.

  python camera_view.py             Open http://<pi-ip>:8000
  python camera_view.py --port 8001

Left: raw frame. Right: what lane_follow.py sees, the steer it would command, and the registered LEGO figure.
Ctrl+C to stop.
"""
import argparse
import time

import cv2
import numpy as np

import lane_follow as lf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    cam = lf.open_camera()
    stream = lf.Stream(args.port)
    detect, steering, traffic = lf.LaneDetector(), lf.Steering(), lf.TrafficLight()
    person = lf.load_person()
    if person:
        person.place = detect.in_lane if lf.PERSON_IN_LANE else None
    print(f"Browser: http://<Pi IP>:{args.port}  | Ctrl+C to quit")

    frames, t0, fps = 0, time.monotonic(), 0.0
    try:
        while True:
            frame = cam.capture_array()
            now = time.monotonic()
            det, junction, dbg = detect(frame)
            light = traffic(frame)
            traffic.draw(dbg)
            found = person(frame, lf.PERSON_X, (lf.PERSON_AT if lf.PERSON_AT is not None else person.stop_at(lf.FRAME_H)) * lf.FRAME_H) if person else None
            if person:
                lf.draw_person(dbg, person, lf.PERSON_AT if lf.PERSON_AT is not None else person.stop_at(lf.FRAME_H))
            raw = frame.copy()
            cv2.putText(raw, f"CAMERA  {fps:4.1f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if det is None:
                state = "LOST"
            else:
                state = f"off {det[0]:+.2f} head {det[1]:+.2f}  steer {steering(*det, now):+.1f}"
            # Line position as a fraction of frame height. That number is course.py at=.
            if detect.junction_y is not None:
                state += f"  JUNCTION {detect.junction_y / lf.FRAME_H:.2f}"
            if detect.crosswalk_y is not None:
                state += f"  CROSSWALK {detect.crosswalk_y / lf.FRAME_H:.2f}"
            if detect.gap_y is not None:
                state += f"  GAP {detect.gap_y / lf.FRAME_H:.2f}"
            if light:
                state += f"  LIGHT {light.upper()}"
            if found:
                state += f"  PERSON {found['feet'] / lf.FRAME_H:.2f}"
            cv2.putText(dbg, f"DETECT  {state}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            stream.update(np.hstack([raw, dbg]))
            stream.keys.take()      # Reads no keys. Ctrl+C on the page ends this, like in the terminal.

            frames += 1
            if now - t0 >= 1:
                fps, frames, t0 = frames / (now - t0), 0, now
                print(f"\r{fps:4.1f} fps  {state}        ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
