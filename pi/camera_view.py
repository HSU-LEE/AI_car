"""카메라 원본 화면 + 차선 인식 화면을 나란히 브라우저로 보기 (모터/서보는 움직이지 않음)

  python camera_view.py             맥 브라우저에서 http://<파이 IP>:8000
  python camera_view.py --port 8001

왼쪽: 카메라 원본 / 오른쪽: lane_follow.py가 보는 인식 결과와 그때 줄 조향각
Ctrl+C로 종료
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
    print(f"브라우저: http://<파이 IP>:{args.port}  | Ctrl+C로 종료")

    frames, t0, fps = 0, time.monotonic(), 0.0
    try:
        while True:
            frame = cam.capture_array()
            now = time.monotonic()
            det, junction, dbg = detect(frame)
            light = traffic(frame)
            traffic.draw(dbg)
            raw = frame.copy()
            cv2.putText(raw, f"CAMERA  {fps:4.1f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if det is None:
                state = "LOST"
            else:
                state = f"off {det[0]:+.2f} head {det[1]:+.2f}  steer {steering(*det, now):+.1f}"
            # 선이 보이는 위치(화면 높이 비율) = course.py at= 에 넣을 값
            if detect.junction_y is not None:
                state += f"  JUNCTION {detect.junction_y / lf.FRAME_H:.2f}"
            if detect.crosswalk_y is not None:
                state += f"  CROSSWALK {detect.crosswalk_y / lf.FRAME_H:.2f}"
            if detect.gap_y is not None:
                state += f"  GAP {detect.gap_y / lf.FRAME_H:.2f}"
            if light:
                state += f"  LIGHT {light.upper()}"
            cv2.putText(dbg, f"DETECT  {state}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            stream.update(np.hstack([raw, dbg]))

            frames += 1
            if now - t0 >= 1:
                fps, frames, t0 = frames / (now - t0), 0, now
                print(f"\r{fps:4.1f} fps  {state}        ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
