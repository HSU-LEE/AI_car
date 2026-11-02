"""신호등 인식 기준값 튜닝 (현장 조명에 맞추기)

맥 브라우저에서 http://<파이 IP>:8000 을 열고 신호등을 카메라 앞에 둔 채 조정:
  왼쪽: 카메라 화면 + 인식된 램프 박스 / 오른쪽: 빨강·초록으로 잡힌 픽셀
  w / s   채도 기준 +5 / -5   (올리면 옅은 색은 무시)
  e / d   밝기 기준 +5 / -5   (올리면 어두운 색은 무시)
  r / f   최소 크기 +5 / -5   (올리면 작은 반사광은 무시)
  Enter   저장 후 종료 (vision.json → lane_follow.py가 다음 실행부터 사용)
  q       저장하지 않고 종료

빨간불/초록불 램프에만 박스가 생기고, 주변 물체·반사광에는 안 생기는 값이 좋다.

화면 아래 "lamp H S V"는 신호등 영역에서 밝고 색이 있는 픽셀의 중간값 (램프 색 확인용)
  H(색상, 0~180): 빨강 0~10 또는 170~180, 초록 40~80 (wait_green은 빨간불 자리에서 35~100까지 초록으로 봄)
  초록불인데 H가 80~100이면 청록빛으로 찍히는 것 → wait_green은 괜찮지만 화면 전체 인식은 못 함
  S/V가 위 기준(S>=, V>=)보다 낮으면 w/s, e/d로 기준을 낮추기
"""
import select
import sys
import termios
import tty

import cv2
import numpy as np

import lane_follow as lf


def lamp_hsv(frame):
    """신호등 영역에서 밝고 색이 있는 픽셀(V>=150, S>=40)의 H, S, V 중간값과 개수"""
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
    stream = lf.Stream(8000)
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

            if not select.select([sys.stdin], [], [], 0)[0]:
                continue
            key = sys.stdin.read(1)
            if key in steps:
                name, step = steps[key]
                setattr(lf, name, int(max(0, min(255 if name != "LIGHT_AREA_MIN" else 2000,
                                                 getattr(lf, name) + step))))
            elif key in ("\n", "\r"):
                lf.save_vision(light_s_min=lf.LIGHT_S_MIN, light_v_min=lf.LIGHT_V_MIN,
                               light_area_min=lf.LIGHT_AREA_MIN)
                print(f"\n저장됨: {lf.VISION_FILE}")
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
