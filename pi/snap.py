"""카메라 사진 한 장 찍기 (lane_follow.py와 같은 해상도/초점)

  python snap.py              snap.jpg 로 저장
  python snap.py out.jpg
"""
import sys
import time

import cv2

import lane_follow as lf


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "snap.jpg"
    cam = lf.open_camera()
    try:
        time.sleep(1)   # 노출이 자리 잡을 때까지
        cv2.imwrite(out, cam.capture_array(), [cv2.IMWRITE_JPEG_QUALITY, 95])
    finally:
        cam.stop()
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
