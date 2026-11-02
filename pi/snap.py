"""Save one camera frame at the same size and focus as lane_follow.py.

  python snap.py              Write snap.jpg.
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
        time.sleep(1)   # Let exposure settle.
        cv2.imwrite(out, cam.capture_array(), [cv2.IMWRITE_JPEG_QUALITY, 95])
    finally:
        cam.stop()
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
