"""적외선 거리 센서(샤프 GP2Y0A21) 장애물 감지 — 켜짐/꺼짐을 안정적으로

배선
  빨강(VCC) -> 5V (물리 2번)
  검정(GND) -> GND (물리 9번)
  노랑(Vo)  -> GPIO17 (물리 11번)

센서 출력은 아날로그라 감지 경계 거리에서 0/1이 빠르게 깜빡인다. 그래서
  1) 백그라운드에서 5ms마다 핀을 읽어 최근 WINDOW 동안 '켜짐' 비율을 구하고
  2) 비율이 ON_RATIO 이상이면 '있음', OFF_RATIO 이하로 떨어져야 '없음' (히스테리시스)
  3) '있음'이 되면 최소 HOLD 동안은 유지
해서 깜빡임 없이 판단한다.

사용 (다른 코드에서)
  from obstacle import Obstacle
  obs = Obstacle()
  if obs.near: ...          # 장애물 있음
  obs.close()

사용법 (단독 테스트)
  python obstacle.py        상태가 바뀔 때만 출력, 0.5초마다 켜짐 비율 표시 (Ctrl+C로 종료)
"""
import threading
import time
from collections import deque

OBSTACLE_PIN = 17
SAMPLE_DT = 0.005       # 핀 읽는 간격(초)
WINDOW = 0.15           # 이 시간(초) 동안의 샘플로 판단
ON_RATIO = 0.6          # 켜짐 비율이 이 이상이면 '있음'
OFF_RATIO = 0.2         # 이 이하로 떨어져야 '없음' (중간 값에서는 이전 상태 유지)
HOLD = 0.3              # '있음'이 되면 최소 이 시간(초)은 유지


class Obstacle:
    def __init__(self, pin=OBSTACLE_PIN):
        from gpiozero import DigitalInputDevice
        self.dev = DigitalInputDevice(pin, pull_up=None, active_state=True)
        self.samples = deque(maxlen=max(1, int(WINDOW / SAMPLE_DT)))
        self.ratio = 0.0
        self._near = False
        self._near_since = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(self.dev.value)
            self.ratio = sum(self.samples) / len(self.samples)
            now = time.monotonic()
            if not self._near and self.ratio >= ON_RATIO:
                self._near, self._near_since = True, now
            elif self._near and self.ratio <= OFF_RATIO and now - self._near_since >= HOLD:
                self._near = False
            time.sleep(SAMPLE_DT)

    @property
    def near(self):
        return self._near

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        self.dev.close()


def main():
    obs = Obstacle()
    last, t_print = None, 0.0
    try:
        while True:
            now = time.monotonic()
            if obs.near != last:
                print(f"{'물체 있음' if obs.near else '없음':6s}  (켜짐 비율 {obs.ratio:.2f})", flush=True)
                last = obs.near
            elif now - t_print >= 0.5:
                print(f"         켜짐 비율 {obs.ratio:.2f}", flush=True)
                t_print = now
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        obs.close()


if __name__ == "__main__":
    main()
