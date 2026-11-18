"""Sharp GP2Y0A21 obstacle detector with a stable on/off reading.

Wiring:
  Red (VCC)   -> 5V (physical pin 2)
  Black (GND) -> GND (physical pin 9)
  Yellow (Vo) -> GPIO17 (physical pin 11)

The analog output chatters at the detection edge, so this:
  1. Samples the pin every 5 ms and keeps the on-ratio over WINDOW.
  2. Turns on at ON_RATIO and off only at OFF_RATIO (hysteresis).
  3. Holds an on reading for at least HOLD seconds.

From other code:
  from obstacle import Obstacle
  obs = Obstacle()
  if obs.near: ...
  obs.close()

  python obstacle.py    Print on change, and the on-ratio every 0.5 s. Ctrl+C to stop.
"""
import threading
import time
from collections import deque

OBSTACLE_PIN = 17
SAMPLE_DT = 0.005       # Sample period, seconds.
WINDOW = 0.15           # Window used for the on-ratio.
ON_RATIO = 0.6          # On-ratio at or above this means an obstacle.
OFF_RATIO = 0.2         # Must fall to this or below before it clears.
HOLD = 0.3              # Minimum time an on reading is kept, seconds.
FAIL_READS = 20         # Failed reads in a row (0.1 s) before the sensor counts as broken.


class Obstacle:
    def __init__(self, pin=OBSTACLE_PIN):
        from gpiozero import DigitalInputDevice
        self.dev = DigitalInputDevice(pin, pull_up=None, active_state=True)
        self.samples = deque(maxlen=max(1, int(WINDOW / SAMPLE_DT)))
        self.ratio = 0.0
        self._near = False
        self._near_since = 0.0
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        fails = 0
        while not self._stop.is_set():
            try:
                value = self.dev.value
                fails = 0
            except Exception as e:      # Kept failing: the thread stops and near reads as an obstacle.
                fails += 1
                if fails >= FAIL_READS:
                    self.error = e
                    print(f"[error] obstacle sensor stopped: {e}", flush=True)
                    return
                time.sleep(SAMPLE_DT)
                continue
            self.samples.append(value)
            self.ratio = sum(self.samples) / len(self.samples)
            now = time.monotonic()
            if not self._near and self.ratio >= ON_RATIO:
                self._near, self._near_since = True, now
            elif self._near and self.ratio <= OFF_RATIO and now - self._near_since >= HOLD:
                self._near = False
            time.sleep(SAMPLE_DT)

    @property
    def near(self):
        """An obstacle in front. A sensor that stopped working reads as one: go_dis() then stops, and a wait for
        clear (start_gate) keeps waiting, instead of driving on with the last reading."""
        return self._near or self.error is not None or (not self._thread.is_alive() and not self._stop.is_set())

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
                print(f"{'object' if obs.near else 'clear':6s}  (on share {obs.ratio:.2f})", flush=True)
                last = obs.near
            elif now - t_print >= 0.5:
                print(f"         on share {obs.ratio:.2f}", flush=True)
                t_print = now
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        obs.close()


if __name__ == "__main__":
    main()
