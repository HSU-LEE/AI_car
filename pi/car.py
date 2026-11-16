"""AICAR steering servo and drive motor.

Wiring (TB6612FNG channel A):
  Servo signal -> GPIO18 (physical pin 12)
  PWMA         -> GPIO13 (physical pin 33), hardware PWM at MOTOR_PWM_HZ
  AIN1         -> GPIO23 (physical pin 16)
  AIN2         -> GPIO24 (physical pin 18)
  STBY, VCC    -> 3.3V
  Common GND

Usage:
  python car.py center          Center the servo.
  python car.py steer 20        Steer for 1s, then recenter (-35..35, negative = left).
                                Wheels should turn right. If they turn left, set STEER_INVERT = True.
  python car.py drive 0.4 2     Drive at speed -1..1 (negative = reverse) for 2 seconds.
  python car.py test            Steer, then forward and reverse.
  python car.py trim -4.2       Shift the saved center trim by -4.2 degrees.
"""
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from time import sleep

SERVO_PIN = 18
MOTOR_PWM, MOTOR_IN1, MOTOR_IN2 = 13, 23, 24

STEER_LIMIT = 26        # Max servo angle either side of the trimmed center.
STEER_SCALE = 1.0       # Servo degrees per commanded degree. 1.0 means they match.
STEER_INVERT = True     # Flip the sign when a positive angle turns the wheels left.
CALIB_FILE = Path(__file__).with_name("calib.json")


def load_calib():
    try:
        return json.loads(CALIB_FILE.read_text())
    except (OSError, ValueError):
        return {}


def load_trim():
    """Center trim saved in calib.json by tune_center.py."""
    return float(load_calib().get("steer_trim", 0.0))


STEER_TRIM = load_trim()
BIAS_MAX = 8    # Largest automatic center correction, in degrees.


MOTOR_PWM_HZ = 20_000   # Motor speed PWM on GPIO13 in hardware: steady whatever the CPU is doing, and above hearing.
                        # (The old gpiozero software PWM ran at 100 Hz and wavered while the vision code kept the CPU busy.)


class SysfsPWM:
    """One channel of the Pi's hardware PWM (sysfs). Channel 0 is GPIO18, channel 1 is GPIO13.

    Requires `dtoverlay=pwm-2chan,pin=18,func=2,pin2=13,func2=4` in /boot/firmware/config.txt.
    """

    def __init__(self, channel, period_ns):
        chips = sorted(Path("/sys/class/pwm").glob("pwmchip*"))
        if not chips:
            raise OSError("No hardware PWM (check the overlay in config.txt)")
        self.chip, self.channel = chips[0], channel
        if int((self.chip / "npwm").read_text()) <= channel:
            raise OSError(f"No hardware PWM channel {channel}")
        self.pwm = self.chip / f"pwm{channel}"
        if not self.pwm.exists():
            (self.chip / "export").write_text(str(channel))
        self._write("duty_cycle", 0)        # A period shorter than the old duty cycle is refused, so clear it first.
        self._write("period", period_ns)
        self.period_ns = period_ns

    def _write(self, name, value):
        # Right after export, udev may not have granted write access yet.
        for _ in range(50):
            try:
                (self.pwm / name).write_text(str(value))
                return
            except PermissionError:
                sleep(0.02)
        (self.pwm / name).write_text(str(value))

    def duty(self, ns):
        self._write("duty_cycle", int(ns))
        self._write("enable", 1)

    def close(self):
        try:
            self._write("duty_cycle", 0)
            self._write("enable", 0)
            (self.chip / "unexport").write_text(str(self.channel))
        except OSError:
            pass


class HardwareServo(SysfsPWM):
    """Servo driven by the GPIO18 hardware PWM (channel 0), steady below 1 degree."""
    PERIOD_NS = 20_000_000   # 50 Hz.

    def __init__(self, min_angle=-90, max_angle=90,
                 min_pulse_width=0.0005, max_pulse_width=0.0025):
        super().__init__(0, self.PERIOD_NS)
        self.min_angle, self.max_angle = min_angle, max_angle
        self.min_ns, self.max_ns = min_pulse_width * 1e9, max_pulse_width * 1e9
        self._angle = None

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        value = max(self.min_angle, min(self.max_angle, value))
        ratio = (value - self.min_angle) / (self.max_angle - self.min_angle)
        self._write("duty_cycle", int(self.min_ns + ratio * (self.max_ns - self.min_ns)))
        self._write("enable", 1)
        self._angle = value


GUARD_SEC = 2.0     # MotorGuard: the motor is switched off if the program gives no sign of life for this long.

# Runs as its own small process (python -c), so it outlives the program it watches. Reads a pipe from it:
# a byte now and then = alive, "q" = closed cleanly, end of file = the program died. Silence or death -> duty 0.
_GUARD_CODE = """
import os, select, signal, sys
duty, limit = sys.argv[1], float(sys.argv[2])
for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(s, signal.SIG_IGN)
def off():
    try:
        with open(duty, "w") as f:
            f.write("0")
    except OSError:
        pass
while True:
    ready, _, _ = select.select([0], [], [], limit)
    if not ready:
        off()
        continue
    data = os.read(0, 4096)
    if not data:
        off()
        break
    if b"q" in data:
        break
"""


class MotorGuard:
    """Switches the motor PWM off if the program driving it dies or freezes.

    Hardware PWM keeps its last duty after the process is gone (kill -9, a crash, out of memory), and the direction
    pins keep their last level too, so the car would drive on at that speed with nothing left to stop it. The watchdog
    thread in lane_follow.py dies with the process. This is a separate process holding the other end of a pipe:
    when the pipe closes or goes quiet for GUARD_SEC, it writes duty 0 (PWMA low = motor braked).
    """

    def __init__(self, duty_file, limit=GUARD_SEC):
        self.proc = subprocess.Popen([sys.executable, "-S", "-c", _GUARD_CODE, str(duty_file), str(limit)],
                                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     start_new_session=True, close_fds=True)     # Own session: Ctrl+C in the terminal does not reach it.
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, args=(limit / 5,), daemon=True)
        self._thread.start()

    def _send(self, data):
        try:
            os.write(self.proc.stdin.fileno(), data)
            return True
        except (OSError, ValueError):
            return False

    def _beat(self, every):
        while not self._stop.wait(every):
            if not self._send(b"."):
                return

    def close(self):
        """Clean exit: the motor is already stopped by the program, so the guard just leaves."""
        self._stop.set()
        self._thread.join(timeout=1)
        self._send(b"q")
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass


class HardwareMotor:
    """TB6612 channel A: speed on PWMA (GPIO13 hardware PWM, MOTOR_PWM_HZ), direction on AIN1 / AIN2.

    Behaves like gpiozero Motor(forward, backward, enable): forward(speed), backward(speed), stop() (both inputs low).
    """

    def __init__(self):
        from gpiozero import DigitalOutputDevice
        self.pwm = SysfsPWM(1, round(1e9 / MOTOR_PWM_HZ))
        self.in1, self.in2 = DigitalOutputDevice(MOTOR_IN1), DigitalOutputDevice(MOTOR_IN2)
        try:
            self.guard = MotorGuard(self.pwm.pwm / "duty_cycle")
        except OSError as e:
            self.guard = None
            print(f"[warning] motor guard not started ({e}): a crash of this program would leave the motor running")

    def _run(self, a, b, speed):
        self.in1.value, self.in2.value = a, b
        self.pwm.duty(max(0.0, min(1.0, speed)) * self.pwm.period_ns)

    def forward(self, speed=1.0):
        self._run(1, 0, speed)

    def backward(self, speed=1.0):
        self._run(0, 1, speed)

    def stop(self):
        self._run(0, 0, 0.0)

    def close(self):
        self.stop()
        if self.guard:
            self.guard.close()
        self.pwm.close()
        self.in1.close()
        self.in2.close()


def make_motor():
    try:
        return HardwareMotor()
    except OSError as e:
        print(f"[warning] {e} -> motor on software PWM (100 Hz, speed wavers while the CPU is busy)")
        from gpiozero import Motor
        return Motor(forward=MOTOR_IN1, backward=MOTOR_IN2, enable=MOTOR_PWM, pwm=True)


def make_servo():
    try:
        return HardwareServo()
    except OSError as e:
        print(f"[warning] {e} -> using software PWM (jitters)")
        from gpiozero import AngularServo
        return AngularServo(SERVO_PIN, min_angle=-90, max_angle=90,
                            min_pulse_width=0.0005, max_pulse_width=0.0025)


class Car:
    def __init__(self):
        self.servo = make_servo()
        self.bias = 0.0     # Learned center offset. Used only when lane_follow.AUTO_TRIM is on.
        self.trim = STEER_TRIM      # Servo center. A newly saved one takes effect here at once, also for the centering in close().
        self.motor = make_motor()

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value):
        self._bias = max(-BIAS_MAX, min(BIAS_MAX, value))

    def load_bias(self):
        self.bias = float(load_calib().get("steer_bias", 0.0))

    def save_bias(self):
        data = load_calib()
        data["steer_bias"] = round(self.bias, 2)
        CALIB_FILE.write_text(json.dumps(data))

    def steer(self, angle):
        angle = max(-STEER_LIMIT, min(STEER_LIMIT, (angle + self.bias) * STEER_SCALE))
        self.servo.angle = (-angle if STEER_INVERT else angle) + self.trim

    def center(self):
        self.steer(0)

    def drive(self, speed):
        speed = max(-1.0, min(1.0, speed))
        if speed > 0:
            self.motor.forward(speed)
        elif speed < 0:
            self.motor.backward(-speed)
        else:
            self.motor.stop()

    def stop(self):
        self.motor.stop()

    def close(self):
        """Stop the motor, center the wheels, then release the pins."""
        self.stop()
        try:
            self.center()
            sleep(0.4)      # Time for the servo to reach center before PWM turns off.
        except OSError:
            pass
        self.servo.close()
        self.motor.close()


def stop_on_signals():
    """Make `aicar stop` (SIGTERM) and a dropped ssh session (SIGHUP) end the program like Ctrl+C, so its
    finally block runs: motor off, wheels centered, pins released. Without this the process just dies mid-drive."""
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "center"
    stop_on_signals()
    car = Car()
    try:
        if cmd == "center":
            car.center()
            sleep(1)
        elif cmd == "steer":
            car.steer(float(sys.argv[2]))
            pulse = 1500 + car.servo.angle / 90 * 1000
            print(f"command {float(sys.argv[2]):+.0f} -> servo angle {car.servo.angle:+.1f} deg (pulse {pulse:.0f} us) | "
                  f"trim {STEER_TRIM:+.1f}, scale {STEER_SCALE}, invert {STEER_INVERT}")
            sleep(1)
            car.center()    # Return to center after the check.
            sleep(0.5)
        elif cmd == "drive":
            car.center()
            car.drive(float(sys.argv[2]))
            sleep(float(sys.argv[3]) if len(sys.argv) > 3 else 1)
        elif cmd == "trim":
            data = load_calib()
            old = float(data.get("steer_trim", 0.0))
            data["steer_trim"] = round(old + float(sys.argv[2]), 1)
            CALIB_FILE.write_text(json.dumps(data))
            print(f"trim {old:+.1f} -> {data['steer_trim']:+.1f} deg saved")
            car.trim = data["steer_trim"]
            car.center()    # Move to the new center so it can be checked.
            sleep(1)
        elif cmd == "test":
            for a in (0, -STEER_LIMIT, STEER_LIMIT, 0):
                print(f"steer {a}")
                car.steer(a)
                sleep(0.7)
            print("forward 0.4")
            car.drive(0.4)
            sleep(1.5)
            car.stop()
            sleep(0.5)
            print("backward 0.4")
            car.drive(-0.4)
            sleep(1.5)
        else:
            print(__doc__)
    finally:
        car.close()


if __name__ == "__main__":
    main()
