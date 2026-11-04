"""AICAR steering servo and drive motor.

Wiring (TB6612FNG channel A):
  Servo signal -> GPIO18 (physical pin 12)
  PWMA         -> GPIO13 (physical pin 33)
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
import sys
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


class HardwareServo:
    """Servo driven by the GPIO18 hardware PWM (sysfs), steady below 1 degree.

    Requires `dtoverlay=pwm,pin=18,func=2` in /boot/firmware/config.txt.
    """
    PERIOD_NS = 20_000_000   # 50 Hz.

    def __init__(self, min_angle=-90, max_angle=90,
                 min_pulse_width=0.0005, max_pulse_width=0.0025):
        chips = sorted(Path("/sys/class/pwm").glob("pwmchip*"))
        if not chips:
            raise OSError("하드웨어 PWM 없음 (config.txt 오버레이 확인)")
        self.chip = chips[0]
        self.pwm = self.chip / "pwm0"
        if not self.pwm.exists():
            (self.chip / "export").write_text("0")
        self._write("period", self.PERIOD_NS)
        self.min_angle, self.max_angle = min_angle, max_angle
        self.min_ns, self.max_ns = min_pulse_width * 1e9, max_pulse_width * 1e9
        self._angle = None

    def _write(self, name, value):
        # Right after export, udev may not have granted write access yet.
        for _ in range(50):
            try:
                (self.pwm / name).write_text(str(value))
                return
            except PermissionError:
                sleep(0.02)
        (self.pwm / name).write_text(str(value))

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

    def close(self):
        try:
            self._write("enable", 0)
            (self.chip / "unexport").write_text("0")
        except OSError:
            pass


def make_servo():
    try:
        return HardwareServo()
    except OSError as e:
        print(f"[경고] {e} -> 소프트웨어 PWM 사용 (떨림 있음)")
        from gpiozero import AngularServo
        return AngularServo(SERVO_PIN, min_angle=-90, max_angle=90,
                            min_pulse_width=0.0005, max_pulse_width=0.0025)


class Car:
    def __init__(self):
        from gpiozero import Motor   # Imported late so replay.py can load this file on a Mac.
        self.servo = make_servo()
        self.bias = 0.0     # Learned center offset. Used only when lane_follow.AUTO_TRIM is on.
        self.trim = STEER_TRIM      # Servo center. A newly saved one takes effect here at once, also for the centering in close().
        self.motor = Motor(forward=MOTOR_IN1, backward=MOTOR_IN2,
                           enable=MOTOR_PWM, pwm=True)

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


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "center"
    car = Car()
    try:
        if cmd == "center":
            car.center()
            sleep(1)
        elif cmd == "steer":
            car.steer(float(sys.argv[2]))
            pulse = 1500 + car.servo.angle / 90 * 1000
            print(f"명령 {float(sys.argv[2]):+.0f} → 서보 각도 {car.servo.angle:+.1f}도 (펄스 {pulse:.0f}us) | "
                  f"중앙보정 {STEER_TRIM:+.1f}, 민감도 {STEER_SCALE}, 반전 {STEER_INVERT}")
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
            print(f"중앙 보정 {old:+.1f} → {data['steer_trim']:+.1f}도 저장")
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
