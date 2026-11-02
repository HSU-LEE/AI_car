"""AICAR 서보(조향) + DC 모터(구동) 제어

배선 (TB6612FNG A채널)
  서보 신호  -> GPIO18 (물리 12번)
  PWMA       -> GPIO13 (물리 33번)
  AIN1       -> GPIO23 (물리 16번)
  AIN2       -> GPIO24 (물리 18번)
  STBY, VCC  -> 3.3V
  GND 공통

사용법
  python car.py center          서보 중앙
  python car.py steer 20        1초간 조향 후 중앙 복귀 (-35 ~ 35, 음수=왼쪽)
                                -> 바퀴가 오른쪽으로 꺾여야 정상. 왼쪽이면 STEER_INVERT = True
  python car.py drive 0.4 2     속도(-1 ~ 1, 음수=후진)로 2초 주행
  python car.py test            조향 + 전진/후진 테스트
  python car.py trim -4.2       서보 중앙 보정값을 -4.2도만큼 옮겨 저장 (직진에서 계속 한쪽으로 꺾을 때)
"""
import json
import sys
from pathlib import Path
from time import sleep

SERVO_PIN = 18
MOTOR_PWM, MOTOR_IN1, MOTOR_IN2 = 13, 23, 24

STEER_LIMIT = 26    # 중앙(보정 후)에서 좌우 최대 서보 각도. 어떤 명령이 와도 이 이상은 안 꺾임
STEER_SCALE = 1.0   # 조향 명령 1도당 실제 서보 회전량 (1.0이면 명령값 = 서보 각도)
STEER_INVERT = True     # + 각도에서 바퀴가 왼쪽으로 꺾이면 True (확인 완료: 반전 필요)
CALIB_FILE = Path(__file__).with_name("calib.json")


def load_calib():
    try:
        return json.loads(CALIB_FILE.read_text())
    except (OSError, ValueError):
        return {}


def load_trim():
    """중앙 보정값. tune_center.py로 측정해 calib.json에 저장"""
    return float(load_calib().get("steer_trim", 0.0))


STEER_TRIM = load_trim()
BIAS_MAX = 8    # 자동 중앙 보정의 최대 범위(도)


class HardwareServo:
    """GPIO18 하드웨어 PWM(sysfs)으로 구동하는 서보. 떨림 없이 1° 미만 단위로 제어

    /boot/firmware/config.txt 에 `dtoverlay=pwm,pin=18,func=2` 필요
    """
    PERIOD_NS = 20_000_000   # 50Hz

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
        # export 직후 udev가 권한을 줄 때까지 잠깐 걸림
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
        from gpiozero import Motor   # 맥에서 replay.py가 이 파일을 import할 수 있게 지연 import
        self.servo = make_servo()
        self.bias = 0.0     # 주행 중 자동으로 배운 중앙 보정 (lane_follow.py의 AUTO_TRIM이 켜져 있을 때만 사용)
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
        self.servo.angle = (-angle if STEER_INVERT else angle) + STEER_TRIM

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
        """모터 정지 → 조향 중앙 → 해제. 어떤 프로그램이 끝나도 바퀴가 중앙으로 돌아옴"""
        self.stop()
        try:
            self.center()
            sleep(0.4)      # 서보가 중앙까지 도는 시간 (PWM을 끄면 그 자리에 멈춤)
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
            car.center()    # 확인 후 중앙 복귀
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
            car.servo.angle = data["steer_trim"]    # 새 중앙으로 바로 이동해서 확인
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
