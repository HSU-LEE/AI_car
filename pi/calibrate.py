"""경기장 자동 캘리브레이션 — 차를 직선 차로 가운데에, 차로 방향으로 놓고 실행

  python calibrate.py           측정 후 vision.json에 저장 (이전 값은 vision.json.bak)
  python calibrate.py --dry     측정만 하고 저장 안 함

순서
  1. 트랙 색       바닥이 밝으면 흰 바닥/검은 선, 어두우면 검은 바닥/흰 선
  2. 초점          오토포커스를 한 번 돌려 트랙 바닥에 맞춘 뒤 그 값으로 고정
  3. 조도          자동 노출/화이트밸런스로 맞춘 뒤 고정. 흰 선이 하얗게 날아가지 않게 줄이고,
                   주행 중 흔들림(번짐)이 없게 노출 시간은 MAX_EXPOSURE_US 이하로 (부족한 밝기는 게인으로)
  4. 차로 폭/소실점 띠마다 양쪽 선 위치를 재서 원근 모델(LANE_WIDTH_PX, HORIZON_Y)을 맞춤
  5. 카메라 중앙    차가 차로 가운데에 있다고 보고 차로 중앙이 화면 아래에서 어디인지 (CAMERA_X)
  6. 확인          저장한 값으로 다시 인식해서 결과 화면을 calib_view.jpg로 저장

자동으로 못 하는 것: 서보 중앙(차가 움직여야 함 → Tune Center), 신호등(켜진 신호등이 필요 → Tune Traffic Light)
"""
import argparse
import json
import shutil
import time

import cv2
import numpy as np

import lane_follow as lf

MAX_EXPOSURE_US = 8000      # 노출 시간 상한(us). 길면 달릴 때 선이 번짐 → 넘으면 게인으로 대신 밝힘
MAX_GAIN = 8.0
CLIP_LEVEL = 250            # 이 밝기 이상은 하얗게 날아간 것으로 봄
CLIP_RATIO = 0.003          # 차선 영역에서 날아간 픽셀이 이 비율을 넘으면 노출을 줄임
FRAMES = 30                 # 차로 측정에 쓰는 프레임 수
VIEW_FILE = lf.VISION_FILE.with_name("calib_view.jpg")


def roi_gray(frame):
    top = int(lf.FRAME_H * lf.ROI_TOP)
    return cv2.cvtColor(frame[top:], cv2.COLOR_BGR2GRAY)


def detect_track_color(frames):
    """바닥 밝기(차선 영역 중간값)로 트랙 색 판단 → "white"(검은 바닥/흰 선) / "black"(흰 바닥/검은 선)"""
    med = float(np.median([np.median(roi_gray(f)) for f in frames]))
    return ("black" if med > 128 else "white"), med


def measure_lanes(frames):
    """띠마다 화면 가운데 기준 왼쪽/오른쪽 선 x의 중간값 → [(y, 왼쪽 x, 오른쪽 x)].
    차로 폭 모델과 상관없이 마스크에서 바로 재므로, 모델 값이 틀려 있어도 측정됨"""
    det = lf.LaneDetector()
    top = int(lf.FRAME_H * lf.ROI_TOP)
    samples = {}
    for f in frames:
        mask = det.line_mask(f[top:])
        h = mask.shape[0] // lf.SLICES
        for i in range(lf.SLICES):
            y1 = mask.shape[0] - (i + 1) * h
            cy = top + y1 + h // 2
            lines, hrun, crosswalk = det.find_lines(mask[y1:y1 + h])
            if hrun is not None or crosswalk:
                continue
            cx = lf.FRAME_W / 2
            lefts, rights = [x for x in lines if x < cx], [x for x in lines if x >= cx]
            if lefts and rights:
                samples.setdefault(cy, []).append((max(lefts), min(rights)))
    bands = []
    for cy, pairs in sorted(samples.items()):
        if len(pairs) >= len(frames) // 2:      # 절반 이상의 프레임에서 양쪽 선이 보인 띠만
            arr = np.array(pairs)
            bands.append((cy, float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))))
    return bands


def fit_geometry(bands):
    """띠별 양쪽 선 → 차로 폭 모델(y=342에서의 폭, 소실점 y), 카메라 중앙, 차의 기울어짐.
    반환: dict 또는 오류 메시지(str)"""
    if len(bands) < 3:
        return f"양쪽 선이 같이 보인 띠가 {len(bands)}개뿐 (3개 이상 필요). 차를 직선 차로 가운데에 놓았는지 확인"
    ys = np.array([b[0] for b in bands], float)
    widths = np.array([b[2] - b[1] for b in bands])
    centers = np.array([(b[1] + b[2]) / 2 for b in bands])
    a, b = np.polyfit(ys, widths, 1)        # 폭 = a*y + b (원근: 아래로 갈수록 넓어짐)
    if a <= 0:
        return "멀수록 차로가 좁아지지 않음 (선을 잘못 잡았을 수 있음). 차를 차로 방향으로 똑바로 놓기"
    horizon = -b / a
    width_342 = a * 342 + b
    resid = float(np.max(np.abs(np.polyval((a, b), ys) - widths)))
    k, c0 = np.polyfit(ys, centers, 1)      # 차로 중앙 x = k*y + c0
    bottom_x = k * lf.FRAME_H + c0
    return {"lane_width_px": round(float(width_342), 1), "horizon_y": round(float(horizon), 1),
            "camera_x": round(float(bottom_x / lf.FRAME_W), 4),
            "tilt": float(k), "fit_error_px": resid, "bands": len(bands)}


def capture(cam, n=1, settle=0.0):
    time.sleep(settle)
    return [cam.capture_array() for _ in range(n)]


def calibrate_focus(cam, controls):
    """오토포커스 한 번 → 렌즈 위치(디옵터). 실패하면 None"""
    cam.set_controls({"AfMode": controls.AfModeEnum.Auto})
    time.sleep(0.2)
    cam.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})
    end = time.monotonic() + 4
    while time.monotonic() < end:
        md = cam.capture_metadata()
        state = md.get("AfState")
        if state == controls.AfStateEnum.Focused:
            return float(md["LensPosition"])
        if state == controls.AfStateEnum.Failed:
            return None
    return None


def calibrate_exposure(cam):
    """자동 노출/화이트밸런스가 자리 잡은 값 → 번짐 없게 노출 상한, 날아가지 않게 줄여서 고정값"""
    cam.set_controls({"AeEnable": True, "AwbEnable": True})
    time.sleep(2.5)
    md = cam.capture_metadata()
    exp, gain = float(md["ExposureTime"]), float(md["AnalogueGain"])
    gains = [float(g) for g in md["ColourGains"]]
    if exp > MAX_EXPOSURE_US:       # 같은 밝기를 짧은 노출 + 높은 게인으로
        gain, exp = min(MAX_GAIN, gain * exp / MAX_EXPOSURE_US), MAX_EXPOSURE_US
    for _ in range(8):              # 흰 선(밝은 쪽)이 하얗게 날아가면 조금씩 줄임
        cam.set_controls({"AeEnable": False, "AwbEnable": False, "ExposureTime": int(exp),
                          "AnalogueGain": gain, "ColourGains": tuple(gains)})
        frame = capture(cam, 1, settle=0.4)[0]
        clipped = float(np.mean(roi_gray(frame) >= CLIP_LEVEL))
        if clipped <= CLIP_RATIO:
            break
        if gain > 1.05:
            gain = max(1.0, gain * 0.8)
        else:
            exp *= 0.8
    return {"ExposureTime": int(exp), "AnalogueGain": round(gain, 3),
            "ColourGains": [round(g, 4) for g in gains]}, clipped


def check_result(frames):
    """저장할 값으로 다시 인식: 위치 오차, 양쪽 선이 다 보인 비율, 마지막 화면"""
    det = lf.LaneDetector()
    offsets, both, dbg = [], 0, None
    for f in frames:
        d, _, dbg = det(f)
        if d is not None:
            offsets.append(d[0])
        both += det.left[0] is not None and det.right[0] is not None
    return (float(np.median(offsets)) if offsets else None), both / max(len(frames), 1), dbg


def annotate(dbg, lines):
    y = 24
    for text in lines:
        cv2.putText(dbg, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(dbg, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="측정만 하고 저장 안 함")
    args = ap.parse_args()
    from libcamera import controls

    print(__doc__.split("\n\n")[0])
    cam = lf.open_camera(locked=False)
    saved, notes = {}, []
    try:
        print("\n[1/6] 트랙 색 ...", flush=True)
        frames = capture(cam, 5, settle=1.5)
        color, med = detect_track_color(frames)
        lf.LINE_COLOR = color
        saved["line_color"] = color
        print(f"      바닥 밝기 {med:.0f} → {'흰 바닥 위 검은 선' if color == 'black' else '검은 바닥 위 흰 선'}")

        print("[2/6] 초점 ...", flush=True)
        lens = calibrate_focus(cam, controls)
        if lens is not None and 0.5 <= lens <= 12:
            saved["lens_position"] = round(lens, 2)
            print(f"      렌즈 {lens:.2f} (약 {100 / lens:.0f}cm에 초점)")
        else:
            notes.append("초점: 오토포커스 실패 → 기존 값 유지")
            print(f"      실패 → 기존 값 {lf.LENS_POSITION} 유지")
        cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                          "LensPosition": saved.get("lens_position", lf.LENS_POSITION)})

        print("[3/6] 조도 (노출 / 게인 / 화이트밸런스) ...", flush=True)
        lock, clipped = calibrate_exposure(cam)
        saved["camera_lock"] = lock
        print(f"      노출 {lock['ExposureTime']}us  게인 {lock['AnalogueGain']}  "
              f"화이트밸런스 R{lock['ColourGains'][0]} B{lock['ColourGains'][1]}  (날아간 픽셀 {clipped:.1%})")

        print("[4/6] 차로 폭 / 소실점 ...", flush=True)
        frames = capture(cam, FRAMES, settle=0.3)
        geo = fit_geometry(measure_lanes(frames))
        geo_ok = not isinstance(geo, str)
        if not geo_ok:
            notes.append("차로: " + geo)
            print("      " + geo)
        else:
            saved["lane_width_px"], saved["horizon_y"] = geo["lane_width_px"], geo["horizon_y"]
            print(f"      띠 {geo['bands']}개 측정  차로 폭 {lf.LANE_WIDTH_PX} → {geo['lane_width_px']}px (y=342)  "
                  f"소실점 {lf.HORIZON_Y} → {geo['horizon_y']}  (오차 최대 {geo['fit_error_px']:.0f}px)")

            print("[5/6] 카메라 중앙 ...", flush=True)
            if abs(geo["tilt"]) > 0.25:
                notes.append("카메라 중앙: 차가 차로와 비스듬해서 건너뜀 (차를 차로 방향으로 똑바로 놓고 다시)")
                print("      차가 차로 방향과 비스듬함 → 건너뜀")
            elif not 0.4 <= geo["camera_x"] <= 0.6:
                notes.append(f"카메라 중앙: {geo['camera_x']}는 너무 치우침 (차를 차로 가운데에 놓고 다시)")
                print(f"      {geo['camera_x']} → 너무 치우쳐서 건너뜀")
            else:
                saved["camera_x"] = geo["camera_x"]
                print(f"      {lf.CAMERA_X} → {geo['camera_x']}")
        saved["white_thr"] = None       # 흰 선 기준은 자동 (고정 노출이라 안정적)

        print("[6/6] 확인 ...", flush=True)
        for key, value in saved.items():        # 이번 실행에서 바로 적용해서 다시 인식
            name = {"line_color": "LINE_COLOR", "lane_width_px": "LANE_WIDTH_PX", "horizon_y": "HORIZON_Y",
                    "camera_x": "CAMERA_X", "white_thr": "WHITE_THR", "lens_position": "LENS_POSITION",
                    "camera_lock": "CAMERA_LOCK"}[key]
            setattr(lf, name, value)
        frames = capture(cam, 20, settle=0.3)
        offset, both, dbg = check_result(frames)
        ok = geo_ok and offset is not None and both >= 0.8 and abs(offset) < 0.1
        off_txt = "-" if offset is None else f"{offset:+.3f}"
        print(f"      양쪽 선 보인 비율 {both:.0%}  위치 오차 {off_txt}  → {'정상' if ok else '확인 필요'}")
        if not ok:
            notes.append("확인: 인식이 불안정 → calib_view.jpg를 보고 차 위치/조명 확인")
        annotate(dbg, [f"CALIBRATION {'OK' if ok else 'CHECK'}  both lines {both:.0%}  offset {off_txt}",
                       f"track {color}  lane {saved.get('lane_width_px', lf.LANE_WIDTH_PX)}px  "
                       f"horizon {saved.get('horizon_y', lf.HORIZON_Y)}  cam_x {saved.get('camera_x', lf.CAMERA_X)}",
                       f"exp {lock['ExposureTime']}us  gain {lock['AnalogueGain']}  "
                       f"lens {saved.get('lens_position', lf.LENS_POSITION)}"])
        cv2.imwrite(str(VIEW_FILE), dbg)
    finally:
        cam.stop()
        cam.close()

    if args.dry:
        print("\n--dry: 저장 안 함")
    else:
        if lf.VISION_FILE.exists():
            shutil.copy(lf.VISION_FILE, lf.VISION_FILE.with_suffix(".json.bak"))
        lf.save_vision(**saved)
        print(f"\n저장: {lf.VISION_FILE}  (이전 값: vision.json.bak)")
    print(json.dumps(saved, ensure_ascii=False))
    for n in notes:
        print("[주의] " + n)
    print("자동으로 못 하는 것: 서보 중앙(Tune Center), 신호등(Tune Traffic Light)")


if __name__ == "__main__":
    main()
