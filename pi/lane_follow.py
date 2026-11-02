"""OpenCV 차선 인식 + 가변 조향 자율주행 + 교차로별 코스 지정

화면 아래쪽(ROI)을 가로 띠 여러 개로 나눠 각 띠에서 흰 선을 찾고, 왼쪽/오른쪽 선을
프레임마다 추적해 차로 중심을 구한다. 중심점들로 직선을 맞춰
  - 위치 오차(차가 차로 중앙에서 벗어난 정도)
  - 방향 오차(차로가 휘어지는 정도)
를 구하고 둘을 합쳐 조향각을 연속으로 정한다. 조향이 클수록 감속한다.
앞을 가로지르는 선(교차로)을 만나면 course.py의 COURSE에 적힌 다음 동작을 실행한다.

사용법 (Pi에서)
  python lane_follow.py             조향만 (모터 꺼짐) - 차를 손으로 밀면서 인식 확인
  python lane_follow.py --drive     실제 주행
  python lane_follow.py --drive --speed 0.3
  python lane_follow.py --drive --start 3     COURSE 3번 동작부터 시작
  python lane_follow.py --record    runs/ 폴더에 영상 저장 (맥에서 replay.py로 분석)
  python lane_follow.py --drive --set KP=1.8 KH=1.0   이번 실행만 설정값 바꾸기

  흰 선 기준값은 자동 캘리브레이션된다. 직접 정하려면 tune_white.py (vision.json에 저장)

  맥 브라우저에서 http://<파이 IP>:8000 을 열면 인식 화면이 보인다. (핫스팟이면 10.42.0.1)
  Ctrl+C(SIGINT)나 SIGTERM으로 정지. ssh 연결이 끊겨도(SIGHUP) 계속 주행한다.
  맥의 aicar go/view는 파이에서 백그라운드로 실행하고 출력은 drive.log에 남긴다.
"""
import argparse
import ast
import json
import queue
import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from car import STEER_LIMIT
from course import COURSE

# ---------------- 설정 ----------------
FRAME_W, FRAME_H = 640, 360
# 카메라: 센서 전체(약 66도)를 씀. 예전에는 FrameRate 60 때문에 가운데 2/3만 잘라 쓰는 1536x864 모드(약 47도)였음.
#   두 모드는 화면 중심이 같아서, 예전 화면의 점 (x, y)는 지금 화면에서 (320 + (x-320)/1.5, 180 + (y-180)/1.5).
#   아래 픽셀/화면 위치 설정은 이 비율로 환산한 값 (옛 값은 각 줄 주석). 트랙 위에서 Detect Only로 확인하며 다듬기
CAMERA_SENSOR = (2304, 1296)    # 센서 전체를 2x2로 묶어 읽는 모드 (최대 56fps). 좁게 보려면 (1536, 864) + 60fps
CAMERA_FPS = 56
HFLIP = VFLIP = False   # 화면이 뒤집혀 보이면 True
CAMERA_X = 0.5          # 차 정중앙이 화면 가로의 어디인지. vision.json의 camera_x가 있으면 그 값을 씀

LINE_MODE = "lane"      # "lane": 양쪽 선 사이로 / "line": 선 하나를 따라감
ROI_TOP = 0.4           # 화면 높이의 이 지점부터 아래만 사용 (작을수록 멀리 봄)
SLICES = 6              # ROI를 나눌 가로 띠 개수

# 흰 선 인식. 기본은 매 프레임 자동 캘리브레이션. tune_white.py로 직접 정하면 vision.json에 저장됨
LINE_COLOR = "white"    # 트랙 종류. "white": 검은 바닥 위 흰 선 / "black": 흰 바닥 위 검은 선
                        #   "black"이면 화면 밝기를 뒤집어서 검은 선을 흰 선처럼 보고 나머지는 똑같이 처리.
                        #   메뉴의 "트랙 색" (aicar track)로 바꾸면 vision.json에 저장됨
WHITE_THR = None        # 흰 선으로 볼 밝기 차(바닥 대비). None이면 자동
WHITE_MIN = 35          # 자동일 때 이보다 낮게는 내려가지 않음 (빈 바닥에서 노이즈를 선으로 잡지 않게)
TOPHAT_K = 23           # 이보다 굵은 밝은 덩어리(반사광, 매트 밖 바닥)는 선이 아님 (반해상도 px) [좁은 화각 35]
USE_MAT_MASK = True     # 검은 매트 바깥(바닥, 나무 테두리)은 무시
LINE_W_MIN, LINE_W_MAX = 3, 87   # 띠 안에서 선 하나의 가로 폭(px) [좁은 화각 4, 130]
MIN_PIXELS = 40         # 선 하나로 인정할 최소 픽셀 수 [좁은 화각 60]
DOUBLE_GAP = 33         # 이 간격(px)보다 가까운 두 선은 이중선 하나로 합침 [좁은 화각 50]
CROSSWALK_LINES = 4     # 한 띠에 선이 이만큼 이상이면 횡단보도로 보고 그 띠는 무시
LANE_WIDTH_PX = 482     # 화면 아래쪽(y=342)에서의 차로 폭(px). 양쪽 선이 보이면 자동 보정됨
                        #   [좁은 화각 녹화 실측 590을 환산. 넓은 화각의 y=342는 예전 화면 밖(더 가까운 바닥)]
HORIZON_Y = 49          # 차로 양쪽 선이 만나는 높이(y). 위로 갈수록 차로 폭이 좁아지는 비율을 정함 [좁은 화각 -16]
REF_LANE_PX = 590       # 위치/방향 오차의 단위 기준: 좁은 화각에서 KP, KH 등을 맞출 때의 차로 폭.
                        #   오차에 REF_LANE_PX / LANE_WIDTH_PX를 곱해 예전과 같은 크기로 만들어서
                        #   KP, KH, OFFSET_PRIORITY, 새 차선 판정 등 조향 쪽 값은 그대로 씀
LEARN_WIDTH = False     # True면 주행 중 차로 폭을 학습. False면 위 실측값으로 고정 (매번 같은 동작)
OUTLIER_PX = 30         # 띠별 중심점이 맞춘 직선에서 이만큼 벗어나면 오인식으로 보고 제외 [좁은 화각 45]

KP = 1.6                # 위치 오차 → 조향. 차로 중앙으로 돌아오는 힘 (클수록 과감)
KH = 0.9                # 방향 오차 → 조향. 커브를 미리 꺾는 힘 (커브 이탈하면 키움)
OFFSET_PRIORITY = 0.6   # 위치 오차가 이만큼 커지면 방향 오차는 무시하고 차로 복귀만 함 (S커브에서 서로 상쇄 방지)
KD = 0.03               # 흔들림 억제 (지그재그가 빠르면 조금 키움, 너무 크면 오히려 떨림)
SMOOTH = 0.15           # 조향 저역통과 (0: 없음, 0.8: 아주 부드럽게. 크면 반응이 늦어 커브 이탈)
AUTO_TRIM = False       # True면 직선에서 계속 한쪽으로 조향할 때 중앙 보정값을 자동으로 고쳐 저장 (주행마다 학습)
AUTO_TRIM_TAU = 15.0    # 자동 중앙 보정 속도(초). 양쪽 선이 다 보이는 직선에서만 학습
AUTO_STEER_MAX = 26     # 자율주행 중 최대 조향각(도). car.py STEER_LIMIT(26)을 넘지 않음
STEER_RATE = 150        # 조향각이 1초에 바뀔 수 있는 최대량(도/초). 작을수록 확확 안 틈

BASE_SPEED = 0.22       # 직진 속도
MIN_SPEED = 0.18        # 최대 조향일 때 속도 (모터가 멈추지 않는 최소값 이상)
HARD_STEER = 21         # 조향각이 이 값(도)을 넘으면 바퀴 저항이 커져 느려지므로 모터값을 조금 올림
HARD_STEER_BOOST = 0.1 # 그때 더하는 모터값 (0이면 안 씀)
KICK_SPEED = 0.6        # 멈춰 있다가 출발할 때 잠깐 주는 속도 (낮은 속도에서는 모터가 안 돌기 때문)
KICK_TIME = 0.25        # 출발 킥 시간(초)
LOST_LIMIT = 15         # (옛 설정, 안 씀) 프레임 수 대신 LOST_HOLD(초)로 판단
LOST_HOLD = 1.0         # 차선을 놓쳐도 이 시간(초) 동안은 마지막 조향을 유지하며 천천히 진행. 그 뒤 정지
STEP_MAX_SEC = 20       # until 동작에 max_sec를 안 쓰면 이 시간이 지나면 경고하고 다음 동작으로 (영원히 안 끝나는 것 방지)
FAR_TOP = 0.3           # slow= 원거리 감지: 화면 높이의 이 지점 ~ ROI_TOP 사이에서도 가로선을 찾음 [좁은 화각 0.2]
FAR_HLINE = 1.2         # 원거리 가로선: 한 줄에서 이어진 흰 픽셀이 그 높이의 차로 폭 × 이 값 이상이어야 인정
                        # 감속에만 쓰고 교차로 판정(until)에는 안 씀 → 곡선을 가로선으로 잘못 봐도 잠깐 느려질 뿐
APPROACH_HOLD = 0.5     # slow=: 선이 잠깐 안 보여도 이 시간(초) 동안은 감속 상태 유지 (오감지 한두 프레임엔 반응 안 하게)
KICK_AFTER_STOP = 0.3   # 이 시간(초) 이상 서 있다가 출발할 때만 출발 킥 (잠깐 멈칫한 것에는 킥 안 함)
LANE_PX_PER_CM = REF_LANE_PX / 20.0     # 1cm가 위치 오차 단위로 몇 px인지 (차로 폭 20cm). shift= 계산용
                                        #   위치 오차가 REF_LANE_PX 기준 단위라서 같은 기준을 씀
# style= 조향 성격: (KP 배수, KH 배수, SMOOTH)
STYLES = {"soft": (0.8, 0.7, 0.35), "normal": (1.0, 1.0, None), "sharp": (1.3, 1.6, 0.05)}

HLINE_RATIO = 0.27      # 띠 폭의 이 비율 이상 이어진 선 = 가로선(교차로) [좁은 화각 0.4]
JUNCTION_Y = 0.5        # 가로선이 화면 높이의 이 지점보다 아래로 내려오면 교차로 동작 시작 (작을수록 일찍)
EDGE_STOP_Y = 0.63      # 차 앞쪽 매트가 화면 높이의 이 지점에서 끝나면(매트 끝) 정지 [좁은 화각 0.7]

JUNCTION_BY_BRANCH = False  # True면 선이 3개 이상 보일 때도 교차로로 봄 (갈림길)
JUNCTION_FRAMES = 3     # 연속 몇 프레임 감지되어야 인정 (오감지 방지). gap/green도 같은 값
GAP_BANDS = 2           # 아래에서 이어지던 선 위로 이 개수만큼 띠에 선이 하나도 없으면 "차선이 끊김"(gap)
CROSSWALK_HOLD_MAX = 1.5    # 차 바로 앞 횡단보도 위에서 조향을 유지하는 최대 시간(초). 그 뒤엔 다시 차선 따라감
TURN_CLEAR_FRAMES = 8   # 회전: 새 차선이 이 프레임 수만큼 잡히면 회전 각도 → 차선 추종 조향으로 완전히 넘어감
STRAIGHT_CORR = 6       # straight(): 차선이 보이면 이 범위(±도) 안에서 보정
CURVE_ASSIST = 0.5      # curve(): 지정 각도와 차선 추종 조향을 섞는 비율 (0: 각도 고정, 1: 차선만)

# 신호등. 현장 조명에 맞춰 tune_light.py로 조정하면 vision.json에 저장됨
LIGHT_Y = (0.0, 0.5)    # 램프를 찾을 높이 범위 (화면 높이 비율). 차선 영역(ROI_TOP)보다 위쪽
LIGHT_SIDE = 0.4        # 화면 왼쪽/오른쪽 이 비율 폭만 봄 (신호등은 레인 옆에 있음) [좁은 화각 0.35]
LIGHT_S_MIN = 110       # 색으로 볼 최소 채도 (0~255)
LIGHT_V_MIN = 170       # 색으로 볼 최소 밝기 (0~255). LED는 스스로 빛나서 주변 물체보다 밝음
LIGHT_AREA_MIN = 11     # 램프 덩어리 최소 크기(px). 작은 반사광 무시 [좁은 화각 25]
LIGHT_AREA_MAX = 1778   # 최대 크기(px). 큰 빨간/초록 물체 무시 [좁은 화각 4000]
LIGHT_G_HUE = (35, 100) # wait_green: 빨간불을 본 그 자리에서만 쓰는 넓은 초록 범위 (LED 초록은 청록빛 H 80~95로도 찍힘)
LIGHT_LOCAL_S_MIN = 60  # 그 자리에서 색으로 볼 최소 채도 (화면 전체용 LIGHT_S_MIN보다 느슨하게: 위치를 알아서 오감지 걱정이 적음)
LIGHT_LOCAL_V_MIN = 100 # 그 자리에서 색으로 볼 최소 밝기 (초록 LED는 빨강보다 어둡게 찍히는 편)
LIGHT_NEAR = 4          # wait_green: 초록불은 빨간불이 있던 자리에서 (빨간 램프 크기 × 이 값) 안에 있어야 인정
                        #   같은 신호등의 램프 간격은 허용하고, 다른 곳의 초록 물체는 무시

LENS_POSITION = 3.0     # 수동 초점(디옵터 = 1/거리m). 3.0 ≈ 33cm. None이면 오토포커스. calibrate.py가 측정해 저장
CAMERA_LOCK = None      # 고정 노출/게인/화이트밸런스 {"ExposureTime": us, "AnalogueGain": x, "ColourGains": [r, b]}.
                        #   None이면 카메라 자동. calibrate.py가 경기장 조명에서 측정해 vision.json에 저장
VISION_FILE = Path(__file__).with_name("vision.json")
# --------------------------------------


def load_vision():
    """vision.json에 저장한 값을 불러옴. 없으면 기본값.
    트랙 색(line_color), 흰 선 기준값(tune_white.py), 신호등 기준값(tune_light.py),
    calibrate.py가 잰 카메라 중앙(camera_x), 차로 폭/소실점, 초점, 노출/화이트밸런스"""
    global LINE_COLOR, WHITE_THR, CAMERA_X, LIGHT_S_MIN, LIGHT_V_MIN, LIGHT_AREA_MIN
    global LANE_WIDTH_PX, HORIZON_Y, LENS_POSITION, CAMERA_LOCK
    try:
        data = json.loads(VISION_FILE.read_text())
    except (OSError, ValueError):
        return
    LINE_COLOR = data.get("line_color", LINE_COLOR)
    WHITE_THR = data.get("white_thr", WHITE_THR)
    CAMERA_X = data.get("camera_x", CAMERA_X)
    LIGHT_S_MIN = data.get("light_s_min", LIGHT_S_MIN)
    LIGHT_V_MIN = data.get("light_v_min", LIGHT_V_MIN)
    LIGHT_AREA_MIN = data.get("light_area_min", LIGHT_AREA_MIN)
    LANE_WIDTH_PX = data.get("lane_width_px", LANE_WIDTH_PX)
    HORIZON_Y = data.get("horizon_y", HORIZON_Y)
    LENS_POSITION = data.get("lens_position", LENS_POSITION)
    CAMERA_LOCK = data.get("camera_lock", CAMERA_LOCK)


def save_vision(**values):
    """vision.json에서 주어진 항목만 바꿔 저장 (다른 항목은 유지)"""
    try:
        data = json.loads(VISION_FILE.read_text())
    except (OSError, ValueError):
        data = {}
    data.update(values)
    VISION_FILE.write_text(json.dumps(data))


load_vision()


_ELLIPSE_RECTS = {}


def ellipse_rects(size):
    """타원 커널을 같은 중심의 직사각형 여러 개로 쪼갬 (합집합 = 원래 타원).
    타원은 볼록하고 대칭이라, 폭 w인 행을 포함하는 '폭 w × 그 폭 이상인 행 전체 높이' 직사각형이
    모두 타원 안에 들어가고, 합치면 정확히 타원이 됨"""
    if size not in _ELLIPSE_RECTS:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        c = size // 2
        reach = {}      # 반폭 → 그 반폭인 행이 중심에서 떨어진 최대 거리
        for r in range(size):
            cols = np.flatnonzero(k[r])
            assert cols[0] + cols[-1] == 2 * c, "타원 커널이 좌우 대칭이 아님"
            hw = c - cols[0]
            reach[hw] = max(reach.get(hw, 0), abs(r - c))
        _ELLIPSE_RECTS[size] = [np.ones((2 * hh + 1, 2 * hw + 1), np.uint8) for hw, hh in reach.items()]
    return _ELLIPSE_RECTS[size]


def tophat_ellipse(img, size):
    """cv2.morphologyEx(img, MORPH_TOPHAT, 타원 커널)과 같은 결과를 더 빠르게.
    합집합 커널로 침식 = 각 직사각형 침식의 최솟값, 팽창 = 최댓값 (직사각형은 OpenCV가 분리해서 빠르게 처리)"""
    rects = ellipse_rects(size)
    eroded = cv2.erode(img, rects[0])
    for r in rects[1:]:
        eroded = cv2.min(eroded, cv2.erode(img, r))
    opened = cv2.dilate(eroded, rects[0])
    for r in rects[1:]:
        opened = cv2.max(opened, cv2.dilate(eroded, r))
    return cv2.subtract(img, opened)


def percentile_u8(img, q):
    """np.percentile(img, q)와 같은 값(선형 보간)을 정렬 없이 히스토그램으로 계산 (uint8 영상용)"""
    n = img.size
    cum = np.cumsum(cv2.calcHist([img], [0], None, [256], [0, 256]).ravel().astype(np.int64))
    pos = (n - 1) * (q / 100)                       # numpy 기본(linear) 방식의 위치 계산을 그대로
    lo = int(np.floor(pos))
    t = pos - lo
    a = float(np.searchsorted(cum, lo, side="right"))                 # lo번째로 작은 값
    b = float(np.searchsorted(cum, min(lo + 1, n - 1), side="right"))
    return b - (b - a) * (1 - t) if t >= 0.5 else a + (b - a) * t   # numpy와 같은 보간식


# ---------------- 인식 화면 그리기 (브라우저 Live View) ----------------
# 색 (BGR). 브라우저 페이지 아래 범례와 같은 색
C_LANE = (255, 210, 60)     # 흰 선으로 인식된 픽셀 (반투명)
C_LEFT, C_RIGHT = (255, 140, 40), (60, 60, 255)    # 왼쪽 / 오른쪽 선
C_PATH = (80, 230, 80)      # 차가 따라갈 차로 중앙
C_JUNC = (255, 0, 255)      # 교차로(가로선)
C_XWALK = (240, 240, 240)   # 횡단보도
C_GAP = (255, 128, 0)       # 차선 끊김
C_AT = (0, 230, 255)        # at= 지점 (이 동작이 끝나는 선)
C_DIM = (110, 110, 110)


def chip(img, text, x, y, color, scale=0.42, dark=None):
    """색 배경 위에 글자를 쓴 작은 이름표. (x, y) = 왼쪽 위. 반환: 오른쪽 끝 x"""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(min(max(x, 0), img.shape[1] - tw - 8))
    y = int(min(max(y, 0), img.shape[0] - th - 8))
    cv2.rectangle(img, (x, y), (x + tw + 8, y + th + 8), color, -1)
    if dark is None:
        dark = sum(color) > 380     # 밝은 배경엔 검은 글자
    cv2.putText(img, text, (x + 4, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (20, 20, 20) if dark else (255, 255, 255), 1, cv2.LINE_AA)
    return x + tw + 8


def dashed_hline(img, y, color, x0=0, x1=None, dash=10, gap=6, thick=1):
    x1 = img.shape[1] if x1 is None else x1
    for x in range(x0, x1, dash + gap):
        cv2.line(img, (x, y), (min(x + dash, x1), y), color, thick, cv2.LINE_AA)


class LaneDetector:
    def __init__(self):
        self.left = [None] * SLICES     # 직전 프레임의 띠별 왼쪽/오른쪽 선 x
        self.right = [None] * SLICES
        self.center = [None] * SLICES
        self.width = [None] * SLICES    # 띠별 차로 폭 학습값
        self.auto_thr = None            # 자동 캘리브레이션된 흰 선 기준값
        self.thr = 0                    # 이번 프레임에 실제로 쓴 기준값
        self.misses = 0                 # 연속으로 차선을 못 찾은 프레임 수
        self.edge = False               # 차 바로 앞에서 매트가 끝남 (주행 정지)
        self.edge_hits = 0
        self.mat_top = 0                # 차 정면에서 매트가 시작되는 가장 위쪽 y (화면 좌표)
        self.crosswalk = False          # 이번 프레임에 차 앞(JUNCTION_Y 아래)에 횡단보도가 보임
        self.crosswalk_any = False      # 화면 어디든 횡단보도가 보임 (지나가는 중)
        self.junction_y = self.crosswalk_y = None
        self.gap_y = None               # 앞에서 차선이 끊기는 곳: 선이 마지막으로 보인 띠의 y. 없으면 None
        self.far_y = None               # ROI 위쪽(원거리)에서 본 가로선 y. slow= 감속용 (Driver가 필요할 때만 채움)
        top = int(FRAME_H * ROI_TOP)
        h = (FRAME_H - top) // SLICES
        self.band_y = [FRAME_H - (i + 0.5) * h for i in range(SLICES)]

    def model_width(self, i):
        """원근 모델: 차로 폭은 소실점(HORIZON_Y)에서 0, 아래로 갈수록 선형으로 넓어짐"""
        return LANE_WIDTH_PX * (self.band_y[i] - HORIZON_Y) / (342 - HORIZON_Y)

    def lane_width(self, i):
        return self.width[i] or self.model_width(i)

    def learn_width(self, i, w):
        if not LEARN_WIDTH:
            return
        # 학습값이 모델에서 크게 벗어나지 않게 제한 (잘못 학습하면 한쪽으로 쏠리는 원인이 됨)
        m = self.model_width(i)
        w = float(np.clip(w, 0.8 * m, 1.25 * m))
        self.width[i] = w if self.width[i] is None else 0.9 * self.width[i] + 0.1 * w

    @staticmethod
    def mat_mask(gray):
        """검은 매트 영역(아래쪽에서 이어진 가장 큰 어두운 영역의 볼록 껍질). 바닥/나무 테두리 제외"""
        small = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4), interpolation=cv2.INTER_AREA)
        thr, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = (small < thr).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))  # 흰 선을 메워 매트를 한 덩어리로 [좁은 화각 9x9]
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((1, 21), np.uint8))  # 횡단보도 줄무늬 사이도 이어줌 [좁은 화각 1x31]
        n, labels = cv2.connectedComponents(dark)
        if n <= 1:
            return np.full_like(gray, 255)
        bottom = labels[-3:].ravel()
        counts = np.bincount(bottom[bottom > 0], minlength=n)
        if counts.max() == 0:
            return np.full_like(gray, 255)
        pts = cv2.findNonZero((labels == counts.argmax()).astype(np.uint8))
        hull = np.zeros_like(small)
        cv2.fillConvexPoly(hull, cv2.convexHull(pts), 255)
        hull = cv2.dilate(hull, np.ones((3, 3), np.uint8))
        return cv2.resize(hull, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    def line_mask(self, roi):
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if LINE_COLOR == "black":
            # 흰 바닥 위 검은 선: 밝기를 뒤집으면 검은 바닥 위 흰 선과 같아짐 (매트 = 흰 바닥 영역)
            gray = cv2.bitwise_not(gray)
        # 탑햇: 주변 바닥보다 밝으면서 가는 것만 남김 → 굵은 반사광/밝은 바닥은 사라짐
        half = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
        top = cv2.resize(tophat_ellipse(half, TOPHAT_K), (gray.shape[1], gray.shape[0]))
        if USE_MAT_MASK:
            mat = self.mat_mask(gray)
            top = cv2.bitwise_and(top, mat)
            # 차 정면(가운데 1/6 폭 = 좁은 화각의 1/4 폭)에서 매트가 어디까지 이어지는지 → 너무 가까이서 끝나면 매트 끝
            cx0 = int(FRAME_W * CAMERA_X)
            front = mat[:, max(cx0 - FRAME_W // 12, 0):cx0 + FRAME_W // 12]
            rows = np.flatnonzero(front.mean(axis=1) > 128)
            mat_top = int(FRAME_H * ROI_TOP) + (rows[0] if len(rows) else front.shape[0])
            self.mat_top = mat_top
            self.edge_hits = self.edge_hits + 1 if mat_top > FRAME_H * EDGE_STOP_Y else 0
            self.edge = self.edge_hits >= 3     # 3프레임 연속일 때만 (한 프레임 오인식으로 서지 않게)

        if WHITE_THR is not None:
            self.thr = WHITE_THR
        else:
            # 자동: 가장 밝은 선 픽셀(상위 1%)의 절반을 기준으로, 프레임 간에는 천천히 따라감
            peak = percentile_u8(top, 99)
            target = max(WHITE_MIN, 0.5 * peak)
            self.auto_thr = target if self.auto_thr is None else 0.9 * self.auto_thr + 0.1 * target
            self.thr = max(WHITE_MIN, self.auto_thr)
        mask = (top >= self.thr).astype(np.uint8) * 255
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def far_hline(self, frame):
        """ROI 위쪽(FAR_TOP ~ ROI_TOP)에서 가로선이 보이는 가장 가까운(아래) y. 없으면 None.
        slow= 감속을 일찍 시작하려고 씀. 기준 밝기는 이번 프레임 ROI에서 구한 self.thr를 그대로 씀"""
        top, bot = int(FRAME_H * FAR_TOP), int(FRAME_H * ROI_TOP)
        if bot - top < 4:
            return None
        gray = cv2.GaussianBlur(cv2.cvtColor(frame[top:bot], cv2.COLOR_BGR2GRAY), (5, 5), 0)
        half = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
        mask = tophat_ellipse(half, TOPHAT_K) >= self.thr
        hits = 0
        for r in range(mask.shape[0] - 1, -1, -1):      # 아래(가까운 쪽)부터
            row = np.r_[0, mask[r].astype(np.int8), 0]
            edges = np.flatnonzero(np.diff(row))
            longest = (edges[1::2] - edges[::2]).max() * 2 if len(edges) else 0
            y = top + 2 * r
            need = FAR_HLINE * LANE_WIDTH_PX * (y - HORIZON_Y) / (342 - HORIZON_Y)
            hits = hits + 1 if longest >= need else 0
            if hits >= 2:                               # 두 줄 이상 이어져야 (잡음 제외)
                return y + 2
        return None

    @staticmethod
    def find_lines(band):
        """띠 하나 → (세로 선들의 x 목록, 가로선 후보 (시작, 끝) 또는 None, 횡단보도 여부)"""
        cols = band.sum(axis=0).astype(np.float64)
        on = (cols > band.shape[0] * 0.2 * 255).astype(np.int8)
        edges = np.flatnonzero(np.diff(np.r_[0, on, 0]))
        runs, hline = [], None
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= FRAME_W * HLINE_RATIO:
                hline = (a, b)
            elif LINE_W_MIN <= b - a <= LINE_W_MAX and cols[a:b].sum() >= MIN_PIXELS * 255:
                runs.append([a, b])
        if len(runs) >= CROSSWALK_LINES:
            # 횡단보도: 폭이 비슷한 줄무늬가 촘촘히 반복됨 (차선들은 서로 멀리 떨어져 있음)
            widths = np.array([b - a for a, b in runs], dtype=float)
            gaps = np.array([runs[k + 1][0] - runs[k][1] for k in range(len(runs) - 1)], dtype=float)
            med = np.median(widths)
            if np.all((widths > 0.5 * med) & (widths < 2 * med)) and np.median(gaps) < 2.5 * med:
                return [], hline, True
        merged = []     # 이중선은 하나로
        for a, b in runs:
            if merged and a - merged[-1][1] < DOUBLE_GAP:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        lines = [float(np.average(np.arange(a, b), weights=cols[a:b] + 1e-6)) for a, b in merged]
        return lines, hline, False

    @staticmethod
    def nearest(x, prev):
        return abs(x - prev) if prev is not None else 1e9

    def assign(self, i, lines, side, ref):
        """띠 i의 선들 중 현재 차로의 왼쪽/오른쪽 선을 고름 → (왼쪽, 오른쪽, 판단 보류한 선)"""
        if not lines:
            return None, None, None
        if side == "left":
            return min(lines), None, None
        if side == "right":
            return None, max(lines), None
        lefts = [x for x in lines if x < ref]
        rights = [x for x in lines if x >= ref]
        l = max(lefts) if lefts else None
        r = min(rights) if rights else None
        w = self.lane_width(i)
        pl, pr = self.left[i], self.right[i]
        if l is not None and r is not None:
            if not 0.6 * w <= r - l <= 1.5 * w:
                # 폭이 말이 안 되면 직전 위치에 더 가까운 선 하나만 믿음
                if self.nearest(l, pl) <= self.nearest(r, pr):
                    r = None
                else:
                    l = None
        if l is None or r is None:
            # 선이 하나만 보이면: 직전 왼/오른선과 확실히 이어지면 그쪽으로,
            # 아니면 판단을 보류했다가 프레임 전체를 보고 정함 (__call__ 참고)
            x = l if l is not None else r
            dl, dr = self.nearest(x, pl), self.nearest(x, pr)
            if min(dl, dr) >= w * 0.25:
                return None, None, x
            l, r = (x, None) if dl <= dr else (None, x)
        return l, r, None

    def find_gap(self, bands, both_near):
        """앞에서 차선이 끊기는지 → 선이 마지막으로 보인 띠의 y (없으면 None).
        아래 띠부터 선이 이어지다가 그 위로 GAP_BANDS개 띠에 선이 하나도 없으면 끊김으로 봄.
        오감지 방지: 맨 아래 띠에 양쪽 선이 다 보일 때만 (급커브에서 위쪽 선이 화면 옆으로 빠지는 것 제외),
        매트 끝이 아닐 때만, 빈 띠가 매트 안일 때만, 가로선/횡단보도 띠가 있으면 아님"""
        if not both_near or self.edge or any(special for _, _, special in bands):
            return None
        k = 0
        while k + 1 < len(bands) and bands[k + 1][1]:    # 아래에서부터 선이 이어진 마지막 띠
            k += 1
        empty = bands[k + 1:k + 1 + GAP_BANDS]
        if len(empty) < GAP_BANDS or any(has for _, has, _ in empty):
            return None
        if USE_MAT_MASK and any(cy < self.mat_top for cy, _, _ in empty):
            return None     # 빈 곳이 매트 밖(바닥)이면 끊긴 게 아니라 매트가 끝난 것
        return int(bands[k][0])

    def __call__(self, frame, side="center"):
        """반환: ((위치오차, 방향오차) 또는 None, 교차로 감지 여부, 디버그 이미지)"""
        top = int(FRAME_H * ROI_TOP)
        mask = self.line_mask(frame[top:])
        h = mask.shape[0] // SLICES

        dbg = frame.copy()
        roi, on = dbg[top:], mask > 0       # 흰 선 픽셀은 반투명으로 칠해서 원래 화면도 보이게
        roi[on] = (0.45 * roi[on] + 0.55 * np.array(C_LANE)).astype(np.uint8)
        cx0 = int(FRAME_W * CAMERA_X)
        dashed_hline(dbg, top, C_DIM)       # 이 선 아래만 차선 인식에 씀
        jy = int(FRAME_H * JUNCTION_Y)
        marks = {}                          # 교차로/횡단보도 이름표 (종류별 한 번, 마지막에 맨 위에 그림)

        junction = False
        self.crosswalk = self.crosswalk_any = False
        self.junction_y = self.crosswalk_y = None   # 가로선/횡단보도가 보인 가장 가까운(아래) 띠의 y. course의 at= 판단용
        ys, ls, rs, pending = [], [], [], []
        bands = []      # 띠별 (y, 선이 있는지, 가로선/횡단보도인지) → 차선 끊김(gap) 판단용
        ref = self.center[0] if self.center[0] is not None else cx0
        for i in range(SLICES):     # 0번이 가장 가까운 띠. 아래에서 위로 올라가며 추적
            y1 = mask.shape[0] - (i + 1) * h
            cy = top + y1 + h // 2
            lines, hrun, crosswalk = self.find_lines(mask[y1:y1 + h])
            hline = hrun is not None
            bands.append((cy, bool(lines), hline or crosswalk))
            if hline and cy >= jy or (JUNCTION_BY_BRANCH and i < 2 and len(lines) >= 3):
                junction = True
            if hline:
                self.junction_y = max(self.junction_y or 0, cy)
            if crosswalk:
                self.crosswalk_any = True
                self.crosswalk_y = max(self.crosswalk_y or 0, cy)
                if cy >= jy:
                    self.crosswalk = True
            if hline or crosswalk:
                color, name = (C_JUNC, "JUNCTION") if hline else (C_XWALK, "CROSSWALK")
                cv2.rectangle(dbg, (0, top + y1 + 2), (5, top + y1 + h - 2), color, -1)   # 양쪽 가장자리 표시
                cv2.rectangle(dbg, (FRAME_W - 6, top + y1 + 2), (FRAME_W - 1, top + y1 + h - 2), color, -1)
                marks.setdefault(name, (color, cy))

            if LINE_MODE == "line":
                l = min(lines, key=lambda x: abs(x - ref)) if lines else None
                r = None
            else:
                l, r, undecided = self.assign(i, lines, side, ref)
                if undecided is not None:
                    pending.append((i, cy, undecided))
            ys.append(cy)
            ls.append(l)
            rs.append(r)
            if l is not None and r is not None:
                ref = (l + r) / 2   # 다음(위) 띠는 이 띠의 중심을 기준으로 좌우 판단
            elif l is not None or r is not None:
                w = self.lane_width(i) / 2
                ref = l + w if l is not None else r - w

        if pending:
            # 보류한 선: 화면 맨 아래까지 연장했을 때 차의 왼쪽에 닿으면 왼쪽 선, 오른쪽이면 오른쪽 선
            if len(pending) >= 2:
                fit = np.polyfit([p[1] for p in pending], [p[2] for p in pending], 1)
                xb = np.polyval(fit, FRAME_H)
            else:
                xb = pending[0][2]
            for i, _, x in pending:
                if xb < cx0 and ls[i] is None:
                    ls[i] = x
                elif xb >= cx0 and rs[i] is None:
                    rs[i] = x
        self.left, self.right = list(ls), list(rs)

        self.gap_y = self.find_gap(bands, ls[0] is not None and rs[0] is not None)
        if self.gap_y is not None:
            dashed_hline(dbg, self.gap_y, C_GAP, thick=2)
            chip(dbg, "LANE ENDS", FRAME_W // 2 - 40, self.gap_y - 22, C_GAP)

        pts = []
        for i, (cy, l, r) in enumerate(zip(ys, ls, rs)):
            half_w = self.lane_width(i) / 2
            if LINE_MODE == "line":
                c = l
            elif l is not None and r is not None:
                c = (l + r) / 2
                self.learn_width(i, r - l)
            elif l is not None:     # 차로 폭은 가로 방향으로 잰 값이라 그대로 반만큼 옮김
                c = l + half_w
            elif r is not None:
                c = r - half_w
            else:
                c = None
            self.center[i] = c
            for x, color in ((l, C_LEFT), (r, C_RIGHT)):
                if x is not None:
                    cv2.circle(dbg, (int(x), cy), 5, (20, 20, 20), -1, cv2.LINE_AA)
                    cv2.circle(dbg, (int(x), cy), 4, color, -1, cv2.LINE_AA)
            if c is not None:
                cv2.circle(dbg, (int(c), cy), 3, C_PATH, -1, cv2.LINE_AA)
                pts.append((cy, c))

        if not pts:
            self.misses += 1
            if self.misses > 5:     # 오래 놓치면(차를 들었다 놓는 등) 추적 기록 초기화
                self.left, self.right, self.center = [None] * SLICES, [None] * SLICES, [None] * SLICES
            self.draw_marks(dbg, marks)
            return None, junction, dbg
        self.misses = 0

        # 다른 띠들과 동떨어진 중심점(오인식) 제외: 두 점씩 골라 만든 직선 중 가장 많은 점이 맞는 것을 채택
        if len(pts) >= 3:
            best = []
            for a in range(len(pts)):
                for b in range(a + 1, len(pts)):
                    (ya, xa), (yb, xb) = pts[a], pts[b]
                    k = (xb - xa) / (yb - ya)
                    inl = [(y, x) for y, x in pts if abs(xa + k * (y - ya) - x) < OUTLIER_PX]
                    if len(inl) > len(best):
                        best = inl
            if len(best) >= 2:
                for y, x in pts:
                    if (y, x) not in best:
                        cv2.circle(dbg, (int(x), y), 4, C_DIM, -1, cv2.LINE_AA)    # 오인식으로 뺀 점
                pts = best

        near_y, far_y = FRAME_H - h // 2, top + h // 2
        if len(pts) >= 2:
            fit = np.polyfit([y for y, _ in pts], [x for _, x in pts], 1)
            x_near, x_far = np.polyval(fit, near_y), np.polyval(fit, far_y)
        else:
            x_near = x_far = pts[0][1]
        cv2.line(dbg, (int(x_near), near_y), (int(x_far), far_y), (20, 20, 20), 6, cv2.LINE_AA)
        cv2.line(dbg, (int(x_near), near_y), (int(x_far), far_y), C_PATH, 3, cv2.LINE_AA)

        # 오차 단위: 화면 반폭 기준 × (REF_LANE_PX / LANE_WIDTH_PX). 화각이 바뀌어 차로가 작게 보여도
        # 같은 차로 이탈은 같은 값이 되게 함 → 조향 게인(KP, KH)과 임계값은 그대로
        half = FRAME_W / 2 * LANE_WIDTH_PX / REF_LANE_PX
        offset = float(np.clip((x_near - cx0) / half, -1, 1))
        heading = float(np.clip((x_far - x_near) / half, -1, 1))
        self.draw_marks(dbg, marks)
        return (offset, heading), junction, dbg

    @staticmethod
    def draw_marks(dbg, marks):
        """교차로/횡단보도 이름표를 다른 표시들 위에 (가장 가까운 띠 높이에)"""
        for name, (color, cy) in marks.items():
            chip(dbg, name, 12, cy - 10, color)


class TrafficLight:
    """신호등 램프 색 판별 → "red" / "green" / None (매 프레임).
    차선 영역(ROI_TOP)보다 위쪽, 화면 좌우(LIGHT_SIDE)만 본다. 두 가지 방법을 같이 씀:
      1. 채도/밝기가 높고 거의 정사각형인 빨강(H<10, >170) / 초록(H 40~80) 덩어리
         (초록 범위를 80에서 끊은 건 청록색 장애물 블록(H 85~95)을 초록불로 잡지 않게)
      2. LED는 가운데가 하얗게 날아가고 테두리에만 색이 남음 → 하얀 덩어리 둘레(링)의 색"""

    RING_S_MIN = 60         # 링 색 판별에 쓰는 최소 채도 (링은 덩어리보다 옅음)
    CORE_V_MIN = 230        # 하얗게 날아간 LED 가운데로 볼 밝기
    CORE_S_MAX = 50         # 가운데로 볼 최대 채도 (거의 흰색만. 옅은 색 테두리까지 가운데로 묶이지 않게)

    def __init__(self):
        self.state = None
        self.boxes = []         # (x, y, w, h, 색) 화면 표시용
        self.local = []
        self.lamps = {}         # 색별 가장 큰 램프 박스 {"red": (x, y, w, h), "green": ...}. wait_green의 위치 비교용
        self.mask_vis = None    # tune_light.py에서 보는 색 마스크 (keep_mask=True일 때만)
        self.keep_mask = False

    @staticmethod
    def regions():
        y0, y1 = int(FRAME_H * LIGHT_Y[0]), int(FRAME_H * LIGHT_Y[1])
        sw = int(FRAME_W * LIGHT_SIDE)
        return [(0, y0, sw, y1), (FRAME_W - sw, y0, FRAME_W, y1)]

    @staticmethod
    def is_red(hue):
        return (hue < 10) | (hue > 170)

    @staticmethod
    def is_green(hue):
        return (hue >= 40) & (hue <= 80)

    @staticmethod
    def lamp_like(w, h, area, area_min):
        """램프 모양: 크기 범위 안, 거의 정사각형, 박스를 꽤 채움(동그라미/네모)"""
        return area_min <= area <= LIGHT_AREA_MAX and 0.6 <= w / h <= 1.6 and area >= 0.45 * w * h

    def __call__(self, frame):
        self.boxes = []
        self.lamps = {}
        self.local = []         # color_at으로 다시 본 자리 (화면 표시용)
        biggest = {"red": 0, "green": 0}

        def add(color, x, y, w, h, area):
            self.boxes.append((x, y, w, h, color))
            if area > biggest[color]:
                biggest[color] = area
                self.lamps[color] = (int(x), int(y), int(w), int(h))

        if self.keep_mask:
            self.mask_vis = np.zeros_like(frame)
        for x0, y0, x1, y1 in self.regions():
            hue, sat, val = cv2.split(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV))
            strong = (sat >= LIGHT_S_MIN) & (val >= LIGHT_V_MIN)

            # 1. 색 덩어리
            for color, mask in (("red", strong & self.is_red(hue)), ("green", strong & self.is_green(hue))):
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                if self.keep_mask:
                    self.mask_vis[y0:y1, x0:x1][mask > 0] = (0, 0, 255) if color == "red" else (0, 255, 0)
                _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for x, y, w, h, area in stats[1:]:
                    if self.lamp_like(w, h, area, LIGHT_AREA_MIN):
                        add(color, x0 + x, y0 + y, w, h, area)

            # 2. 가운데가 하얗게 날아간 LED: 둘레 링의 색
            core = ((val >= self.CORE_V_MIN) & (sat <= self.CORE_S_MAX)).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if not self.lamp_like(w, h, area, max(LIGHT_AREA_MIN // 2, 4)):
                    continue
                pad = max(2, min(w, h) // 2)
                sy = slice(max(y - pad, 0), y + h + pad)
                sx = slice(max(x - pad, 0), x + w + pad)
                blob = (labels[sy, sx] == i).astype(np.uint8)
                ring = cv2.dilate(blob, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)).astype(bool) & (blob == 0)
                colored = ring & (sat[sy, sx] >= self.RING_S_MIN) & (val[sy, sx] >= LIGHT_V_MIN // 2)
                if colored.sum() < 0.3 * ring.sum():
                    continue
                h_ring = hue[sy, sx][colored]
                color = ("red" if self.is_red(h_ring).mean() > 0.6 else
                         "green" if self.is_green(h_ring).mean() > 0.6 else None)
                if color:
                    add(color, x0 + x, y0 + y, w, h, area)

        # wait_green이 '빨간불이었던 자리'들을 다시 볼 수 있게 빨간 램프 후보를 전부 남김
        self.lamps["red_all"] = [(int(x), int(y), int(w), int(h)) for x, y, w, h, c in self.boxes if c == "red"]
        if biggest["red"] or biggest["green"]:
            self.state = "red" if biggest["red"] >= biggest["green"] else "green"
        else:
            self.state = None
        return self.state

    def color_at(self, frame, box):
        """빨간불이었던 박스 자리(주변 번짐 포함)만 다시 봐서 "red" / "green" / None.
        위치를 아니까 색 기준을 넓게 씀: 초록은 청록빛까지(LIGHT_G_HUE), 채도/밝기 기준도 낮게.
        LED 가운데가 하얗게 날아가도 둘레 번짐에 색이 남아서 잡힘"""
        x, y, w, h = box
        pad = max(w, h)
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, FRAME_W), min(y + h + pad, FRAME_H)
        hue, sat, val = cv2.split(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV))
        colored = (sat >= LIGHT_LOCAL_S_MIN) & (val >= LIGHT_LOCAL_V_MIN)
        red = int((colored & self.is_red(hue)).sum())
        green = int((colored & (hue >= LIGHT_G_HUE[0]) & (hue <= LIGHT_G_HUE[1])).sum())
        need = max(LIGHT_AREA_MIN // 2, int(0.15 * w * h))
        color = "green" if green >= need and green > 2 * red else "red" if red >= need and red > 2 * green else None
        self.local.append((x0, y0, x1, y1, color))
        return color

    def draw(self, dbg, regions=True):
        """regions: 램프를 찾는 영역(좌우 위쪽) 테두리도 그릴지. 주행 화면은 신호 대기 중에만 그림"""
        if regions:
            for x0, y0, x1, y1 in self.regions():
                cv2.rectangle(dbg, (x0, y0), (x1 - 1, y1 - 1), C_DIM, 1)
        for x0, y0, x1, y1, color in self.local:    # wait_green이 다시 보는 '빨간불이었던 자리' (노랑)
            cv2.rectangle(dbg, (x0, y0), (x1 - 1, y1 - 1), C_AT, 1, cv2.LINE_AA)
            if color:
                chip(dbg, color.upper(), x0, y1 + 2, C_AT, 0.35)
        for x, y, w, h, color in self.boxes:
            bgr = (40, 40, 255) if color == "red" else (60, 220, 60)
            cv2.rectangle(dbg, (x - 4, y - 4), (x + w + 4, y + h + 4), bgr, 2, cv2.LINE_AA)
            chip(dbg, color.upper(), x - 4, y - 22, bgr, 0.38)


class Steering:
    """(위치오차, 방향오차) → 조향각(도). P + 필터링된 D + 저역통과로 가변 조향"""

    def __init__(self):
        self.prev_err = 0.0
        self.prev_t = None
        self.d = 0.0
        self.angle = 0.0

        self.out_t = None

    def lane(self, offset, heading, now, kp=1.0, kh=1.0):
        """차선 추종 조향 목표(도). 출력은 하지 않음. kp, kh: style= 배수"""
        # 차로 중앙에서 멀어질수록 방향 오차의 비중을 줄임 → 차선 끝에 붙었을 때는 무조건 복귀부터
        err = kp * KP * offset + kh * KH * heading * max(0.0, 1 - abs(offset) / OFFSET_PRIORITY)
        if self.prev_t is not None and now > self.prev_t:
            # 프레임 간 미분은 노이즈가 커서 그대로 쓰면 지그재그의 원인이 됨 → 필터링
            self.d = 0.7 * self.d + 0.3 * (err - self.prev_err) / (now - self.prev_t)
        self.prev_err, self.prev_t = err, now
        return float(np.clip(err + KD * self.d, -1, 1)) * AUTO_STEER_MAX

    def output(self, target, now, smooth=None):
        """목표 각도 → 부드럽게, 변화 속도 제한을 걸어 실제 조향각으로. 모든 코스 동작이 이걸 거침"""
        dt = now - self.out_t if self.out_t is not None and now > self.out_t else 1 / 30
        self.out_t = now
        smooth = SMOOTH if smooth is None else smooth
        target = smooth * self.angle + (1 - smooth) * target
        step = STEER_RATE * dt      # 한 프레임에 꺾을 수 있는 최대량
        self.angle += float(np.clip(target - self.angle, -step, step))
        return self.angle

    def __call__(self, offset, heading, now):
        return self.output(self.lane(offset, heading, now), now)


class CourseRunner:
    """COURSE의 동작을 위에서부터 순서대로 실행.
    각 동작은 sec(정해진 시간) 또는 until(교차로/횡단보도/새 차선/차선 끊김/초록불) 조건으로 끝나고,
    min_sec ~ max_sec 범위를 지킨다. speed는 그 동작 동안만 적용된다."""

    EVENT_NAMES = {"junction": "교차로", "crosswalk": "횡단보도", "lane": "새 차선",
                   "gap": "차선 끊김", "green": "초록불", "obstacle": "장애물 감지", "clear": "장애물 없어짐",
                   "lost": "차선 놓침"}
    STATE_EVENTS = ("lane", "green", "clear")   # '지금 상태'로 판단: 써도 다시 셀 필요 없음 (장애물은 한 번 쓰면 없어졌다 다시 나타나야 셈)
    AT_EVENTS = ("junction", "crosswalk", "gap")    # at= (화면 어디까지 내려와야 하는지)을 쓸 수 있는 이벤트

    def __init__(self, course, start=1):
        from course import _flat
        self.course = _flat(course)     # 묶음(리스트)이 섞여 있어도 펼쳐서 씀 → COURSE에 _flat 안 써도 됨
        self.idx = start - 1        # 지금 실행 중인 동작 번호 (0부터)
        self.step_start = None
        self.done = False
        self.hold = None            # straight(angle="hold"): 시작 순간의 조향각
        self.red_hits = 0           # wait_green: 빨간불 연속 감지 프레임 수
        self.red_seen = False       # wait_green: 이 동작에서 빨간불을 확실히 봤는지 (빨강 → 초록으로 바뀌어야 출발)
        self.red_box = None         # wait_green: 마지막으로 본 빨간 램프 박스 (초록불은 이 근처여야 인정)
        self.red_boxes = []         # wait_green: 빨간불일 때 보인 빨간 램프 후보 전부 (그 자리들을 다시 봄)
        self.local_light = None     # wait_green: 그 자리들을 다시 본 결과 (Driver가 매 프레임 채움)
        # 이벤트별: 연속 감지 프레임 수, 무장 여부 (출발선처럼 처음부터 보이던 것, 방금 쓴 것은 한 번 사라져야 다시 셈)
        # 초록불, 장애물/없음은 처음부터 그 상태여도 바로 인정해야 하므로 항상 무장
        self.hits = {"junction": 0, "crosswalk": 0, "lane": 0, "gap": 0, "green": 0, "obstacle": 0, "clear": 0,
                     "lost": 0}
        self.armed = {"junction": False, "crosswalk": False, "lane": True, "gap": False, "green": True,
                      "obstacle": True, "clear": True, "lost": True}
        self.since = {k: None for k in self.hits}   # 이벤트별: 지금 상태가 시작된 시각 (hold_sec 판단용)

    def step(self):
        step = self.course[self.idx] if self.idx < len(self.course) else None
        if step is not None and step["angle"] is None:     # right()/left() 기본: 차가 꺾을 수 있는 최대 각도
            step = dict(step, angle=AUTO_STEER_MAX if step["do"] == "right" else -AUTO_STEER_MAX)
        return step

    @property
    def speed(self):
        step = self.step()
        return None if step is None else step["speed"]

    def begin(self, now):
        self.step_start = now
        self.hits["lane"] = 0
        self.hold = None
        self.red_hits, self.red_seen, self.red_box = 0, False, None
        self.red_boxes, self.local_light = [], None
        step = self.step()
        if step is None:
            print(f"\n[코스 끝] 이제부터 차선만 따라감")
            return
        if step["do"] == "straight" and step["until"] == "lane":
            # 끊기기 전 차선이 아직 보여서 바로 "새 차선"으로 끝나지 않게: 한 번 사라졌다가 다시 잡혀야 끝
            self.armed["lane"] = False
        if step["sec"] is not None:
            how = f"{step['sec']}초"
        else:
            how = f"{self.EVENT_NAMES[step['until']]}까지" if step["until"] else ""
            if step.get("at") is not None:
                how += f" (화면 {step['at']} 지점)"
            if step.get("hold"):
                how += f" {step['hold']}초 유지"
            if step["min"]:
                how += f" 최소 {step['min']}초"
            if step["max"] is not None:
                how += f" 최대 {step['max']}초"
        if step["do"] in ("left", "right", "curve"):
            angle = f" {step['angle']:+.0f}도"
        elif step["do"] == "straight" and step["angle"] != 0:
            angle = " 시작 각도 유지" if step["angle"] == "hold" else f" {step['angle']:+.0f}도"
        else:
            angle = ""
        speed = f" 속도 {step['speed']}" if step["speed"] is not None else ""
        print(f"\n[{self.idx + 1}/{len(self.course)}] {step['do']}{angle} {how}{speed}")

    def near_red(self, box):
        """초록 램프가 빨간 램프가 있던 자리 근처인지 (같은 신호등). 램프 크기에 비례한 거리로 판단"""
        if self.red_box is None or box is None:
            return False
        rx, ry, rw, rh = self.red_box
        gx, gy, gw, gh = box
        dist = ((rx + rw / 2 - gx - gw / 2) ** 2 + (ry + rh / 2 - gy - gh / 2) ** 2) ** 0.5
        return dist <= LIGHT_NEAR * max(rw, rh)

    def watch(self, junction, crosswalk, det, ys=None, light=None, lamps=None, obstacle=None, now=0.0):
        """이벤트별 연속 감지 프레임 수 갱신.
        ys = {"junction": y, "crosswalk": y, "gap": y}: 화면에서 그 선이 보인 가장 아래 y.
        지금 동작에 at=이 있으면 그 이벤트는 선이 화면의 at 지점까지 내려와야 감지로 침.
        light = 신호등 색 ("red" / "green" / None).
        초록불은 이 동작에서 빨간불을 JUNCTION_FRAMES 연속으로 본 뒤에만 셈 (빨강 → 초록으로 바뀌어야 출발).
        처음부터 보이던 초록 물체나, 차가 오기 전부터 켜져 있던 초록불로는 출발하지 않음.
        lamps = 색별 램프 박스 (TrafficLight.lamps): 초록은 빨간 램프가 있던 자리 근처(LIGHT_NEAR)여야 인정.
        lamps를 안 주면 위치는 비교하지 않음.
        obstacle = 앞 적외선 센서에 물체가 있는지 (센서가 없으면 None → 항상 '없음')"""
        ys = ys or {}
        self.red_hits = self.red_hits + 1 if light == "red" else 0
        if self.red_hits >= JUNCTION_FRAMES:
            self.red_seen = True
            if lamps and lamps.get("red"):
                self.red_box = lamps["red"]
            # 본 적 있는 빨간 자리는 지우지 않고 쌓음: 램프가 초록으로 바뀌면 그 빨간 박스는 사라지지만
            # (다른 빨간 물체 때문에 여전히 '빨강'이어도) 그 자리를 계속 다시 봐야 하므로
            for box in (lamps or {}).get("red_all", []):
                if len(self.red_boxes) < 6 and not any(
                        abs(box[0] - b[0]) < max(b[2], 4) and abs(box[1] - b[1]) < max(b[3], 4) for b in self.red_boxes):
                    self.red_boxes.append(box)
        green_here = lamps is None or self.near_red(lamps.get("green"))
        seen = {"junction": junction, "crosswalk": crosswalk,
                "lane": not junction and det is not None and abs(det[0]) < 0.6 and abs(det[1]) < 0.5,
                "gap": ys.get("gap") is not None,
                # 초록: 화면 전체에서 빨간불 근처에 초록불이 보이거나, 빨간불이었던 자리가 초록(청록 포함)으로 바뀜
                "green": self.red_seen and ((light == "green" and green_here) or self.local_light == "green"),
                "obstacle": bool(obstacle), "clear": not obstacle,
                "lost": det is None}
        step = self.step()
        if step is not None and step.get("at") is not None and step["until"] in self.AT_EVENTS:
            y = ys.get(step["until"])
            seen[step["until"]] = y is not None and y >= step["at"] * FRAME_H - 1    # -1: 0.55*360=198.00000000000003 같은 오차
        for k, v in seen.items():
            self.hits[k] = self.hits[k] + 1 if v else 0
            self.since[k] = (self.since[k] if self.since[k] is not None else now) if v else None
            if not v:
                self.armed[k] = True

    def happened(self, event, now=0.0, hold=0.0):
        need = TURN_CLEAR_FRAMES if event == "lane" else JUNCTION_FRAMES
        held = hold <= 0 or (self.since[event] is not None and now - self.since[event] >= hold)
        return self.armed[event] and self.hits[event] >= need and held

    def update(self, junction, now, det=None, crosswalk=False, ys=None, light=None, lamps=None, obstacle=None):
        if self.step_start is None:
            self.begin(now)
        self.watch(junction, crosswalk, det, ys, light, lamps, obstacle, now)
        while (step := self.step()) is not None:
            if step["do"] == "finish":
                self.done = True
                return
            elapsed = now - self.step_start
            if step["sec"] is not None:
                reason = "시간" if elapsed >= step["sec"] else None
            elif step["max"] is not None and elapsed >= step["max"]:
                reason = "최대 시간"
            elif step["max"] is None and step["until"] is not None and step["do"] != "wait" and elapsed >= STEP_MAX_SEC:
                reason = f"안전 시간 {STEP_MAX_SEC}초 (조건이 안 와서 넘어감, max_sec로 조절)"
            elif (step["until"] is not None and elapsed >= step["min"]
                  and self.happened(step["until"], now, step.get("hold", 0.0))):
                reason = self.EVENT_NAMES[step["until"]]
                if step["until"] not in self.STATE_EVENTS:
                    self.armed[step["until"]] = False   # 같은 선을 다음 동작에서 또 세지 않게
            else:
                reason = None
            if reason is None:
                return
            print(f"\n[{self.idx + 1} 끝] {reason} ({elapsed:.1f}초)")
            self.idx += 1
            self.begin(now)

    def lane_side(self, now):
        step = self.step()
        return (step or {}).get("side") or "center"

    def opt(self, key, default=None):
        """지금 동작의 옵션 (shift, steer, style ...). 코스가 끝났거나 안 썼으면 default"""
        step = self.step()
        value = (step or {}).get(key)
        return default if value is None else value

    def turning(self, now):
        """회전 동작 중이면서 아직 차선 추종으로 넘어가기 전"""
        step = self.step()
        if step is None or step["do"] not in ("left", "right"):
            return False
        return now - self.step_start < step["min"] or self.hits["lane"] < TURN_CLEAR_FRAMES

    def following(self):
        step = self.step()
        return step is None or step["do"] == "follow"

    def target(self, lane, now, current=0.0):
        """지금 동작의 조향 목표(도). lane = 차선 추종 조향(차선이 안 보이면 None),
        current = 지금 실제 조향각 (straight(angle="hold")가 시작 순간 각도를 기억하는 데 씀).
        고정 각도는 없음: 모든 동작이 차선이 보이는 만큼 차선 추종 조향을 섞어 씀"""
        step = self.step()
        if step is None or step["do"] in ("follow", "stop", "wait", "finish"):
            return lane
        angle = step["angle"]
        if step["do"] == "straight":
            if angle == "hold":
                if self.hold is None:
                    self.hold = current
                angle = self.hold
            if lane is None:
                return angle
            return angle + float(np.clip(lane - angle, -STRAIGHT_CORR, STRAIGHT_CORR))
        if lane is None:
            return angle
        if step["do"] == "curve":
            return angle + CURVE_ASSIST * (lane - angle)
        # 회전: 최소 시간 동안은 지정 각도로, 그 뒤 새 차선이 잡힐수록 차선 추종 조향으로 넘어감
        if now - self.step_start < step["min"]:
            return angle
        w = max(0.0, 1 - self.hits["lane"] / TURN_CLEAR_FRAMES)
        return w * angle + (1 - w) * lane

    def stopped(self, now):
        step = self.step()
        return step is not None and step["do"] in ("stop", "wait")

    def needs_light(self):
        """신호등 인식이 필요한 동작인지 (wait_green). 아닐 때는 신호등 계산을 건너뜀"""
        step = self.step()
        return step is not None and step["do"] == "wait" and step["until"] == "green"

    def label(self):
        return f"{min(self.idx + 1, len(self.course))}/{len(self.course)}"

    def status(self, now):
        step = self.step()
        if step is None:
            return "END"
        if step["do"] in ("left", "right", "curve"):
            return f"{step['do'].upper()} {step['angle']:+.0f}"
        if step["do"] == "wait":
            if step["until"] == "green":
                return "WAIT GREEN" if self.red_seen else "WAIT RED"
            return {"obstacle": "WAIT OBSTACLE", "clear": "WAIT CLEAR"}.get(step["until"], "WAIT")
        if step["do"] == "straight" and step["angle"] == "hold":
            return "STRAIGHT HOLD"
        return step["do"].upper()


def speed_for(angle, base):
    """조향이 클수록 감속"""
    if base <= MIN_SPEED:   # 동작별로 아주 느린 속도를 정했으면 그대로
        return base
    ratio = min(abs(angle) / STEER_LIMIT, 1.0)
    return base - (base - MIN_SPEED) * ratio


class Stream:
    """디버그 화면을 브라우저로 송출.
    페이지가 최신 화면을 한 장씩 받아감(/frame.jpg): 와이파이가 잠깐 끊기거나 프로그램이 다시 시작돼도
    페이지가 알아서 다시 연결하고, 늦게 받는 쪽은 밀린 화면 없이 항상 최신 화면만 받음.
    JPEG 인코딩은 메인 루프가 아니라 브라우저 요청 스레드에서, 보는 사람이 있을 때만 함"""

    LEGEND = [("#3cd2ff", "흰 선으로 인식"), ("#288cff", "왼쪽 선"), ("#ff3c3c", "오른쪽 선"),
              ("#50e650", "따라갈 차로 중앙 / 화살표 = 지금 조향"), ("#ff00ff", "교차로 가로선 (점선: 멀리서 미리 본 것)"),
              ("#f0f0f0", "횡단보도"), ("#0080ff", "차선 끊김"), ("#ffe600", "이 동작이 끝나는 지점 (at)"),
              ("#ff2828", "빨간불"), ("#3cdc3c", "초록불")]

    PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px -apple-system,sans-serif}
header{display:flex;align-items:center;gap:10px;padding:8px 14px;max-width:1400px;margin:0 auto}
header b{color:#fff} #st{margin-left:auto;display:flex;align-items:center;gap:6px;color:#aaa;font-size:13px}
#dot{width:10px;height:10px;border-radius:50%;background:#666}
.wrap{position:relative;max-width:1400px;margin:0 auto}
img{display:block;width:100%;background:#000;min-height:120px}
#msg{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
 background:rgba(0,0,0,.55);color:#fff;font-size:18px;font-weight:600}
.lg{display:flex;flex-wrap:wrap;gap:6px 16px;padding:10px 14px;max-width:1400px;margin:0 auto}
.lg span{display:flex;align-items:center;gap:6px} .lg i{width:14px;height:14px;border-radius:3px;display:inline-block}
</style></head><body>
<header><b>__TITLE__</b><small style="color:#888">실시간 인식 화면</small>
<span id="st"><i id="dot"></i><span id="stt">연결 중…</span></span></header>
<div class="wrap"><img id="v" alt=""><div id="msg">연결이 끊겼어요 · 다시 연결하는 중…</div></div>
<div class="lg">__LEGEND__</div>
<script>
// 최신 화면을 한 장씩 받아옴. 실패하면 잠깐 쉬고 계속 다시 시도 (프로그램이 다시 켜지면 자동으로 이어짐)
const img = document.getElementById("v"), msg = document.getElementById("msg"),
      dot = document.getElementById("dot"), stt = document.getElementById("stt");
let seq = -1, fails = 0, shown = [], url = null;
function status(ok, text) { dot.style.background = ok ? "#3cdc3c" : "#ff4040"; stt.textContent = text; }
async function loop() {
  for (;;) {
    const ctl = new AbortController(), t = setTimeout(() => ctl.abort(), 3000);
    try {
      const r = await fetch("/frame.jpg?after=" + seq, {cache: "no-store", signal: ctl.signal});
      clearTimeout(t);
      if (r.status === 204) {     // 1초 동안 새 화면 없음 (프로그램은 켜져 있음)
        fails = 0; msg.style.display = "none"; status(true, "연결됨 · 새 화면 없음"); continue;
      }
      if (!r.ok) throw new Error(r.status);
      seq = +r.headers.get("X-Seq");
      const u = URL.createObjectURL(await r.blob());
      img.src = u; if (url) URL.revokeObjectURL(url); url = u;
      fails = 0; msg.style.display = "none";
      const now = performance.now(); shown.push(now); shown = shown.filter(x => now - x < 2000);
      status(true, "연결됨 · " + (shown.length / 2).toFixed(0) + " fps");
    } catch (e) {
      clearTimeout(t); fails++;
      if (fails >= 2) { msg.style.display = "flex"; status(false, "끊김 · 다시 연결 중"); }
      await new Promise(ok => setTimeout(ok, Math.min(300 * fails, 2000)));
    }
  }
}
loop();
</script></body></html>"""

    def __init__(self, port, title="AICAR", legend=False):
        import logging
        from flask import Flask, Response
        logging.getLogger("werkzeug").setLevel(logging.ERROR)   # 한 장마다 찍히는 요청 기록 끄기
        self.img, self.seq = None, 0            # 메인 루프가 넣은 최신 화면과 번호
        self.jpg, self.jpg_seq = None, -1       # 마지막으로 인코딩한 JPEG와 그 화면 번호
        self.lock = threading.Lock()
        app = Flask(__name__)
        # LEGEND 색은 화면 색(BGR)을 웹 색(RGB)으로 바꾼 것
        items = "".join(f'<span><i style="background:{c}"></i>{t}</span>' for c, t in self.LEGEND) if legend else ""
        page = self.PAGE.replace("__TITLE__", title).replace("__LEGEND__", items)

        @app.route("/")
        def index():
            return Response(page, mimetype="text/html", headers={"Cache-Control": "no-store"})

        @app.route("/frame.jpg")
        def frame():
            # after보다 새 화면이 나올 때까지 최대 1초 기다렸다가 최신 화면 하나만 보냄
            from flask import request
            after = request.args.get("after", -1, type=int)
            if after > self.seq:        # 프로그램이 다시 시작돼서 번호가 처음부터: 그냥 최신 화면
                after = -1
            end = time.monotonic() + 1.0
            while (self.img is None or self.seq <= after) and time.monotonic() < end:
                time.sleep(0.01)
            jpg, seq = self.encoded()
            if jpg is None or seq <= after:
                return Response(status=204)
            return Response(jpg, mimetype="image/jpeg",
                            headers={"X-Seq": str(seq), "Cache-Control": "no-store"})

        @app.route("/video")
        def video():
            """예전 방식(MJPEG 연속 스트림). 다른 프로그램에서 주소로 바로 볼 때용"""
            def gen():
                last = -1
                while True:
                    time.sleep(0.05)
                    jpg, seq = self.encoded()
                    if jpg is not None and seq != last:
                        last = seq
                        yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

        threading.Thread(target=lambda: app.run(host="0.0.0.0", port=port, threaded=True),
                         daemon=True).start()

    def update(self, img):
        """화면만 넘겨 둠 (인코딩 안 함). img는 이후에 다시 고치지 않는 새 배열이어야 함"""
        self.img = img
        self.seq += 1

    def encoded(self):
        """최신 화면의 JPEG. 새 화면이 있을 때만 인코딩 (브라우저가 여러 개여도 한 번)"""
        with self.lock:
            img, seq = self.img, self.seq
            if img is not None and seq != self.jpg_seq:
                self.jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
                self.jpg_seq = seq
            return self.jpg, self.jpg_seq


class Recorder:
    """원본 프레임 + 로그를 runs/<시각>/ 에 저장. 맥에서 replay.py로 다시 돌려볼 수 있음.
    JPEG 저장은 별도 스레드에서 해서 주행 루프가 기다리지 않음 (close()에서 남은 것까지 다 저장)"""

    def __init__(self):
        self.dir = Path(__file__).with_name("runs") / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True)
        self.log = open(self.dir / "log.csv", "w")
        self.log.write("frame,t,offset,heading,junction,steer,speed,step,status\n")
        self.n = 0
        self.queue = queue.Queue(maxsize=300)   # 저장이 밀리면(약 5초치) 루프가 잠깐 기다림. 프레임은 안 버림
        self.writer = threading.Thread(target=self._save_frames, daemon=True)
        self.writer.start()
        print(f"녹화: {self.dir}")

    def _save_frames(self):
        while (item := self.queue.get()) is not None:
            path, frame = item
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def write(self, frame, t, det, junction, steer, speed, course, now):
        self.queue.put((str(self.dir / f"{self.n:05d}.jpg"), frame))
        off, head = det if det else ("", "")
        self.log.write(f"{self.n},{t:.3f},{off},{head},{int(junction)},{steer:.2f},{speed:.2f},"
                       f"{course.idx},{course.status(now)}\n")
        self.n += 1

    def close(self):
        self.queue.put(None)
        self.writer.join()
        self.log.close()


class Driver:
    """카메라 한 프레임 → (조향각, 속도). 실제 주행(main)과 시뮬레이터가 똑같이 이걸 씀"""

    def __init__(self, course, start=1, base_speed=BASE_SPEED, obstacle=None):
        self.detect, self.traffic, self.steering = LaneDetector(), TrafficLight(), Steering()
        self.course = CourseRunner(course, start)
        self.base_speed = base_speed
        self.obs = obstacle
        self.lost_since = None      # 차선을 놓친 시각
        self.stopped_since = 0.0    # 속도 0이 된 시각 (출발 킥 판단)
        self.kick_until = 0.0
        self.cw_since = None
        self.speed = 0.0
        self.edge = False
        self.done = False
        self.det, self.junction, self.light = None, False, None
        self.approach = (0.0, -1.0, -1)    # slow=: (감속 정도 0~1, 마지막으로 선을 본 시각, 동작 번호)

    def approach_speed(self, base, now):
        """slow=: 교차로/횡단보도/끊김 선이 화면 위쪽에 처음 보이면 base로 시작해서,
        선이 끝나는 지점(at, 없으면 JUNCTION_Y)까지 내려오는 동안 slow 속도까지 점점 줄임"""
        course = self.course
        step = course.step()
        slow = course.opt("slow")
        if slow is None or step["until"] not in CourseRunner.AT_EVENTS:
            return base
        level, seen_t, idx = self.approach
        if idx != course.idx:       # 새 동작: 처음부터
            level, seen_t = 0.0, -1.0
        y = {"junction": self.detect.junction_y, "crosswalk": self.detect.crosswalk_y,
             "gap": self.detect.gap_y}[step["until"]]
        far = step["until"] == "junction"
        if y is None and far:
            y = self.detect.far_y       # 아직 ROI에 안 들어왔으면 원거리에서 본 가로선
        if y is not None:
            start = FRAME_H * (FAR_TOP if far else ROI_TOP)
            end = (course.opt("at") or JUNCTION_Y) * FRAME_H
            level = max(level, float(np.clip((y - start) / max(end - start, 1.0), 0.0, 1.0)))
            seen_t = now
        elif now - seen_t > APPROACH_HOLD:
            level = 0.0
        self.approach = (level, seen_t, course.idx)
        slow = max(min(slow, base), MIN_SPEED)      # 모터가 서지 않는 최소 속도 아래로는 안 내림
        return base - (base - slow) * level

    # ---------- 인식 화면 (브라우저 Live View / replay) ----------
    PANEL_W = 240
    EVENT_EN = {"junction": "JUNCTION", "crosswalk": "CROSSWALK", "lane": "NEW LANE", "gap": "LANE ENDS",
                "green": "GREEN LIGHT", "obstacle": "OBSTACLE", "clear": "CLEAR", "lost": "LANE LOST"}
    DO_EN = {"follow": "FOLLOW LANE", "straight": "STRAIGHT", "right": "TURN RIGHT", "left": "TURN LEFT",
             "curve": "CURVE", "stop": "STOP", "wait": "WAIT", "finish": "FINISH"}

    def step_text(self, step):
        """동작 하나 → (큰 글자, 작은 글자 줄들)"""
        if step is None:
            return "COURSE END", ["follow lane"]
        title = self.DO_EN[step["do"]]
        if step["do"] in ("left", "right", "curve"):
            title += f" {abs(step['angle']):.0f}"
        if step["sec"]:
            cond = f"for {step['sec']:g}s"
        elif step["until"]:
            cond = "until " + self.EVENT_EN[step["until"]] + (f" @{step['at']:g}" if step.get("at") else "")
        else:
            cond = ""
        tags = []
        for key, fmt in (("speed", "spd {:g}"), ("slow", "slow->{:g}"), ("side", "{} line"),
                         ("shift", "shift {:+g}cm"), ("steer", "max {:g}deg"), ("style", "{}")):
            if step.get(key) is not None and not (key == "speed" and step["do"] in ("stop", "wait")):
                tags.append(fmt.format(step[key]))
        return title, [c for c in (cond, "  ".join(tags[:2]), "  ".join(tags[2:])) if c]

    def view(self, dbg, now, fps=None):
        """카메라 인식 화면 + 오른쪽 정보 패널을 합친 화면"""
        course, detect, steering = self.course, self.detect, self.steering
        step = course.step()
        self.traffic.draw(dbg, regions=course.needs_light())

        # at= 지점: 교차로/횡단보도 선이 이 점선까지 내려오면 이 동작이 끝남
        if step is not None and step["until"] in CourseRunner.AT_EVENTS and step["sec"] is None:
            at = step.get("at") or JUNCTION_Y       # at을 안 쓰면 JUNCTION_Y에서 끝남
            ay = int(at * FRAME_H)
            dashed_hline(dbg, ay, C_AT, thick=2)
            chip(dbg, f"END @{at:g}", FRAME_W - 90, ay - 22, C_AT, 0.38)

        # 차 위치(아래 가운데)와 지금 조향 방향 화살표
        cx0, by = int(FRAME_W * CAMERA_X), FRAME_H - 6
        a = np.radians(steering.angle * 1.6)
        tip = (int(cx0 + 70 * np.sin(a)), int(by - 70 * np.cos(a)))
        cv2.arrowedLine(dbg, (cx0, by), tip, (20, 20, 20), 7, cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(dbg, (cx0, by), tip, (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.3)

        # 가운데 위 큰 알림 (가장 중요한 하나만)
        lost = self.det is None
        banner = None
        if step is None or step["do"] == "finish" or self.done:
            banner = ("FINISH", (60, 160, 60)) if step is not None else None
        elif self.edge:
            banner = ("MAT EDGE - STOP", (40, 40, 220))
        elif step["do"] == "wait" and step["until"] == "green":
            banner = ("RED LIGHT - WAIT", (40, 40, 220)) if course.red_seen else ("FINDING RED LIGHT", (120, 100, 80))
        elif step["do"] == "wait":
            banner = ({"clear": "WAIT: REMOVE OBSTACLE", "obstacle": "WAIT: OBSTACLE"}.get(step["until"], "WAIT"),
                      (160, 120, 40))
        elif self.speed == 0 and not course.stopped(now) and step is not None:
            banner = ("STOPPED (LANE LOST)" if lost else "STOPPED", (40, 40, 220))
        elif lost and self.lost_since is not None:
            banner = (f"LANE LOST {now - self.lost_since:.1f}s", (0, 140, 255))
        if banner:
            (tw, _), _ = cv2.getTextSize(banner[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x = (FRAME_W - tw) // 2 - 12
            over = dbg.copy()
            cv2.rectangle(over, (x, 8), (x + tw + 24, 44), banner[1], -1)
            cv2.addWeighted(over, 0.8, dbg, 0.2, 0, dbg)
            cv2.putText(dbg, banner[0], (x + 12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # ---- 오른쪽 패널 ----
        W = self.PANEL_W
        p = np.full((FRAME_H, W, 3), 32, np.uint8)
        white, grey = (240, 240, 240), (150, 150, 150)

        def text(t, x, y, scale=0.45, color=white, thick=1):
            cv2.putText(p, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

        def bar(y, label, frac, value, color, mark=None):
            text(label, 12, y)
            text(value, W - 12 - cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0], y)
            cv2.rectangle(p, (12, y + 6), (W - 12, y + 16), (70, 70, 70), -1)
            cv2.rectangle(p, (12, y + 6), (12 + int((W - 24) * float(np.clip(frac, 0, 1))), y + 16), color, -1)
            if mark is not None:
                mx = 12 + int((W - 24) * float(np.clip(mark, 0, 1)))
                cv2.line(p, (mx, y + 3), (mx, y + 19), C_AT, 2)

        # 지금 동작 / 다음 동작
        idx, n = min(course.idx + 1, len(course.course)), len(course.course)
        text(f"STEP {idx}/{n}", 12, 24, 0.5, grey)
        el = now - course.step_start if course.step_start is not None else 0.0
        lim = step and (step["sec"] or step["max"])
        t = f"{el:.1f}s" + (f" / {lim:g}s" if lim else "")
        text(t, W - 12 - cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0], 24, 0.45, grey)
        cv2.rectangle(p, (12, 31), (W - 12, 34), (70, 70, 70), -1)
        cv2.rectangle(p, (12, 31), (12 + int((W - 24) * (course.idx / max(n, 1))), 34), C_PATH, -1)
        title, lines = self.step_text(step)
        text(title, 12, 60, 0.62, white, 2)
        y = 80
        for line in lines:
            text(line, 12, y, 0.42, C_AT if line.startswith("until") else grey)
            y += 18
        nxt = course.course[course.idx + 1] if course.idx + 1 < n else None
        text("next: " + (self.step_text(nxt)[0] if nxt else "-"), 12, 134, 0.42, grey)
        cv2.line(p, (12, 146), (W - 12, 146), (70, 70, 70), 1)

        # 속도 / 조향
        slow = course.opt("slow")
        bar(168, "SPEED", self.speed / 0.6, f"{self.speed:.2f}", C_PATH if self.speed > 0 else (40, 40, 220),
            None if slow is None else slow / 0.6)
        text("STEER", 12, 206)
        st = steering.angle
        t = f"{'L' if st < -0.5 else 'R' if st > 0.5 else ''} {abs(st):.0f} deg"
        text(t, W - 12 - cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0], 206)
        mid = W // 2
        cv2.rectangle(p, (12, 212), (W - 12, 222), (70, 70, 70), -1)
        sx = mid + int((W // 2 - 12) * float(np.clip(st / AUTO_STEER_MAX, -1, 1)))
        cv2.rectangle(p, (min(mid, sx), 212), (max(mid, sx), 222), (255, 200, 80), -1)
        cv2.line(p, (mid, 209), (mid, 225), white, 1)
        cv2.line(p, (12, 236), (W - 12, 236), (70, 70, 70), 1)

        # 상태 표시 (켜진 것만 색)
        on_light = course.local_light or self.light     # 빨간불이었던 자리를 다시 본 결과가 더 정확
        items = [("LANE", True, (60, 200, 60) if not lost else (40, 40, 220), "OK" if not lost else "LOST"),
                 ("JUNCTION", detect.junction_y is not None or detect.far_y is not None, C_JUNC, None),
                 ("CROSSWALK", detect.crosswalk_any, C_XWALK, None),
                 ("LANE ENDS", detect.gap_y is not None, C_GAP, None),
                 ("LIGHT", on_light is not None,
                  (40, 40, 255) if on_light == "red" else (60, 220, 60), (on_light or "").upper() or None),
                 ("OBSTACLE", bool(self.obs and self.obs.near), (0, 140, 255), None)]
        for k, (name, on, color, val) in enumerate(items):
            x, yy = 12 + (k % 2) * ((W - 24) // 2 + 2), 246 + (k // 2) * 26
            w = (W - 24) // 2 - 2
            cv2.rectangle(p, (x, yy), (x + w, yy + 20), color if on else (55, 55, 55), -1)
            label = name if val is None else f"{name} {val}"
            dark = on and sum(color) > 380
            cv2.putText(p, label, (x + 6, yy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (20, 20, 20) if dark else (white if on else grey), 1, cv2.LINE_AA)

        # 아래: 흰 선 기준, fps
        track = "white line" if LINE_COLOR == "white" else "black line"
        foot = f"{track}  thr {detect.thr:.0f} {'auto' if WHITE_THR is None else 'set'}"
        if fps:
            foot += f"   {fps:.0f} fps"
        text(foot, 12, FRAME_H - 12, 0.4, grey)
        return np.hstack([dbg, p])

    def tick(self, frame, now, show=False):
        """반환: (조향각 또는 None(유지), 모터 명령, 디버그 이미지)"""
        course, detect, steering = self.course, self.detect, self.steering
        if course.turning(now):
            # 회전 중에는 차가 빠르게 돌아서 직전 프레임의 선 위치 기억이 오히려 틀림 → 매 프레임 새로 판단
            detect.left, detect.right, detect.center = [None] * SLICES, [None] * SLICES, [None] * SLICES
        det, junction, dbg = detect(frame, course.lane_side(now))
        step = course.step()
        detect.far_y = None
        if step is not None and step.get("slow") is not None and step["until"] == "junction":
            detect.far_y = detect.far_hline(frame)      # 필요할 때만 계산 (fps 절약)
            if detect.far_y is not None:
                dashed_hline(dbg, detect.far_y, C_JUNC, dash=6, gap=6)
                chip(dbg, "JUNCTION AHEAD", FRAME_W - 130, detect.far_y - 20, C_JUNC, 0.38)
        light = self.traffic(frame) if course.needs_light() or show else None
        course.local_light = None
        if step is not None and step["do"] == "wait" and step["until"] == "green" and course.red_seen:
            # 빨간불이었던 자리들 중 하나라도 초록으로 바뀌었는지 (다른 빨간 물체가 있어도 램프 자리는 바뀜)
            colors = [self.traffic.color_at(frame, box) for box in course.red_boxes]
            course.local_light = "green" if "green" in colors else ("red" if "red" in colors else None)
        course.update(junction, now, det, detect.crosswalk,
                      {"junction": detect.junction_y, "crosswalk": detect.crosswalk_y, "gap": detect.gap_y},
                      light, self.traffic.lamps, self.obs.near if self.obs else None)
        self.det, self.junction, self.light = det, junction, light
        if course.done:
            self.done = True
            return None, 0.0, dbg

        base = course.speed if course.speed is not None else self.base_speed
        base = self.approach_speed(base, now)
        kp, kh, smooth = STYLES[course.opt("style", "normal")]
        if det is not None:
            # shift=+3: 차로 중앙보다 3cm 오른쪽을 따라감.
            # offset은 '차로 중앙이 카메라보다 오른쪽에 있는 정도'라, 차가 오른쪽에 붙으려면 차로 중앙이 왼쪽(-)에
            # 보여야 함 → 그 상태를 0으로 맞추도록 shift만큼 더함
            det = (det[0] + course.opt("shift", 0.0) * LANE_PX_PER_CM / (FRAME_W / 2), det[1])
        # 차 바로 앞 횡단보도: 줄무늬 때문에 차선 인식이 흔들리므로 잠깐 지금 조향을 유지하며 통과
        self.cw_since = (self.cw_since or now) if detect.crosswalk else None
        if self.cw_since is not None and now - self.cw_since < CROSSWALK_HOLD_MAX:
            lane = steering.angle
        else:
            lane = steering.lane(*det, now, kp, kh) if det is not None else None
        target = course.target(lane, now, steering.angle)
        steer = None
        if target is None:      # follow 중 차선을 놓침: 직전 조향 유지 (LOST_HOLD초까지)
            self.lost_since = self.lost_since if self.lost_since is not None else now
        else:
            self.lost_since = None
            limit = course.opt("steer", AUTO_STEER_MAX)
            steer = steering.output(float(np.clip(target, -limit, limit)), now, smooth)
        lost_long = self.lost_since is not None and now - self.lost_since > LOST_HOLD
        # 매트 끝 정지: 차선이 보이는 동안이나 회전 중에는 안 함 (오감지로 서는 것 방지)
        self.edge = detect.edge and det is None and not course.turning(now)
        if course.stopped(now) or lost_long or self.edge:
            speed = 0.0
        elif self.lost_since is not None:
            speed = min(base, MIN_SPEED)     # 놓친 동안은 천천히
        elif course.following():
            speed = speed_for(steering.angle, base)   # 차선 따라갈 때만 많이 꺾으면 감속
        else:
            speed = base        # 회전/직진/커브는 지정한 속도 그대로
        if speed > 0 and self.lost_since is None and abs(steering.angle) > HARD_STEER:
            speed += HARD_STEER_BOOST   # 많이 꺾으면 바퀴가 끌려 느려지므로 보충 (차선 놓친 동안은 제외)
        if speed == 0:
            if self.speed > 0:
                self.stopped_since = now
        elif self.speed == 0 and now - self.stopped_since >= KICK_AFTER_STOP:
            self.kick_until = now + KICK_TIME   # 실제로 서 있다가 출발할 때만 킥
        self.speed = speed
        cmd = KICK_SPEED if speed > 0 and now < self.kick_until else speed
        return steer, cmd, dbg


def open_camera(locked=True):
    """locked: calibrate.py가 저장한 고정 노출/화이트밸런스(CAMERA_LOCK)를 씀. False면 카메라 자동"""
    from libcamera import Transform, controls
    from picamera2 import Picamera2

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},   # OpenCV BGR 순서
        sensor={"output_size": CAMERA_SENSOR},
        transform=Transform(hflip=HFLIP, vflip=VFLIP),
        controls={"FrameRate": CAMERA_FPS}))
    cam.start()
    if LENS_POSITION is None:
        cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    else:
        cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": LENS_POSITION})
    if locked and CAMERA_LOCK:
        # 경기장 조명에서 잰 값으로 고정: 매트 반사나 사람 그림자에 밝기/색이 흔들리지 않게
        cam.set_controls({"AeEnable": False, "AwbEnable": False,
                          "ExposureTime": int(CAMERA_LOCK["ExposureTime"]),
                          "AnalogueGain": float(CAMERA_LOCK["AnalogueGain"]),
                          "ColourGains": tuple(CAMERA_LOCK["ColourGains"])})
    return cam


def stop_on_signal(*_):
    raise KeyboardInterrupt


def main():
    from car import Car

    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", action="store_true", help="모터 구동 (없으면 조향만)")
    ap.add_argument("--speed", type=float, default=BASE_SPEED)
    ap.add_argument("--start", type=int, default=1, help="COURSE 몇 번 동작부터 시작할지")
    ap.add_argument("--record", action="store_true", help="runs/ 폴더에 프레임과 로그 저장")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="KEY=VALUE 로 이번 실행만 설정 덮어쓰기 (예: KP=1.8)")
    args = ap.parse_args()

    for kv in args.set:
        key, val = kv.split("=", 1)
        if key not in globals() or not key.isupper():
            raise SystemExit(f"--set: 알 수 없는 설정 {key}")
        try:
            new = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            new = val       # "line" 같은 문자열
        print(f"{key}: {globals()[key]} -> {new}")
        globals()[key] = new

    # 백그라운드(&)로 시작하면 SIGINT가 무시된 채로 시작되므로 직접 다시 등록해야 Ctrl+C / kill -INT로 멈춤
    signal.signal(signal.SIGINT, stop_on_signal)
    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)   # ssh가 끊겨도 계속 주행 (정지는 Ctrl+C / aicar stop)

    car = Car()
    try:
        from obstacle import Obstacle
        obs = Obstacle()
    except Exception as e:      # 센서/모듈이 없어도 주행은 되게
        print(f"[경고] 장애물 센서 사용 안 함: {e}")
        obs = None
    if AUTO_TRIM:
        car.load_bias()     # 이전 주행에서 배운 중앙 보정을 이어서 사용
    cam = open_camera()
    driver = Driver(COURSE, args.start, args.speed, obs)
    stream = None if args.no_stream else Stream(args.port, "AICAR 주행", legend=True)
    rec = Recorder() if args.record else None
    print("모터", "ON" if args.drive else "OFF (조향만)",
          f"| 코스 {len(COURSE)}단계, {args.start}번부터 | Ctrl+C로 정지")

    frames, t0, t_start = 0, time.monotonic(), time.monotonic()
    fps = None
    prev_now = t_start
    try:
        while True:
            frame = cam.capture_array()
            now = time.monotonic()
            show = stream is not None and frames % 2 == 0      # 이번 프레임을 브라우저로 보낼지
            steer, cmd, dbg = driver.tick(frame, now, show)
            if driver.done:
                print("\n코스 완료")
                break
            if steer is not None:
                car.steer(steer)
            if driver.edge:
                print("\r[매트 끝] 정지        ", end="", flush=True)
            car.drive(cmd if args.drive else 0)
            course, detect, steering, traffic = driver.course, driver.detect, driver.steering, driver.traffic
            det, junction, light, speed = driver.det, driver.junction, driver.light, driver.speed

            # 자동 중앙 보정: 직선을 달리는데 계속 한쪽으로 꺾고 있다면 서보 중앙이 틀어진 것
            both = sum(l is not None and r is not None for l, r in zip(detect.left, detect.right))
            if (AUTO_TRIM and args.drive and speed > 0 and course.following() and det is not None
                    and both >= SLICES - 2 and abs(det[1]) < 0.05 and abs(steering.angle) < 8):
                car.bias += steering.angle * (now - prev_now) / AUTO_TRIM_TAU
            prev_now = now

            if rec:
                rec.write(frame, now - t_start, det, junction, steering.angle, speed, course, now)
            frames += 1
            if show:
                stream.update(driver.view(dbg, now, fps))
            if time.monotonic() - t0 >= 1:
                fps = frames / (time.monotonic() - t0)
                print(f"\r{fps:4.1f} fps  "
                      f"steer {steering.angle:+5.1f}  speed {speed:.2f}  "
                      f"step {course.label()} {course.status(now):9s} "
                      + (f"center {car.bias:+.1f} " if AUTO_TRIM else "")
                      + (f"light {light} " if light else "")
                      + ("GAP " if detect.gap_y is not None else "")
                      + f"{'LOST' if det is None else ''}    ", end="", flush=True)
                frames, t0 = 0, time.monotonic()
    except KeyboardInterrupt:
        print("\n정지")
    finally:
        if AUTO_TRIM and args.drive:
            car.save_bias()
            print(f"\n자동 중앙 보정 저장: {car.bias:+.1f}도")
        car.close()
        cam.stop()
        if obs:
            obs.close()
        if rec:
            rec.close()
            print(f"녹화 저장: {rec.dir} ({rec.n} 프레임)")


if __name__ == "__main__":
    main()
