"""Check installed modules and whether the camera and GPIO stack import."""
import importlib

MODULES = ["numpy", "cv2", "picamera2", "gpiozero", "lgpio", "smbus2", "spidev", "serial",
           "flask", "flask_sock", "flask_socketio", "fastapi", "uvicorn", "websockets",
           "scipy", "pandas", "matplotlib", "sklearn", "torch", "torchvision",
           "onnxruntime", "ai_edge_litert", "mediapipe", "ultralytics", "imutils", "simple_pid"]

for name in MODULES:
    try:
        mod = importlib.import_module(name)
        print(f"  OK   {name:16} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  FAIL {name:16} {e}")

try:
    from picamera2 import Picamera2
    print("카메라:", Picamera2.global_camera_info() or "연결된 카메라 없음")
except Exception as e:
    print("카메라 확인 실패:", e)
