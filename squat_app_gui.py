"""
Squat Assistant – Stable & Accurate Version
Fixes:
- Rep counter reliability
- Landmark visibility checks
- Spine (torso) detection
- Frame flicker
- Sound in EXE
"""

import sys
import os
import time
import math
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from PySide6 import QtCore, QtGui, QtWidgets

# -------------------- RESOURCE PATH (EXE SAFE) --------------------

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # PyInstaller
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# -------------------- SOUND --------------------

if sys.platform.startswith("win"):
    import winsound

def play_click():
    try:
        winsound.PlaySound(
            resource_path("click.wav"),
            winsound.SND_FILENAME | winsound.SND_ASYNC
        )
    except Exception:
        pass

# -------------------- MATH UTILITIES --------------------

def angle(a, b, c):
    a, b, c = map(lambda x: np.array(x, dtype=np.float32), (a, b, c))
    ba, bc = a - b, c - b
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return 180.0
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))

class MedianSmoother:
    def __init__(self, size=7):
        self.buf = deque(maxlen=size)

    def update(self, v):
        self.buf.append(v)
        return sorted(self.buf)[len(self.buf)//2]

# -------------------- REP COUNTER --------------------

class RepCounter:
    def __init__(self):
        self.state = "up"
        self.count = 0
        self.hold = 0

    def update(self, knee_angle):
        DOWN_TH = 100
        UP_TH = 150
        HOLD_FRAMES = 3

        if self.state == "up":
            if knee_angle < DOWN_TH:
                self.hold += 1
                if self.hold >= HOLD_FRAMES:
                    self.state = "down"
                    self.hold = 0
            else:
                self.hold = 0

        elif self.state == "down":
            if knee_angle > UP_TH:
                self.hold += 1
                if self.hold >= HOLD_FRAMES:
                    self.count += 1
                    self.state = "up"
                    self.hold = 0
            else:
                self.hold = 0

# -------------------- VIDEO THREAD --------------------

class VideoThread(QtCore.QThread):
    frame_signal = QtCore.Signal(QtGui.QImage)
    status_signal = QtCore.Signal(object, int)

    def __init__(self):
        super().__init__()
        self.running = True
        self.active = False
        self.blur = True
        self.countdown = -1
        self.last_tick = time.time()

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.4
        )

        self.knee_s = MedianSmoother()
        self.spine_s = MedianSmoother()
        self.counter = RepCounter()

    def start_workout(self):
        self.counter = RepCounter()
        self.countdown = 3
        self.active = False
        self.blur = False
        self.last_tick = time.time()

    def stop_workout(self):
        self.active = False
        self.blur = True

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            good = None

            # -------- COUNTDOWN --------
            if self.countdown >= 0:
                if time.time() - self.last_tick >= 1:
                    self.countdown -= 1
                    self.last_tick = time.time()
                text = "GO" if self.countdown == 0 else str(self.countdown)
                cv2.putText(frame, text, (w//2-80, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 8)
                if self.countdown == 0:
                    self.active = True
                    self.countdown = -1

            if self.blur:
                frame = cv2.GaussianBlur(frame, (31,31), 0)

            # -------- POSE --------
            if self.active:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.pose.process(rgb)

                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    px = lambda i: (int(lm[i].x*w), int(lm[i].y*h))

                    vis = lambda i: lm[i].visibility > 0.45

                    needed = [23,24,25,26,27,28,11,12]
                    if sum(vis(i) for i in needed) >= 4:
                        lh, rh = px(23), px(24)
                        lk, rk = px(25), px(26)
                        la, ra = px(27), px(28)
                        ls, rs = px(11), px(12)

                        knee = self.knee_s.update(min(
                            angle(lh, lk, la),
                            angle(rh, rk, ra)
                        ))

                        mid_hip = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
                        mid_sh = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)

                        spine = self.spine_s.update(
                            angle(mid_sh, mid_hip, (mid_hip[0], mid_hip[1]-100))
                        )

                        self.counter.update(knee)

                        good = knee <= 100 and spine <= 25
                        color = (0,255,0) if good else (0,0,255)

                        for a,b in [(lh,lk),(lk,la),(rh,rk),(rk,ra),(mid_sh,mid_hip)]:
                            cv2.line(frame, a, b, color, 5)

            self.status_signal.emit(good, self.counter.count)

            img = QtGui.QImage(frame.data, w, h, 3*w, QtGui.QImage.Format_BGR888)
            self.frame_signal.emit(img)

            QtCore.QThread.msleep(30)  # stable FPS

        cap.release()

    def close(self):
        self.running = False
        self.wait()

# -------------------- GUI --------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Squat Assistant")
        self.resize(1300, 750)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        self.video = QtWidgets.QLabel()
        self.video.setFixedSize(850,650)
        self.video.setStyleSheet("background:black; border-radius:18px;")
        layout.addWidget(self.video)

        panel = QtWidgets.QFrame()
        panel.setFixedWidth(360)
        panel.setStyleSheet("background:rgba(30,30,30,0.92); border-radius:28px;")
        p = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(panel)

        self.reps = QtWidgets.QLabel("0")
        self.reps.setAlignment(QtCore.Qt.AlignCenter)
        self.reps.setStyleSheet("color:white; font-size:120px; font-weight:800;")
        p.addWidget(self.reps)

        self.badge = QtWidgets.QLabel("READY")
        self.badge.setAlignment(QtCore.Qt.AlignCenter)
        self.badge.setFixedHeight(80)
        self.badge.setStyleSheet("background:#2979FF; color:white; font-size:28px; border-radius:40px;")
        p.addWidget(self.badge)

        self.start = QtWidgets.QPushButton("START")
        self.stop = QtWidgets.QPushButton("STOP")
        self.start.setStyleSheet("background:#00E676; font-size:22px; padding:18px; font-weight:800;")
        self.stop.setStyleSheet("background:#FF1744; color:white; font-size:18px; padding:16px; font-weight:700;")
        p.addWidget(self.start)
        p.addWidget(self.stop)

        self.thread = VideoThread()
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.status_signal.connect(self.update_status)
        self.thread.start()

        self.start.clicked.connect(lambda: (play_click(), self.thread.start_workout()))
        self.stop.clicked.connect(lambda: (play_click(), self.thread.stop_workout()))

    def update_frame(self, img):
        self.video.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def update_status(self, good, reps):
        self.reps.setText(str(reps))
        if good is None:
            self.badge.setText("READY")
            self.badge.setStyleSheet("background:#2979FF; color:white; font-size:28px;")
        elif good:
            self.badge.setText("GOOD FORM")
            self.badge.setStyleSheet("background:#00E676; color:#003300; font-size:28px;")
        else:
            self.badge.setText("BAD FORM")
            self.badge.setStyleSheet("background:#FF1744; color:white; font-size:28px;")

    def closeEvent(self, e):
        self.thread.close()
        e.accept()

# -------------------- ENTRY --------------------

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
