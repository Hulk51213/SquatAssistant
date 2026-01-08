import sys
import time
import math
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtMultimedia import QSoundEffect


# -------------------- Math Utilities --------------------

def angle(a, b, c):
    a, b, c = map(lambda x: np.array(x, dtype=np.float32), (a, b, c))
    ba, bc = a - b, c - b
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return 180.0
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))


def torso_angle(mid_sh, mid_hip):
    v = np.array(mid_sh) - np.array(mid_hip)
    v = v / (np.linalg.norm(v) + 1e-6)
    vertical = np.array([0.0, -1.0])
    dot = np.dot(v, vertical)
    return math.degrees(math.acos(np.clip(dot, -1.0, 1.0)))


class Smoother:
    def __init__(self, n=6):
        self.buf = deque(maxlen=n)

    def update(self, v):
        self.buf.append(v)
        return sum(self.buf) / len(self.buf)


class DecisionSmoother:
    def __init__(self, n=7):
        self.buf = deque(maxlen=n)

    def update(self, v):
        if v is not None:
            self.buf.append(v)
        if self.buf.count(True) >= 5:
            return True
        if self.buf.count(False) >= 5:
            return False
        return None


# -------------------- Video Thread --------------------

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

        self.reps = 0
        self.state = "up"

        self.pose = mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.knee_s = Smoother()
        self.torso_s = Smoother()
        self.form_smoother = DecisionSmoother()

    def start_workout(self):
        self.reps = 0
        self.state = "up"
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
            raw_form = None

            if self.countdown >= 0:
                if time.time() - self.last_tick >= 1:
                    self.countdown -= 1
                    self.last_tick = time.time()

                txt = "GO" if self.countdown == 0 else str(self.countdown)
                cv2.putText(frame, txt, (w//2 - 80, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 8)

                if self.countdown == 0:
                    self.active = True
                    self.countdown = -1

            if self.blur:
                frame = cv2.GaussianBlur(frame, (31,31), 0)

            if self.active:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.pose.process(rgb)

                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark

                    if min(lm[25].visibility, lm[26].visibility,
                           lm[27].visibility, lm[28].visibility) >= 0.5:

                        p = lambda i: (int(lm[i].x * w), int(lm[i].y * h))
                        lh, lk, la = p(23), p(25), p(27)
                        rh, rk, ra = p(24), p(26), p(28)
                        ls, rs = p(11), p(12)

                        mid_hip = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
                        mid_sh = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)

                        knee_angle = min(
                            angle(lh, lk, la),
                            angle(rh, rk, ra)
                        )
                        knee_angle = self.knee_s.update(knee_angle)
                        torso = self.torso_s.update(torso_angle(mid_sh, mid_hip))

                        if knee_angle < 120:
                            depth_ok = knee_angle <= 100
                            hip_ok = mid_hip[1] > max(lk[1], rk[1])
                            torso_ok = torso <= 30
                            raw_form = depth_ok and hip_ok and torso_ok

                            color = (0,230,118) if raw_form else (255,23,68)
                            for a,b in [(lh,lk),(lk,la),(rh,rk),(rk,ra),(mid_sh,mid_hip)]:
                                cv2.line(frame, a, b, color, 6)

                        if self.state == "up" and knee_angle < 95:
                            self.state = "down"
                        elif self.state == "down" and knee_angle > 160:
                            if raw_form:
                                self.reps += 1
                            self.state = "up"

            final_form = self.form_smoother.update(raw_form)
            self.status_signal.emit(final_form, self.reps)

            img = QtGui.QImage(frame.data, w, h, 3*w, QtGui.QImage.Format_BGR888)
            self.frame_signal.emit(img)
            QtCore.QThread.msleep(1)

        cap.release()
        self.pose.close()

    def close(self):
        self.running = False
        self.wait()


# -------------------- GUI --------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Squat Assistant")
        self.showFullScreen()

        self.click_sound = QSoundEffect()
        self.click_sound.setSource(QtCore.QUrl.fromLocalFile("click.wav"))
        self.click_sound.setVolume(0.25)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(24)

        self.video = QtWidgets.QLabel()
        self.video.setFixedSize(850,650)
        self.video.setStyleSheet("background:black; border-radius:18px;")
        layout.addWidget(self.video)

        panel = QtWidgets.QFrame()
        panel.setFixedWidth(360)
        panel.setStyleSheet("""
            background: rgba(255,255,255,0.18);
            border-radius: 28px;
            border: 1px solid rgba(255,255,255,0.25);
        """)
        p = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(panel)

        self.reps = QtWidgets.QLabel("0")
        self.reps.setAlignment(QtCore.Qt.AlignCenter)
        self.reps.setStyleSheet("color:white; font-size:120px; font-weight:800;")
        p.addWidget(self.reps)

        self.badge = QtWidgets.QLabel("READY")
        self.badge.setAlignment(QtCore.Qt.AlignCenter)
        self.badge.setFixedHeight(78)
        self.badge.setStyleSheet("background:#2979FF; color:white; font-size:28px; border-radius:39px;")
        p.addWidget(self.badge)

        self.start = QtWidgets.QPushButton("START")
        self.start.setFixedHeight(78)
        self.start.setStyleSheet("background:#00E676; font-size:26px; font-weight:900; border-radius:22px;")
        p.addWidget(self.start)

        self.stop = QtWidgets.QPushButton("STOP")
        self.stop.setFixedHeight(64)
        self.stop.setStyleSheet("background:#FF1744; color:white; font-size:22px; font-weight:800; border-radius:18px;")
        p.addWidget(self.stop)

        self.thread = VideoThread()
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.status_signal.connect(self.update_status)
        self.thread.start()

        self.start.clicked.connect(lambda: (self.click_sound.play(), self.thread.start_workout()))
        self.stop.clicked.connect(lambda: (self.click_sound.play(), self.thread.stop_workout()))

    def update_frame(self, img):
        self.video.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def update_status(self, good, reps):
        self.reps.setText(str(reps))
        if good is None:
            self.badge.setText("READY")
        elif good:
            self.badge.setText("GOOD FORM")
            self.badge.setStyleSheet("background:#00E676; color:#003300; font-size:28px; border-radius:39px;")
        else:
            self.badge.setText("BAD FORM")
            self.badge.setStyleSheet("background:#FF1744; color:white; font-size:28px; border-radius:39px;")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def closeEvent(self, e):
        self.thread.close()
        e.accept()


# -------------------- ENTRY --------------------

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    sys.exit(app.exec())
