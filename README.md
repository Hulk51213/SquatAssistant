# RepRight 🏋️‍♂️  
AI-Powered Exercise Form & Rep Detection using Google MediaPipe

---

## 🚀 Overview
**RepRight** is a real-time computer vision fitness assistant that analyzes squat form and counts repetitions using a standard webcam.  
It provides instant visual and audio feedback to help users maintain proper form and reduce injury risk — all without wearable devices.

---

## ✨ Key Features
- ✅ Real-time squat rep counting  
- ✅ Full-body pose detection  
- ✅ Spine alignment & posture validation  
- ✅ Form stability smoothing (no flicker)  
- ✅ Desktop UI with instant feedback  
- ✅ Runs fully offline (privacy-friendly)

---

## 🧠 How It Works
RepRight uses **Google MediaPipe Pose** to extract 33 human body landmarks per frame.

From these landmarks, the system:
- Computes knee joint angles to detect squat depth
- Analyzes torso (spine) angle for posture correctness
- Applies temporal smoothing to stabilize predictions
- Uses a state machine to reliably count full reps

---

## 🛠️ Tech Stack
- **Google MediaPipe Pose** – real-time pose estimation  
- **Python** – core logic  
- **OpenCV** – video processing  
- **NumPy** – mathematical analysis  
- **PySide6 (Qt)** – desktop UI  
- **PyInstaller** – Windows executable packaging  

## 📦 Installation (Source)
```bash
git clone https://github.com/Hulk51213/SquatAssistant
