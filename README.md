# 🚗 Driver Drowsiness Detection System (DDDS)

A **real-time driver drowsiness detection system**, the project includes **three different implementations**, each based on a different detection method:

| Version | Library | Description |
|----------|----------|-------------|
| 🧩 **DDDS** | `OpenCV` | Uses Haar Cascades to detect eyes and faces. |
| 🧠 **DDDS_DLIB** | `Dlib` | Uses facial landmarks to track eye movement. |
| 🎯 **DDDS_MP** | `Google MediaPipe` | Uses Face Mesh for accurate and stable detection. |

---

## 📁 Project Structure

DDDS_upload/
├── DDDSthefolder/ # Implementation using Haar cascades
│ └── DDDS.py
├── DDDS_DLIBthefolder/ # Implementation using DLIB facial landmarks
│ └── DDDS_DLIB.py
└── DDDS_MPthefolder/ # Implementation using MediaPipe Face Mesh
└── DDDS_MP.py

---

## ⚙️ Setup & Installation

## 1️⃣ Clone this repository:

```bash
git clone https://github.com/EmunaS/DDDS.git
cd DDDS
```

## 2️⃣ Install the dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

## ▶️🚀 Run the Project
You can run any of the implementations directly.
 To run each version, use the following commands in your terminal:

```bash
# Run the HAAR version
python DDDSthefolder/DDDS_HAAR.py

# Run the DLIB version
python DDDS_DLIBthefolder/DDDS_DLIB.py

# Run the MediaPipe version
python DDDS_MPthefolder/DDDS_MP.py
```
## 📊Output

Each implementation opens your webcam and analyzes the driver’s eyes and face in real time.
When drowsiness is detected, the system triggers an alert.

## 🧠 About

This project was developed as part of a computer vision learning project,
exploring different real-time facial detection methods for driver safety.

## 👩‍💻 Author

**Emuna S.**  
[GitHub Profile →](https://github.com/EmunaS)

💡
- Make sure your webcam is connected and accessible.
- Make sure you have the correct Python version (3.11 recommended).
- If you are using DLIB, ensure dlib or dlib-bin is installed depending on your system.

## ⭐ Acknowledgements

This project was built using the following amazing open-source libraries:

- [OpenCV](https://opencv.org/) — Computer vision library for image and video processing.
- [Dlib](http://dlib.net/) — Toolkit for machine learning and computer vision, used here for facial landmark detection.
- [MediaPipe](https://developers.google.com/mediapipe) — Google’s framework for building perception pipelines, used here for face and eye tracking.
- [Python](https://www.python.org/) — The programming language powering the entire project.





