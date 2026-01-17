# 📦 Box-Sense AI: Intelligent Vision Counter

**Box-Sense AI** is a real-time computer vision application designed to automate inventory tracking on conveyor belts. It uses a custom-trained YOLOv8 model to detect, track, and count moving objects (SKUs) with high precision, providing live analytics via a modern Streamlit dashboard.

---

## ✨ Key Features

*   **Real-Time Detection**: Instantly detects objects entering the camera frame.
*   **"River Flow" Tracking**: Smart logic that assigns unique IDs to objects and counts them only when they pass through a defined central point (ROI).
*   **Stuck Object Alert**: Automatically notifies if an object remains stationary in the scanning zone for >5 seconds.
*   **Live Dashboard**: Monitor counts, system health, and SKU inventory in a beautiful "Obsidian" themed interface.
*   **ROI Calibration**: Adjustable "Region of Interest" to mask out background noise and focus on the conveyor belt.
*   **Batch Reporting**: Export session logs to CSV for inventory management.

---

## 🛠️ Tech Stack

*   **Language**: Python 3.12+
*   **Vision Core**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [OpenCV](https://opencv.org/)
*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Data Processing**: NumPy, Pandas
*   **Hardware Support**: Webcam / USB Cameras

---

## 🧠 Model Training (Google Colab)

The heart of this system is a custom **YOLOv8 Nano** model (`roboflow_v2.pt`).

*   **Dataset**: Custom labeled dataset hosted on [Roboflow](https://roboflow.com/).
*   **Training Environment**: Trained using **Google Colab** (T4 GPU) for accelerated performance.
*   **Process**:
    1.  Dataset augmentation (flip, rotate, noise) in Roboflow.
    2.  Exported to YOLOv8 format.
    3.  Fine-tuned for 20 epochs on Colab.
    4.  Weights exported for local inference.

---

## 💻 Installation Guide

### Prerequisites
*   [Python 3.10 or higher](https://www.python.org/downloads/)
*   [Git](https://git-scm.com/)

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd AI-VISION
```

### 2. Set Up Virtual Environment

#### 🪟 Windows
```powershell
# Create Environment
python -m venv env

# Activate
.\env\Scripts\activate
```

#### 🍎 Mac / 🐧 Linux
```bash
# Create Environment
python3 -m venv env

# Activate
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note**: If you have a dedicated NVIDIA GPU, ensure you install the CUDA version of PyTorch for faster inference.

---

## 🚀 How to Run

### 🪟 Windows (One-Click)
Simply double-click the **`run_app.bat`** file on your desktop/folder.

OR via Terminal:
```powershell
streamlit run dashboard.py
```

### 🍎 Mac / Linux
Open your terminal and run:
```bash
streamlit run dashboard.py
```

---

## 📖 User Guide

1.  **Dashboard**: View live metrics (Total Units, Active SKUs).
2.  **Inventory**: Add or remove Product IDs (SKUs) that you want to track.
3.  **Vision Ops**: 
    *   Select the SKU you are running.
    *   Click **Start** to open the camera feed.
    *   Watch as boxes are counted automatically!
4.  **System**: Use the sliders to adjust the Green Box (ROI) to match your camera angle.

---

## 🤝 Credits
Developed by **Rakshith R ** | Powered by **YOLOv8** & **Streamlit**
