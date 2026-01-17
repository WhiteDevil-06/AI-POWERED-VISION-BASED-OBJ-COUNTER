# 👁️ AI Vision Box Counter

**High-Performance Object Detection & Counting System**
> Custom trained YOLOv8 model for accurate inventory management and logistics.

---

## � Installation & Usage

You can choose between the **Automatic Shortcut** (Recommended) or the **Manual Setup**.

### ⚡ Method 1: The Shortcut (Automatic)
*Best for first-time setup or quick launching (Windows & Mac).*

**For Windows Users 🪟:**
1.  Open the folder.
2.  Double-click **`setup_and_run.bat`**.
3.  *That's it! It automatically installs Python libraries and launches the App.*

**For Mac / Linux Users 🍎:**
1.  Open Terminal.
2.  Run: `bash setup_and_run.sh`

---

### 🛠️ Method 2: Manual Installation
*For developers who prefer full control.*

**1. Create a Virtual Environment:**
```bash
python -m venv env
```

**2. Activate Environment:**
*   **Windows:** `.\env\Scripts\activate`
*   **Mac/Linux:** `source env/bin/activate`

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the Dashboard:**
```bash
streamlit run dashboard.py
```

---

## ✨ Key Features

*   **⚡ Real-Time Detection**: Powered by Ultralytics YOLOv8 (Custom `roboflow_v2.pt` model).
*   **🎯 Precise Counting**: Advanced Centroid Tracking with ROI filtering.
*   **💾 Data Persistence**: Automatically saves Inventory (`skus.json`) and Logs (`logs.json`).
*   **🛡️ Robust Camera System**: Auto-detects external webcams and recovers from disconnection.
*   **📊 Batch Management**:
    *   **Auto-Save**: Logs data automatically when the target count is reached.
    *   **Fail-Safe**: "Session Stopped" modal allows saving partial batches.

## 📂 Project Structure

*   `dashboard.py`: **Main Application** (Streamlit Interface).
*   `setup_and_run.bat`: **Windows Auto-Launcher**.
*   `setup_and_run.sh`: **Mac Auto-Launcher**.
*   `requirements.txt`: Python dependencies.
*   `roboflow_v2.pt`: Custom trained model weights.
*   `phase4_roi.py`: Vision logic helper.
