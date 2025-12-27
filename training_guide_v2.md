# 🎓 Step-by-Step Training Guide (Zero to Hero)

## STEP 1: Get the Data (Roboflow)
1.  **Go to this URL**: [Roboflow Universe - Box Dataset](https://universe.roboflow.com/yogi-tri-a/box-carton-package-detection)
    *   *Note: You can search "Cardboard Box" if you want a different one, but this one is good.*
2.  Click the big button: **Download Dataset**.
3.  A popup appears. Select Format: **YOLOv8**.
4.  Select **"Show Download Code"** (Do NOT download the zip file to your laptop).
5.  **COPY** the code that looks like this:
    ```python
    !pip install roboflow
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_KEY_HERE")
    project = rf.workspace(...).project(...)
    version = project.version(1)
    dataset = version.download("yolov8")
    ```
6.  Keep this code safe.

## STEP 2: Google Colab Setup
1.  **Open this Link**: [Google Colab](https://colab.research.google.com/)
2.  Click **File -> Upload Notebook**.
3.  Upload the `colab_training.ipynb` file I put on your Desktop.
4.  **IMPORTANT:** Go to top menu: **Runtime -> Change runtime type** -> Select **T4 GPU** -> Save.
    *   *If you don't do this, it will be slow.*

## STEP 3: Run the Training
1.  **Cell 1**: Click the "Play" button (Installs YOLO). Wait for green checkmark.
2.  **Cell 2**: Paste your **Roboflow Code** (from Step 1) into this cell where it says `# PASTE YOUR ROBOFLOW CODE HERE`. Click Play.
3.  **Cell 3**: Click Play. This starts the training.
    *   You will see a progress bar for "Epoch 1/50", "Epoch 2/50"...
    *   It typically takes **10-15 minutes**.

## STEP 4: Download Your "Brain"
1.  **Cell 4**: Click Play.
2.  It should automatically download a file named `best.pt`.
3.  **Move this file** to your project folder: `C:\Users\raksh\OneDrive\Desktop\AI-VISION`.

## STEP 5: Integrate
Once you have `best.pt`:
1.  Tell me "I have the file".
2.  I will update the code to use:
    ```python
    model = YOLO("best.pt")
    ```
3.  We run Phase 4 again, and it will be **perfectly stable** because it only knows "Boxes".
