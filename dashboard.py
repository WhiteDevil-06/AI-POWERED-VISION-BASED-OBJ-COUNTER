import streamlit as st
import time
import pandas as pd
import numpy as np
from datetime import datetime
import cv2
import os
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg') # Fix for thread crashes
from ultralytics import YOLO

# --- CONFIGURATION & STATE ---
def init_state():
    if 'skus' not in st.session_state:
        st.session_state['skus'] = [
            {"sku_id": "IPHONE-15-PM", "name": "iPhone 15 Pro Max", "units_per_box": 10},
            {"sku_id": "NIKE-AIR-90", "name": "Nike Air Max 90", "units_per_box": 12},
            {"sku_id": "PS5-CONSOLE", "name": "PlayStation 5 Slim", "units_per_box": 1},
        ]
    
    if 'logs' not in st.session_state:
        # Start with empty logs for production
        st.session_state['logs'] = []

# --- CENTROID TRACKER (COPIED FROM PHASE 4) ---
class CentroidTracker:
    def __init__(self, max_disappeared=80, max_distance=250):
        self.next_object_id = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # CALC DISTANCE (PURE NUMPY)
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    dx = object_centroids[i][0] - input_centroids[j][0]
                    dy = object_centroids[i][1] - input_centroids[j][1]
                    D[i, j] = np.sqrt(dx*dx + dy*dy)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                if D[row][col] > self.max_distance: continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])
        return self.objects

# --- MODEL LOADER (CACHED) ---
@st.cache_resource
def load_yolo_model():
    path = "roboflow_v2.pt" if os.path.exists("roboflow_v2.pt") else "yolov8n.pt"
    
    # FIX FOR PYTORCH 2.6+
    try:
        import torch
        _orig_load = torch.load
        def safe_load(*args, **kwargs):
            if 'weights_only' not in kwargs: kwargs['weights_only'] = False
            return _orig_load(*args, **kwargs)
        torch.load = safe_load
        
        model = YOLO(path)
        
        # Restore original
        torch.load = _orig_load
        
        # Warmup
        model(np.zeros((100,100,3), dtype='uint8'), verbose=False)
        return model, path
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, "Error"

def apply_obsidian_theme():
    st.markdown("""
    <style>
        /* MAIN APP BACKGROUND */
        .stApp {
            background-color: #0E0E10; /* Matte Black */
            color: #FFFFFF;
        }
        
        /* HIDE DEFAULT HEADER */
        header {visibility: hidden;}
        
        /* CARDS (OBSIDIAN STYLE) */
        .metric-card {
            background-color: #1C1C1E; /* Dark Charcoal */
            border: 1px solid #333333;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .metric-value {
            font-size: 36px;
            font-weight: 700;
            color: #FFFFFF;
            font-family: 'Segoe UI', sans-serif;
            letter-spacing: -1px;
            line-height: 1.2;
        }
        .metric-label {
            font-size: 12px;
            color: #A1A1AA; /* Light Grey */
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 8px;
            letter-spacing: 1px;
        }
        
        /* TABS (FULL WIDTH & CLEAN) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
            padding: 10px 0px;
            border-bottom: 1px solid #333;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            flex-grow: 1; /* Force full width */
            background-color: #1C1C1E;
            color: #A1A1AA;
            border: 1px solid #333;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }
        .stTabs [aria-selected="true"] {
            background-color: #6366F1 !important; /* Indigo */
            color: white !important;
            border-color: #6366F1 !important;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
        }
        .stTabs [data-baseweb="tab"]:hover {
            border-color: #6366F1;
            color: white;
        }
        
        /* TABLES */
        [data-testid="stDataFrame"] {
            border: 1px solid #333333;
            border-radius: 8px;
            background-color: #1C1C1E;
        }
        
        /* INPUTS & SLIDERS */
        .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
            background-color: #1C1C1E !important;
            color: white !important;
            border: 1px solid #333333 !important;
            border-radius: 8px;
            height: 45px; /* Taller inputs */
        }
        
        /* BUTTONS (STANDARDIZED) */
        .stButton > button {
            background-color: #1C1C1E;
            color: #FFFFFF;
            border: 1px solid #333333;
            border-radius: 8px;
            font-weight: 600;
            height: 45px; /* Match input height */
            transition: all 0.2s;
            width: 100%; /* Full width in container */
        }
        .stButton > button:hover {
            border-color: #6366F1;
            color: #6366F1;
            background-color: #2D2D30;
        }
        
        /* PRIMARY ACTION BUTTONS */
        button[kind="primary"] {
            background-color: #6366F1 !important;
            border: none !important;
            color: white !important;
            font-size: 15px !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }
        button[kind="primary"]:hover {
            background-color: #4F46E5 !important;
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.5);
            transform: translateY(-1px);
        }

        /* DELETE BUTTON (RED TEXT ONLY) */
        button[kind="secondary"] {
            border-color: #EF4444 !important;
            color: #EF4444 !important;
            background-color: transparent !important;
        }
        button[kind="secondary"]:hover {
            background-color: rgba(239, 68, 68, 0.1) !important;
            color: #EF4444 !important;
        }
        
        /* CUSTOM DIVIDER */
        hr {
            border-color: #333333;
            margin: 30px 0;
        }
        
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Box-Sense AI", layout="wide", page_icon="📦")
    init_state()
    apply_obsidian_theme()

    # --- TOP BAR ---
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("<h3 style='margin:0; padding:0; color:white;'>📦 BOX-SENSE <span style='color:#6366F1'>AI</span></h3>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<p style='text-align:right; color:#A1A1AA; margin:0;'>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- TABS ---
    tab_dash, tab_inv, tab_vis, tab_rep, tab_sys = st.tabs(["DASHBOARD", "INVENTORY", "VISION OPS", "REPORTS", "SYSTEM"])

    # --- TAB 1: DASHBOARD ---
    with tab_dash:
        # 1. METRICS
        total_units = sum(log['total_units'] for log in st.session_state['logs'])
        unique_skus = len(st.session_state['skus'])
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""<div class="metric-card"><div class="metric-label">TOTAL UNITS COUNTED</div><div class="metric-value">{total_units:,}</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card"><div class="metric-label">ACTIVE SKUs</div><div class="metric-value">{unique_skus}</div></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card"><div class="metric-label">SYSTEM HEALTH</div><div class="metric-value" style="color:#10B981">100%</div></div>""", unsafe_allow_html=True)

        # 2. FULL ITEM LIST (Aggregated)
        st.markdown("##### 📦 INVENTORY SUMMARY")
        
        # Aggregate Data Calculation
        inventory_summary = []
        for sku in st.session_state['skus']:
            sku_id = sku['sku_id']
            # Filter logs for this SKU
            relevant_logs = [l for l in st.session_state['logs'] if l['sku'] == sku_id]
            total_boxes = sum(l['boxes'] for l in relevant_logs)
            total_u = total_boxes * sku['units_per_box']
            
            inventory_summary.append({
                "SKU ID": sku_id,
                "PRODUCT NAME": sku['name'],
                "UNITS/BOX": sku['units_per_box'],
                "TOTAL BOXES": total_boxes,
                "TOTAL UNITS": total_u
            })
            
        df_inv = pd.DataFrame(inventory_summary)
        # Display as a clean table
        st.dataframe(
            df_inv, 
            use_container_width=True, 
            height=400,
            hide_index=True
        )

    # --- TAB 2: INVENTORY ---
    with tab_inv:
        c1, c2 = st.columns([1, 2])
        
        # LEFT: ADD SKU FORM
        with c1:
            st.markdown("""<div class="metric-card">""", unsafe_allow_html=True)
            st.markdown("##### ➕ NEW SKU")
            new_id = st.text_input("SKU ID", placeholder="e.g. A-101").upper()
            new_name = st.text_input("NAME", placeholder="e.g. Widget A")
            new_units = st.number_input("UNITS PER BOX", 1, 10000, 10)
            
            if st.button("ADD SKU", type="primary", use_container_width=True):
                if new_id and new_name:
                    # Check duplicate
                    if any(s['sku_id'] == new_id for s in st.session_state['skus']):
                        st.error("SKU ID already exists!")
                    else:
                        st.session_state['skus'].append({
                            "sku_id": new_id,
                            "name": new_name,
                            "units_per_box": new_units
                        })
                        st.success(f"Added {new_id}")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.warning("Please fill all fields.")
            st.markdown("""</div>""", unsafe_allow_html=True)

        # RIGHT: LIST & DELETE
        with c2:
            st.markdown("##### 📝 SKU DATABASE")
            
            # Show Table
            df_view = pd.DataFrame(st.session_state['skus'])
            st.dataframe(df_view, use_container_width=True, hide_index=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Delete Logic
            cd1, cd2 = st.columns([3, 1])
            with cd1:
                del_target = st.selectbox("SELECT SKU TO DELETE", [s['sku_id'] for s in st.session_state['skus']])
            with cd2:
                # Vertical align spacer (Label height is approx 28px)
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("DELETE", type="secondary", use_container_width=True):
                    st.session_state['skus'] = [s for s in st.session_state['skus'] if s['sku_id'] != del_target]
                    st.toast(f"Deleted {del_target}")
                    time.sleep(1)
                    st.rerun()

    # --- TAB 3: VISION OPS (INTEGRATED) ---
    with tab_vis:
        # Header Control Bar
        c_sel, c_exp, c_act = st.columns([2, 1, 1])
        active_sku = c_sel.selectbox("ACTIVE SKU", [s['sku_id'] for s in st.session_state['skus']], key='vis_sku')
        expected = c_exp.number_input("EXPECTED", 1, 1000, 100, key='vis_exp')
        
        # Toggle Button Logic
        if 'cam_active' not in st.session_state: st.session_state['cam_active'] = False
        
        btn_label = "⏹ STOP" if st.session_state['cam_active'] else "▶ START"
        btn_type = "secondary" if st.session_state['cam_active'] else "primary"
        
        if c_act.button(btn_label, type=btn_type, use_container_width=True):
            st.session_state['cam_active'] = not st.session_state['cam_active']
            st.rerun()

        # Visual Feed Area
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        # --- THE VISION LOGIC ---
        if st.session_state['cam_active']:
            # 1. Init Model
            model, model_path = load_yolo_model()
            if model is None:
                st.error("Model Failed to Load")
                st.stop()
            
            # 2. Init Camera
            cap = cv2.VideoCapture(0) # Default
            if not cap.isOpened() or not cap.read()[0]:
                cap = cv2.VideoCapture(1) # Backup
            
            if not cap.isOpened():
                st.error("❌ NO CAMERA DETECTED")
                st.session_state['cam_active'] = False
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # 3. Init Tracker & Vars
                tracker = CentroidTracker(max_disappeared=80, max_distance=250)
                roi_rect = (400, 150, 480, 400)
                start_time = time.time()
                system_active = False
                total_count = 0
                counted_ids = set()
                stuck_log = {}
                
                # Streamlit UI Elements
                st_frame = st.image([]) # Placeholder for video
                stop_btn = st.button("🔴 EMERGENCY STOP", key='emg_stop')

                # 4. LOOP
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Logic copied from phase4_roi.py
                    rx, ry, rw, rh = roi_rect
                    roi_color = (0, 255, 0)
                    
                    detections = []
                    results = model(frame, conf=0.5, verbose=False)
                    
                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            if cls_id in [0, 39, 41]:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                detections.append((x1, y1, x2, y2))
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 1)
                                cv2.putText(frame, f"{int(conf*100)}%", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

                    if not system_active:
                        if len(detections) > 0: system_active = True
                        elif (time.time() - start_time) > 60: system_active = True

                    if system_active:
                        objects = tracker.update(detections)
                        ids_to_kill = []
                        
                        for (obj_id, centroid) in list(objects.items()):
                            cx, cy = centroid
                            in_roi = (rx < cx < (rx + rw)) and (ry < cy < (ry + rh))
                            
                            color = (0, 255, 0) if in_roi else (0, 100, 255)
                            cv2.putText(frame, f"ID {obj_id}", (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            cv2.circle(frame, (cx, cy), 4, color, -1)

                            if in_roi:
                                if obj_id not in counted_ids:
                                    total_count += 1
                                    counted_ids.add(obj_id)
                                    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 5)
                                
                                if obj_id not in stuck_log: stuck_log[obj_id] = time.time()
                                elif (time.time() - stuck_log[obj_id]) > 5.0:
                                    msg = f"NOTIFICATION: OBJECT {obj_id} STUCK"
                                    cv2.putText(frame, msg, (rx, ry + rh + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 3)
                            else:
                                if obj_id in stuck_log: del stuck_log[obj_id]
                                if cx > (rx + rw + 50): ids_to_kill.append(obj_id)
                        
                        for k in ids_to_kill: tracker.deregister(k)

                    # HUD
                    roi_thick = 3 if system_active else 1
                    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), roi_color, roi_thick)
                    cv2.putText(frame, f"COUNT: {total_count} / {expected}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                    # Display in Streamlit (BGR -> RGB)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, channels="RGB")
                    
                    # Stop Condition
                    if stop_btn:
                        st.session_state['cam_active'] = False
                        cap.release()
                        st.rerun()

                cap.release()

        else:
            st.info("ℹ️ CAMERA OFFLINE. CLICK START TO BEGIN SESSION.")
            st.image("https://placehold.co/1280x720/0E0E10/1C1C1E.png?text=SYSTEM+READY", use_container_width=True)

    # --- TAB 4: REPORTS ---
    with tab_rep:
        c1, c2 = st.columns([3, 1])
        c1.markdown("##### 📊 SESSION HISTORY LOGS")
        
        # Export Button
        df_logs = pd.DataFrame(st.session_state['logs'])
        csv = df_logs.to_csv(index=False).encode('utf-8')
        c2.download_button(
            "📥 EXPORT CSV",
            csv,
            "box_hunter_logs.csv",
            "text/csv",
            key='download-csv',
            type="primary",
            use_container_width=True
        )
        
        st.dataframe(df_logs, use_container_width=True, hide_index=True)

    # --- TAB 5: SYSTEM ---
    with tab_sys:
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("##### 📐 ROI CALIBRATION")
            st.caption("Adjust the slider to match your conveyor belt Area of Interest.")
            
            rx = st.slider("X OFFSET", 0, 640, 400)
            ry = st.slider("Y OFFSET", 0, 480, 150)
            rw = st.slider("WIDTH", 100, 600, 480)
            rh = st.slider("HEIGHT", 100, 400, 400)
            
        with c2:
            # LIVE PREVIEW MOCKUP
            st.markdown("##### 👁️ PREVIEW")
            
            # Generate black canvas
            preview = np.zeros((720, 1280, 3), dtype=np.uint8)
            # Draw ROI
            cv2.rectangle(preview, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            # Add text
            cv2.putText(preview, f"ROI: {rw}x{rh}", (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            st.image(preview, channels="BGR", use_container_width=True)

if __name__ == "__main__":
    main()
