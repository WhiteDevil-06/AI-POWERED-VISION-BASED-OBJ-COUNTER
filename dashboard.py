import streamlit as st
import time
import pandas as pd
import numpy as np
from datetime import datetime
import cv2
import os
import json
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
from ultralytics import YOLO

CONFIG_FILE = "vision_config.json"

# --- CONFIGURATION & PERSISTENCE ---
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f: return json.load(f)
        except: pass
    return {"rx": 400, "ry": 150, "rw": 480, "rh": 400}

def save_config(rx, ry, rw, rh):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({"rx": rx, "ry": ry, "rw": rw, "rh": rh}, f)
        return True
    except:
        return False

SKU_FILE = "skus.json"
LOG_FILE = "logs.json"

def load_data():
    skus = [
        {"sku_id": "IPHONE-15-PM", "name": "iPhone 15 Pro Max", "units_per_box": 10},
        {"sku_id": "NIKE-AIR-90", "name": "Nike Air Max 90", "units_per_box": 12},
        {"sku_id": "PS5-CONSOLE", "name": "PlayStation 5 Slim", "units_per_box": 1},
    ]
    logs = []
    
    if os.path.exists(SKU_FILE):
        try: 
            with open(SKU_FILE, 'r') as f: skus = json.load(f)
        except: pass
        
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f: logs = json.load(f)
        except: pass
        
    return skus, logs

def save_data():
    try:
        with open(SKU_FILE, 'w') as f: json.dump(st.session_state['skus'], f)
        with open(LOG_FILE, 'w') as f: json.dump(st.session_state['logs'], f)
    except: pass

def init_state():
    if 'skus' not in st.session_state or 'logs' not in st.session_state:
        s, l = load_data()
        st.session_state['skus'] = s
        st.session_state['logs'] = l
    
    # Load Roi Config
    if 'roi_cfg' not in st.session_state:
        st.session_state['roi_cfg'] = load_config()

# --- CENTROID TRACKER (TUNED 7.5) ---
class CentroidTracker:
    def __init__(self, max_disappeared=80, max_distance=300):
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

# --- MODEL LOADER (OPTIMIZED) ---
@st.cache_resource
def load_yolo_model():
    path = "roboflow_v2.pt" if os.path.exists("roboflow_v2.pt") else "yolov8n.pt"
    try:
        import torch
        _orig_load = torch.load
        def safe_load(*args, **kwargs):
            if 'weights_only' not in kwargs: kwargs['weights_only'] = False
            return _orig_load(*args, **kwargs)
        torch.load = safe_load
        
        model = YOLO(path)
        
        torch.load = _orig_load
        # LIGHT WARMUP (1x1 px) - Faster
        model(np.zeros((32,32,3), dtype='uint8'), verbose=False)
        return model, path
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, "Error"

# --- THEME (RESTORED FROM PHASE 7) ---
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
        
        inventory_summary = []
        for sku in st.session_state['skus']:
            sku_id = sku['sku_id']
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
        st.dataframe(df_inv, use_container_width=True, height=400, hide_index=True)

    # --- TAB 2: INVENTORY (RESTORED DELETE UI) ---
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
                    if any(s['sku_id'] == new_id for s in st.session_state['skus']):
                        st.error("SKU ID already exists!")
                    else:
                        st.session_state['skus'].append({
                            "sku_id": new_id,
                            "name": new_name,
                            "units_per_box": new_units
                        })
                        save_data()
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
            
            # Delete Logic (RESTORED)
            cd1, cd2 = st.columns([3, 1])
            with cd1:
                del_target = st.selectbox("SELECT SKU TO DELETE", [s['sku_id'] for s in st.session_state['skus']])
            with cd2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("DELETE", type="secondary", use_container_width=True):
                    st.session_state['skus'] = [s for s in st.session_state['skus'] if s['sku_id'] != del_target]
                    save_data()
                    st.toast(f"Deleted {del_target}")
                    time.sleep(1)
                    st.rerun()

    # --- TAB 3: VISION OPS ---
    with tab_vis:
        c_sel, c_exp, c_act = st.columns([2, 1, 1])
        active_sku_id = c_sel.selectbox("ACTIVE SKU", [s['sku_id'] for s in st.session_state['skus']], key='vis_sku')
        expected = c_exp.number_input("EXPECTED", 1, 1000, 100, key='vis_exp')
        
        if 'cam_active' not in st.session_state: st.session_state['cam_active'] = False
        
        if 'current_count' not in st.session_state: st.session_state['current_count'] = 0
        
        btn_label = "⏹ STOP & SAVE" if st.session_state['cam_active'] else "▶ START SESSION"
        btn_type = "secondary" if st.session_state['cam_active'] else "primary"
        
        c_act.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if c_act.button(btn_label, type=btn_type, use_container_width=True):
            if st.session_state['cam_active']:
                # STOPPING -> Trigger Save Modal
                st.session_state['pending_save'] = {
                    'count': st.session_state.get('current_count', 0), 
                    'sku': active_sku_id, 
                    'reason': 'USER STOPPED'
                }
                st.session_state['cam_active'] = False
            else:
                # STARTING
                st.session_state['cam_active'] = True
                st.session_state['current_count'] = 0 # Reset
            st.rerun()

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        if st.session_state['cam_active']:
            # ROBUST CAMERA SEARCH
            def get_camera():
                # Order: Index 1 (External - DSHOW/Default), then Index 0 (Internal)
                attempts = [
                    (1, cv2.CAP_DSHOW, "Index 1 (Ext - DirectShow)"),
                    (1, cv2.CAP_ANY,   "Index 1 (Ext - Default)"),
                    (0, cv2.CAP_DSHOW, "Index 0 (Int - DirectShow)"),
                    (0, cv2.CAP_ANY,   "Index 0 (Int - Default)"),
                ]
                
                for idx, backend, name in attempts:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        # Verify we can actually read
                        ret, _ = cap.read()
                        if ret:
                            return cap, name
                        cap.release()
                return None, None

            with st.spinner("STARTING VISION SYSTEM... (Searching Cameras)"):
                model, _ = load_yolo_model()
                cap, cam_name = get_camera()
                
            if cap is None:
                st.error("❌ CRTICAL ERROR: NO WORKING CAMERA FOUND.")
                st.info("Please unplug/replug your webcam and restart the app.")
                st.session_state['cam_active'] = False
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # LOAD CONFIG
                cfg = st.session_state['roi_cfg']
                roi_rect = (cfg['rx'], cfg['ry'], cfg['rw'], cfg['rh'])
                
                tracker = CentroidTracker(max_disappeared=80, max_distance=300)
                total_count = st.session_state.get('current_count', 0)
                st_frame = st.image([])
                # REMOVED INNER STOP BUTTON
                
                counted_ids = set()
                stuck_log = {}
                start_ts = time.time()
                system_active = False

                # ... (Inside Camera Loop)
                while True:
                    try:
                        ret, frame = cap.read()
                        if not ret: 
                            st.warning("Lost Camera Feed. Retrying...")
                            time.sleep(0.1)
                            continue
                    except cv2.error:
                        st.error("Camera Hardware Error. Please Restart.")
                        break
                    
                    rx, ry, rw, rh = roi_rect
                    detections = []
                    
                    # 1. DETECTION (NO FILTER - TRUST CUSTOM MODEL)
                    results = model(frame, conf=0.5, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            # REMOVED: if int(box.cls[0]) in [0, 39, 41]: 
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            detections.append((x1, y1, x2, y2))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 3)
                            cv2.putText(frame, f"{int(float(box.conf[0])*100)}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

                    if not system_active:
                        if len(detections) > 0 or (time.time()-start_ts > 60): system_active = True

                    if system_active:
                        objects = tracker.update(detections)
                        ids_kill = []
                        
                        for (oid, cent) in list(objects.items()):
                            cx, cy = cent
                            in_roi = (rx < cx < rx+rw) and (ry < cy < ry+rh)
                            col = (0, 255, 0) if in_roi else (0, 100, 255)
                            cv2.putText(frame, f"ID {oid}", (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
                            cv2.circle(frame, (cx, cy), 8, col, -1)

                            if in_roi:
                                if oid not in counted_ids:
                                    total_count += 1
                                    st.session_state['current_count'] = total_count # SYNC STATE
                                    counted_ids.add(oid)
                                    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 6) # FLASH ROI
                                
                                if oid not in stuck_log: stuck_log[oid] = time.time()
                                elif (time.time() - stuck_log[oid]) > 5.0:
                                    cv2.putText(frame, f"NOTIFICATION: OBJECT {oid} STUCK", (rx, ry+rh+40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 4)
                            else:
                                if oid in stuck_log: del stuck_log[oid]
                                if cx > (rx+rw+50): ids_kill.append(oid)
                        
                        for k in ids_kill: tracker.deregister(k)

                    # HUD
                    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 3)
                    cv2.putText(frame, f"COUNT: {total_count} / {expected}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
                    
                    ts = datetime.now().strftime("%H:%M:%S")
                    cv2.putText(frame, ts, (1050, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                    
                    model_display_name = "MODEL: " + os.path.basename(getattr(model, 'active_path', 'roboflow_v2.pt'))
                    cv2.putText(frame, model_display_name, (30, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

                    st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    
                    # 3. AUTO STOP (Target Reached)
                    if total_count >= expected:
                        u_per_box = next((s['units_per_box'] for s in st.session_state['skus'] if s['sku_id'] == active_sku_id), 1)
                        st.session_state['logs'].append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "batch_id": f"#{len(st.session_state['logs'])+1000}",
                            "sku": active_sku_id,
                            "boxes": total_count,
                            "total_units": total_count * u_per_box
                        })
                        save_data()
                        st.session_state['batch_success_msg'] = f"{total_count} BOXES COMPLETED ({active_sku_id})"
                        st.session_state['cam_active'] = False
                        break
                    
                    # FORCE SYNC STATE (CRITICAL FIX FOR ZERO COUNT)
                    st.session_state['current_count'] = total_count

                cap.release()
                st.rerun()

        # --- SUCCESS MESSAGE (Target Reached) - SMALLER & CLEANER ---
        if 'batch_success_msg' in st.session_state:
            if 'pending_save' in st.session_state: del st.session_state['pending_save']
            
            # Simple Green Notification
            st.success(f"✅ {st.session_state['batch_success_msg']}") 
            if st.button("START NEW BATCH", type="primary", use_container_width=True):
                del st.session_state['batch_success_msg']
                st.rerun()
        
        # --- SAVE/DISCARD DIALOG (Post-Loop) ---
        elif 'pending_save' in st.session_state:
            data = st.session_state['pending_save']
            
            # Since Manual Stop is the ONLY way to get here now:
            header_text = "⏹ SESSION STOPPED"
            header_color = "#6366F1" # Indigo (Matches App Theme)
            box_border = "#6366F1"
            
            # Premium Modal UI
            # Premium Modal UI
            html_content = f"""
            <div style="background: linear-gradient(135deg, #1C1C1E 0%, #2C2C2E 100%); padding: 40px; border-radius: 16px; border: 2px solid {box_border}; box-shadow: 0 0 40px rgba(99, 102, 241, 0.2); text-align: center; max-width: 600px; margin: 0 auto; margin-bottom: 30px;">
                <div style="font-size: 20px; font-weight: 700; color: {header_color}; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 15px;">
                    {header_text}
                </div>
                <div style="font-size: 90px; font-weight: 800; color: white; line-height: 1; text-shadow: 0 0 30px {header_color}44; margin-bottom: 10px;">
                    {data['count']}
                </div>
                <div style="font-size: 20px; color: #E5E7EB; margin-bottom: 40px;">
                    BOXES OF <span style="color: {header_color}; font-weight:bold;">{data['sku']}</span>
                </div>
                <p style="color: #6B7280; font-size: 14px;">Confirm your count. Save to database or discard.</p>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Centered Buttons
            c_spacer_l, c_save, c_discard, c_spacer_r = st.columns([1, 2, 2, 1])
            
            if c_save.button("✅ SAVE & LOG", type="primary", use_container_width=True):
                if data['count'] > 0:
                    u_per_box = next((s['units_per_box'] for s in st.session_state['skus'] if s['sku_id'] == data['sku']), 1)
                    st.session_state['logs'].append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "batch_id": f"#{len(st.session_state['logs'])+1000}",
                        "sku": data['sku'],
                        "boxes": data['count'],
                        "total_units": data['count'] * u_per_box
                    })
                    save_data()
                    st.toast("Batch Saved Successfully!")
                del st.session_state['pending_save']
                time.sleep(1)
                st.rerun()
                
            if c_discard.button("❌ DISCARD DATA", type="secondary", use_container_width=True):
                del st.session_state['pending_save']
                st.toast("Batch Discarded")
                time.sleep(1)
                st.rerun()
        
        elif not st.session_state['cam_active']:
            st.info("ℹ️ CAMERA OFFLINE. CLICK START TO BEGIN SESSION.")
            st.image("https://placehold.co/1280x720/0E0E10/1C1C1E.png?text=SYSTEM+READY", use_container_width=True)

    # --- TAB 4: REPORTS ---
    with tab_rep:
        c1, c2 = st.columns([3, 1])
        c1.markdown("##### 📊 SESSION HISTORY LOGS")
        df_logs = pd.DataFrame(st.session_state['logs'])
        csv = df_logs.to_csv(index=False).encode('utf-8')
        c2.download_button("📥 EXPORT CSV", csv, "logs.csv", "text/csv", type="primary", use_container_width=True)
        st.dataframe(df_logs, use_container_width=True, hide_index=True)

    # --- TAB 5: SYSTEM ---
    with tab_sys:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("##### 📐 ROI CALIBRATION")
            cfg = st.session_state['roi_cfg']
            rx = st.slider("X", 0, 1280, cfg['rx'])
            ry = st.slider("Y", 0, 720, cfg['ry'])
            rw = st.slider("W", 50, 800, cfg['rw'])
            rh = st.slider("H", 50, 600, cfg['rh'])
            
            if st.button("💾 SAVE CONFIGURATION", type="primary"):
                if save_config(rx, ry, rw, rh):
                    st.session_state['roi_cfg'] = {"rx": rx, "ry": ry, "rw": rw, "rh": rh}
                    st.success("Configuration Saved!")
                    
        with c2:
            st.markdown("##### 👁️ PREVIEW")
            prev = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.rectangle(prev, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            cv2.putText(prev, "ROI PREVIEW", (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            st.image(prev, channels="BGR", use_container_width=True)

if __name__ == "__main__":
    main()
