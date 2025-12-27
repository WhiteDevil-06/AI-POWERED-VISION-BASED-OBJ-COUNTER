import streamlit as st
import time
import pandas as pd
import numpy as np
from datetime import datetime
import cv2

# --- CONFIGURATION & STATE ---
def init_state():
    if 'skus' not in st.session_state:
        st.session_state['skus'] = [
            {"sku_id": "IPHONE-15-PM", "name": "iPhone 15 Pro Max", "units_per_box": 10},
            {"sku_id": "NIKE-AIR-90", "name": "Nike Air Max 90", "units_per_box": 12},
            {"sku_id": "PS5-CONSOLE", "name": "PlayStation 5 Slim", "units_per_box": 1},
        ]
    
    if 'logs' not in st.session_state:
        # Dummy logs for demo
        st.session_state['logs'] = [
            {"time": "10:15:22", "batch_id": "#8821", "sku": "NIKE-AIR-90", "boxes": 50, "total_units": 600},
            {"time": "11:05:00", "batch_id": "#8822", "sku": "IPHONE-15-PM", "boxes": 20, "total_units": 200},
        ]

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

    # --- TAB 3: VISION OPS ---
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
        
        if st.session_state['cam_active']:
            st.warning("🔴 LIVE FEED SIMULATION (CONNECTING TO YOLO MODEL...)")
            # Placeholder for camera feed
            st.image("https://placehold.co/1280x720/0E0E10/333333.png?text=CAMERA+ACTIVE\\nDETECTING+BOXES...", use_container_width=True)
            
            # Overlay Metrics (Simulated)
            m1, m2, m3 = st.columns(3)
            m1.metric("BOXES DETECTED", "12")
            m2.metric("CURRENT CONFIDENCE", "88%")
            m3.metric("FPS", "30")
            
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
            
            rx = st.slider("X OFFSET", 0, 640, 150)
            ry = st.slider("Y OFFSET", 0, 480, 100)
            rw = st.slider("WIDTH", 100, 600, 340)
            rh = st.slider("HEIGHT", 100, 400, 200)
            
        with c2:
            # LIVE PREVIEW MOCKUP
            st.markdown("##### 👁️ PREVIEW")
            
            # Generate black canvas
            preview = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw ROI
            cv2.rectangle(preview, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            # Add text
            cv2.putText(preview, f"ROI: {rw}x{rh}", (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            st.image(preview, channels="BGR", use_container_width=True)

if __name__ == "__main__":
    main()
