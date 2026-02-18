import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import plotly.graph_objects as go
import io

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# REPLACE THESE WITH YOUR ACTUAL DETAILS
API_KEY = "xf4IxH3qirhiaeGu1w7b" 
MODEL_ID = "pl_co/1"   

# LOGO PATH
LOGO_PATH = "logo.png"

# CLASS NAMES (Must match Roboflow exactly)
CLASS_1_NAME = "white" 
CLASS_2_NAME = "black"
CLASS_3_NAME = "circlip"        

# ==========================================
# 🎨 CINEMATIC UI STYLING
# ==========================================
st.set_page_config(
    page_title="VISION SYSTEM | QC INSPECTION",
    layout="wide",
    page_icon="👁️",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Dark Industrial Look
st.markdown("""
    <style>
        .stApp { background-color: #0b0c10; color: #c5c6c7; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* HEADER */
        .header-title {
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            font-size: 2.2rem;
            color: #ffffff;
            letter-spacing: 2px;
            text-transform: uppercase;
            text-shadow: 0 0 10px rgba(102, 252, 241, 0.3);
            border-bottom: 2px solid #66fcf1;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        /* STATUS INDICATORS */
        .status-card {
            background-color: #1f2833;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border: 1px solid #333;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
            margin-bottom: 15px;
        }
        .status-pass { border: 2px solid #00ff00; color: #00ff00; text-shadow: 0 0 15px #00ff00; }
        .status-fail { border: 2px solid #ff0000; color: #ff0000; text-shadow: 0 0 15px #ff0000; }
        .status-text { font-size: 2rem; font-weight: 900; letter-spacing: 3px; }
        .status-sub { font-size: 1rem; opacity: 0.8; text-transform: uppercase; }

        /* BUTTONS */
        div.stButton > button {
            background: linear-gradient(45deg, #1f4068, #162447);
            color: #66fcf1;
            border: 1px solid #66fcf1;
            font-weight: bold;
            font-size: 18px;
            letter-spacing: 1px;
            text-transform: uppercase;
            padding: 12px 24px;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background: #66fcf1;
            color: #0b0c10;
            box-shadow: 0 0 15px #66fcf1;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🔄 SESSION STATE
# ==========================================
if 'ok_count' not in st.session_state:
    st.session_state.ok_count = 0
if 'nok_count' not in st.session_state:
    st.session_state.nok_count = 0

# ==========================================
# 🧠 LOGIC FUNCTIONS
# ==========================================

def run_inference(image):
    """
    Uses multipart/form-data to fix 'Internal Error'.
    """
    CONFIDENCE = 40  
    OVERLAP = 30     

    # 1. Convert PIL Image to JPEG Bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    # 2. Construct URL
    url = (
        f"https://detect.roboflow.com/{MODEL_ID}"
        f"?api_key={API_KEY}"
        f"&confidence={CONFIDENCE}"
        f"&overlap={OVERLAP}"
    )

    # 3. Send POST Request
    try:
        response = requests.post(
            url,
            files={"file": img_bytes} 
        )
        if response.status_code != 200:
            return {"error": True, "message": f"Status {response.status_code}: {response.text}"}
        return response.json()
    except Exception as e:
        return {"error": True, "message": str(e)}

def draw_bbox(image, predictions):
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    if 'predictions' in predictions:
        for p in predictions['predictions']:
            x, y, w, h = int(p['x']), int(p['y']), int(p['width']), int(p['height'])
            label = p['class']
            conf = p['confidence'] * 100
            
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            if label == CLASS_3_NAME:
                color = (0, 255, 0) # Green
            else:
                color = (255, 255, 0) # Cyan/Yellowish
                
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
            
            text = f"{label} {conf:.1f}%"
            cv2.putText(img_cv, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                   
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

def evaluate_logic(predictions):
    if 'predictions' not in predictions:
        return False
        
    found_classes = [p['class'] for p in predictions['predictions']]
    
    has_c1 = CLASS_1_NAME in found_classes
    has_c2 = CLASS_2_NAME in found_classes
    has_c3 = CLASS_3_NAME in found_classes
    
    if (has_c1 and has_c3) or (has_c2 and has_c3):
        return True
    return False

# ==========================================
# 🖥️ MAIN UI LAYOUT
# ==========================================

# HEADER
col_h1, col_h2 = st.columns([1, 8])
with col_h1:
    try:
        st.image(LOGO_PATH, width=90)
    except:
        st.markdown("🌐")
with col_h2:
    st.markdown('<div class="header-title"> CIRCLIP CHECK VISION SYSTEM</div>', unsafe_allow_html=True)

# MAIN CONTENT
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📷 INPUT SOURCE")
    input_method = st.radio("", ("Live Camera", "Local Upload"), horizontal=True)
    
    image_file = None
    if input_method == "Live Camera":
        image_file = st.camera_input("CAPTURE")
    else:
        image_file = st.file_uploader("UPLOAD FILE", type=['jpg', 'png', 'jpeg'])

    if image_file:
        process_btn = st.button("▶ INITIATE SCAN", use_container_width=True)
        
        if process_btn:
            with st.spinner("PROCESSING..."):
                try:
                    image = Image.open(image_file)
                    preds = run_inference(image)
                    
                    if 'error' in preds:
                        st.error(f"API Error: {preds.get('message', 'Unknown Error')}")
                    else:
                        result_img = draw_bbox(image, preds)
                        is_pass = evaluate_logic(preds)
                        
                        if is_pass:
                            st.session_state.ok_count += 1
                            st.session_state['last_status'] = 'PASS'
                        else:
                            st.session_state.nok_count += 1
                            st.session_state['last_status'] = 'FAIL'
                            
                        st.session_state['last_raw_img'] = image
                        st.session_state['last_result_img'] = result_img
                        st.session_state['last_preds'] = preds
                        
                except Exception as e:
                    st.error(f"SYSTEM ERROR: {e}")

    # --- RESULT DISPLAY ---
    if 'last_result_img' in st.session_state:
        st.markdown("---")
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.markdown("#### 📄 RAW IMAGE")
            st.image(st.session_state['last_raw_img'], use_container_width=True)
            
        with img_col2:
            st.markdown("#### 🎯 DETECTED RESULT")
            st.image(st.session_state['last_result_img'], use_container_width=True)

with col2:
    st.markdown("### 📊 SYSTEM STATUS")
    
    if 'last_status' in st.session_state:
        if st.session_state['last_status'] == 'PASS':
            st.markdown("""
                <div class="status-card status-pass">
                    <div class="status-text">✅ PASS</div>
                    <div class="status-sub">COMPONENT OK</div>
                </div>
            """, unsafe_allow_html=True)
        else:
             st.markdown("""
                <div class="status-card status-fail">
                    <div class="status-text">❌ FAIL</div>
                    <div class="status-sub">CIRCLIP MISSING / ERROR</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="status-card" style="border-color: #45a29e; color: #45a29e;">
                <div class="status-text">STANDBY</div>
                <div class="status-sub">WAITING FOR INPUT</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 PRODUCTION METRICS")
    
    total = st.session_state.ok_count + st.session_state.nok_count
    
    if total > 0:
        fig = go.Figure(data=[go.Pie(
            labels=['PASS (OK)', 'FAIL (NOK)'],
            values=[st.session_state.ok_count, st.session_state.nok_count],
            hole=.5,
            marker=dict(colors=['#00ff00', '#ff0000']),
            textinfo='label+percent'
        )])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c5c6c7'),
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("TOTAL", total)
        c2.metric("OK", st.session_state.ok_count)
        c3.metric("NOK", st.session_state.nok_count)