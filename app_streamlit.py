import streamlit as st
import os
import time
import math
import sys

from PIL import Image
import pandas as pd
import gdown

# -------------------------------------------------------------------
# Configuration & CSS Setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="DeepBovine AI",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Earthy Dark Theme
st.markdown("""
    <style>
    :root {
        --deep-wood: #121212;
        --earthy-brown: #2C2520;
        --sage-green: #7BAE7F;
        --sage-hover: #659368;
        --terracotta: #D17B8F;
        --sandy-cream: #EAE6DB;
        --card-bg: rgba(44, 37, 32, 0.6);
    }
    .stApp {
        background-color: var(--deep-wood);
        color: var(--sandy-cream);
    }
    /* Headers */
    h1, h2, h3 {
        color: var(--sage-green) !important;
        font-family: 'Outfit', sans-serif !important;
    }
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: var(--terracotta) !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--sandy-cream) !important;
        font-size: 1.1rem !important;
        opacity: 0.8;
    }
    /* Button */
    .stButton > button {
        background-color: var(--sage-green) !important;
        color: var(--deep-wood) !important;
        border: none !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        background-color: var(--sage-hover) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(123, 174, 127, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Language Translations
# -------------------------------------------------------------------
TRANSLATIONS = {
    "English": {
        "title": "🌾 DeepBovine AI",
        "subtitle": "### Intelligent Cattle Weight & Nutritional Management",
        "config": "⚙️ Configuration",
        "lang": "🗣️ Language",
        "ai_engine": "1. AI Inference Engine",
        "model_select": "Select Model:",
        "c_profile": "2. Cattle Profile",
        "breed": "Breed Type:",
        "stage": "Growth Stage:",
        "stages": ["Calf / Growing Heifer", "Dry Cow", "Milking Cow"],
        "milk_yield": "Expected Milk Yield (Liters/Day):",
        "upload_sec": "📸 1. Image Upload",
        "upload_side": "Upload Side View Image",
        "upload_rear": "Upload Rear View Image",
        "calc_btn": "Calculate Weight & Nutritional Needs",
        "warn_upload": "⚠️ Please upload both Side and Rear images to proceed.",
        "analyzing": "🧠 Analyzing Images using Computer Vision...",
        "success": "✅ Prediction Complete in {}s!",
        "est_weight": "### ⚖️ Estimated Live Weight",
        "bw": "Body Weight",
        "ratio": "Cattle/Sticker Ratio",
        "status": "Status",
        "masks": "#### Detection Masks",
        "tracker": "### 🥗 Daily Nutritional Requirement Tracker",
        "optimal_feed": "Optimal feed ratios calculated for a **{}** weighing approximately **{} kg**.",
        "dry": "🌾 Dry Fodder (Bhusa)",
        "green": "🌿 Green Fodder",
        "conc": "🥘 Concentrates",
        "diet_comp": "#### Diet Composition Breakdown",
        "faq_title": "📚 Frequently Asked Questions & Feed Guide",
        "q_dry": "🌾 What is Dry Fodder (Bhusa)?",
        "a_dry": "**Dry fodder** consists of dried agricultural by-products like wheat straw (bhusa), paddy straw, dry grasses, and maize stover.\nIt is low in moisture (around 10%) but high in essential fiber. Fiber acts as the \"scratch factor\" necessary for healthy rumen function, rumination (cud-chewing), and maintaining butterfat levels in milk.",
        "q_green": "🌿 What is Green Fodder?",
        "a_green": "**Green fodder** includes lush, fresh crops such as Napier grass, Berseem, Lucerne, or freshly cut millet.\nIt is highly palatable, very digestible, and has high moisture content. It is absolutely crucial for maintaining milk production, hydration, and supplying natural vitamins (especially Vitamin A) to the animal.",
        "q_conc": "🥘 What are Concentrates?",
        "a_conc": "**Concentrates** are nutrient-dense, easily digestible feed mixes packed with energy and protein.\nCommon examples include mustard cake, cotton seed cake, crushed grains, bran, and commercial cattle feed pellets. They are given in smaller quantities to supplement the base diet, specifically targeted for boosting milk yield, pregnancy support, or rapid growth in calves.",
        "q_why": "⚖️ Why does the recommended diet change?",
        "a_why": "Feed is calculated primarily on **Dry Matter Intake (DMI)**, which scales directly with the live body weight (typically ~2.5% of body weight).\nHowever, the composition shifts based on the life stage:\n*   **Milking Cows** need heavy concentrates for energy to produce milk.\n*   **Calves** need higher protein ratios for muscular development.\n*   **Dry Cows** need basic maintenance diets composed mostly of roughage."
    },
    "हिन्दी": {
        "title": "🌾 डीपबोवाइन एआई (DeepBovine AI)",
        "subtitle": "### बुद्धिमान पशु वजन और पोषण प्रबंधन",
        "config": "⚙️ विन्यास (Configuration)",
        "lang": "🗣️ भाषा (Language)",
        "ai_engine": "1. एआई मॉडल (AI Engine)",
        "model_select": "मॉडल चुनें:",
        "c_profile": "2. पशु प्रोफ़ाइल (Profile)",
        "breed": "नस्ल का प्रकार:",
        "stage": "विकास का चरण:",
        "stages": ["बछड़ा (Calf)", "सूखी गाय (Dry Cow)", "दुधारू गाय (Milking Cow)"],
        "milk_yield": "अपेक्षित दूध उत्पादन (लीटर/दिन):",
        "upload_sec": "📸 1. चित्र अपलोड करें",
        "upload_side": "साइड व्यू (बगल का) चित्र अपलोड करें",
        "upload_rear": "रियर व्यू (पीछे का) चित्र अपलोड करें",
        "calc_btn": "वजन और पोषण की गणना करें",
        "warn_upload": "⚠️ आगे बढ़ने के लिए कृपया साइड और रियर दोनों चित्र अपलोड करें।",
        "analyzing": "🧠 कंप्यूटर विजन द्वारा विश्लेषण किया जा रहा है...",
        "success": "✅ भविष्यवाणी {} सेकंड में पूरी हुई!",
        "est_weight": "### ⚖️ अनुमानित जीवित वजन",
        "bw": "शरीर का वजन",
        "ratio": "पशु/स्टिकर अनुपात",
        "status": "स्थिति",
        "masks": "#### डिटेक्शन मास्क",
        "tracker": "### 🥗 दैनिक पोषण आवश्यकता",
        "optimal_feed": "लगभग **{} किलो** वजन वाली **{}** के लिए इष्टतम फ़ीड अनुपात।",
        "dry": "🌾 सूखा चारा (भूसा)",
        "green": "🌿 हरा चारा",
        "conc": "🥘 दाना (Concentrates)",
        "diet_comp": "#### आहार संरचना (Diet Breakdown)",
        "faq_title": "📚 अक्सर पूछे जाने वाले प्रश्न और फ़ीड गाइड",
        "q_dry": "🌾 सूखा चारा (भूसा) क्या है?",
        "a_dry": "**सूखे चारे** में सूखी घास, गेहूं का भूसा, धान का पुआल और मक्के की कड़बी शामिल हैं।\nयह फाइबर से भरपूर होता है जो पाचन और रूमेन (पेट) के स्वास्थ्य के लिए आवश्यक है।",
        "q_green": "🌿 हरा चारा क्या है?",
        "a_green": "**हरे चारे** में नेपियर घास, बरसीम, ल्यूसर्न या ताजी कटी हुई बाजरे जैसी फसलें शामिल हैं।\nयह स्वादिष्ट होता है और पानी व विटामिन (विशेषकर विटामिन ए) की पूर्ति करता है। दुधारू जानवरों के लिए यह बहुत महत्वपूर्ण है।",
        "q_conc": "🥘 दाना (Concentrates) क्या है?",
        "a_conc": "**दाना (Concentrates)** ऊर्जा और प्रोटीन से भरपूर फ़ीड मिश्रण हैं।\nइनमें सरसों की खली, बिनौला, मक्का, चोकर और वाणिज्यिक पशु चारा (पेलट्स) शामिल हैं। इसे दूध उत्पादन बढ़ाने या बछड़ों के तेजी से विकास के लिए दिया जाता है।",
        "q_why": "⚖️ अनुशंसित आहार क्यों बदलता है?",
        "a_why": "फ़ीड की गणना मुख्य रूप से लाइव वजन (DMI) के आधार पर की जाती है।\nलेकिन आवश्यकता जीवन चरण के आधार पर बदलती है:\n*   **दुधारू गायों** को दूध बनाने की ऊर्जा के लिए भारी दाने (concentrates) की आवश्यकता होती है।\n*   **बछड़ों** को मांसपेशियों के विकास के लिए अधिक प्रोटीन चाहिए।\n*   **सूखी गायों** को मुख्य रूप से केवल चारे या रखरखाव आहार की आवश्यकता होती है।"
    }
}

# -------------------------------------------------------------------
# Data Dictionaries for Nutritional Module
# -------------------------------------------------------------------
BASE_NUTRITION_CHART = {
    150: (1.25, 11.0, 1.0),
    175: (1.50, 13.5, 1.2),
    200: (1.75, 16.5, 1.5),
    225: (2.00, 19.0, 1.7),
    250: (2.25, 21.0, 2.0),
    275: (2.75, 23.5, 2.0),
    300: (3.00, 26.0, 2.5),
    325: (3.25, 29.0, 2.5),
    350: (3.50, 31.0, 3.0),
    375: (3.75, 33.5, 3.0),
    400: (4.00, 36.5, 3.5),
    425: (4.25, 38.0, 4.0),
    450: (4.50, 40.0, 4.5),
    475: (4.75, 42.0, 5.0),
    500: (5.00, 45.0, 5.5)
}

def get_nearest_feed_bracket(weight):
    if weight < 150: return 150
    if weight > 500: return 500
    return 25 * round(weight / 25)

def calculate_daily_feed(weight, profile_stage, milk_yield):
    bracket = get_nearest_feed_bracket(weight)
    dry_base, green_base, conc_base = BASE_NUTRITION_CHART[bracket]
    
    # Adjust based on array index matching
    if profile_stage in ["Calf / Growing Heifer", "बछड़ा (Calf)"]:
        conc_base += 0.5 
    elif profile_stage in ["Milking Cow", "दुधारू गाय (Milking Cow)"]:
        conc_base += (milk_yield / 2.0)
    
    return dry_base, green_base, conc_base
    
# -------------------------------------------------------------------
# Backend Inference Integration Setup
# -------------------------------------------------------------------
import sys
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

# Auto-download models from Google Drive on first boot
def ensure_models_downloaded():
    sentinel = os.path.join(project_root, "models", "v1", "seg", "iter_40000.pth")
    if not os.path.exists(sentinel):
        st.info("⏳ First boot: Downloading model weights (~2.3 GB). This may take 3–5 minutes...")
        import traceback
        try:
            gdown.download_folder(
                id="1h0GxqjjuxZnmrIdhdHI731jnDv3AbzkU",
                output=os.path.join(project_root, "models"),
                quiet=False,
                use_cookies=False
            )
            st.success("✅ Models downloaded successfully! Reloading...")
            st.rerun()
        except Exception as e:
            st.error("🚨 Google Drive blocked the automated 2.3GB download via Captcha/Rate Limit! Please use the HuggingFace upload method we discussed instead.")
            st.code(traceback.format_exc())

ensure_models_downloaded()

@st.cache_resource
def load_inference_modules():
    from inference import inference_optimized 
    return inference_optimized

def save_uploaded_file(uploaded_file, filename):
    upload_dir = os.path.join(project_root, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# -------------------------------------------------------------------
# Streamlit App UI
# -------------------------------------------------------------------

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Language Toggle
    current_lang = st.radio("🗣️ Language / भाषा", ["English", "हिन्दी"], horizontal=True)
    t = TRANSLATIONS[current_lang]
    
    st.markdown("---")
    st.header(t["config"])
    
    st.subheader(t["c_profile"])
    breed = st.selectbox(
        t["breed"],
        options=["Local/Indigenous", "Exotic/Crossbreed"]
    )
    stage = st.selectbox(
        t["stage"],
        options=t["stages"]
    )
    
    milk_yield = 0.0
    if stage in ["Milking Cow", "दुधारू गाय (Milking Cow)"]:
        milk_yield = st.slider(t["milk_yield"], min_value=0.0, max_value=30.0, value=10.0, step=0.5)

st.title(t["title"])
st.markdown(t["subtitle"])

# Main App Body
st.markdown("---")
st.subheader(t["upload_sec"])
col1, col2 = st.columns(2)

with col1:
    side_img_file = st.file_uploader(t["upload_side"], type=["jpg", "jpeg", "png"])
    if side_img_file:
        st.image(side_img_file, use_column_width=True, caption="Side View")

with col2:
    rear_img_file = st.file_uploader(t["upload_rear"], type=["jpg", "jpeg", "png"])
    if rear_img_file:
        st.image(rear_img_file, use_column_width=True, caption="Rear View")

if st.button(t["calc_btn"]):
    if not side_img_file or not rear_img_file:
        st.warning(t["warn_upload"])
    else:
        with st.spinner(t["analyzing"]):
            # Setup
            side_path = save_uploaded_file(side_img_file, "temp_side.jpg")
            rear_path = save_uploaded_file(rear_img_file, "temp_rear.jpg")
            inf_opt = load_inference_modules()
            
            # Run Inference
            result = None
            start_time = time.time()
            try:
                result = inf_opt.predict(side_path, rear_path)
            except Exception as e:
                st.error(f"Error during inference: {str(e)}")
            
            duration = round(time.time() - start_time, 2)
            
            if result and result.get("weight", 0) > 0:
                weight = round(result["weight"], 2)
                ratio = round(result.get("ratio", 0), 2)
                
                st.success(t["success"].format(duration))
                
                # Render Prediction Metrics
                st.markdown("---")
                st.markdown(t["est_weight"])
                
                metric_col1, metric_col2, metric_col3 = st.columns([2, 1, 1])
                with metric_col1:
                    st.metric(t["bw"], f"{weight} kg")
                with metric_col2:
                    st.metric(t["ratio"], f"{ratio}")
                with metric_col3:
                    st.metric(t["status"], result.get("remarks", "OK"))
                    
                # Render Output Images
                st.markdown(t["masks"])
                out_col1, out_col2 = st.columns(2)
                try:
                    if os.path.exists("side_seg_output.jpg") and os.path.exists("rear_seg_output.jpg"):
                        with out_col1:
                            st.image("side_seg_output.jpg", use_column_width=True)
                        with out_col2:
                            st.image("rear_seg_output.jpg", use_column_width=True)
                except:
                    pass

                # Render Feed Calculator
                st.markdown("---")
                st.markdown(t["tracker"])
                st.markdown(t["optimal_feed"].format(stage, weight))
                
                dry_f, green_f, conc_f = calculate_daily_feed(weight, stage, milk_yield)
                
                feed_col1, feed_col2, feed_col3 = st.columns(3)
                with feed_col1:
                    st.metric(t["dry"], f"{dry_f:.1f} kg")
                with feed_col2:
                    st.metric(t["green"], f"{green_f:.1f} kg")
                with feed_col3:
                    st.metric(t["conc"], f"{conc_f:.1f} kg")
                
                # Simple Data Chart
                st.markdown(t["diet_comp"])
                diet_df = pd.DataFrame({
                    "Feed Type": ["Dry Fodder", "Green Fodder", "Concentrates"],
                    "Amount (kg)": [dry_f, green_f, conc_f]
                })
                st.bar_chart(diet_df.set_index("Feed Type"), height=300)

            else:
                msg = result.get("remarks", "Failed to detect cattle or sticker.") if result else "Inference failure."
                st.error(f"❌ Error: {msg}")

# FAQ Section
st.markdown("---")
st.subheader(t["faq_title"])

# Wait to ensure t is defined out of scope for sidebar. We must fetch t again just in case.
# Oh wait, Streamlit runs top-to-bottom so t is perfectly defined here.
col_faq1, col_faq2 = st.columns(2)

with col_faq1:
    with st.expander(t["q_dry"]):
        st.write(t["a_dry"])

    with st.expander(t["q_green"]):
        st.write(t["a_green"])

with col_faq2:
    with st.expander(t["q_conc"]):
        st.write(t["a_conc"])

    with st.expander(t["q_why"]):
        st.write(t["a_why"])
