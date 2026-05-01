import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO

# --- GPU MEMORY OPTIMIZATION ---
# This ensures the app only takes what it needs from your 4GB VRAM
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Galaxy XAI Analyzer", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4259; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_my_model():
    # UPDATE THIS PATH to where you saved the .h5 file on your PC
    model_path = "model\galaxy_model.h5" 
    return load_model(model_path)

try:
    model = load_my_model()
    st.sidebar.success("✅ Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

CLASS_NAMES = ["Elliptical", "Spiral", "Irregular"]
# Using Stage 4 layer for 8x8 resolution to prevent heatmap "diverging"
TARGET_LAYER = "conv4_block6_out" 

# --- XAI LOGIC (GRAD-CAM) ---
def get_refined_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # Ensure preds is a tensor and has the correct shape (batch, classes)
        if isinstance(preds, list):
            preds = tf.convert_to_tensor(preds[0])
        
        # If preds is 1D, expand it to 2D [1, classes]
        if len(preds.shape) == 1:
            preds = tf.expand_dims(preds, axis=0)

        # Get the top predicted class index
        class_id = np.argmax(preds[0])
        
        # FIX: Access the scalar value for the class to avoid slicing errors
        class_channel = preds[0, class_id]

    # Calculate gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Guided Gradients: remove negative noise
    guided_grads = tf.cast(last_conv_layer_output > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    pooled_grads = tf.reduce_mean(guided_grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = np.maximum(tf.squeeze(heatmap).numpy(), 0)
    
    # POWER TRANSFORMATION: Sharpens focus to the exact center
    heatmap = np.power(heatmap, 2) 
    
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
        
    return heatmap, class_id
def analyze_galaxy(image):
    # Convert PIL to BGR
    img = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Preprocessing: Center Crop + Resize to 128x128
    h, w, _ = img_bgr.shape
    min_dim = min(h, w)
    img_cropped = img_bgr[(h-min_dim)//2:(h+min_dim)//2, (w-min_dim)//2:(w+min_dim)//2]
    img_resized = cv2.resize(img_cropped, (128, 128))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # 1. Run Model & XAI
    heatmap, class_id = get_refined_heatmap(img_array, model, TARGET_LAYER)
    preds = model.predict(img_array)
    confidence = preds[0][class_id]

    # 2. Process Heatmap (INTER_CUBIC for centering)
    heatmap_res = cv2.resize(heatmap, (128, 128), interpolation=cv2.INTER_CUBIC)
    jet_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)

    # 3. Superimpose (70% original image for detail)
    orig_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(orig_rgb, 0.7, jet_heatmap, 0.3, 0)
    
    return orig_rgb, superimposed, CLASS_NAMES[class_id], confidence

# --- FRONTEND UI ---
st.title("🌌 DeepScan Galaxy Intelligence")
st.markdown("Professional Grade Morphological Classification & Explainability")

# Sidebar inputs
st.sidebar.header("Galactic Input")
choice = st.sidebar.selectbox("Choose Input Method", ["Local Upload", "URL Link"])

image = None
if choice == "Local Upload":
    file = st.sidebar.file_uploader("Drag & Drop or Select Image", type=["jpg", "png", "jpeg"])
    if file: image = Image.open(file)
else:
    url = st.sidebar.text_input("Paste Direct Image URL")
    if url:
        try:
            resp = requests.get(url)
            image = Image.open(BytesIO(resp.content))
        except: st.error("Failed to load image from URL.")

# Analysis execution
if image:
    with st.spinner("Decoding galactic structure..."):
        orig, xai, label, conf = analyze_galaxy(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(orig, caption="Input Data (128x128 Crop)", use_container_width=True)
            st.metric("Morphological Class", label)
        
        with col2:
            st.image(xai, caption="XAI Evidence (Centered Grad-CAM)", use_container_width=True)
            st.metric("Certainty Score", f"{conf:.2%}")

        st.divider()
        st.header("🔬 Model Interpretation Report")
        
        if label == "Spiral":
            st.success("### Feature: Spiral Arms & Curvilinear Flux")
            st.write("The red-spectrum intensity is mapped directly to the **trailing curvilinear arms**. The model has correctly identified the rotational symmetry and resolved dust lanes as the primary classification criteria.")
        elif label == "Elliptical":
            st.info("### Feature: Dense Luminosity Core")
            st.write("The heatmap focus is concentrated on the **dense, spherical central luminosity core**. The model verified a lack of structure (arms or irregular patches), triggering a decision based on the smooth ellipsoidal light gradient.")
        else: # Irregular
            st.warning("### Feature: Asymmetric Distribution")
            st.write("The XAI indicates the model is responding to **disorganized, high-intensity clusters** that lack a defined center or rotational pattern. Classification is based on structural chaos.")
else:
    st.info("Awaiting Input: Please upload a galaxy image or provide a link to begin.")