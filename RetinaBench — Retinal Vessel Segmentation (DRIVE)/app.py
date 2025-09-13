import json, time, numpy as np, cv2, onnxruntime as ort
from pathlib import Path
from PIL import Image
import streamlit as st
import pandas as pd

st.set_page_config(page_title="RetinaBench (ONNX)", layout="wide")

MODELS = ["unet","attn_unet","resunet","deeplabv3","unetpp","pspnet"]
MODEL_DIR = Path("retinabench_artifacts/retinabench")

@st.cache_resource
def load_sessions():
    sessions, thresholds, shapes = {}, {}, {}
    for m in MODELS:
        onnx_path = MODEL_DIR/m/"best.onnx"
        thr_path  = MODEL_DIR/m/"threshold.json"
        if onnx_path.exists():
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            sessions[m] = sess
            thresholds[m] = float(json.load(open(thr_path))["best_threshold"]) if thr_path.exists() else 0.5
            ishape = sess.get_inputs()[0].shape  # [1,1,H,W] or [1,1,None,None]
            shapes[m] = ishape
    return sessions, thresholds, shapes

def preprocess_green(pil_img):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    g = bgr[...,1].astype(np.float32)
    g = (g - g.mean()) / (g.std() if g.std()>0 else 1.0)
    g = np.clip(g, -3, 3); g = ((g+3)/6*255).astype(np.uint8)
    g = cv2.createCLAHE(2.0,(8,8)).apply(g)
    g = (255.0*(g/255.0)**(1/1.2)).astype(np.uint8)
    return g, bgr

def pad_to_32(img):
    h,w = img.shape; H=(h+31)//32*32; W=(w+31)//32*32
    pad = np.zeros((H,W), np.uint8); pad[:h,:w] = img
    return pad,(h,w),(H,W)

def center_crop_or_pad(img, H, W):
    h,w = img.shape
    if h<H or w<W:
        buf = np.zeros((max(h,H), max(w,W)), img.dtype)
        buf[:h,:w] = img
        img = buf; h,w = img.shape
    y0 = (h - H)//2 if h>H else 0
    x0 = (w - W)//2 if w>W else 0
    return img[y0:y0+H, x0:x0+W], (y0,x0)

def overlay_rgb(rgb, mask_u8, color=(0,255,255), alpha=0.55, contour=False):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if contour:
        cnts,_ = cv2.findContours((mask_u8>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = bgr.copy(); cv2.drawContours(out, cnts, -1, color, 2)
    else:
        layer = np.zeros_like(bgr); layer[mask_u8>0] = color
        out = cv2.addWeighted(bgr, 1-alpha, layer, alpha, 0)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def run_one(sess, ishape, img_u8, thr):
    # handles dynamic (None,None) or fixed (e.g., 576x576)
    xpad, (h,w), (Hp,Wp) = pad_to_32(img_u8)
    ipt = sess.get_inputs()[0].name

    # dynamic input -> simple path
    if ishape[2] is None or ishape[3] is None:
        x = xpad.astype(np.float32)/255.0; x = x[None, None, :, :]
        t0 = time.time()
        y  = sess.run(None, {ipt: x})[0][0,0]
        lat = (time.time()-t0)*1000
        y = y[:h, :w]
    else:
        # fixed size -> center crop OR pad, then paste output back into a canvas
        H_exp, W_exp = int(ishape[2]), int(ishape[3])

        # prepare model input
        xfit, _ = center_crop_or_pad(xpad, H_exp, W_exp)
        x = xfit.astype(np.float32)/255.0; x = x[None, None, :, :]

        # run
        t0 = time.time()
        yfit = sess.run(None, {ipt: x})[0][0,0]  # (H_exp, W_exp)
        lat = (time.time()-t0)*1000

        # canvas must be large enough for either path (original padded OR model size)
        baseH, baseW = max(Hp, H_exp), max(Wp, W_exp)
        ypad = np.zeros((baseH, baseW), np.float32)

        # center the model output on that canvas
        y0 = (baseH - H_exp)//2 if baseH > H_exp else 0
        x0 = (baseW - W_exp)//2 if baseW > W_exp else 0
        ypad[y0:y0+H_exp, x0:x0+W_exp] = yfit

        # crop back to original (un-padded) size
        y = ypad[:h, :w]

    mask = (y > thr).astype(np.uint8) * 255
    return mask, lat

sessions, thresholds, shapes = load_sessions()

st.sidebar.title("RetinaBench (ONNX)")
available = [m for m in MODELS if m in sessions]
sel = st.sidebar.multiselect("Select models", available, default=available)
use_saved_thr = st.sidebar.checkbox("Use saved per-model thresholds", True)
user_thr = st.sidebar.slider("Global threshold", 0.10, 0.90, 0.50, 0.01)
alpha = st.sidebar.slider("Overlay strength", 0.1, 0.9, 0.55, 0.05)

st.title("Multi-Model Retinal Vessel Segmentation")
uploaded = st.file_uploader("Upload a fundus image (PNG/JPG/TIF)", type=["png","jpg","jpeg","tif","tiff"])

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    st.image(pil, caption="Input", use_container_width=True)

    green, bgr = preprocess_green(pil)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rows=[]; masks={}
    cols = st.columns(3)
    for i,m in enumerate(sel):
        thr = thresholds[m] if use_saved_thr else user_thr
        mask, lat = run_one(sessions[m], shapes[m], green, thr)
        masks[m] = mask
        with cols[i % 3]:
            st.subheader(m)
            st.image(overlay_rgb(rgb, mask, (0,255,255), alpha, contour=False), caption="Overlay (filled)")
            st.image(overlay_rgb(rgb, mask, (255,0,0), alpha=1.0, contour=True), caption="Overlay (contour)")
            st.download_button(f"Download mask ({m})",
                               data=cv2.imencode(".png", mask)[1].tobytes(),
                               file_name=f"{m}_mask.png", mime="image/png")
        rows.append((m, thr, int((mask>0).sum()), round(lat,1)))
    st.dataframe(pd.DataFrame(rows, columns=["model","threshold","vessel_pixels","latency_ms"]), use_container_width=True)

    if len(masks)>=2:
        prob_stack = np.stack([(masks[m]>0).astype(np.float32) for m in sel], 0)
        ens = (prob_stack.mean(0) > 0.5).astype(np.uint8)*255
        st.subheader("Ensemble (mean>0.5)")
        st.image(overlay_rgb(rgb, ens, (0,200,0), alpha))
        st.download_button("Download mask (ensemble)",
                           data=cv2.imencode(".png", ens)[1].tobytes(),
                           file_name="ensemble_mask.png", mime="image/png")
else:
    st.info("Upload a fundus image to run the models.")
