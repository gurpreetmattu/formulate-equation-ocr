import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

import streamlit as st
import torch
import numpy as np
import io
import json
import cv2
from PIL import Image, ImageEnhance
from latex2mathml.converter import convert as latex2mathml_convert
import torch.nn as nn
import torch.nn.functional as F
import re

# =========================
# CONFIGURATION
# =========================

PROJECT_NAME = "Automated Printed Equation Recognition & LaTeX/MathML Conversion from Images"
VOCAB_PATH = r"D:\equation_recogination_app\vocab.json"
CHECKPOINT_PATH = r"D:\equation_recogination_app\model_checkpoint\full_checkpoint.pt"
IMG_HEIGHT, IMG_WIDTH = 160, 1024
ROW_BI_DIM = 64
HIDDEN_DIM = 512
EMB_DIM = 384
NUM_LAYERS = 2
DROPOUT = 0.3
BEAM_WIDTH = 5
MAX_LEN = 160
device = torch.device("cpu")

st.set_page_config(page_title=PROJECT_NAME, layout="wide")

# ========== SESSION STATE FOR LOGIN ==========
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# ========== UTILS ==========
def preprocess_equation_image_for_inference(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img_np = np.array(img)
    _, thresh = cv2.threshold(img_np, 240, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(255 - thresh)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img_np[y:y+h, x:x+w]
    pil_img = Image.fromarray(cropped)
    sharpness = ImageEnhance.Sharpness(pil_img).enhance(2.0)
    contrast = ImageEnhance.Contrast(sharpness).enhance(1.5)
    img = np.array(contrast)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    img_norm = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    return img_tensor, img

def fix_mathml_block_tag(mathml):
    # Make MathML display="block" and clean up xmlns if needed
    mathml = re.sub(r'display="inline"', 'display="block"', mathml)
    mathml = re.sub(r'<(/?)ns0:', r'<\1', mathml)
    mathml = re.sub(r'xmlns:ns0="[^"]+"', 'xmlns="http://www.w3.org/1998/Math/MathML"', mathml)
    return mathml

# ========== MODEL AND VOCAB LOAD ==========
@st.cache_resource(show_spinner=True, ttl=7200)
def load_model_and_vocab():
    # ---- Define Model Classes ----
    class DepthwiseSeparableConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
            super().__init__()
            self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.act = nn.ReLU(inplace=True)
        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.bn(x)
            x = self.act(x)
            return x
    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size=7):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x_cat = torch.cat([avg_out, max_out], dim=1)
            attn = self.conv(x_cat)
            attn = self.sigmoid(attn)
            return x * attn
    class LWDSCSA_Encoder(nn.Module):
        def __init__(self, in_channels=1, hidden_dims=[32, 64, 128], strides=[2, 2, 1], row_hidden_dim=64):
            super().__init__()
            assert len(hidden_dims) == len(strides)
            layers = []
            prev_dim = in_channels
            for hd, s in zip(hidden_dims, strides):
                layers.append(DepthwiseSeparableConv(prev_dim, hd, 3, stride=s, padding=1))
                prev_dim = hd
            self.dsc_blocks = nn.Sequential(*layers)
            self.spatial_att = SpatialAttention()
            self.row_hidden_dim = row_hidden_dim
            self.row_bilstm = nn.LSTM(
                input_size=hidden_dims[-1],
                hidden_size=row_hidden_dim,
                batch_first=True,
                bidirectional=True
            )
        def forward(self, x):
            feat = self.dsc_blocks(x)
            feat = self.spatial_att(feat)
            B, C, H, W = feat.shape
            feat_reshape = feat.permute(0, 2, 3, 1).contiguous().view(B*H, W, C)
            row_out, _ = self.row_bilstm(feat_reshape)
            row_out = row_out.reshape(B, H, W, 2*self.row_hidden_dim)
            row_out = row_out.permute(0, 2, 1, 3).contiguous()
            row_out = row_out.reshape(B, W, H*2*self.row_hidden_dim)
            return row_out
    class SequenceModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
            super().__init__()
            self.bilstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        def forward(self, x):
            output, _ = self.bilstm(x)
            return output
    class LuongAttention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
        def forward(self, decoder_hidden, encoder_outputs):
            attn_energies = torch.bmm(
                encoder_outputs, decoder_hidden.unsqueeze(2)
            ).squeeze(2)
            attn_weights = torch.softmax(attn_energies, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            return context, attn_weights
    class LuongDecoder(nn.Module):
        def __init__(self, vocab_size, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb_dim)
            self.rnn = nn.LSTM(emb_dim + enc_hidden_dim, dec_hidden_dim, batch_first=True)
            self.attn = LuongAttention(enc_hidden_dim)
            self.out = nn.Linear(dec_hidden_dim, vocab_size)
            self.hidden_proj = nn.Linear(dec_hidden_dim, enc_hidden_dim)
            self.rnn_layers = 1
            self.dec_hidden_dim = dec_hidden_dim
        def forward(self, input_token, prev_hidden, encoder_outputs):
            emb = self.embedding(input_token).unsqueeze(1)
            hidden_for_attn = prev_hidden[0].squeeze(0)
            hidden_for_attn = self.hidden_proj(hidden_for_attn)
            context, attn_weights = self.attn(hidden_for_attn, encoder_outputs)
            context = context.unsqueeze(1)
            rnn_input = torch.cat([emb, context], dim=2)
            output, hidden = self.rnn(rnn_input, prev_hidden)
            output = output.squeeze(1)
            logits = self.out(output)
            return logits, hidden, attn_weights

    # --- Load Vocab ---
    with open(VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)
    idx_to_token = {v: k for k, v in vocab.items()}
    sos_token_id = vocab["<SOS>"]
    eos_token_id = vocab["<EOS>"]
    device = torch.device("cpu")

    encoder = LWDSCSA_Encoder(in_channels=1, hidden_dims=[32, 64, 128], strides=[2, 2, 1], row_hidden_dim=ROW_BI_DIM).to(device)
    seq_model = SequenceModel(input_dim=5120, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    decoder = LuongDecoder(len(vocab), emb_dim=EMB_DIM, enc_hidden_dim=2*HIDDEN_DIM, dec_hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    seq_model.load_state_dict(ckpt["seq_model"])
    decoder.load_state_dict(ckpt["decoder"])

    return encoder, seq_model, decoder, idx_to_token, sos_token_id, eos_token_id, device

encoder, seq_model, decoder, idx_to_token, sos_token_id, eos_token_id, device = load_model_and_vocab()

# ========== INFERENCE AND CONVERSION ==========
def predict_latex(image_tensor, encoder, seq_model, decoder, sos_token_id, eos_token_id, idx_to_token, max_len=MAX_LEN, beam_width=5, mode="greedy", device="cpu"):
    encoder.eval(); seq_model.eval(); decoder.eval()
    with torch.no_grad():
        features = encoder(image_tensor.to(device))
        encoder_outputs = seq_model(features)
        batch_size = 1
        hidden = (
            torch.zeros(1, batch_size, decoder.dec_hidden_dim, device=device),
            torch.zeros(1, batch_size, decoder.dec_hidden_dim, device=device)
        )
        if mode == "greedy":
            input_token = torch.full((batch_size,), sos_token_id, dtype=torch.long, device=device)
            pred_ids = []
            for t in range(max_len):
                logits, hidden, _ = decoder(input_token, hidden, encoder_outputs)
                next_token = logits.argmax(dim=1)
                if next_token.item() == eos_token_id:
                    break
                pred_ids.append(next_token.item())
                input_token = next_token
            pred_tokens = [idx_to_token.get(idx, "<UNK>") for idx in pred_ids]
            return " ".join(pred_tokens)
        elif mode == "beam":
            decoded_tokens = beam_search_decode(
                encoder, seq_model, decoder, image_tensor.squeeze(0),
                sos_token_id, eos_token_id, beam_width=beam_width,
                max_len=max_len, device=device
            )
            pred_tokens = [idx_to_token.get(idx, "<UNK>") for idx in decoded_tokens]
            return " ".join(pred_tokens)
        else:
            raise ValueError("Unknown decoding mode. Use 'greedy' or 'beam'.")

def beam_search_decode(encoder, seq_model, decoder, image, sos_token_id, eos_token_id, beam_width=BEAM_WIDTH, max_len=MAX_LEN, device='cpu'):
    encoder.eval(); seq_model.eval(); decoder.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        features = encoder(image)
        encoder_outputs = seq_model(features)
        batch_size = 1
        h_0 = torch.zeros(1, batch_size, decoder.dec_hidden_dim, device=device)
        c_0 = torch.zeros(1, batch_size, decoder.dec_hidden_dim, device=device)
        hidden = (h_0, c_0)
        beams = [([sos_token_id], hidden, 0.0, False)]
        for _ in range(max_len):
            new_beams = []
            for tokens, h, score, finished in beams:
                if finished:
                    new_beams.append((tokens, h, score, True))
                    continue
                input_token = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
                logits, h_new, _ = decoder(input_token, h, encoder_outputs)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                for log_p, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                    next_tokens = tokens + [idx]
                    next_score = score + log_p
                    next_finished = (idx == eos_token_id)
                    new_beams.append((next_tokens, h_new, next_score, next_finished))
            new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]
            beams = new_beams
            if all([b[3] for b in beams]):
                break
        best_tokens = beams[0][0][1:]
        if eos_token_id in best_tokens:
            best_tokens = best_tokens[:best_tokens.index(eos_token_id)]
        return best_tokens

# --- Predict & MathML Conversion ---
def predict_and_convert_mathml(img_tensor, encoder, seq_model, decoder, sos_token_id, eos_token_id, idx_to_token, max_len=160, beam_width=5, device="cpu"):
    img_tensor = img_tensor.to(device)
    # Greedy
    try:
        greedy_pred = predict_latex(img_tensor, encoder, seq_model, decoder, sos_token_id, eos_token_id, idx_to_token, mode="greedy")
    except Exception as e:
        greedy_pred = f"[Prediction error: {e}]"
    try:
        greedy_mathml = latex2mathml_convert(greedy_pred)
        greedy_mathml = fix_mathml_block_tag(greedy_mathml)
    except Exception as e:
        greedy_mathml = f"[Conversion error: {e}]"
    # Beam
    try:
        beam_pred = predict_latex(img_tensor, encoder, seq_model, decoder, sos_token_id, eos_token_id, idx_to_token, beam_width=beam_width, mode="beam")
    except Exception as e:
        beam_pred = f"[Prediction error: {e}]"
    try:
        beam_mathml = latex2mathml_convert(beam_pred)
        beam_mathml = fix_mathml_block_tag(beam_mathml)
    except Exception as e:
        beam_mathml = f"[Conversion error: {e}]"
    return greedy_pred, greedy_mathml, beam_pred, beam_mathml

# ========== STREAMLIT PAGE LAYOUT ==========
PAGES = ["Equation OCR", "Examples", "About"]
page = st.sidebar.selectbox("Navigation", PAGES)

if page == "Equation OCR":
    st.title(PROJECT_NAME)
    if not st.session_state["logged_in"]:
        st.subheader("Login")
        username = st.text_input("Enter your name to continue", "")
        if st.button("Login") and username.strip():
            st.session_state["logged_in"] = True
            st.session_state["username"] = username.strip()
            st.success(f"Welcome, {username}!")
        st.stop()
    else:
        st.markdown(f"**Logged in as:** `{st.session_state['username']}`")
        st.header("Upload an equation image")
        uploaded = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "bmp", "tiff"])
        if uploaded:
            file_bytes = uploaded.read()
            try:
                pil_image = Image.open(io.BytesIO(file_bytes))
                png_bytes = io.BytesIO()
                pil_image.save(png_bytes, format="PNG")
                png_bytes = png_bytes.getvalue()
                img_tensor, processed_img_for_plot = preprocess_equation_image_for_inference(png_bytes)
            except Exception as e:
                st.error(f"Image preprocessing failed: {e}")
                st.stop()

            st.subheader("Preprocessed Image for Model")
            st.image(processed_img_for_plot, caption="Preprocessed Input", width=600)

            # Inference
            greedy_pred, greedy_mathml, beam_pred, beam_mathml = predict_and_convert_mathml(
                img_tensor, encoder, seq_model, decoder,
                sos_token_id, eos_token_id, idx_to_token,
                max_len=MAX_LEN, beam_width=BEAM_WIDTH, device=device,
            )

            # --- Output Section ---
            st.subheader("Model Outputs")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Greedy Decoding (LaTeX):**")
                st.code(greedy_pred or "[error]", language=None)
                st.markdown("**Greedy Decoding (MathML):**")
                st.code(greedy_mathml or "[error]", language=None)
            with col2:
                st.markdown("**Beam Search Decoding (LaTeX):**")
                st.code(beam_pred or "[error]", language=None)
                st.markdown("**Beam Search Decoding (MathML):**")
                st.code(beam_mathml or "[error]", language=None)

            # --- Visual Comparison (both render, fallback if one fails) ---
            st.subheader("Visual Comparison")
            #st.markdown("**Input Image (left) | Rendered LaTeX - Greedy (middle) | Rendered LaTeX - Beam (right)**")
            render_col1, render_col2, render_col3 = st.columns(3)
            with render_col1:
                st.markdown("**Input Image:**")
                st.image(processed_img_for_plot, width=400)
            with render_col2:
                st.markdown("**Greedy LaTeX Render:**")
                try:
                    st.latex(greedy_pred)
                except Exception:
                    st.markdown("_LaTeX render failed for Greedy._")
            with render_col3:
                st.markdown("**Beam LaTeX Render:**")
                try:
                    st.latex(beam_pred)
                except Exception:
                    st.markdown("_LaTeX render failed for Beam._")

elif page == "Examples":
    st.title("Example Equations & Results")
    EXAMPLES_DIR = r"D:\equation_recogination_app\examples"
    META_PATH = os.path.join(EXAMPLES_DIR, "examples.json")

    if not os.path.exists(META_PATH):
        st.warning("No example metadata found. Please add examples/examples.json and images to use this page.")
        st.stop()

    with open(META_PATH, "r", encoding="utf-8") as f:
        examples_meta = json.load(f)

    st.info("Browse a gallery of diverse test cases. For each, view the input image, model's predicted LaTeX and MathML, and rendering success status.")

    for fname, meta in examples_meta.items():
        img_path = os.path.join(EXAMPLES_DIR, fname)
        if not os.path.exists(img_path):
            st.error(f"Image missing: {fname}")
            continue

        # --- Inference on example ---
        with open(img_path, "rb") as fimg:
            file_bytes = fimg.read()
        img_tensor, processed_img_for_plot = preprocess_equation_image_for_inference(file_bytes)
        greedy_pred, greedy_mathml, beam_pred, beam_mathml = predict_and_convert_mathml(
            img_tensor, encoder, seq_model, decoder,
            sos_token_id, eos_token_id, idx_to_token,
            max_len=MAX_LEN, beam_width=BEAM_WIDTH, device=device,
        )

        st.markdown(f"---\n### Example: {meta.get('desc', fname)}")

        # Side-by-side: Input | Greedy | Beam
        
        
        st.image(processed_img_for_plot, caption="Input", width=400)
        st.markdown("**Ground Truth LaTeX:**")
        st.code(meta["latex"], language=None)
        
        st.markdown("**Greedy Decoding (LaTeX):**")
        st.code(greedy_pred, language=None)
        st.markdown("**Greedy Decoding (MathML):**")
        if greedy_mathml.startswith("[Conversion error"):
            st.error("MathML conversion failed")
        else:
            st.code(greedy_mathml, language=None)
        st.markdown("**Greedy LaTeX Render:**")
        try:
            st.latex(greedy_pred)
            st.success("Render: Success")
        except Exception:
            st.markdown("_LaTeX render failed._")
            st.error("Render: Failed")
        
        st.markdown("**Beam Search Decoding (LaTeX):**")
        st.code(beam_pred, language=None)
        st.markdown("**Beam Search Decoding (MathML):**")
        if beam_mathml.startswith("[Conversion error"):
            st.error("MathML conversion failed")
        else:
            st.code(beam_mathml, language=None)
        st.markdown("**Beam LaTeX Render:**")
        try:
            st.latex(beam_pred)
            st.success("Render: Success")
        except Exception:
            st.markdown("_LaTeX render failed._")
            st.error("Render: Failed")


elif page == "About":
    st.title("About This Project")
    st.markdown("""
    ## Automated Printed Equation Recognition & LaTeX/MathML Conversion from Images

    This web application provides a robust, research-grade pipeline for converting **images of printed mathematical equations** into both LaTeX and MathML formats, using state-of-the-art deep learning techniques.  
    It empowers researchers, educators, and students to digitize complex printed math for seamless integration into **LaTeX papers, Word documents, and web platforms**.

    ---
    ### **Key Features**
    - **End-to-end OCR:** Convert printed equation images directly to editable LaTeX & MathML.
    - **Dual Decoding:** See outputs from both Greedy and Beam Search strategies.
    - **Side-by-side Visual Comparison:** Instantly verify model output against the original image.
    - **Copy & Download:** Extract LaTeX/MathML easily for use in papers, Word, or other tools.
    - **Example Gallery:** Explore a variety of equation types and see model performance.

    ---
    ### **Technical Highlights**
    - **Model Architecture:**  
      Custom **LWDSC-SA (Lightweight Depthwise Separable Conv + Spatial Attention)** encoder  
      → **BiLSTM sequence model**  
      → **Luong Attention Decoder**
    - **Dataset:** Based on im2latex-100k.
    - **Image Preprocessing:** Adaptive cropping, enhancement, normalization.
    - **GPU-Optimized:** Trained and inferenced on RTX 2060 for high throughput.
    - **Conversion Pipeline:**  
      `Image → LaTeX (Deep Learning) → MathML (latex2mathml Python library)`
    - **Limitations:**  
      - Best on high-quality scans of printed (not handwritten) equations  
      - May require post-editing for deeply nested or unusual formulas

    ---
    ### **Usage Instructions**
    1. **Login** with your name.
    2. **Upload a clear, cropped image** (PNG/JPG/BMP/TIFF) of a printed math equation.
    3. **Review the LaTeX & MathML outputs** (both Greedy and Beam Search).
    4. **Compare visual renderings** and copy code as needed.
    5. Explore the **Examples** page for more sample use cases.

    ---
    ### **Future Enhancements**
    - Handwritten and mixed-content equation support
    - Batch processing and advanced analytics
    - More robust MathML for Word and WYSIWYG editors
    - API endpoint for integration with other tools

    ---
    ### **Contact & Acknowledgments**
    - **Lead Developer:** Gurpreet Singh, MSc Data Science & Spatial Analytics  
      Symbiosis Institute of Geoinformatics, Pune
    - **Contact:** gurpreetsinghmattu2002@gmail.com 
    - **Inspired by:** Mathpix, im2latex, PyTorch & Streamlit communities  
    - **Source code:** NA

    ---
    _© 2025 Gurpreet Singh. For academic, non-commercial use only._
    """)