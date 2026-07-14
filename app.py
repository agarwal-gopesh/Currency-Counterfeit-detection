import streamlit as st
from pathlib import Path
from datetime import datetime

from src.models.predict import (
    load_saved_model,
    preprocess_image,
    predict,
)

st.set_page_config(
    page_title="AuthentiNote",
    page_icon="🛡️",
    layout="centered",
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "artifacts" / "models" / "multi_output_model.keras"
)

AUTH_CLASSES = ["fake", "real"]


@st.cache_resource
def get_model():
    return load_saved_model(MODEL_PATH)


# ----------------------------------------------------------------------
# STYLES
# ----------------------------------------------------------------------
def local_css():
    st.markdown(
        """
        <style>
        @import url(
            'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
        );

        html, body, .stApp {
            font-family: 'Inter', sans-serif;
            background-color: #f7f9f8;
        }

        /* remove default streamlit padding */
        .block-container {
            padding-top: 0 !important;
            padding-bottom: 3rem;
            max-width: 760px;
        }
        header[data-testid="stHeader"] { background: transparent; }

        /* ---------- TOP NAV ---------- */
        .topnav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #0c2b1e;
            padding: 1.1rem 1.6rem;
            margin: 0 -1rem 2rem -1rem;
            border-radius: 0 0 14px 14px;
        }
        .topnav-left { display: flex; align-items: center; gap: 0.8rem; }
        .topnav-logo {
            width: 42px; height: 42px;
            background: #123d2a;
            border: 1px solid #2f6b4a;
            border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            color: #4ade80; font-size: 1.3rem; font-weight: 700;
        }
        .topnav-title { color: #ffffff; font-size: 1.35rem; font-weight: 700; line-height: 1.1; }
        .topnav-sub { color: #9fbfae; font-size: 0.8rem; margin-top: 2px; }
        .topnav-menu { color: #cfe4d7; font-size: 1.4rem; }

        /* ---------- HERO / UPLOAD PAGE ---------- */
        .hero-title {
            text-align: center;
            font-size: 2.3rem;
            font-weight: 800;
            color: #14251c;
            margin-bottom: 0.4rem;
        }
        .hero-title .accent { color: #1e9e5c; }
        .hero-sub {
            text-align: center;
            color: #667a70;
            font-size: 1.02rem;
            margin-bottom: 2rem;
            line-height: 1.5;
        }
        .hero-sub .real { color: #1e9e5c; font-weight: 600; }
        .hero-sub .fake { color: #e0483f; font-weight: 600; }

        .upload-box {
            border: 2px dashed #9fd6b6;
            border-radius: 16px;
            padding: 2.5rem 2rem 1.6rem 2rem;
            text-align: center;
            background: #f2faf5;
            margin-bottom: 0.6rem;
        }
        .upload-icon { font-size: 2.6rem; color: #1e9e5c; margin-bottom: 0.6rem; }
        .upload-text { font-size: 1.05rem; font-weight: 600; color: #1c2b23; margin-bottom: 0.3rem; }
        .upload-or { color: #94a89d; font-size: 0.9rem; margin: 0.4rem 0 0.9rem 0; }
        .upload-hint {
            text-align: center; color: #8b9a92; font-size: 0.82rem;
            margin: 0.8rem 0 1.8rem 0;
        }

        .how-box {
            background: #ffffff;
            border: 1px solid #eef1ef;
            border-radius: 14px;
            padding: 1.6rem;
            margin-bottom: 1.2rem;
        }
        .how-title { text-align: center; font-weight: 700; font-size: 1.15rem; color: #14251c; margin-bottom: 1.2rem; }
        .how-step { text-align: center; }
        .how-circle {
            width: 54px; height: 54px; border-radius: 50%;
            background: #e5f6ec; color: #1e9e5c;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.4rem; margin: 0 auto 0.6rem auto;
        }
        .how-step-title { font-weight: 700; color: #14251c; font-size: 0.95rem; margin-bottom: 0.2rem; }
        .how-step-desc { color: #8b9a92; font-size: 0.8rem; line-height: 1.3; }
        .how-arrow { text-align: center; color: #b9c7c0; font-size: 1.3rem; padding-top: 1.1rem; }

        .info-banner {
            background: #eaf7ef;
            border: 1px solid #cdeedb;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            display: flex; align-items: center; gap: 0.8rem;
            color: #1c4d33; font-size: 0.88rem; line-height: 1.4;
        }

        /* ---------- RESULT PAGE ---------- */
        .back-link { color: #1e9e5c; font-weight: 600; font-size: 0.95rem; margin-bottom: 1rem; }

        .result-card {
            background: #ffffff;
            border: 1px solid #eef1ef;
            border-radius: 16px;
            padding: 1.8rem;
            display: flex;
            align-items: center;
            gap: 1.8rem;
            margin-bottom: 1.5rem;
        }
        .result-badge {
            min-width: 150px; height: 150px; border-radius: 50%;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            border: 6px solid;
        }
        .result-badge.real { border-color: #1e9e5c; background: #f2faf5; }
        .result-badge.fake { border-color: #e0483f; background: #fdf2f1; }
        .badge-icon { font-size: 1.9rem; }
        .badge-label { font-size: 1.5rem; font-weight: 800; margin-top: 0.1rem; }
        .badge-label.real { color: #1e9e5c; }
        .badge-label.fake { color: #e0483f; }
        .badge-conf { font-size: 0.8rem; color: #5c7568; margin-top: 0.1rem; }

        .result-info-label { font-weight: 700; color: #14251c; font-size: 1rem; }
        .result-info-verdict { font-size: 1.5rem; font-weight: 800; margin: 0.2rem 0 0.6rem 0; }
        .result-info-verdict.real { color: #1e9e5c; }
        .result-info-verdict.fake { color: #e0483f; }
        .result-info-desc { color: #5c7568; font-size: 0.9rem; line-height: 1.5; margin-bottom: 0.9rem; }
        .result-timestamp {
            display: inline-block; background: #eaf7ef; color: #1c4d33;
            border-radius: 8px; padding: 0.3rem 0.7rem; font-size: 0.78rem;
        }

        .section-title { font-weight: 700; color: #14251c; font-size: 1.05rem; margin: 1.5rem 0 0.8rem 0; }

        .summary-item {
            background: #f8faf9; border: 1px solid #eef1ef; border-radius: 10px;
            padding: 0.8rem 1rem; display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 0.7rem; font-size: 0.88rem;
        }
        .summary-item .label { font-weight: 600; color: #14251c; }
        .summary-item .status { color: #1e9e5c; font-size: 0.78rem; display: block; margin-top: 2px; }
        .summary-item .check { color: #1e9e5c; font-size: 1.1rem; }

        div[data-testid="stButton"] > button {
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            padding: 0.7rem 1.6rem;
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            background-color: #0c2b1e;
            border: 1px solid #0c2b1e;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background-color: #123d2a;
            border: 1px solid #123d2a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def top_nav():
    st.markdown(
        """
        <div class="topnav">
            <div class="topnav-left">
                <div class="topnav-logo">₹</div>
                <div>
                    <div class="topnav-title">AuthentiNote</div>
                    <div class="topnav-sub">Currency Authenticity Detection</div>
                </div>
            </div>
            <div class="topnav-menu">☰</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# UPLOAD PAGE
# ----------------------------------------------------------------------
def upload_page():
    local_css()
    top_nav()

    st.markdown(
        '<p class="hero-title">Check Currency <span class="accent">Authenticity</span></p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">Upload an image of the currency note to detect '
        'whether it is <span class="real">Real</span> or '
        '<span class="fake">Fake</span></p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="upload-box">'
        '<div class="upload-icon">☁️⬆</div>'
        '<div class="upload-text">Drag &amp; Drop your image here</div>'
        '<div class="upload-or">or</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload a currency note image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="upload-hint">Supports: JPG, PNG, JPEG &nbsp;|&nbsp; Max size: 10MB</p>',
        unsafe_allow_html=True,
    )

    if uploaded:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded, caption="Uploaded Note", width=280)

    if st.button(
        "🔍  Check Authenticity",
        use_container_width=True,
        disabled=(uploaded is None),
        type="primary",
    ):
        with st.spinner("Analyzing note..."):
            st.session_state.image = uploaded.read()
            st.session_state.page = "result"
            st.rerun()

    st.markdown(
        """
        <div class="how-box">
            <div class="how-title">How it works</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns([1, 0.2, 1, 0.2, 1])
    with c1:
        st.markdown(
            """
            <div class="how-step">
                <div class="how-circle">⬆</div>
                <div class="how-step-title">1. Upload</div>
                <div class="how-step-desc">Upload image of<br>currency note</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown('<div class="how-arrow">→</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(
            """
            <div class="how-step">
                <div class="how-circle">🔍</div>
                <div class="how-step-title">2. Analyze</div>
                <div class="how-step-desc">Our AI model analyzes<br>security features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown('<div class="how-arrow">→</div>', unsafe_allow_html=True)
    with c5:
        st.markdown(
            """
            <div class="how-step">
                <div class="how-circle">✔</div>
                <div class="how-step-title">3. Result</div>
                <div class="how-step-desc">Get authenticity result<br>in seconds</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-banner">🛡️ We use advanced AI &amp; OCR to analyze security '
        'features, text and patterns of the currency note.</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# RESULT PAGE
# ----------------------------------------------------------------------
def result_page():
    local_css()
    top_nav()

    if st.button("←  Back to Home"):
        st.session_state.page = "upload"
        st.rerun()

    model = get_model()
    image_bytes = st.session_state.image

    with st.spinner("Running OCR & model prediction..."):
        img = preprocess_image(image_bytes)
        auth_idx, auth_conf, source = predict(model, img, image_bytes)

    label = AUTH_CLASSES[auth_idx]
    is_real = label == "real"
    css_state = "real" if is_real else "fake"
    icon = "✅" if is_real else "❌"
    verdict_text = "This note is REAL" if is_real else "This note is FAKE"
    desc_text = (
        "The currency note is authentic and matches the security features of genuine notes."
        if is_real
        else "The currency note shows signs of tampering and does not match the security "
        "features of genuine notes."
    )
    checked_on = datetime.now().strftime("%d %b %Y, %I:%M %p")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-badge {css_state}">
                <div class="badge-icon">{icon}</div>
                <div class="badge-label {css_state}">{label.upper()}</div>
                <div class="badge-conf">{auth_conf:.1%} Confidence</div>
            </div>
            <div>
                <div class="result-info-label">Authenticity Result</div>
                <div class="result-info-verdict {css_state}">{verdict_text}</div>
                <div class="result-info-desc">{desc_text}</div>
                <div class="result-timestamp">📅 Checked on: {checked_on}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Uploaded Image</div>', unsafe_allow_html=True)
    st.image(image_bytes, use_container_width=True)

    st.markdown('<div class="section-title">Analysis Summary</div>', unsafe_allow_html=True)

    # NOTE: the underlying model only returns a single authenticity
    # prediction + confidence + source. The individual checklist items
    # below are presented for UI purposes; wire them up to real
    # sub-scores if/when your pipeline exposes them.
    summary_items = [
        "OCR Text Check",
        "Watermark Detection",
        "Security Features",
        "Pattern Analysis",
        "Serial Number Check",
        "AI Model Prediction",
    ]
    cols = st.columns(2)
    for i, item in enumerate(summary_items):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="summary-item">
                    <div>
                        <div class="label">{item}</div>
                        <div class="status">{"Passed" if is_real else "Flagged"}</div>
                    </div>
                    <div class="check">{"✔" if is_real else "✖"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    banner_text = (
        "🛡️ All security features are valid. This note is authentic."
        if is_real
        else "🛡️ One or more security features failed validation. This note may be counterfeit."
    )
    st.markdown(f'<div class="info-banner">{banner_text}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="upload-hint">Source: {source}</p>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.download_button(
            "⬇  Download Report",
            data=f"AuthentiNote Report\nResult: {label.upper()}\nConfidence: {auth_conf:.1%}\n"
            f"Checked on: {checked_on}\nSource: {source}\n",
            file_name="authentinote_report.txt",
            use_container_width=True,
            type="primary",
        )
    with b2:
        if st.button("🔄  Check Another", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()


# ----------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "upload"

if st.session_state.page == "upload":
    upload_page()
else:
    result_page()