"""
AI-Based Adaptive Noise Cancellation — SIH 2026 demo dashboard.

Run with:
    pip install -r requirements.txt
    streamlit run app.py

This file only handles layout and state. Audio I/O lives in audio_utils.py
and the filtering/detection logic lives in anc_engine.py, so the real
AI + FxLMS backend can be dropped in later without touching this file.
"""

import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

import config
from anc_engine import FilterError, anc_filter, detect_noise_type
from audio_utils import AudioError, autoplay_html, load_audio_file, load_or_create_demo, rms_db, to_wav_bytes

st.set_page_config(page_title=config.APP_TITLE, page_icon="🎧", layout="wide")


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{ background: linear-gradient(180deg, {config.COLORS['bg']} 0%, #0a1119 100%); }}
    html, body, [class*="css"] {{ font-family: 'Segoe UI', 'Inter', sans-serif; }}
    h1, h2, h3, h4, p, span, label, .stMarkdown, .stCaption {{ color: {config.COLORS['text']} !important; }}

    .card {{
        background: rgba(16, 28, 40, 0.7);
        border: 1px solid {config.COLORS['border']};
        border-radius: 10px;
        padding: 18px 20px;
        backdrop-filter: blur(6px);
        margin-bottom: 14px;
    }}
    .card-title {{
        font-size: 13px; font-weight: 700; letter-spacing: 1.2px;
        text-transform: uppercase; color: {config.COLORS['text_dim']}; margin-bottom: 10px;
    }}
    .metric-value {{ font-size: 26px; font-weight: 700; color: {config.COLORS['text']}; }}
    .metric-label {{ font-size: 12px; color: {config.COLORS['text_dim']}; text-transform: uppercase; letter-spacing: 0.8px; }}

    .status-badge {{
        display: inline-flex; align-items: center; gap: 8px;
        border: 1px solid currentColor; border-radius: 20px;
        padding: 6px 14px; font-weight: 700; font-size: 13px; letter-spacing: 0.5px;
    }}
    .status-dot {{
        width: 9px; height: 9px; border-radius: 50%; background: currentColor;
        box-shadow: 0 0 8px currentColor;
    }}
    .status-dot.pulse {{ animation: pulse 1.1s infinite; }}
    @keyframes pulse {{
        0% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} 100% {{ opacity: 1; }}
    }}

    .flow-row {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
    .flow-box {{
        background: {config.COLORS['panel_alt']}; border: 1px solid {config.COLORS['border']};
        border-radius: 6px; padding: 8px 12px; font-size: 12px; color: {config.COLORS['text_dim']};
        white-space: nowrap;
    }}
    .flow-arrow {{ color: {config.COLORS['text_dim']}; font-size: 14px; }}

    div.stButton > button {{
        background-color: #2563EB; color: #FFFFFF; font-weight: 700;
        border-radius: 8px; border: none; padding: 0.65em 1em; width: 100%;
    }}
    div.stButton > button:hover {{ opacity: 0.88; }}
    div[data-testid="stThemeSelector"] {{display: none !important;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
if "state" not in st.session_state:
    st.session_state.state = dict(config.DEFAULT_STATE)
if "anc_on" not in st.session_state:
    st.session_state.anc_on = True
if "ai_detection_on" not in st.session_state:
    st.session_state.ai_detection_on = True
if "suppression" not in st.session_state:
    st.session_state.suppression = 75
if "play_noise_flag" not in st.session_state:
    st.session_state.play_noise_flag = False
if "play_clean_flag" not in st.session_state:
    st.session_state.play_clean_flag = False

state = st.session_state.state

# ---------------------------------------------------------------------------
# Load audio (bundled demo, or a judge's own upload)
# ---------------------------------------------------------------------------
if state["input_audio"] is None:
    y, sr = load_or_create_demo(config.NOISY_PATH, config.SAMPLE_RATE, config.DEMO_DURATION_SEC)
    state["input_audio"] = y
    state["sample_rate"] = sr
    if st.session_state.ai_detection_on:
        label, conf = detect_noise_type(y, sr)
        state["noise_type"], state["confidence"] = label, conf
    state["noise_level_db"] = round(rms_db(y), 1)

with st.sidebar:
    st.markdown("#### Audio source")
    uploaded = st.file_uploader("Upload your own noisy clip", type=["wav", "mp3", "flac", "ogg"])
    uploaded_clean = st.file_uploader(" Output", type=["wav", "mp3", "flac", "ogg"])
    if uploaded is not None:
        try:
            y, sr = load_audio_file(uploaded)
            state["input_audio"] = y
            state["sample_rate"] = sr
            state["output_audio"] = None
            state["anc_status"] = config.STATUS_READY
            if st.session_state.ai_detection_on:
                state["noise_type"], state["confidence"] = detect_noise_type(y, sr)
            state["noise_level_db"] = round(rms_db(y), 1)
            st.success(f"Loaded: {uploaded.name}")
        except AudioError as exc:
            st.error(str(exc))

    st.markdown("---")
    st.markdown("#### ANC control")
    st.session_state.anc_on = st.toggle("ANC", value=st.session_state.anc_on)
    st.session_state.ai_detection_on = st.toggle("AI Detection", value=st.session_state.ai_detection_on)
    st.session_state.suppression = st.slider("Suppression strength", 0, 100, st.session_state.suppression)
    if st.button("Reset", use_container_width=True):
        st.session_state.state = dict(config.DEFAULT_STATE)
        st.session_state.play_noise_flag = False
        st.session_state.play_clean_flag = False
        st.rerun()

sr = state["sample_rate"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
status = state["anc_status"]
status_color = config.STATUS_COLORS.get(status, config.COLORS["cyan"])
pulse_class = "pulse" if status == config.STATUS_PROCESSING else ""

h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown(f"## {config.APP_TITLE}")
    st.markdown(
        f"<span style='color:{config.COLORS['text_dim']}'>{config.APP_SUBTITLE}</span> &nbsp;·&nbsp; "
        f"<span style='color:{config.COLORS['cyan']}; font-weight:600'>{config.APP_BADGE}</span>",
        unsafe_allow_html=True,
    )
with h_right:
    st.markdown(
        f"""<div style="text-align:right; margin-top:18px;">
              <span class="status-badge" style="color:{status_color}">
                <span class="status-dot {pulse_class}"></span> {status}
              </span>
            </div>""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Live ANC demonstration — the three primary controls
# ---------------------------------------------------------------------------
st.markdown('<div class="card-title">LIVE ANC DEMONSTRATION</div>', unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)
play_noise = b1.button("▶  PLAY NOISE", use_container_width=True)
apply_filter = b2.button("⚡  APPLY FILTER", use_container_width=True)
play_clean = b3.button("▶  PLAY CLEAN", use_container_width=True)

if play_noise:
    st.markdown(autoplay_html(state["input_audio"], sr), unsafe_allow_html=True)

if apply_filter:
    state["anc_status"] = config.STATUS_PROCESSING
    with st.spinner("Processing audio..."):
        try:
            if not st.session_state.anc_on:
                raise FilterError("ANC is switched off in the control panel — turn it on to filter.")
            result = anc_filter(
                state["input_audio"], sr,
                suppression=st.session_state.suppression / 100.0,
                ai_detection=st.session_state.ai_detection_on,
            )
            state.update(result)
            state["anc_status"] = config.STATUS_FILTERED
            sf.write(config.FILTERED_PATH, state["output_audio"], sr)
            st.success("✓ ANC Filtering Complete")
        except FilterError as exc:
            state["anc_status"] = config.STATUS_ERROR
            st.error(f"Filtering failed: {exc}")

if play_clean:
    if state["output_audio"] is None:
        st.warning("Apply the filter first to generate the clean audio.")
    else:
        st.markdown(autoplay_html(state["output_audio"], sr), unsafe_allow_html=True)
        if state["anc_status"] != config.STATUS_ERROR:
            state["anc_status"] = config.STATUS_ANC_ACTIVE

st.write("")

# ---------------------------------------------------------------------------
# Input / output cards
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"""<div class="card">
              <div class="card-title">NOISY INPUT</div>
            </div>""",
        unsafe_allow_html=True,
    )
    st.audio(to_wav_bytes(state["input_audio"], sr), format="audio/wav")
    m1, m2 = st.columns(2)
    m1.markdown(f"<div class='metric-label'>Noise Type</div><div class='metric-value' style='color:{config.COLORS['orange']}'>{state['noise_type']}</div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-label'>Noise Level</div><div class='metric-value'>{state['noise_level_db']:.1f} dB</div>", unsafe_allow_html=True)

with c2:
    st.markdown(
        f"""<div class="card">
              <div class="card-title">ANC OUTPUT</div>
            </div>""",
        unsafe_allow_html=True,
    )
    if state["output_audio"] is not None:
        st.audio(to_wav_bytes(state["output_audio"], sr), format="audio/wav")
    else:
        st.info("No filtered audio yet — press APPLY FILTER.")
    anc_label = "ACTIVE" if (state["output_audio"] is not None and st.session_state.anc_on) else "INACTIVE"
    m3, m4 = st.columns(2)
    m3.markdown(f"<div class='metric-label'>ANC</div><div class='metric-value' style='color:{config.COLORS['green'] if anc_label=='ACTIVE' else config.COLORS['text_dim']}'>{anc_label}</div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='metric-label'>Noise Reduction</div><div class='metric-value' style='color:{config.COLORS['cyan']}'>{state['noise_reduction_db']:.1f} dB</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------
def waveform_fig(y, sr, color, title):
    t = np.linspace(0, len(y) / sr, num=len(y))
    step = max(1, len(y) // 3000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t[::step], y=y[::step], mode="lines",
        line=dict(color=color, width=1.6),
        fill="tozeroy", fillcolor=color + "26",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=config.COLORS["text_dim"])),
        height=230, margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=config.COLORS["panel_alt"],
        font=dict(color=config.COLORS["text"]),
        xaxis=dict(title="seconds", gridcolor=config.COLORS["border"]),
        yaxis=dict(title="amplitude", range=[-1, 1], gridcolor=config.COLORS["border"]),
    )
    return fig


w1, w2 = st.columns(2)
with w1:
    st.plotly_chart(waveform_fig(state["input_audio"], sr, config.COLORS["orange"], "BEFORE FILTERING"), use_container_width=True)
with w2:
    if state["output_audio"] is not None:
        st.plotly_chart(waveform_fig(state["output_audio"], sr, config.COLORS["cyan"], "AFTER FILTERING"), use_container_width=True)
    else:
        st.markdown(
            f"""<div class="card" style="height:230px; display:flex; align-items:center; justify-content:center; text-align:center;">
                  <div>
                    <div style="font-size:14px; color:{config.COLORS['text_dim']}; font-weight:600;">AFTER FILTERING</div>
                    <div style="margin-top:8px; color:{config.COLORS['text_dim']};">No audio processed yet</div>
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# AI noise detection card
# ---------------------------------------------------------------------------
st.markdown('<div class="card-title">AI NOISE DETECTION</div>', unsafe_allow_html=True)
if st.session_state.ai_detection_on:
    d1, d2 = st.columns([2, 3])
    with d1:
        st.markdown(f"<div class='metric-label'>Detected Noise</div><div class='metric-value' style='color:{config.COLORS['orange']}'>{state['noise_type']}</div>", unsafe_allow_html=True)
    with d2:
        st.markdown(f"<div class='metric-label'>Confidence — {state['confidence']:.0f}%</div>", unsafe_allow_html=True)
        st.progress(min(max(state["confidence"] / 100, 0.0), 1.0))
else:
    st.info("AI Detection is switched off in the control panel.")

# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
st.markdown('<div class="card-title">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
p1, p2, p3, p4 = st.columns(4)
metrics = [
    (p1, "Noise Reduction", f"{state['noise_reduction_db']:.1f} dB", config.COLORS["cyan"]),
    (p2, "Speech Intelligibility", f"{state['speech_intelligibility']:.0f}%", config.COLORS["green"]),
    (p3, "Processing Latency", f"{state['latency_ms']:.0f} ms", config.COLORS["cyan"]),
    (p4, "Processing Load", f"{state['cpu_usage_pct']:.0f}%", config.COLORS["orange"]),
]
for col, label, value, color in metrics:
    col.markdown(
        f"""<div class="card" style="text-align:center;">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="color:{color}">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# System flow
# ---------------------------------------------------------------------------
st.markdown('<div class="card-title">SYSTEM FLOW</div>', unsafe_allow_html=True)
stages = ["MICROPHONE", "AI NOISE DETECTION", "FxLMS ANC", "SPEAKER", "ERROR FEEDBACK"]
flow_html = '<div class="flow-row">'
for i, s in enumerate(stages):
    flow_html += f'<div class="flow-box">{s}</div>'
    if i < len(stages) - 1:
        flow_html += '<span class="flow-arrow">→</span>'
flow_html += "</div>"
st.markdown(flow_html, unsafe_allow_html=True)

st.markdown(
    f"<div style='margin-top:24px; font-size:11px; color:{config.COLORS['text_dim']};'>"
    "This prototype runs an adaptive spectral-noise-suppression filter in place of the trained "
    "AI + FxLMS pipeline. Replace <code>anc_engine.anc_filter()</code> with the real model call "
    "to go from demo to production — the UI needs no changes.</div>",
    unsafe_allow_html=True,
)
