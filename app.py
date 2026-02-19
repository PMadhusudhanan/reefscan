"""
ReefScan · Marine Intelligence Dashboard
========================================
A professional Streamlit dashboard for coral reef health analysis.
Backend detection logic is unchanged from original notebook.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import streamlit.components.v1 as components

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReefScan · Marine Intelligence",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0F1E;
    color: #E2EEF9;
}

/* ── Hide default chrome ── */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ── Main content padding ── */
.block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px;
}

/* ── HERO HEADER ── */
.hero-header {
    background: linear-gradient(135deg, #0B1E3D 0%, #0D2B50 40%, #0A3060 70%, #062040 100%);
    border-radius: 0 0 28px 28px;
    padding: 2.8rem 3rem 2.4rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid rgba(0, 180, 255, 0.15);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: "";
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 90% 20%, rgba(0,180,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 40% 60% at 10% 80%, rgba(0,100,200,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.22em;
    color: #00B4FF; text-transform: uppercase;
    margin-bottom: 0.7rem; opacity: 0.9;
}
.hero-title {
    font-size: 2.2rem; font-weight: 800;
    color: #F0F8FF; margin: 0 0 0.5rem;
    letter-spacing: -0.03em; line-height: 1.15;
}
.hero-title span { color: #00D4FF; }
.hero-subtitle {
    font-size: 0.9rem; color: #5E8CAA;
    font-weight: 400; max-width: 520px; line-height: 1.6;
}
.hero-pills {
    display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 1.4rem;
}
.pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.1em;
    padding: 4px 12px; border-radius: 100px; font-weight: 500;
    border: 1px solid; text-transform: uppercase;
}
.pill-blue   { color: #00D4FF; border-color: rgba(0,212,255,0.35); background: rgba(0,212,255,0.07); }
.pill-green  { color: #00E59B; border-color: rgba(0,229,155,0.35); background: rgba(0,229,155,0.07); }
.pill-orange { color: #FFB74D; border-color: rgba(255,183,77,0.35); background: rgba(255,183,77,0.07); }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #0D1525 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem; }

.sidebar-logo {
    display: flex; align-items: center; gap: 0.6rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.4rem;
}
.sidebar-logo-icon {
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, #0069AA, #00D4FF);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; box-shadow: 0 0 14px rgba(0,212,255,0.3);
}
.sidebar-brand { font-size: 1rem; font-weight: 800; color: #E2EEF9; letter-spacing: -0.02em; }
.sidebar-brand span { color: #00D4FF; }
.sidebar-version {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem; color: #2D5A78; letter-spacing: 0.1em;
}

.sidebar-section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.18em;
    color: #2D5A78; text-transform: uppercase;
    margin: 1.2rem 0 0.6rem;
}

.info-block {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-top: 0.8rem;
}
.info-block-title { font-size: 0.72rem; font-weight: 600; color: #00D4FF; margin-bottom: 0.4rem; }
.info-block p { font-size: 0.72rem; color: #4A7A96; line-height: 1.6; margin: 0; }

.stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.stat-row:last-child { border-bottom: none; }
.stat-label { font-size: 0.71rem; color: #3A6580; }
.stat-value { font-family: 'JetBrains Mono', monospace; font-size: 0.71rem; color: #E2EEF9; font-weight: 600; }

/* ── STATUS ALERT ── */
.alert {
    display: flex; align-items: flex-start; gap: 1rem;
    padding: 1.1rem 1.4rem; border-radius: 14px;
    margin-bottom: 1.6rem; border: 1px solid;
}
.alert-icon { font-size: 1.5rem; line-height: 1; flex-shrink: 0; }
.alert-title { font-size: 0.95rem; font-weight: 700; margin-bottom: 2px; }
.alert-desc  { font-size: 0.78rem; opacity: 0.7; }
.alert-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem; font-weight: 700;
    margin-left: auto; align-self: center; flex-shrink: 0;
}
.alert-good    { background: rgba(0,229,155,0.07); border-color: rgba(0,229,155,0.25); color: #00E59B; }
.alert-warning { background: rgba(255,183,77,0.07); border-color: rgba(255,183,77,0.25); color: #FFB74D; }
.alert-danger  { background: rgba(255,80,80,0.08);  border-color: rgba(255,80,80,0.28);  color: #FF6B6B; }

/* ── KPI CARDS ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.8rem;
}
.kpi-card {
    border-radius: 16px;
    padding: 1.5rem 1.5rem 1.3rem;
    border: 1px solid rgba(255,255,255,0.07);
    position: relative; overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.45);
}
.kpi-stripe {
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px; border-radius: 16px 16px 0 0;
}
.kpi-glow {
    position: absolute; bottom: -25px; right: -25px;
    width: 90px; height: 90px; border-radius: 50%; opacity: 0.06;
}
.kpi-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1rem;
}
.kpi-icon-wrap {
    width: 38px; height: 38px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.kpi-trend {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; padding: 2px 7px; border-radius: 100px;
    font-weight: 600; border: 1px solid;
}
.kpi-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-value {
    font-size: 3rem; font-weight: 800; line-height: 1;
    letter-spacing: -0.03em; margin-bottom: 0.8rem;
}
.kpi-value sup { font-size: 1.3rem; font-weight: 500; opacity: 0.8; }
.kpi-progress-track {
    height: 5px; border-radius: 3px;
    background: rgba(255,255,255,0.06); overflow: hidden;
    margin-bottom: 0.55rem;
}
.kpi-progress-fill { height: 100%; border-radius: 3px; }
.kpi-footer { font-size: 0.68rem; opacity: 0.45; letter-spacing: 0.01em; }

/* bleaching card */
.kc-bleach { background: linear-gradient(150deg, #111C2E, #162436); }
.kc-bleach .kpi-label  { color: #90CAF9; }
.kc-bleach .kpi-value  { color: #C5DEF8; }
.kc-bleach .kpi-stripe { background: linear-gradient(90deg, #546E7A, #90CAF9); }
.kc-bleach .kpi-glow   { background: #90CAF9; }
.kc-bleach .kpi-icon-wrap { background: rgba(144,202,249,0.1); }
.kc-bleach .kpi-trend  { color: #90CAF9; border-color: rgba(144,202,249,0.3); background: rgba(144,202,249,0.07); }
.kc-bleach .kpi-progress-fill { background: linear-gradient(90deg, #546E7A, #90CAF9); }

/* algae card */
.kc-algae { background: linear-gradient(150deg, #0D1E14, #102418); }
.kc-algae .kpi-label  { color: #69F0AE; }
.kc-algae .kpi-value  { color: #A0F5CF; }
.kc-algae .kpi-stripe { background: linear-gradient(90deg, #1B5E35, #69F0AE); }
.kc-algae .kpi-glow   { background: #69F0AE; }
.kc-algae .kpi-icon-wrap { background: rgba(105,240,174,0.1); }
.kc-algae .kpi-trend  { color: #69F0AE; border-color: rgba(105,240,174,0.3); background: rgba(105,240,174,0.07); }
.kc-algae .kpi-progress-fill { background: linear-gradient(90deg, #1B5E35, #69F0AE); }

/* sediment card */
.kc-sediment { background: linear-gradient(150deg, #1A1408, #221A0C); }
.kc-sediment .kpi-label  { color: #FFCC80; }
.kc-sediment .kpi-value  { color: #FFE0A0; }
.kc-sediment .kpi-stripe { background: linear-gradient(90deg, #7C5210, #FFCC80); }
.kc-sediment .kpi-glow   { background: #FFCC80; }
.kc-sediment .kpi-icon-wrap { background: rgba(255,204,128,0.1); }
.kc-sediment .kpi-trend  { color: #FFCC80; border-color: rgba(255,204,128,0.3); background: rgba(255,204,128,0.07); }
.kc-sediment .kpi-progress-fill { background: linear-gradient(90deg, #7C5210, #FFCC80); }

/* health card */
.kc-health { background: linear-gradient(150deg, #071826, #09203A); }
.kc-health .kpi-label  { color: #00D4FF; }
.kc-health .kpi-value  { color: #60E8FF; }
.kc-health .kpi-stripe { background: linear-gradient(90deg, #005B7F, #00D4FF); }
.kc-health .kpi-glow   { background: #00D4FF; }
.kc-health .kpi-icon-wrap { background: rgba(0,212,255,0.1); }
.kc-health .kpi-trend  { color: #00D4FF; border-color: rgba(0,212,255,0.3); background: rgba(0,212,255,0.07); }
.kc-health .kpi-progress-fill { background: linear-gradient(90deg, #005B7F, #00D4FF); }

/* ── CHART PANEL ── */
.chart-panel {
    background: #0D1525;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    box-shadow: 0 6px 28px rgba(0,0,0,0.35);
    margin-bottom: 1.5rem;
}
.chart-panel-header {
    padding: 1rem 1.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex; align-items: center; justify-content: space-between;
}
.chart-panel-title {
    font-size: 0.85rem; font-weight: 700; color: #E2EEF9;
    display: flex; align-items: center; gap: 0.5rem;
}
.chart-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 3px 9px; border-radius: 100px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08); color: #3A6580;
}
.chart-panel-footer {
    padding: 0.65rem 1.4rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; color: #2D5A78;
}

/* ── INFO ICON + TOOLTIP ── */
.info-wrap {
    position: relative; display: inline-flex;
    align-items: center; margin-left: 0.5rem;
}
.info-icon {
    width: 20px; height: 20px; border-radius: 50%;
    background: rgba(0,212,255,0.10);
    border: 1px solid rgba(0,212,255,0.30);
    color: #00D4FF; font-size: 0.65rem; font-weight: 700;
    display: inline-flex; align-items: center; justify-content: center;
    cursor: pointer; font-family: 'JetBrains Mono', monospace;
    transition: background 0.2s, border-color 0.2s;
    flex-shrink: 0; user-select: none;
}
.info-icon:hover, .info-icon.active {
    background: rgba(0,212,255,0.22);
    border-color: rgba(0,212,255,0.65);
}
.info-tooltip {
    display: none;
    position: absolute; bottom: calc(100% + 10px); right: 0;
    width: 280px;
    background: #0A1830;
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem; font-weight: 400;
    color: #8FBDD3; line-height: 1.6;
    box-shadow: 0 -8px 36px rgba(0,0,0,0.55);
    z-index: 99999;
    pointer-events: auto;
}
.info-tooltip.visible {
    display: block;
}
.info-tooltip::before {
    content: "";
    position: absolute; bottom: -6px; right: 8px;
    width: 10px; height: 10px;
    background: #0A1830;
    border-right: 1px solid rgba(0,212,255,0.25);
    border-bottom: 1px solid rgba(0,212,255,0.25);
    transform: rotate(45deg);
}
.info-tooltip strong { color: #00D4FF; display: block; margin-bottom: 0.3rem; font-size: 0.78rem; }

/* ── IMAGE PANEL ── */
.img-panel {
    background: #0D1525;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px; overflow: hidden;
    box-shadow: 0 6px 28px rgba(0,0,0,0.35);
    height: 100%;
}
.img-panel-header {
    padding: 1rem 1.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex; align-items: center; justify-content: space-between;
}
.img-panel-title { font-size: 0.85rem; font-weight: 700; color: #E2EEF9; }
.img-panel-footer {
    padding: 0.65rem 1.4rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; color: #2D5A78;
}

/* ── SECTION HEADING ── */
.section-heading {
    display: flex; align-items: center; gap: 0.8rem;
    margin: 2rem 0 1rem;
}
.section-heading-text {
    font-size: 0.65rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.2em; text-transform: uppercase; color: #2D5A78;
    white-space: nowrap;
}
.section-heading-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,180,255,0.15), transparent);
}

/* ── FOOTER ── */
.dash-footer {
    margin-top: 3rem; padding: 1.5rem 0;
    border-top: 1px solid rgba(255,255,255,0.05);
    display: flex; align-items: center; justify-content: space-between;
}
.dash-footer-left { font-size: 0.72rem; color: #2D5A78; }
.dash-footer-left strong { color: #3A7A96; }
.dash-footer-right {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; color: #1E3D50; letter-spacing: 0.1em;
}

/* ── EMPTY STATE ── */
.empty-state {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; text-align: center;
    padding: 5rem 2rem;
    background: #0D1525;
    border: 1px dashed rgba(0,180,255,0.15);
    border-radius: 20px; margin-top: 2rem;
}
.empty-icon {
    width: 72px; height: 72px; border-radius: 20px;
    background: linear-gradient(135deg, #062040, #0D3060);
    border: 1px solid rgba(0,180,255,0.2);
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; margin-bottom: 1.4rem;
    box-shadow: 0 0 32px rgba(0,180,255,0.12);
}
.empty-title { font-size: 1.1rem; font-weight: 700; color: #2D5A78; margin-bottom: 0.5rem; }
.empty-desc  { font-size: 0.82rem; color: #1E3D50; max-width: 360px; line-height: 1.7; }

/* ── Streamlit upload widget override ── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(0,180,255,0.03) !important;
    border: 1.5px dashed rgba(0,180,255,0.25) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(0,180,255,0.5) !important;
}
/* sidebar scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1A3550; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Inject tooltip JS via components (st.markdown strips <script> tags) ───────
components.html("""
<script>
(function() {
  // Target the parent Streamlit document from this component iframe
  var doc = window.parent.document;

  function initTooltips() {
    doc.querySelectorAll('.info-icon').forEach(function(icon) {
      if (icon.dataset.tipBound) return;
      icon.dataset.tipBound = '1';
      icon.addEventListener('click', function(e) {
        e.stopPropagation();
        var wrap = icon.closest('.info-wrap');
        var tip  = wrap ? wrap.querySelector('.info-tooltip') : null;
        if (!tip) return;
        var isOpen = tip.classList.contains('visible');
        doc.querySelectorAll('.info-tooltip.visible').forEach(function(t){ t.classList.remove('visible'); });
        doc.querySelectorAll('.info-icon.active').forEach(function(i){ i.classList.remove('active'); });
        if (!isOpen) { tip.classList.add('visible'); icon.classList.add('active'); }
      });
    });
    doc.addEventListener('click', function() {
      doc.querySelectorAll('.info-tooltip.visible').forEach(function(t){ t.classList.remove('visible'); });
      doc.querySelectorAll('.info-icon.active').forEach(function(i){ i.classList.remove('active'); });
    }, { once: false });
  }

  // Re-bind whenever Streamlit re-renders the DOM
  var obs = new MutationObserver(function() { initTooltips(); });
  obs.observe(doc.body, { childList: true, subtree: true });
  initTooltips();
})();
</script>
""", height=0)


# ──────────────────────────────────────────────────────────────────────────────
# ── BACKEND LOGIC (UNCHANGED FROM ORIGINAL NOTEBOOK) ─────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def analyze_image(pil_image):
    image = np.array(pil_image.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (500, 500))
    img   = image / 255.0

    total_pixels = img.shape[0] * img.shape[1]

    # Bleaching = very bright (white) pixels
    bleach_mask = np.logical_and.reduce((
        img[:, :, 0] > 0.8,
        img[:, :, 1] > 0.8,
        img[:, :, 2] > 0.8,
    ))

    # Algae bloom = dominant green
    algae_mask = np.logical_and(
        img[:, :, 1] > 0.45,
        img[:, :, 1] > img[:, :, 0],
    )

    # Sediment = brownish (red dominant, low blue)
    sediment_mask = np.logical_and(
        img[:, :, 0] > 0.4,
        img[:, :, 2] < 0.3,
    )

    bleach_percent   = (np.sum(bleach_mask)   / total_pixels) * 100
    algae_percent    = (np.sum(algae_mask)    / total_pixels) * 100
    sediment_percent = (np.sum(sediment_mask) / total_pixels) * 100
    healthy_percent  = max(0, 100 - (bleach_percent + algae_percent + sediment_percent))

    return {
        "bleach":   round(bleach_percent,   2),
        "algae":    round(algae_percent,    2),
        "sediment": round(sediment_percent, 2),
        "health":   round(healthy_percent,  2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# ── CHART FUNCTIONS ───────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
CHART_BG   = "#0D1525"
C_BLEACH   = "#90CAF9"
C_ALGAE    = "#69F0AE"
C_SEDIMENT = "#FFCC80"
C_HEALTH   = "#00D4FF"


def _savefig(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                facecolor=CHART_BG, edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf


def chart_bar(R: dict) -> io.BytesIO:
    """Vertical bar chart -- pollution & health breakdown."""
    labels = ["Coral\nBleaching", "Algae\nBloom", "Sediment", "Marine\nHealth"]
    values = [R["bleach"], R["algae"], R["sediment"], R["health"]]
    colors = [C_BLEACH, C_ALGAE, C_SEDIMENT, C_HEALTH]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    x = np.arange(len(labels))
    width = 0.52

    # grey 100% background track
    ax.bar(x, [100] * 4, width=width, color="#FFFFFF", alpha=0.04, zorder=1)

    # coloured value bars
    bars = ax.bar(x, values, width=width, color=colors, alpha=0.90, zorder=3)

    for bar, val, col in zip(bars, values, colors):
        # top-edge accent line
        ax.plot(
            [bar.get_x() + 0.04, bar.get_x() + bar.get_width() - 0.04],
            [val, val],
            color=col, linewidth=3, solid_capstyle="round", zorder=5, alpha=0.85,
        )
        # value label above bar
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            f"{val:.2f}%",
            ha="center", va="bottom",
            fontsize=10, fontweight="700",
            color=col, fontfamily="monospace",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5, color="#B0CCDE", fontweight="600")
    ax.set_ylim(0, 118)
    ax.set_ylabel("Coverage  (%)", color="#2D5A78", fontsize=9, labelpad=10)
    ax.set_title("Pollution & Health Coverage Analysis",
                 color="#C5DEF8", fontsize=11.5, fontweight="bold",
                 pad=16, loc="left")
    ax.tick_params(axis="x", colors="#2D5A78", labelsize=9, length=0)
    ax.tick_params(axis="y", colors="#2D5A78", labelsize=8.5)
    ax.grid(axis="y", color="#FFFFFF", alpha=0.04, linestyle="--", linewidth=0.8)

    # Marine Health Score annotation box
    score = R["health"]
    ax.text(
        0.985, 0.97,
        f"Marine Health Score\n{score:.2f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="#00D4FF", fontfamily="monospace",
        bbox=dict(facecolor=(0.024, 0.118, 0.188, 0.85),
                  edgecolor=(0.0, 0.83, 1.0, 0.4),
                  boxstyle="round,pad=0.5", linewidth=1),
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=1.6)
    return _savefig(fig)


def chart_donut(R: dict) -> io.BytesIO:
    """Donut chart — ecosystem composition breakdown."""
    labels = ["Bleaching", "Algae", "Sediment", "Healthy Water"]
    values = [R["bleach"], R["algae"], R["sediment"], R["health"]]
    colors = [C_BLEACH, C_ALGAE, C_SEDIMENT, C_HEALTH]
    safe   = [max(v, 0.01) for v in values]

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    wedges, _, autotexts = ax.pie(
        safe, colors=colors, startangle=90,
        autopct="%1.1f%%", pctdistance=0.76,
        wedgeprops=dict(width=0.52, edgecolor=CHART_BG, linewidth=2.5),
    )
    for at in autotexts:
        at.set_color("#D6EAF8"); at.set_fontsize(8); at.set_fontweight("600")

    # centre text
    ax.text(0,  0.13, f"{R['health']:.1f}%",
            ha="center", va="center", fontsize=21, fontweight="800",
            color="#00D4FF", fontfamily="monospace")
    ax.text(0, -0.10, "HEALTH INDEX",
            ha="center", va="center", fontsize=6.5,
            color="#2D5A78", fontfamily="monospace", fontstyle="normal")

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(colors, labels)]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False,
              fontsize=7.5, labelcolor="#4A7A96")
    ax.set_title("Ecosystem Composition", color="#C5DEF8",
                 fontsize=10.5, fontweight="bold", pad=10)
    plt.tight_layout(pad=1.2)
    return _savefig(fig)


def chart_gauge(R: dict) -> io.BytesIO:
    """Semi-circular health gauge — Critical / Poor / Fair / Good."""
    score = R["health"]

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.set_aspect("equal"); ax.axis("off")

    R_OUT, R_IN = 1.0, 0.60
    zones = [
        (180, 135, "#C0392B", "Critical"),
        (135,  90, "#E67E22", "Poor"),
        ( 90,  45, "#F4D03F", "Fair"),
        ( 45,   0, "#27AE60", "Good"),
    ]

    for a0, a1, col, lbl in zones:
        th = np.linspace(np.radians(a0), np.radians(a1), 120)
        xo, yo = R_OUT * np.cos(th), R_OUT * np.sin(th)
        xi, yi = R_IN  * np.cos(th), R_IN  * np.sin(th)
        ax.fill(np.concatenate([xo, xi[::-1]]),
                np.concatenate([yo, yi[::-1]]),
                color=col, alpha=0.88, zorder=2, linewidth=0)
        mid = np.radians((a0 + a1) / 2)
        rm  = (R_IN + R_OUT) / 2
        ax.text(rm * np.cos(mid), rm * np.sin(mid), lbl,
                ha="center", va="center", fontsize=6.2,
                color="white", fontweight="bold",
                rotation=(a0 + a1) / 2 - 90, zorder=5)

    # divider gaps
    for deg in [135, 90, 45]:
        rd = np.radians(deg)
        ax.plot([R_IN * np.cos(rd), R_OUT * np.cos(rd)],
                [R_IN * np.sin(rd), R_OUT * np.sin(rd)],
                color=CHART_BG, linewidth=3.5, zorder=4)

    # inner fill
    th2 = np.linspace(0, np.pi, 200)
    ax.fill(np.concatenate([R_IN * np.cos(th2), [0]]),
            np.concatenate([R_IN * np.sin(th2), [0]]),
            color="#0D1525", zorder=3)

    # tick labels
    for deg, lbl in [(180, "0"), (135, "25"), (90, "50"), (45, "75"), (0, "100")]:
        rd = np.radians(deg)
        tx = (R_OUT + 0.15) * np.cos(rd)
        ty = (R_OUT + 0.15) * np.sin(rd)
        ha = "right" if deg > 91 else ("left" if deg < 89 else "center")
        ax.text(tx, ty, lbl, ha=ha, va="center",
                fontsize=6.5, color="#2D5A78", fontfamily="monospace")

    # needle
    needle_deg = 180 - (score / 100) * 180
    nd = np.radians(needle_deg)
    NL = 0.80
    ax.plot([0, NL * np.cos(nd)], [0, NL * np.sin(nd)],
            color="#FFFFFF", linewidth=2.8, solid_capstyle="round", zorder=7)
    ax.plot(NL * np.cos(nd), NL * np.sin(nd),
            "o", color="#00D4FF", markersize=5.5, zorder=8)
    ax.plot(0, 0, "o", color="#00D4FF", markersize=13, zorder=9)
    ax.plot(0, 0, "o", color="#0D1525", markersize=8,  zorder=10)

    # score readout
    ax.text(0, 0.31, f"{score:.1f}%",
            ha="center", va="center",
            fontsize=18, fontweight="800",
            color="#00D4FF", fontfamily="monospace", zorder=11)
    ax.text(0, 0.12, "Marine Health Score",
            ha="center", va="center",
            fontsize=6.2, color="#2D5A78",
            fontfamily="monospace", zorder=11)

    ax.set_xlim(-1.40, 1.40)
    ax.set_ylim(-0.20, 1.32)
    ax.set_title("Health Gauge", color="#C5DEF8",
                 fontsize=10.5, fontweight="bold", pad=8)
    plt.tight_layout(pad=0.5)
    return _savefig(fig)


def chart_recovery(R: dict) -> io.BytesIO:
    """Marine Recovery Simulation — logistic growth curves."""
    months = np.linspace(0, 50, 300)
    boost  = R["health"] / 100.0

    def logi(t, L, k, t0):
        return L / (1 + np.exp(-k * (t - t0)))

    eco = {
        "Coral Reef":      (dict(L=0.88, k=0.18, t0=8),  dict(L=0.98, k=0.22, t0=6),  "#90CAF9", "#42A5F5"),
        "Seagrass Meadow": (dict(L=0.80, k=0.13, t0=12), dict(L=0.95, k=0.17, t0=9),  "#A5D6A7", "#66BB6A"),
        "Mangrove Forest": (dict(L=0.72, k=0.10, t0=16),
                            dict(L=min(0.88 + boost * 0.5, 1.0), k=0.13, t0=12),
                            "#CE93D8", "#AB47BC"),
    }

    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    legend_lines, legend_labels = [], []
    for name, (bp, ip, cb, ci) in eco.items():
        yb = logi(months, **bp)
        yi = np.clip(logi(months, **ip) * (1 + boost * 0.15), 0, 1)
        lb, = ax.plot(months, yb, "--", lw=1.8, color=cb, alpha=0.75)
        li, = ax.plot(months, yi, "-",  lw=2.5, color=ci)
        ax.fill_between(months, yb, yi, color=ci, alpha=0.07)
        legend_lines  += [lb, li]
        legend_labels += [f"{name} — Baseline", f"{name} — Improved"]

    ax.axhline(1.0, color="#FFFFFF", alpha=0.06, lw=1, ls=":")
    ax.text(49, 1.013, "Full Recovery",
            color="#2D5A78", fontsize=7.5, ha="right", fontfamily="monospace")

    ax.text(
        0.015, 0.97,
        f"  Health Boost  +{R['health']:.0f}%  ",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8.5, color="#00D4FF", fontfamily="monospace",
        bbox=dict(facecolor=(0.024, 0.118, 0.188, 0.85),
                  edgecolor=(0.0, 0.83, 1.0, 0.4),
                  boxstyle="round,pad=0.45", linewidth=1),
    )

    ax.legend(legend_lines, legend_labels,
              loc="lower right", ncol=1, frameon=True,
              framealpha=0.10, edgecolor=(1.0, 1.0, 1.0, 0.07),
              facecolor="#080E17", fontsize=8, labelcolor="#4A7A96")

    ax.set_xlim(0, 50); ax.set_ylim(0, 1.10)
    ax.set_xlabel("Time (Months)",  color="#2D5A78", fontsize=9.5, labelpad=8)
    ax.set_ylabel("Recovery Level", color="#2D5A78", fontsize=9.5, labelpad=8)
    ax.set_title(f"Marine Recovery Simulation  ·  Boost: {R['health']:.0f}%",
                 color="#C5DEF8", fontsize=11.5, fontweight="bold", pad=16, loc="left")
    ax.tick_params(axis="x", colors="#2D5A78", labelsize=8.5)
    ax.tick_params(axis="y", colors="#2D5A78", labelsize=8.5)
    ax.grid(color="#FFFFFF", alpha=0.04, linestyle="--", linewidth=0.8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor((1.0, 1.0, 1.0, 0.07))
        ax.spines[spine].set_linewidth(0.8)

    plt.tight_layout(pad=1.6)
    return _savefig(fig)


# ──────────────────────────────────────────────────────────────────────────────
# ── STATUS HELPER ─────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def get_status(score):
    if score >= 65:
        return "alert-good",    "✅", "Good Health",
    elif score >= 35:
        return "alert-warning", "⚠️", "Moderate Stress"
    else:
        return "alert-danger",  "🚨", "Critical Condition"


def get_status_desc(score):
    if score >= 65:
        return "Ecosystem conditions are stable. Reef biodiversity appears well-maintained."
    elif score >= 35:
        return "Ecosystem shows signs of environmental stress. Monitoring recommended."
    else:
        return "Severe degradation detected. Immediate intervention may be required."


# ──────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-icon">🪸</div>
      <div>
        <div class="sidebar-brand">Reef<span>Scan</span></div>
        <div class="sidebar-version">v4.0 · MARINE INTELLIGENCE</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">📂 Upload Images (up to 5)</div>',
                unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Reef images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload up to 5 underwater reef photos for spectral analysis",
    )
    if uploaded_files and len(uploaded_files) > 5:
        st.warning("⚠️ Maximum 5 images at a time. Only the first 5 will be analysed.")
        uploaded_files = uploaded_files[:5]

    st.markdown('<div class="sidebar-section-title">ℹ️ Detection Method</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block">
      <div class="info-block-title">Spectral Pixel Analysis</div>
      <p>Each pixel is classified by its RGB channel ratios using threshold-based rules derived from marine biology research.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">🎯 Thresholds</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 0.2rem 0;">
      <div class="stat-row">
        <span class="stat-label">🪸 Bleaching</span>
        <span class="stat-value">R,G,B > 0.80</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">🌿 Algae</span>
        <span class="stat-value">G > 0.45</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">🟫 Sediment</span>
        <span class="stat-value">R > 0.40, B < 0.30</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">🖼 Resize</span>
        <span class="stat-value">500 × 500 px</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">📊 Output Charts</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 0.2rem 0;">
      <div class="stat-row"><span class="stat-label">Bar Chart</span><span class="stat-value">Per image</span></div>
      <div class="stat-row"><span class="stat-label">Donut Chart</span><span class="stat-value">Composition</span></div>
      <div class="stat-row"><span class="stat-label">Health Gauge</span><span class="stat-value">Score</span></div>
      <div class="stat-row"><span class="stat-label">Recovery Sim</span><span class="stat-value">50-month</span></div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ── HERO HEADER ───────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-eyebrow">🌊 &nbsp; Environmental AI Platform</div>
  <h1 class="hero-title">Coral Reef <span>Health Monitor</span></h1>
  <p class="hero-subtitle">
    Upload an underwater reef photograph to instantly detect coral bleaching,
    algae blooms, and sediment levels using spectral pixel analysis — and
    generate a complete ecosystem health report.
  </p>
  <div class="hero-pills">
    <span class="pill pill-blue">Spectral Analysis</span>
    <span class="pill pill-green">Algae Detection</span>
    <span class="pill pill-orange">Sediment Mapping</span>
    <span class="pill pill-blue">Recovery Simulation</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ── MAIN CONTENT ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def section_head(label: str):
    st.markdown(f"""
    <div class="section-heading">
      <span class="section-heading-text">{label}</span>
      <div class="section-heading-line"></div>
    </div>""", unsafe_allow_html=True)



# ── Info icon tooltip helper ──────────────────────────────────────────────
def info_icon(title: str, body: str) -> str:
    """Return HTML for the i icon with a hover tooltip."""
    return (
        '<div class="info-wrap">'
        '<div class="info-icon">i</div>'
        '<div class="info-tooltip">'
        f'<strong>{title}</strong>'
        f'{body}'
        '</div></div>'
    )


# ── helper to render one full image report ────────────────────────────────
def render_report(uf, pil_image, R, idx):
    """Render the full analysis report for a single image."""

    # Status alert
    cls, icon, title = get_status(R["health"])
    desc = get_status_desc(R["health"])
    st.markdown(f"""
    <div class="alert {cls}">
      <div class="alert-icon">{icon}</div>
      <div>
        <div class="alert-title">{title}</div>
        <div class="alert-desc">{desc}</div>
      </div>
      <div class="alert-score">{R["health"]:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    section_head(f"01 · Detection Metrics")
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card kc-bleach">
        <div class="kpi-stripe"></div><div class="kpi-glow"></div>
        <div class="kpi-header"><div class="kpi-icon-wrap">🪸</div><div class="kpi-trend">Bleaching</div></div>
        <div class="kpi-label">Coral Bleaching</div>
        <div class="kpi-value">{R["bleach"]:.1f}<sup>%</sup></div>
        <div class="kpi-progress-track"><div class="kpi-progress-fill" style="width:{min(R["bleach"],100)}%"></div></div>
        <div class="kpi-footer">Bright / white pixel coverage</div>
      </div>
      <div class="kpi-card kc-algae">
        <div class="kpi-stripe"></div><div class="kpi-glow"></div>
        <div class="kpi-header"><div class="kpi-icon-wrap">🌿</div><div class="kpi-trend">Algae</div></div>
        <div class="kpi-label">Algae Bloom</div>
        <div class="kpi-value">{R["algae"]:.1f}<sup>%</sup></div>
        <div class="kpi-progress-track"><div class="kpi-progress-fill" style="width:{min(R["algae"],100)}%"></div></div>
        <div class="kpi-footer">Green-dominant pixel ratio</div>
      </div>
      <div class="kpi-card kc-sediment">
        <div class="kpi-stripe"></div><div class="kpi-glow"></div>
        <div class="kpi-header"><div class="kpi-icon-wrap">🟫</div><div class="kpi-trend">Sediment</div></div>
        <div class="kpi-label">Sediment Level</div>
        <div class="kpi-value">{R["sediment"]:.1f}<sup>%</sup></div>
        <div class="kpi-progress-track"><div class="kpi-progress-fill" style="width:{min(R["sediment"],100)}%"></div></div>
        <div class="kpi-footer">Turbid / brown-toned pixels</div>
      </div>
      <div class="kpi-card kc-health">
        <div class="kpi-stripe"></div><div class="kpi-glow"></div>
        <div class="kpi-header"><div class="kpi-icon-wrap">💧</div><div class="kpi-trend">Health</div></div>
        <div class="kpi-label">Marine Health Score</div>
        <div class="kpi-value">{R["health"]:.1f}<sup>%</sup></div>
        <div class="kpi-progress-track"><div class="kpi-progress-fill" style="width:{min(R["health"],100)}%"></div></div>
        <div class="kpi-footer">Overall ecosystem index</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Image + bar chart
    section_head("02 · Image & Coverage Analysis")
    col_img, col_bar = st.columns([1, 1.7], gap="large")
    with col_img:
        st.markdown(f"""
        <div class="img-panel">
          <div class="img-panel-header">
            <span class="img-panel-title">📷 &nbsp;Uploaded Image</span>
            <span class="chart-badge">Input</span>
          </div>""", unsafe_allow_html=True)
        st.image(pil_image, use_container_width=True)
        st.markdown(f"""
          <div class="img-panel-footer">
            <span>{uf.name}</span>
            <span>{pil_image.width} × {pil_image.height} px</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with col_bar:
        st.markdown("""
        <div class="chart-panel">
          <div class="chart-panel-header">
            <span class="chart-panel-title">📊 &nbsp;Pollution &amp; Health Coverage</span>
            <div style="display:flex;align-items:center;gap:0.5rem;">
              <span class="chart-badge">Bar Chart</span>
              <div class="info-wrap">
                <div class="info-icon">i</div>
                <div class="info-tooltip">
                  <strong>📊 Pollution &amp; Health Coverage</strong>
                  Shows what percentage of pixels in your image were classified into each category. <b style="color:#90CAF9">Blue</b> = Coral Bleaching (bright/white pixels), <b style="color:#69F0AE">Green</b> = Algae Bloom (green-dominant), <b style="color:#FFCC80">Amber</b> = Sediment (brown/turbid), <b style="color:#00D4FF">Cyan</b> = Marine Health. A tall cyan bar is a good sign.
                </div>
              </div>
            </div>
          </div>""", unsafe_allow_html=True)
        st.image(chart_bar(R), use_container_width=True)
        st.markdown("""
          <div class="chart-panel-footer">
            <span>Vertical bars show pixel coverage per category</span>
            <span>Health = 100% − (Bleach + Algae + Sediment)</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # Donut + gauge
    section_head("03 · Composition & Health Gauge")
    col_donut, col_gauge = st.columns([1, 1], gap="large")
    with col_donut:
        st.markdown("""
        <div class="chart-panel">
          <div class="chart-panel-header">
            <span class="chart-panel-title">🥧 &nbsp;Ecosystem Composition</span>
            <div style="display:flex;align-items:center;gap:0.5rem;">
              <span class="chart-badge">Donut</span>
              <div class="info-wrap">
                <div class="info-icon">i</div>
                <div class="info-tooltip">
                  <strong>🥧 Ecosystem Composition</strong>
                  Each slice shows the proportional share of the image per category. The centre number is your <b style="color:#00D4FF">Marine Health Score</b>. A large cyan slice = mostly healthy reef. Large white (Bleaching) or green (Algae) slices signal environmental stress that needs monitoring.
                </div>
              </div>
            </div>
          </div>""", unsafe_allow_html=True)
        st.image(chart_donut(R), use_container_width=True)
        st.markdown("""
          <div class="chart-panel-footer">
            <span>Proportional breakdown of all detected categories</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with col_gauge:
        st.markdown("""
        <div class="chart-panel">
          <div class="chart-panel-header">
            <span class="chart-panel-title">🎯 &nbsp;Health Gauge</span>
            <div style="display:flex;align-items:center;gap:0.5rem;">
              <span class="chart-badge">Semi-circular</span>
              <div class="info-wrap">
                <div class="info-icon">i</div>
                <div class="info-tooltip">
                  <strong>🎯 Health Gauge</strong>
                  The needle points to your Marine Health Score on a 0–100 arc. <b style="color:#C0392B">Red (0–25)</b> = Critical degradation. <b style="color:#E67E22">Orange (25–50)</b> = Poor. <b style="color:#F4D03F">Yellow (50–75)</b> = Fair. <b style="color:#27AE60">Green (75–100)</b> = Good. Aim to push the needle into the green zone.
                </div>
              </div>
            </div>
          </div>""", unsafe_allow_html=True)
        st.image(chart_gauge(R), use_container_width=True)
        st.markdown("""
          <div class="chart-panel-footer">
            <span style="color:#C0392B;font-weight:600;">● Critical 0–25</span>&nbsp;&nbsp;
            <span style="color:#E67E22;font-weight:600;">● Poor 25–50</span>&nbsp;&nbsp;
            <span style="color:#F4D03F;font-weight:600;">● Fair 50–75</span>&nbsp;&nbsp;
            <span style="color:#27AE60;font-weight:600;">● Good 75–100</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # Recovery simulation
    section_head("04 · Marine Recovery Simulation")
    st.markdown("""
    <div class="chart-panel">
      <div class="chart-panel-header">
            <span class="chart-panel-title">📈 &nbsp;Recovery Curves</span>
            <div style="display:flex;align-items:center;gap:0.5rem;">
              <span class="chart-badge">50-Month Projection</span>
              <div class="info-wrap">
                <div class="info-icon">i</div>
                <div class="info-tooltip">
                  <strong>📈 Marine Recovery Simulation</strong>
                  Projects ecosystem recovery over 50 months. <b style="color:#8FBDD3">Dashed lines</b> = natural baseline recovery with no intervention. <b style="color:#8FBDD3">Solid lines</b> = boosted recovery path based on your Health Score. A bigger gap between dashed and solid means greater potential benefit from active reef restoration.
                </div>
              </div>
            </div>
      </div>""", unsafe_allow_html=True)
    st.image(chart_recovery(R), use_container_width=True)
    st.markdown(f"""
      <div class="chart-panel-footer">
        <span>Dashed = Baseline &nbsp;|&nbsp; Solid = Improved with health boost</span>
        <span>Boost derived from Health Score: {R["health"]:.1f}%</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Raw data
    section_head("05 · Raw Detection Data")
    with st.expander("🔬 &nbsp;View full precision values"):
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🪸 Bleaching",          f'{R["bleach"]:.2f}%')
        c2.metric("🌿 Algae Bloom",         f'{R["algae"]:.2f}%')
        c3.metric("🟫 Sediment",            f'{R["sediment"]:.2f}%')
        c4.metric("💧 Marine Health Score",  f'{R["health"]:.2f}%')
        st.markdown("<br>", unsafe_allow_html=True)


# ── MAIN RENDER ──────────────────────────────────────────────────────────────
if uploaded_files:
    # ── Summary comparison bar (shown only when >1 image) ────────────────────
    if len(uploaded_files) > 1:
        section_head("🌐 · Multi-Image Comparison Overview")

        all_results = []
        for uf in uploaded_files:
            img = Image.open(uf)
            all_results.append((uf.name, analyze_image(img)))

        # Comparison table
        cols = st.columns(len(all_results))
        for col, (name, R) in zip(cols, all_results):
            cls, icon, title = get_status(R["health"])
            col.markdown(f"""
            <div class="chart-panel" style="text-align:center;padding:1rem 0.5rem;">
              <div style="font-size:1.6rem;margin-bottom:0.3rem;">{icon}</div>
              <div style="font-size:0.7rem;color:#6B93AF;font-family:monospace;
                          margin-bottom:0.5rem;white-space:nowrap;overflow:hidden;
                          text-overflow:ellipsis;">{name[:20]}</div>
              <div style="font-size:2rem;font-weight:800;color:#00D4FF;
                          font-family:monospace;line-height:1;">{R["health"]:.1f}<span style="font-size:0.9rem">%</span></div>
              <div style="font-size:0.65rem;color:#3A6580;margin-top:0.2rem;">Health Score</div>
              <div style="height:4px;background:rgba(255,255,255,0.05);
                          border-radius:2px;margin:0.7rem 0.5rem 0;overflow:hidden;">
                <div style="height:100%;width:{min(R["health"],100)}%;
                            background:linear-gradient(90deg,#005B7F,#00D4FF);
                            border-radius:2px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Comparison chart — all images side by side
        section_head("📊 · Side-by-Side Metric Comparison")
        names  = [r[0][:15] for r in all_results]
        bleach = [r[1]["bleach"]   for r in all_results]
        algae  = [r[1]["algae"]    for r in all_results]
        sedim  = [r[1]["sediment"] for r in all_results]
        health = [r[1]["health"]   for r in all_results]

        import numpy as np_cmp
        x = np_cmp.arange(len(names))
        w = 0.18

        fig_cmp, ax_cmp = plt.subplots(figsize=(max(8, len(names)*2.5), 5))
        fig_cmp.patch.set_facecolor(CHART_BG)
        ax_cmp.set_facecolor(CHART_BG)

        b1 = ax_cmp.bar(x - 1.5*w, bleach, w, color=C_BLEACH,  alpha=0.90, label="Bleaching")
        b2 = ax_cmp.bar(x - 0.5*w, algae,  w, color=C_ALGAE,   alpha=0.90, label="Algae")
        b3 = ax_cmp.bar(x + 0.5*w, sedim,  w, color=C_SEDIMENT,alpha=0.90, label="Sediment")
        b4 = ax_cmp.bar(x + 1.5*w, health, w, color=C_HEALTH,  alpha=0.90, label="Health")

        for bars, col in [(b1,C_BLEACH),(b2,C_ALGAE),(b3,C_SEDIMENT),(b4,C_HEALTH)]:
            for bar in bars:
                h = bar.get_height()
                ax_cmp.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                            f"{h:.1f}%", ha="center", va="bottom",
                            fontsize=7.5, fontweight="700",
                            color=col, fontfamily="monospace")

        ax_cmp.set_xticks(x)
        ax_cmp.set_xticklabels(names, fontsize=10, color="#B0CCDE", fontweight="600")
        ax_cmp.set_ylim(0, 118)
        ax_cmp.set_ylabel("Coverage (%)", color="#2D5A78", fontsize=9, labelpad=8)
        ax_cmp.set_title("All Images — Metric Comparison",
                         color="#C5DEF8", fontsize=11.5, fontweight="bold",
                         pad=14, loc="left")
        ax_cmp.tick_params(axis="x", length=0, labelsize=9)
        ax_cmp.tick_params(axis="y", colors="#2D5A78", labelsize=8.5)
        ax_cmp.grid(axis="y", color="#FFFFFF", alpha=0.04, linestyle="--", linewidth=0.8)
        ax_cmp.legend(frameon=False, fontsize=8.5, labelcolor="#6B93AF",
                      loc="upper right", ncol=4)
        for spine in ax_cmp.spines.values():
            spine.set_visible(False)
        plt.tight_layout(pad=1.4)

        cmp_buf = io.BytesIO()
        fig_cmp.savefig(cmp_buf, format="png", dpi=150, bbox_inches="tight",
                        facecolor=CHART_BG)
        cmp_buf.seek(0)
        plt.close(fig_cmp)

        st.markdown("""
        <div class="chart-panel">
          <div class="chart-panel-header">
                <span class="chart-panel-title">📊 &nbsp;Multi-Image Comparison</span>
            <div style="display:flex;align-items:center;gap:0.5rem;">
              <span class="chart-badge">Multi-image</span>
              <div class="info-wrap">
                <div class="info-icon">i</div>
                <div class="info-tooltip">
                  <strong>📊 Multi-Image Comparison</strong>
                  Each group of 4 bars = one uploaded image. Compares <b style="color:#90CAF9">Bleaching</b>, <b style="color:#69F0AE">Algae</b>, <b style="color:#FFCC80">Sediment</b>, and <b style="color:#00D4FF">Health</b> scores side by side across all images. Images with a taller cyan (Health) bar and shorter other bars are in better condition.
                </div>
              </div>
            </div>
          </div>""", unsafe_allow_html=True)
        st.image(cmp_buf, use_container_width=True)
        st.markdown("""
          <div class="chart-panel-footer">
            <span>Each group of 4 bars = one uploaded image</span>
            <span>Blue=Bleaching · Green=Algae · Amber=Sediment · Cyan=Health</span>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-image tabs ────────────────────────────────────────────────────────
    tab_labels = [f"🖼 {uf.name[:18]}" for uf in uploaded_files]
    tabs = st.tabs(tab_labels)

    for i, (tab, uf) in enumerate(zip(tabs, uploaded_files)):
        with tab:
            pil_image = Image.open(uf)
            R = analyze_image(pil_image)
            render_report(uf, pil_image, R, i)

else:
    # ── EMPTY STATE ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">🌊</div>
      <div class="empty-title">Awaiting Image Upload</div>
      <div class="empty-desc">
        Use the <strong>sidebar</strong> on the left to upload up to
        <strong>5 underwater reef photographs</strong> at once.<br><br>
        Each image gets a full analysis tab. When multiple images are uploaded,
        a side-by-side comparison overview is shown automatically.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ── FOOTER ────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-footer">
  <div class="dash-footer-left">
    🪸 &nbsp;<strong>ReefScan</strong> · Marine Intelligence Dashboard &nbsp;·&nbsp;
    Built with Streamlit &amp; OpenCV
  </div>
  <div class="dash-footer-right">
    SPECTRAL ANALYSIS ENGINE · v3.0
  </div>
</div>
""", unsafe_allow_html=True)
