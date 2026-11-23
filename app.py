"""Streamlit frontend: Evidence-Based Fake News Detection using NLP + RAG."""
from __future__ import annotations

import html
import json
from string import Template

from src.sentiment import analyze_sentiment
from src.ner import extract_entities
from src.summarizer import summarize_article
from src.retriever import fetch_page_text
from src.utils import is_url
import streamlit as st
from src.ocr import extract_text_from_image
from src import database
from src import auth
from src.config import load_settings
from src.errors import ConfigError, FakeNewsError, InputError
from src.models import CONTRADICTED, INSUFFICIENT, SUPPORTED
from src.pipeline import analyze
from src.personalized_news import NEWS_CATEGORIES, clean_snippet, fetch_personalized_news
from src.topic_modeling import run_topic_modeling

st.set_page_config(page_title="Evidence-Based Fake News Detection", page_icon="🔍", layout="wide")

# --------------------------------------------------------------------------- theme
# The theme is a pure UI preference kept in st.session_state["theme"] ("light" / "dark").
# The toggle sits in the top-right corner of every page (including the login page).
PALETTES = {
    "light": dict(
        scheme="light", bg="#f4f6fa", surface="#ffffff", surface2="#f0f3f8", border="#e2e8f0",
        text="#0f172a", muted="#64748b", accent="#2563eb", accent_soft="#dbeafe", accent_text="#ffffff",
        ok="#15803d", bad="#b91c1c", warn="#a16207", pill_text="#ffffff",
        shadow="0 1px 2px rgba(15,23,42,0.06), 0 6px 18px rgba(15,23,42,0.06)"),
    "dark": dict(
        scheme="dark", bg="#0b1220", surface="#121c2e", surface2="#18243a", border="#263552",
        text="#e5ecf6", muted="#94a3b8", accent="#60a5fa", accent_soft="#1e3a5f", accent_text="#0b1220",
        ok="#34d399", bad="#f87171", warn="#fbbf24", pill_text="#0b1220",
        shadow="0 1px 2px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.35)"),
}

if "theme" not in st.session_state:
    try:
        _base = st.get_option("theme.base")  # respect a dark base theme from .streamlit/config.toml
    except Exception:
        _base = None
    st.session_state["theme"] = "dark" if _base == "dark" else "light"

THEME = st.session_state["theme"] if st.session_state["theme"] in PALETTES else "light"
P = PALETTES[THEME]

COLORS = {SUPPORTED: P["ok"], CONTRADICTED: P["bad"], INSUFFICIENT: P["warn"]}
NEUTRAL = P["muted"]
TOPIC_COLOR = P["accent"]
LOW_TOPIC_CONFIDENCE = 0.4
EXAMPLES = {
    "Debunked claim": "Humans only use 10 percent of their brains.",
    "Verifiable claim": "India's Chandrayaan-3 landed near the Moon's south pole in August 2023.",
    "Multi-claim text": (
        "The World Health Organization declared COVID-19 a global pandemic on 11 March 2020. "
        "Officials also said that 5G towers spread the coronavirus across Europe. "
        "Vitamin C cures the common cold in 24 hours, according to a viral post."),
}

CSS = Template("""
<style>
:root {
  --bg: $bg; --surface: $surface; --surface-2: $surface2; --border: $border; --text: $text;
  --muted: $muted; --accent: $accent; --accent-soft: $accent_soft; --accent-text: $accent_text;
  --ok: $ok; --bad: $bad; --warn: $warn; --pill-text: $pill_text; --shadow: $shadow;
  color-scheme: $scheme;
}

/* ---------- page canvas ---------- */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Inter, Roboto, "Helvetica Neue",
               "Noto Sans Devanagari", Arial, sans-serif;
}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
  background: var(--surface); border-right: 1px solid var(--border);
}
[data-testid="stSidebarCollapseButton"] svg, [data-testid="collapsedControl"] svg {color: var(--muted);}
.block-container {max-width: 1100px; padding-top: 1.4rem; padding-bottom: 3rem;}
#MainMenu, footer {visibility: hidden;}

/* ---------- typography ---------- */
.stApp h1 {font-weight: 700 !important; font-size: 1.85rem !important; letter-spacing: -0.015em;
           margin-bottom: 0.15rem !important; color: var(--text);}
.stApp h2, .stApp h3, .stApp h4 {color: var(--text); letter-spacing: -0.01em;}
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li, .stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stCheckbox"] p {color: var(--text);}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stImageCaption"] {color: var(--muted);}
.stApp [data-testid="stMarkdownContainer"] a {color: var(--accent); text-decoration: none; font-weight: 600;}
.stApp [data-testid="stMarkdownContainer"] a:hover {text-decoration: underline;}
.stApp hr {border-color: var(--border);}
.subtitle {color: var(--muted); font-size: 0.97rem; margin-bottom: 1.1rem;}
.section-label {font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
                color: var(--muted); margin: 1.3rem 0 0.5rem 0;}
.meta-line {font-size: 0.82rem; color: var(--muted);}
.hist-meta {font-size: 0.78rem; color: var(--muted);}

/* ---------- brand / account ---------- */
.brand {display: flex; align-items: center; gap: 10px; font-weight: 700; letter-spacing: 0.02em; color: var(--text);}
.brand-mark {width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center;
             justify-content: center; font-size: 18px; background: var(--accent-soft); border: 1px solid var(--border);}
.account {display: flex; align-items: center; gap: 12px; padding: 12px; border-radius: 12px;
          background: var(--surface-2); border: 1px solid var(--border); margin: 4px 0 10px 0;}
.avatar {width: 38px; height: 38px; border-radius: 50%; background: var(--accent); color: var(--accent-text);
         display: flex; align-items: center; justify-content: center; font-weight: 700; flex: none;}
.acct-email {font-size: 0.9rem; font-weight: 600; word-break: break-all; color: var(--text);}

/* ---------- cards ---------- */
.card {background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
       padding: 18px 22px; margin-bottom: 14px; box-shadow: var(--shadow);}
.card-kicker {font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: var(--muted);}
.card-title {font-size: 1.6rem; font-weight: 700; margin: 2px 0 6px 0;}
.verdict-card {background: var(--surface); border: 1px solid var(--border); border-left: 6px solid var(--c);
               border-radius: 14px; padding: 20px 24px; box-shadow: var(--shadow);}
.verdict-kicker {font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: var(--muted);}
.verdict-title {font-size: 1.8rem; font-weight: 700; color: var(--c); margin: 2px 0 8px 0;}
.verdict-summary {font-size: 0.97rem; color: var(--text); opacity: 0.92;}

.bar {height: 8px; border-radius: 999px; background: var(--surface-2); border: 1px solid var(--border);
      overflow: hidden; margin-top: 6px;}
.bar > div {height: 100%; background: var(--c); border-radius: 999px;}
.conf-row {display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--muted);}

.tile {background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px;
       text-align: center; box-shadow: var(--shadow);}
.tile-v {font-size: 1.7rem; font-weight: 700; line-height: 1.2;}
.tile-l {font-size: 0.72rem; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); margin-top: 2px;}

.pill {display: inline-block; padding: 3px 11px; border-radius: 999px; color: var(--pill-text); font-size: 0.78rem;
       font-weight: 700; letter-spacing: 0.02em;}
.claim-text {font-size: 1.04rem; font-weight: 600; margin: 4px 0 10px 0; color: var(--text);}

.ev {background: var(--surface); border: 1px solid var(--border); border-top: 3px solid var(--accent);
     border-radius: 10px; padding: 12px 16px; margin: 6px 0 12px 0;}
.ev-head {font-size: 0.92rem; font-weight: 600; margin-bottom: 2px;}
.ev-head a {text-decoration: none;}
.ev-head a:hover {text-decoration: underline;}
.eid {display: inline-block; font-size: 0.72rem; font-weight: 700; padding: 1px 8px; margin-right: 8px;
      border-radius: 6px; border: 1px solid var(--accent); color: var(--accent);}
.ev-meta {font-size: 0.76rem; color: var(--muted); margin-bottom: 6px;}
.ev-text {font-size: 0.89rem; line-height: 1.55; color: var(--text);}

/* ---------- inputs ---------- */
.stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"],
.stApp [data-baseweb="base-input"], .stApp [data-baseweb="textarea"] {
  background-color: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important;
}
.stApp [data-baseweb="input"], .stApp [data-baseweb="textarea"], .stApp [data-baseweb="select"] > div {border-radius: 10px;}
.stApp input::placeholder, .stApp textarea::placeholder {color: var(--muted) !important; opacity: 0.85;}
.stApp [data-baseweb="input"]:focus-within, .stApp [data-baseweb="textarea"]:focus-within,
.stApp [data-baseweb="select"] > div:focus-within {border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-soft);}
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul, [data-baseweb="menu"] {
  background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border);
}
[data-baseweb="popover"] li[role="option"] {color: var(--text) !important;}
[data-baseweb="popover"] li[role="option"]:hover {background: var(--surface-2) !important;}
.stApp [data-testid="stFileUploaderDropzone"] {background: var(--surface); border: 1.5px dashed var(--border);
                                               border-radius: 12px;}
.stApp [data-testid="stFileUploaderDropzone"] small, .stApp [data-testid="stFileUploaderDropzone"] span {color: var(--muted);}

/* ---------- buttons ---------- */
.stApp .stButton > button, .stApp .stDownloadButton > button, .stApp [data-testid="stFormSubmitButton"] > button {
  border-radius: 10px; border: 1px solid var(--border); background: var(--surface); color: var(--text);
  font-weight: 600; padding: 0.45rem 1rem; transition: border-color 0.15s, color 0.15s, box-shadow 0.15s, filter 0.15s;
}
.stApp .stButton > button p, .stApp .stDownloadButton > button p,
.stApp [data-testid="stFormSubmitButton"] > button p {color: inherit;}
.stApp .stButton > button:hover, .stApp .stDownloadButton > button:hover,
.stApp [data-testid="stFormSubmitButton"] > button:hover {border-color: var(--accent); color: var(--accent);}
.stApp .stButton > button[kind="primary"], .stApp .stButton > button[data-testid="stBaseButton-primary"],
.stApp [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"],
.stApp [data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--accent); border-color: var(--accent); color: var(--accent-text);
}
.stApp .stButton > button[kind="primary"]:hover, .stApp .stButton > button[data-testid="stBaseButton-primary"]:hover,
.stApp [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"]:hover,
.stApp [data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
  color: var(--accent-text); filter: brightness(1.08); box-shadow: 0 4px 14px var(--accent-soft);
}

/* ---------- tabs ---------- */
.stApp [data-baseweb="tab-list"] {gap: 0.25rem; border-bottom: 1px solid var(--border);}
.stApp button[data-baseweb="tab"] {background: transparent; color: var(--muted); font-weight: 600; padding: 0.65rem 1.05rem;}
.stApp button[data-baseweb="tab"] p {color: inherit;}
.stApp button[data-baseweb="tab"]:hover {color: var(--text);}
.stApp button[data-baseweb="tab"][aria-selected="true"] {color: var(--accent);}
.stApp [data-baseweb="tab-highlight"] {background-color: var(--accent) !important; height: 3px; border-radius: 3px;}
.stApp [data-baseweb="tab-border"] {background-color: transparent !important;}

/* ---------- containers, expanders, metrics, alerts ---------- */
.stApp [data-testid="stVerticalBlockBorderWrapper"] {border-color: var(--border); border-radius: 14px;}
.stApp [data-testid="stExpander"] {background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
                                  box-shadow: var(--shadow);}
.stApp [data-testid="stExpander"] details, .stApp [data-testid="stExpander"] summary {border: none; color: var(--text);}
.stApp [data-testid="stExpander"] summary p {color: var(--text); font-weight: 600;}
.stApp [data-testid="stMetric"] {background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
                                padding: 12px 16px; box-shadow: var(--shadow);}
.stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricLabel"] p {color: var(--muted);}
.stApp [data-testid="stMetricValue"] {color: var(--text);}
.stApp [data-testid="stAlert"] {border-radius: 12px;}
.stApp [data-testid="stProgress"] div[role="progressbar"] > div, .stApp .stProgress > div > div > div > div {
  background-color: var(--accent);
}
.stApp [data-testid="stSpinner"] {color: var(--muted);}

@supports selector(div:has(span)) {
  .stApp [data-testid="stAlert"] {background: var(--surface); border: 1px solid var(--border); border-left: 5px solid var(--accent);}
  .stApp [data-testid="stAlert"] p {color: var(--text);}
  .stApp [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {border-left-color: var(--ok);}
  .stApp [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {border-left-color: var(--warn);}
  .stApp [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {border-left-color: var(--bad);}

  /* radio buttons as segmented pills */
  .stApp div[role="radiogroup"] {gap: 0.5rem;}
  .stApp div[role="radiogroup"] > label {background: var(--surface); border: 1px solid var(--border);
        border-radius: 999px; padding: 0.35rem 1rem; margin: 0; cursor: pointer; transition: all 0.15s;}
  .stApp div[role="radiogroup"] > label > div:first-child {display: none;}
  .stApp div[role="radiogroup"] > label:hover {border-color: var(--accent);}
  .stApp div[role="radiogroup"] > label:has(input:checked) {background: var(--accent); border-color: var(--accent);}
  .stApp div[role="radiogroup"] > label:has(input:checked) p {color: var(--accent-text); font-weight: 700;}
}
</style>
""")
st.markdown(CSS.substitute(**P), unsafe_allow_html=True)


def _toggle_theme() -> None:
    st.session_state["theme"] = "light" if st.session_state.get("theme") == "dark" else "dark"


def theme_toggle() -> None:
    """Light / dark switch shown in the top-right corner."""
    dark = THEME == "dark"
    st.button(
        "☀️  Light" if dark else "🌙  Dark",
        key="theme_toggle",
        on_click=_toggle_theme,
        help="Switch to light theme" if dark else "Switch to dark theme",
        use_container_width=True,
    )


# ---------------------------------------------------------
# AUTHENTICATION: Login and Registration
# ---------------------------------------------------------

# Load settings before authentication so we know the database path.
base_settings = load_settings()

# Check whether the user already has a valid session.
if "auth_user" not in st.session_state:
    token = st.session_state.get("auth_token")

    if token:
        st.session_state["auth_user"] = auth.get_session_user(
            base_settings.db_path, token
        )

    else:
        st.session_state["auth_user"] = None


# Show login/register screen when the user is not authenticated.
if not st.session_state["auth_user"]:

    top_l, top_r = st.columns([6, 1.3])
    with top_l:
        st.markdown('<div class="brand"><div class="brand-mark">🔍</div>NEWSINTEL AI</div>',
                    unsafe_allow_html=True)
    with top_r:
        theme_toggle()

    st.write("")
    _, auth_col, _ = st.columns([1, 1.5, 1])

    with auth_col:
        st.title("NEWSINTEL AI")
        st.markdown("### Welcome! Please log in to continue.")

        login_tab, register_tab = st.tabs(["Login", "Register"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email address")
                password = st.text_input("Password", type="password")

                login_button = st.form_submit_button(
                    "Login", type="primary", use_container_width=True
                )

                if login_button:
                    user = auth.authenticate_user(
                        base_settings.db_path, email, password
                    )

                    if user:
                        token = auth.create_session(
                            base_settings.db_path, user["id"]
                        )

                        st.session_state["auth_token"] = token
                        st.session_state["auth_user"] = user

                        st.success("Login successful!")
                        st.rerun()

                    else:
                        st.error("Invalid email or password.")

        with register_tab:
            with st.form("register_form"):
                new_email = st.text_input("Email address")
                new_password = st.text_input(
                    "Password", type="password"
                )
                confirm_password = st.text_input(
                    "Confirm password", type="password"
                )

                register_button = st.form_submit_button(
                    "Create Account", type="primary",
                    use_container_width=True
                )

                if register_button:
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")

                    else:
                        success, message = auth.register_user(
                            base_settings.db_path,
                            new_email,
                            new_password
                        )

                        if success:
                            st.success(
                                "Account created! Please log in."
                            )
                        else:
                            st.error(message)

    # Stop here until the user logs in.
    st.stop()

# --------------------------------------------------------------------- html helpers
def esc(value) -> str:
    return html.escape(str(value), quote=True)


def pill(verdict: str) -> str:
    return f'<span class="pill" style="background:{COLORS.get(verdict, NEUTRAL)}">{esc(verdict)}</span>'


def conf_bar(value: float, color: str) -> str:
    pct = max(0, min(100, round(value * 100)))
    return (f'<div class="conf-row"><span>Confidence</span><span>{pct}%</span></div>'
            f'<div class="bar" style="--c:{color}"><div style="width:{pct}%"></div></div>')


def tile(label: str, value, color: str = "inherit") -> str:
    return (f'<div class="tile"><div class="tile-v" style="color:{color}">{esc(value)}</div>'
            f'<div class="tile-l">{esc(label)}</div></div>')


def evidence_html(p: dict, color: str) -> str:
    url = p["url"] if str(p["url"]).startswith(("http://", "https://")) else "#"
    return (f'<div class="ev" style="border-top-color:{color}">'
            f'<div class="ev-head"><span class="eid" style="border-color:{color};color:{color}">{esc(p["eid"])}</span>'
            f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(p["title"][:120])}</a></div>'
            f'<div class="ev-meta">{esc(p["domain"])} &nbsp;·&nbsp; {esc(p["credibility_label"])} '
            f'&nbsp;·&nbsp; Relevance {p["similarity"]:.2f}</div>'
            f'<div class="ev-text">{esc(p["text"])}</div></div>')


def evidence_block(title: str, passages: list, color: str) -> None:
    st.markdown(f'<div class="section-label">{esc(title)} ({len(passages)})</div>', unsafe_allow_html=True)
    for p in passages:
        st.markdown(evidence_html(p, color), unsafe_allow_html=True)


# ------------------------------------------------------------------------ results
def render_topic(topic: dict) -> None:
    """Detected Topic card (same look as the Sentiment card below it)."""
    label = topic.get("label", "General")
    detected = label != "General"
    color = TOPIC_COLOR if detected else NEUTRAL
    confidence = float(topic.get("confidence") or 0.0)

    if not detected:
        detail = f'<div class="meta-line">{esc(topic.get("note") or "No clear topic signal.")}</div>'
    else:
        detail = conf_bar(confidence, color)
        if confidence < LOW_TOPIC_CONFIDENCE:
            detail += '<div class="meta-line">Low confidence: only a few topic keywords were found.</div>'
        alts = ", ".join(f'{esc(a["label"])} ({a["confidence"]:.0%})' for a in topic.get("alternatives") or [])
        if alts:
            detail += f'<div class="meta-line" style="margin-top:6px;">Also related: {alts}</div>'

    st.markdown('<div class="section-label">Detected Topic</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card" style="border-left:6px solid {color};">'
        f'<div class="card-kicker">Main category of this news</div>'
        f'<div class="card-title" style="color:{color};">{esc(label)}</div>'
        f'{detail}</div>', unsafe_allow_html=True)


def render_result(res: dict, key_prefix: str = "main") -> None:
    verdict = res["overall_verdict"]
    color = COLORS.get(verdict, NEUTRAL)
    results = res["claim_results"]

    st.markdown(
        f'<div class="verdict-card" style="--c:{color}">'
        f'<div class="verdict-kicker">Overall assessment</div>'
        f'<div class="verdict-title">{esc(verdict)}</div>'
        f'<div class="verdict-summary">{esc(res["summary"])}</div>'
        f'{conf_bar(res["overall_confidence"], color)}</div>', unsafe_allow_html=True)



    counts = {v: sum(r["verdict"] == v for r in results) for v in (SUPPORTED, CONTRADICTED, INSUFFICIENT)}
    st.write("")
    cols = st.columns(4)
    cols[0].markdown(tile("Claims checked", len(results)), unsafe_allow_html=True)
    cols[1].markdown(tile("Supported", counts[SUPPORTED], COLORS[SUPPORTED]), unsafe_allow_html=True)
    cols[2].markdown(tile("Contradicted", counts[CONTRADICTED], COLORS[CONTRADICTED]), unsafe_allow_html=True)
    cols[3].markdown(tile("Insufficient", counts[INSUFFICIENT], COLORS[INSUFFICIENT]), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Claim-level analysis</div>', unsafe_allow_html=True)
    for cr in results:
        claim = cr["claim"]
        short = claim["text"] if len(claim["text"]) <= 85 else claim["text"][:84] + "…"
        with st.expander(f'{claim["id"]}  ·  {cr["verdict"]}  ·  {short}', expanded=len(results) <= 2):
            c = COLORS.get(cr["verdict"], NEUTRAL)
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f'<div class="meta-line">Claim {esc(claim["id"])}</div>'
                            f'<div class="claim-text">{esc(claim["text"])}</div>'
                            f'{pill(cr["verdict"])}'
                            f'&nbsp;&nbsp;<span class="meta-line">{cr["n_documents"]} pages retrieved</span>',
                            unsafe_allow_html=True)
            with right:
                st.markdown(conf_bar(cr["confidence"], c), unsafe_allow_html=True)

            if cr.get("error"):
                st.warning(f'This claim could not be fully verified: {cr["error"]}')

            st.markdown('<div class="section-label">Reasoning</div>', unsafe_allow_html=True)
            st.write(cr["reasoning"])
            for point in cr["key_points"]:
                st.markdown(f"- {point}")

            sup, con = cr["supporting"], cr["contradicting"]
            if sup and con:
                a, b = st.columns(2)
                with a:
                    evidence_block("Supporting evidence", sup, COLORS[SUPPORTED])
                with b:
                    evidence_block("Contradicting evidence", con, COLORS[CONTRADICTED])
            elif sup:
                evidence_block("Supporting evidence", sup, COLORS[SUPPORTED])
            elif con:
                evidence_block("Contradicting evidence", con, COLORS[CONTRADICTED])
            if cr["other"]:
                with st.expander(f'Other retrieved passages ({len(cr["other"])})'):
                    for p in cr["other"]:
                        st.markdown(evidence_html(p, NEUTRAL), unsafe_allow_html=True)

    st.caption(f'Report generated {res["created_at"]}. Verdicts are produced from retrieved web evidence '
               f'and should be confirmed against the cited sources.')
    st.download_button("Download report (JSON)", json.dumps(res, indent=2, ensure_ascii=False),
                       file_name="fact_check_report.json", mime="application/json",
                       key=f"download_{key_prefix}")


# ------------------------------------------------------------------------ sidebar

settings = load_settings()

with st.sidebar:
    st.markdown("### Account")
    st.divider()

    st.markdown("**Logged in as:**")
    _email = str(st.session_state["auth_user"]["email"])
    st.markdown(
        f'<div class="account"><div class="avatar">{esc(_email[:1].upper() or "?")}</div>'
        f'<div class="acct-email">{esc(_email)}</div></div>',
        unsafe_allow_html=True)

    if st.button("Logout", use_container_width=True):
        auth.logout_user(
            settings.db_path,
            st.session_state.get("auth_token")
        )

        st.session_state.pop("auth_token", None)
        st.session_state.pop("auth_user", None)
        st.session_state.pop("result", None)
        st.session_state.pop("history_view", None)

        st.rerun()


# --------------------------------------------------------------------------- main
head_l, head_r = st.columns([6, 1.3])
with head_l:
    st.title("Evidence-Based Fake News Detection")
    st.markdown('<div class="subtitle">Claim verification with retrieved evidence and cited reasoning</div>',
                unsafe_allow_html=True)
with head_r:
    theme_toggle()

tab_check, tab_summarize, tab_history, tab_news = st.tabs(
    ["Fact-check", "Summarize", "History", "Personalized News"]
)
with tab_check:
    def _set_example(text: str) -> None:
        st.session_state["input_text"] = text

    def _clear() -> None:
        st.session_state["input_text"] = ""
        st.session_state["image_extracted_text"] = ""
        st.session_state.pop("result", None)
        st.session_state.pop("show_nlp_lab", None)

    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose input type",
        ["Text Input", "Image Upload"],
        horizontal=True
    )

    if input_mode == "Text Input":
        user_text = st.text_area(
            "Headline, claim, article text or URL",
            key="input_text",
            height=160,
            placeholder="Paste a headline, claim, article or link to verify…",
            label_visibility="collapsed"
        )

    else:
        uploaded_image = st.file_uploader(
            "Upload a news image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded image", width=400)

            if st.button("Extract Text"):
                try:
                    extracted_text = extract_text_from_image(uploaded_image)
                    st.session_state["image_extracted_text"] = extracted_text
                except Exception as exc:
                    st.error(f"Could not extract text: {exc}")

        if "image_extracted_text" in st.session_state:
            st.markdown("**Extracted Text (you can edit it):**")

            user_text = st.text_area(
                "Extracted text",
                key="image_extracted_text",
                height=160,
                label_visibility="collapsed"
            )
        else:
            user_text = ""
            st.info("Upload an image and click Extract Text.")

    ex_cols = st.columns([1, 1, 1, 1.4])
    for col, (name, text) in zip(ex_cols, EXAMPLES.items()):
        col.button(name, on_click=_set_example, args=(text,), key=f"ex_{name}")

    
    act1, act2, act3, _ = st.columns([1, 1, 1, 5])

    run = act1.button("Verify", type="primary")
    act2.button("Clear", on_click=_clear)

# Run fact-checking when Verify is clicked
    if run:
        st.session_state["show_nlp_lab"] = False

        bar, status = st.progress(0.0), st.empty()

        def on_progress(frac: float, msg: str) -> None:
            bar.progress(min(max(frac, 0.0), 1.0))
            status.caption(msg)

        try:
            result = analyze(
                user_text,
                settings,
                progress=on_progress,
                user_id=st.session_state["auth_user"]["id"]
            )

            sentiment_result = analyze_sentiment(user_text)
            ner_result = extract_entities(user_text)

            result_dict = result.to_dict()
            result_dict["sentiment"] = sentiment_result
            result_dict["ner"] = ner_result

            st.session_state["result"] = result_dict

        except (InputError, ConfigError) as exc:
            st.session_state.pop("result", None)
            st.warning(str(exc))

        except FakeNewsError as exc:
            st.session_state.pop("result", None)
            st.error(f"The check could not be completed. {exc}")

        except Exception as exc:
            st.session_state.pop("result", None)
            st.error(f"Unexpected error: {exc}")

        finally:
            bar.empty()
            status.empty()


# Display NLP Lab button and saved results
# This section must remain OUTSIDE if run

    if "result" in st.session_state:

        if act3.button("NLP Lab", key="open_nlp_lab"):
            st.session_state["show_nlp_lab"] = True
            st.rerun()

        st.divider()

        if st.session_state.get("show_nlp_lab", False):
            if st.button("← Back to News Results", key="back_to_news"):
                st.session_state["show_nlp_lab"] = False
                st.rerun()

            # NLP Lab View
            st.markdown("## NLP Lab")
            st.caption("Explore the NLP techniques applied to your submitted news article.")

            # 1. Text Preprocessing
            st.markdown("### 1. Text Preprocessing")

            preprocessing = st.session_state["result"].get("preprocessing")

            if preprocessing:
                with st.expander("View Text Preprocessing Results", expanded=True):

                    st.markdown("**Cleaned Text**")
                    st.write(preprocessing["cleaned_text"])

                    st.markdown("**Tokens**")
                    st.write(preprocessing["tokens"])

                    st.markdown("**After Stopword Removal**")
                    st.write(preprocessing["filtered_tokens"])

                    st.markdown("**After Lemmatization**")
                    st.write(preprocessing["lemmatized_tokens"])

                    st.markdown("**Final Processed Text**")
                    st.write(preprocessing["processed_text"])

            else:
                st.warning("Preprocessing results are not available.")


            
            # 2. Sentiment Analysis
            st.markdown("### 2. Sentiment Analysis")

            sentiment = st.session_state["result"].get("sentiment")

            if sentiment:
                with st.expander("View Sentiment Analysis Results", expanded=True):

                    label = sentiment["label"]

                    st.markdown(f"**Overall Sentiment: {label}**")

                    st.metric("Compound Score", sentiment["compound"])

                    score_cols = st.columns(3)

                    score_cols[0].metric(
                        "Positive",
                        f'{sentiment["positive"]:.1%}'
                    )

                    score_cols[1].metric(
                        "Negative",
                        f'{sentiment["negative"]:.1%}'
                    )

                    score_cols[2].metric(
                        "Neutral",
                        f'{sentiment["neutral"]:.1%}'
                    )

            else:
                st.warning("Sentiment analysis results are not available.")


            # 3. Named Entity Recognition
            st.markdown("### 3. Named Entity Recognition")

            ner = st.session_state["result"].get("ner")

            if ner:
                with st.expander("View Named Entity Recognition Results", expanded=True):

                    entities = ner.get("entities", [])
                    entity_counts = ner.get("entity_counts", {})

                    if entities:
                        st.markdown("**Extracted Entities**")

                        st.dataframe(
                            [
                                {
                                    "Entity": entity["text"],
                                    "Category": entity["label"],
                                    "Description": entity["description"]
                                }
                                for entity in entities
                            ],
                            use_container_width=True,
                            hide_index=True
                        )

                        st.markdown("**Entity Category Summary**")

                        if entity_counts:
                            count_cols = st.columns(min(len(entity_counts), 4))

                            for index, (category, count) in enumerate(entity_counts.items()):
                                with count_cols[index % len(count_cols)]:
                                    st.metric(category, count)

                    else:
                        st.info("No named entities were identified in this article.")

            else:
                st.warning("NER results are not available.")

            
            # 3. Topic Detection
            st.markdown("### 3. Topic Detection")

            topic = st.session_state["result"].get("topic")

            if topic:
                with st.expander("View Topic Detection Results", expanded=True):

                    render_topic(topic)

            else:
                st.warning("Topic detection results are not available.")


            # 4. TF-IDF and Similarity Analysis
            st.markdown("### 4. TF-IDF & Similarity Analysis")

            claim_results = st.session_state["result"].get("claim_results", [])

            if claim_results:

                for claim_result in claim_results:

                    claim = claim_result["claim"]

                    st.markdown(f"**Claim: {claim['text']}**")

                    evidence_groups = {
                        "Supporting Evidence": claim_result.get("supporting", []),
                        "Contradicting Evidence": claim_result.get("contradicting", []),
                        "Other Retrieved Evidence": claim_result.get("other", [])
                    }

                    found_evidence = False

                    for group_name, passages in evidence_groups.items():

                        if passages:
                            found_evidence = True

                            with st.expander(group_name):

                                for passage in passages:

                                    score = float(passage.get("similarity", 0))

                                    st.markdown(f"**{passage['eid']} - {passage['title']}**")

                                    st.write(passage["text"])

                                                                    
                                    st.metric("Hybrid Similarity", f"{score:.1%}")

                                    st.progress(max(0.0, min(score, 1.0)))


                                    st.caption(
                                        f"Source: {passage['domain']}"
                                    )

                                    st.markdown(
                                        f"[Visit Source]({passage['url']})"
                                    )

                                    st.divider()

                    if not found_evidence:
                        st.info("No ranked evidence passages are available for this claim.")

            else:
                st.warning("No claim-level similarity results are available.")




        else:
            # News Results View
            render_result(st.session_state["result"])


with tab_summarize:
    st.markdown("## Intelligent News Summarization")
    st.write(
        "Generate short summaries, detailed explanations "
        "and important key points from news articles."
    )

    summary_input_type = st.radio(
        "Choose article input type",
        ["Paste Article", "Article URL"],
        horizontal=True,
        key="summary_input_type"
    )

    if summary_input_type == "Paste Article":
        article_text = st.text_area(
            "Paste your news article here",
            height=250,
            placeholder="Enter the complete news article...",
            key="summary_article_text"
        )
    else:
        article_url = st.text_input(
            "Enter news article URL",
            placeholder="https://example.com/news/article",
            key="summary_article_url"
        )

    if st.button("Generate Summary", key="generate_summary"):
        try:
            # Get article text
            if summary_input_type == "Paste Article":
                article_content = article_text.strip()
                source_url = ""

                if not article_content:
                    st.warning("Please paste a news article first.")
                    st.stop()

            else:
                if not article_url.strip():
                    st.warning("Please enter an article URL.")
                    st.stop()

                if not is_url(article_url.strip()):
                    st.warning("Please enter a valid URL.")
                    st.stop()

                with st.spinner("Extracting article from URL..."):
                    article_title, article_content = fetch_page_text(
                        article_url.strip(),
                        settings
                    )

                source_url = article_url.strip()

                if not article_content.strip():
                    st.error("Could not extract article text from this URL.")
                    st.stop()

            # Generate summary
            with st.spinner("Generating summary..."):
                summary_result = summarize_article(
                    article_content,
                    settings
                )

            summary_data = json.loads(summary_result)

            st.session_state["summary_result"] = summary_data
            st.session_state["summary_source_url"] = source_url

            st.success("Summary generated successfully!")

        except Exception as e:
            st.error(f"Summarization failed: {e}")

    if "summary_result" in st.session_state:
        summary = st.session_state["summary_result"]

        st.divider()
        st.markdown("## Generated Summary")

        # Short Summary
        with st.container(border=True):
            st.markdown("### 📝 Short Summary")
            st.write(summary["short_summary"])

        # Detailed Summary
        with st.container(border=True):
            st.markdown("### 📖 Detailed Summary")
            st.write(summary["detailed_summary"])

        # Important Key Points
        with st.container(border=True):
            st.markdown("### 🔑 Important Key Points")

            source_url = st.session_state.get("summary_source_url", "")
            for point in summary["key_points"]:
                st.markdown(f"- {point}")

        if source_url:
            st.divider()
            st.markdown("### 🔗 Original Article")
            st.markdown(f"[Read Full Article]({source_url})")

with tab_history:
    try:
        rows = database.list_history(
            settings.db_path,
            limit=100,
            user_id=st.session_state["auth_user"]["id"]
        )
    except Exception as exc:
        rows = []
        st.error(f"Could not read history: {exc}")

    if not rows:
        st.info("No saved checks yet. Completed analyses appear here automatically.")
    else:
        st.markdown(f'<div class="section-label">Saved analyses ({len(rows)})</div>', unsafe_allow_html=True)
        for r in rows:
            with st.container(border=True):
                c1, c2, c3 = st.columns([7, 1, 1])
                preview = r["input_text"][:110] + ("…" if len(r["input_text"]) > 110 else "")
                c1.markdown(f'{pill(r["overall_verdict"])} &nbsp;<span class="meta-line">'
                            f'{r["overall_confidence"]:.0%} confidence</span><br>{esc(preview)}<br>'
                            f'<span class="hist-meta">#{r["id"]} · {esc(r["created_at"])}</span>',
                            unsafe_allow_html=True)
                if c2.button("Open", key=f'open_{r["id"]}'):
                    st.session_state["history_view"] = database.get_analysis(
                        settings.db_path,
                        r["id"],
                        user_id=st.session_state["auth_user"]["id"]
                    )
                if c3.button("Delete", key=f'del_{r["id"]}'):
                    database.delete_analysis(
                        settings.db_path,
                        r["id"],
                        user_id=st.session_state["auth_user"]["id"]
                    )
                    view = st.session_state.get("history_view")
                    if view and view.get("id") == r["id"]:
                        st.session_state.pop("history_view", None)
                    st.rerun()

        with st.expander("Danger zone"):
            if st.checkbox("I understand this permanently deletes all saved analyses"):
                if st.button("Clear all history"):
                    database.clear_history(
                        settings.db_path,
                        user_id=st.session_state["auth_user"]["id"]
                    )
                    st.session_state.pop("history_view", None)
                    st.rerun()

        if st.session_state.get("history_view"):
            st.divider()
            render_result(st.session_state["history_view"], key_prefix="history")

with tab_news:
    st.markdown("## 📰 Personalized News")
    st.write(
        "Choose a news category or enter your own topic "
        "to discover the latest news, each with an AI-written summary."
    )

    selected_category = st.selectbox(
        "Choose a news category",
        NEWS_CATEGORIES,
        key="news_category"
    )

    if selected_category == "Other / Customize":
        custom_topic = st.text_input(
            "Enter your preferred news topic",
            placeholder="e.g. Artificial Intelligence, SpaceX, Cybersecurity",
            key="custom_news_topic"
        )
        selected_topic = custom_topic.strip()
    else:
        selected_topic = selected_category

    if st.button("🔍 Get News", key="get_personalized_news"):
        if not selected_topic:
            st.warning("Please select or enter a news topic.")
        else:
            with st.spinner("Fetching the latest news and writing summaries (this can take a minute)..."):
                try:
                    news_results = fetch_personalized_news(selected_topic, settings)
                    st.session_state["personalized_news_results"] = news_results
                    st.session_state["personalized_news_topic"] = selected_topic
                    st.session_state.pop("topic_modeling_result", None)
                except (InputError, ConfigError) as e:
                    st.warning(str(e))
                except FakeNewsError as e:
                    st.error(f"Could not fetch news: {e}")
                except Exception as e:
                    st.error(f"Could not fetch news: {e}")

    if "personalized_news_results" in st.session_state:
        news_results = st.session_state["personalized_news_results"]
        news_topic = st.session_state.get("personalized_news_topic", "")

        st.divider()
        st.markdown(f"### Latest News: {news_topic}")

        if not news_results:
            st.info("No news articles found. Try another topic.")
        else:
            if st.button("🧩 Topic Modeling", key="run_topic_modeling"):
                with st.spinner("Discovering topics..."):
                    try:
                        st.session_state["topic_modeling_result"] = run_topic_modeling(news_results)
                    except Exception as e:
                        st.session_state["topic_modeling_result"] = {
                            "status": "error",
                            "message": f"Topic modeling could not be completed: {e}",
                        }

            tm = st.session_state.get("topic_modeling_result")
            if tm:
                with st.expander("Topic Modeling Results (NMF)", expanded=True):
                    if tm["status"] in ("empty", "insufficient"):
                        st.info(tm["message"])
                    elif tm["status"] != "ok":
                        st.warning(tm["message"])
                    else:
                        st.markdown(f"**{tm['n_topics']} topics discovered across {tm['n_articles']} articles**")

                        st.markdown("**Topic Distribution**")
                        st.bar_chart(
                            {d["label"]: d["article_count"] for d in tm["distribution"]},
                            horizontal=True,
                        )

                        for tp in tm["topics"]:
                            n = tp["article_count"]
                            with st.expander(f'{tp["label"]}  ({n} article{"" if n == 1 else "s"}, {tp["share"]:.0%})'):
                                st.markdown("**Keywords:** " + ", ".join(k["term"] for k in tp["keywords"]))
                                if tp["articles"]:
                                    for a in tp["articles"]:
                                        st.markdown(f"- [{a['title']}]({a['url']})" if a["url"] else f"- {a['title']}")
                                else:
                                    st.caption("No articles were grouped under this topic.")

                        if tm["unassigned"]:
                            st.caption("No clear topic: " + "; ".join(a["title"] for a in tm["unassigned"]))
                        if tm["skipped"]:
                            st.caption(f'{len(tm["skipped"])} article(s) skipped because they had too little text.')

            for article in news_results:
                with st.container(border=True):
                    # Headline
                    st.markdown(f"### {article.get('title', 'Untitled')}")

                    # AI summary (falls back to the cleaned search preview if none was generated)
                    summary = (article.get("summary") or "").strip()
                    if summary:
                        st.markdown("**AI summary**")
                        st.write(summary)
                        if article.get("summary_source") == "snippet":
                            st.caption(
                                "ℹ️ Summary is based on the search preview because "
                                "the full article could not be opened."
                            )
                    else:
                        st.write(clean_snippet(article.get("content", "")) or "No description available.")
                        reason = (article.get("summary_error") or "").strip()
                        st.caption(
                            "⚠️ AI summary unavailable"
                            + (f": {reason[:120]}" if reason else ".")
                        )

                    # Source name from the URL
                    link = article.get("url", "")
                    source = link.split("//")[-1].split("/")[0].replace("www.", "")
                    st.caption(f"📰 Source: {source}")
                    st.caption(f"📅 Published: {article.get('published_date', 'Date unavailable')}")

                    # Article link
                    if link:
                        st.markdown(f"[Read Full Article ↗]({link})")