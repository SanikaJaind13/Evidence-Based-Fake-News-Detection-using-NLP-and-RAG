"""Streamlit frontend: Evidence-Based Fake News Detection using NLP + RAG."""
from __future__ import annotations

import html
import json

import streamlit as st
from src.ocr import extract_text_from_image
from src import database
from src import auth
from src.config import load_settings
from src.errors import ConfigError, FakeNewsError, InputError
from src.models import CONTRADICTED, INSUFFICIENT, SUPPORTED
from src.pipeline import analyze

st.set_page_config(page_title="Evidence-Based Fake News Detection", page_icon="🔍", layout="wide")

# --------------------------------------------------------------------------- theme
COLORS = {SUPPORTED: "#1f7a4d", CONTRADICTED: "#b3261e", INSUFFICIENT: "#8a6d1a"}
NEUTRAL = "#6b7280"
EXAMPLES = {
    "Debunked claim": "Humans only use 10 percent of their brains.",
    "Verifiable claim": "India's Chandrayaan-3 landed near the Moon's south pole in August 2023.",
    "Multi-claim text": (
        "The World Health Organization declared COVID-19 a global pandemic on 11 March 2020. "
        "Officials also said that 5G towers spread the coronavirus across Europe. "
        "Vitamin C cures the common cold in 24 hours, according to a viral post."),
}

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2.2rem; padding-bottom: 3rem;}
#MainMenu, footer {visibility: hidden;}
h1 {font-weight: 650 !important; letter-spacing: -0.01em; margin-bottom: 0.2rem !important;}
.subtitle {color: #6b7280; font-size: 0.95rem; margin-bottom: 1.2rem;}
.section-label {font-size: 0.72rem; font-weight: 650; letter-spacing: 0.09em; text-transform: uppercase;
                color: #6b7280; margin: 1.1rem 0 0.4rem 0;}

.verdict-card {border: 1px solid rgba(128,128,128,0.28); border-left: 6px solid var(--c);
               border-radius: 8px; padding: 18px 22px; background: rgba(128,128,128,0.05);}
.verdict-kicker {font-size: 0.72rem; font-weight: 650; letter-spacing: 0.09em; text-transform: uppercase; color: #6b7280;}
.verdict-title {font-size: 1.7rem; font-weight: 650; color: var(--c); margin: 2px 0 8px 0;}
.verdict-summary {font-size: 0.95rem; opacity: 0.9;}

.bar {height: 8px; border-radius: 999px; background: rgba(128,128,128,0.22); overflow: hidden; margin-top: 6px;}
.bar > div {height: 100%; background: var(--c); border-radius: 999px;}
.conf-row {display: flex; justify-content: space-between; font-size: 0.8rem; color: #6b7280;}

.tile {border: 1px solid rgba(128,128,128,0.28); border-radius: 8px; padding: 12px 14px; text-align: center;}
.tile-v {font-size: 1.6rem; font-weight: 650; line-height: 1.2;}
.tile-l {font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; color: #6b7280;}

.pill {display: inline-block; padding: 2px 10px; border-radius: 4px; color: #fff; font-size: 0.78rem;
       font-weight: 600; letter-spacing: 0.02em;}
.claim-text {font-size: 1.02rem; font-weight: 550; margin: 4px 0 10px 0;}
.meta-line {font-size: 0.82rem; color: #6b7280;}

.ev {border: 1px solid rgba(128,128,128,0.25); border-top: 3px solid var(--accent); border-radius: 6px;
     padding: 10px 14px; margin: 6px 0 12px 0; background: rgba(128,128,128,0.04);}
.ev-head {font-size: 0.9rem; font-weight: 600; margin-bottom: 2px;}
.ev-head a {text-decoration: none;}
.ev-head a:hover {text-decoration: underline;}
.eid {display: inline-block; font-size: 0.72rem; font-weight: 700; padding: 1px 7px; margin-right: 8px;
      border-radius: 4px; border: 1px solid var(--accent); color: var(--accent);}
.ev-meta {font-size: 0.76rem; color: #6b7280; margin-bottom: 6px;}
.ev-text {font-size: 0.88rem; line-height: 1.5;}
.hist-meta {font-size: 0.78rem; color: #6b7280;}
</style>
""", unsafe_allow_html=True)


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
    return (f'<div class="ev" style="--accent:{color}">'
            f'<div class="ev-head"><span class="eid">{esc(p["eid"])}</span>'
            f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(p["title"][:120])}</a></div>'
            f'<div class="ev-meta">{esc(p["domain"])} &nbsp;·&nbsp; {esc(p["credibility_label"])} '
            f'&nbsp;·&nbsp; Relevance {p["similarity"]:.2f}</div>'
            f'<div class="ev-text">{esc(p["text"])}</div></div>')


def evidence_block(title: str, passages: list, color: str) -> None:
    st.markdown(f'<div class="section-label">{esc(title)} ({len(passages)})</div>', unsafe_allow_html=True)
    for p in passages:
        st.markdown(evidence_html(p, color), unsafe_allow_html=True)


# ------------------------------------------------------------------------ results
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

# ------------------------------------------------------------------------ sidebar

settings = load_settings()

with st.sidebar:
    st.markdown("### Account")
    st.divider()

    st.markdown("**Logged in as:**")
    st.write(st.session_state["auth_user"]["email"])

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
st.title("Evidence-Based Fake News Detection")
st.markdown('<div class="subtitle">Claim verification with retrieved evidence and cited reasoning</div>',
            unsafe_allow_html=True)

tab_check, tab_history = st.tabs(["Fact-check", "History"])

with tab_check:
    def _set_example(text: str) -> None:
        st.session_state["input_text"] = text

    def _clear() -> None:
        st.session_state["input_text"] = ""
        st.session_state["image_extracted_text"] = ""
        st.session_state.pop("result", None)

    
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

act1, act2, _ = st.columns([1, 1, 6])
run = act1.button("Verify", type="primary")
act2.button("Clear", on_click=_clear)

if run:
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
        st.session_state["result"] = result.to_dict()
    except (InputError, ConfigError) as exc:
        st.session_state.pop("result", None)
        st.warning(str(exc))
    except FakeNewsError as exc:
        st.session_state.pop("result", None)
        st.error(f"The check could not be completed. {exc}")
    except Exception as exc:  # last-resort safety net so the app never crashes
        st.session_state.pop("result", None)
        st.error(f"Unexpected error: {exc}")
    finally:
        bar.empty()
        status.empty()

    if "result" in st.session_state:
        st.divider()
        render_result(st.session_state["result"])

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

