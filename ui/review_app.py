"""Broker review page (Streamlit): submit a call, read the transcript and summary, approve the CRM diff.

    streamlit run ui/review_app.py
Env: BC_API_URL (default http://localhost:8000), BC_UI_API_KEY (default dev-key).
"""

import os

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("BC_API_URL", "http://localhost:8000")
KEY = os.environ.get("BC_UI_API_KEY", "dev-key")
http = httpx.Client(base_url=API, headers={"X-API-Key": KEY}, timeout=600)

st.set_page_config(page_title="Broker Copilot", layout="wide")
st.markdown(
    """
<style>
:root { --ink:#1b1b1b; --muted:#4a4a4a; --line:#d9d4ca; --paper:#faf8f4; --accent:#7a3b12; }
html, body, [class*="css"], .stApp { background: var(--paper); color: var(--ink); font-family: "Times New Roman", serif; }
h1, h2, h3 { font-family: "GFS Didot", "Didot", "Times New Roman", serif !important; color: var(--ink); letter-spacing: 0; }
label, .stMetric label, th { font-family: "Times New Roman", serif !important; font-weight: 700 !important; color: var(--ink) !important; }
.seg { padding: 6px 10px; margin: 4px 0; border-left: 3px solid var(--line); direction: rtl; text-align: right; }
.seg.broker { border-color: #8a8a8a; background: #f1efe9; }
.seg.client { border-color: var(--accent); background: #fff; }
.who { font-weight: 700; font-size: 0.85rem; color: var(--muted); direction: ltr; text-align: left; }
.flag { color: var(--accent); font-weight: 700; }
textarea { unicode-bidi: plaintext; }  /* each line takes its own direction: Arabic right, English left */
[class*="st-key-followup"] textarea { direction: rtl; text-align: right; }
</style>
""",
    unsafe_allow_html=True,
)


def get(path: str, **params):
    r = http.get(path, params=params)
    r.raise_for_status()
    return r.json()


st.title("Broker Copilot")

leads = get("/api/leads")
lead_names = {"(no lead)": None} | {f"{lead['lead_id']} - {lead['name']}": lead["lead_id"] for lead in leads}

with st.sidebar:
    st.header("New call")
    lead_label = st.selectbox("Lead", list(lead_names))
    mode = st.radio("Input", ["Audio", "Text / WhatsApp export"], horizontal=True)
    if mode == "Audio":
        audio = st.file_uploader("Recording or voice note", type=["wav", "mp3", "m4a", "ogg", "opus", "webm"])
        consent = st.checkbox("Client consented to recording")
        if st.button("Process", disabled=not (audio and consent)):
            with st.spinner("Transcribing and summarizing..."):
                r = http.post("/api/calls/audio", files={"file": (audio.name, audio.getvalue())},
                              data={"lead_id": lead_names[lead_label] or "", "consent": "true", "channel": "upload"})
            st.session_state["call_id"] = r.json().get("id") if r.is_success else st.error(r.text)
    else:
        text = st.text_area("Paste 'broker: / client:' lines or a WhatsApp export", height=200)
        if st.button("Process", disabled=not text.strip()):
            with st.spinner("Summarizing..."):
                r = http.post("/api/calls/text", json={"lead_id": lead_names[lead_label], "text": text})
            st.session_state["call_id"] = r.json().get("id") if r.is_success else st.error(r.text)

    st.header("Calls")
    calls = get("/api/calls", limit=30)
    for c in calls:
        label = f"#{c['id']} {c['status']} - {(c['headline'] or c['channel'])[:50]}"
        if st.button(label, key=f"call{c['id']}", use_container_width=True):
            st.session_state["call_id"] = c["id"]

call_id = st.session_state.get("call_id") or (calls[0]["id"] if calls else None)
if not call_id:
    st.info("Submit a call from the sidebar.")
    st.stop()

call = get(f"/api/calls/{call_id}")
summary = call.get("summary") or {}
c1, c2, c3, c4 = st.columns(4)
c1.metric("Status", call["status"])
c2.metric("Audio", f"{(call.get('duration_s') or 0):.0f}s")
c3.metric("Processing", f"{(call.get('processing_s') or 0):.1f}s")
c4.metric("Cost", f"${(call.get('usage') or {}).get('total_usd', 0):.4f}")
if call["status"] == "failed":
    st.error(call["error"])
    st.stop()
if (call.get("usage") or {}).get("fallback_reason"):
    st.warning("LLM unavailable, rules extractor used: " + call["usage"]["fallback_reason"])

st.subheader(summary.get("headline", ""))
left, right = st.columns([1.1, 1])

with left:
    st.markdown("### Transcript")
    for i, seg in enumerate((call.get("transcript") or {}).get("segments", [])):
        m, s = divmod(int(seg["start"]), 60)
        st.markdown(
            f'<div class="seg {seg["speaker"]}"><div class="who">[{i}] {m:02d}:{s:02d} {seg["speaker"]}</div>'
            f'{seg["text"]}</div>', unsafe_allow_html=True)

with right:
    st.markdown("### CRM changes")
    changes = call.get("crm_changes") or []
    flagged = set(call.get("flagged_fields") or [])
    if flagged:
        st.markdown(f'<span class="flag">Evidence not found for: {", ".join(sorted(flagged))}</span>',
                    unsafe_allow_html=True)
    approved, rejected, overrides = [], [], {}
    editable = call["status"] == "review"
    for ch in changes:
        if ch["status"] == "unchanged":
            continue
        quote = "; ".join(e["quote"] for e in ch.get("evidence", [])[:2])
        cols = st.columns([1.2, 1.2, 0.6, 1])
        cols[0].markdown(f"**{ch['field']}**  \n<small>was: {ch['old']}</small>", unsafe_allow_html=True)
        new_val = cols[1].text_input("value", str(ch["new"]), key=f"v{ch['field']}", label_visibility="collapsed",
                                     disabled=not editable, help=quote or None)
        cols[2].markdown(f"{ch['confidence']:.2f}")
        if ch["status"] == "needs_review":
            if cols[3].checkbox("approve", key=f"a{ch['field']}", disabled=not editable):
                approved.append(ch["field"])
        elif not cols[3].checkbox("auto", value=True, key=f"r{ch['field']}", disabled=not editable):
            rejected.append(ch["field"])
        if new_val != str(ch["new"]):
            overrides[ch["field"]] = new_val
    if editable and st.button("Approve and write to CRM", type="primary"):
        r = http.post(f"/api/calls/{call_id}/approve",
                      json={"approved_fields": approved, "rejected_fields": rejected, "overrides": overrides})
        st.success("CRM updated") if r.is_success else st.error(r.text)
        st.rerun()

    st.markdown("### Objections")
    for o in summary.get("objections", []):
        st.markdown(f"**{o['category']}**: {o['text']}  \n_Suggested:_ {o.get('suggested_response') or '-'}")
    st.markdown("### Commitments")
    for cm in summary.get("commitments", []):
        st.markdown(f"**{cm['by']}** {cm['text']} {('- due ' + cm['due']) if cm.get('due') else ''}")
    st.markdown(f"**Next action:** {summary.get('next_action', '')}  \n"
                f"**Follow-up:** {(summary.get('follow_up_date') or {}).get('value')}  \n"
                f"**Sentiment:** {summary.get('sentiment')}")

st.markdown("### Matched units")
if call.get("matches"):
    df = pd.DataFrame(call["matches"])[["unit_id", "title", "area", "price_egp", "down_payment_pct",
                                        "installment_years", "delivery", "score"]]
    st.dataframe(df, hide_index=True, use_container_width=True)
st.markdown("### Follow-up draft")
st.text_area("WhatsApp message (edit, then send from WhatsApp)", call.get("followup_draft") or "", height=180,
             key=f"followup_{call['id']}")

with st.expander("Monitoring"):
    st.json(get("/api/metrics"))
