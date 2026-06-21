import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path

DATA = Path("data/working_annotations.csv")
LABELS = ["analysis", "hot_take", "reaction"]

# Injected into an invisible iframe each rerun; window.parent gives the Streamlit doc.
# We swap the handler reference so only one listener is ever live.
SHORTCUT_JS = """<script>
(function () {
    const doc = window.parent.document;
    const KEY_MAP = {'1': 'analysis', '2': 'hot_take', '3': 'reaction'};

    function clickLabel(text) {
        for (const el of doc.querySelectorAll('label')) {
            if (el.textContent.trim() === text) { el.click(); break; }
        }
    }
    function clickSaveNext() {
        for (const btn of doc.querySelectorAll('button')) {
            if (btn.innerText.trim() === 'Save & Next') { btn.click(); break; }
        }
    }
    function handler(e) {
        const el = doc.activeElement;
        const typing = el && ((el.tagName === 'INPUT' && el.type !== 'radio') || el.tagName === 'TEXTAREA');
        if (typing) return;
        if (KEY_MAP[e.key]) { e.preventDefault(); clickLabel(KEY_MAP[e.key]); }
        else if (e.key === 'Enter' || e.key === 'ArrowRight') { e.preventDefault(); clickSaveNext(); }
    }
    if (window.parent._tmHandler) {
        doc.removeEventListener('keydown', window.parent._tmHandler);
    }
    window.parent._tmHandler = handler;
    doc.addEventListener('keydown', handler);
})();
</script>"""


def load():
    return pd.read_csv(DATA, dtype=str).fillna("")


def save_row(df, idx, label, notes):
    df.at[idx, "label"] = label
    df.at[idx, "notes"] = notes
    df.to_csv(DATA, index=False)


st.set_page_config(page_title="TakeMeter Review", layout="wide")
df = load()

unlabeled_idx = df.index[df["label"] == ""].tolist()
first_unlabeled = unlabeled_idx[0] if unlabeled_idx else len(df) - 1

if "idx" not in st.session_state:
    st.session_state.idx = first_unlabeled

idx = st.session_state.idx
row = df.iloc[idx]

labeled = df[df["label"] != ""]
dist = {lbl: int((labeled["label"] == lbl).sum()) for lbl in LABELS}
dist_str = "  |  ".join(f"**{lbl}** {n}" for lbl, n in dist.items())

st.title(f"TakeMeter Review — {len(labeled)}/{len(df)} labeled")
st.markdown(dist_str)
st.markdown("---")

hide_draft = st.toggle("Hide draft label (blind calibration pass)")

col_prev, col_next, col_jump, _ = st.columns([1, 1, 2, 6])
with col_prev:
    if st.button("← Prev", disabled=idx == 0):
        st.session_state.idx -= 1
        st.rerun()
with col_next:
    if st.button("Next →", disabled=idx >= len(df) - 1):
        st.session_state.idx += 1
        st.rerun()
with col_jump:
    if st.button("Jump to first unlabeled"):
        st.session_state.idx = first_unlabeled
        st.rerun()

st.caption(f"Row {idx + 1} of {len(df)}  —  {row['story_title']}")

with st.container(border=True):
    st.markdown(row["text"])

if not hide_draft:
    draft = row["label_suggested"]
    st.info(f"Draft: **{draft}**" if draft in LABELS else f"Draft: `{draft}` (unrecognised)")

default = row["label"] if row["label"] in LABELS else (
    row["label_suggested"] if row["label_suggested"] in LABELS else LABELS[0]
)
chosen = st.radio(
    "Final label  (keys: **1** analysis · **2** hot\_take · **3** reaction · **Enter / →** save & next)",
    LABELS, index=LABELS.index(default), horizontal=True, key=f"label_{idx}",
)
notes = st.text_input("Notes (optional)", value=row["notes"], key=f"notes_{idx}")

if st.button("Save & Next", type="primary"):
    fresh = load()
    save_row(fresh, idx, chosen, notes)
    if idx < len(fresh) - 1:
        st.session_state.idx += 1
    st.rerun()

components.html(SHORTCUT_JS, height=0)
