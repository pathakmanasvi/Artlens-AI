"""Streamlit UI. Run: streamlit run app.py"""
import streamlit as st

from src.pipeline import analyze
from src.preprocessing import load_image

st.set_page_config(page_title="ArtLens AI", page_icon="🖼️", layout="wide")
st.title("ArtLens AI")
st.caption("Photograph or upload a painting or drawing to get its style, estimated period and a description.")

source = st.radio("Input", ["Camera", "Upload"], horizontal=True)
file = st.camera_input("Take a photo") if source == "Camera" else st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "webp"])
col_a, col_b = st.columns(2)
auto_crop = col_a.checkbox("Auto-crop canvas", value=True)
use_llm = col_b.checkbox("Generate description (needs API key)", value=True)

if file is None:
    st.stop()

img = load_image(file)
left, right = st.columns([1, 1.2])
left.image(img, caption="Your image", use_container_width=True)

with st.spinner("Analysing..."):
    try:
        res = analyze(img, crop=auto_crop, use_llm=use_llm)
    except FileNotFoundError:
        right.error("Models or reference index not found. Run the training scripts first (see README).")
        st.stop()

for w in res["quality"]["warnings"]:
    right.warning(w)

if res["identification"] == "known":
    right.success("Recognised as a known artwork")
    right.subheader(res.get("title") or "Untitled")
    right.write(f"**Artist:** {res.get('artist')}")
    right.write(f"**Year:** {res.get('year')}  |  **Age:** {res.get('age_years')} years")
    right.write(f"**Style:** {res.get('style')}")
    right.progress(min(res["confidence"], 1.0), text=f"Match similarity {res['confidence']:.0%}")
else:
    right.info("No confident match. Showing a visual estimate.")
    right.write("**Likely period**")
    for label, p in res["period_estimate"]:
        right.progress(p, text=f"{label} ({p:.0%})")
    right.write("**Likely style**")
    for label, p in res["style_estimate"]:
        right.progress(p, text=f"{label} ({p:.0%})")
    if res["confidence"] < 0.4:
        right.warning("Low confidence. Treat this as a rough guess.")

right.caption(res["note"])

desc = res.get("description")
if desc:
    st.divider()
    st.subheader("Description")
    if "error" in desc:
        st.warning(desc["error"])
    else:
        for key, val in desc.items():
            st.markdown(f"**{key.replace('_', ' ').title()}**  \n{val}")

if res["similar_works"]:
    st.divider()
    st.subheader("Most similar works in the reference set")
    for m in res["similar_works"]:
        st.write(f"- {m.get('title')} — {m.get('artist')} ({m.get('year')}) · similarity {m['similarity']:.2f}")
