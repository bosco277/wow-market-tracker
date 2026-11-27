"""WoW Market Desk: WoW Forever only (v0.7.48). Reads the daily tables, item names and house map from the Forever dataset on Hugging Face.

The page itself is forever_page.py (screens) and forever_lib.py (data helpers). The retail page that used to live here was removed in v0.7.48;
it is in the git history. Secrets: HF_TOKEN (read access to the Forever dataset), optional FOREVER_HF_REPO."""
import streamlit as st

st.set_page_config(page_title="WoW Market Desk · WoW Forever", layout="wide")


def _stretch() -> dict:
    """Streamlit 1.50+ wants width="stretch"; older versions only know use_container_width. Use whichever this server has."""
    try:
        v = tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        v = (0, 0)
    return {"width": "stretch"} if v >= (1, 50) else {"use_container_width": True}


import forever_page  # noqa: E402  (after set_page_config, which must be the first Streamlit call)

forever_page.render(_stretch())
