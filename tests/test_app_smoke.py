"""Runs app.py (WoW Forever only since v0.7.48) against stand-ins for Streamlit and Plotly (neither is installed where the tests run), on
a pretend Forever dataset folder. It cannot judge how the page looks; it catches typos, wrong column names and crashes on empty or old tables."""
import os
import runpy
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_app_dir = os.path.join(_root, "frontend") if os.path.exists(os.path.join(_root, "frontend", "app.py")) else _root


class Any:
    """Accepts any call, attribute, context manager or iteration."""
    calls = []

    def __init__(self, name="x"):
        self._name = name

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return Any(f"{self._name}.{k}")

    def __call__(self, *a, **k):
        Any.calls.append((self._name, a, k))
        return Any(self._name + "()")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter([Any(), Any(), Any(), Any()])


class Col(Any):
    """A column or tab: anything called on it is done by the page itself, as in Streamlit."""
    def __init__(self, st):
        self._st = st

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return getattr(self._st, k)


def fake_streamlit(selected_row=0, found=None, version="1.51.0", only_for=None):  # only_for: a column name; only tables that have it get the clicked row
    st = types.ModuleType("streamlit")
    st.__version__ = version
    st.set_page_config = lambda **k: None
    st.cache_data = lambda *a, **k: (lambda f: f)
    st.secrets = {}
    st.session_state = {}
    st.query_params = {"watch": "2"}
    st.columns = lambda spec, **k: [Col(st) for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda names: [Col(st) for _ in names]
    st.selectbox = lambda label, options, index=0, key=None, **k: found if key == "find_item" else (options[index] if index is not None and len(options) else None)
    st.radio = lambda label, options, index=0, **k: options[index]
    st.slider = lambda label, lo, hi, value=None, *a, **k: value
    st.text_input = lambda *a, **k: ""
    st.multiselect = lambda label, options, default=None, key=None, **k: st.session_state.get(key, default or [])
    st.checkbox = lambda *a, **k: False
    st.number_input = lambda *a, **k: 0.0
    ev = types.SimpleNamespace(selection=types.SimpleNamespace(rows=[selected_row] if selected_row is not None else []))
    seen = []

    nothing = types.SimpleNamespace(selection=types.SimpleNamespace(rows=[]))

    def dataframe(df, **k):
        seen.append((df, k))
        return ev if only_for is None or only_for in df.columns else nothing
    st.dataframe = dataframe
    st.seen = seen
    st.column_config = Any("column_config")
    for name in ("title", "caption", "error", "warning", "info", "success", "write", "markdown", "subheader", "metric", "table", "plotly_chart",
                 "divider", "expander", "stop", "code", "download_button"):
        setattr(st, name, Any(name))
    return st


def run_app(forever_dir=None, **kw):  # kw: selected_row, found, version, only_for. forever_dir: a pretend Forever dataset folder
    st = fake_streamlit(**kw)

    def stop():
        raise SystemExit(0)                  # as in Streamlit: nothing after st.stop() runs
    st.stop = stop
    if forever_dir:
        os.environ["FOREVER_LOCAL_DIR"] = forever_dir
    px = types.ModuleType("plotly")
    go = types.ModuleType("plotly.graph_objects")
    for name in ("Figure", "Scatter", "Bar", "Heatmap"):
        setattr(go, name, Any(name))
    px.graph_objects = go
    orig_read = pd.read_parquet
    st.parquet_reads = []

    def no_parquet(path, *a, **k):           # the app must not read retail tables any more
        st.parquet_reads.append(path)
        raise FileNotFoundError(path)
    pd.read_parquet = no_parquet
    saved_path = list(sys.path)
    sys.path.insert(0, _app_dir)
    for m in ("lib", "forever_lib", "forever_page"):
        sys.modules.pop(m, None)
    try:
        with mock.patch.dict(sys.modules, {"streamlit": st, "plotly": px, "plotly.graph_objects": go}):
            try:
                runpy.run_path(os.path.join(_app_dir, "app.py"), run_name="__main__")
            except SystemExit:
                pass
    finally:
        pd.read_parquet = orig_read
        sys.path[:] = saved_path
        os.environ.pop("FOREVER_LOCAL_DIR", None)
    return st


class TestAppIsForeverOnly(unittest.TestCase):
    """v0.7.48: the app shows WoW Forever only. The page's own behaviour is tested in test_forever_frontend.py."""

    def test_no_retail_table_is_read_and_there_is_no_game_switch(self):
        with tempfile.TemporaryDirectory() as t:
            st = run_app(forever_dir=t)
        self.assertEqual(st.parquet_reads, [])
        with open(os.path.join(_app_dir, "app.py"), encoding="utf-8") as f:
            src = f.read()
        for word in ("import lib", "wow-market-data-lake", "st.radio", "GAMES"):
            self.assertNotIn(word, src)


if __name__ == "__main__":
    unittest.main()
