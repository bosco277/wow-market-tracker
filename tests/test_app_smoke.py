"""Runs frontend/app.py against stand-ins for Streamlit and Plotly (neither is installed where the tests run), with
small made-up tables. It cannot judge how the page looks; it catches typos, wrong column names and crashes on empty or old tables."""
import os
import runpy
import sys
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


def fake_streamlit(selected_row=0, found=None, version="1.51.0"):
    st = types.ModuleType("streamlit")
    st.__version__ = version
    st.set_page_config = lambda **k: None
    st.cache_data = lambda *a, **k: (lambda f: f)
    st.secrets = {}
    st.session_state = {}
    st.query_params = {"watch": "2"}
    st.columns = lambda spec, **k: [Col(st) for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda names: [Col(st) for _ in names]
    st.selectbox = lambda label, options, **k: found
    st.multiselect = lambda label, options, default=None, key=None, **k: st.session_state.get(key, default or [])
    st.checkbox = lambda *a, **k: False
    st.number_input = lambda *a, **k: 0.0
    ev = types.SimpleNamespace(selection=types.SimpleNamespace(rows=[selected_row] if selected_row is not None else []))
    seen = []

    def dataframe(df, **k):
        seen.append((df, k))
        return ev
    st.dataframe = dataframe
    st.seen = seen
    st.column_config = Any("column_config")
    for name in ("title", "caption", "error", "warning", "info", "write", "markdown", "subheader", "metric", "table", "plotly_chart",
                 "divider", "expander", "stop"):
        setattr(st, name, Any(name))
    return st


def tables():
    n = 30
    ts = pd.date_range("2026-10-28 00:00", periods=n * 6, freq="h")
    items = pd.DataFrame({
        "item_id": [1, 2, 3], "name": ["Iceweb Spider Silk", "Saronite Ore", ""], "watched": [False, False, False],
        "floor_gold": [1.0, 20.0, 0.5], "daily_low": [900, 5, 100], "daily_est": [1000, 10, 150], "daily_high": [1100, 40, 200],
        "confidence": ["High", "Low", "Medium"], "confidence_reason": ["a", "b", "c"], "now_percentile": [0.5, 0.95, 0.05],
        "band_p10": [0.8, 15.0, 0.4], "band_p90": [1.2, 25.0, 0.7], "wall_est_gold": [1.1, 22.0, 0.6], "wall_low_gold": [1.0, 20.0, 0.5],
        "wall_high_gold": [1.2, 24.0, 0.7], "engine_version": "0.7.6"})
    rng = np.random.default_rng(1)
    hourly = pd.concat([pd.DataFrame({
        "item_id": i, "ts": ts, "floor_gold": 10 * i + rng.normal(0, 0.1, len(ts)), "units_listed": rng.integers(900, 1100, len(ts)),
        "sold_low": 0.5, "sold_est": 1.0, "sold_high": 2.0, "spike": False, "blip": False}) for i in (1, 2, 3)], ignore_index=True)
    shelves = pd.DataFrame({
        "item_id": [1, 2], "name": ["Iceweb Spider Silk", "Saronite Ore"], "verdict": ["candidate", "weak"], "units": [10, 20],
        "avg_price_gold": [1.0, 2.0], "breakeven_gold": [1.05, 2.1], "target_gold": [1.5, 3.0], "net_margin": [0.3, 0.4],
        "net_margin_stress": [0.2, 0.3], "profit_gold": [4.0, 9.0], "profit_stress_gold": [2.0, 5.0], "cost_gold": [10.0, 40.0],
        "sell_hours_low": [3.0, np.inf], "sell_hours_est": [2.0, 4.0], "sell_hours_high": [1.0, 3.0], "confidence": ["Medium"] * 2,
        "daily_low": [1, 2], "daily_est": [3, 4], "daily_high": [5, 6], "engine_version": "0.7.6",
        **{c: ["pass", "warn"] for c in ("test_price", "test_speed", "test_size", "test_margin", "test_regime", "test_book")},
        "long_median_gold": [1.2, 2.5], "long_p10_gold": [1.0, 2.0], "long_p90_gold": [1.4, 3.0], "long_days": [7.0, 5.0]})
    flips = pd.DataFrame({"item_id": [1, 1], "t1": pd.to_datetime(["2026-10-31 17:33:42", "2026-11-01 04:33:45"]),
                          "floor_before_gold": [2.9, 2.7], "floor_after_gold": [3.9, 3.7], "jump": [1.35, 1.37], "units_removed": [93.0, 78.0],
                          "ended_by": ["reverted", "end_of_data"], "revert_hours": [3.0, np.nan], "followup_hours": [3.0, 1.0],
                          "likely_flip": [True, False]})
    cal = pd.DataFrame({"data_last_ts": [pd.Timestamp("2026-11-01 12:00")], "median_range_ratio": [2.9], "n_high": [1], "n_medium": [1], "n_low": [1]})
    return dict(items=items, item_hourly=hourly, profile=pd.DataFrame(), calibration=cal, calibration_history=cal,
                experiment=pd.DataFrame(), flips=flips, shelves=shelves, backtest_summary=pd.DataFrame())


def run_app(data, **kw):  # kw: selected_row, found, version
    st = fake_streamlit(**kw)
    px = types.ModuleType("plotly")
    go = types.ModuleType("plotly.graph_objects")
    for name in ("Figure", "Scatter", "Bar", "Heatmap"):
        setattr(go, name, Any(name))
    px.graph_objects = go
    os.environ["DERIVED_DIR"] = "/nowhere"
    orig_read = pd.read_parquet
    pd.read_parquet = lambda path, *a, **k: data[os.path.basename(path)[:-len(".parquet")]].copy() if os.path.basename(path)[:-8] in data else (_ for _ in ()).throw(FileNotFoundError(path))
    saved_path = list(sys.path)
    sys.path.insert(0, _app_dir)
    sys.modules.pop("lib", None)
    try:
        with mock.patch.dict(sys.modules, {"streamlit": st, "plotly": px, "plotly.graph_objects": go}):
            try:
                runpy.run_path(os.path.join(_app_dir, "app.py"), run_name="__main__")
            except SystemExit:
                pass
    finally:
        pd.read_parquet = orig_read
        sys.path[:] = saved_path
        os.environ.pop("DERIVED_DIR", None)
    return st


class TestAppSmoke(unittest.TestCase):
    def test_runs_and_tables_hold_numbers(self):
        st = run_app(tables(), selected_row=0)
        self.assertGreaterEqual(len(st.seen), 2)                      # Market table and Flips table
        for df, k in st.seen:
            self.assertNotIn("use_container_width", k)               # the deprecated argument is gone on Streamlit 1.51
            self.assertEqual(k.get("width"), "stretch")
        market = next(d for d, k in st.seen if "Sold/day best" in d and "Gold/day best (g)" in d)
        self.assertTrue(pd.api.types.is_numeric_dtype(market["Sold/day best"]))
        flips = next(d for d, k in st.seen if "Profit at target (g)" in d)
        events = next(d for d, k in st.seen if "Units bought out" in d)                 # the item view's past-buyouts table
        self.assertEqual(len(events), 2)
        self.assertTrue(pd.api.types.is_numeric_dtype(flips["Profit at target (g)"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(flips["Hours to sell (slow case)"]))

    def test_old_streamlit_still_works(self):
        out = run_app(tables(), selected_row=0, version="1.40.0")
        for df, k in out.seen:
            self.assertTrue(k.get("use_container_width"))
            self.assertNotIn("width", k)

    def test_search_box_shows_item_and_watch_list_from_address(self):
        st = run_app(tables(), selected_row=None, found="Saronite Ore (2)")
        market = next(d for d, k in st.seen if "Watched" in d)
        self.assertTrue(market.loc[market["Item ID"] == 2, "Watched"].iloc[0])      # ?watch=2
        self.assertFalse(market.loc[market["Item ID"] == 1, "Watched"].iloc[0])

    def test_survives_old_shelves_empty_hourly_and_no_flips(self):
        d = tables()
        d["shelves"] = d["shelves"].drop(columns=["test_regime", "long_median_gold", "daily_est"])
        run_app(d, selected_row=0)
        d = tables()
        d["item_hourly"] = pd.DataFrame()
        d["shelves"] = pd.DataFrame()
        run_app(d, selected_row=1)


def ledger_for_smoke(graded=True):
    """Forecasts for items 1 and 2 (not 3): one guess day, 1 / 3 / 7 days ahead, three methods. With graded=True the 1-day targets already have their real price."""
    rows = []
    for i, origin in ((1, 1.0), (2, 20.0)):
        for h, td in ((1, "2026-10-31"), (3, "2026-11-02"), (7, "2026-11-06")):
            for m in ("naive", "ema25", "kinked"):
                rows.append(dict(made_day="2026-10-30", item_id=i, horizon=h, method=m, guess=origin, lo50=origin * 0.9, hi50=origin * 1.1, lo80=origin * 0.8, hi80=origin * 1.25,
                                 origin=origin, jump_bin=2, target_day=td, actual=origin * 1.05 if (graded and h == 1) else np.nan, made_at="2026-10-31T06:17:00Z", engine_version="0.7.30"))
    return pd.DataFrame(rows)


def named(label):
    return [(a, k) for n, a, k in Any.calls if n == label]


class TestForecastViews(unittest.TestCase):
    def test_item_chart_gets_a_forecast_and_the_tab_gets_a_scoreboard(self):
        d = tables()
        d["forecast_ledger"] = ledger_for_smoke()
        Any.calls.clear()
        st = run_app(d, selected_row=0)                                # the Market and Flips tables each open an item; items 1 and 2 both have forecasts
        scatter = [k for a, k in named("Scatter")]
        lines = [k for k in scatter if str(k.get("name", "")).startswith("forecast: ")]
        self.assertGreaterEqual(len(lines), 1)
        self.assertTrue(all(k["name"] == "forecast: Stays put" for k in lines))                    # the honest baseline is the default
        self.assertTrue(all(len(k["x"]) == 4 for k in lines))                                      # the daily price it starts from, then 1, 3 and 7 days ahead
        names = [k.get("name") for k in scatter]
        for nm in ("forecast 80% range", "forecast 50% range", "daily price (median of the day)"):
            self.assertIn(nm, names)
        past = next(df for df, k in st.seen if "Result" in df)                                      # this item's list of forecasts
        self.assertEqual(len(past), 9)
        self.assertEqual(set(past["Result"]), {"waiting", "inside 50% range"})                     # 1-day targets graded (1.05 x the guess, inside 0.9 - 1.1); 3 and 7 days wait
        board = next(df for df, k in st.seen if "Inside 80% range (%)" in df)
        self.assertEqual(len(board), 3)                                                            # only the 1-day forecasts are graded: three methods
        self.assertTrue(pd.api.types.is_numeric_dtype(board["Typical miss (%)"]))
        self.assertTrue(any(str(k.get("key", "")).startswith("fc_") for a, k in named("plotly_chart")))

    def test_item_without_forecasts_gets_an_explanation(self):
        d = tables()
        d["forecast_ledger"] = ledger_for_smoke()
        Any.calls.clear()
        st = run_app(d, selected_row=None, found="item 3")             # item 3 is not in the ledger
        self.assertFalse([k for a, k in named("Scatter") if str(k.get("name", "")).startswith("forecast")])
        self.assertTrue(any("No forecast for this item yet" in str(a[0]) for a, k in named("caption")))
        self.assertFalse(any("Result" in df for df, k in st.seen))

    def test_nothing_graded_yet_shows_the_waiting_message(self):
        d = tables()
        d["forecast_ledger"] = ledger_for_smoke(graded=False)
        Any.calls.clear()
        st = run_app(d, selected_row=0)
        self.assertTrue(any("Nothing is graded yet" in str(a[0]) for a, k in named("info")))
        waiting = next(df for df, k in st.seen if "Forecasts waiting" in df)
        self.assertEqual(list(waiting["Forecasts waiting"]), [6, 6, 6])        # 2 items x 3 methods at each horizon

    def test_no_ledger_file_at_all(self):
        Any.calls.clear()
        st = run_app(tables(), selected_row=0)                                 # no forecast_ledger table: the page loads it as empty
        self.assertTrue(any("No forecasts recorded yet" in str(a[0]) for a, k in named("info")))
        self.assertFalse([k for a, k in named("Scatter") if str(k.get("name", "")).startswith("forecast")])
        self.assertFalse(any("Result" in df for df, k in st.seen))


if __name__ == "__main__":
    unittest.main()
