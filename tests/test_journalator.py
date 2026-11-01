import os
import sys
import tempfile
import unittest
import zlib

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
from engine import journalator as j  # noqa: E402

ALPHA = j._ALPHABET


class Enc:
    """Writes the same binary form as LibSerialize (the subset Journalator uses) so the reader can be tested without a real file."""

    def __init__(self):
        self.out = bytearray([1])
        self.strs, self.tabs = {}, {}

    def int_(self, n):
        if 0 <= n < 128:
            self.out.append((n << 1) | 1)
        elif -4096 < n < 4096:
            v = abs(n)
            self.out += bytes([((v & 15) << 4) | (8 if n < 0 else 0) | 4, v >> 4])
        else:
            for idx, size in ((1, 2), (3, 3), (5, 4), (7, 8)):
                if n < 256 ** size:
                    self.out.append(idx << 3)
                    self.out += n.to_bytes(size, "big")
                    return
            raise ValueError(n)

    def str_(self, s):
        if len(s) > 2 and s in self.strs:
            self.out += bytes([26 << 3, self.strs[s]])
            return
        b = s.encode()
        if len(b) > 2:
            self.strs[s] = len(self.strs) + 1
        if len(b) < 16:
            self.out.append((len(b) << 4) | 2)
        else:
            self.out += bytes([14 << 3, len(b)])
        self.out += b

    def val(self, v):
        if isinstance(v, bool):
            self.out.append((12 if v else 13) << 3)
        elif isinstance(v, int):
            self.int_(v)
        elif isinstance(v, str):
            self.str_(v)
        elif isinstance(v, dict):
            if id(v) in self.tabs:
                self.out += bytes([29 << 3, self.tabs[id(v)]])
                return
            self.tabs[id(v)] = len(self.tabs) + 1
            if len(v) < 16:
                self.out.append((len(v) << 4) | 6)
            else:
                self.out += bytes([17 << 3, len(v)])
            for k, x in v.items():
                self.val(k)
                self.val(x)
        elif isinstance(v, list):
            self.tabs[id(v)] = len(self.tabs) + 1
            if len(v) < 16:
                self.out.append((len(v) << 4) | 10)
            else:
                self.out += bytes([20 << 3, len(v)])
            for x in v:
                self.val(x)
        else:
            raise TypeError(v)


def encode_print(b):
    bits = n = 0
    out = []
    for x in b:
        bits |= x << n
        n += 8
        while n >= 6:
            out.append(ALPHA[bits & 63])
            bits >>= 6
            n -= 6
    if n:
        out.append(ALPHA[bits & 63])
    return "".join(out)


def pack(logs):
    e = Enc()
    e.val(logs)
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    return encode_print(c.compress(bytes(e.out)) + c.flush())


T0 = 1793470000          # Sat 2026-10-31 14:06:40 EDT


def link(i, name):
    return f"|cnIQ1:|Hitem:{i}::::::::65:252:::::::::|h[{name}]|h|r"


def sample_logs():
    src = {"character": "Kruzzik", "realm": "Skullcrusher", "faction": "Horde"}
    inv = [
        dict(itemName="Large Prismatic Shard", source=src, invoiceType="buyer", count=96, time=T0 + 100, value=3010900, consignment=0,
             itemLink=link(22449, "Large Prismatic Shard"), deposit=0),
        dict(itemName="Fiery Core", source=src, invoiceType="buyer", count=354, time=T0 + 110, value=1639300, consignment=0,
             itemLink=link(17010, "Fiery Core"), deposit=0),
        dict(itemName="Large Prismatic Shard", source=src, playerName="Multiple Buyers", invoiceType="seller", count=96, value=6950400,
             time=T0 + 11000, consignment=347520, deposit=100),
        dict(itemName="Fiery Core", source=src, playerName="Multiple Buyers", invoiceType="seller", count=145, value=928000,
             time=T0 + 20000, consignment=46400, deposit=43500),
    ]
    post = [
        dict(itemName="Large Prismatic Shard", buyout=72400, source=src, count=96, time=T0 + 200, itemLink=link(22449, "Large Prismatic Shard"), deposit=100),
        dict(itemName="Fiery Core", buyout=6400, source=src, count=354, time=T0 + 210, itemLink=link(17010, "Fiery Core"), deposit=106200),
        dict(itemName="Fiery Core", buyout=5700, source=src, count=209, time=T0 + 40000, itemLink=link(17010, "Fiery Core"), deposit=62700),
    ]
    fail = [dict(itemName="Fiery Core", time=T0 + 39000, count=209, source=src, itemLink=link(17010, "Fiery Core"), failedType="expired")]
    return {"Invoices": inv, "Posting": post, "Failures": fail, "Trades": [], "Version": 15}


def write_lua(logs, extra=""):
    data = pack(logs)
    text = ('\nJOURNALATOR_CONFIG = {\n["debug"] = false,\n}\nJOURNALATOR_LOGS = nil\nJOURNALATOR_ARCHIVE = {\n["stores"] = {\n["SometimesLocked"] = {\n'
            f'["Logs-{T0}"] = {{\n["timestamp"] = {T0 + 99999},\n["version"] = 1,\n["data"] = "{data}",\n}},\n}},\n["RawData"] = {{\n}},\n}},\n'
            '["internalVersion"] = 1,\n}\nJOURNALATOR_ARCHIVE_TIMES = {\n' + f"{T0},\n}}\n" + extra)
    fd, path = tempfile.mkstemp(suffix=".lua")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


class TestDecode(unittest.TestCase):
    def test_print_encoding_round_trip(self):
        b = bytes(range(256)) * 3
        self.assertEqual(j.decode_print(encode_print(b))[:len(b)], b)

    def test_numbers_strings_tables_and_references(self):
        shared = {"character": "Kruzzik", "realm": "Skullcrusher"}
        obj = {"list": [{"n": v, "src": shared, "name": "Fiery Core"} for v in (0, 127, 128, 150, 165, 2000, 4095, 70000, 154900, 2499800, 1793470496)]}
        e = Enc()
        e.val(obj)
        out = j._Reader(bytes(e.out)).value()
        self.assertEqual([x["n"] for x in out["list"]], [0, 127, 128, 150, 165, 2000, 4095, 70000, 154900, 2499800, 1793470496])
        self.assertTrue(all(x["src"] is out["list"][0]["src"] for x in out["list"]))         # table references resolve to the same table
        self.assertEqual({x["name"] for x in out["list"]}, {"Fiery Core"})                    # string references resolve
        self.assertEqual(out["list"][0]["src"]["realm"], "Skullcrusher")

    def test_negative_12_bit_and_long_strings(self):
        e = Enc()
        e.val({"neg": -300, "long": "x" * 40, "flag": True, "off": False})
        out = j._Reader(bytes(e.out)).value()
        self.assertEqual((out["neg"], out["long"], out["flag"], out["off"]), (-300, "x" * 40, True, False))

    def test_bad_payload_is_a_clear_error(self):
        with self.assertRaises(ValueError):
            j._Reader(b"\x02abc")


class TestTables(unittest.TestCase):
    def setUp(self):
        self.path = write_lua(sample_logs())
        self.logs = j.read_savedvars(self.path)
        self.t = j.tables(self.logs)

    def tearDown(self):
        os.remove(self.path)

    def test_read_file_and_clean_tables(self):
        self.assertEqual((len(self.logs["Invoices"]), len(self.logs["Posting"]), len(self.logs["Failures"])), (4, 3, 1))
        self.assertEqual(self.logs["warnings"], [])
        b, s = self.t["purchases"], self.t["sales"]
        self.assertEqual(list(b["item_id"]), [22449, 17010])
        self.assertAlmostEqual(b["unit_gold"].iloc[0], 3010900 / 96 / 1e4)
        self.assertAlmostEqual(s["net_gold"].iloc[0], (6950400 - 347520) / 1e4)              # the 5% cut comes off; the deposit comes back separately
        self.assertAlmostEqual(s["cut_gold"].iloc[0], s["gross_gold"].iloc[0] * 0.05)
        self.assertTrue(s["item_id"].notna().all())                                            # sale invoices have no link: filled in from the name
        self.assertEqual(str(self.t["postings"]["time_utc"].dt.tz), "UTC")

    def test_matching_sales_expiries_and_deposits(self):
        L = j.match_listings(self.t)
        self.assertEqual(list(L["status"]), ["sold", "part sold, rest expired", "open"])
        lps, fc1, fc2 = L.iloc[0], L.iloc[1], L.iloc[2]
        self.assertEqual((lps["units_sold"], lps["units_open"]), (96, 0))
        self.assertEqual((fc1["units_sold"], fc1["units_expired"], fc1["units_open"]), (145, 209, 0))
        self.assertAlmostEqual(fc1["deposit_lost_gold"], 106200 * 209 / 354 / 1e4)
        self.assertEqual((fc2["units_sold"], fc2["units_open"]), (0, 209))
        self.assertAlmostEqual(lps["hours_to_collect"], (11000 - 200) / 3600.0)
        self.assertEqual(L["units_sold"].sum(), self.t["sales"]["units"].sum())              # every sold unit is accounted for
        self.assertAlmostEqual(L["net_gold"].sum(), self.t["sales"]["net_gold"].sum())

    def test_flip_ledger_numbers(self):
        F = j.flip_ledger(self.t).set_index("item_id")
        lps = F.loc[22449]
        self.assertAlmostEqual(lps["profit_on_sold_gold"], (6950400 - 347520 - 3010900) / 1e4)
        self.assertAlmostEqual(lps["margin_on_sold"], (6950400 - 347520) / 3010900 - 1)
        fc = F.loc[17010]
        cost_per = 1639300 / 354
        self.assertAlmostEqual(fc["cost_of_sold_gold"], cost_per * 145 / 1e4)
        self.assertAlmostEqual(fc["unsold_at_cost_gold"], cost_per * 209 / 1e4)
        self.assertAlmostEqual(fc["deposit_lost_gold"], 106200 * 209 / 354 / 1e4)

    def test_empty_and_unarchived_files_give_warnings(self):
        fd, p = tempfile.mkstemp(suffix=".lua")
        with os.fdopen(fd, "w") as f:
            f.write('JOURNALATOR_LOGS = {\n["Invoices"] = {},\n}\nJOURNALATOR_ARCHIVE = {\n}\n')
        try:
            w = j.read_savedvars(p)["warnings"]
            self.assertEqual(len(w), 2)
            self.assertTrue(any("/reload" in x for x in w))
        finally:
            os.remove(p)
        self.assertTrue(j.flip_ledger({}).empty)
        self.assertTrue(j.match_listings({}).empty)

    def test_same_record_in_two_archive_blocks_is_counted_once(self):
        p2 = write_lua(sample_logs())
        try:
            with open(p2) as f:
                text = f.read()
            block = text[text.index('["Logs-'):text.index('},\n["RawData"]')]
            text = text.replace(block, block + block.replace(f"Logs-{T0}", f"Logs-{T0 + 1}"))
            with open(p2, "w") as f:
                f.write(text)
            self.assertEqual(len(j.read_savedvars(p2)["Invoices"]), 4)
        finally:
            os.remove(p2)


if __name__ == "__main__":
    unittest.main()


class TestLedgerCommand(unittest.TestCase):
    def test_command_prints_and_writes_files_without_a_snapshot_store(self):
        import contextlib
        import io
        sys.path.insert(0, os.path.join(_root, "tools"))
        import lake
        path = write_lua(sample_logs())
        with tempfile.TemporaryDirectory() as d:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    lake.main(["ledger", "--log", path, "--out", d])
            finally:
                os.remove(path)
            out = buf.getvalue()
            self.assertIn("EACH LISTING", out)
            self.assertIn("Fiery Core", out)
            self.assertIn("TOTAL: profit on sold units", out)
            self.assertTrue(os.path.exists(os.path.join(d, "ledger_flips.csv")))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                lake.main(["ledger", "--log", os.path.join(d, "missing.lua"), "--out", d])
            self.assertIn("File not found", buf.getvalue())


class TestLedgerPrivacy(unittest.TestCase):
    def run_cmd(self, argv, fetch=None):
        import contextlib
        import io
        sys.path.insert(0, os.path.join(_root, "tools"))
        import lake
        buf = io.StringIO()
        orig = lake._fetch_private_file
        if fetch is not None:
            lake._fetch_private_file = fetch
        try:
            with contextlib.redirect_stdout(buf):
                lake.main(argv)
        finally:
            lake._fetch_private_file = orig
        return buf.getvalue()

    def test_hf_path_is_read_and_no_names_reach_the_log_or_files(self):
        path = write_lua(sample_logs())
        asked = []
        with tempfile.TemporaryDirectory() as d:
            try:
                out = self.run_cmd(["ledger", "--log", "hf:system/Journalator.lua", "--out", d], fetch=lambda rel: asked.append(rel) or path)
            finally:
                os.remove(path)
            self.assertEqual(asked, ["system/Journalator.lua"])
            self.assertIn("TOTAL: profit on sold units", out)
            blob = out
            for f in os.listdir(d):
                if f.endswith(".csv"):
                    with open(os.path.join(d, f)) as fh:
                        blob += fh.read()
            for secret in ("Kruzzik", "Skullcrusher", "Multiple Buyers", "Horde"):
                self.assertNotIn(secret, blob)

    def test_missing_token_or_file_gives_a_plain_message(self):
        import lake
        old = os.environ.pop("HF_TOKEN", None)
        try:
            self.assertIsNone(lake._fetch_private_file("system/Journalator.lua"))
        finally:
            if old:
                os.environ["HF_TOKEN"] = old
        out = self.run_cmd(["ledger", "--log", "hf:system/Journalator.lua"], fetch=lambda rel: None)
        self.assertNotIn("Traceback", out)
