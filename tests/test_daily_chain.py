import os
import re
import sys
import unittest

import yaml

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

WF = os.path.join(_root, ".github", "workflows", "daily_chain.yml")
COPY = os.path.join(_root, "tools", "daily_chain.yml")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _load(path=WF):
    text = _read(path)
    return text, yaml.safe_load(text)


def _steps(doc):
    return doc["jobs"]["chain"]["steps"]


class TestDailyChainWorkflow(unittest.TestCase):
    def test_copy_in_tools_is_identical(self):
        self.assertEqual(_read(WF), _read(COPY))

    def test_manual_start_only(self):
        _, doc = _load()
        trig = doc.get("on", doc.get(True))  # PyYAML reads the bare key `on` as True
        self.assertEqual(list(trig), ["workflow_dispatch"])
        self.assertEqual(trig["workflow_dispatch"]["inputs"]["days"]["default"], "4")
        self.assertEqual(trig["workflow_dispatch"]["inputs"]["end_date"]["default"], "")

    def test_never_two_at_once(self):
        _, doc = _load()
        self.assertEqual(doc["concurrency"]["group"], "daily-chain")
        self.assertFalse(doc["concurrency"]["cancel-in-progress"])

    def test_daily_runs_before_the_ledger_and_both_upload(self):
        _, doc = _load()
        runs = [s.get("run", "") for s in _steps(doc)]
        i_daily = next(i for i, r in enumerate(runs) if "lake.py daily" in r)
        i_ledger = next(i for i, r in enumerate(runs) if "lake.py forecast_ledger" in r)
        self.assertLess(i_daily, i_ledger)
        self.assertIn("--upload", runs[i_daily])
        self.assertIn("--upload", runs[i_ledger])
        self.assertIn("--store hf", runs[i_daily])
        self.assertIn("--store hf", runs[i_ledger])

    def test_stops_if_daily_fails(self):
        # `set -o pipefail` makes a failing python (not tee) fail the step, and a failed step skips the next one unless it says always()
        _, doc = _load()
        steps = _steps(doc)
        daily = next(s for s in steps if "lake.py daily" in s.get("run", ""))
        ledger = next(s for s in steps if "lake.py forecast_ledger" in s.get("run", ""))
        self.assertIn("set -o pipefail", daily["run"])
        self.assertIn("set -o pipefail", ledger["run"])
        self.assertNotIn("if", ledger)
        self.assertNotIn("continue-on-error", daily)

    def test_only_the_hf_token_is_used(self):
        text, _ = _load()
        secrets = set(re.findall(r"secrets\.([A-Z0-9_]+)", text))
        self.assertEqual(secrets, {"HF_TOKEN"})

    def test_end_date_is_passed_only_when_given(self):
        _, doc = _load()
        daily = next(s for s in _steps(doc) if "lake.py daily" in s.get("run", ""))
        self.assertIn('if [ -n "$END" ]', daily["run"])
        self.assertIn("--end $END", daily["run"])
        self.assertEqual(daily["env"]["DAYS"], "${{ inputs.days || '4' }}")

    def test_same_download_cache_as_market_tools(self):
        mt = _read(os.path.join(_root, ".github", "workflows", "pick_items.yml"))
        text, _ = _load()
        for needle in ("path: .lake_cache", "key: lake-cache-${{ github.run_id }}", "restore-keys: lake-cache-"):
            self.assertIn(needle, text)
            self.assertIn(needle, mt)

    def test_installs_the_same_packages_as_market_tools(self):
        mt = _read(os.path.join(_root, ".github", "workflows", "pick_items.yml"))
        line = "pip install pandas numpy pyarrow requests tabulate huggingface_hub boto3 scipy"
        text, _ = _load()
        self.assertIn(line, text)
        self.assertIn(line, mt)

    def test_no_account_or_character_names(self):
        text, _ = _load()
        for word in ("bosco", "kohler", "Boscoe"):
            self.assertNotIn(word.lower(), text.lower())


class TestWorkerFiles(unittest.TestCase):
    """The Worker lives outside this repo; if its folder sits next to the engine (as in the build sandbox) check that cron and workflow name agree."""

    def test_worker_cron_and_workflow_agree(self):
        wdir = os.path.join(os.path.dirname(_root), "cloudflare-worker")
        if not os.path.isdir(wdir):
            self.skipTest("cloudflare-worker folder not next to the engine")
        js = _read(os.path.join(wdir, "worker.js"))
        toml = _read(os.path.join(wdir, "wrangler.toml"))
        keys = re.findall(r'^\s*"([^"]+)":\s*\{\s*workflow:\s*"([^"]+)"', js, flags=re.M)
        crons = re.findall(r'"([^"]+)"', re.search(r"crons\s*=\s*\[(.*?)\]", toml, flags=re.S).group(1))
        self.assertEqual(sorted(k for k, _ in keys), sorted(crons))
        for _, wf in keys:
            self.assertTrue(os.path.exists(os.path.join(_root, ".github", "workflows", wf)), wf)


if __name__ == "__main__":
    unittest.main()
