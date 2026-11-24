"""Run the judge proxy that applies config/judge_policy.json to GraphRAG-Bench evaluator requests.

    python scripts/run_judge_proxy.py --max-usd 0.40 --port 8765 --ledger-out results/judge_ledger_<system>.json

Then run the official evaluator UNMODIFIED against it, for each system with the same settings:

    set LLM_API_KEY=unused-by-proxy
    python -m Evaluation.generation_eval --mode API --model gpt-6-luna --base_url http://127.0.0.1:8765/v1 ...

The proxy holds the real OPENAI_API_KEY; the evaluator never sees it. Stop with Ctrl-C; the ledger (policy
hash, request/refusal/failure counts, removed parameters, measured usage and cost) is written on exit.
--dry-run starts and stops the server without an API key and makes no upstream calls.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarks.judge_proxy import JudgeLedger, JudgeProxyServer, load_judge_policy, make_openai_sender  # noqa: E402
from src.core.config import load_experiment_config, validate_environment  # noqa: E402
from src.integrations.ragu_metering import UsageMeter  # noqa: E402
from src.core.env import load_dotenv_if_present  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-usd", type=float, required=True, help="Hard spend cap; calls are refused after it")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ledger-out", default="results/judge_ledger.json")
    parser.add_argument("--label", default="", help="Free-text label, for example the system being scored")
    parser.add_argument("--dry-run", action="store_true", help="Start/stop only; no key, no upstream calls")
    args = parser.parse_args()

    load_dotenv_if_present()
    policy = load_judge_policy()
    meter = UsageMeter(max_usd=args.max_usd)
    ledger = JudgeLedger()
    if args.dry_run:
        def send(path, payload):  # never reached in a dry run
            raise RuntimeError("dry run: upstream disabled")
    else:
        validate_environment()
        config = load_experiment_config()
        send = make_openai_sender(os.environ.get("OPENAI_BASE_URL") or config["base_url"], os.environ["OPENAI_API_KEY"])

    server = JudgeProxyServer(policy, ledger, meter, send, port=0 if args.dry_run else args.port)
    server.start()
    print(f"Judge proxy listening at {server.base_url} (judge {policy.judge_model}, omits {list(policy.omit_parameters)}, "
          f"cap ${args.max_usd}, policy hash {policy.policy_hash[:12]}).")
    stop = threading.Event()
    if not args.dry_run:
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        print("Press Ctrl-C to stop and write the ledger.")
        stop.wait()
    server.stop()
    record = {"label": args.label, "is_mocked": args.dry_run, **ledger.summary(policy, meter)}
    out = (ROOT / args.ledger_out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Ledger written to {out}")


if __name__ == "__main__":
    main()
