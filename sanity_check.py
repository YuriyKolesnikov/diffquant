# sanity_check.py
"""
DiffQuant — standalone sanity check entry point.

Run this before any training to verify:
    1. Gradient flows from PnL loss to every model parameter.
    2. Model learns long bias on a synthetic uptrend.
    3. Model learns short bias on a synthetic downtrend.

Usage:
    python sanity_check.py --config configs/experiments/itransformer_hybrid.py
    python sanity_check.py --config configs/experiments/lstm_hybrid.py
"""

import argparse
import logging
import sys

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers = [logging.StreamHandler(sys.stdout)],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="DiffQuant sanity checks")
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()

    from configs.experiments      import load_config
    from model.policy_network     import PolicyNetwork
    from simulator.diff_simulator import DiffSimulator, SimConfig
    from sanity.checks            import run_all_checks

    cfg   = load_config(args.config)
    model = PolicyNetwork(cfg)
    sim   = DiffSimulator(SimConfig(
        commission_rate = cfg.simulator.commission_rate,
        slippage_rate   = cfg.simulator.slippage_rate,
    ))

    ok = run_all_checks(model, sim, cfg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

