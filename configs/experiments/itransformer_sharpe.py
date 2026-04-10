# configs/experiments/itransformer_sharpe.py
"""
Primary experiment: iTransformer backbone with the full hybrid loss.

iTransformer (Liu et al., ICLR 2024) treats each feature channel as a token,
with attention operating across the variable dimension rather than time.
This inductive bias aligns well with financial multivariate series where
cross-variable dependencies (price-volume, spread-volatility) are more
stable than local temporal patterns.

The hybrid loss balances four objectives simultaneously:
    -Sharpe      — core risk-adjusted return objective
    turnover     — keeps position changes proportional to signal strength
    drawdown     — discourages extended underwater periods mid-episode
    terminal     — ensures the model closes positions before the window ends

# Sanity checks
python sanity_check.py --config configs/experiments/itransformer_sharpe.py

# Train
python train.py --config configs/experiments/itransformer_sharpe.py --skip-sanity

# Finding the best thresholds on the VAL dataset
python optimize_thresholds.py \
    --config configs/experiments/itransformer_sharpe.py \
    --trials 50 \
    --objective sharpe

# Evaluate | Default ["test", "backtest"]
python evaluate.py --config configs/experiments/itransformer_sharpe.py --mode test

# Backtest
python evaluate.py --config configs/experiments/itransformer_sharpe.py --mode backtest

# Evaluate "test" + final model
python evaluate.py \
  --config configs/experiments/itransformer_sharpe.py \
  --checkpoint output/itransformer_sharpe/models/final.pth \
  --mode test

# Evaluate "backtest" + final model
python evaluate.py \
  --config configs/experiments/itransformer_sharpe.py \
  --checkpoint output/itransformer_sharpe/models/final.pth \
  --mode backtest

# Optuna — hyperparameter search in debug config
# Quick run: 5 trials, each with 10 epochs
python optimize.py --config configs/experiments/itransformer_sharpe.py --trials 50

# Compare
python compare.py

# rm -r output/itransformer_sharpe; rm -rf data_cache/*
"""

from configs.base_config import MasterConfig, iTransformerConfig, SplitConfig

cfg = MasterConfig(experiment_name="itransformer_sharpe")

# ── Splits ────────────────────────────────────────────────────────────────────

cfg.data.splits = SplitConfig(
    train_start  = "2024-01-01",
    train_end    = "2025-03-31",
    val_end      = "2025-06-30",
    test_end     = "2025-09-30",
    backtest_end = "2025-12-31"
)

# ── Backbone ──────────────────────────────────────────────────────────────────
cfg.backbone.type = "itransformer"

cfg.backbone.itransformer = iTransformerConfig(
    d_model  = 32,
    n_heads  = 2,
    n_layers = 4,
    d_ff     = 64,
    dropout  = 0.1,
)

# ── Loss ──────────────────────────────────────────────────────────────────────
# Key components and their role:
#   lambda_sharpe:      core risk-adjusted return objective
#   lambda_turnover:    light commission drag penalty (not the primary control)
#   lambda_drawdown:    log-stable equity drawdown — smoother gradients than cumprod
#   lambda_terminal:    encourages flat position at episode end
#   lambda_bias:        |mean(position)| — prevents buy&hold / always-long collapse
#   lambda_flat_target: (flat_soft - flat_target)^2 — prevents permanent cash collapse
cfg.loss.type            = "sharpe"

cfg.loss.lambda_sharpe   = 1.0
cfg.loss.lambda_turnover = 0.01
cfg.loss.lambda_drawdown = 0.05
cfg.loss.lambda_terminal = 0.01
cfg.loss.lambda_bias     = 0.25

"""
Diagnostics:
    flat collapse (flat → 1.0):    increase lambda_flat_target or reduce lambda_turnover
    buy&hold (flat → 0, long → 1): increase lambda_bias
    churning (commission > 1%):    increase lambda_turnover
"""
cfg.loss.lambda_flat_target = 0.5
cfg.loss.flat_target        = 0.20 # target: 20% of bars in cash
cfg.loss.flat_eps           = 0.05 # sync with cfg.eval.exit_thr
cfg.loss.flat_k             = 40.0

# ── Training ──────────────────────────────────────────────────────────────────
cfg.training.num_epochs    = 30
cfg.training.batch_size    = 64
cfg.training.lr            = 1e-3
cfg.training.weight_decay  = 1e-5
cfg.training.warmup_epochs = 2
cfg.training.gradient_clip = 1.0
cfg.training.val_freq      = 2
cfg.training.full_bptt     = True # strict end-to-end differentiability
cfg.training.mirror_augmentation = True

# ── Data ──────────────────────────────────────────────────────────────────────
cfg.data.timeframe_min     = 30
cfg.data.context_len       = 96
cfg.data.horizon_len       = 24
cfg.data.stride            = 24

cfg.data.preset            = "ohlcv" # ohlc ohlcv full custom
# used only when preset="custom"
# cfg.data.feature_columns   = ["high", "low", "close", "volume"]
cfg.data.add_time_features = False
cfg.data.add_typical_price = False

cfg.data.add_rolling_vol = True
cfg.data.vol_window      = 12
cfg.data.vol_window_vol  = 12

# ── Policy head ───────────────────────────────────────────────────────────────
cfg.policy.dense_dims    = [64, 32]
cfg.policy.tau_direction = 1.0
cfg.policy.tau_gate      = 1.0
cfg.policy.dropout       = 0.1

# ── Simulator ─────────────────────────────────────────────────────────────────
cfg.simulator.commission_rate = 0.0004
cfg.simulator.slippage_rate   = 0.0003

# ── Evaluation ────────────────────────────────────────────────────────────────
cfg.eval.enter_long_thr         = 0.1  # 0.2
cfg.eval.enter_short_thr        = 0.1  # 0.2
cfg.eval.exit_thr               = 0.05 # must match cfg.loss.flat_eps above
cfg.eval.min_delta_to_rebalance = 0.03

