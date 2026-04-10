# configs/base_config.py
"""
DiffQuant — Base Configuration
================================
Single source of truth for all hyperparameters across the pipeline.

Design principles:
    - Every parameter has a documented rationale, not just a description.
    - Derived quantities (e.g. n_features) are computed automatically.
    - Configs are fully serializable via .model_dump() for checkpoint storage.
    - Experiment configs inherit from MasterConfig and override selectively.

Usage:
    from configs.base_config import MasterConfig

    cfg = MasterConfig(experiment_name="my_run")
    cfg.training.lr = 1e-4  # override specific fields
"""

from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field, computed_field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

class SplitConfig(BaseModel):
    """
    Strict temporal boundaries for dataset splits.
    All splits are non-overlapping. The order must be:
        train_end < val_end < test_end < backtest_end

    Backtest period is held out until final evaluation only —
    it must never be used for any form of model selection.
    """
    #     Train    : 2021-01-01 → 2025-03-31  (~4.25 years)
    #     Val      : 2025-04-01 → 2025-06-30  (3 months — model selection)
    #     Test     : 2025-07-01 → 2025-09-30  (3 months — out-of-sample)
    #     Backtest : 2025-10-01 → 2025-12-31  (3 months — final hold-out)

    train_start:  str | None = None        # None = use all data from beginning
    train_end:    str = "2025-03-31"
    val_end:      str = "2025-06-30"
    test_end:     str = "2025-09-30"
    backtest_end: str = "2025-12-31"


# Feature presets — named subspaces for quick experimentation.
# Using a dict here (not Enum) keeps it extensible without subclassing.
FEATURE_PRESETS: dict[str, list[str]] = {
    # Pure price action — minimal input, maximum interpretability.
    "ohlc":    ["open", "high", "low", "close"],

    # Price + volume — standard starting point for most research.
    "ohlcv":   ["open", "high", "low", "close", "volume"],

    # All raw exchange fields — full information from the source .npz.
    "full":    ["open", "high", "low", "close", "volume", "num_trades"],
}
_SOURCE_COLUMNS = {"open", "high", "low", "close", "volume", "num_trades"}


class DataConfig(BaseModel):
    """
    Controls raw data loading, aggregation, and sample construction.

    Feature selection works in two modes:

        preset="ohlc" / "ohlcv" / "full"
            Selects a named subspace. feature_columns is ignored.
            Use this for standard experiments and ablations.

        preset="custom"
            Uses feature_columns exactly as specified.
            Any column name present in the source .npz is valid.

    Optional computed features (stacked after the base columns):

        add_typical_price: bool
            Volume-weighted average price, computed during aggregation
            from 1-min source data. Only meaningful when timeframe_min > 1.
            Adds 1 channel.

        add_time_features: bool
            Cyclic encoding of hour-of-day and day-of-week.
            [sin_hour, cos_hour, sin_dow, cos_dow] — 4 channels.
            Captures intraday and weekly seasonality patterns.

    The final channel count (n_features) is derived automatically
    and exposed as a computed field consumed by the backbone configs.
    """

    timeframe_min: int = 5   # [1, 5, 10, 30, 60]

    # Sample geometry
    context_len: int = 96   # bars → model input window
    horizon_len: int = 24   # bars → differentiable simulator window
    stride:      int = 24   # sampling step; stride=horizon_len → non-overlapping

    # Feature selection
    preset: Literal["ohlc", "ohlcv", "full", "custom"] = "ohlcv"
    feature_columns: List[str] = []   # used only when preset="custom"
    # custom subset
    # cfg.data.preset = "custom"
    # cfg.data.feature_columns = ["close", "volume"]

    # Optional computed channels
    add_typical_price: bool = False  # (H+L+C)/3 log-return; not true VWAP
    add_time_features: bool = True
    add_rolling_vol:   bool = False  # log(rolling_std(close_ret, vol_window))

    # Rolling normalisation windows — causal estimators, no look-ahead.
    # vol_window bars for rolling volume mean and rolling volatility std.
    # Default 48 bars = 24h at 30-min, 8h at 5-min, 4h at 1-min (approximately).
    # Increase for smoother estimates; decrease for faster regime adaptation.
    vol_window: int = Field(default=48, gt=0)
    vol_window_vol: int = Field(default=20, gt=0)

    # Simulator needs raw close prices — referenced by name, never by index.
    close_column: str = "close"

    splits: SplitConfig = SplitConfig()

    @model_validator(mode="after")
    def _validate(self) -> "DataConfig":
        # Validate preset
        if self.preset == "custom" and not self.feature_columns:
            raise ValueError(
                "preset='custom' requires feature_columns to be non-empty."
            )
        if self.preset not in {*FEATURE_PRESETS, "custom"}:
            raise ValueError(
                f"Unknown preset: {self.preset!r}. "
                f"Valid: {list(FEATURE_PRESETS)} + ['custom']"
            )
        # Validate typical_price requirement
        if self.add_typical_price and self.timeframe_min == 1:
            raise ValueError(
                "add_typical_price=True requires timeframe_min > 1. "
                "Typical price on 1-min bars adds no information beyond close."
            )
        
        # close_column is extracted from raw source bars (not from feature_columns).
        # Validate it is a recognised source column.
        if self.close_column not in _SOURCE_COLUMNS:
            raise ValueError(
                f"close_column={self.close_column!r} is not a valid source column. "
                f"Valid: {sorted(_SOURCE_COLUMNS)}"
            )
    
        return self

    @computed_field
    @property
    def active_feature_set(self) -> List[str]:
        # Resolve base columns from preset at call time — same logic as resolve_n_features().
        if self.preset == "custom":
            cols = list(self.feature_columns)
        else:
            cols = list(FEATURE_PRESETS.get(self.preset, []))
        if self.add_typical_price:
            cols.append("typical_price")
        if getattr(self, "add_rolling_vol", False):
            cols.append("rolling_vol")
        if self.add_time_features:
            cols.extend(["sin_hour", "cos_hour", "sin_dow", "cos_dow"])
        return cols
    
    def resolve_n_features(self) -> int:
        """
        Runtime computation of feature count.
        Bypasses pydantic computed_field to guarantee correct value
        after post-init preset mutations in experiment configs.
        """
        if self.preset == "custom":
            base = len(self.feature_columns) if self.feature_columns else 0
        else:
            base = len(FEATURE_PRESETS.get(self.preset, []))
        if self.add_typical_price:
            base += 1
        if getattr(self, "add_rolling_vol", False):
            base += 1
        if self.add_time_features:
            base += 4
        return base


# ─────────────────────────────────────────────────────────────────────────────
# Model backbones
# ─────────────────────────────────────────────────────────────────────────────

class iTransformerConfig(BaseModel):
    """
    iTransformer backbone hyperparameters.
    Reference: Liu et al., "iTransformer: Inverted Transformers are Effective
    for Time Series Forecasting", ICLR 2024.

    Key idea: each feature channel becomes a token (not each timestep).
    Attention models inter-channel dependencies across the full context window.
    This is architecturally superior for financial data where channels
    (price, volume, typical_price) exhibit stable cross-variable relationships.

    Output dimension: n_channels * d_model (flattened for the policy head).

    d_model=128, n_layers=4 is a practical default for financial time series:
    large enough to capture non-linear dependencies, small enough to train
    on a single GPU without overfitting on ~20k samples.
    """
    d_model:  int   = 128
    n_heads:  int   = 8     # must divide d_model evenly
    n_layers: int   = 4
    d_ff:     int   = 256   # feedforward expansion; typically 2* d_model
    dropout:  float = 0.1

    # use_input_norm


class LSTMConfig(BaseModel):
    """
    LSTM encoder backbone hyperparameters.

    The LSTM processes the context window sequentially and returns
    the final hidden state as the feature vector.

    Bidirectional is valid here: at inference time the full context window
    [t-ctx : t] is observed before any trading decision is made,
    so future context information within the window does not leak.

    hidden_size=128 matches iTransformer d_model for fair comparison.
    num_layers=2 with dropout=0.2 is standard for financial time series
    to prevent overfitting while maintaining sufficient representational capacity.
    """
    hidden_size:   int   = 128
    num_layers:    int   = 2
    dropout:       float = 0.2   # applied between LSTM layers (num_layers > 1)
    bidirectional: bool  = True  # valid: full context is observed before decision


class BackboneConfig(BaseModel):
    """
    Selects and configures the feature extraction backbone.
    Only the active backbone's parameters are used at runtime.
    """
    type: Literal["itransformer", "lstm"] = "itransformer"

    itransformer: iTransformerConfig = iTransformerConfig()
    lstm:         LSTMConfig         = LSTMConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Policy head
# ─────────────────────────────────────────────────────────────────────────────

class PolicyConfig(BaseModel):
    """
    Two-headed policy output: direction * gate.

    Output equation:
        position = tanh(direction_raw / τ_dir) * sigmoid(gate_raw / τ_gate)

    direction ∈ (-1, +1): encodes the alpha signal — which way and how strongly.
    gate      ∈ ( 0,  1): encodes confidence — whether to trade at all.
    position  ∈ (-1, +1): the actual weight sent to the differentiable simulator.

    The gate head is initialized with a negative bias (-1.0) so the model
    starts near-flat, preventing large random trades in early training.

    Temperature parameters (τ_dir, τ_gate):
        τ > 1.0 → softer decisions (more continuous, easier gradients early on)
        τ < 1.0 → sharper decisions (approaching ±1 / 0-1 boundaries)
        τ = 1.0 → standard tanh/sigmoid behavior (sensible default)

    Extras vector fed alongside backbone features (additional_feats=4):
        [prev_position, prev_delta_position, time_elapsed, time_remaining]
        These give the model path-dependent context without action one-hot encoding,
        which would introduce discrete variables incompatible with backprop.
    """
    dense_dims:       List[int] = [128, 64]
    tau_direction: float = Field(default=1.0, gt=0.0,
        description="Temperature for direction head. Must be strictly positive.")
    tau_gate:      float = Field(default=1.0, gt=0.0,
        description="Temperature for gate head. Must be strictly positive.")
    dropout:          float     = 0.1
    additional_feats: int       = 4


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable simulator
# ─────────────────────────────────────────────────────────────────────────────

class SimulatorConfig(BaseModel):
    """
    Differentiable trading simulator parameters.

    Transaction cost model:
        gross_pnl_t = prev_position_t * return_t
        cost_t      = smooth_abs(Δposition_t) * (commission + slippage)
        net_pnl_t   = gross_pnl_t - cost_t

    smooth_abs(x) = sqrt(x² + ε) replaces |x| to ensure C∞ differentiability.
    The standard abs() has an undefined subgradient at x=0, causing
    gradient instability when the model is near the flat zone.

    Cost rates match Binance USDT-margined Futures defaults:
        commission_rate = 0.0004  (0.04% taker fee)
        slippage_rate   = 0.0003  (conservative 0.03% mid-price slippage estimate)

    Market impact (Almgren-Chriss quadratic model) is disabled by default.
    Enable for large-capital experiments where order size affects price:
        cost_t += market_impact_eta * Δposition_t²
    """
    commission_rate:   float = 0.0004
    slippage_rate:     float = 0.0003
    smooth_abs_eps:    float = 1e-6
    market_impact_eta: float = 0.0   # 0.0 = disabled


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class LossConfig(BaseModel):
    """
    Training objective configuration.

    Three loss modes (cfg.loss.type):
        sharpe  — maximise Sharpe ratio: -(μ / sigma)
        sortino — maximise Sortino ratio: -(μ / sigma_down)
                  Fully differentiable via ReLU downside std, no graph breaks.
        hybrid  — weighted combination of financial objectives and
                  behavioural regularisers. Recommended for production.

    Hybrid mode core objective (cfg.loss.type_hybrid):
        sharpe_hybrid  — uses Sharpe as the core objective (default)
        sortino_hybrid — uses Sortino as the core objective

    Hybrid loss structure:

        L = λ_sharpe      · (-Sharpe or -Sortino)
          + λ_turnover    · mean(smooth_abs(Δpos[1:]))
          + λ_drawdown    · mean(log_running_max - log_equity)
          + λ_terminal    · smooth_abs(pos_T)
          + λ_flat_target · (flat_soft - flat_target)²
          + λ_bias        · |mean(position)|

    Behavioural regularisers:

        flat_target:
            Target fraction of bars in cash. (flat_soft - flat_target)² penalises
            both flat collapse (flat → 1) and permanent full exposure (flat → 0).
            flat_soft = sigmoid(flat_k · (flat_eps - |pos|)) — differentiable proxy.
            flat_eps must equal cfg.eval.exit_thr for loss / eval consistency.

        bias:
            |mean(position)| penalises always-long or always-short behaviour.
            Proved critical for breaking long bias on trending BTC training data.

    Diagnostics:
        flat collapse (flat → 1.0):    increase lambda_flat_target or reduce lambda_turnover
        buy&hold (flat → 0, long → 1): increase lambda_bias
        churning (commission > 1%):    increase lambda_turnover
    """
    type:        Literal["sharpe", "sortino", "hybrid"]     = "hybrid"
    type_hybrid: Literal["sharpe_hybrid", "sortino_hybrid"] = "sharpe_hybrid"

    # Weighted loss components
    lambda_sharpe:      float = Field(default=1.0,  ge=0.0)
    lambda_turnover:    float = Field(default=0.01, ge=0.0)
    lambda_drawdown:    float = Field(default=0.05, ge=0.0)
    lambda_terminal:    float = Field(default=0.01, ge=0.0)
    lambda_flat_target: float = Field(default=0.0,  ge=0.0)
    lambda_bias:        float = Field(default=0.0,  ge=0.0)

    # Flat target parameters — flat_eps must equal cfg.eval.exit_thr
    flat_target: float = Field(default=0.20, ge=0.0, le=1.0)
    flat_eps:    float = Field(default=0.05, ge=0.0)
    flat_k:      float = Field(default=40.0, gt=0.0)

    # Numerical stability in Sharpe / Sortino denominator
    eps: float = Field(default=1e-8, gt=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    """
    Training loop hyperparameters.

    Optimizer: AdamW with (β₁=0.9, β₂=0.95).
        β₂=0.95 (vs default 0.999) provides faster adaptation
        to non-stationary financial gradients.

    LR schedule: Linear warmup → Cosine annealing.
        Warmup prevents large early gradient steps through the
        long BPTT chain before the policy has meaningful outputs.
        Cosine annealing provides smooth convergence without
        manual step-decay tuning.

    gradient_clip: Clips gradient L2 norm to prevent explosion
        through the recurrent episode rollout. Critical for LSTM,
        important for iTransformer.

    num_workers: DataLoader workers. 0 = main process (safe for debugging).
        Set to 2–4 for training runs on machines with sufficient RAM.

    val_freq: Run walk-forward validation every N epochs.
        Walk-forward is slower than batched validation (~52s for 3-month val
        at 5-min resolution on CPU), so we don't run it every epoch.
    """
    num_epochs:    int   = 300
    batch_size:    int   = 64
    lr:            float = 3e-4
    weight_decay:  float = 1e-5
    gradient_clip: float = 1.0
    warmup_epochs: int   = 15
    val_freq:      int   = 10
    num_workers:   int   = 0

    # If True: gradients flow through the full recurrent extras chain
    # (prev_pos, prev_delta) — strict end-to-end differentiability.
    # If False: extras are detached — truncated BPTT, shorter gradient paths,
    # lower explosion risk on unstable initializations.
    # Gradient clipping handles stability in both modes.
    full_bptt: bool = True

    # If True: with probability 0.5 each batch, inverts price returns and
    # flips positions. Forces the model to learn symmetric long/short behaviour.
    # Useful when training data is predominantly trending in one direction.
    # Adds ~50% compute overhead per epoch.
    mirror_augmentation: bool = False

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation & backtest
# ─────────────────────────────────────────────────────────────────────────────

class EvalConfig(BaseModel):
    """
    Walk-forward evaluation and backtest execution parameters.

    Hysteresis thresholds prevent rapid oscillation ("chatter") at
    decision boundaries. Two separate thresholds create a dead-band:
        - entry requires |position| > enter_thr  (strong signal needed to open)
        - exit  requires |position| < exit_thr   (weak signal triggers close)

    This asymmetry is intentional: it is cheaper to miss a small opportunity
    than to repeatedly open and close positions, paying commission both ways.

    min_delta_to_rebalance: minimum absolute position change to trigger a trade.
        Changes smaller than this are ignored to avoid micro-transactions
        that drain capital via commission without meaningful exposure change.

    Threshold interpretation:
        position >  enter_long_thr  → enter / hold long
        position < -enter_short_thr → enter / hold short
        |position| < exit_thr       → close position (flat)
        otherwise                   → maintain current position (hysteresis zone)
    """
    enter_long_thr:         float = 0.15
    enter_short_thr:        float = 0.15
    exit_thr:               float = 0.05
    min_delta_to_rebalance: float = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

class PathsConfig(BaseModel):
    """
    Filesystem paths for data, cache, and output artifacts.

    source_data: path to the raw 1-minute .npz file downloaded from HuggingFace.
    cache_dir:   processed datasets are cached here keyed by config MD5 hash.
                 Delete this directory to force full rebuild.
    output_dir:  all artifacts for one experiment are written here:
                 model checkpoints, logs, plots, reports.
                 The {experiment_name} token is resolved by MasterConfig.
    """
    source_data: str = "data_source/btcusdt_1min_2021_2025.npz"
    cache_dir:   str = "data_cache/"
    output_dir:  str = "output/{experiment_name}"


# ─────────────────────────────────────────────────────────────────────────────
# Master config
# ─────────────────────────────────────────────────────────────────────────────

class MasterConfig(BaseModel):
    """
    DiffQuant master configuration.

    All experiment configs inherit from this class and selectively override
    fields. This ensures every experiment has complete, explicit hyperparameters
    with no hidden defaults.

    experiment_name: used as the output directory name and in all logs.
                     Choose descriptive names: "itransformer_hybrid_v1",
                     not "run1" or "test".

    device: "cpu" for development and debugging (always works),
            "cuda" for training runs (requires NVIDIA GPU with CUDA),
            "mps" for Apple Silicon (M1/M2/M3 Macs).

    seed: controls all sources of randomness (torch, numpy).
          Fixed seed is mandatory for reproducible research.
          Change seed across runs to measure variance, not to cherry-pick.
    """
    experiment_name: str = "diffquant_base"
    device:          str = "cpu"
    seed:            int = 42

    data:      DataConfig      = DataConfig()
    backbone:  BackboneConfig  = BackboneConfig()
    policy:    PolicyConfig    = PolicyConfig()
    simulator: SimulatorConfig = SimulatorConfig()
    loss:      LossConfig      = LossConfig()
    training:  TrainingConfig  = TrainingConfig()
    eval:      EvalConfig      = EvalConfig()
    paths:     PathsConfig     = PathsConfig()

    @model_validator(mode="after")
    def _resolve_paths(self) -> "MasterConfig":
        """Substitute {experiment_name} token in output_dir."""
        self.paths.output_dir = self.paths.output_dir.format(
            experiment_name=self.experiment_name
        )
        return self

    def backbone_output_dim(self) -> int:
        """
        Compute the flat feature vector size produced by the active backbone.
        Used by PolicyNetwork to size the policy head input layer.

        iTransformer: n_channels * d_model
        LSTM:         hidden_size * (2 if bidirectional else 1)
        """

        n = self.data.resolve_n_features()
        if self.backbone.type == "itransformer":
            return n * self.backbone.itransformer.d_model
        elif self.backbone.type == "lstm":
            dirs = 2 if self.backbone.lstm.bidirectional else 1
            return self.backbone.lstm.hidden_size * dirs
        else:
            raise ValueError(f"Unknown backbone type: {self.backbone.type!r}")

