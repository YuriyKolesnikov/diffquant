# training/trainer.py
"""
DiffTrainer: end-to-end differentiable training loop.

Episode rollout mechanics:
    At each step t ∈ [0, H-1], the model receives:
        window : full_seq[:, t : t+ctx, :]   sliding context window
        extras : [prev_pos, prev_delta, t/H, (H-t)/H]  path context

    Normalisation is per-sample per-step (z-score on the context window).
    Extras are detached — gradient flows through positions_list only,
    not through the extras chain. This avoids excessively long BPTT
    while preserving the full gradient through the position sequence.

Validation:
    Walk-forward on the continuous val series — one forward pass per bar,
    no episode resets, position tracked across the entire period.
    This mirrors live trading and is the only honest validation metric.
"""

import logging
from pathlib import Path
from typing  import Optional

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange

from configs.base_config   import MasterConfig
from data.dataset          import TradingDataset
from data.normalization    import normalize_context
from model.policy_network  import PolicyNetwork
from simulator.diff_simulator import DiffSimulator, SimConfig
from simulator.losses      import compute_loss
from evaluation.walk_forward import WalkForwardEvaluator
from utils.logging_utils import MetricsLogger
from utils.visualization  import Visualizer

log = logging.getLogger(__name__)


class DiffTrainer:

    def __init__(self, cfg: MasterConfig):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)

        self.out_dir = Path(cfg.paths.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Model
        self.model = PolicyNetwork(cfg).to(self.device)
        assert self.model.n_params > 0
        n_feat = cfg.data.resolve_n_features()
        log.info(
            "PolicyNetwork: %s params | backbone=%s | loss=%s | n_features=%d",
            f"{self.model.n_params:,}", cfg.backbone.type, cfg.loss.type, n_feat,
        )

        # Simulator
        self.simulator = DiffSimulator(SimConfig(
            commission_rate   = cfg.simulator.commission_rate,
            slippage_rate     = cfg.simulator.slippage_rate,
            smooth_abs_eps    = cfg.simulator.smooth_abs_eps,
            market_impact_eta = cfg.simulator.market_impact_eta,
        ))

        self._evaluator = WalkForwardEvaluator(self.model, cfg)

        # Optimiser: β₂=0.95 for faster adaptation to non-stationary gradients.
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr           = cfg.training.lr,
            weight_decay = cfg.training.weight_decay,
            betas        = (0.9, 0.95),
        )

        # LR schedule: linear warmup → cosine decay.
        warmup = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor = 0.05,
            total_iters  = cfg.training.warmup_epochs,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max   = max(1, cfg.training.num_epochs - cfg.training.warmup_epochs),
            eta_min = 1e-7,
        )
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers = [warmup, cosine],
            milestones = [cfg.training.warmup_epochs],
        )

        self.best_val_sharpe = -float("inf")
        self.logger     = MetricsLogger(str(self.out_dir))
        self.visualizer = Visualizer(str(self.out_dir / "plots"))
        self._history: list[dict] = []

    # ── Episode rollout ───────────────────────────────────────────────────────

    def _rollout(
        self,
        full_seq: torch.Tensor,    # (B, ctx+hor, F)
        closes:   torch.Tensor,    # (B, hor+1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Slide the policy over the horizon, collecting positions.
        Returns positions (B, H) — in computation graph — and closes.
        """
        cfg = self.cfg
        B   = full_seq.shape[0]
        ctx = cfg.data.context_len
        hor = cfg.data.horizon_len

        full_seq = full_seq.to(self.device)
        closes   = closes.to(self.device)

        positions_list: list[torch.Tensor] = []
        prev_pos   = torch.zeros(B, 1, device=self.device)
        prev_delta = torch.zeros(B, 1, device=self.device)

        for t in range(hor):
            # Sliding context window.
            window_raw = full_seq[:, t : t + ctx, :]        # (B, ctx, F)

            # Per-sample z-score — uses context window stats only.
            window, _, _ = normalize_context(window_raw)

            # Extras: path context, fully detached from graph.
            t_elapsed   = torch.full((B, 1), t / hor,       device=self.device)
            t_remaining = torch.full((B, 1), (hor - t) / hor, device=self.device)

            if self.cfg.training.full_bptt:
                extras = torch.cat([prev_pos, prev_delta, t_elapsed, t_remaining], dim=1)
            else:
                extras = torch.cat([prev_pos.detach(), prev_delta.detach(), t_elapsed, t_remaining], dim=1)

            pos_t = self.model(window, extras)               # (B, 1) — in graph
            positions_list.append(pos_t)

            prev_delta = (pos_t - prev_pos).detach()
            prev_pos   = pos_t.detach()

        positions = torch.cat(positions_list, dim=1)         # (B, H)
        return positions, closes

    # ── Training epoch ────────────────────────────────────────────────────────
    
    def _train_epoch(self, loader: DataLoader) -> dict:
        if len(loader) == 0:
            raise ValueError(
                f"DataLoader has 0 batches. "
                f"Train dataset may be smaller than batch_size={self.cfg.training.batch_size}. "
                f"Reduce batch_size or check split boundaries."
            )

        self.model.train()
        losses, pnls = [], []

        use_mirror = self.cfg.training.mirror_augmentation

        for full_seq, closes in loader:
            self.optimizer.zero_grad()

            positions, closes = self._rollout(full_seq, closes)
            step_pnl, _, _    = self.simulator.simulate(closes, positions)
            loss              = compute_loss(step_pnl, positions, self.cfg)

            # ── Mirror augmentation ───────────────────────────────────────────
            # With probability 0.5, reflect prices around the first bar close.
            # Reflected series has the same volatility structure but inverted
            # direction — forces symmetric long/short learning.
            # Loss is averaged 50/50 with the original episode.
            if use_mirror and torch.rand(1).item() > 0.5:
                closes_aug = 2.0 * closes[:, 0:1] - closes   # (B, hor+1)

                # full_seq_aug = full_seq.clone()
                # n_price_ch = sum(
                #     1 for c in self.cfg.data.active_feature_set
                #     if c in ("open", "high", "low", "close")
                # )
                # full_seq_aug[:, :, :n_price_ch] = -full_seq[:, :, :n_price_ch]

                price_cols = {"open", "high", "low", "close"}
                price_mask = torch.tensor(
                    [c in price_cols for c in self.cfg.data.active_feature_set],
                    dtype=torch.bool, device=full_seq.device,
                )  # (F,)

                full_seq_aug = full_seq.clone()
                full_seq_aug[:, :, price_mask] = -full_seq[:, :, price_mask]

                positions_aug, closes_aug = self._rollout(full_seq_aug, closes_aug)
                pnl_aug, _, _  = self.simulator.simulate(closes_aug, -positions_aug)
                loss_aug       = compute_loss(pnl_aug, -positions_aug, self.cfg)

                loss = 0.5 * loss + 0.5 * loss_aug

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.gradient_clip
            )
            self.optimizer.step()

            losses.append(loss.item())
            pnls.append(step_pnl.detach())

        flat = torch.cat([p.reshape(-1) for p in pnls])
        return {
            "train_loss":   sum(losses) / len(losses),
            "train_sharpe": (flat.mean() / (flat.std() + 1e-8)).item(),
        }

    # ── Walk-forward validation ───────────────────────────────────────────────

    def _validate(self, raw_val: dict) -> dict:
        """Delegates to WalkForwardEvaluator — single source of truth."""
        result = self._evaluator.run(raw_val, mode="val")
        m      = result.metrics
        return {
            "val_sharpe":       m["sharpe"],
            "val_sortino":      m["sortino"],
            "val_final_ret":    m["final_return_pct"],
            "val_max_drawdown": m["max_drawdown_pct"],
            "val_turnover":     m["turnover"],
            "val_flat_frac":    m["flat_fraction"],
            "val_total_comm":   m["total_commission_pct"],
        }

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save(self, name: str) -> None:
        path = self.out_dir / "models" / name
        path.parent.mkdir(exist_ok=True)
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_sharpe":      self.best_val_sharpe,
            "cfg":                  self.cfg.model_dump(),
            "history":              self._history,
        }, path)

    # ── Main training loop ────────────────────────────────────────────────────

    def train(
        self,
        train_dataset: TradingDataset,
        raw_val:       dict,
    ) -> None:
        """
        Parameters
        ----------
        train_dataset : TradingDataset built from the train split.
        raw_val       : dict with keys raw_features, raw_closes, raw_timestamps
                        (continuous val series for walk-forward evaluation).
        """
        cfg    = self.cfg
        loader = DataLoader(
            train_dataset,
            batch_size  = cfg.training.batch_size,
            shuffle     = True,
            num_workers = cfg.training.num_workers,
            pin_memory  = (cfg.device != "cpu"),
            drop_last   = True,    # ensures consistent batch sizes for loss stats
        )

        log.info(
            "Training: %s | epochs=%d | batch=%d | train_samples=%s",
            cfg.experiment_name,
            cfg.training.num_epochs,
            cfg.training.batch_size,
            f"{len(train_dataset):,}",
        )

        progress = trange(
            1, cfg.training.num_epochs + 1,
            desc="Training",
            leave=True,
            dynamic_ncols=True,
        )

        for epoch in progress:
            train_m = self._train_epoch(loader)
            self.scheduler.step()

            # Update progress bar with current training metrics.
            progress.set_description(
                f"[{epoch}/{cfg.training.num_epochs}]  "
                f"loss={train_m['train_loss']:+.4f}  "
                f"sharpe={train_m['train_sharpe']:+.4f}"
            )

            if epoch % cfg.training.val_freq == 0:
                val_m  = self._validate(raw_val)
                self.logger.log_val(epoch, train_m, val_m)

                sharpe = val_m.get("val_sharpe", -999.0)
                marker = ""
                if sharpe > self.best_val_sharpe:
                    self.best_val_sharpe = sharpe
                    self._save("best.pth")
                    marker = "  ★ best"

                # Update progress bar with val metrics and log to file.
                progress.set_description(
                    f"[{epoch}/{cfg.training.num_epochs}]  "
                    f"loss={train_m['train_loss']:+.4f}  "
                    f"sharpe={train_m['train_sharpe']:+.4f}  |  "
                    f"val_sharpe={val_m.get('val_sharpe', 0.0):+.4f}  "
                    f"val_ret={val_m.get('val_final_ret', 0.0):+.2f}%"
                    f"{marker}"
                )

                log.info(
                    "[%4d/%d]  loss=%+.4f  tr_sharpe=%+.4f  |  "
                    "val_sharpe=%+.4f  val_ret=%+.2f%%  "
                    "dd=%.3f  flat=%.3f  turn=%.4f%s",
                    epoch, cfg.training.num_epochs,
                    train_m["train_loss"],
                    train_m["train_sharpe"],
                    val_m.get("val_sharpe",       0.0),
                    val_m.get("val_final_ret",    0.0),
                    val_m.get("val_max_drawdown", 0.0),
                    val_m.get("val_flat_frac",    0.0),
                    val_m.get("val_turnover",     0.0),
                    marker,
                )

        self._save("final.pth")
        self.visualizer.plot_training_history(self.logger.get_history())
        log.info(
            "Training complete. Best val Sharpe: %+.4f  → %s",
            self.best_val_sharpe,
            self.out_dir / "models" / "best.pth",
        )

