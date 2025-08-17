from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config.model_config import HRMConfig
from src.utils.logging import lprint, LoggingLevels as ll

# Adaptive Computation Time (ACT)
class ACT(nn.Module):
    def __init__(self, hidden_size: int, context_size: int, eps: float = 0.01):
        super().__init__()
        self.eps = torch.tensor(eps, dtype=torch.float32)  # Converti eps in tensore
        self.proj = nn.Sequential(
            nn.Linear(hidden_size + context_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, low_state: torch.Tensor, context: torch.Tensor, max_steps: int) -> Tuple[int, torch.Tensor]:
        """
        Determina il numero di passi bassi e i pesi di arresto.

        Args:
            low_state (torch.Tensor): low state layer [B, hidden_size]
            context (torch.Tensor): Context [B, context_size]
            max_steps (int): max number of low level steps

        Returns:
            tuple: number of steps, stop weitghts ([S, B])
        """
        try:
            B, H = low_state.shape
            device = low_state.device

            halting_prob = torch.zeros(B, device=device, dtype=torch.float32)  # [B]
            remainders = []
            step_probs = []
            steps = torch.zeros(B, device=device, dtype=torch.int32)  # [B]

            for t in range(max_steps):
                p = self.proj(torch.cat([low_state, context], dim=-1)).squeeze(-1)  # [B]
                p = torch.clamp(p, 0.01, 0.99)  # [B]
                new_halt = halting_prob + p  # [B]
                still_running = (halting_prob < (1 - self.eps.to(device))).float()  # [B]
                step_prob = torch.where(
                    new_halt <= (1 - self.eps.to(device)),
                    p,
                    (1 - self.eps.to(device)) - halting_prob
                )  # [B]
                step_prob = step_prob * still_running  # [B]
                step_probs.append(step_prob)
                halting_prob = torch.clamp(new_halt, max=1 - self.eps.to(device))  # [B]
                remainders.append(1 - halting_prob)
                steps += (halting_prob < (1 - self.eps.to(device))).to(torch.int32)  # [B]

            if len(step_probs) == 0:
                step_probs = [torch.zeros(B, device=device, dtype=torch.float32)]
                steps = torch.ones(B, device=device, dtype=torch.int32)

            step_weights = torch.stack(step_probs, dim=0)  # [S, B]
            total_steps = steps.max().item()  # Numero massimo di passi nel batch

            return total_steps, step_weights
        except Exception as e:
            lprint(ll.ERROR, f"Error in ACT forward: {str(e)}")
            raise

# Low-level reasoner
class LowLevelCore(nn.Module):
    def __init__(self, input_size: int, context_size: int, hidden_size: int):
        super().__init__()
        self.cell = nn.GRUCell(context_size + context_size, hidden_size)  # Modificato: usa context_size invece di input_size

    def forward_step(self, x_t: torch.Tensor, context: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.cell(torch.cat([x_t, context], dim=-1), h)

# High-level controller
class HighLevelCore(nn.Module):
    def __init__(self, in_size: int, hidden_size: int):
        super().__init__()
        self.cell = nn.GRUCell(in_size, hidden_size)

    def forward_step(self, features: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.cell(features, h)


class FusionModel(nn.Module):
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_in = nn.Linear(cfg.input_size, cfg.context_size)
        self.low = LowLevelCore(cfg.input_size, cfg.context_size, cfg.low_hidden)
        self.high = HighLevelCore(cfg.low_hidden + cfg.context_size, cfg.high_hidden)
        self.to_context = nn.Linear(cfg.high_hidden, cfg.context_size)
        self.act = ACT(cfg.low_hidden, cfg.context_size, eps=cfg.act_eps)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.high_hidden) if cfg.batch_norm else nn.Identity(),
            nn.Dropout(cfg.dropout_prob),
            nn.Linear(cfg.high_hidden, cfg.num_targets),  # Output for all targets
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        x: [B, T] — teacher outputs, where T is num_targets, each element is a scalar
        Returns predictions [B, T] and info dict with diagnostics.
        """
        try:
            B, T = x.shape
            cfg = self.cfg

            # Aggiungi dimensione per input_size=1
            x = x.unsqueeze(-1)  # [B, T, 1]

            # Initial states
            high_h = torch.zeros(B, cfg.high_hidden, device=x.device)
            low_h = torch.zeros(B, cfg.low_hidden, device=x.device)

            # Precompute token features
            token_feats = self.embed_in(x)  # [B, T, context_size]

            # High-level loop
            total_inner_steps = 0
            for n in range(cfg.max_high_steps):
                context = self.to_context(high_h)  # [B, context_size]
                steps, step_weights = self.act(low_h, context, cfg.max_low_steps)
                total_inner_steps += steps

                # Low-level loop: iterate over teacher outputs (T dimension)
                for s in range(steps):
                    for t in range(T):  # Iterate over each target's output
                        x_t = token_feats[:, t, :]  # [B, context_size]
                        low_h = self.low.forward_step(x_t, context, low_h)

                if cfg.one_step_detach:
                    low_h_ = low_h.detach() + (low_h - low_h.detach())
                else:
                    low_h_ = low_h

                features = torch.cat([low_h_, context], dim=-1)
                high_h = self.high.forward_step(features, high_h)

            outputs = self.classifier(high_h)  # [B, num_targets]
            info = {"total_inner_steps": total_inner_steps}
            return outputs, info
        except Exception as e:
            lprint(ll.ERROR, f"Error in FusionModel forward: {str(e)}")
            raise