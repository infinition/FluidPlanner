"""
FluidPlanner: Hierarchical Planning via Multi-Scale Reaction-Diffusion PDEs.

Architecture: 3 PDE levels with decreasing diffusion rates.
- Level 0 (fast diffusion):  local actions, motor control
- Level 1 (medium diffusion): sub-goals, tactical decisions
- Level 2 (slow diffusion):  global plan, strategic objective

Top-down coupling: level 2 -> level 1 -> level 0 (learned forcing terms).
The hierarchy emerges from the diffusion time constants, not from manual design.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        return x / (rms + self.eps) * self.scale


class ReactionMLP(nn.Module):
    """Per-cell nonlinear reaction: R(u) = W2 * GELU(W1 * u)."""
    def __init__(self, d: int, expand: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * expand),
            nn.GELU(),
            nn.Linear(d * expand, d),
        )

    def forward(self, x):
        return self.net(x)


class Laplacian2DSimple(nn.Module):
    """Discrete 2D Laplacian with learnable per-channel diffusion coefficients."""
    def __init__(self, channels: int, n_dilations: int = 3):
        super().__init__()
        self.channels = channels
        dilations = [1, 2, 4][:n_dilations]
        self.dilations = dilations
        # Learnable diffusion coefficient per (dilation, channel)
        self.diff_logits = nn.Parameter(torch.zeros(n_dilations, channels))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, C, H, W) -> diffusion output (B, C, H, W)."""
        out = torch.zeros_like(u)
        for i, d in enumerate(self.dilations):
            coeff = 0.25 * torch.sigmoid(self.diff_logits[i])  # (C,) in [0, 0.25]
            # 2D Laplacian: sum of 4 neighbors - 4*center
            pad_u = F.pad(u, [d, d, d, d], mode="reflect")
            lap = (
                pad_u[:, :, d:-d, 2*d:]     +  # right
                pad_u[:, :, d:-d, :-2*d]     +  # left
                pad_u[:, :, 2*d:, d:-d]      +  # down
                pad_u[:, :, :-2*d, d:-d]     -  # up
                4 * u
            )
            out = out + coeff[None, :, None, None] * lap
        return out


class MemoryPump(nn.Module):
    """O(1) global memory: gated accumulation of spatially pooled features."""
    def __init__(self, d: int):
        super().__init__()
        self.forget_gate = nn.Linear(d, d)
        self.input_gate = nn.Linear(d, d)
        self.candidate = nn.Linear(d, d)

    def forward(self, h: torch.Tensor, x_pooled: torch.Tensor) -> torch.Tensor:
        f = torch.sigmoid(self.forget_gate(x_pooled))
        i = torch.sigmoid(self.input_gate(x_pooled))
        c = torch.tanh(self.candidate(x_pooled))
        return h * f + i * c


class PDELevel(nn.Module):
    """
    One level of the hierarchical PDE.

    Integrates: du/dt = D*Lap(u) + R(u) + alpha_mem*h + forcing
    for T steps with learnable dt.
    """
    def __init__(self, d_model: int, D_init: float = 0.1, max_steps: int = 6):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps

        self.diffusion = Laplacian2DSimple(d_model, n_dilations=3)
        self.reaction = ReactionMLP(d_model, expand=2)
        self.memory = MemoryPump(d_model)
        self.norm = RMSNorm(d_model)

        # Learnable time step and memory mixing
        self.log_dt = nn.Parameter(torch.tensor(math.log(0.1)))
        self.alpha_mem = nn.Parameter(torch.tensor(0.1))
        self.alpha_force = nn.Parameter(torch.tensor(0.1))

        # Initialize diffusion coefficients based on D_init
        with torch.no_grad():
            # inverse sigmoid of D_init/0.25
            init_logit = math.log(D_init / (0.25 - D_init + 1e-8))
            self.diffusion.diff_logits.fill_(init_logit)

    def forward(self, u: torch.Tensor, forcing: torch.Tensor = None):
        """
        u: (B, C, H, W) feature field
        forcing: (B, C, H, W) top-down signal from higher level (optional)
        Returns: evolved u, info dict with diagnostics
        """
        B, C, H, W = u.shape
        dt = torch.exp(self.log_dt).clamp(0.01, 0.3)
        h = torch.zeros(B, C, device=u.device, dtype=u.dtype)

        step_energies = []

        for t in range(self.max_steps):
            # Diffusion
            diff = self.diffusion(u)

            # Flatten for reaction + memory
            u_flat = u.permute(0, 2, 3, 1).reshape(B * H * W, C)
            react = self.reaction(u_flat).reshape(B, H, W, C).permute(0, 3, 1, 2)

            # Global memory
            u_pooled = u.mean(dim=(2, 3))  # (B, C)
            h = self.memory(h, u_pooled)
            h_broadcast = h[:, :, None, None].expand_as(u)

            # PDE update
            du = diff + react + self.alpha_mem * h_broadcast
            if forcing is not None:
                du = du + self.alpha_force * forcing

            u = u + dt * du

            # Normalize every 2 steps
            if (t + 1) % 2 == 0:
                u_flat = u.permute(0, 2, 3, 1).reshape(B * H * W, C)
                u_flat = self.norm(u_flat)
                u = u_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

            step_energies.append(u.abs().mean().item())

        info = {
            "dt": dt.item(),
            "h_norm": h.norm().item(),
            "step_energies": step_energies,
        }
        return u, info


class FluidPlanner(nn.Module):
    """
    3-Level Hierarchical PDE Planner.

    Level 2 (slow): global plan (quasi-static objective field)
    Level 1 (medium): sub-goals (tactical waypoints)
    Level 0 (fast): local actions (motor commands)

    Top-down forcing: level2 -> level1 -> level0.
    """
    def __init__(
        self,
        grid_channels: int = 6,
        d_model: int = 64,
        max_steps_per_level: int = 6,
        n_actions: int = 4,
    ):
        super().__init__()
        self.d_model = d_model

        # Encoder: grid (6 channels) -> d_model features
        self.encoder = nn.Sequential(
            nn.Conv2d(grid_channels, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
        )

        # Goal encoder: goal position -> d_model features (broadcast)
        self.goal_encoder = nn.Sequential(
            nn.Conv2d(1, d_model // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model // 4, d_model, 3, padding=1),
        )

        # 3 PDE levels with decreasing diffusion rates
        self.level2 = PDELevel(d_model, D_init=0.01, max_steps=max_steps_per_level)  # slow
        self.level1 = PDELevel(d_model, D_init=0.05, max_steps=max_steps_per_level)  # medium
        self.level0 = PDELevel(d_model, D_init=0.15, max_steps=max_steps_per_level)  # fast

        # Top-down coupling projections
        self.coupling_2to1 = nn.Conv2d(d_model, d_model, 1)
        self.coupling_1to0 = nn.Conv2d(d_model, d_model, 1)

        # Action head: pool agent location features -> action logits
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_actions),
        )

        # Phase classifier (auxiliary): what hierarchical level is active?
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, 3),  # 3 phases: key, door, goal
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, grid: torch.Tensor, return_fields: bool = False):
        """
        grid: (B, 6, H, W) multi-channel grid encoding
        Returns: action_logits (B, 4), and optionally intermediate fields
        """
        B, _, H, W = grid.shape

        # Encode grid
        u = self.encoder(grid)  # (B, d_model, H, W)

        # Encode goal (channel 4 = GOAL)
        goal_map = grid[:, 4:5, :, :]  # (B, 1, H, W)
        goal_features = self.goal_encoder(goal_map)
        u = u + goal_features

        # Level 2: global plan (slow diffusion)
        u2, info2 = self.level2(u)

        # Top-down: level2 -> level1
        force_1 = self.coupling_2to1(u2)
        u1, info1 = self.level1(u, forcing=force_1)

        # Top-down: level1 -> level0
        force_0 = self.coupling_1to0(u1)
        u0, info0 = self.level0(u, forcing=force_0)

        # Extract features at agent position
        agent_mask = grid[:, 5, :, :]  # (B, H, W), channel 5 = AGENT
        mask_sum = agent_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)  # (B, 1, 1)
        # Pool features at agent location: (B, C, H, W) * (B, 1, H, W) -> sum -> (B, C)
        agent_features = (u0 * agent_mask.unsqueeze(1)).sum(dim=(2, 3)) / mask_sum.squeeze(-1)  # (B, d_model)

        # Action prediction
        action_logits = self.action_head(agent_features)  # (B, 4)

        # Phase prediction (auxiliary)
        phase_logits = self.phase_head(agent_features)  # (B, 3)

        output = {
            "action_logits": action_logits,
            "phase_logits": phase_logits,
            "info": {"level0": info0, "level1": info1, "level2": info2},
        }

        if return_fields:
            output["fields"] = {
                "u0": u0.detach(),
                "u1": u1.detach(),
                "u2": u2.detach(),
                "input": u.detach(),
            }

        return output


class FlatBaseline(nn.Module):
    """Single-level baseline (no hierarchy) for ablation."""
    def __init__(self, grid_channels=6, d_model=64, n_actions=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(grid_channels, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
        )
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_actions),
        )

    def forward(self, grid, return_fields=False):
        B = grid.shape[0]
        features = self.encoder(grid)
        agent_mask = grid[:, 5, :, :]
        mask_sum = agent_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        agent_feat = (features * agent_mask.unsqueeze(1)).sum(dim=(2, 3)) / mask_sum.squeeze(-1)
        logits = self.action_head(agent_feat)
        output = {"action_logits": logits, "phase_logits": torch.zeros(B, 3, device=grid.device)}
        if return_fields:
            output["fields"] = {"u0": features.detach()}
        return output
