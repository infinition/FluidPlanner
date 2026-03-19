"""
FluidPlanner Pure Emergence: ZERO hierarchical bias.

A single PDE with uniform channels. No pre-defined levels, no top-down
coupling, no phase head. The question: does temporal stratification
emerge spontaneously from reaction-diffusion dynamics?

If after training some channels naturally evolve slowly (plan) and others
rapidly (actions), that is genuine emergence -- not architectural bias.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        return x / (rms + self.eps) * self.scale


class UniformLaplacian2D(nn.Module):
    """Discrete 2D Laplacian with learnable per-channel diffusion.
    All channels start EQUAL -- no pre-imposed scale separation."""

    def __init__(self, channels, n_dilations=3):
        super().__init__()
        self.dilations = [1, 2, 4][:n_dilations]
        # ALL channels initialized identically (D=0.05)
        init_logit = math.log(0.05 / (0.25 - 0.05))  # sigmoid^-1(0.05/0.25)
        self.diff_logits = nn.Parameter(torch.full((n_dilations, channels), init_logit))

    def forward(self, u):
        out = torch.zeros_like(u)
        for i, d in enumerate(self.dilations):
            coeff = 0.25 * torch.sigmoid(self.diff_logits[i])
            pad_u = F.pad(u, [d, d, d, d], mode="reflect")
            lap = (
                pad_u[:, :, d:-d, 2*d:] +
                pad_u[:, :, d:-d, :-2*d] +
                pad_u[:, :, 2*d:, d:-d] +
                pad_u[:, :, :-2*d, d:-d] -
                4 * u
            )
            out = out + coeff[None, :, None, None] * lap
        return out


class UniformReaction(nn.Module):
    """Per-cell MLP. Same for all channels -- no hierarchy."""
    def __init__(self, d, expand=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * expand),
            nn.GELU(),
            nn.Linear(d * expand, d),
        )

    def forward(self, x):
        return self.net(x)


class UniformMemory(nn.Module):
    """Single O(1) memory -- no multi-level hierarchy."""
    def __init__(self, d):
        super().__init__()
        self.forget = nn.Linear(d, d)
        self.input_gate = nn.Linear(d, d)
        self.candidate = nn.Linear(d, d)

    def forward(self, h, x_pooled):
        return h * torch.sigmoid(self.forget(x_pooled)) + \
               torch.sigmoid(self.input_gate(x_pooled)) * torch.tanh(self.candidate(x_pooled))


class PureEmergencePDE(nn.Module):
    """
    Single-level PDE with uniform initialization.

    ZERO hierarchical bias:
    - All channels start with identical diffusion coefficients
    - No top-down coupling (there's only one level)
    - No phase head (we don't pre-suppose number of phases)
    - Single memory pump (not stratified)

    After training, we measure per-channel temporal dynamics to see
    if stratification emerged spontaneously.
    """

    def __init__(self, d_model=128, max_steps=12, n_dilations=3):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps

        self.diffusion = UniformLaplacian2D(d_model, n_dilations)
        self.reaction = UniformReaction(d_model)
        self.memory = UniformMemory(d_model)
        self.norm = RMSNorm(d_model)

        # Single learnable dt -- same for all channels
        self.log_dt = nn.Parameter(torch.tensor(math.log(0.1)))
        self.alpha_mem = nn.Parameter(torch.tensor(0.1))

    def forward(self, u):
        """
        u: (B, C, H, W)
        Returns: evolved u, per-channel diagnostics
        """
        B, C, H, W = u.shape
        dt = torch.exp(self.log_dt).clamp(0.01, 0.3)
        h = torch.zeros(B, C, device=u.device, dtype=u.dtype)

        # Record per-channel states at each step for emergence analysis
        channel_history = []

        for t in range(self.max_steps):
            # Snapshot per-channel energy before update
            channel_energy = u.abs().mean(dim=(0, 2, 3))  # (C,)
            channel_history.append(channel_energy.detach())

            # Diffusion
            diff = self.diffusion(u)

            # Reaction (per-voxel)
            u_flat = u.permute(0, 2, 3, 1).reshape(B * H * W, C)
            react = self.reaction(u_flat).reshape(B, H, W, C).permute(0, 3, 1, 2)

            # Memory
            u_pooled = u.mean(dim=(2, 3))
            h = self.memory(h, u_pooled)
            h_broad = h[:, :, None, None].expand_as(u)

            # PDE step -- identical for all channels
            du = diff + react + self.alpha_mem * h_broad
            u = u + dt * du

            # Normalize every 2 steps
            if (t + 1) % 2 == 0:
                u_flat = u.permute(0, 2, 3, 1).reshape(B * H * W, C)
                u_flat = self.norm(u_flat)
                u = u_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Compute per-channel dynamics (how much each channel changed over steps)
        channel_history = torch.stack(channel_history)  # (T, C)
        # Per-channel velocity: mean absolute change between consecutive steps
        channel_velocity = (channel_history[1:] - channel_history[:-1]).abs().mean(dim=0)  # (C,)

        info = {
            "channel_velocity": channel_velocity,  # THE emergence metric
            "channel_history": channel_history,
            "dt": dt.item(),
        }
        return u, info


class FluidPlannerPure(nn.Module):
    """
    Pure emergence planner. Single PDE, no hierarchy.
    After training, analyze channel_velocity to detect stratification.
    """

    def __init__(self, grid_channels=6, d_model=128, max_steps=12, n_actions=4):
        super().__init__()
        self.d_model = d_model

        self.encoder = nn.Sequential(
            nn.Conv2d(grid_channels, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
        )

        # Single PDE -- no levels, no hierarchy
        self.pde = PureEmergencePDE(d_model, max_steps)

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, grid, return_fields=False):
        B, _, H, W = grid.shape

        u = self.encoder(grid)
        u_evolved, pde_info = self.pde(u)

        # Extract agent features
        agent_mask = grid[:, 5, :, :] if grid.shape[1] > 5 else grid[:, 2, :, :]
        mask_sum = agent_mask.sum(dim=(1, 2)).clamp(min=1e-8).unsqueeze(1)  # (B, 1)
        agent_feat = (u_evolved * agent_mask.unsqueeze(1)).sum(dim=(2, 3)) / mask_sum  # (B, C)

        action_logits = self.action_head(agent_feat)

        output = {
            "action_logits": action_logits,
            "phase_logits": torch.zeros(B, 3, device=grid.device),  # dummy for compat
            "channel_velocity": pde_info["channel_velocity"],
            "channel_history": pde_info["channel_history"],
            "info": pde_info,
        }

        if return_fields:
            # Split channels into 4 groups for visualization
            c = self.d_model
            q = c // 4
            output["fields"] = {
                "channels_0_25": u_evolved[:, :q].detach(),
                "channels_25_50": u_evolved[:, q:2*q].detach(),
                "channels_50_75": u_evolved[:, 2*q:3*q].detach(),
                "channels_75_100": u_evolved[:, 3*q:].detach(),
            }

        return output
