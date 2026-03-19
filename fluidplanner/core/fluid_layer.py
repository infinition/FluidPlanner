"""
fluid_layer.py — Core reaction-diffusion layers for FluidVLA

Design goals preserved:
  - local, iterative, PDE-like computation
  - explicit diffusion + reaction + lightweight memory
  - adaptive compute via equilibrium / turbulence
  - no attention

Main improvements in this version:
  1. Early-stop metric is separated from differentiable regularization.
  2. Stop metric is measured on a low-resolution probe to avoid being dominated
     by tiny local pixel jitters.
  3. min_steps + stop_patience give more stable adaptive compute.
  4. Extra diagnostics are returned to understand why/when stopping happens.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .diffusion import Laplacian2D, LaplacianSpatioTemporal
except ImportError:
    from diffusion import Laplacian2D, LaplacianSpatioTemporal


class RMSNorm(nn.Module):
    """Simple RMSNorm over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class MemoryPump(nn.Module):
    """
    Lightweight global memory accumulator.

    h_t = h_{t-1} + gate(u) * value(u)

    The memory is reset at every forward call.
    This preserves O(1) memory with respect to spatial resolution.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

    def forward(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return h + torch.sigmoid(self.gate(u)) * torch.tanh(self.value(u))


class ReactionMLP(nn.Module):
    """
    Per-position reaction term R(u).

    Diffusion propagates information locally.
    Reaction transforms features at each position.
    """

    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = max(channels * expansion, channels)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class _FluidLayerBase(nn.Module):
    """Shared utilities for 2D and video fluid layers."""

    def __init__(
        self,
        channels: int,
        max_steps: int,
        dt: float,
        epsilon: float,
        alpha: float,
        use_pde: bool,
        norm_type: str,
        norm_every: int,
        stop_patience: int,
        min_steps: int,
        stop_probe_hw: int,
    ):
        super().__init__()
        self.channels = channels
        self.max_steps = int(max_steps)
        self.epsilon = float(epsilon)
        self.use_pde = bool(use_pde)
        self.norm_every = max(1, int(norm_every))
        self.stop_patience = max(1, int(stop_patience))
        self.min_steps = max(1, int(min_steps))
        self.stop_probe_hw = max(1, int(stop_probe_hw))

        self.reaction = ReactionMLP(channels)
        self.memory = MemoryPump(channels)

        # Learnable global coefficients
        self.alpha_global = nn.Parameter(torch.tensor(float(alpha)))
        self.log_dt = nn.Parameter(torch.log(torch.tensor(float(dt))))

        norm_type = norm_type.lower()
        if norm_type == "rmsnorm":
            self.norm = RMSNorm(channels)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(channels)
        else:
            raise ValueError("norm_type must be 'rmsnorm' or 'layernorm'")

    def _dt(self) -> torch.Tensor:
        # Wider than before for tunability, but still clamped for stability.
        return self.log_dt.exp().clamp(0.005, 0.35)

    def _alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_global)

    @staticmethod
    def _safe_zero_like_scalar(ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)

    def _should_stop(self, history: List[float], step_idx: int) -> bool:
        """
        Stop only:
          - in eval mode
          - after a minimum number of internal steps
          - if the last `stop_patience` stop-metrics are all below epsilon
        """
        if self.training or self.epsilon <= 0.0:
            return False
        if (step_idx + 1) < self.min_steps:
            return False
        if len(history) < self.stop_patience:
            return False
        window = history[-self.stop_patience:]
        return max(window) < self.epsilon


class FluidLayer2D(_FluidLayerBase):
    """
    2D reaction-diffusion layer.

    Update at each internal step:
      u <- u + dt * [diffusion(u) + reaction(u) + alpha*h_global + alpha_local*h_local]

    Key distinction:
      - stop_turbulence: for adaptive stopping only
      - diff_turbulence: for logging / optional regularization
    """

    def __init__(
        self,
        channels: int,
        dilations: List[int] = [1, 4, 16],
        max_steps: int = 12,
        dt: float = 0.1,
        epsilon: float = 0.08,
        alpha: float = 0.1,
        use_pde: bool = True,
        norm_type: str = "rmsnorm",
        norm_every: int = 2,
        local_memory_hw: int = 4,
        signed_diffusion: bool = False,
        diffusion_scale: float = 0.25,
        stop_patience: int = 2,
        min_steps: int = 3,
        stop_probe_hw: int = 8,
    ):
        super().__init__(
            channels=channels,
            max_steps=max_steps,
            dt=dt,
            epsilon=epsilon,
            alpha=alpha,
            use_pde=use_pde,
            norm_type=norm_type,
            norm_every=norm_every,
            stop_patience=stop_patience,
            min_steps=min_steps,
            stop_probe_hw=stop_probe_hw,
        )

        self.local_memory_hw = max(1, int(local_memory_hw))

        self.diffusion = Laplacian2D(
            channels=channels,
            dilations=dilations,
            signed_diffusion=signed_diffusion,
            diffusion_scale=diffusion_scale,
        )

        self.local_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.alpha_local = nn.Parameter(torch.tensor(alpha * 0.5))

    def _make_stop_probe(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build a compact low-resolution representation for stopping decisions.
        This makes stopping less sensitive to tiny local fluctuations.
        """
        probe = F.adaptive_avg_pool2d(
            x, output_size=(min(self.stop_probe_hw, x.shape[-2]), min(self.stop_probe_hw, x.shape[-1]))
        )
        return probe

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        b, c, h, w = u.shape
        device = u.device

        h_state = torch.zeros(b, c, device=device, dtype=u.dtype)

        stop_history: List[float] = []
        diff_turbulences: List[torch.Tensor] = []
        step_energies: List[torch.Tensor] = []  # v8: energie 100% differentiable

        equilibrium_step = self.max_steps

        # Probe used only for adaptive stop comparisons.
        prev_probe = self._make_stop_probe(u).detach()

        for step in range(self.max_steps):
            diff = self.diffusion(u) if self.use_pde else torch.zeros_like(u)

            u_flat = u.permute(0, 2, 3, 1).reshape(b, h * w, c)
            diff_flat = diff.permute(0, 2, 3, 1).reshape(b, h * w, c)

            react = self.reaction(u_flat)

            # Global O(1) memory
            pooled = u_flat.mean(dim=1)
            h_state = self.memory(h_state, pooled)
            h_global = h_state.unsqueeze(1).expand(-1, h * w, -1)

            # Cheap local low-resolution memory
            local_mem = F.adaptive_avg_pool2d(
                u, output_size=(self.local_memory_hw, self.local_memory_hw)
            )
            local_mem = self.local_proj(local_mem)
            local_mem = F.interpolate(local_mem, size=(h, w), mode="bilinear", align_corners=False)
            local_mem = local_mem.permute(0, 2, 3, 1).reshape(b, h * w, c)

            dt = self._dt()
            alpha_global = self._alpha()
            alpha_local = F.softplus(self.alpha_local)

            du = diff_flat + react + alpha_global * h_global + alpha_local * local_mem

            # Candidate state before normalization.
            u_candidate = u_flat + dt * du

            # Normalization only every N steps to avoid erasing the dynamics too aggressively.
            u_flat_next = u_candidate
            if (step + 1) % self.norm_every == 0:
                u_flat_next = self.norm(u_flat_next)

            u_candidate_img = u_candidate.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            u_next = u_flat_next.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

            # Stop metric on low-resolution probe, not full-res tensor.
            current_probe = self._make_stop_probe(u_candidate_img).detach()
            stop_turb = (current_probe - prev_probe).abs().mean() / (prev_probe.abs().mean() + 1e-8)

            # Differentiable turbulence for logging / eq-loss style usage.
            step_energy = du.abs().mean()
            lap_energy = diff_flat.abs().mean() if self.use_pde else self._safe_zero_like_scalar(u)
            diff_turb = stop_turb + 0.05 * step_energy + 0.01 * lap_energy
            diff_turbulences.append(diff_turb)
            step_energies.append(step_energy)  # v8: energie brute du update

            stop_val = float(stop_turb.item())
            stop_history.append(stop_val)

            if stop_val < self.epsilon and equilibrium_step == self.max_steps:
                equilibrium_step = step + 1

            u = u_next
            prev_probe = current_probe

            if self._should_stop(stop_history, step):
                break

        diff_turb_mean = (
            torch.stack(diff_turbulences).mean()
            if diff_turbulences
            else self._safe_zero_like_scalar(u)
        )

        # v8: energie PDE 100% differentiable
        step_energy_mean = (
            torch.stack(step_energies).mean()
            if step_energies
            else self._safe_zero_like_scalar(u)
        )

        return u, {
            "steps_used": len(stop_history),
            "stop_history": stop_history,
            "turbulence_history": stop_history,  # alias for backward compatibility
            "equilibrium_step": equilibrium_step,
            "final_turbulence": stop_history[-1] if stop_history else 0.0,
            "min_turbulence": min(stop_history) if stop_history else 0.0,
            "diff_turbulence": diff_turb_mean,
            "step_energy": step_energy_mean,  # v8: 100% differentiable
            "pde_active": self.use_pde,
        }


class FluidLayerVideo(_FluidLayerBase):
    """
    Video reaction-diffusion layer.

    Same philosophy as FluidLayer2D, but diffusion also propagates through time.
    Input/output layout: (B, C, T, H, W)

    We again use a compact low-resolution stop probe to avoid letting tiny
    frame-level noise prevent early stopping forever.
    """

    def __init__(
        self,
        channels: int,
        spatial_dilations: List[int] = [1, 4, 16],
        temporal_dilations: List[int] = [1, 2],
        max_steps: int = 12,
        dt: float = 0.1,
        epsilon: float = 0.08,
        causal_time: bool = True,
        use_pde: bool = True,
        norm_type: str = "rmsnorm",
        norm_every: int = 2,
        local_memory_hw: int = 4,
        signed_diffusion: bool = False,
        diffusion_scale: float = 0.25,
        temporal_mode: str = "backward_diff",
        stop_patience: int = 2,
        min_steps: int = 3,
        stop_probe_hw: int = 8,
        stop_probe_t: int = 2,
    ):
        super().__init__(
            channels=channels,
            max_steps=max_steps,
            dt=dt,
            epsilon=epsilon,
            alpha=0.1,
            use_pde=use_pde,
            norm_type=norm_type,
            norm_every=norm_every,
            stop_patience=stop_patience,
            min_steps=min_steps,
            stop_probe_hw=stop_probe_hw,
        )

        self.local_memory_hw = max(1, int(local_memory_hw))
        self.stop_probe_t = max(1, int(stop_probe_t))

        self.diffusion = LaplacianSpatioTemporal(
            channels=channels,
            spatial_dilations=spatial_dilations,
            temporal_dilations=temporal_dilations,
            causal_time=causal_time,
            temporal_mode=temporal_mode,
            signed_diffusion=signed_diffusion,
            diffusion_scale=diffusion_scale,
        )

        self.local_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.alpha_local = nn.Parameter(torch.tensor(0.05))

    def _make_stop_probe(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build a small low-resolution video probe:
          (B, C, T, H, W) -> pooled (B, C, t_probe, h_probe, w_probe)
        """
        t_probe = min(self.stop_probe_t, x.shape[2])
        h_probe = min(self.stop_probe_hw, x.shape[3])
        w_probe = min(self.stop_probe_hw, x.shape[4])
        probe = F.adaptive_avg_pool3d(x, output_size=(t_probe, h_probe, w_probe))
        return probe

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        b, c, t, h, w = u.shape
        n = t * h * w
        device = u.device

        h_state = torch.zeros(b, c, device=device, dtype=u.dtype)

        stop_history: List[float] = []
        diff_turbulences: List[torch.Tensor] = []
        step_energies: List[torch.Tensor] = []  # v8: energie 100% differentiable

        equilibrium_step = self.max_steps
        prev_probe = self._make_stop_probe(u).detach()

        for step in range(self.max_steps):
            diff = self.diffusion(u) if self.use_pde else torch.zeros_like(u)

            u_flat = u.permute(0, 2, 3, 4, 1).reshape(b, n, c)
            diff_flat = diff.permute(0, 2, 3, 4, 1).reshape(b, n, c)

            react = self.reaction(u_flat)

            # Global O(1) memory
            pooled = u_flat.mean(dim=1)
            h_state = self.memory(h_state, pooled)
            h_global = h_state.unsqueeze(1).expand(-1, n, -1)

            # Cheap low-resolution local video memory
            t_local = max(1, min(t, 2))
            local_mem = F.adaptive_avg_pool3d(
                u,
                output_size=(t_local, self.local_memory_hw, self.local_memory_hw),
            )
            local_mem = self.local_proj(local_mem)
            local_mem = F.interpolate(local_mem, size=(t, h, w), mode="trilinear", align_corners=False)
            local_mem = local_mem.permute(0, 2, 3, 4, 1).reshape(b, n, c)

            dt = self._dt()
            alpha_global = self._alpha()
            alpha_local = F.softplus(self.alpha_local)

            du = diff_flat + react + alpha_global * h_global + alpha_local * local_mem
            u_candidate = u_flat + dt * du

            u_flat_next = u_candidate
            if (step + 1) % self.norm_every == 0:
                u_flat_next = self.norm(u_flat_next)

            u_candidate_vid = u_candidate.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
            u_next = u_flat_next.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()

            # Stop metric on compact probe
            current_probe = self._make_stop_probe(u_candidate_vid).detach()
            stop_turb = (current_probe - prev_probe).abs().mean() / (prev_probe.abs().mean() + 1e-8)

            step_energy = du.abs().mean()
            lap_energy = diff_flat.abs().mean() if self.use_pde else self._safe_zero_like_scalar(u)
            diff_turb = stop_turb + 0.05 * step_energy + 0.01 * lap_energy
            diff_turbulences.append(diff_turb)
            step_energies.append(step_energy)  # v8: energie brute du update

            stop_val = float(stop_turb.item())
            stop_history.append(stop_val)

            if stop_val < self.epsilon and equilibrium_step == self.max_steps:
                equilibrium_step = step + 1

            u = u_next
            prev_probe = current_probe

            if self._should_stop(stop_history, step):
                break

        diff_turb_mean = (
            torch.stack(diff_turbulences).mean()
            if diff_turbulences
            else self._safe_zero_like_scalar(u)
        )

        # v8: energie PDE 100% differentiable
        step_energy_mean = (
            torch.stack(step_energies).mean()
            if step_energies
            else self._safe_zero_like_scalar(u)
        )

        return u, {
            "steps_used": len(stop_history),
            "stop_history": stop_history,
            "turbulence_history": stop_history,  # alias for backward compatibility
            "equilibrium_step": equilibrium_step,
            "final_turbulence": stop_history[-1] if stop_history else 0.0,
            "min_turbulence": min(stop_history) if stop_history else 0.0,
            "diff_turbulence": diff_turb_mean,
            "step_energy": step_energy_mean,  # v8: 100% differentiable
            "pde_active": self.use_pde,
        }