"""
diffusion.py — Multi-scale Laplacian operators for FluidVLA

Core idea:
  - diffusion is local and physics-inspired
  - kernels are fixed stencils (not attention)
  - only the diffusion coefficients are learned

This preserves the project's paradigm:
  local reaction-diffusion dynamics instead of global pairwise attention.
"""

from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


TemporalMode = Literal["backward_diff", "symmetric_laplacian"]


class Laplacian1D(nn.Module):
    """
    1D multi-scale discrete Laplacian.

    Useful for sequences or sensor streams.

    Kernel at dilation d:
      [1, -2, 1]

    Causal mode uses left-only padding for autoregressive settings.
    """

    def __init__(
        self,
        channels: int,
        dilations: List[int] = [1, 4, 16],
        causal: bool = False,
        signed_diffusion: bool = False,
        diffusion_scale: float = 0.25,
    ):
        super().__init__()
        self.dilations = list(dilations)
        self.causal = causal
        self.signed_diffusion = signed_diffusion
        self.diffusion_scale = diffusion_scale

        # Learnable coefficient per channel and per scale.
        self.D = nn.Parameter(torch.ones(len(self.dilations), channels) * 0.1)

        # Fixed physics kernel.
        self.register_buffer("kernel", torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32))

    def _coeff(self, param: torch.Tensor) -> torch.Tensor:
        if self.signed_diffusion:
            return self.diffusion_scale * torch.tanh(param)
        return F.softplus(param)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, C, L)
        returns: (B, C, L)
        """
        b, c, length = u.shape
        out = torch.zeros_like(u)

        kernel = self.kernel.view(1, 1, 3).expand(c, 1, 3)
        for i, d in enumerate(self.dilations):
            if self.causal:
                padded = F.pad(u, (2 * d, 0), mode="replicate")
            else:
                padded = F.pad(u, (d, d), mode="replicate")

            lap = F.conv1d(padded, kernel, dilation=d, groups=c)
            lap = lap[..., :length]
            coeff = self._coeff(self.D[i]).view(1, c, 1)
            out = out + coeff * lap

        return out


class Laplacian2D(nn.Module):
    """
    2D multi-scale Laplacian.

    5-point stencil at dilation d:
      u(x+d,y) + u(x-d,y) + u(x,y+d) + u(x,y-d) - 4u(x,y)
    """

    def __init__(
        self,
        channels: int,
        dilations: List[int] = [1, 4, 16],
        signed_diffusion: bool = False,
        diffusion_scale: float = 0.25,
    ):
        super().__init__()
        self.dilations = list(dilations)
        self.signed_diffusion = signed_diffusion
        self.diffusion_scale = diffusion_scale
        self.D = nn.Parameter(torch.ones(len(self.dilations), channels) * 0.1)

        kernel_2d = torch.zeros(3, 3, dtype=torch.float32)
        kernel_2d[1, 0] = 1.0
        kernel_2d[1, 2] = 1.0
        kernel_2d[0, 1] = 1.0
        kernel_2d[2, 1] = 1.0
        kernel_2d[1, 1] = -4.0
        self.register_buffer("kernel_2d", kernel_2d)

    def _coeff(self, param: torch.Tensor) -> torch.Tensor:
        if self.signed_diffusion:
            return self.diffusion_scale * torch.tanh(param)
        return F.softplus(param)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, C, H, W)
        returns: (B, C, H, W)
        """
        b, c, h, w = u.shape
        out = torch.zeros_like(u)
        kernel = self.kernel_2d.view(1, 1, 3, 3).expand(c, 1, 3, 3)

        for i, d in enumerate(self.dilations):
            padded = F.pad(u, (d, d, d, d), mode="replicate")
            lap = F.conv2d(padded, kernel, dilation=d, groups=c)
            lap = lap[:, :, :h, :w]
            coeff = self._coeff(self.D[i]).view(1, c, 1, 1)
            out = out + coeff * lap

        return out


class LaplacianSpatioTemporal(nn.Module):
    """
    3D spatio-temporal Laplacian for video.

    Output = spatial diffusion + temporal diffusion.

    Notes on temporal modes:
      - backward_diff:
          strictly causal, uses only past frames.
          This is the correct mode for real-time robotics.
      - symmetric_laplacian:
          centered temporal Laplacian using past and future.
          Useful only for offline experiments.
    """

    def __init__(
        self,
        channels: int,
        spatial_dilations: List[int] = [1, 4, 16],
        temporal_dilations: List[int] = [1, 2],
        causal_time: bool = True,
        temporal_mode: TemporalMode = "backward_diff",
        signed_diffusion: bool = False,
        diffusion_scale: float = 0.25,
    ):
        super().__init__()
        self.spatial_dilations = list(spatial_dilations)
        self.temporal_dilations = list(temporal_dilations)
        self.causal_time = causal_time
        self.temporal_mode = temporal_mode
        self.signed_diffusion = signed_diffusion
        self.diffusion_scale = diffusion_scale

        if self.causal_time and self.temporal_mode == "symmetric_laplacian":
            raise ValueError(
                "temporal_mode='symmetric_laplacian' is not strictly causal. "
                "Use causal_time=False for that mode."
            )

        self.D_spatial = nn.Parameter(torch.ones(len(self.spatial_dilations), channels) * 0.1)
        self.D_temporal = nn.Parameter(torch.ones(len(self.temporal_dilations), channels) * 0.05)

        kernel_2d = torch.zeros(3, 3, dtype=torch.float32)
        kernel_2d[1, 0] = 1.0
        kernel_2d[1, 2] = 1.0
        kernel_2d[0, 1] = 1.0
        kernel_2d[2, 1] = 1.0
        kernel_2d[1, 1] = -4.0
        self.register_buffer("kernel_2d", kernel_2d)

    def _coeff(self, param: torch.Tensor) -> torch.Tensor:
        if self.signed_diffusion:
            return self.diffusion_scale * torch.tanh(param)
        return F.softplus(param)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, C, T, H, W)
        returns: (B, C, T, H, W)
        """
        b, c, t, h, w = u.shape
        out = torch.zeros_like(u)

        # --- Spatial diffusion applied independently on each frame ---
        # Flatten time into batch because the spatial stencil is frame-local.
        u_spatial = u.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        spatial_out = torch.zeros_like(u_spatial)
        kernel = self.kernel_2d.view(1, 1, 3, 3).expand(c, 1, 3, 3)

        for i, d in enumerate(self.spatial_dilations):
            padded = F.pad(u_spatial, (d, d, d, d), mode="replicate")
            lap = F.conv2d(padded, kernel, dilation=d, groups=c)
            lap = lap[:, :, :h, :w]
            coeff = self._coeff(self.D_spatial[i]).view(1, c, 1, 1)
            spatial_out = spatial_out + coeff * lap

        spatial_out = spatial_out.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        out = out + spatial_out

        # --- Temporal diffusion ---
        for i, dt in enumerate(self.temporal_dilations):
            # Past context is always available.
            u_past = F.pad(u, (0, 0, 0, 0, dt, 0), mode="replicate")[:, :, :t, :, :]

            if self.causal_time:
                # Strictly causal backward difference.
                temporal_lap = u_past - u
            else:
                if self.temporal_mode == "backward_diff":
                    temporal_lap = u_past - u
                else:
                    u_future = F.pad(u, (0, 0, 0, 0, 0, dt), mode="replicate")[:, :, dt : dt + t, :, :]
                    temporal_lap = u_past + u_future - 2.0 * u

            coeff = self._coeff(self.D_temporal[i]).view(1, c, 1, 1, 1)
            out = out + coeff * temporal_lap

        return out
