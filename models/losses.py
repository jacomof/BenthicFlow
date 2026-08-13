"""Reconstruction and adversarial losses for RAE training.

Combines L1, LPIPS, SILog depth loss, and DINO-backed GAN discriminator loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Differentiable augmentation (Zhao et al. 2020)
# ---------------------------------------------------------------------------
# Three augmentations: color (brightness/saturation/contrast), translation, cutout.
# Applied to the discriminator's input only, identically to real and fake. The
# operations are differentiable so gradients flow into the generator.


def _rand_brightness(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)


def _rand_saturation(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2) + mean


def _rand_contrast(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + mean


def _rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    B, C, H, W = x.shape
    sh, sw = int(H * ratio + 0.5), int(W * ratio + 0.5)
    th = torch.randint(-sh, sh + 1, (B, 1, 1), device=x.device)
    tw = torch.randint(-sw, sw + 1, (B, 1, 1), device=x.device)
    grid_b = torch.arange(B, device=x.device).view(B, 1, 1)
    grid_h = torch.arange(H, device=x.device).view(1, H, 1)
    grid_w = torch.arange(W, device=x.device).view(1, 1, W)
    grid_h = torch.clamp(grid_h + th + 1, 0, H + 1)
    grid_w = torch.clamp(grid_w + tw + 1, 0, W + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    out = (
        x_pad.permute(0, 2, 3, 1)
        .contiguous()[grid_b, grid_h, grid_w]
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return out


def _rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    B, C, H, W = x.shape
    ch, cw = int(H * ratio + 0.5), int(W * ratio + 0.5)
    cy = torch.randint(0, H, (B, 1, 1), device=x.device)
    cx = torch.randint(0, W, (B, 1, 1), device=x.device)
    grid_b = torch.arange(B, device=x.device).view(B, 1, 1)
    grid_h = torch.arange(H, device=x.device).view(1, H, 1)
    grid_w = torch.arange(W, device=x.device).view(1, 1, W)
    mask = torch.ones(B, H, W, device=x.device)
    h_in = (grid_h >= cy - ch // 2) & (grid_h < cy - ch // 2 + ch)
    w_in = (grid_w >= cx - cw // 2) & (grid_w < cx - cw // 2 + cw)
    cut = h_in & w_in
    mask[grid_b.expand_as(cut), grid_h.expand_as(cut), grid_w.expand_as(cut)] = (
        1.0 - cut.float()
    )
    return x * mask.unsqueeze(1)


def diff_augment(
    x: torch.Tensor, policy: str = "color,translation,cutout"
) -> torch.Tensor:
    """Apply DiffAugment policies in sequence. Inputs in [0, 1]."""
    if not policy:
        return x
    for p in policy.split(","):
        p = p.strip()
        if p == "color":
            x = _rand_brightness(x)
            x = _rand_saturation(x)
            x = _rand_contrast(x)
        elif p == "translation":
            x = _rand_translation(x)
        elif p == "cutout":
            x = _rand_cutout(x)
        else:
            raise ValueError(f"Unknown DiffAugment policy: {p}")
    return x.clamp(0, 1)


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------


class LPIPSLoss(nn.Module):
    """lpips.LPIPS wrapper. Inputs in [0, 1]; converted to [-1, 1] internally."""

    def __init__(self, net: str = "vgg"):
        super().__init__()
        import lpips

        self.lpips = lpips.LPIPS(net=net, verbose=False)
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        self.lpips.eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lpips(pred * 2 - 1, target * 2 - 1).mean()


# ---------------------------------------------------------------------------
# DINO-S/8-backed discriminator
# ---------------------------------------------------------------------------


class DinoDiscriminator(nn.Module):
    """Frozen DINO-S/8 backbone with a small trainable convolutional head.

    Following StyleGAN-T / RAE paper: extract spatial token features at the
    8x8-patch resolution from a frozen DINO-S/8, reshape to a feature map,
    and run a small ConvNet head to produce a logit map for hinge-GAN loss.

    Input: RGB in [0, 1], 224x224. Internally converted to ImageNet stats.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, head_channels: int = 256):
        super().__init__()
        # Load frozen DINO-S/8 from torch hub.
        backbone = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone

        # DINO-S/8 has hidden dim 384. With 224 input and patch 8 we get a
        # 28x28 token grid. We extract patch tokens (drop CLS), reshape to a
        # [B, 384, 28, 28] feature map, then run the head.
        self.feat_dim = 384
        self.feat_grid = 224 // 8  # 28

        head_dim = head_channels
        self.head = nn.Sequential(
            nn.Conv2d(self.feat_dim, head_dim, 3, padding=1),
            nn.GroupNorm(8, head_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(head_dim, head_dim, 3, padding=1, stride=2),
            nn.GroupNorm(8, head_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(head_dim, head_dim, 3, padding=1),
            nn.GroupNorm(8, head_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(head_dim, 1, 3, padding=1),
        )

        self.register_buffer(
            "mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
        )

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Run frozen DINO and return [B, 384, 28, 28] patch features."""
        # Normalize to ImageNet stats inside the module so callers can pass [0,1]
        x = (x - self.mean) / self.std

        # FIXED: Removed torch.no_grad() here to allow gradients to flow
        # back to the generator. The backbone parameters are already frozen,
        # so this is memory safe.
        tokens = self.backbone.get_intermediate_layers(x, n=1)[0]  # [B, 1+N, D]

        patches = tokens[:, 1:, :]  # drop CLS
        B, N, D = patches.shape
        g = self.feat_grid
        return patches.transpose(1, 2).view(B, D, g, g).contiguous()

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        feats = self._extract_patch_tokens(rgb)
        return self.head(feats)


# Public alias so the rest of the codebase doesn't need to know which D it is
Discriminator = DinoDiscriminator


# ---------------------------------------------------------------------------
# Adaptive GAN weight (VQGAN / RAE)
# ---------------------------------------------------------------------------


def adaptive_lambda(
    rec_grad: torch.Tensor,
    gan_grad: torch.Tensor,
    eps: float = 1e-4,
    max_value: float = 1e4,
) -> torch.Tensor:
    return torch.clamp(
        rec_grad.norm() / (gan_grad.norm() + eps), max=max_value
    ).detach()


# ---------------------------------------------------------------------------
# Scale-invariant depth loss (Eigen et al. 2014)
# ---------------------------------------------------------------------------


def silog_loss(
    pred: torch.Tensor, target: torch.Tensor, lam: float = 0.85, eps: float = 1e-3
) -> torch.Tensor:
    """Scale-invariant log loss, computed per-image then averaged.
    g = log(pred) - log(target); loss = mean_b( E[g^2] - lam * E[g]^2 ),
    where the expectations are over each image's pixels. Subtracting the
    per-image mean error makes the loss invariant to a per-image multiplicative
    depth scale, so deployments at different absolute depth scales contribute
    """
    pred = pred.clamp_min(eps)
    target = target.clamp_min(eps)
    g = (torch.log(pred) - torch.log(target)).flatten(1)  # (B, H*W)
    per_image = g.pow(2).mean(dim=1) - lam * g.mean(dim=1).pow(2)
    return per_image.clamp_min(0).mean()


def multiscale_gradient_loss(
    pred: torch.Tensor, target: torch.Tensor, num_scales: int = 4, eps: float = 1e-3
) -> torch.Tensor:
    """Multi-scale gradient matching (MiDaS-style), in log-depth space.
    Penalises the difference between the spatial gradients of pred and target
    at several resolutions. Because the GT depth here is smooth, this term is
    near-zero for clean output but strongly penalises high-frequency grain in
    the prediction (e.g. texture leaking from the RGB LPIPS/GAN losses through
    """
    diff = torch.log(pred.clamp_min(eps)) - torch.log(target.clamp_min(eps))
    total = diff.new_zeros(())
    for _ in range(num_scales):
        gx = (diff[:, :, :, 1:] - diff[:, :, :, :-1]).abs().mean()
        gy = (diff[:, :, 1:, :] - diff[:, :, :-1, :]).abs().mean()
        total = total + gx + gy
        if diff.shape[-1] < 2 or diff.shape[-2] < 2:
            break
        diff = F.avg_pool2d(diff, kernel_size=2)
    return total


# ---------------------------------------------------------------------------
# Combined loss with phase scheduling
# ---------------------------------------------------------------------------


class RAELoss(nn.Module):
    """RGBD reconstruction loss with phase-scheduled LPIPS and GAN.

    Generator-side: L1(rgb) + L1(depth) [+ LPIPS(rgb)] [+ GAN(rgb)].
    Discriminator-side: hinge GAN on RGB only, with DiffAugment on inputs.
    """

    def __init__(
        self,
        lpips_start_epoch: int = 6,
        gan_start_epoch: int = 8,
        w_l1_rgb: float = 1.0,
        w_depth: float = 1.0,
        w_lpips: float = 1.0,
        w_gan: float = 0.75,
        depth_loss: str = "silog",
        silog_lambda: float = 0.85,
        silog_eps: float = 1e-3,
        w_depth_grad: float = 0.5,
        depth_grad_scales: int = 4,
        diffaug_policy: str = "color,translation,cutout",
    ):
        super().__init__()
        self.lpips_start = lpips_start_epoch
        self.gan_start = gan_start_epoch
        self.w_l1_rgb = w_l1_rgb
        self.w_depth = w_depth
        self.depth_loss = depth_loss
        self.silog_lambda = silog_lambda
        self.silog_eps = silog_eps
        self.w_depth_grad = w_depth_grad
        self.depth_grad_scales = depth_grad_scales
        self.w_lpips = w_lpips
        self.w_gan = w_gan
        self.diffaug_policy = diffaug_policy

        self.lpips = LPIPSLoss(net="vgg")
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    @property
    def lpips_active(self) -> bool:
        return self.epoch >= self.lpips_start

    @property
    def gan_active(self) -> bool:
        return self.epoch >= self.gan_start

    # ----- generator-side loss -----

    def generator_loss(
        self,
        rgbd_pred: torch.Tensor,
        rgbd_target: torch.Tensor,
        last_layer: nn.Parameter | None = None,
        disc: nn.Module | None = None,
    ) -> dict:
        rgb_p, depth_p = rgbd_pred[:, :3], rgbd_pred[:, 3:4]
        rgb_t, depth_t = rgbd_target[:, :3], rgbd_target[:, 3:4]

        l_l1_rgb = F.l1_loss(rgb_p, rgb_t)
        if self.depth_loss == "silog":
            l_depth = silog_loss(
                depth_p, depth_t, lam=self.silog_lambda, eps=self.silog_eps
            )
            depth_key = "silog_depth"
        else:
            l_depth = F.l1_loss(depth_p, depth_t)
            depth_key = "l1_depth"

        loss = self.w_l1_rgb * l_l1_rgb + self.w_depth * l_depth
        out = {"l1_rgb": l_l1_rgb.detach(), depth_key: l_depth.detach()}

        if self.w_depth_grad > 0:
            l_depth_grad = multiscale_gradient_loss(
                depth_p, depth_t, num_scales=self.depth_grad_scales, eps=self.silog_eps
            )
            loss = loss + self.w_depth_grad * l_depth_grad
            out["depth_grad"] = l_depth_grad.detach()

        if self.lpips_active:
            l_lp = self.lpips(rgb_p, rgb_t)
            loss = loss + self.w_lpips * l_lp
            out["lpips"] = l_lp.detach()

        if self.gan_active and disc is not None:
            # DiffAugment applied in the same way to the discriminator's view
            # of the fake. Real images are augmented in the discriminator step.
            rgb_p_aug = diff_augment(rgb_p, self.diffaug_policy)
            logits_fake = disc(rgb_p_aug)
            l_g = -logits_fake.mean()

            if last_layer is not None and last_layer.requires_grad:
                rec_total = self.w_l1_rgb * l_l1_rgb + (
                    self.w_lpips * l_lp if self.lpips_active else 0.0
                )
                rec_grad = torch.autograd.grad(
                    rec_total, last_layer, retain_graph=True
                )[0]
                gan_grad = torch.autograd.grad(l_g, last_layer, retain_graph=True)[0]
                lam = adaptive_lambda(rec_grad, gan_grad)
            else:
                lam = torch.tensor(1.0, device=rgb_p.device)

            loss = loss + self.w_gan * lam * l_g
            out["gan_g"] = l_g.detach()
            out["gan_lambda"] = lam.detach()

        out["total"] = loss
        return out

    # ----- discriminator-side loss -----

    def discriminator_loss(
        self, disc: nn.Module, rgb_real: torch.Tensor, rgb_fake: torch.Tensor
    ) -> dict:
        if not self.gan_active:
            return {"d_total": torch.tensor(0.0, device=rgb_real.device)}

        rgb_real_aug = diff_augment(rgb_real, self.diffaug_policy)
        rgb_fake_aug = diff_augment(rgb_fake.detach(), self.diffaug_policy)

        # Single forward through the (DDP-wrapped) discriminator: real and fake
        # are concatenated so there is exactly one forward per backward, which is
        # the DDP-safe pattern. Two separate forwards on a DDP module risk the
        # reducer's "marked ready only once" error. The head uses GroupNorm
        # (per-sample), so concatenation is numerically identical to two forwards.
        logits = disc(torch.cat([rgb_real_aug, rgb_fake_aug], dim=0))
        logits_real, logits_fake = logits.chunk(2, dim=0)

        l_real = F.relu(1.0 - logits_real).mean()
        l_fake = F.relu(1.0 + logits_fake).mean()
        l = 0.5 * (l_real + l_fake)
        return {"d_total": l, "d_real": l_real.detach(), "d_fake": l_fake.detach()}
