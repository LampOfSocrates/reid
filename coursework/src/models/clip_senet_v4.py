import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

try:
    from transformers import CLIPModel
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

__all__ = ["clip_senet_v4"]

_TINYCLIP_HF  = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
_SEM_DIM      = 512    # TinyCLIP-ViT-8M/16 visual_projection output dim
_CNN_DIM      = 768    # ConvNeXt-small pooled output
_FUSED_DIM    = 2048   # T_u and T after fusion (paper Eq. 3 / 5)
_AFEM_GROUPS  = 32     # paper ablation Table V optimal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_kaiming(m):
    cls = m.__class__.__name__
    if "Linear" in cls:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif "BatchNorm1d" in cls and m.affine:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def _init_classifier(m):
    if "Linear" in m.__class__.__name__:
        nn.init.normal_(m.weight, std=0.001)


# ---------------------------------------------------------------------------
# CNN backbone  (ConvNeXt-small — timm, ImageNet pretrained, 768-d output)
# ---------------------------------------------------------------------------

class _ConvNeXtSmall(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self._m = timm.create_model("convnext_small", pretrained=pretrained, num_classes=0)

    def forward(self, x):
        return self._m(x)  # [B, 768]


# ---------------------------------------------------------------------------
# Semantic encoder  (TinyCLIP-ViT-8M/16)
#
# freeze_backbone=True  (default): only visual_projection + BN are trained.
#   The ViT weights are CLIP-pretrained and already semantically rich —
#   freezing them cuts ~7M params from the backward pass every step.
#   Set freeze_backbone=False to fine-tune everything end-to-end.
#
# Resize: always interpolate to 224×224 unconditionally so torch.compile
#   sees a static graph (no dynamic shape branch → no graph break).
# ---------------------------------------------------------------------------

class _TinyCLIPSemNet(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        assert _HF_AVAILABLE, "Run: pip install transformers"
        clip = CLIPModel.from_pretrained(_TINYCLIP_HF) if pretrained \
               else CLIPModel(CLIPModel.config_class.from_pretrained(_TINYCLIP_HF))
        self.vision_model      = clip.vision_model
        self.visual_projection = clip.visual_projection  # hidden → 256-d
        self.bn                = nn.BatchNorm1d(_SEM_DIM)

        if freeze_backbone:
            for p in self.vision_model.parameters():
                p.requires_grad = False

    def forward(self, x):
        # Unconditional resize — no dynamic branch, keeps torch.compile graph intact
        x = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)
        pooled = self.vision_model(pixel_values=x).pooler_output
        return self.bn(self.visual_projection(pooled))  # [B, 256]


# ---------------------------------------------------------------------------
# AFEM  (paper Eq. 4 — grouped FC with learnable scalar weights, NOT squeeze-excite)
#
# T_s' = f_linear(T_s) + Σ_i  w_i ⊙ f_linear(T_s)_group_i
#       = f_linear(T_s) · (1 + w_expanded)
#
# f_linear : Linear(in→out) → BN → ReLU
# w_i      : one learnable scalar per group, init ~ N(0,1)  (paper §III-C)
# Groups   : out_dim split into G=32 contiguous chunks
#
# group_weights expanded via view+expand — avoids repeat_interleave allocation
# on every forward pass and is torch.compile friendly.
# ---------------------------------------------------------------------------

class AFEM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_groups: int = 32):
        super().__init__()
        assert out_dim % num_groups == 0, \
            f"out_dim ({out_dim}) must be divisible by num_groups ({num_groups})"
        self.f_linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.group_weights = nn.Parameter(torch.randn(num_groups))
        self._num_groups = num_groups
        self._group_size = out_dim // num_groups

    def forward(self, x):
        T_f = self.f_linear(x)                                         # [B, out_dim]
        w   = self.group_weights.view(self._num_groups, 1) \
                                .expand(self._num_groups, self._group_size) \
                                .reshape(-1)                            # [out_dim]
        return T_f + T_f * w                                           # [B, out_dim]


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss (Khosla et al. 2020)
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        logits   = torch.matmul(features, features.T) / self.temp
        labels   = labels.contiguous().view(-1, 1)
        mask     = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(labels.shape[0], device=device).view(-1, 1), 0,
        )
        mask        = mask * logits_mask
        exp_logits  = torch.exp(logits) * logits_mask
        log_prob    = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return -mean_log_prob_pos.mean()


# ---------------------------------------------------------------------------
# Main model
#
# Architecture (paper Fig. 2 / Eq. 1–5):
#   T_a  = ConvNeXt-small(x)                 [B,  768]
#   T_s  = BN(TinyCLIP-ViT-8M(x))            [B,  256]
#   T_u  = FC(cat(T_s, T_a))  → 2048-d       Eq. 3
#   T_s' = AFEM(T_s)          → 2048-d       Eq. 4
#   T    = T_u + T_s'                         Eq. 5
#   ŷ    = classifier(T)       (train only)
#
# Loss: L = L_CE + L_SupCon  (paper Eq. 8, equal weights)
# NOTE: pass --lambda-htri 0 at train time — paper does NOT use triplet loss.
#
# Multi-stream removed: torch.compile(reduce-overhead) uses CUDA graphs which
# require all ops on the default stream — mixing streams breaks graph capture.
# ---------------------------------------------------------------------------

class CLIPSENet_V4(nn.Module):
    def __init__(
        self,
        num_classes,
        loss="softmax",
        pretrained=True,
        freeze_sem_backbone: bool = True,
        supcon_weight: float = 1.0,
        supcon_temperature: float = 0.07,
        **kwargs,
    ):
        super().__init__()
        self.loss    = loss
        self.app_net = _ConvNeXtSmall(pretrained=pretrained)
        self.sem_net = _TinyCLIPSemNet(pretrained=pretrained,
                                       freeze_backbone=freeze_sem_backbone)

        self.fuse       = nn.Linear(_SEM_DIM + _CNN_DIM, _FUSED_DIM)   # Eq. 3
        self.afem       = AFEM(_SEM_DIM, _FUSED_DIM, _AFEM_GROUPS)     # Eq. 4
        self.classifier = nn.Linear(_FUSED_DIM, num_classes, bias=False)

        self.fuse.apply(_init_kaiming)
        self.classifier.apply(_init_classifier)

        self._supcon        = SupConLoss(temperature=supcon_temperature)
        self._supcon_weight = supcon_weight

    def forward(self, x):
        f_app     = self.app_net(x)                                # [B, 2048]
        T_s       = self.sem_net(x)                                # [B,  512]
        T_u       = self.fuse(torch.cat([T_s, f_app], dim=1))     # [B, 2048]  Eq. 3
        T_s_prime = self.afem(T_s)                                 # [B, 2048]  Eq. 4
        T         = T_u + T_s_prime                                # [B, 2048]  Eq. 5

        if self.training:
            return self.classifier(T), T
        return T

    def aux_loss(self, features, pids):
        return self._supcon_weight * self._supcon(features, pids)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def clip_senet_v4(num_classes, loss="softmax", pretrained=True, **kwargs):
    return CLIPSENet_V4(num_classes, loss=loss, pretrained=pretrained, **kwargs)
