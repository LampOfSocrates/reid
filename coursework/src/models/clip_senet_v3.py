import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class _ConvNeXtSmall_Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self._m = timm.create_model("convnext_small", pretrained=pretrained, num_classes=0)

    def forward(self, x):
        return self._m(x)  # [B, 768]

__all__ = ["clip_senet_v3_ibn_supcon"]


# --- Initialization Helpers ---
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)


# --- Adaptive Fine-grained Enhancement Module (squeeze-excite + residual) ---
class AFEM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 8, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attn(x) + x


# --- Supervised Contrastive Loss (Khosla et al. 2020) ---
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        logits = torch.div(torch.matmul(features, features.T), self.temp)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(labels.shape[0], device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return -mean_log_prob_pos.mean()


# --- Version 3: ResNet50-IBN-a + TinyViT + AFEM with SupCon aux loss ---
class CLIP_SENet_V3_IBN_SupCon(nn.Module):
    def __init__(
        self,
        num_classes,
        loss="softmax",
        pretrained=True,
        supcon_weight: float = 0.5,
        supcon_temperature: float = 0.07,
        **kwargs,
    ):
        super().__init__()
        self.loss = loss
        self.app_net = _ConvNeXtSmall_Backbone(pretrained=pretrained)
        self.sem_net = timm.create_model(
            "vit_tiny_patch16_224", pretrained=pretrained, num_classes=0
        )
        self.afem = AFEM(192)

        self.reduce = nn.Linear(768 + 192, 512)
        self.bn_neck = nn.BatchNorm1d(512)
        self.bn_neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(512, num_classes, bias=False)

        self.reduce.apply(weights_init_kaiming)
        self.bn_neck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        self._supcon = SupConLoss(temperature=supcon_temperature)
        self._supcon_weight = supcon_weight
        self._sem_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def forward(self, x):
        f_app = self.app_net(x)

        if self._sem_stream is not None:
            with torch.cuda.stream(self._sem_stream):
                f_sem = self.afem(self.sem_net(x))
            torch.cuda.current_stream().wait_stream(self._sem_stream)
        else:
            f_sem = self.afem(self.sem_net(x))

        combined = torch.cat([f_app, f_sem], dim=1)
        feat = self.reduce(combined)
        bn_feat = self.bn_neck(feat)
        if self.training:
            return self.classifier(bn_feat), feat
        return bn_feat

    def aux_loss(self, features, pids):
        # main.py adds this term to the total training loss when defined.
        return self._supcon_weight * self._supcon(features, pids)


# --- Factory ---
def clip_senet_v3_ibn_supcon(num_classes, loss="softmax", pretrained=True, **kwargs):
    return CLIP_SENet_V3_IBN_SupCon(
        num_classes, loss=loss, pretrained=pretrained, **kwargs
    )
