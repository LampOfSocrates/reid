import torch
import torch.nn as nn
import timm

__all__ = ["clip_senet_v1_dual", "clip_senet_v2_frozen"]

# --- Initialization Helpers ---
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None: nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)

# --- Version 1: Dual-Stream (Fine-tune) ---
class CLIP_SENet_V1_Dual(nn.Module):
    """
    Appearance (CNN) + Semantic (TinyCLIP) fused.
    Both streams are TRAINABLE.
    """
    def __init__(self, num_classes, pretrained=True, **kwargs):
        super().__init__()
        # 1. Appearance Stream
        self.cnn = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
        # 2. Semantic Stream (Distilled CLIP)
        self.tiny_clip = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, num_classes=0)
        
        # Combined dim: 2048 (ResNet) + 192 (TinyViT) = 2240
        self.reduce = nn.Linear(2240, 512)
        self.bn_neck = nn.BatchNorm1d(512)
        self.bn_neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(512, num_classes, bias=False)

        self.reduce.apply(weights_init_kaiming)
        self.bn_neck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        f_app = self.cnn(x)
        f_sem = self.tiny_clip(x)
        combined = torch.cat([f_app, f_sem], dim=1)
        
        feat = self.reduce(combined)
        bn_feat = self.bn_neck(feat)
        
        if self.training:
            return self.classifier(bn_feat), feat # (outputs, features)
        return bn_feat

# --- Version 2: Single-Stream (Frozen) ---
class CLIP_SENet_V2_Frozen(nn.Module):
    """
    Full CLIP ViT backbone. 
    Backbone is FROZEN; only AFEM and Head are TRAINABLE.
    """
    def __init__(self, num_classes, pretrained=True, **kwargs):
        super().__init__()
        self.backbone = timm.create_model("vit_base_patch32_clip_224", pretrained=pretrained)
        
        # Freeze Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # AFEM (Squeeze-and-Excitation on CLS token)
        self.afem = nn.Sequential(
            nn.Linear(768, 48, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(48, 768, bias=False),
            nn.Sigmoid()
        )
        
        self.reduce = nn.Linear(768, 512)
        self.bn_neck = nn.BatchNorm1d(512)
        self.bn_neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(512, num_classes, bias=False)

        self.afem.apply(weights_init_kaiming)
        self.reduce.apply(weights_init_kaiming)
        self.bn_neck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        # CLIP requirement: 224x224
        if x.shape[2:] != (224, 224):
            x = nn.functional.interpolate(x, size=(224, 224), mode="bilinear")
        
        with torch.no_grad():
            # Extract CLS token [Batch, 768]
            f = self.backbone.forward_features(x)[:, 0, :]
        
        # Apply Adaptive Enhancement
        f = f * self.afem(f)
        
        feat = self.reduce(f)
        bn_feat = self.bn_neck(feat)
        
        if self.training:
            return self.classifier(bn_feat), feat # (outputs, features)
        return bn_feat

# --- Factory Functions ---
def clip_senet_v1_dual(num_classes, **kwargs):
    return CLIP_SENet_V1_Dual(num_classes, **kwargs)

def clip_senet_v2_frozen(num_classes, **kwargs):
    return CLIP_SENet_V2_Frozen(num_classes, **kwargs)