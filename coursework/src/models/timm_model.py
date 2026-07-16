import torch
import torch.nn as nn
import timm

__all__ = ["swin_t_fc512", "swin_t_custom", "deit_s_fc512"]

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class SwinT_ReID_Custom(nn.Module):
    def __init__(self, num_classes, loss='softmax', pretrained=True, **kwargs):
        super().__init__()
        self.loss = loss
        kwargs.pop('use_gpu', None)
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0,
            **kwargs
        )
        
        self.embed_dim = self.backbone.num_features 
        self.bn_neck = nn.BatchNorm1d(self.embed_dim)
        self.bn_neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.embed_dim, num_classes, bias=False)

        self.bn_neck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        global_feat = self.backbone(x) 
        bn_feat = self.bn_neck(global_feat)
        
        if self.training:
            logits = self.classifier(bn_feat)
            # Match training loop: (outputs, features)
            return logits, global_feat
        return bn_feat

class SwinT_FC512(nn.Module):
    def __init__(self, num_classes, loss='softmax', pretrained=True, **kwargs):
        super().__init__()
        self.loss = loss
        kwargs.pop('use_gpu', None)
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0,
            **kwargs
        )
        
        self.in_features = self.backbone.num_features
        self.feature_reduce = nn.Linear(self.in_features, 512)
        self.bn_neck = nn.BatchNorm1d(512)
        self.bn_neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(512, num_classes, bias=False)

        self.feature_reduce.apply(weights_init_kaiming)
        self.bn_neck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        global_feat = self.backbone(x)
        reduced_feat = self.feature_reduce(global_feat)
        bn_feat = self.bn_neck(reduced_feat)
        
        if self.training:
            logits = self.classifier(bn_feat)
            # Match training loop: (outputs, features)
            return logits, reduced_feat
        return bn_feat

class DeiT_S_FC512(nn.Module):
    """
    DeiT-Small/16 backbone with BNNeck + 512-d bottleneck.

    Architecture: deit_small_patch16_224 (384-d CLS) → Linear(384→512) → BNNeck
    Training returns (logits, pre_BN_feat) so triplet acts on un-normalised 512-d.
    Eval returns BN-normalised 512-d for distance comparison (BoT decoupling).

    Published VeRi-776 reference: 76.3 mAP / 95.5 R1 (TransReID Table 2, DeiT-S/16).
    """
    def __init__(self, num_classes, loss='softmax', pretrained=True, **kwargs):
        super().__init__()
        self.loss = loss
        kwargs.pop('use_gpu', None)
        self.backbone = timm.create_model(
            'deit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            **kwargs
        )

        self.in_features = self.backbone.num_features  # 384
        self.feature_reduce = nn.Linear(self.in_features, 512)
        self.bn_neck = nn.BatchNorm1d(512)
        self.bn_neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(512, num_classes, bias=False)

        self.feature_reduce.apply(weights_init_kaiming)
        self.bn_neck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        global_feat = self.backbone(x)               # (B, 384) CLS token
        reduced_feat = self.feature_reduce(global_feat)  # (B, 512) pre-BN
        bn_feat = self.bn_neck(reduced_feat)          # (B, 512) BN-normalised

        if self.training:
            logits = self.classifier(bn_feat)
            return logits, reduced_feat  # (outputs, features) for train loop
        return bn_feat                   # retrieval feature for eval


# --- Factory Functions ---

def swin_t_custom(num_classes, loss='softmax', pretrained=True, **kwargs):
    return SwinT_ReID_Custom(num_classes, loss, pretrained, **kwargs)

def swin_t_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    return SwinT_FC512(num_classes, loss, pretrained, **kwargs)

def deit_s_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    return DeiT_S_FC512(num_classes, loss, pretrained, **kwargs)