import torch
import torch.nn as nn
from .cross_entropy import CrossEntropyLabelSmooth
from .triplet import TripletLoss

class ReIDLoss(nn.Module):
    """
    Unified class connecting CE and Triplet Loss. 
    Handles multilogit sequences such as PCB gracefully.
    """
    def __init__(self, num_classes, epsilon=0.1, margin=0.3, agw_weighted=False):
        super(ReIDLoss, self).__init__()
        self.ce_loss = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=epsilon)
        self.triplet_loss = TripletLoss(margin=margin, normalize_feature=agw_weighted)
    
    def forward(self, logits, features, targets):
        """
        Args:
            logits: Single tensor (B, num_classes) or List of Tensors for PCB
            features: Single feature tensor or List of Tensors for PCB
            targets: target classes (B)
        """
        if isinstance(logits, list):
            # E.g., PCB outputs a list of 6 logits and 6 features
            ce_loss_val = 0
            triplet_loss_val = 0
            for i in range(len(logits)):
                ce_loss_val += self.ce_loss(logits[i], targets)
                triplet_loss_val += self.triplet_loss(features[i], targets)
            
            ce_loss_val /= len(logits)
            triplet_loss_val /= len(logits)
        
        else:
            # Standard single output models (BoT, AGW, TransReID, CLIP-SENet)
            ce_loss_val = self.ce_loss(logits, targets)
            triplet_loss_val = self.triplet_loss(features, targets)
            
        total_loss = ce_loss_val + triplet_loss_val
        return total_loss
