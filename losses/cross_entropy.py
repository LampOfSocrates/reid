import torch
import torch.nn as nn

class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross-entropy loss with label smoothing to prevent overfitting.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        
        # Smooth targets
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        
        # Cross entropy computes: sum(targets * -log_probs)
        loss = (- targets * log_probs).mean(0).sum()
        return loss
