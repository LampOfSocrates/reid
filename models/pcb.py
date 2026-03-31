import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def weights_init_kaiming(m):
    # Initializes weights using Kaiming He's method, which is good for layers followed by ReLU.
    # It sets weights to small random numbers, helping to maintain a stable variance of outputs across layers.
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    # Initializes the final classification weights with a very small standard deviation
    # to avoid extreme initial predictions.
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class PCB(nn.Module):
    """
    Part-based Convolutional Baseline (PCB) architecture.
    Instead of looking at the image as one whole piece, PCB splits the image 
    horizontally into multiple "parts" (or strips) and classifies each part independently.
    This helps the model focus on local details (like a car's bumper or roof) 
    that might be missed if we only look at the global picture.
    """
    def __init__(self, num_classes, num_parts=6):
        super(PCB, self).__init__()
        self.num_parts = num_parts # By default, split into 6 horizontal parts
        self.num_classes = num_classes # The number of unique identities to predict
        self.in_planes = 2048 # ResNet-50 outputs features of size 2048

        # 1. Load a pre-trained ResNet-50 model as our backbone feature extractor.
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # 2. Remove the standard average pool and classifier layers since we'll build our own.
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        
        # 3. Modify the last convolution layer (layer4). 
        # By default, ResNet shrinks the image spatial size here (stride=2).
        # We change it to stride=1 so the output "feature map" remains larger,
        # giving us more spatial resolution to split the image into parts.
        self.base[7][0].conv2.stride = (1, 1)
        self.base[7][0].downsample[0].stride = (1, 1)

        # 4. A pooling layer that mathematically squashes the feature map
        # into exactly 'num_parts' vertically, and 1 horizontally.
        self.avgpool = nn.AdaptiveAvgPool2d((self.num_parts, 1))
        
        # We need independent Batch Norm Neck (BN-Neck) and Classifiers 
        # for EACH of the parts we split the image into.
        self.bottlenecks = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        
        for i in range(self.num_parts):
            # Create a Batch Normalization layer for the separated part
            bottleneck = nn.BatchNorm1d(self.in_planes)
            bottleneck.bias.requires_grad_(False)
            bottleneck.apply(weights_init_kaiming)
            self.bottlenecks.append(bottleneck)
            
            # Create a linear classifier for the separated part
            classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            classifier.apply(weights_init_classifier)
            self.classifiers.append(classifier)

    def forward(self, x):
        # Step A: Extract deep features from the image using ResNet backbone.
        features = self.base(x)      # Output shape: (Batch Size, 2048, Height, Width)
        
        # Step B: Pool the features spatially into 6 horizontal parts.
        features = self.avgpool(features) # Output shape: (Batch Size, 2048, 6, 1)
        
        part_features = []       # To store features before normalization
        part_features_bn = []    # To store features after normalization
        part_logits = []         # To store prediction probabilities
        
        # Step C: Loop through each of the 6 parts individually.
        for i in range(self.num_parts):
            # Extract the raw feature vector for part 'i'
            f_i = features[:, :, i, 0] # Output shape: (B, 2048)
            
            # Normalize the feature using its dedicated BN layer
            f_bn_i = self.bottlenecks[i](f_i)
            
            part_features.append(f_i)
            part_features_bn.append(f_bn_i)
            
            # Step D: Only run the classifiers if we are training the model.
            if self.training:
                part_logits.append(self.classifiers[i](f_bn_i))
                
        # During inference (testing), we combine all 6 normalized parts
        # into one giant feature vector for comparing distance.
        if not self.training:
            return torch.cat(part_features_bn, dim=1) # Shape: (B, 2048 * 6)
            
        # During training, return both the 6 predictions and 6 raw features 
        # to calculate multiple individual losses later.
        return part_logits, part_features
