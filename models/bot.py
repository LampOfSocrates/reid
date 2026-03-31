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

class BoT(nn.Module):
    """
    Bag of Tricks (BoT) network architecture.
    This is one of the strongest standard baselines in Re-ID.
    It takes a classic standard network (ResNet-50) and applies specific
    'tricks' like removing downsampling and adding a Batch-Norm Neck 
    layer to balance two very different types of training losses.
    """
    def __init__(self, num_classes):
        super(BoT, self).__init__()
        # 1. Load a pre-trained ResNet-50 model.
        # This backbone has seen lots of real-world images (ImageNet dataset)
        # and inherently knows how to extract great edges, shapes, and features!
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # 2. Extract specific layers of the ResNet.
        # We drop the final "average pooling" and "classifier" layers because
        # we will build our own custom Re-ID head.
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        
        # 3. Modification Trick: Keep spatial resolution large.
        # In a generic ResNet, the 4th block halves the image size (stride=2).
        # We deliberately set the stride to 1 here so the output image 'feature map'
        # remains double the size. This retains much more fine-grained detail of the car!
        self.base[7][0].conv2.stride = (1, 1)
        self.base[7][0].downsample[0].stride = (1, 1)
        
        # 4. Global Average Pooling (GAP) Layer.
        # This mathematically squashes our large feature map into a single summary vector of size 2048.
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.num_classes = num_classes # How many unique car identities we have
        self.in_planes = 2048 # ResNet50 outputs a vector of size 2048

        # 5. The "BN-Neck" Trick.
        # Re-ID uses two losses: Identity Loss (Cross-Entropy) and Metric Loss (Triplet).
        # Triplet Loss works best when features are spread out (not normalized).
        # Identity Loss works best when features are normalized on a hypersphere.
        # The BN-Neck separates them! Triplet Loss uses the raw 'features' before this BN layer,
        # while Identity Loss uses the 'features_bn' after this BN layer.
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # We disable the bias shift for smooth metric learning
        self.bottleneck.apply(weights_init_kaiming)

        # 6. Classifier Identity layer.
        # Calculates the probabilities that the extracted features belong to 
        # a certain car identity out of `num_classes` possible identities!
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        # Step A: Pass the raw image pixels through our powerful ResNet backbone
        features = self.base(x)      # Output shape: (Batch Size, 2048, Height, Width)
        
        # Step B: Squash the height and width spatial features into a 1x1 summary!
        features = self.gap(features) # Output shape: (Batch Size, 2048, 1, 1)
        
        # Flatten out the empty 1x1 dimensions to just give us a 1D vector per image
        features = features.view(features.size(0), -1) # Output shape: (Batch Size, 2048)
        
        # Step C: Pass those raw features right through the BN-Neck normalizing buffer.
        features_bn = self.bottleneck(features)
        
        # During inference (testing), we do NOT care about classifying the car ID. 
        # All we want is the numeric vector 'features_bn' to calculate Euclidean distances.
        if not self.training:
            return features_bn
            
        # Step D: During training, feed the normalized bottleneck features into the linear classifier.
        # We return both the Identity log-probabilities (for Cross-Entropy) and the 
        # raw unnormalized features (for Triplet Metric Learning).
        logits = self.classifier(features_bn)
        return logits, features
