import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
    """
    Non-local block to capture "long-range dependencies".
    Normally, CNNs only look at small patches of an image at a time (like 3x3 pixels).
    This block allows a pixel in the top-left corner to talk directly to a pixel 
    in the bottom-right corner, helping the model understand global context 
    (like how the front tire relates to the back window of the car!).
    """
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2 # Shrinking channels saves massive memory speed
        
        # We need 3 different "views" of the image to perform the self-attention math
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        # After we figure out the attention, we blow the channels back up to the original size
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            nn.BatchNorm2d(self.in_channels)
        )
        # Initialize as 0 so at the very start of training, this block doesn't ruin the pre-trained ResNet
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, x):
        b, c, h, w = x.size() # Batch, Channels, Height, Width
        
        # 1. Flatten the height/width into one long line of pixels
        # g_x shape: (Batch, Channels/2, Height * Width). The permute flips it.
        g_x = self.g(x).view(b, self.inter_channels, -1).permute(0, 2, 1)
        
        # 2. Do the exact same to our other two sets
        theta_x = self.theta(x).view(b, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(b, self.inter_channels, -1)
        
        # 3. The attention math! We multiply theta and phi.
        # This gives us a giant (Height*Width x Height*Width) grid.
        # Every pixel scores how much it should "care" about every other pixel!
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1) # Softmax turns those scores into percentages (adding to 100%)
        
        # 4. We apply those percentage weights to our actual pixel values (g_x).
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_channels, h, w) # Shape it back into a square picture!
        
        W_y = self.W(y)
        
        # 5. Add the newly enhanced pixel data on top of the original image (Residual connection)
        z = W_y + x
        return z

class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    Standard Global Average Pooling takes a strict mathematical average of the picture.
    Max Pooling takes only the single brightest/loudest pixel.
    GeM acts as a slidable scale between the two! 
    By setting 'p=3', it allows the model to focus on the brighter/sharper 
    features without completely ignoring the softer background details.
    """
    def __init__(self, p=3, eps=1e-6):
        super(GeMPooling, self).__init__()
        # Trainable parameter 'p'. We start it at 3, but the neural net can adjust it natively later!
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps # A tiny number to stop division-by-zero math crash errors

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

from .bot import weights_init_kaiming, weights_init_classifier

class AGW(nn.Module):
    """
    AGW stands for: Attention (Non-local block), Generalized mean pooling, 
    and Weighted triplet loss. This is one of the leading CNN-based Re-ID architectures!
    """
    def __init__(self, num_classes):
        super(AGW, self).__init__()
        
        # 1. Load the pre-trained ResNet-50 visual brain
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        
        # 2. Maintain high resolution! Change the stride from 2 to 1 in the final block.
        self.base[7][0].conv2.stride = (1, 1)
        self.base[7][0].downsample[0].stride = (1, 1)
        
        # 3. Insert the Non-Local block deep inside the network (right after the 3rd layer block)
        # Layer 3 outputs exactly 1024 channels.
        self.nl_block = NonLocalBlock(1024) 
        
        # 4. Use GeM Pool instead of regular Average Pool
        self.pool = GeMPooling(p=3.0)
        
        self.num_classes = num_classes
        self.in_planes = 2048 # ResNet's final layer outputs 2048 dimensions

        # 5. Prepare the famous "BN-Neck" to buffer the conflicting losses
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # 6. Final linear prediction layer to guess the specific car Identity
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        # Step A: Pass the image through the first layers of ResNet manually.
        # Why manually? Because we need to carefully inject our Non-Local block in the middle!
        for i in range(7):
            x = self.base[i](x)
            
        # Step B: At the end of layer 3, apply our "long-range attention"
        x = self.nl_block(x) 
        
        # Step C: Give it back to ResNet to finish the final deep convolution block (layer 4)
        x = self.base[7](x)  # Shape out: (B, 2048, H, W)
        
        # Step D: Compress the height and width down to 1x1 using our GeM pool math
        features = self.pool(x)       
        features = features.view(features.size(0), -1) # Shape: (B, 2048)
        
        # Step E: Normalize via the bottle-neck for Metric Training
        features_bn = self.bottleneck(features)
        
        # Testing mode: just return the clean math vectors to calculate distances!
        if not self.training:
            return features_bn
            
        # Training mode: spit out ID probabilities, and the unnormalized base features.
        logits = self.classifier(features_bn)
        return logits, features
