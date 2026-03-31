import torch
import torch.nn as nn
import timm
from .bot import weights_init_kaiming, weights_init_classifier

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for the Fine-Grained Enhancement Module.
    This module works like an 'attention mechanism'. It figures out which 
    feature channels are most important for distinguishing the car and boosts 
    their signal, while dampening the less useful channels.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # 'Squeeze' step: Average pooling condenses the spatial information into one single number per channel.
        # This gives us a global summary of what each feature channel represents.
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 'Excitation' step: Two fully connected (Linear) layers that learn
        # the complex relationships between the different feature channels.
        self.fc = nn.Sequential(
            # First, compress the channels into a smaller dimension (reduction step)
            nn.Linear(channels, channels // reduction, bias=False),
            # Apply ReLU for non-linearity (allowing it to learn complex functions)
            nn.ReLU(inplace=True),
            # Expand it back to the original number of channels
            nn.Linear(channels // reduction, channels, bias=False),
            # Sigmoid squashes the output between 0 and 1. These serve as 'weights' or percentages of importance!
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, seq_len = x.size() # b=batch_size, c=channels, seq_len=sequence length (1 in our case)
        
        # Squeeze the input sequence (from B x C x SeqLen -> B x C)
        y = self.avg_pool(x).view(b, c)
        
        # Pass through the linear layers to get our importance weights (from B x C -> B x C x 1)
        y = self.fc(y).view(b, c, 1)
        
        # Excite the original input: multiply original features by our calculated importance weights!
        return x * y.expand_as(x)

class CLIPSENet(nn.Module):
    """
    CLIP-SENet: Employs a 'frozen' CLIP image encoder with an Adaptive
    Fine-grained Enhancement Module (AFEM) based on SENet principles.
    Instead of training a neural network from scratch, it leverages the 
    powerful generic vision features learned by OpenAI's CLIP model!
    """
    def __init__(self, num_classes):
        super(CLIPSENet, self).__init__()
        self.num_classes = num_classes # The amount of distinct car identities we're trying to learn
        
        # 1. Load the pre-trained CLIP Vision Transformer (ViT-B/32) using the 'timm' library.
        # This model has already learned amazing representations by looking at millions
        # of image-text pairs from the web. 
        self.clip_backbone = timm.create_model('vit_base_patch32_clip_224', pretrained=True)
        
        # 2. FREEZE the visual backbone.
        # We deliberately stop the CLIP backbone from 'learning' or updating its parameters.
        # This preserves its robust, pre-trained generalized semantic knowledge and saves huge memory!
        for param in self.clip_backbone.parameters():
            param.requires_grad = False
            
        self.in_planes = self.clip_backbone.embed_dim # The output feature size of CLIP (usually 768)
        
        # 3. Adopt the Adaptive Fine-Grained Enhancement Module (AFEM) using our SEBlock defined above.
        # We need this because raw CLIP features are 'general purpose'. 
        # This module will adaptively focus only on the fine-grained details needed specifically for Vehicle Re-ID.
        self.afem = SEBlock(self.in_planes, reduction=16)
        
        # 4. Batch Normalization Neck (BN-Neck).
        # Normalizes the enhanced feature space to prepare it for smooth metric distance calculations.
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        # 5. Linear Classification layer.
        # This calculates probabilities for what identity ID this car belongs to during training.
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        # Step A: Ensure the input image size is exactly 224x224.
        # Standard Re-ID runs at 320x320, but because our CLIP model was originally 
        # trained on 224x224, and because we frozen its position embeddings, 
        # we must explicitly resize the image down right here.
        b, c, h, w = x.shape
        if h != 224 or w != 224:
            x_clip = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            x_clip = x
            
        # Step B: Get frozen semantic features out of the CLIP backbone
        # We wrap this in torch.no_grad() as an extra safety measure to disable gradient tracking
        with torch.no_grad():
            # Pass our image into the frozen visual backbone
            clip_features = self.clip_backbone.forward_features(x_clip) 
            
        # Step C: Extract the "Class" (CLS) token.
        # In a generic vision transformer, the CLS token (the first token in the sequence)
        # acts as a global summary representation of the entire image structure and semantics.
        cls_feat = clip_features[:, 0, :] # Shape: (Batch, 768)
        
        # Step D: Apply our fine-grained enhancement attention.
        # Because our SEBlock expects a sequence dimension, we artificially add an 
        # extra dimension of '1' to trick it into processing our 1D classification vector.
        cls_seq = cls_feat.unsqueeze(-1) # Shape: (Batch, 768, 1)
        
        # Let the neural attention highlight important attributes!
        enhanced_feat = self.afem(cls_seq) # Output shape: (Batch, 768, 1)
        
        # Remove the extra dimension we added earlier.
        features = enhanced_feat.squeeze(-1) # Back to Shape: (Batch, 768)
        
        # Step E: Normalize via the BN-Neck
        features_bn = self.bottleneck(features)
        
        # Return features directly if we are predicting/testing our model.
        if not self.training:
            return features_bn
            
        # During training, predict probabilities using the classifier branch.
        logits = self.classifier(features_bn)
        return logits, features
