import torch
import torch.nn as nn
import timm

def weights_init_classifier(m):
    # This function initializes the weights of our classification layer.
    # We set the weights to very small random numbers (normal distribution with 0.001 std),
    # and the biases to exactly zero. This helps the network start training smoothly.
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class TransReID(nn.Module):
    """
    TransReID architecture using ViT (Vision Transformer) 
    with Side Information Embedding (SIE).
    This model turns images into feature vectors, incorporating extra info
    like which camera took the picture or the viewpoint of the vehicle.
    """
    def __init__(self, num_classes, num_cameras=20, num_views=4):
        super(TransReID, self).__init__()
        self.num_classes = num_classes
        
        # 1. Load a pre-trained Vision Transformer (ViT) base model using the 'timm' library.
        # Pre-training gives the model generic knowledge about images.
        self.base = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.in_planes = self.base.embed_dim # The feature size (usually 768 for ViT-Base)
        
        # 2. Side Information Embedding (SIE) for camera and viewpoint.
        # These are learned vectors (parameters) that carry extra metadata information
        # to help the model distinguish between cars captured on different cameras.
        self.sie_cam = nn.Parameter(torch.zeros(num_cameras, 1, self.in_planes))
        self.sie_view = nn.Parameter(torch.zeros(num_views, 1, self.in_planes))
        
        # Initialize these embeddings with small random numbers.
        nn.init.normal_(self.sie_cam, std=0.02)
        nn.init.normal_(self.sie_view, std=0.02)
        
        # 3. Batch Normalization Neck (BN-Neck).
        # This layer acts as a buffer between the metric learning loss (Triplet Loss)
        # and the identity classification loss (Cross Entropy). It helps balance the learning.
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False) # We don't want to shift the features here
        self.bottleneck.apply(weights_init_classifier)
        
        # 4. Identity Classifier.
        # This linear layer predicts which specific vehicle 'ID' the image belongs to.
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, cam_id=None, view_id=None):
        B = x.shape[0] # B is the batch size (number of images)
        
        # Step A: Convert the image into a sequence of small patches (patch embedding)
        x = self.base.patch_embed(x)
        
        # Step B: Add Side Information Embedding (SIE) if available.
        # This tells the model *where* (camera) and *how* (view) the car was seen.
        if cam_id is not None:
            cam_emb = self.sie_cam[cam_id]
            x = x + cam_emb
            
        if view_id is not None:
            view_emb = self.sie_view[view_id]
            x = x + view_emb
            
        # Step C: Add Positional Embeddings.
        # Since transformers don't understand order natively, we add learned positional 
        # embeddings so the model knows where each patch was originally located in the image.
        cls_tokens = self.base.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # Attach the special "Class" (CLS) token at the start
        x = x + self.base.pos_embed
        x = self.base.pos_drop(x)
        
        # Step D: Pass the sequence through the Transformer blocks.
        # Here, self-attention mechanisms compare every patch to every other patch.
        for blk in self.base.blocks:
            x = blk(x)
        x = self.base.norm(x)
        
        # Step E: Extract the final representation.
        # The first token is our CLS token. It has gathered information from all other patches
        # and serves as the global feature vector representing the entire image.
        features = x[:, 0]
        
        # Step F: Pass through the BN-Neck to normalize the feature space.
        features_bn = self.bottleneck(features)
        
        # During inference (testing), we only need the feature vector to calculate distances.
        if not self.training:
            return features_bn
            
        # During training, we predict the Identity (ID) using our classifier.
        # We also return the raw features to apply the Triplet Loss.
        logits = self.classifier(features_bn)
        return logits, features
