import torch
import torch.nn as nn

def normalize(x, axis=-1):
    """
    Normalizing a tensor to unit length along the specified dimension.
    Imagine making all arrows the exact same length (length 1), 
    so we only care about the *direction* they point, not how long they are!
    """
    # We add 1e-12 (a tiny number) to prevent dividing by zero
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(x, y):
    """
    Computes Euclidean distance between two matrices of feature vectors.
    Euclidean distance is just the straight-line distance between two points in space.
    We compute this highly efficiently using matrix multiplications instead of slow loops.
    The formula is: (x - y)^2 = x^2 + y^2 - 2*x*y
    """
    m, n = x.size(0), y.size(0)
    
    # Calculate x^2
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    
    # Calculate y^2
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    
    # Add them together
    dist = xx + yy
    
    # Subtract 2*x*y (the -2 is added inside the function)
    dist.addmm_(1, -2, x, y.t())
    
    # Take the square root to get the final straight line distance!
    # Clamp prevents tiny negative numbers caused by floating point computer errors.
    dist = dist.clamp(min=1e-12).sqrt()  
    return dist

class TripletLoss(nn.Module):
    """
    Triplet loss with hard positive/negative mining.
    This is the core of "Metric Learning". 
    Instead of guessing a class, Triplet Loss looks at 3 images at a time:
      1. Anchor: A baseline image (e.g., Red Honda Civic).
      2. Positive: Another image of the EXACT SAME Red Honda Civic.
      3. Negative: An image of a DIFFERENT car.
      
    Goal: Pull the Positive closer to the Anchor, and push the Negative away by at least a 'margin'.
    "Hard mining" means we only look at the hardest examples 
    (the furthest Positive and the closest Negative in the batch) to learn faster!
    """
    def __init__(self, margin=0.3, normalize_feature=False):
        super(TripletLoss, self).__init__()
        self.margin = margin # How much further away the negative must be than the positive
        self.normalize_feature = normalize_feature # Whether to make all vectors length 1
        
        # PyTorch's built-in ranking loss that enforces: Dist(Anchor, Negative) > Dist(Anchor, Positive) + Margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw numeric feature vectors for every image in the batch (Batch, 2048)
            targets: True vehicle IDs for those images (Batch)
        """
        # Step A: Normalize if requested (used mainly alongside Attention models)
        if self.normalize_feature:
            inputs = normalize(inputs, axis=-1)

        # Step B: Calculate the straight-line distance between EVERY image and EVERY OTHER image in the batch
        dist_mat = euclidean_dist(inputs, inputs)
        
        N = dist_mat.size(0) # Number of images
        
        # Step C: Create True/False matrices
        # is_pos: True if two images share the exact same vehicle ID (Target)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())
        
        # is_neg: True if two images have DIFFERENT vehicle IDs
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())

        # Step D: Hard Positive/Negative Mining!
        # Search for the hardest positive (the biggest distance between the same car)
        dist_ap = dist_mat[is_pos].view(N, -1).max(1, keepdim=True)[0]
        
        # Search for the hardest negative (the shortest distance between different cars)
        dist_an = dist_mat[is_neg].view(N, -1).min(1, keepdim=True)[0]
        
        # Step E: Apply the margin ranking loss
        y = torch.ones_like(dist_an) # '1' means we want "dist_an" to be larger than "dist_ap"
        
        # This calculates how badly the model failed, passing it back as the 'loss' to learn from
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
