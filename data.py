import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def get_transforms(is_train=True):
    """
    Returns data transforms for vehicle Re-ID.
    As per CLIP-SENet requirements, we resize to 320x320.
    """
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    if is_train:
        return T.Compose([
            T.Resize((320, 320)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize_transform,
            T.RandomErasing(p=0.5, scale=(0.02, 0.4), value=[0.485, 0.456, 0.406])
        ])
    else:
        return T.Compose([
            T.Resize((320, 320)),
            T.ToTensor(),
            normalize_transform
        ])

class VeRiDataset(Dataset):
    """
    Dataset loader for VeRi-776.
    Assumes filenames are in format: {identity}_{camera}_{...}.jpg
    """
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        
        self.image_paths = []
        self.pids = []
        self.camids = []
        
        # Load dataset
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if not filename.endswith('.jpg'):
                    continue
                # VeRi format: ID_cCam_Seq_Frame.jpg, e.g., 0002_c001_00030635_0.jpg
                parts = filename.split('_')
                if len(parts) >= 2:
                    pid = int(parts[0])
                    # Ensure camera ID parses properly (e.g. c001 -> 1)
                    camid = int(parts[1][1:]) if parts[1].startswith('c') else int(parts[1])
                    
                    self.image_paths.append(os.path.join(data_dir, filename))
                    self.pids.append(pid)
                    self.camids.append(camid)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        pid = self.pids[idx]
        camid = self.camids[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, pid, camid, img_path

def get_dataloader(data_dir, batch_size=64, is_train=True, num_workers=4):
    """
    Creates a DataLoader for the VeRi dataset.
    """
    transform = get_transforms(is_train=is_train)
    dataset = VeRiDataset(data_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True
    )
