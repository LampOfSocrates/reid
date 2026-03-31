import torch
import numpy as np
from models.bot import BoT
from models.agw import AGW
from models.transreid import TransReID
from models.pcb import PCB
from models.clip_senet import CLIPSENet
from losses.reid_loss import ReIDLoss
from eval import calculate_rank1_map

def test_models_forward():
    print("\n--- Starting Model Initializations ---")
    print("Note: This steps take time on the first run as PyTorch downloads heavy pre-trained weights (ResNet, ViT, CLIP)...")
    b, c, h, w = 2, 3, 320, 320
    dummy_input = torch.randn(b, c, h, w)
    dummy_labels = torch.randint(0, 10, (b,))
    
    models = {}
    
    print("\n[1/5] Initializing BoT (Downloading ResNet-50 weights if missing)...")
    models['BoT'] = BoT(num_classes=10)
    print("      -> BoT initialized successfully!")

    print("[2/5] Initializing AGW (Uses ResNet-50 weights)...")
    models['AGW'] = AGW(num_classes=10)
    print("      -> AGW initialized successfully!")

    print("[3/5] Initializing TransReID (Downloading ViT-Base weights if missing)...")
    models['TransReID'] = TransReID(num_classes=10)
    print("      -> TransReID initialized successfully!")

    print("[4/5] Initializing PCB (Uses ResNet-50 weights)...")
    models['PCB'] = PCB(num_classes=10)
    print("      -> PCB initialized successfully!")

    print("[5/5] Initializing CLIP-SENet (Downloading CLIP ViT-B/32 weights if missing)...")
    models['CLIPSENet'] = CLIPSENet(num_classes=10)
    print("      -> CLIP-SENet initialized successfully!")
    
    print("\n--- Testing Forward Passes ---")
    criterion = ReIDLoss(num_classes=10)
    
    for name, model in models.items():
        try:
            model.train()
            if name == 'TransReID':
                logits, features = model(dummy_input, cam_id=torch.zeros(b, dtype=torch.long))
            else:
                logits, features = model(dummy_input)
                
            loss = criterion(logits, features, dummy_labels)
            loss.backward()
            print(f"[{name}] Forward & Backward SUCCESS")
        except Exception as e:
            print(f"[{name}] ERROR: {e}")

def test_eval():
    print("Testing eval function...")
    qf = torch.randn(2, 256)
    gf = torch.randn(4, 256)
    from eval import euclidean_distance
    dist = euclidean_distance(qf, gf)
    
    q_pids = np.array([1, 2])
    g_pids = np.array([1, 1, 2, 3])
    q_camids = np.array([1, 1])
    g_camids = np.array([2, 2, 2, 2])
    
    r1, map_val = calculate_rank1_map(dist, q_pids, g_pids, q_camids, g_camids)
    print(f"Eval SUCCESS. Rank-1: {r1:.4f}, mAP: {map_val:.4f}")

if __name__ == "__main__":
    test_models_forward()
    test_eval()
