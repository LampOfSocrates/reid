import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import csv
from data import get_dataloader
from losses.reid_loss import ReIDLoss
from models.bot import BoT
from models.agw import AGW
from models.transreid import TransReID
from models.pcb import PCB
from models.clip_senet import CLIPSENet
import subprocess


def train(model_name, epochs, batch_size, data_dir, use_gpu=True):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Num classes for VeRi-776 train set is 576.
    num_classes = 576

    # Initialize Model
    if model_name == 'bot':
        model = BoT(num_classes)
        agw_weighted = False
    elif model_name == 'agw':
        model = AGW(num_classes)
        agw_weighted = True
    elif model_name == 'transreid':
        model = TransReID(num_classes)
        agw_weighted = False
    elif model_name == 'pcb':
        model = PCB(num_classes)
        agw_weighted = False
    elif model_name == 'clip_senet':
        model = CLIPSENet(num_classes)
        agw_weighted = False
    else:
        raise ValueError("Invalid model name")

    model = model.to(device)

    # Loss and Optimizer
    criterion = ReIDLoss(num_classes=num_classes, agw_weighted=agw_weighted).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00035, weight_decay=5e-4)

    # Dataloader
    train_loader = get_dataloader(data_dir, batch_size=batch_size, is_train=True)

    # Metric collection setup
    metrics_csv_path = f"metrics_{model_name}.csv"
    with open(metrics_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss"])

    print(f"Metrics will be logged to {metrics_csv_path}")

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # TQDM wrapper
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, pids, camids, _ in pbar:
            imgs = imgs.to(device)
            pids = pids.to(device)
            camids = camids.to(device)

            optimizer.zero_grad()

            # Forward pass
            if model_name == 'transreid':
                logits, features = model(imgs, cam_id=camids)
            else:
                logits, features = model(imgs)

            loss = criterion(logits, features, pids)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished, Average Loss: {avg_loss:.4f}")

        # Save metrics to CSV
        with open(metrics_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss])

    print("Training completed. Invoking plot_metrics to render outputs...")
    try:
        subprocess.run(["python", "plot_metrics.py", "--csv", metrics_csv_path, "--output", f"plot_{model_name}.png"], check=True)
    except Exception as e:
        print(f"Failed to plot metrics magically: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vehicle Re-ID Training Script")
    parser.add_argument('--model', type=str, default='bot', choices=['bot', 'agw', 'transreid', 'pcb', 'clip_senet'])
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_dir', type=str, default='datasets/VeRi/image_train')
    args = parser.parse_args()

    train(args.model, args.epochs, args.batch_size, args.data_dir)
