import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from datasets_loader import get_dataloader
from models.hybrid_model import build_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--client_path", type=str, required=True,
                        help="Path to prepared client dataset")
    parser.add_argument("--client_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resnet_type", type=str, default="resnet50")
    parser.add_argument("--save_dir", type=str, default="client_outputs")

    return parser.parse_args()



def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()
    running_loss = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)



def evaluate(model, loader, device):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:,1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)



def save_metrics(labels, preds, probs, save_path):

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds).tolist()

    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0

    report = classification_report(labels, preds)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm
    }

    # Save JSON
    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save Classification Report
    with open(os.path.join(save_path, "classification_report.txt"), "w") as f:
        f.write(report)

    print("\n===== Evaluation Metrics =====")
    print(json.dumps(metrics, indent=4))
    print("\nClassification Report:\n", report)


def main():

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = os.path.join(args.save_dir, args.client_name)
    os.makedirs(save_path, exist_ok=True)


    train_loader = get_dataloader(
        args.client_path,
        split="train",
        batch_size=args.batch_size
    )

    test_loader = get_dataloader(
        args.client_path,
        split="test",
        batch_size=args.batch_size
    )


    model = build_model(args.resnet_type).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0


    for epoch in range(args.epochs):

        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        labels, preds, probs = evaluate(model, test_loader, device)

        acc = accuracy_score(labels, preds)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Accuracy: {acc:.4f}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),
                       os.path.join(save_path, "best_model.pth"))


    labels, preds, probs = evaluate(model, test_loader, device)

    save_metrics(labels, preds, probs, save_path)

    print("\nTraining Completed for", args.client_name)


if __name__ == "__main__":
    main()