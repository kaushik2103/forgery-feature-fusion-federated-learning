# ============================================================
# Client Training Script (FedAvg Compatible)
# ============================================================

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

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
    roc_auc_score,
    roc_curve
)

from datasets_loader import get_dataloader
from models.hybrid_model import build_model


# ============================================================
# Argument Parser
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--client_path", type=str, required=True)
    parser.add_argument("--client_name", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=12)

    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--resnet_type", type=str, default="resnet50")

    parser.add_argument("--global_model", type=str, default=None)

    parser.add_argument("--save_dir", type=str, default="client_outputs")

    return parser.parse_args()


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    running_loss = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, device):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in tqdm(loader, desc="Testing", leave=False):

            images = images.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:,1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(labels, preds, probs):

    acc = accuracy_score(labels, preds)

    precision = precision_score(labels, preds, zero_division=0)

    recall = recall_score(labels, preds, zero_division=0)

    f1 = f1_score(labels, preds, zero_division=0)

    auc = roc_auc_score(labels, probs)

    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    }, cm


# ============================================================
# Plot Functions
# ============================================================

def plot_training_curve(history, save_path):

    df = pd.DataFrame(history)

    plt.figure(figsize=(8,5))

    plt.plot(df["epoch"], df["accuracy"], label="Accuracy")
    plt.plot(df["epoch"], df["train_loss"], label="Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Value")

    plt.legend()

    plt.title("Training Curve")

    plt.savefig(os.path.join(save_path,"training_curve.png"))

    plt.close()


def plot_confusion_matrix(cm, save_path):

    plt.figure(figsize=(5,5))

    plt.imshow(cm, cmap="Blues")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,cm[i,j],ha="center",va="center")

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(os.path.join(save_path,"confusion_matrix.png"))

    plt.close()


def plot_roc(labels, probs, save_path):

    fpr, tpr, _ = roc_curve(labels, probs)

    plt.figure()

    plt.plot(fpr,tpr)

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.title("ROC Curve")

    plt.savefig(os.path.join(save_path,"roc_curve.png"))

    plt.close()


# ============================================================
# Save Results
# ============================================================

def save_results(metrics, cm, labels, preds, probs, save_path):

    report = classification_report(labels, preds)

    with open(os.path.join(save_path,"metrics.json"),"w") as f:
        json.dump(metrics,f,indent=4)

    with open(os.path.join(save_path,"classification_report.txt"),"w") as f:
        f.write(report)

    np.save(os.path.join(save_path,"confusion_matrix.npy"),cm)

    plot_confusion_matrix(cm,save_path)

    plot_roc(labels,probs,save_path)

    pd.DataFrame({
        "label":labels,
        "prediction":preds,
        "probability":probs
    }).to_csv(
        os.path.join(save_path,"predictions.csv"),
        index=False
    )

    print("\nFinal Metrics")

    print(json.dumps(metrics,indent=4))


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = os.path.join(args.save_dir,args.client_name)

    os.makedirs(save_path,exist_ok=True)

    print("\nDevice:",device)

    # ========================================================
    # Load Data
    # ========================================================

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

    dataset_size = len(train_loader.dataset)

    print("Client Dataset Size:", dataset_size)

    # ========================================================
    # Model
    # ========================================================

    model = build_model(args.resnet_type).to(device)

    if args.global_model and os.path.exists(args.global_model):

        print("\nLoading Global Model")

        model.load_state_dict(
            torch.load(args.global_model,map_location=device)
        )

    # ========================================================
    # Training Setup
    # ========================================================

    class_weights = torch.tensor([1.0,1.5]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2,
        factor=0.5
    )

    best_acc = 0

    history = []

    patience = 3

    early_stop_counter = 0

    # ========================================================
    # Training Loop
    # ========================================================

    for epoch in range(1,args.epochs+1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        labels,preds,probs = evaluate(model,test_loader,device)

        metrics,cm = compute_metrics(labels,preds,probs)

        scheduler.step(metrics["accuracy"])

        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss

        history.append(metrics)

        print("Epoch",epoch,"Accuracy",metrics["accuracy"])

        if metrics["accuracy"] > best_acc:

            best_acc = metrics["accuracy"]

            torch.save(
                model.state_dict(),
                os.path.join(save_path,"best_model.pth")
            )

            early_stop_counter = 0

        else:

            early_stop_counter += 1

            if early_stop_counter >= patience:

                print("Early stopping")

                break

    torch.save(
        model.state_dict(),
        os.path.join(save_path,"last_model.pth")
    )

    # ========================================================
    # Save dataset size for FedAvg
    # ========================================================

    with open(os.path.join(save_path,"client_info.json"),"w") as f:

        json.dump({
            "client_name":args.client_name,
            "dataset_size":dataset_size
        },f,indent=4)

    # ========================================================
    # Save Training History
    # ========================================================

    with open(os.path.join(save_path,"training_history.json"),"w") as f:
        json.dump(history,f,indent=4)

    pd.DataFrame(history).to_csv(
        os.path.join(save_path,"epoch_metrics.csv"),
        index=False
    )

    plot_training_curve(history,save_path)

    # ========================================================
    # Final Evaluation
    # ========================================================

    labels,preds,probs = evaluate(model,test_loader,device)

    final_metrics,cm = compute_metrics(labels,preds,probs)

    save_results(final_metrics,cm,labels,preds,probs,save_path)

    print("\nClient Training Completed")


if __name__ == "__main__":
    main()