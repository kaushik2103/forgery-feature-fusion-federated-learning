import os
import json
import copy
import torch
import argparse
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from datasets_loader import get_dataloader
from models.hybrid_model import build_model
from server_aggregation import fusion_aggregation


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resnet_type", type=str, default="resnet50")

    parser.add_argument("--client_paths", nargs="+", required=True)
    parser.add_argument("--global_test_path", type=str,
                        default="balanced_data/global_test")

    parser.add_argument("--save_dir", type=str, default="federated_results")

    return parser.parse_args()


def local_train(model, train_loader, epochs, device, lr):

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, loader, device):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:,1].cpu().numpy())

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    }


def main():

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize global model
    global_model = build_model(args.resnet_type).to(device)

    round_metrics = []
    best_acc = 0

    # Global Test Loader
    global_loader = get_dataloader(
        args.global_test_path,
        split="test",
        batch_size=args.batch_size
    )


    for round_num in range(args.rounds):

        print(f"\n========== Round {round_num+1}/{args.rounds} ==========")

        client_models = []


        for client_id, client_path in enumerate(args.client_paths):

            print(f"\nClient {client_id+1} Training...")

            # Load client data
            train_loader = get_dataloader(
                client_path,
                split="train",
                batch_size=args.batch_size
            )

            # Copy global model
            local_model = copy.deepcopy(global_model)

            # Local train
            local_model = local_train(
                local_model,
                train_loader,
                args.local_epochs,
                device,
                args.lr
            )

            client_models.append(local_model)


        weights = [1.0] * len(client_models)

        global_model = fusion_aggregation(client_models, weights)
        global_model.to(device)


        metrics = evaluate(global_model, global_loader, device)

        print("\nGlobal Metrics:", metrics)

        round_metrics.append({
            "round": round_num+1,
            **metrics
        })

        # Save best model
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save(
                global_model.state_dict(),
                os.path.join(args.save_dir, "best_global_model.pth")
            )


    with open(os.path.join(args.save_dir, "round_metrics.json"), "w") as f:
        json.dump(round_metrics, f, indent=4)

    print("\nFederated Training Completed")
    print("Best Global Accuracy:", best_acc)


if __name__ == "__main__":
    main()