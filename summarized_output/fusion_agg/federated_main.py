import os
import json
import copy
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from datasets_loader import get_dataloader
from models.hybrid_model import build_model
from server_aggregation import fuse_models



def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--resnet_type", type=str, default="resnet50")

    parser.add_argument("--client_paths", nargs="+", required=True)

    parser.add_argument("--global_test_path", type=str, required=True)

    parser.add_argument("--save_dir", type=str, default="federated_results")

    return parser.parse_args()



def local_train(model, train_loader, epochs, device, lr):

    model.train()

    class_weights = torch.tensor([1.0, 1.5]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2,
        factor=0.5
    )

    for epoch in range(epochs):

        epoch_loss = 0

        progress = tqdm(train_loader, desc=f"Local Epoch {epoch+1}", leave=False)

        for images, labels in progress:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_loss += loss.item()

            progress.set_postfix(loss=loss.item())

        epoch_loss /= len(train_loader)

        scheduler.step(epoch_loss)

    return model, epoch_loss



def evaluate(model, loader, device):

    model.eval()

    preds = []
    labels = []
    probs = []

    with torch.no_grad():

        for images, y in tqdm(loader, desc="Evaluation", leave=False):

            images = images.to(device)

            outputs = model(images)

            prob = torch.softmax(outputs, dim=1)

            pred = torch.argmax(prob, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.numpy())
            probs.extend(prob[:,1].cpu().numpy())

    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    acc = accuracy_score(labels, preds)

    precision = precision_score(labels, preds, zero_division=0)

    recall = recall_score(labels, preds, zero_division=0)

    f1 = f1_score(labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0

    cm = confusion_matrix(labels, preds)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    }

    return metrics, labels, preds, probs, cm



def plot_curves(round_metrics, save_dir):

    df = pd.DataFrame(round_metrics)

    plt.figure()

    plt.plot(df["round"], df["accuracy"])

    plt.xlabel("Round")

    plt.ylabel("Accuracy")

    plt.title("Global Accuracy")

    plt.savefig(os.path.join(save_dir,"accuracy_curve.png"))

    plt.close()

    plt.figure()

    plt.plot(df["round"], df["f1_score"])

    plt.xlabel("Round")

    plt.ylabel("F1 Score")

    plt.title("Global F1 Score")

    plt.savefig(os.path.join(save_dir,"f1_curve.png"))

    plt.close()


def main():

    args = parse_args()

    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)

    print("\nUsing device:", device)

    with open(os.path.join(args.save_dir,"federated_config.json"),"w") as f:

        json.dump(vars(args),f,indent=4)

    global_model = build_model(args.resnet_type).to(device)

    round_metrics = []
    client_history = []

    best_acc = 0

    global_loader = get_dataloader(
        args.global_test_path,
        split="test",
        batch_size=args.batch_size
    )


    for round_num in range(args.rounds):

        print("\n==============================")
        print(f"Round {round_num+1}/{args.rounds}")
        print("==============================")

        client_models = []
        client_round_metrics = []


        for client_id, client_path in enumerate(args.client_paths):

            print(f"\nTraining Client {client_id+1}")

            train_loader = get_dataloader(
                client_path,
                split="train",
                batch_size=args.batch_size
            )

            test_loader = get_dataloader(
                client_path,
                split="test",
                batch_size=args.batch_size
            )

            local_model = copy.deepcopy(global_model)

            local_model, train_loss = local_train(
                local_model,
                train_loader,
                args.local_epochs,
                device,
                args.lr
            )

            metrics, _, _, _, _ = evaluate(local_model, test_loader, device)

            metrics["client"] = client_id+1
            metrics["round"] = round_num+1
            metrics["train_loss"] = train_loss

            client_round_metrics.append(metrics)

            print("Client Metrics:", metrics)

            client_models.append(local_model)

        client_history.extend(client_round_metrics)


        global_model = fuse_models(client_models, args.resnet_type)

        global_model.to(device)

        torch.save(
            global_model.state_dict(),
            os.path.join(args.save_dir,f"global_model_round_{round_num+1}.pth")
        )


        metrics, labels, preds, probs, cm = evaluate(
            global_model,
            global_loader,
            device
        )

        metrics["round"] = round_num+1

        round_metrics.append(metrics)

        print("\nGlobal Metrics:", metrics)

        pd.DataFrame({
            "label":labels,
            "prediction":preds,
            "probability":probs
        }).to_csv(
            os.path.join(args.save_dir,f"predictions_round_{round_num+1}.csv"),
            index=False
        )

        np.save(
            os.path.join(args.save_dir,f"confusion_matrix_round_{round_num+1}.npy"),
            cm
        )

        if metrics["accuracy"] > best_acc:

            best_acc = metrics["accuracy"]

            torch.save(
                global_model.state_dict(),
                os.path.join(args.save_dir,"best_global_model.pth")
            )

            print("New best global model saved")

        torch.cuda.empty_cache()


    with open(os.path.join(args.save_dir,"round_metrics.json"),"w") as f:

        json.dump(round_metrics,f,indent=4)

    with open(os.path.join(args.save_dir,"client_metrics.json"),"w") as f:

        json.dump(client_history,f,indent=4)

    pd.DataFrame(round_metrics).to_csv(
        os.path.join(args.save_dir,"round_metrics.csv"),
        index=False
    )

    pd.DataFrame(client_history).to_csv(
        os.path.join(args.save_dir,"client_metrics.csv"),
        index=False
    )

    plot_curves(round_metrics,args.save_dir)

    print("\nFederated Training Completed")

    print("Best Global Accuracy:",best_acc)



if __name__ == "__main__":
    main()