import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

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



def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--client_dirs", nargs="+", required=True)

    parser.add_argument(
        "--global_test_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="server_outputs"
    )

    parser.add_argument(
        "--resnet_type",
        type=str,
        default="resnet50"
    )

    return parser.parse_args()


def load_client_models(client_dirs, device, resnet_type):

    models = []
    sample_counts = []

    print("\nLoading client models...")

    for path in client_dirs:

        model = build_model(resnet_type)

        weight_path = os.path.join(path, "best_model.pth")

        if not os.path.exists(weight_path):
            raise FileNotFoundError(weight_path)

        model.load_state_dict(
            torch.load(weight_path, map_location=device)
        )

        model.to(device)
        model.eval()

        models.append(model)

        # load dataset size
        info_file = os.path.join(path, "client_info.json")

        if os.path.exists(info_file):

            with open(info_file) as f:
                info = json.load(f)

            sample_counts.append(info["dataset_size"])

        else:

            print("WARNING: client_info.json missing, using equal weight")
            sample_counts.append(1)

    return models, sample_counts



def fedavg(models_list, sample_counts, resnet_type):

    print("\nPerforming FedAvg Aggregation")

    global_model = build_model(resnet_type)

    global_dict = global_model.state_dict()

    client_dicts = [m.state_dict() for m in models_list]

    total_samples = sum(sample_counts)

    # compute normalized weights
    weights = [c / total_samples for c in sample_counts]

    print("Client sample counts:", sample_counts)
    print("Aggregation weights:", weights)

    for key in tqdm(global_dict.keys(), desc="Aggregating Parameters"):

        try:

            global_dict[key] = sum(
                weights[i] * client_dicts[i][key]
                for i in range(len(models_list))
            )

        except:

            # fallback safety
            global_dict[key] = client_dicts[0][key]

    global_model.load_state_dict(global_dict)

    return global_model



def evaluate(model, loader, device):

    model.eval()

    preds = []
    labels = []
    probs = []

    with torch.no_grad():

        for images, y in tqdm(loader, desc="Global Evaluation"):

            images = images.to(device)

            outputs = model(images)

            prob = torch.softmax(outputs, dim=1)

            pred = torch.argmax(prob, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.numpy())
            probs.extend(prob[:,1].cpu().numpy())

    return np.array(labels), np.array(preds), np.array(probs)


def plot_confusion_matrix(cm, save_dir):

    plt.figure(figsize=(5,5))

    plt.imshow(cm, cmap="Blues")

    plt.title("Confusion Matrix")

    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,cm[i,j],ha="center",va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir,"confusion_matrix.png"))

    plt.close()


def plot_roc(labels, probs, save_dir):

    fpr, tpr, _ = roc_curve(labels, probs)

    plt.figure()

    plt.plot(fpr, tpr)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.savefig(os.path.join(save_dir,"roc_curve.png"))

    plt.close()


def save_results(labels, preds, probs, cm, save_dir):

    acc = accuracy_score(labels, preds)

    precision = precision_score(labels, preds, zero_division=0)

    recall = recall_score(labels, preds, zero_division=0)

    f1 = f1_score(labels, preds, zero_division=0)

    auc = roc_auc_score(labels, probs)

    report = classification_report(labels, preds)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    }

    with open(os.path.join(save_dir,"global_metrics.json"),"w") as f:
        json.dump(metrics,f,indent=4)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(save_dir,"evaluation_metrics.csv"),
        index=False
    )

    with open(os.path.join(save_dir,"classification_report.txt"),"w") as f:
        f.write(report)

    np.save(os.path.join(save_dir,"confusion_matrix.npy"),cm)

    plot_confusion_matrix(cm,save_dir)

    plot_roc(labels,probs,save_dir)

    pd.DataFrame({
        "label":labels,
        "prediction":preds,
        "probability":probs
    }).to_csv(
        os.path.join(save_dir,"predictions.csv"),
        index=False
    )

    print("\nGLOBAL METRICS")

    print(json.dumps(metrics,indent=4))


def main():

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)

    models_list, sample_counts = load_client_models(
        args.client_dirs,
        device,
        args.resnet_type
    )

    global_model = fedavg(
        models_list,
        sample_counts,
        args.resnet_type
    )

    global_model.to(device)

    torch.save(
        global_model.state_dict(),
        os.path.join(args.save_dir,"global_model.pth")
    )

    global_loader = get_dataloader(
        args.global_test_path,
        split="test",
        batch_size=32
    )

    labels, preds, probs = evaluate(global_model, global_loader, device)

    cm = confusion_matrix(labels, preds)

    save_results(
        labels,
        preds,
        probs,
        cm,
        args.save_dir
    )

    print("\nFedAvg Aggregation Completed")



if __name__ == "__main__":
    main()