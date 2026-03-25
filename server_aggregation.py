import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
    parser.add_argument("--global_test_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="server_outputs")
    parser.add_argument("--resnet_type", type=str, default="resnet50")

    return parser.parse_args()


def load_client_models(client_dirs, device, resnet_type):

    models = []
    accuracies = []

    print("\nLoading client models...")

    for path in client_dirs:

        model = build_model(resnet_type)

        weight_path = os.path.join(path, "best_model.pth")

        if not os.path.exists(weight_path):
            raise FileNotFoundError(weight_path)

        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()

        models.append(model)

        metrics_path = os.path.join(path, "metrics.json")

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                acc = json.load(f)["accuracy"]
        else:
            acc = 1.0

        accuracies.append(acc)

    return models, accuracies


def compute_feature_similarity(models, loader, device):

    features = []

    with torch.no_grad():

        for model in models:

            feats = []

            for images, _ in loader:

                images = images.to(device)

                f = model.extract_features(images)

                feats.append(f.mean(dim=0).cpu())

            feats = torch.stack(feats).mean(dim=0)

            features.append(feats)

    n = len(features)

    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            sim_matrix[i][j] = F.cosine_similarity(
                features[i], features[j], dim=0
            ).item()

    return sim_matrix


def compute_entropy(model, loader, device):

    entropies = []

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            probs = torch.softmax(model(images), dim=1)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            entropies.extend(entropy.cpu().numpy())

    return np.mean(entropies)


def domain_aware_fusion(models_list, accuracies, loader, device, resnet_type):

    print("\n🚀 ADVANCED DOMAIN-AWARE FUSION")

    global_model = build_model(resnet_type)
    global_dict = global_model.state_dict()

    client_dicts = [m.state_dict() for m in models_list]
    n = len(models_list)

    sim_matrix = compute_feature_similarity(models_list, loader, device)

    diversity = 1 - sim_matrix.mean(axis=1)
    diversity = diversity / (diversity.sum() + 1e-8)

    accs = np.array(accuracies)
    accs = accs / (accs.sum() + 1e-8)


    entropy_scores = np.array([
        compute_entropy(m, loader, device)
        for m in models_list
    ])

    confidence = 1 / (entropy_scores + 1e-8)
    confidence = confidence / (confidence.sum() + 1e-8)

    combined = 0.5 * accs + 0.3 * diversity + 0.2 * confidence

    weights = np.exp(combined) / np.sum(np.exp(combined))

    print("🔥 Final Aggregation Weights:", weights)

    for key in tqdm(global_dict.keys(), desc="Fusing Layers"):

        try:

            if "resnet" in key or "xception" in key:

                global_dict[key] = sum(
                    weights[i] * client_dicts[i][key]
                    for i in range(n)
                )

            elif "fusion" in key:

                div_w = np.exp(diversity) / np.sum(np.exp(diversity))

                global_dict[key] = sum(
                    div_w[i] * client_dicts[i][key]
                    for i in range(n)
                )

            elif "classifier" in key:

                temp = 1.5
                conf_w = weights ** (1 / temp)
                conf_w = conf_w / conf_w.sum()

                global_dict[key] = sum(
                    conf_w[i] * client_dicts[i][key]
                    for i in range(n)
                )

            else:

                global_dict[key] = sum(
                    weights[i] * client_dicts[i][key]
                    for i in range(n)
                )

        except:
            global_dict[key] = client_dicts[0][key]

    global_model.load_state_dict(global_dict)

    return global_model, weights, sim_matrix


def evaluate(model, loader, device):

    model.eval()

    preds, labels, probs = [], [], []

    with torch.no_grad():

        for images, y in tqdm(loader, desc="Global Evaluation"):

            images = images.to(device)

            outputs = model(images)

            prob = torch.softmax(outputs, dim=1)
            pred = torch.argmax(prob, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.numpy())
            probs.extend(prob[:, 1].cpu().numpy())

    return np.array(labels), np.array(preds), np.array(probs)


def save_results(labels, preds, probs, cm, weights, sim, save_dir):

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1_score": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs)
    }

    with open(os.path.join(save_dir,"global_metrics.json"),"w") as f:
        json.dump(metrics,f,indent=4)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(save_dir,"metrics.csv"), index=False
    )

    np.save(os.path.join(save_dir,"confusion_matrix.npy"), cm)
    np.save(os.path.join(save_dir,"similarity_matrix.npy"), sim)

    with open(os.path.join(save_dir,"aggregation_weights.json"),"w") as f:
        json.dump(weights.tolist(),f,indent=4)

    print("\nGLOBAL METRICS")
    print(json.dumps(metrics, indent=4))


def main():

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)

    models_list, accs = load_client_models(
        args.client_dirs,
        device,
        args.resnet_type
    )

    global_loader = get_dataloader(
        args.global_test_path,
        split="test",
        batch_size=32
    )

    global_model, weights, sim = domain_aware_fusion(
        models_list,
        accs,
        global_loader,
        device,
        args.resnet_type
    )

    torch.save(
        global_model.state_dict(),
        os.path.join(args.save_dir,"global_model.pth")
    )

    labels, preds, probs = evaluate(global_model, global_loader, device)

    cm = confusion_matrix(labels, preds)

    save_results(labels, preds, probs, cm, weights, sim, args.save_dir)

    print("\n✅ Server Fusion Completed")


if __name__ == "__main__":
    main()