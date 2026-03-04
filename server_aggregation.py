import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

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

    parser.add_argument("--client_dirs", nargs="+", required=True,
                        help="Paths to client output folders")
    parser.add_argument("--client_weights", nargs="+", type=float,
                        help="Client importance weights (same order)")
    parser.add_argument("--global_test_path", type=str,
                        default="prepared_data/global_test")
    parser.add_argument("--save_dir", type=str,
                        default="server_outputs")
    parser.add_argument("--resnet_type", type=str,
                        default="resnet50")

    return parser.parse_args()



def load_client_models(client_dirs, device, resnet_type):

    models_list = []

    for path in client_dirs:

        model = build_model(resnet_type)
        weight_path = os.path.join(path, "best_model.pth")

        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()

        models_list.append(model)

    return models_list



def fusion_aggregation(models_list, weights):

    global_model = build_model()
    global_dict = global_model.state_dict()

    client_dicts = [m.state_dict() for m in models_list]

    total_weight = sum(weights)
    weights = [w/total_weight for w in weights]

    for key in global_dict.keys():


        if "resnet" in key or "xception" in key:

            global_dict[key] = sum(
                weights[i] * client_dicts[i][key]
                for i in range(len(models_list))
            )


        elif "fusion" in key:

            global_dict[key] = sum(
                weights[i] * client_dicts[i][key]
                for i in range(len(models_list))
            )


        elif "classifier" in key:

            # Give slightly higher importance to last client (deepfake)
            deepfake_boost = 1.2

            adjusted_weights = weights.copy()
            adjusted_weights[-1] *= deepfake_boost

            norm = sum(adjusted_weights)
            adjusted_weights = [w/norm for w in adjusted_weights]

            global_dict[key] = sum(
                adjusted_weights[i] * client_dicts[i][key]
                for i in range(len(models_list))
            )

        else:
            global_dict[key] = sum(
                weights[i] * client_dicts[i][key]
                for i in range(len(models_list))
            )

    global_model.load_state_dict(global_dict)

    return global_model


def evaluate(model, loader, device):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Global Evaluation"):

            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:,1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def save_results(labels, preds, probs, save_path):

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

    with open(os.path.join(save_path, "global_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(save_path, "global_classification_report.txt"), "w") as f:
        f.write(report)

    print("\n===== GLOBAL EVALUATION =====")
    print(json.dumps(metrics, indent=4))
    print("\nClassification Report:\n", report)


def main():

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)

    # Default equal weights if not provided
    if args.client_weights is None:
        weights = [1.0] * len(args.client_dirs)
    else:
        weights = args.client_weights

    models_list = load_client_models(
        args.client_dirs,
        device,
        args.resnet_type
    )

    global_model = fusion_aggregation(models_list, weights)
    global_model.to(device)

    # Save global model
    torch.save(global_model.state_dict(),
               os.path.join(args.save_dir, "global_model.pth"))

    global_loader = get_dataloader(
        args.global_test_path,
        split="test",
        batch_size=32
    )

    labels, preds, probs = evaluate(global_model, global_loader, device)

    save_results(labels, preds, probs, args.save_dir)

    print("\nServer Aggregation Completed")


if __name__ == "__main__":
    main()