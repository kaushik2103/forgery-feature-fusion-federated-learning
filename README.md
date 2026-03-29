# Forgery Feature Fusion Federated Learning
The system trains a single universal forgery detector collaboratively using 4 decentralized clients without sharing raw images. The system collaboratively learns a universal facial forgery detector by combining local hybrid feature learning with server-side fusion aggregation across heterogeneous domains.


## How to run the clients(example):
`python client_train.py \
    --client_path prepared_data/client1_casia \
    --client_name casia \
    --epochs 10`

`python client_train.py \
    --client_path prepared_data/client2_siw \
    --client_name siw`

`python client_train.py \
    --client_path prepared_data/client3_ff \
    --client_name ff`

## How to run the server(example):
`python server_aggregation.py \
    --client_dirs \
    client_outputs/casia \
    client_outputs/siw \
    client_outputs/ff \
    --client_weights 1 1 1`

## How to run federated_main.py(example):
`python federated_main.py \
    --rounds 5 \
    --local_epochs 2 \
    --client_paths \
    prepared_data/client1_casia \
    prepared_data/client2_siw \
    prepared_data/client3_ff`

`python federated_main.py --rounds 5 --local_epochs 2 --batch_size 32 --client_paths prepared_data/client1_casia prepared_data/client2_siw prepared_data/client3_ff`

This command I have used for training and testing:
`python federated_main.py --rounds 20  --local_epochs 5 --batch_size 12 --lr 0.001 --client_paths ff_clients\client1 ff_clients\client2 --global_test_path ff_clients\global_test`

New command to be use: `python federated_main.py --rounds 20 --local_epochs 5 --batch_size 12 --lr 0.0003 --client_paths ff_clients/client1 ff_clients/client2 --global_test_path ff_clients/global_test`

## Before moving towards the project overview and results, I would like you say that: the results are calculated for both 10 and 20 rounds for each aggregation methods, However due to size of the models & results I didn't added the results for 20 rounds.

## Project Overview

This project implements a **Federated Learning (FL) framework for Deepfake Detection** using a **hybrid CNN–Xception architecture** trained on the **FaceForensics++ dataset**.

The goal of this work is to evaluate how different **federated aggregation strategies** affect the performance of a distributed deepfake detection system where data remains **locally stored on clients**.

Two aggregation strategies were implemented and evaluated:

1. **Fusion Aggregation (Model Fusion Strategy)**
2. **Federated Averaging (FedAvg Strategy)**

Both methods follow the **standard federated learning workflow** where clients train local models and the server aggregates them to create a global model.

---

# Dataset

The experiments use a **reduced version of the FaceForensics++ dataset** to make federated training computationally feasible.

The dataset includes the following manipulation types:

* Deepfakes
* FaceSwap
* Face2Face
* NeuralTextures
* FaceShifter
* DeepFakeDetection

Data is split into:

```
Client 1
   train/
   test/

Client 2
   train/
   test/

Global Test Dataset
   real/
   fake/
```

Each client receives **different manipulation domains**, simulating **non-IID federated data distribution**.

---

# System Architecture

The project uses a **Hybrid Deepfake Detection Model** combining two feature extractors:

* **ResNet backbone**
* **Xception backbone**

These extractors generate complementary facial artifact features which are then merged using a **feature fusion layer** followed by a **binary classifier**.

High-level architecture:

---

# Federated Learning Pipeline

The training process follows the standard **federated learning cycle**:

1. **Server initializes a global model**
2. **Global model is sent to clients**
3. **Clients train locally on private datasets**
4. **Clients send model weights back to server**
5. **Server aggregates models**
6. **New global model is redistributed**
7. **Process repeats for multiple rounds**

---

# Phase 1: Fusion Aggregation

In the first phase, **Model Fusion Aggregation** was implemented.

Instead of traditional averaging, **layer-wise fusion weights** were applied during aggregation.

Example strategy:

* Backbone layers averaged
* Feature fusion layers weighted
* Classifier layers weighted separately

This approach allows the server to **combine domain-specific knowledge from different clients more effectively**.

### Observed Behavior

* Accuracy fluctuates between rounds
* Performance **gradually improves over rounds**
* Graph shows **oscillating but increasing accuracy**

### Performance

```
Fusion Aggregation Accuracy ≈ 90%
```

Graph behavior:


This oscillating behavior occurs because **each client specializes in different manipulation domains**, and fusion allows the model to gradually combine those features.

---

# Phase 2: FedAvg Aggregation

The second phase implemented **Federated Averaging (FedAvg)**.

FedAvg performs **weighted averaging of model parameters** based on client dataset sizes.

Aggregation formula:

```
W_global = Σ (nk / N) * Wk
```

Where:

```
Wk = client model weights
nk = number of samples at client k
N = total samples
```

### Observed Behavior

* Accuracy stabilizes early
* Global model converges quickly
* Performance becomes **steady but lower than fusion**

### Performance

```
FedAvg Accuracy ≈ 84%
```

Graph behavior:

FedAvg stabilizes early because it **smooths the parameters of client models**, which can suppress domain-specific knowledge.

---

# Comparison of Aggregation Strategies

| Aggregation Method | Accuracy | Behavior                   |
| ------------------ | -------- | -------------------------- |
| Fusion Aggregation | ~90%     | Oscillating but increasing |
| FedAvg             | ~84%     | Stable but lower           |

### Key Insight

Fusion aggregation performs better because:

* Clients contain **different manipulation domains**
* Fusion preserves **domain-specific representations**
* FedAvg averages these representations and **reduces specialization**

---

# Experimental Observations

### Fusion Aggregation

Advantages:

* Better global accuracy
* Captures manipulation artifacts from different domains
* Gradual performance improvement across rounds

Disadvantages:

* Slightly unstable training curve
* Requires more careful layer weighting

---

### FedAvg

Advantages:

* Simpler and more stable training
* Standard federated learning baseline
* Lower communication complexity

Disadvantages:

* Lower accuracy in non-IID settings
* Loss of domain-specific features

---

# Result Summary

```
Fusion Aggregation Accuracy  ≈ 90%
FedAvg Aggregation Accuracy  ≈ 84%
```

The experiments show that **aggregation strategy significantly impacts federated deepfake detection performance**, especially under **non-IID data distributions**.

---

# Project Outputs

All experiment results are stored in the **`summarized` folder**, including:

```
summarized/

fusion_results/
    accuracy_curve.png
    f1_curve.png
    confusion_matrix.npy
    predictions.csv

fedavg_results/
    accuracy_curve.png
    f1_curve.png
    confusion_matrix.npy
    predictions.csv
```

These files contain detailed evaluation metrics and visualization of training behavior.

---

# Conclusion

This project demonstrates that **aggregation strategy plays a crucial role in federated deepfake detection systems**.

Fusion aggregation significantly outperforms FedAvg in this scenario due to the **non-IID distribution of manipulation types across clients**.

Key findings:

* Federated learning can effectively train deepfake detection models without centralized data sharing.
* Aggregation strategy strongly influences model performance.
* Domain-aware fusion methods can outperform traditional FedAvg under heterogeneous data conditions.

Future work could explore:

* Adaptive aggregation strategies
* Client importance weighting
* Domain-aware federated optimization methods.
