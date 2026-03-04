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

