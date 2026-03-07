import os
import shutil
import random
from pathlib import Path

random.seed(42)

SOURCE = "datasets/dataset_processed_split"
TARGET = "ff_clients"
GLOBAL_TEST = "ff_clients/global_test"

# ============================================================
# Balanced Dataset Sizes
# ============================================================

TRAIN_REAL = 2000
TEST_REAL = 500

TRAIN_FAKE_TOTAL = 2000
TEST_FAKE_TOTAL = 500

GLOBAL_REAL = 2000
GLOBAL_FAKE_TOTAL = 2000


# ============================================================
# Fake Types
# ============================================================

client1_fake = [
    "Deepfakes",
    "FaceSwap",
    "Face2Face"
]

client2_fake = [
    "NeuralTextures",
    "FaceShifter",
    "DeepFakeDetection"
]

all_fake_types = client1_fake + client2_fake


# ============================================================
# Copy Images
# ============================================================

def copy_images(src_folder, dst_folder, max_images):

    os.makedirs(dst_folder, exist_ok=True)

    images = os.listdir(src_folder)

    random.shuffle(images)

    images = images[:min(max_images, len(images))]

    copied = 0

    for img in images:

        src = os.path.join(src_folder, img)

        new_name = f"{Path(src_folder).name}_{img}"

        dst = os.path.join(dst_folder, new_name)

        shutil.copy(src, dst)

        copied += 1

    return copied


# ============================================================
# Build Client
# ============================================================

def build_client(client_name, fake_types):

    print("\nPreparing", client_name)

    fake_train_each = TRAIN_FAKE_TOTAL // len(fake_types)
    fake_test_each = TEST_FAKE_TOTAL // len(fake_types)

    stats = {
        "train_real": 0,
        "train_fake": 0,
        "test_real": 0,
        "test_fake": 0
    }

    for split in ["train", "test"]:
        for label in ["real", "fake"]:
            os.makedirs(
                f"{TARGET}/{client_name}/{split}/{label}",
                exist_ok=True
            )

    # ---------------- Real ----------------

    stats["train_real"] = copy_images(
        f"{SOURCE}/train/Real",
        f"{TARGET}/{client_name}/train/real",
        TRAIN_REAL
    )

    stats["test_real"] = copy_images(
        f"{SOURCE}/test/Real",
        f"{TARGET}/{client_name}/test/real",
        TEST_REAL
    )

    # ---------------- Fake ----------------

    for fake_type in fake_types:

        stats["train_fake"] += copy_images(
            f"{SOURCE}/train/{fake_type}",
            f"{TARGET}/{client_name}/train/fake",
            fake_train_each
        )

        stats["test_fake"] += copy_images(
            f"{SOURCE}/test/{fake_type}",
            f"{TARGET}/{client_name}/test/fake",
            fake_test_each
        )

    print("Train Real :", stats["train_real"])
    print("Train Fake :", stats["train_fake"])
    print("Test Real  :", stats["test_real"])
    print("Test Fake  :", stats["test_fake"])


# ============================================================
# Build Global Test
# ============================================================

def build_global_test():

    print("\nPreparing GLOBAL TEST DATASET")

    os.makedirs(GLOBAL_TEST, exist_ok=True)

    fake_each = GLOBAL_FAKE_TOTAL // len(all_fake_types)

    real_dst = f"{GLOBAL_TEST}/Real"
    os.makedirs(real_dst, exist_ok=True)

    real_count = copy_images(
        f"{SOURCE}/test/Real",
        real_dst,
        GLOBAL_REAL
    )

    fake_count = 0

    for fake_type in all_fake_types:

        dst = f"{GLOBAL_TEST}/{fake_type}"

        fake_count += copy_images(
            f"{SOURCE}/test/{fake_type}",
            dst,
            fake_each
        )

    print("Global Real :", real_count)
    print("Global Fake :", fake_count)
    print("Global Total:", real_count + fake_count)


# ============================================================
# Run Script
# ============================================================

if __name__ == "__main__":

    build_client("client1", client1_fake)

    build_client("client2", client2_fake)

    build_global_test()

    print("\nBalanced Clients and Global Test Dataset Prepared")