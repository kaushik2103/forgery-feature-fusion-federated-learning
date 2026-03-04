import os
import random
import shutil

random.seed(42)

SOURCE_ROOT = "prepared_data"
TARGET_ROOT = "balanced_data"

# Target sizes (CASIA size)
TARGET_TRAIN = 1655
TARGET_TEST = 1000

CLIENTS = [
    "client1_casia",
    "client2_siw",
    "client3_ff"
]



def list_images(folder):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg",".png",".jpeg"))
    ]


def balance_split(client, split, target_total):

    src_client = os.path.join(SOURCE_ROOT, client)
    dst_client = os.path.join(TARGET_ROOT, client)

    os.makedirs(dst_client, exist_ok=True)

    real_src = os.path.join(src_client, split, "real")
    fake_src = os.path.join(src_client, split, "fake")

    real_imgs = list_images(real_src)
    fake_imgs = list_images(fake_src)

    # Compute proportional sampling
    total = len(real_imgs) + len(fake_imgs)

    if total <= target_total:
        # If smaller than target → copy all
        sampled_real = real_imgs
        sampled_fake = fake_imgs
    else:
        real_ratio = len(real_imgs) / total
        fake_ratio = len(fake_imgs) / total

        target_real = int(target_total * real_ratio)
        target_fake = target_total - target_real

        sampled_real = random.sample(real_imgs, min(target_real, len(real_imgs)))
        sampled_fake = random.sample(fake_imgs, min(target_fake, len(fake_imgs)))

    # Create destination folders
    for label in ["real","fake"]:
        os.makedirs(os.path.join(dst_client, split, label), exist_ok=True)

    # Copy sampled images
    for img in sampled_real:
        shutil.copy(
            os.path.join(real_src,img),
            os.path.join(dst_client,split,"real",img)
        )

    for img in sampled_fake:
        shutil.copy(
            os.path.join(fake_src,img),
            os.path.join(dst_client,split,"fake",img)
        )

    print(f"\n{client} - {split}")
    print(f"   Real : {len(sampled_real)}")
    print(f"   Fake : {len(sampled_fake)}")
    print(f"   Total: {len(sampled_real)+len(sampled_fake)}")


if __name__ == "__main__":

    os.makedirs(TARGET_ROOT, exist_ok=True)

    for client in CLIENTS:

        print(f"\nBalancing {client} ...")

        balance_split(client, "train", TARGET_TRAIN)
        balance_split(client, "test", TARGET_TEST)

    print("\nDataset balancing completed.")