import os

def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])


def count_client(client_path):

    print(f"\nChecking: {client_path}")

    splits = ["train", "test"]

    total_real = 0
    total_fake = 0

    for split in splits:
        real_path = os.path.join(client_path, split, "real")
        fake_path = os.path.join(client_path, split, "fake")

        real_count = count_images(real_path)
        fake_count = count_images(fake_path)

        total_real += real_count
        total_fake += fake_count

        print(f"\n{split.upper()} SET")
        print(f"Real : {real_count}")
        print(f"Fake : {fake_count}")
        print(f"Total: {real_count + fake_count}")

    print("\nOVERALL")
    print(f"Total Real : {total_real}")
    print(f"Total Fake : {total_fake}")
    print(f"Grand Total: {total_real + total_fake}")


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":

    clients = [
        "balanced_data/client1_casia",
        "balanced_data/client2_siw",
        "balanced_data/client3_ff",
        "balanced_data/global_test"
    ]

    for client in clients:

        if "global_test" in client:
            print(f"\nChecking Global Test Dataset")

            real_count = count_images(os.path.join(client, "real"))
            fake_count = count_images(os.path.join(client, "fake"))

            print(f"     Real : {real_count}")
            print(f"     Fake : {fake_count}")
            print(f"     Total: {real_count + fake_count}")

        else:
            count_client(client)

    print("\nDataset counting completed.")