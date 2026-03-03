import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

LABEL_MAP = {
    "real": 0,
    "fake": 1
}

def get_transforms(img_size=224, train=True):

    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])


class ForgeryDataset(Dataset):

    def __init__(self, root_dir, split="train", img_size=224):
        """
        root_dir:
            prepared_data/client1_casia
            prepared_data/client2_siw
            prepared_data/client3_ff
            prepared_data/global_test

        split:
            train / test (ignored for global)
        """

        self.samples = []
        self.transform = get_transforms(img_size, train=(split=="train"))

        # Global dataset case
        if "global_test" in root_dir:
            self._load_global(root_dir)

        else:
            self._load_client(root_dir, split)

        print(f"Loaded {len(self.samples)} images from {root_dir} ({split})")


    def _load_client(self, root_dir, split):

        for label_name in ["real", "fake"]:

            folder = os.path.join(root_dir, split, label_name)

            if not os.path.exists(folder):
                continue

            label = LABEL_MAP[label_name]

            for img in os.listdir(folder):
                path = os.path.join(folder, img)
                self.samples.append((path, label))


    def _load_global(self, root_dir):

        for label_name in ["real", "fake"]:

            folder = os.path.join(root_dir, label_name)

            if not os.path.exists(folder):
                continue

            label = LABEL_MAP[label_name]

            for img in os.listdir(folder):
                path = os.path.join(folder, img)
                self.samples.append((path, label))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # corrupted image safety
            return self.__getitem__((idx+1) % len(self.samples))

        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataloader(root_dir, split="train", batch_size=32, img_size=224, num_workers=2):

    dataset = ForgeryDataset(root_dir, split, img_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split=="train"),
        num_workers=num_workers,
        pin_memory=True
    )

    return loader