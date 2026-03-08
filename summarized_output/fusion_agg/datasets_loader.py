import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def get_transforms(img_size=224, train=True):
    """
    224 works best for hybrid ResNet + Xception fusion
    Faster training and stable accuracy
    """

    if train:

        return transforms.Compose([

            transforms.Resize((img_size, img_size)),

            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    else:

        return transforms.Compose([

            transforms.Resize((img_size, img_size)),

            transforms.ToTensor(),

            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])


class ForgeryDataset(Dataset):

    def __init__(self, root_dir, split="train", img_size=224):

        self.samples = []
        self.root_dir = root_dir
        self.split = split

        self.transform = get_transforms(
            img_size,
            train=(split == "train")
        )


        split_path = os.path.join(root_dir, split)

        if os.path.exists(split_path):
            dataset_path = split_path
        else:
            # global dataset case
            dataset_path = root_dir

        self._load_dataset(dataset_path)

        print(f"\nLoaded {len(self.samples)} samples from {dataset_path}")



    def _load_dataset(self, dataset_path):

        real_count = 0
        fake_count = 0

        for class_name in os.listdir(dataset_path):

            folder = os.path.join(dataset_path, class_name)

            if not os.path.isdir(folder):
                continue

            if class_name.lower() == "real":
                label = 0
            else:
                label = 1

            for img in os.listdir(folder):

                if not img.lower().endswith(VALID_EXTENSIONS):
                    continue

                img_path = os.path.join(folder, img)

                self.samples.append((img_path, label))

                if label == 0:
                    real_count += 1
                else:
                    fake_count += 1

        # Shuffle samples
        random.shuffle(self.samples)

        print("\nDataset Statistics")
        print("------------------")
        print("Real :", real_count)
        print("Fake :", fake_count)
        print("Total:", real_count + fake_count)



    def __len__(self):
        return len(self.samples)



    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")

        except:
            # Handle corrupted images
            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx)

        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)



def get_dataloader(
        root_dir,
        split="train",
        batch_size=32,
        img_size=224,
        num_workers=2
):

    dataset = ForgeryDataset(
        root_dir=root_dir,
        split=split,
        img_size=img_size
    )

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=(split == "train"),

        num_workers=num_workers,

        pin_memory=True,

        drop_last=(split == "train")

    )

    return loader