import os
import tempfile

import torchvision
from tqdm.auto import tqdm

CLASSES = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)


def main():
    for split in ["train", "test"]:
        out_dir = f"mnist_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.MNIST(
                root=tmp_dir, train=split == "train", download=True
            )

        print("dumping images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            image.save(filename)


if __name__ == "__main__":
    main()