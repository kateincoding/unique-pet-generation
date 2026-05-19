"""CelebA dataset for facial attribute prediction."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    """CelebA dataset filtered to selected facial attributes.

    Args:
        root: Path to celeba directory containing img_align_celeba/, list_attr_celeba.txt,
              and list_eval_partition.txt
        split: One of 'train', 'val', 'test'
        selected_attributes: List of attribute names to use
        transform: Albumentations transform pipeline
    """

    SPLIT_MAP = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        selected_attributes: list[str] | None = None,
        transform=None,
    ):
        self.root = Path(root)
        self.img_dir = self.root / "img_align_celeba"
        self.transform = transform

        if split not in self.SPLIT_MAP:
            raise ValueError(f"split must be one of {list(self.SPLIT_MAP.keys())}")

        partitions = self._load_partitions()
        all_attrs, attr_names = self._load_attributes()

        split_mask = partitions == self.SPLIT_MAP[split]
        self.filenames = [f for f, m in zip(all_attrs.keys(), split_mask) if m]

        if selected_attributes is not None:
            self.attr_indices = [attr_names.index(a) for a in selected_attributes]
            self.attr_names = selected_attributes
        else:
            self.attr_indices = list(range(len(attr_names)))
            self.attr_names = attr_names

        all_labels = np.array(list(all_attrs.values()))
        self.labels = all_labels[split_mask][:, self.attr_indices].astype(np.float32)

    def _load_partitions(self) -> np.ndarray:
        path = self.root / "list_eval_partition.txt"
        partitions = []
        with open(path) as f:
            first_line = f.readline().strip()
            # Detect CSV (Kaggle) vs space-separated (original)
            if "," in first_line:
                # CSV with header: "image_id,partition"
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        partitions.append(int(parts[1]))
            else:
                # Original format: no header, space-separated
                parts = first_line.split()
                if len(parts) == 2:
                    partitions.append(int(parts[1]))
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        partitions.append(int(parts[1]))
        return np.array(partitions)

    def _load_attributes(self) -> tuple[dict, list[str]]:
        path = self.root / "list_attr_celeba.txt"
        with open(path) as f:
            first_line = f.readline().strip()

            if "," in first_line:
                # CSV format (Kaggle): header is "image_id,attr1,attr2,..."
                attr_names = first_line.split(",")[1:]
                attrs = {}
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 2:
                        continue
                    filename = parts[0]
                    values = [(int(v) + 1) // 2 for v in parts[1:]]
                    attrs[filename] = values
            else:
                # Original format: first line is count, second is attr names
                attr_names = f.readline().strip().split()
                attrs = {}
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    filename = parts[0]
                    values = [(int(v) + 1) // 2 for v in parts[1:]]
                    attrs[filename] = values

        return attrs, attr_names

    def compute_pos_weight(self) -> torch.Tensor:
        """Compute pos_weight for BCEWithLogitsLoss to handle class imbalance."""
        pos_counts = self.labels.sum(axis=0)
        neg_counts = len(self.labels) - pos_counts
        weights = neg_counts / np.clip(pos_counts, 1, None)
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_dir / self.filenames[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        label = self.labels[idx]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, torch.tensor(label, dtype=torch.float32)
