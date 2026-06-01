"""Tests for CelebA dataset and transforms.

These tests verify the Dataset class logic without requiring the actual CelebA data.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from pet_gen.data.celeba_dataset import CelebADataset
from pet_gen.data.transforms import get_eval_transform, get_train_transform


@pytest.fixture
def mock_celeba(tmp_path):
    """Create a minimal mock CelebA dataset for testing."""
    img_dir = tmp_path / "img_align_celeba"
    img_dir.mkdir()

    n_images = 20
    attr_names = ["Black_Hair", "Blond_Hair", "Smiling"]
    filenames = []

    for i in range(n_images):
        fname = f"{i:06d}.jpg"
        filenames.append(fname)
        # Create a small dummy JPEG
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (178, 218, 3), dtype=np.uint8))
        img.save(img_dir / fname)

    # Write attribute file
    with open(tmp_path / "list_attr_celeba.txt", "w") as f:
        f.write(f"{n_images}\n")
        f.write(" ".join(attr_names) + "\n")
        for fname in filenames:
            attrs = " ".join([str(np.random.choice([-1, 1])) for _ in attr_names])
            f.write(f"{fname} {attrs}\n")

    # Write partition file: 14 train, 3 val, 3 test
    with open(tmp_path / "list_eval_partition.txt", "w") as f:
        for i, fname in enumerate(filenames):
            if i < 14:
                partition = 0
            elif i < 17:
                partition = 1
            else:
                partition = 2
            f.write(f"{fname} {partition}\n")

    return tmp_path


def test_dataset_splits(mock_celeba):
    train = CelebADataset(mock_celeba, split="train")
    val = CelebADataset(mock_celeba, split="val")
    test = CelebADataset(mock_celeba, split="test")

    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3


def test_dataset_selected_attributes(mock_celeba):
    ds = CelebADataset(mock_celeba, split="train", selected_attributes=["Black_Hair", "Smiling"])
    assert ds.attr_names == ["Black_Hair", "Smiling"]
    _, label = ds[0]
    assert label.shape == (2,)


def test_dataset_output_types(mock_celeba):
    ds = CelebADataset(mock_celeba, split="train")
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert image.shape[0] == 3  # channels first
    assert label.dtype == torch.float32


def test_dataset_labels_binary(mock_celeba):
    ds = CelebADataset(mock_celeba, split="train")
    for i in range(len(ds)):
        _, label = ds[i]
        assert torch.all((label == 0) | (label == 1))


def test_pos_weight(mock_celeba):
    ds = CelebADataset(mock_celeba, split="train")
    pw = ds.compute_pos_weight()
    assert pw.shape == (3,)
    assert torch.all(pw > 0)


def test_transforms_output_shape():
    train_t = get_train_transform(224)
    eval_t = get_eval_transform(224)

    dummy = np.random.randint(0, 255, (178, 218, 3), dtype=np.uint8)

    train_out = train_t(image=dummy)["image"]
    eval_out = eval_t(image=dummy)["image"]

    assert train_out.shape == (3, 224, 224)
    assert eval_out.shape == (3, 224, 224)


def test_invalid_split(mock_celeba):
    with pytest.raises(ValueError, match="split must be"):
        CelebADataset(mock_celeba, split="invalid")
