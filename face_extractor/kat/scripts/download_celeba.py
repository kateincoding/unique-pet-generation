"""Download CelebA dataset from Kaggle using kagglehub."""

import shutil
from pathlib import Path
import kagglehub

DATA_ROOT = Path("./data/celeba")


def download_celeba(data_root: Path | None = None):
    root = data_root or DATA_ROOT

    img_dir = root / "img_align_celeba"
    attr_file = root / "list_attr_celeba.txt"
    partition_file = root / "list_eval_partition.txt"

    if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) > 200000:
        print("CelebA already downloaded. Skipping.")
        return root

    print("Downloading CelebA from Kaggle...")
    kaggle_path = Path(kagglehub.dataset_download("jessicali9530/celeba-dataset"))
    print(f"Downloaded to: {kaggle_path}")

    root.mkdir(parents=True, exist_ok=True)

    # Kaggle CelebA structure: <path>/img_align_celeba/img_align_celeba/*.jpg
    # and <path>/list_attr_celeba.csv, <path>/list_eval_partition.csv
    # Need to locate the actual files
    kaggle_img_dir = kaggle_path / "img_align_celeba" / "img_align_celeba"
    if not kaggle_img_dir.exists():
        kaggle_img_dir = kaggle_path / "img_align_celeba"

    if kaggle_img_dir.exists() and not img_dir.exists():
        print(f"Symlinking images: {kaggle_img_dir} -> {img_dir}")
        img_dir.symlink_to(kaggle_img_dir)

    # Copy annotation files — Kaggle version may be .csv or .txt
    _copy_annotations(kaggle_path, root)

    n_images = len(list(img_dir.glob("*.jpg")))
    print(f"CelebA ready: {n_images} images")
    return root


def _copy_annotations(kaggle_path: Path, dest: Path):
    """Find and copy attribute + partition annotation files."""
    # Attribute file
    for candidate in [
        kaggle_path / "list_attr_celeba.txt",
        kaggle_path / "list_attr_celeba.csv",
    ]:
        if candidate.exists():
            shutil.copy2(candidate, dest / "list_attr_celeba.txt")
            print(f"  Copied {candidate.name}")
            break
    else:
        # Search recursively
        found = list(kaggle_path.rglob("list_attr_celeba.*"))
        if found:
            shutil.copy2(found[0], dest / "list_attr_celeba.txt")
            print(f"  Copied {found[0]}")

    # Partition file
    for candidate in [
        kaggle_path / "list_eval_partition.txt",
        kaggle_path / "list_eval_partition.csv",
    ]:
        if candidate.exists():
            shutil.copy2(candidate, dest / "list_eval_partition.txt")
            print(f"  Copied {candidate.name}")
            break
    else:
        found = list(kaggle_path.rglob("list_eval_partition.*"))
        if found:
            shutil.copy2(found[0], dest / "list_eval_partition.txt")
            print(f"  Copied {found[0]}")


if __name__ == "__main__":
    download_celeba()
