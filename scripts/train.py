"""Main training script for facial feature extraction baseline."""

import sys
from pathlib import Path

from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from pet_gen.data.celeba_dataset import CelebADataset
from pet_gen.data.transforms import get_eval_transform, get_train_transform
from pet_gen.models.feature_model import FacialFeatureModel
from pet_gen.training.losses import create_bce_loss
from pet_gen.training.trainer import Trainer, get_device


def main():
    # Load config: base + CLI overrides
    base_cfg = OmegaConf.load("configs/base.yaml")
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    print(f"Device: {get_device()}")

    # Data
    train_transform = get_train_transform(cfg.data.image_size)
    eval_transform = get_eval_transform(cfg.data.image_size)

    train_dataset = CelebADataset(
        root=cfg.data.root,
        split="train",
        selected_attributes=list(cfg.data.selected_attributes),
        transform=train_transform,
    )
    val_dataset = CelebADataset(
        root=cfg.data.root,
        split="val",
        selected_attributes=list(cfg.data.selected_attributes),
        transform=eval_transform,
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # Model
    num_attrs = len(cfg.data.selected_attributes)
    model = FacialFeatureModel(
        backbone_name=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        embedding_dim=cfg.model.embedding_dim,
        num_attributes=num_attrs,
        dropout=cfg.model.dropout,
    )

    # Loss with class imbalance weighting
    pos_weight = train_dataset.compute_pos_weight().to(get_device())
    criterion = create_bce_loss(pos_weight)

    # Optimizer (initially only head params since backbone is frozen)
    head_params = list(model.embedding_head.parameters()) + list(
        model.attribute_head.parameters()
    )
    optimizer = torch.optim.AdamW(
        head_params,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=cfg.training.checkpoint_dir,
        use_amp=cfg.training.use_amp,
        gradient_clip_norm=cfg.training.gradient_clip_norm,
        freeze_backbone_epochs=cfg.model.freeze_backbone_epochs,
        attr_names=list(cfg.data.selected_attributes),
    )

    trainer.fit(
        epochs=cfg.training.epochs,
        early_stopping_patience=cfg.training.early_stopping_patience,
    )


if __name__ == "__main__":
    main()
