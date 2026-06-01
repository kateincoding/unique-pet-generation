"""Training loop with backbone freezing, mixed precision, and early stopping."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pet_gen.training.metrics import compute_metrics


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion,
        optimizer,
        scheduler=None,
        device: torch.device | None = None,
        checkpoint_dir: str = "./checkpoints",
        use_amp: bool = True,
        gradient_clip_norm: float = 1.0,
        freeze_backbone_epochs: int = 5,
        attr_names: list[str] | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_clip_norm = gradient_clip_norm
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.attr_names = attr_names

        # AMP only on CUDA (MPS doesn't fully support GradScaler)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.model.to(self.device)
        self.best_map = 0.0
        self.writer = SummaryWriter()

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    output = self.model(images)
                    loss = self.criterion(output["logits"], labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(images)
                loss = self.criterion(output["logits"], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        all_logits = []
        all_targets = []
        total_loss = 0.0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.model(images)
            loss = self.criterion(output["logits"], labels)
            total_loss += loss.item()

            all_logits.append(output["logits"].cpu().numpy())
            all_targets.append(labels.cpu().numpy())

        logits = np.concatenate(all_logits)
        targets = np.concatenate(all_targets)
        metrics = compute_metrics(logits, targets, self.attr_names)
        metrics["val_loss"] = total_loss / len(self.val_loader)
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(state, self.checkpoint_dir / "latest.pt")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pt")

    def fit(self, epochs: int, early_stopping_patience: int = 5):
        patience_counter = 0

        # Freeze backbone initially
        self.model.freeze_backbone()
        print(f"Backbone frozen for first {self.freeze_backbone_epochs} epochs")

        for epoch in range(1, epochs + 1):
            # Unfreeze backbone after warmup
            if epoch == self.freeze_backbone_epochs + 1:
                self.model.unfreeze_backbone()
                # Rebuild optimizer with param groups for differential LR
                lr = self.optimizer.param_groups[0]["lr"]
                self.optimizer = torch.optim.AdamW(
                    self.model.get_param_groups(lr, backbone_lr_scale=0.1),
                    weight_decay=self.optimizer.param_groups[0].get("weight_decay", 1e-4),
                )
                print("Backbone unfrozen with 0.1x learning rate")

            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            is_best = val_metrics["mAP"] > self.best_map
            if is_best:
                self.best_map = val_metrics["mAP"]
                patience_counter = 0
            else:
                patience_counter += 1

            self.save_checkpoint(epoch, val_metrics, is_best)

            # Log to tensorboard
            self.writer.add_scalar("train/loss", train_loss, epoch)
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"val/{key}", value, epoch)

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"mAP={val_metrics['mAP']:.4f} "
                f"acc={val_metrics['mean_accuracy']:.4f}"
                f"{' *BEST*' if is_best else ''}"
            )

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.writer.close()
        print(f"Training complete. Best mAP: {self.best_map:.4f}")
