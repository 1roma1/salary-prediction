import os
import math
import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import R2Score

from tqdm import tqdm
from pathlib import Path

from src.base.utils import get_or_create_experiment
from src.base.registries import LossRegistry, ModelRegistry, OptimizerRegistry


class Trainer:
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def _train_epoch(
        self,
        train_dataloader: DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
        device: str,
        grad_accum: int = 1,
    ) -> float:
        model.train()
        train_loss = 0.0
        batch_iterator = tqdm(train_dataloader)
        for batch_idx, (descriptions, features, labels, mask) in enumerate(
            batch_iterator
        ):
            descriptions, features, labels, mask = (
                descriptions.to(device),
                features.to(device),
                labels.to(device),
                mask.to(device),
            )

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(descriptions, features, mask)
                loss = loss_fn(outputs.reshape(-1), labels)

            loss = loss / grad_accum
            loss.backward()

            if ((batch_idx + 1) % grad_accum == 0) or (
                batch_idx + 1 == len(train_dataloader)
            ):
                optimizer.step()
                if lr_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
        return train_loss / len(train_dataloader)

    def _val_epoch(
        self,
        val_dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        device: str,
    ) -> float:
        model.eval()
        val_loss = 0
        r2_score_metric = R2Score()
        with torch.no_grad():
            for descriptions, features, labels, mask in val_dataloader:
                descriptions, features, labels, mask = (
                    descriptions.to(device),
                    features.to(device),
                    labels.to(device),
                    mask.to(device),
                )
                y_pred = model(descriptions, features, mask)
                val_loss += loss_fn(y_pred.reshape(-1), labels).item()
                r2_score_metric.update(y_pred.reshape(-1), labels)
        r2_score = r2_score_metric.compute()
        val_loss /= len(val_dataloader)
        return val_loss, r2_score

    def train(
        self,
        experiment_name: str,
        checkpoints_dir: str,
        train_config: dict,
        data_config: dict,
    ) -> None:
        device = train_config["device"]

        model = ModelRegistry.get(train_config["model"])(
            **train_config["model_params"]
        )
        model = model.to(device)

        optimizer = OptimizerRegistry.get(train_config["optimizer"])(
            params=model.parameters(), **train_config["optimizer_params"]
        )
        lr_scheduler = None
        if train_config.get("lr_scheduler"):
            lr_scheduler = OptimizerRegistry.get(
                train_config.get("lr_scheduler")
            )(optimizer, **train_config.get("lr_scheduler_params"))

        loss_fn = LossRegistry.get(train_config["loss_fn"])()

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=train_config["batch_size"],
            shuffle=True,
            num_workers=train_config["num_workers"],
            pin_memory=train_config["pin_memory"],
            drop_last=True,
        )
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=train_config["batch_size"],
            num_workers=train_config["num_workers"],
            pin_memory=train_config["pin_memory"],
        )

        experiment_id = get_or_create_experiment(
            experiment_name=experiment_name
        )
        mlflow.set_experiment(experiment_id=experiment_id)

        with mlflow.start_run(experiment_id=experiment_id) as run:
            best_loss = math.inf
            for epoch in range(1, train_config["epochs"] + 1):
                train_loss = self._train_epoch(
                    train_dataloader,
                    model,
                    optimizer,
                    lr_scheduler,
                    loss_fn,
                    device,
                    train_config.get("grad_accum", 1),
                )
                val_loss, r2_score = self._val_epoch(
                    val_dataloader, model, loss_fn, device
                )

                print(
                    f"Epoch {epoch}: train loss - {train_loss} | "
                    f"val loss - {val_loss} | r2 score: {r2_score}"
                )

                if val_loss < best_loss:
                    best_loss = val_loss
                    print(f"Save model with validation loss: {best_loss:2.5f}")
                    model.eval()
                    os.makedirs(
                        Path(checkpoints_dir, run.info.run_name), exist_ok=True
                    )
                    torch.save(
                        model.state_dict(),
                        Path(
                            checkpoints_dir,
                            run.info.run_name,
                            f"{train_config["model"]}.pth",
                        ),
                    )

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("r2_score", r2_score, step=epoch)

            mlflow.log_params(
                {
                    "model": train_config.get("model"),
                    "optimizer": train_config.get("optimizer"),
                    "lr_scheduler": train_config.get("lr_scheduler"),
                    "loss_fn": train_config.get("loss_fn"),
                    "device": train_config.get("device"),
                    "grad_accum": train_config.get("grad_accum"),
                    "epochs": train_config.get("epochs"),
                    "batch_size": train_config.get("batch_size"),
                    "num_workers": train_config.get("num_workers"),
                    "pin_memory": train_config.get("pin_memory"),
                    "tokenzier": data_config.get("tokenizer"),
                    "max_length": data_config.get("max_length"),
                }
            )
            mlflow.log_params(train_config.get("model_params", {}))
            mlflow.log_params(train_config.get("optimizer_params", {}))
            mlflow.log_params(train_config.get("lr_scheduler_params", {}))
            mlflow.log_params(train_config.get("loss_fn_params", {}))
