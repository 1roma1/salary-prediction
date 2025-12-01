import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from pathlib import Path

from src.models.bert import BertModel


def get_or_create_experiment(experiment_name: str) -> str:
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


class Trainer:
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def _train_epoch(
        self,
        train_dataloader: DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: str,
    ) -> float:
        model.train()
        train_loss = 0.0
        batch_iterator = tqdm(train_dataloader)
        for inputs, labels, masks in batch_iterator:
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(inputs, masks)
                loss = loss_fn(outputs.view(-1), labels.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
        return train_loss / len(train_dataloader)

    def _val_epoch(
        self, val_dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: str
    ) -> tuple[float, float]:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, masks in val_dataloader:
                inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
                y_pred = model(inputs, masks)
                val_loss += loss_fn(y_pred.view(-1), labels.view(-1)).item()
        val_loss /= len(val_dataloader)
        return val_loss

    def train(
        self,
        experiment_name: str,
        checkpoints_dir: str,
        train_config: dict,
    ) -> None:
        device = train_config["device"]

        # model_class = NAME_TO_MODEL[train_config["name"]]
        # model = model_class(**train_config["model_params"])
        model = BertModel(**train_config["model_params"])
        model = model.to(device)

        optimizer = optim.AdamW(params=model.parameters(), **train_config["optimizer_params"])
        loss_fn = nn.MSELoss()
        # loss_fn = AAMSoftmax(margin=0.2)

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=train_config["batch_size"],
            shuffle=True,
            num_workers=train_config["num_workers"],
            pin_memory=train_config["pin_memory"],
        )
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=train_config["batch_size"],
            num_workers=train_config["num_workers"],
            pin_memory=train_config["pin_memory"],
        )

        experiment_id = get_or_create_experiment(experiment_name=experiment_name)
        mlflow.set_experiment(experiment_id=experiment_id)

        with mlflow.start_run(experiment_id=experiment_id) as run:
            best_loss = 1000000
            for epoch in range(1, train_config["epochs"] + 1):
                train_loss = self._train_epoch(train_dataloader, model, optimizer, loss_fn, device)
                val_loss = self._val_epoch(val_dataloader, model, loss_fn, device)

                print(f"Epoch {epoch}: train loss - {train_loss} | val loss - {val_loss}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    print(f"Save model with validation loss: {best_loss:2.5f}")
                    model.eval()
                    torch.save(
                        model.state_dict(),
                        Path(checkpoints_dir) / f"{train_config['name']}-{run.info.run_name}.pth",
                    )

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

            mlflow.log_params(
                {
                    "model": train_config["name"],
                    "device": train_config["device"],
                    "epochs": train_config["epochs"],
                    "batch_size": train_config["batch_size"],
                }
            )
            mlflow.log_params(train_config["model_params"])
            mlflow.log_params(train_config["optimizer_params"])
