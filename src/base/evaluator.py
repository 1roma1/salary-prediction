import tempfile
import onnx
import mlflow
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError

from src.base.utils import get_or_create_experiment
from src.base.registries import LossRegistry, ModelRegistry


def to_onnx(model: nn.Module, sample: tuple):
    model.eval()
    model = model.to("cpu")
    text = sample[0].unsqueeze(0)
    features = sample[1].unsqueeze(0)
    mask = sample[2].unsqueeze(0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir)
        torch.onnx.export(
            model,
            (text, features, mask),
            path / "model.onnx",
            dynamo=False,
            external_data=False,
        )
        return onnx.load(path / "model.onnx")


class Evaluator:
    def __init__(
        self,
        test_dataset: Dataset,
    ) -> None:
        self.test_dataset = test_dataset

    def _evaluate(
        self,
        test_dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        device: str,
    ) -> tuple[float, float]:
        model.eval()
        test_loss = 0
        r2 = R2Score().to(device)
        rmse = MeanSquaredError(squared=False).to(device)
        mae = MeanAbsoluteError().to(device)
        with torch.no_grad():
            for descriptions, features, labels, mask in test_dataloader:
                descriptions, features, labels, mask = (
                    descriptions.to(device),
                    features.to(device),
                    labels.to(device),
                    mask.to(device),
                )
                y_pred = model(descriptions, features, mask)
                test_loss += loss_fn(y_pred.reshape(-1), labels).item()
                r2.update(y_pred.reshape(-1), labels)
                rmse.update(
                    torch.expm1(y_pred.reshape(-1)), torch.expm1(labels)
                )
                mae.update(
                    torch.expm1(y_pred.reshape(-1)), torch.expm1(labels)
                )
        r2_score = r2.compute()
        rmse_score = rmse.compute()
        mae_score = mae.compute()
        test_loss /= len(test_dataloader)
        return test_loss, r2_score.item(), rmse_score.item(), mae_score.item()

    def evaluate(
        self,
        experiment_name: str,
        checkpoint: str,
        train_config: dict,
        data_config: dict,
    ) -> None:

        device = train_config["device"]

        model = ModelRegistry.get(train_config["model"])(
            **train_config["model_params"]
        )
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        model = model.to(device)

        loss_fn = LossRegistry.get(train_config["loss_fn"])()

        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=train_config["batch_size"],
            shuffle=True,
            num_workers=train_config["num_workers"],
            pin_memory=train_config["pin_memory"],
        )

        experiment_id = get_or_create_experiment(
            experiment_name=experiment_name
        )
        mlflow.set_experiment(experiment_id=experiment_id)

        with mlflow.start_run(experiment_id=experiment_id):
            test_loss, r2, rmse, mae = self._evaluate(
                test_dataloader,
                model,
                loss_fn,
                device,
            )
            print(
                f"Test loss - {test_loss} | r2: {r2} | rmse: {rmse} "
                f"| mae: {mae}"
            )

            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            mlflow.log_params(
                {
                    "model": train_config.get("model"),
                    "loss_fn": train_config.get("loss_fn"),
                    "device": train_config.get("device"),
                    "num_workers": train_config.get("num_workers"),
                    "pin_memory": train_config.get("pin_memory"),
                    "features": data_config.get("features"),
                }
            )
            mlflow.log_params(train_config.get("model_params", {}))
            mlflow.log_params(train_config.get("loss_fn_params", {}))
            mlflow.log_params(data_config.get("params", {}))

            text_sample, feature_sample, _, mask_sample = self.test_dataset[0]
            onnx_model = to_onnx(
                model, (text_sample, feature_sample, mask_sample)
            )
            mlflow.onnx.log_model(
                onnx_model,
                "model",
                save_as_external_data=False,
            )
