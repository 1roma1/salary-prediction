import mlflow
import onnx_ir
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.base.utils import get_or_create_experiment
from src.base.registries import LossRegistry, ModelRegistry


def to_onnx(model: nn.Module, sample: torch.Tensor):
    model.eval()
    model = model.to("cpu")
    input = sample.unsqueeze(0)
    return onnx_ir.to_proto(torch.onnx.export(model, input, dynamo=True).model)


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
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                y_pred = model(inputs)
                test_loss += loss_fn(y_pred, labels).item()

        test_loss /= len(test_dataloader)
        return test_loss

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
            test_loss = self._evaluate(
                test_dataloader,
                len(self.test_dataset.label_to_int),
                model,
                loss_fn,
                device,
            )
            print(f"Train loss - {test_loss}")

            mlflow.log_metric("test_loss", test_loss)

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

            onnx_model = to_onnx(model, self.test_dataset[0][0])
            mlflow.onnx.log_model(
                onnx_model,
                "model",
                input_example=self.test_dataset[0][0].numpy()[np.newaxis],
                save_as_external_data=False,
            )
