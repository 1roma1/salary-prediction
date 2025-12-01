import torch

model.eval()
model = model.to("cpu")
input = sample.unsqueeze(0)
torch.onnx.export(model, (input), "model.onnx", dynamo=True, external_data=False)
