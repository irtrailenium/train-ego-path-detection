import json
import os
import random
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import yaml

from src.utils.common import (
    set_seeds,
    split_dataset,
)
from src.utils.dataset import PathsDataset
from src.utils.evaluate import IoUEvaluator, LatencyEvaluator

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

methods = [  # methods to evaluate
    "classification",
    "regression",
    "segmentation",
]
backbones = [  # backbones to evaluate
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "resnet18",
    "resnet34",
    "resnet50",
]
runtimes = [  # runtimes to evaluate
    "pytorch",
    "tensorrt",
]
metrics = [  # metrics to evaluate
    "iou",
    "latency",
]

basepath = os.path.dirname(__file__)
models_to_eval = [
    f
    for f in os.listdir(os.path.join(basepath, "weights"))
    if os.path.isdir(os.path.join(basepath, "weights", f))
]

if "iou" in metrics:
    with open(os.path.join("configs", "global.yaml")) as f:
        config = yaml.safe_load(f)
    images_path = config["images_path"]
    annotations_path = config["annotations_path"]
    set_seeds(config["seed"])
    with open(annotations_path) as json_file:
        indices = list(range(len(json.load(json_file).keys())))
    random.shuffle(indices)
    proportions = (config["train_prop"], config["val_prop"], config["test_prop"])
    train_indices, val_indices, test_indices = split_dataset(indices, proportions)
    test_dataset = PathsDataset(
        images_path, annotations_path, test_indices, config, "segmentation"
    )

stats = []
for model in models_to_eval:
    with open(os.path.join("weights", model, "config.yaml")) as f:
        model_config = yaml.safe_load(f)
    method = model_config["method"]
    backbone = model_config["backbone"]
    if backbone not in backbones or method not in methods:
        continue
    for runtime in runtimes:
        if "latency" in metrics:
            time.sleep(30)  # cooldown
            latency_evaluator = LatencyEvaluator(
                model_path=os.path.join("weights", model),
                runtime=runtime,
                device=device,
            )
            latency = latency_evaluator.evaluate()
        if "iou" in metrics:
            iou_evaluator = IoUEvaluator(
                dataset=test_dataset,
                model_path=os.path.join("weights", model),
                runtime=runtime,
                device=device,
            )
            iou = iou_evaluator.evaluate()
        precision = "amx" if runtime == "tensorrt" else "fp32"
        stats.append(
            f"{runtime},{backbone},{precision},{method},{model}"
            + (f",{latency * 1000:.2f}" if "latency" in metrics else "")
            + (f",{iou:.5f}" if "iou" in metrics else "")
            + "\n"
        )
stats.sort()

os.makedirs("output", exist_ok=True)
with open(os.path.join("output", "eval.csv"), "w") as f:
    f.write("runtime,backbone,precision,method,model")
    f.write(",latency" if "latency" in metrics else "")
    f.write(",iou" if "iou" in metrics else "")
    f.write("\n")
    f.writelines(stats)
