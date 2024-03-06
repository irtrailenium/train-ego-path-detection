import importlib
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import v2 as transforms

from ..nn.model import ClassificationNet, RegressionNet, SegmentationNet
from .autocrop import Autocropper
from .common import to_scaled_tensor
from .postprocessing import (
    classifications_to_rails,
    regression_to_rails,
    scale_mask,
    scale_rails,
)


class Detector:
    def __init__(self, model_path, crop_coords, runtime, device):
        """Interface to infer the train ego-path detection model using PyTorch or TensorRT.

        Args:
            model_path (str): Path to the trained model directory (containing config.yaml and best.pt)
            crop_coords (tuple or str or None): Coordinates to use for cropping the input image before inference:
                - If tuple, should be the inclusive absolute coordinates (xleft, ytop, xright, ybottom) of the fixed region.
                - If str, should be "auto" to use automatic cropping.
                - If None, no cropping is performed.
            runtime (str): Runtime to use for model inference ("pytorch" or "tensorrt").
            device (str): Device to use for model inference ("cpu", "cuda", "cuda:x" or "mps").
        """
        self.model_path = model_path
        self.runtime = runtime
        self.device = torch.device(device)
        with open(os.path.join(self.model_path, "config.yaml")) as f:
            self.config = yaml.safe_load(f)
        if isinstance(crop_coords, tuple) and len(crop_coords) == 4:
            self.crop_coords = crop_coords
        elif crop_coords == "auto":
            self.crop_coords = Autocropper(self.config)
        else:
            self.crop_coords = None

        if self.runtime == "pytorch":
            self.model = self.init_model_pytorch()
        elif self.runtime == "tensorrt":
            # lazy imports
            self.trt = importlib.import_module("tensorrt")
            self.cuda = importlib.import_module("pycuda.driver")
            os.environ["CUDA_MODULE_LOADING"] = "LAZY"
            # convert model to tensorrt if not already done
            if not os.path.exists(os.path.join(self.model_path, "best.trt")):
                self.convert_to_tensorrt()
            # init cuda context on device
            self.cuda.init()
            device = 0 if device == "cuda" else int(device.split(":")[-1])
            self.ctx = self.cuda.Device(device).retain_primary_context()
            self.ctx.push()
            self.exectx, self.bindings, self.shapes = self.init_model_tensorrt()
        else:
            raise ValueError

    def __del__(self):
        if self.runtime == "tensorrt":
            self.ctx.pop()

    def get_crop_coords(self):
        return (
            self.crop_coords()
            if isinstance(self.crop_coords, Autocropper)
            else self.crop_coords
        )

    def init_model_pytorch(self):
        if self.config["method"] == "classification":
            model = ClassificationNet(
                backbone=self.config["backbone"],
                input_shape=tuple(self.config["input_shape"]),
                anchors=self.config["anchors"],
                classes=self.config["classes"],
                pool_channels=self.config["pool_channels"],
                fc_hidden_size=self.config["fc_hidden_size"],
            )
        elif self.config["method"] == "regression":
            model = RegressionNet(
                backbone=self.config["backbone"],
                input_shape=tuple(self.config["input_shape"]),
                anchors=self.config["anchors"],
                pool_channels=self.config["pool_channels"],
                fc_hidden_size=self.config["fc_hidden_size"],
            )
        elif self.config["method"] == "segmentation":
            model = SegmentationNet(
                backbone=self.config["backbone"],
                decoder_channels=tuple(self.config["decoder_channels"]),
            )
        model.to(self.device).eval()
        model.load_state_dict(
            torch.load(
                os.path.join(self.model_path, "best.pt"), map_location=self.device
            )
        )
        return model

    def init_model_tensorrt(self):
        runtime = self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR))
        with open(os.path.join(self.model_path, "best.trt"), "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        exectx = engine.create_execution_context()
        shapes = tuple(
            [tuple(engine.get_binding_shape(i)) for i in range(engine.num_bindings)]
        )
        bindings = [
            self.cuda.mem_alloc(np.prod(shape).item() * np.dtype(np.float32).itemsize)
            for shape in shapes
        ]
        return exectx, bindings, shapes

    def convert_to_tensorrt(self, precision="fp16"):
        pytorch_model = self.init_model_pytorch()
        dummy_input = torch.rand((1, *self.config["input_shape"])).to(self.device)
        torch.onnx.export(pytorch_model, dummy_input, "temp.onnx")
        trt_logger = self.trt.Logger(self.trt.Logger.ERROR)
        builder = self.trt.Builder(trt_logger)
        flag = 1 << (int)(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        config = builder.create_builder_config()
        parser = self.trt.OnnxParser(network, trt_logger)
        with open("temp.onnx", "rb") as model:
            parser.parse(model.read())
        if precision == "fp16":
            config.set_flag(self.trt.BuilderFlag.FP16)
        engine = builder.build_engine(network, config)
        with open(os.path.join(self.model_path, "best.trt"), "wb") as f:
            f.write(engine.serialize())
        os.remove("temp.onnx")

    def infer_model_pytorch(self, img):
        tensor = to_scaled_tensor(img).unsqueeze(0).to(self.device)
        tensor = transforms.Resize(self.config["input_shape"][1:][::-1])(tensor)
        with torch.inference_mode():
            pred = self.model(tensor)
        return pred.cpu().numpy()

    def infer_model_tensorrt(self, img):
        tensor = transforms.Compose(
            [
                to_scaled_tensor,
                transforms.Resize(self.config["input_shape"][1:][::-1]),
            ]
        )(img).contiguous()
        tensor = tensor.numpy()  # convert to numpy
        self.cuda.memcpy_htod(self.bindings[0], tensor)  # copy input to GPU
        self.exectx.execute_v2(self.bindings)  # infer model
        pred = np.empty(self.shapes[1], dtype=np.float32)  # allocate output
        self.cuda.memcpy_dtoh(pred, self.bindings[1])  # copy output to CPU
        return pred

    def detect(self, img):
        """Detects the train ego-path on an image using the model.

        Args:
            img (PIL.Image.Image): Input image on which detection is to be performed.

        Returns:
            list or PIL.Image.Image: Train ego-path detection result, whose type depends on the method used:
                - Classification/Regression: List containing the left and right rails lists of rails point coordinates (x, y).
                - Segmentation: PIL.Image.Image representing the binary mask of detected region.
        """     
        original_shape = img.size
        crop_coords = self.get_crop_coords()
        if crop_coords is not None:
            xleft, ytop, xright, ybottom = crop_coords
            img = img.crop((xleft, ytop, xright + 1, ybottom + 1))

        if self.runtime == "pytorch":
            pred = self.infer_model_pytorch(img)
        elif self.runtime == "tensorrt":
            pred = self.infer_model_tensorrt(img)

        if self.config["method"] == "classification":
            clf = pred.reshape(2, self.config["anchors"], self.config["classes"] + 1)
            clf = np.argmax(clf, axis=2)
            rails = classifications_to_rails(clf, self.config["classes"])
            rails = scale_rails(rails, crop_coords, original_shape)
            rails = np.round(rails).astype(int)
            res = rails.tolist()
        elif self.config["method"] == "regression":
            traj = pred[:, :-1].reshape(2, self.config["anchors"])
            ylim = 1 / (1 + np.exp(-pred[:, -1].item()))  # sigmoid
            rails = regression_to_rails(traj, ylim)
            rails = scale_rails(rails, crop_coords, original_shape)
            rails = np.round(rails).astype(int)
            res = rails.tolist()
        elif self.config["method"] == "segmentation":
            mask = pred.squeeze(0).squeeze(0)
            mask = (mask > 0).astype(np.uint8) * 255
            mask = Image.fromarray(mask)
            res = scale_mask(mask, crop_coords, original_shape)

        if isinstance(self.crop_coords, Autocropper):
            self.crop_coords.update(original_shape, res)

        return res
