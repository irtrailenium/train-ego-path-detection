import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from PIL import Image

from src.utils.interface import Detector
from src.utils.visualization import draw_egopath

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

base_path = os.path.dirname(__file__)
os.makedirs(os.path.join(base_path, "output"), exist_ok=True)

classification_model_path = os.path.join(base_path, "weights", "fortuitous-goat-12")
regression_model_path = os.path.join(base_path, "weights", "chromatic-laughter-5")
segmentation_model_path = os.path.join(base_path, "weights", "twinkling-rocket-21")

img = Image.open(os.path.join(base_path, "data", "egopath.jpg"))

to_test = (
    list(range(1, 5))
    # + [5]  # uncomment to test inference with TensorRT
)

# 1) Inference without cropping
if 1 in to_test:
    detector = Detector(
        model_path=regression_model_path,
        crop_coords=None,
        runtime="pytorch",
        device=device,
    )
    egopath = detector.detect(img)
    vis = draw_egopath(img, egopath)
    vis.save(os.path.join(base_path, "output", "demo1.jpg"))

# 2) Inference with manual cropping
if 2 in to_test:
    crop_coords = (580, 270, 1369, 1079)
    detector = Detector(regression_model_path, crop_coords, "pytorch", device)
    egopath = detector.detect(img)
    vis = draw_egopath(img, egopath, crop_coords=detector.get_crop_coords())
    vis.save(os.path.join(base_path, "output", "demo2.jpg"))

# 3) Inference with automatic cropping
if 3 in to_test:
    detector = Detector(regression_model_path, "auto", "pytorch", device)
    for _ in range(50):  # multiple iterations to get a stable crop
        crop_coords = detector.get_crop_coords()
        egopath = detector.detect(img)
    vis = draw_egopath(img, egopath, crop_coords=crop_coords)
    vis.save(os.path.join(base_path, "output", "demo3.jpg"))

# 4) Methods (classification, regression, segmentation) can be used interchangeably
if 4 in to_test:
    crop_coords = (580, 270, 1369, 1079)
    models = (
        classification_model_path,
        regression_model_path,
        segmentation_model_path,
    )
    dst = Image.new("RGB", (img.width * len(models), img.height))
    for i, model_path in enumerate(models):
        detector = Detector(model_path, crop_coords, "pytorch", device)
        pred = detector.detect(img)
        vis = draw_egopath(img, pred, crop_coords=detector.get_crop_coords())
        dst.paste(vis, (img.width * i, 0))
    dst.save(os.path.join(base_path, "output", "demo4.jpg"))

# 5) Inference with TensorRT
if 5 in to_test:
    crop_coords = (580, 270, 1369, 1079)
    detector = Detector(
        model_path=regression_model_path,
        crop_coords=crop_coords,
        runtime="tensorrt",  # only this line changes from the previous examples
        device=device,
    )
    egopath = detector.detect(img)
    vis = draw_egopath(img, egopath, crop_coords=detector.get_crop_coords())
    vis.save(os.path.join(base_path, "output", "demo5.jpg"))
