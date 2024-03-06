import argparse
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image

from src.utils.common import simple_logger
from src.utils.interface import Detector
from src.utils.visualization import draw_egopath


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ego-Path Detection Inference Script")
    parser.add_argument(
        "model",
        type=str,
        help="Name of the trained model to use (e.g., 'chromatic-laughter-5').",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input file (image or video).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the destination directory for the output file. If not specified, the output is saved in the same directory as the input file.",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default="auto",
        help="Coordinates to use for cropping the input image or video ('auto' for automatic cropping, 'x_left,y_top,x_right,y_bottom' inclusive absolute coordinates for manual cropping, or 'none' to disable cropping).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Inference starting point in the input video in seconds. If not specified, starts from the beginning.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Inference ending point in the input video in seconds. If not specified, processes the video until the end.",
    )
    parser.add_argument(
        "--show-crop",
        action="store_true",
        help="If enabled, displays the crop boundaries in the visual output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"]
        + [f"cuda:{x}" for x in range(torch.cuda.device_count())],
        help="Device to use ('cpu', 'cuda', 'cuda:x' or 'mps').",
    )

    return parser.parse_args()


def main(args):
    logger = simple_logger(__name__, "info")
    base_path = os.path.dirname(__file__)
    # Parse crop coordinates
    if args.crop == "auto":
        crop_coords = "auto"
    elif args.crop == "none":
        crop_coords = None
    else:
        crop_coords = tuple(map(int, args.crop.split(",")))

    detector = Detector(
        model_path=os.path.join(base_path, "weights", args.model),
        crop_coords=crop_coords,
        runtime="pytorch",
        device=args.device,
    )

    extension = os.path.splitext(args.input)[1]
    outname = f"{os.path.splitext(os.path.basename(args.input))[0]}_out{extension}"
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, outname)
    else:
        output_path = os.path.join(os.path.dirname(args.input), outname)

    logger.info(f"\nDetecting ego-path in {args.input}...")

    if extension in [".jpg", ".jpeg", ".png"]:
        frame = Image.open(args.input)
        for _ in range(50 if crop_coords == "auto" else 1):
            crop = detector.get_crop_coords() if args.show_crop else None
            res = detector.detect(frame)
        draw_egopath(frame, res, crop_coords=crop).save(output_path)

    elif extension in [".mp4", ".avi"]:
        cap = cv2.VideoCapture(args.input)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        start_frame = int(args.start * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        max_frames = (
            int(args.end * fps) - start_frame if args.end is not None else total_frames
        )
        current_frame = 0
        logger.info(
            f"\nFRAMES: {total_frames}"
            + f" | RESOLUTION: {frame_width}x{frame_height}"
            + f" | FPS: {fps}"
        )
        progress_bar = simple_logger(f"{__name__}_progress", "info", terminator="\r")
        while current_frame < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            crop = detector.get_crop_coords() if args.show_crop else None
            res = detector.detect(frame)
            vis = draw_egopath(frame, res, crop_coords=crop)
            vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
            out.write(vis)
            current_frame += 1
            progress_bar.info(
                f"Processed {current_frame:0{len(str(max_frames))}}/{max_frames} frames"
                + f" ({current_frame/max_frames*100:.2f}%)"
            )
        cap.release()
        out.release()
        logger.info("")

    else:
        raise NotImplementedError

    logger.info(f"\nInference complete. Output saved to {output_path}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
