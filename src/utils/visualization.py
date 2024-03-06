import numpy as np
from PIL import Image, ImageDraw


def draw_egopath(img, egopath, opacity=0.5, color=(0, 189, 80), crop_coords=None):
    """Overlays the train ego-path on the input image.

    Args:
        img (PIL.Image.Image): Input image on which rails are to be visualized.
        egopath (list or numpy.ndarray): Ego-path to be visualized on the image, either as a list of points (classification/regression) or as a mask (segmentation).
        opacity (float, optional): Opacity level of the overlay. Defaults to 0.5.
        color (tuple, optional): Color of the overlay. Defaults to (0, 189, 80).
        crop_coords (tuple, optional): Crop coordinates used during inference. If provided, a red rectangle will be drawn around the cropped region. Defaults to None.

    Returns:
        PIL.Image.Image: Image with the ego-path overlay.
    """
    vis = img.copy()
    if isinstance(egopath, list):  # classification/regression
        left_rail, right_rail = egopath
        if not left_rail or not right_rail:
            return vis
        points = left_rail + right_rail[::-1]
        mask = Image.new("RGBA", vis.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask)
        draw.polygon([tuple(xy) for xy in points], fill=color + (int(255 * opacity),))
        vis.paste(mask, (0, 0), mask)
    elif isinstance(egopath, Image.Image):  # segmentation
        mask = Image.fromarray(np.array(egopath) * opacity).convert("L")
        colored_mask = Image.new("RGBA", mask.size, color + (0,))
        colored_mask.putalpha(mask)
        vis.paste(colored_mask, (0, 0), colored_mask)
    if crop_coords is not None:
        draw = ImageDraw.Draw(vis)
        draw.rectangle(crop_coords, outline=(255, 0, 0), width=1)
    return vis
