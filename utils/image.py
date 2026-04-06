from PIL import Image


def preprocess_image(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize((448, 448), Image.Resampling.LANCZOS)


def filter_image(example: dict) -> bool:
    w, h = example["image"].size
    return w > 56 and h > 56
