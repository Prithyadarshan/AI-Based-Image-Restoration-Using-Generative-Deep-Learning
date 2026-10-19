import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset

SEED = 7
IMAGE_SIZE = 64


def make_clean_image(seed, size=IMAGE_SIZE):
    rng = random.Random(seed)
    base = tuple(rng.randint(20, 230) for _ in range(3))
    image = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(image)
    for _ in range(7):
        x1, y1 = rng.randrange(size), rng.randrange(size)
        x2, y2 = rng.randrange(size), rng.randrange(size)
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        color = tuple(rng.randint(0, 255) for _ in range(3))
        if rng.random() < 0.5:
            draw.ellipse((xmin, ymin, xmax, ymax), fill=color)
        else:
            draw.rectangle((xmin, ymin, xmax, ymax), fill=color)
    return image


def to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


class RestorationPairs(Dataset):
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        clean_pil = make_clean_image(SEED + index)
        blur_radius = 0.6 + (index % 4) * 0.35
        degraded_pil = clean_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        clean = to_tensor(clean_pil)
        degraded = to_tensor(degraded_pil)
        degraded = (degraded + torch.randn_like(degraded) * 0.12).clamp(-1, 1)
        return degraded, clean


train_loader = DataLoader(
    RestorationPairs(1200), batch_size=32, shuffle=True, num_workers=0
)
preview_input, preview_target = next(iter(train_loader))
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
for ax, tensor, title in zip(
    axes,
    [preview_input[0], preview_target[0]],
    ["Degraded input", "Clean target"],
):
    ax.imshow(((tensor.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
