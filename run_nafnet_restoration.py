from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from google.colab import files


print(
    "Upload either 1 blurry image for restoration, or exactly 2 images "
    "(degraded first, clean target second) to compare."
)
uploaded = files.upload()

if len(uploaded) == 0:
    print("No images uploaded.")
else:
    uploaded_items = list(uploaded.items())
    degraded_name, degraded_bytes = uploaded_items[0]
    input_path = Path("/content") / degraded_name
    input_path.write_bytes(degraded_bytes)
    degraded_image = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")

    restored_path = Path("/content/restored_image.png")
    command = [
        sys.executable,
        "-m",
        "basicsr.demo",
        "-opt",
        "options/test/GoPro/NAFNet-width64.yml",
        "--input_path",
        str(input_path),
        "--output_path",
        str(restored_path),
    ]
    result = subprocess.run(
        command, cwd="/content/NAFNet", capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("NAFNet inference failed; see the command output above.")
    restored_image = Image.open(restored_path).convert("RGB")

    if len(uploaded_items) >= 2:
        target_name, target_bytes = uploaded_items[1]
        target_path = Path("/content") / target_name
        target_path.write_bytes(target_bytes)
        target_image = ImageOps.exif_transpose(Image.open(target_path)).convert("RGB")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        images = [degraded_image, restored_image, target_image]
        titles = [f"Input: {degraded_name}", "Restored", f"Target: {target_name}"]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        images = [degraded_image, restored_image]
        titles = [f"Input: {degraded_name}", "NAFNet Restored"]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    print("Restored image saved to:", restored_path)
