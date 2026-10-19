from pathlib import Path
import subprocess
import sys
import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from google.colab import files


print(
    "Upload 1 clean target image to generate a degraded input, OR 2 images: "
    "degraded input first, clean target second."
)
uploaded = files.upload()
if len(uploaded) not in (1, 2):
    raise ValueError(
        f"Upload 1 clean image or 2 paired images; received {len(uploaded)}."
    )

uploaded_items = list(uploaded.items())
if len(uploaded_items) == 1:
    target_name, target_bytes = uploaded_items[0]
    target_image = ImageOps.exif_transpose(
        Image.open(io.BytesIO(target_bytes))
    ).convert("RGB")
    image_array = cv2.cvtColor(np.asarray(target_image), cv2.COLOR_RGB2BGR)
    kernel_size = 21
    blur_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cv2.line(
        blur_kernel,
        (0, kernel_size // 2),
        (kernel_size - 1, kernel_size // 2),
        1.0,
        thickness=1,
    )
    rotation = cv2.getRotationMatrix2D(
        (kernel_size // 2, kernel_size // 2), 25, 1.0
    )
    blur_kernel = cv2.warpAffine(blur_kernel, rotation, (kernel_size, kernel_size))
    blur_kernel /= blur_kernel.sum()
    degraded_array = cv2.filter2D(
        image_array, -1, blur_kernel, borderType=cv2.BORDER_REFLECT
    )
    degraded_image = Image.fromarray(
        cv2.cvtColor(degraded_array, cv2.COLOR_BGR2RGB)
    )
    degraded_name = "synthetic_degraded_" + target_name
else:
    degraded_name, degraded_bytes = uploaded_items[0]
    target_name, target_bytes = uploaded_items[1]
    degraded_image = ImageOps.exif_transpose(
        Image.open(io.BytesIO(degraded_bytes))
    ).convert("RGB")
    target_image = ImageOps.exif_transpose(
        Image.open(io.BytesIO(target_bytes))
    ).convert("RGB")

input_path = Path("/content/degraded_input.png")
target_path = Path("/content") / target_name
target_image.save(target_path)
degraded_image.save(input_path)
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

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, image, title in zip(
    axes,
    [degraded_image, restored_image, target_image],
    [f"Input: {degraded_name}", "Restored", f"Target: {target_name}"],
):
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, image, title in zip(
    axes,
    [degraded_image, target_image],
    ["Degraded Input", "Clean Target"],
):
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
print("Restored image saved to:", restored_path)
