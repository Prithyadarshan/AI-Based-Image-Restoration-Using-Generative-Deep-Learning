from google.colab import files
from PIL import Image, ImageOps
import io

print("Select your images (upload 1 for restoration, or exactly 2 to compare with a clean target).")
uploaded = files.upload()

if len(uploaded) == 0:
    print("No images uploaded.")
else:
    uploaded_items = list(uploaded.items())
    degraded_name, degraded_bytes = uploaded_items[0]
    degraded_image = ImageOps.exif_transpose(
        Image.open(io.BytesIO(degraded_bytes)).convert("RGB")
    )
    original_size = degraded_image.size

    model_input = (
        to_tensor(
            degraded_image.resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
            )
        )
        .unsqueeze(0)
        .to(device)
    )
    generator.eval()
    with torch.no_grad():
        restored_tensor = generator(model_input).cpu()[0]
    restored_small = Image.fromarray(
        (
            (
                (restored_tensor.permute(1, 2, 0).numpy() + 1) / 2
            ).clip(0, 1)
            * 255
        ).astype(np.uint8)
    )
    restored_image = restored_small.resize(
        original_size, Image.Resampling.LANCZOS
    )

    if len(uploaded) >= 2:
        target_name, target_bytes = uploaded_items[1]
        target_image = ImageOps.exif_transpose(
            Image.open(io.BytesIO(target_bytes)).convert("RGB")
        )
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        images = [degraded_image, restored_image, target_image]
        titles = [
            f"Degraded input: {degraded_name}",
            "Restored estimate",
            f"Clean target: {target_name}",
        ]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        images = [degraded_image, restored_image]
        titles = [f"Degraded input: {degraded_name}", "Restored estimate"]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
