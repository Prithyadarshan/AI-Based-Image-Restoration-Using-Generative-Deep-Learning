import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


def visualize_restoration_results(generator, dataset, device, count=6):
    generator.eval()
    sample_count = min(count, len(dataset))
    loader = DataLoader(
        dataset, batch_size=sample_count, shuffle=False, num_workers=0
    )
    degraded, clean = next(iter(loader))

    with torch.no_grad():
        restored = generator(degraded.to(device)).cpu()

    fig, axes = plt.subplots(
        sample_count, 3, figsize=(8, 2.4 * sample_count), squeeze=False
    )
    for row in range(sample_count):
        for col, (tensor, title) in enumerate(
            [
                (degraded[row], "Input"),
                (restored[row], "Restored"),
                (clean[row], "Target"),
            ]
        ):
            axes[row, col].imshow(
                ((tensor.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
            )
            axes[row, col].set_title(title)
            axes[row, col].axis("off")

    plt.tight_layout()
    return fig, axes
