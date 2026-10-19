import numpy as np
import torch
from torch import nn


def psnr(prediction, target):
    mse = torch.mean((prediction - target) ** 2).item()
    return float("inf") if mse == 0 else 10 * np.log10(4.0 / mse)


def evaluate_restoration(generator, degraded, clean, device):
    generator.eval()
    with torch.no_grad():
        predictions = generator(degraded.to(device)).cpu()

    mean_absolute_error = nn.functional.l1_loss(predictions, clean).item()
    mean_psnr = np.mean(
        [psnr(predictions[index], clean[index]) for index in range(len(clean))]
    )

    print(f"Mean absolute error: {mean_absolute_error:.4f}")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")
    return {
        "predictions": predictions,
        "mean_absolute_error": mean_absolute_error,
        "mean_psnr": mean_psnr,
    }
