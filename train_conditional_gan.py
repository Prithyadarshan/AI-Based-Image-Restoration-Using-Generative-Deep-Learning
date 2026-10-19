import torch
from torch import nn

from conditional_gan_models import device, discriminator, generator
from synthetic_restoration_dataset import train_loader

EPOCHS = 8
L1_WEIGHT = 80.0


def train():
    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=2e-4, betas=(0.5, 0.999)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)
    )
    history = {"generator": [], "discriminator": []}

    for epoch in range(EPOCHS):
        g_total = 0.0
        d_total = 0.0

        for degraded, clean in train_loader:
            degraded = degraded.to(device)
            clean = clean.to(device)
            fake = generator(degraded)

            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(degraded, clean)
            fake_logits = discriminator(degraded, fake.detach())
            d_loss = (
                criterion(real_logits, torch.ones_like(real_logits))
                + criterion(fake_logits, torch.zeros_like(fake_logits))
            ) * 0.5
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            fake_logits = discriminator(degraded, fake)
            g_adv = criterion(fake_logits, torch.ones_like(fake_logits))
            g_recon = nn.functional.l1_loss(fake, clean)
            g_loss = g_adv + L1_WEIGHT * g_recon
            g_loss.backward()
            optimizer_g.step()

            g_total += g_loss.item()
            d_total += d_loss.item()

        history["generator"].append(g_total / len(train_loader))
        history["discriminator"].append(d_total / len(train_loader))
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"G {history['generator'][-1]:.3f} | "
            f"D {history['discriminator'][-1]:.3f}"
        )

    return history


if __name__ == "__main__":
    train()
