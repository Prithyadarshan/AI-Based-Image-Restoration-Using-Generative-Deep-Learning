import torch
from torch import nn


def down(in_channels, out_channels, normalize=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not normalize)
    ]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = down(3, 64, False)
        self.d2 = down(64, 128)
        self.d3 = down(128, 256)
        self.d4 = down(256, 512)
        self.u1 = up(512, 256)
        self.u2 = up(512, 128)
        self.u3 = up(256, 64)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        a = self.d1(x)
        b = self.d2(a)
        c = self.d3(b)
        d = self.d4(c)
        x = self.u1(d)
        x = self.u2(torch.cat([x, c], 1))
        x = self.u3(torch.cat([x, b], 1))
        return self.out(torch.cat([x, a], 1))


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            down(6, 64, False),
            down(64, 128),
            down(128, 256),
            nn.Conv2d(256, 1, 4, 1, 1),
        )

    def forward(self, degraded, candidate):
        return self.net(torch.cat([degraded, candidate], dim=1))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = PatchDiscriminator().to(device)
print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
