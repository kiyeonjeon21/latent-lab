"""GAN training - DCGAN on MNIST."""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from rich.console import Console

from latent_lab.models.torch_utils import get_device, seed_everything

console = Console()


def run(cfg: DictConfig) -> None:
    """Train a DCGAN on MNIST."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    from latent_lab.experiments.tracker import log_metrics

    seed_everything(cfg.training.seed)
    device = get_device()
    latent_dim = cfg.model.get("latent_dim", 100)

    transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    train_ds = torchvision.datasets.MNIST(
        cfg.data.get("path", "data/raw"), train=True, download=True, transform=transform
    )
    loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 784),
                nn.Tanh(),
            )

        def forward(self, z):
            return self.net(z).view(-1, 1, 28, 28)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    G = Generator().to(device)
    D = Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(cfg.training.epochs):
        g_loss_total, d_loss_total = 0.0, 0.0
        for real_images, _ in loader:
            bs = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            # Train Discriminator
            z = torch.randn(bs, latent_dim, device=device)
            fake_images = G(z).detach()
            d_loss = criterion(D(real_images), real_labels) + criterion(D(fake_images), fake_labels)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # Train Generator
            z = torch.randn(bs, latent_dim, device=device)
            g_loss = criterion(D(G(z)), real_labels)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

        n = len(loader)
        log_metrics({"g_loss": g_loss_total / n, "d_loss": d_loss_total / n}, step=epoch)
        console.print(
            f"  Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"G Loss: {g_loss_total / n:.4f} | D Loss: {d_loss_total / n:.4f}"
        )

    torch.save({"G": G.state_dict(), "D": D.state_dict()}, f"models/checkpoints/{cfg.name}.pt")
    console.print(f"[green]GAN saved.[/green]")
