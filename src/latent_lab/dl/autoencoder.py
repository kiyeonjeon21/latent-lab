"""Autoencoder and VAE training."""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from rich.console import Console

from latent_lab.models.torch_utils import get_device, seed_everything

console = Console()


def run(cfg: DictConfig) -> None:
    """Train autoencoder or VAE."""
    variant = cfg.model.get("variant", "ae")
    if variant == "vae":
        train_vae(cfg)
    else:
        train_ae(cfg)


def train_ae(cfg: DictConfig) -> None:
    """Train a convolutional autoencoder on MNIST."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from latent_lab.experiments.tracker import log_metrics

    seed_everything(cfg.training.seed)
    device = get_device()

    transform = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.MNIST(cfg.data.get("path", "data/raw"), train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)

    latent_dim = cfg.model.get("latent_dim", 32)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(), nn.Linear(64 * 7 * 7, latent_dim),
            )
        def forward(self, x):
            return self.net(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
            self.net = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
            )
        def forward(self, z):
            return self.net(self.fc(z).view(-1, 64, 7, 7))

    encoder, decoder = Encoder().to(device), Decoder().to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.training.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(cfg.training.epochs):
        total_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            loss = criterion(decoder(encoder(images)), images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_ds)
        log_metrics({"recon_loss": avg_loss}, step=epoch)
        console.print(f"  Epoch {epoch + 1}/{cfg.training.epochs} | Recon Loss: {avg_loss:.6f}")

    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, f"models/checkpoints/{cfg.name}.pt")
    console.print(f"[green]Autoencoder saved.[/green]")


def train_vae(cfg: DictConfig) -> None:
    """Train a Variational Autoencoder on MNIST."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from latent_lab.experiments.tracker import log_metrics

    seed_everything(cfg.training.seed)
    device = get_device()
    latent_dim = cfg.model.get("latent_dim", 16)

    transform = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.MNIST(cfg.data.get("path", "data/raw"), train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 400), nn.ReLU())
            self.fc_mu = nn.Linear(400, latent_dim)
            self.fc_logvar = nn.Linear(400, latent_dim)
            self.decoder = nn.Sequential(nn.Linear(latent_dim, 400), nn.ReLU(), nn.Linear(400, 784), nn.Sigmoid())

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)

        def decode(self, z):
            return self.decoder(z).view(-1, 1, 28, 28)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    def vae_loss(recon, x, mu, logvar):
        bce = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld

    for epoch in range(cfg.training.epochs):
        total_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            recon, mu, logvar = model(images)
            loss = vae_loss(recon, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_ds)
        log_metrics({"vae_loss": avg_loss}, step=epoch)
        console.print(f"  Epoch {epoch + 1}/{cfg.training.epochs} | VAE Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"models/checkpoints/{cfg.name}.pt")
    console.print(f"[green]VAE saved.[/green]")
