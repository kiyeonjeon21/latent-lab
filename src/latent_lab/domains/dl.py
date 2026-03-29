"""Deep Learning experiment domain - CNN, Autoencoder, GAN, Diffusion on PyTorch MPS."""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from rich.console import Console

from latent_lab.models.torch_utils import get_device, seed_everything

console = Console()


def run_experiment(cfg: DictConfig) -> None:
    """Run a DL experiment."""
    from latent_lab.experiments.tracker import log_config, setup_tracking, track_run

    setup_tracking(f"dl-{cfg.name}")

    with track_run(run_name=cfg.name, tags={"domain": "dl"}):
        log_config(cfg)

        task = cfg.get("task", "train_cnn")

        match task:
            case "train_cnn":
                _train_cnn(cfg)
            case "train_autoencoder":
                _train_autoencoder(cfg)
            case "train_gan":
                _train_gan(cfg)
            case "diffusion_inference":
                _diffusion_inference(cfg)
            case _:
                console.print(f"[red]Unknown DL task: {task}[/red]")


# ---------------------------------------------------------------------------
# CNN Training
# ---------------------------------------------------------------------------
def _train_cnn(cfg: DictConfig) -> None:
    """Train a CNN classifier on CIFAR-10 or custom dataset."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    from latent_lab.experiments.tracker import log_metrics

    seed_everything(cfg.training.seed)
    device = get_device()
    console.print(f"[cyan]Device: {device}[/cyan]")

    # Data
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    data_name = cfg.data.get("name", "cifar10")
    data_dir = cfg.data.get("path", "data/raw")

    if data_name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        num_classes = 10
    elif data_name == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        console.print(f"[red]Dataset '{data_name}' not supported. Use cifar10/cifar100.[/red]")
        return

    bs = cfg.training.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4)

    # Model
    model_name = cfg.model.get("name", "resnet18")
    if cfg.model.get("pretrained"):
        import timm

        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    else:
        model = _build_simple_cnn(num_classes)

    model = model.to(device)
    console.print(f"[cyan]Model: {model_name}, Params: {sum(p.numel() for p in model.parameters()):,}[/cyan]")

    # Training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.training.epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.training.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total

        # Evaluate
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_correct += (outputs.argmax(1) == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total

        log_metrics(
            {"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc},
            step=epoch,
        )
        console.print(
            f"  Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
        )

    # Save
    save_path = f"models/checkpoints/{cfg.name}.pt"
    torch.save(model.state_dict(), save_path)
    console.print(f"[green]Model saved to {save_path}[/green]")


def _build_simple_cnn(num_classes: int) -> nn.Module:
    """Simple CNN for CIFAR-scale images."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, num_classes),
    )


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------
def _train_autoencoder(cfg: DictConfig) -> None:
    """Train a convolutional autoencoder."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    from latent_lab.experiments.tracker import log_metrics

    seed_everything(cfg.training.seed)
    device = get_device()

    transform = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        cfg.data.get("path", "data/raw"), train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)

    latent_dim = cfg.model.get("latent_dim", 32)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, latent_dim),
            )

        def forward(self, x):
            return self.net(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
            self.net = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid(),
            )

        def forward(self, z):
            x = self.fc(z).view(-1, 64, 7, 7)
            return self.net(x)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.training.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(cfg.training.epochs):
        total_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            z = encoder(images)
            recon = decoder(z)
            loss = criterion(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_ds)
        log_metrics({"recon_loss": avg_loss}, step=epoch)
        console.print(f"  Epoch {epoch + 1}/{cfg.training.epochs} | Recon Loss: {avg_loss:.6f}")

    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
               f"models/checkpoints/{cfg.name}.pt")
    console.print(f"[green]Autoencoder saved.[/green]")


# ---------------------------------------------------------------------------
# GAN
# ---------------------------------------------------------------------------
def _train_gan(cfg: DictConfig) -> None:
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


# ---------------------------------------------------------------------------
# Diffusion (inference only — training requires too much compute)
# ---------------------------------------------------------------------------
def _diffusion_inference(cfg: DictConfig) -> None:
    """Run Stable Diffusion inference on MPS."""
    from diffusers import StableDiffusionPipeline

    device = get_device()
    model_id = cfg.model.get("pretrained", "stabilityai/stable-diffusion-2-1-base")
    console.print(f"[cyan]Loading {model_id}...[/cyan]")

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    prompt = cfg.get("prompt", "a photo of a cat in space, digital art")
    num_steps = cfg.get("num_inference_steps", 30)

    console.print(f"[cyan]Generating: '{prompt}'[/cyan]")
    image = pipe(prompt, num_inference_steps=num_steps).images[0]

    output_path = f"reports/figures/{cfg.name}.png"
    image.save(output_path)
    console.print(f"[green]Image saved to {output_path}[/green]")
