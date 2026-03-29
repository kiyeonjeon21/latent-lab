"""CNN training - classification, transfer learning."""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from rich.console import Console

from latent_lab.models.torch_utils import get_device, seed_everything

console = Console()


def run(cfg: DictConfig) -> None:
    """Train a CNN classifier."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from latent_lab.experiments.tracker import log_metrics

    seed_everything(cfg.training.seed)
    device = get_device()
    console.print(f"[cyan]Device: {device}[/cyan]")

    transform = T.Compose([
        T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
        T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
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
        console.print(f"[red]Dataset '{data_name}' not supported.[/red]")
        return

    bs = cfg.training.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4)

    model_name = cfg.model.get("name", "resnet18")
    if cfg.model.get("pretrained"):
        import timm
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    else:
        model = _build_simple_cnn(num_classes)

    model = model.to(device)
    console.print(f"[cyan]Model: {model_name}, Params: {sum(p.numel() for p in model.parameters()):,}[/cyan]")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
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

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_correct += (outputs.argmax(1) == labels).sum().item()
                test_total += labels.size(0)
        test_acc = test_correct / test_total

        log_metrics({"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}, step=epoch)
        console.print(f"  Epoch {epoch + 1}/{cfg.training.epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    save_path = f"models/checkpoints/{cfg.name}.pt"
    torch.save(model.state_dict(), save_path)
    console.print(f"[green]Model saved to {save_path}[/green]")


def _build_simple_cnn(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes),
    )
