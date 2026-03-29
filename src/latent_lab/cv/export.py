"""Model export - CoreML, ONNX."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Export model to CoreML or ONNX."""
    export_format = cfg.get("export_format", "coreml")
    match export_format:
        case "coreml":
            export_coreml(cfg)
        case "onnx":
            export_onnx(cfg)


def export_coreml(cfg: DictConfig) -> None:
    """Export YOLO model to CoreML."""
    from ultralytics import YOLO
    model_path = cfg.model.get("pretrained", "yolo11n.pt")
    model = YOLO(model_path)
    model.export(format="coreml")
    console.print(f"[green]Exported {model_path} to CoreML[/green]")


def export_onnx(cfg: DictConfig) -> None:
    """Export YOLO model to ONNX."""
    from ultralytics import YOLO
    model_path = cfg.model.get("pretrained", "yolo11n.pt")
    model = YOLO(model_path)
    model.export(format="onnx")
    console.print(f"[green]Exported {model_path} to ONNX[/green]")
