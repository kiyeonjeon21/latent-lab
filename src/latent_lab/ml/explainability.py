"""Model explainability - SHAP, feature importance."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Run explainability analysis."""
    method = cfg.get("method", "shap")
    match method:
        case "shap":
            run_shap(cfg)
        case "feature_importance":
            run_feature_importance(cfg)
        case _:
            console.print(f"[red]Unknown method: {method}[/red]")


def run_shap(cfg: DictConfig) -> None:
    """SHAP analysis on a trained model."""
    import shap
    from sklearn.model_selection import train_test_split

    from latent_lab.ml.classification import _build_model, _load_data
    from latent_lab.experiments.tracker import log_artifact

    X, y = _load_data(cfg)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=cfg.training.seed)

    model = _build_model(cfg.model.get("name", "random_forest"), cfg, cfg.training.seed)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:100])

    console.print(f"[green]SHAP values computed for {X_test[:100].shape[0]} samples[/green]")
    console.print("[cyan]Use shap.summary_plot() in a notebook for visualization.[/cyan]")


def run_feature_importance(cfg: DictConfig) -> None:
    """Feature importance from tree-based models."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    from latent_lab.ml.classification import _build_model, _load_data

    X, y = _load_data(cfg)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=cfg.training.seed)

    model = _build_model(cfg.model.get("name", "random_forest"), cfg, cfg.training.seed)
    model.fit(X_train, y_train)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        console.print("[bold]Feature Importance:[/bold]")
        for i, idx in enumerate(indices[:10]):
            console.print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")
    else:
        console.print("[yellow]Model does not have feature_importances_ attribute.[/yellow]")
