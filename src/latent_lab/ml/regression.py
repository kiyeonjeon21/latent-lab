"""Regression experiments."""

from omegaconf import DictConfig
from rich.console import Console

from latent_lab.ml.classification import _build_model, _load_data

console = Console()


def run(cfg: DictConfig) -> None:
    """Run regression experiment."""
    _run_regression(cfg)


def _run_regression(cfg: DictConfig) -> None:
    """Regression experiment."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    from latent_lab.experiments.tracker import log_metrics

    algorithm = cfg.model.get("name", "random_forest")
    seed = cfg.training.get("seed", 42)

    X, y = _load_data(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = _build_model(algorithm, cfg, seed, task="regression")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    log_metrics({"mse": mse, "mae": mae, "r2": r2})
    console.print(f"[green]MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}[/green]")
