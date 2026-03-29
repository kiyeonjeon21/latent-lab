"""Hyperparameter tuning with Optuna."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Run Optuna hyperparameter search."""
    import optuna
    from sklearn.model_selection import cross_val_score

    from latent_lab.ml.classification import _build_model, _load_data

    X, y = _load_data(cfg)
    n_trials = cfg.get("n_trials", 50)
    algorithm = cfg.model.get("name", "xgboost")
    seed = cfg.training.get("seed", 42)

    def objective(trial):
        params = {}
        if algorithm in ("xgboost", "lightgbm", "random_forest"):
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
            params["max_depth"] = trial.suggest_int("max_depth", 2, 12)
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)

        # Override config with trial params
        from omegaconf import OmegaConf
        trial_cfg = OmegaConf.merge(cfg, {"model": params, "training": {"learning_rate": params.get("learning_rate", 0.1)}})
        model = _build_model(algorithm, trial_cfg, seed)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    console.print(f"[green]Best accuracy: {study.best_value:.4f}[/green]")
    console.print(f"[green]Best params: {study.best_params}[/green]")

    from latent_lab.experiments.tracker import log_metrics, log_params
    log_params(study.best_params)
    log_metrics({"best_cv_accuracy": study.best_value})
