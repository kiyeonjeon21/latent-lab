"""Classification with sklearn/XGBoost/LightGBM."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Run classification experiment."""
    _run_classification(cfg)


def _run_classification(cfg: DictConfig) -> None:
    """Classification with sklearn/XGBoost/LightGBM."""
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import cross_val_score, train_test_split

    from latent_lab.experiments.tracker import log_metrics

    algorithm = cfg.model.get("name", "random_forest")
    seed = cfg.training.get("seed", 42)

    # Load data
    X, y = _load_data(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    console.print(f"[cyan]Train: {X_train.shape}, Test: {X_test.shape}[/cyan]")

    # Build model
    model = _build_model(algorithm, cfg, seed)
    console.print(f"[cyan]Training {algorithm}...[/cyan]")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    console.print(f"[green]CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})[/green]")

    # Fit and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    log_metrics({"accuracy": acc, "f1_weighted": f1, "cv_mean": cv_scores.mean()})
    console.print(f"[green]Test Accuracy: {acc:.4f}, F1: {f1:.4f}[/green]")
    console.print(classification_report(y_test, y_pred))


def _load_data(cfg: DictConfig):
    """Load dataset by name or from file."""
    import numpy as np

    data_name = cfg.data.get("name", "iris")

    # Built-in datasets
    match data_name:
        case "iris":
            from sklearn.datasets import load_iris

            data = load_iris()
            return data.data, data.target
        case "wine":
            from sklearn.datasets import load_wine

            data = load_wine()
            return data.data, data.target
        case "digits":
            from sklearn.datasets import load_digits

            data = load_digits()
            return data.data, data.target
        case "boston" | "california":
            from sklearn.datasets import fetch_california_housing

            data = fetch_california_housing()
            return data.data, data.target
        case "breast_cancer":
            from sklearn.datasets import load_breast_cancer

            data = load_breast_cancer()
            return data.data, data.target
        case _:
            # Load from file
            import polars as pl

            path = cfg.data.get("path", "")
            if not path:
                console.print(f"[red]Unknown dataset: {data_name}, and no path provided[/red]")
                raise ValueError(f"Unknown dataset: {data_name}")

            df = pl.read_csv(path) if path.endswith(".csv") else pl.read_parquet(path)
            target_col = cfg.data.get("target_column", df.columns[-1])
            feature_cols = [c for c in df.columns if c != target_col]
            return df[feature_cols].to_numpy(), df[target_col].to_numpy()


def _build_model(algorithm: str, cfg: DictConfig, seed: int, task: str = "classification"):
    """Build a model by algorithm name."""
    match algorithm:
        case "random_forest":
            if task == "regression":
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(
                    n_estimators=cfg.model.get("n_estimators", 100),
                    max_depth=cfg.model.get("max_depth", None),
                    random_state=seed,
                )
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=cfg.model.get("n_estimators", 100),
                max_depth=cfg.model.get("max_depth", None),
                random_state=seed,
            )
        case "xgboost":
            import xgboost as xgb

            if task == "regression":
                return xgb.XGBRegressor(
                    n_estimators=cfg.model.get("n_estimators", 100),
                    max_depth=cfg.model.get("max_depth", 6),
                    learning_rate=cfg.training.get("learning_rate", 0.1),
                    random_state=seed,
                    tree_method="hist",
                )
            return xgb.XGBClassifier(
                n_estimators=cfg.model.get("n_estimators", 100),
                max_depth=cfg.model.get("max_depth", 6),
                learning_rate=cfg.training.get("learning_rate", 0.1),
                random_state=seed,
                tree_method="hist",
            )
        case "lightgbm":
            import lightgbm as lgb

            if task == "regression":
                return lgb.LGBMRegressor(
                    n_estimators=cfg.model.get("n_estimators", 100),
                    max_depth=cfg.model.get("max_depth", -1),
                    learning_rate=cfg.training.get("learning_rate", 0.1),
                    random_state=seed,
                )
            return lgb.LGBMClassifier(
                n_estimators=cfg.model.get("n_estimators", 100),
                max_depth=cfg.model.get("max_depth", -1),
                learning_rate=cfg.training.get("learning_rate", 0.1),
                random_state=seed,
            )
        case "svm":
            if task == "regression":
                from sklearn.svm import SVR

                return SVR()
            from sklearn.svm import SVC

            return SVC(random_state=seed)
        case "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=1000, random_state=seed)
        case _:
            console.print(f"[yellow]Unknown algorithm '{algorithm}', defaulting to RandomForest[/yellow]")
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=100, random_state=seed)
