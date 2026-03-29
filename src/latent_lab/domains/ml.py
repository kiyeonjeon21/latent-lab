"""Classical ML experiment domain - sklearn, XGBoost, LightGBM."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run_experiment(cfg: DictConfig) -> None:
    """Run a classical ML experiment."""
    from latent_lab.experiments.tracker import log_config, log_metrics, setup_tracking, track_run

    setup_tracking(f"ml-{cfg.name}")

    with track_run(run_name=cfg.name, tags={"domain": "ml"}):
        log_config(cfg)

        task = cfg.get("task", "classify")

        match task:
            case "classify":
                _run_classification(cfg)
            case "regress":
                _run_regression(cfg)
            case "cluster":
                _run_clustering(cfg)
            case _:
                console.print(f"[red]Unknown ML task: {task}[/red]")


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


def _run_clustering(cfg: DictConfig) -> None:
    """Clustering experiment."""
    from sklearn.metrics import calinski_harabasz_score, silhouette_score

    from latent_lab.experiments.tracker import log_metrics

    algorithm = cfg.model.get("name", "kmeans")
    seed = cfg.training.get("seed", 42)

    X, _ = _load_data(cfg)

    match algorithm:
        case "kmeans":
            from sklearn.cluster import KMeans

            n_clusters = cfg.model.get("n_clusters", 3)
            model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        case "dbscan":
            from sklearn.cluster import DBSCAN

            model = DBSCAN(eps=cfg.model.get("eps", 0.5), min_samples=cfg.model.get("min_samples", 5))
        case _:
            console.print(f"[red]Unknown clustering algorithm: {algorithm}[/red]")
            return

    labels = model.fit_predict(X)
    n_labels = len(set(labels) - {-1})

    if n_labels > 1:
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        log_metrics({"silhouette": sil, "calinski_harabasz": ch, "n_clusters": n_labels})
        console.print(f"[green]Clusters: {n_labels}, Silhouette: {sil:.4f}, CH: {ch:.4f}[/green]")


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
