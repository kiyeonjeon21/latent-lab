"""Clustering experiments."""

from omegaconf import DictConfig
from rich.console import Console

from latent_lab.ml.classification import _load_data

console = Console()


def run(cfg: DictConfig) -> None:
    """Run clustering experiment."""
    _run_clustering(cfg)


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
