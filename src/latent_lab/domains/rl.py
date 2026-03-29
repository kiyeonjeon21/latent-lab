"""Reinforcement Learning experiment domain."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run_experiment(cfg: DictConfig) -> None:
    """Run an RL experiment."""
    from latent_lab.experiments.tracker import log_config, log_metrics, setup_tracking, track_run

    setup_tracking(f"rl-{cfg.name}")

    with track_run(run_name=cfg.name, tags={"domain": "rl"}):
        log_config(cfg)
        _run_sb3(cfg)


def _run_sb3(cfg: DictConfig) -> None:
    """Train with Stable-Baselines3."""
    import gymnasium as gym
    from stable_baselines3 import PPO, A2C, SAC

    env_name = cfg.data.get("name", "CartPole-v1")
    algorithm = cfg.model.get("name", "PPO")
    total_timesteps = cfg.training.get("max_steps", 100_000)

    console.print(f"[cyan]Environment: {env_name}, Algorithm: {algorithm}[/cyan]")

    env = gym.make(env_name)

    algo_map = {"PPO": PPO, "A2C": A2C, "SAC": SAC}
    algo_cls = algo_map.get(algorithm, PPO)

    model = algo_cls("MlpPolicy", env, verbose=1, seed=cfg.training.seed)
    console.print(f"[cyan]Training for {total_timesteps} timesteps...[/cyan]")
    model.learn(total_timesteps=total_timesteps)

    save_path = f"models/checkpoints/{cfg.name}"
    model.save(save_path)
    console.print(f"[green]Model saved to {save_path}[/green]")
