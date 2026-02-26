"""Train a PPO agent on the Credit Card Debt environment.

Usage:
    python scripts/train_ppo.py
    python scripts/train_ppo.py --config configs/train/ppo_default.yaml
    python scripts/train_ppo.py --timesteps 5000   # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from src.envs.credit_env import CreditCardDebtEnv
from src.utils.config import load_env_config, load_train_config
from src.agents.callbacks import CreditMetricsCallback, EpisodeLoggerCallback


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_env(env_config_path: str, seed: int = 0):
    """Return a factory function that creates a CreditCardDebtEnv."""
    def _init():
        config = load_env_config(env_config_path)
        env = CreditCardDebtEnv(config=config)
        env.reset(seed=seed)
        return env
    return _init


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train PPO on CreditCardDebtEnv")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/ppo_default.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for training: 'auto' (use CUDA if available), 'cuda', 'cpu', or 'cuda:0' etc.",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_train_config(config_path)

    total_timesteps = args.timesteps or cfg.get("total_timesteps", 500_000)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    n_envs = cfg.get("n_envs", 4)
    env_config_path = cfg.get("env_config", "configs/env/default_3card.yaml")
    log_dir = cfg.get("log_dir", "runs/")
    eval_freq = cfg.get("eval_freq", 10_000)
    eval_episodes = cfg.get("eval_episodes", 50)
    checkpoint_freq = cfg.get("checkpoint_freq", 50_000)

    device = args.device
    print(f"Training PPO for {total_timesteps:,} timesteps")
    print(f"  device: {device}  |  envs: {n_envs}  |  seed: {seed}  |  log_dir: {log_dir}")
    print(f"  env_config: {env_config_path}")
    print()

    # ── Build vectorized training env ────────────────────────────────
    env_fns = [make_env(env_config_path, seed=seed + i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)  # Track episode returns/lengths

    # ── Build eval env (single, deterministic) ───────────────────────
    eval_env = DummyVecEnv([make_env(env_config_path, seed=seed + 1000)])
    eval_env = VecMonitor(eval_env)

    # ── PPO hyperparameters ──────────────────────────────────────────
    policy_kwargs = cfg.get("policy_kwargs", {})
    # Convert net_arch from YAML format to SB3 format
    if "net_arch" in policy_kwargs:
        net_arch = policy_kwargs["net_arch"]
        if isinstance(net_arch, dict):
            # YAML: {pi: [256, 256], vf: [256, 256]} → SB3 dict format
            policy_kwargs["net_arch"] = dict(net_arch)

    model = PPO(
        policy=cfg.get("policy", "MlpPolicy"),
        env=vec_env,
        device=device,
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        n_steps=cfg.get("n_steps", 2048),
        batch_size=cfg.get("batch_size", 64),
        n_epochs=cfg.get("n_epochs", 10),
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        clip_range=float(cfg.get("clip_range", 0.2)),
        ent_coef=float(cfg.get("ent_coef", 0.01)),
        vf_coef=float(cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.5)),
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        tensorboard_log=log_dir,
        seed=seed,
        verbose=1,
    )

    # ── Callbacks ────────────────────────────────────────────────────
    credit_cb = CreditMetricsCallback(verbose=0)

    # Derive scenario tag from env config filename (e.g. "default_3card")
    scenario_tag = Path(env_config_path).stem  # e.g. "default_3card"
    episode_logger_cb = EpisodeLoggerCallback(
        scenario_tag=scenario_tag,
        algo_tag="PPO",
        log_freq=1,   # print every episode; increase to reduce noise
        verbose=0,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(Path("models") / "best"),
        log_path=str(Path(log_dir) / "eval_logs"),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        verbose=1,
    )

    callbacks = CallbackList([credit_cb, episode_logger_cb, eval_cb])

    # ── Train ────────────────────────────────────────────────────────
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # ── Save final model ─────────────────────────────────────────────
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ppo_latest"
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}.zip")
    print(f"TensorBoard logs in {log_dir}")
    print(f"  View with: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
