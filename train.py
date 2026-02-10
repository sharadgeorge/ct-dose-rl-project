"""
Training script for CT Dose Optimization RL Agent

Trains a PPO or DQN agent to minimize radiation dose while maintaining image quality.

Usage:
    python train.py --algorithm PPO --timesteps 100000
    python train.py --algorithm DQN --timesteps 50000 --phantom chest
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from envs.ct_dose_env import CTDoseEnv, CTDoseEnvContinuous, ScanConfig


def make_env(
    config: ScanConfig,
    phantom_type: str = "shepp_logan",
    rank: int = 0,
    seed: int = 0,
) -> callable:
    """Create environment factory function."""
    
    def _init():
        env = CTDoseEnv(
            config=config,
            phantom_type=phantom_type,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    
    return _init


def make_continuous_env(
    config: ScanConfig,
    phantom_type: str = "shepp_logan",
    rank: int = 0,
    seed: int = 0,
) -> callable:
    """Create continuous action environment factory function."""
    
    def _init():
        env = CTDoseEnvContinuous(
            config=config,
            phantom_type=phantom_type,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    
    return _init


def train(args):
    """Main training function."""
    
    # Create configuration
    config = ScanConfig(
        n_angles=args.n_angles,
        image_size=256,
        mA_levels=(50, 100, 150, 200, 250),
        noise_scale=args.noise_scale,
        noise_exponent=args.noise_exponent,
        het_noise_scale=args.het_noise_scale,
        step_dose_penalty=args.step_dose_penalty,
        dose_weight=args.dose_weight,
        quality_weight=args.quality_weight,
        ssim_percentile=args.ssim_percentile,
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.algorithm}_{args.phantom}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Configuration: {config}")
    
    # Create environments
    print("Creating environments...")
    
    if args.algorithm == "SAC":
        # SAC requires continuous actions
        env_factory = make_continuous_env
    else:
        env_factory = make_env
    
    if args.n_envs > 1:
        train_env = SubprocVecEnv([
            env_factory(config, args.phantom, rank=i, seed=args.seed)
            for i in range(args.n_envs)
        ])
    else:
        train_env = DummyVecEnv([
            env_factory(config, args.phantom, seed=args.seed)
        ])
    
    # Evaluation environment
    eval_env = DummyVecEnv([
        env_factory(config, args.phantom, seed=args.seed + 1000)
    ])
    
    # Create model
    print(f"Creating {args.algorithm} model...")
    
    common_kwargs = {
        "verbose": 1,
        "tensorboard_log": str(output_dir / "tensorboard"),
        "seed": args.seed,
    }
    
    if args.algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_angles * 10,  # Collect 10 episodes before update
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            **common_kwargs,
        )
    elif args.algorithm == "DQN":
        model = DQN(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            target_update_interval=500,
            **common_kwargs,
        )
    elif args.algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(args.timesteps // 20, 1000),
        n_eval_episodes=10,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 10, 5000),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="model",
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # Train
    print(f"Training for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    model.save(output_dir / "final_model")
    print(f"Final model saved to {output_dir / 'final_model'}")
    
    # Save configuration
    with open(output_dir / "config.txt", "w") as f:
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Phantom: {args.phantom}\n")
        f.write(f"Timesteps: {args.timesteps}\n")
        f.write(f"n_angles: {args.n_angles}\n")
        f.write(f"noise_scale: {args.noise_scale}\n")
        f.write(f"noise_exponent: {args.noise_exponent}\n")
        f.write(f"het_noise_scale: {args.het_noise_scale}\n")
        f.write(f"step_dose_penalty: {args.step_dose_penalty}\n")
        f.write(f"dose_weight: {args.dose_weight}\n")
        f.write(f"quality_weight: {args.quality_weight}\n")
        f.write(f"ssim_percentile: {args.ssim_percentile}\n")
        f.write(f"learning_rate: {args.learning_rate}\n")
        f.write(f"seed: {args.seed}\n")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train CT Dose Optimization RL Agent")
    
    # Algorithm
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "DQN", "SAC"],
        help="RL algorithm (PPO, DQN, or SAC for continuous)"
    )
    
    # Environment
    parser.add_argument(
        "--phantom",
        type=str,
        default="shepp_logan",
        choices=["shepp_logan", "ellipses", "chest"],
        help="Phantom type for simulation"
    )
    parser.add_argument(
        "--n_angles",
        type=int,
        default=60,
        help="Number of projection angles per scan"
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.5,
        help="Global noise amplitude"
    )
    parser.add_argument(
        "--noise_exponent",
        type=float,
        default=0.08,
        help="Exponential noise strength (thick paths get more noise)"
    )
    parser.add_argument(
        "--het_noise_scale",
        type=float,
        default=0.5,
        help="Heterogeneity noise scale (density transitions get more noise)"
    )
    parser.add_argument(
        "--step_dose_penalty",
        type=float,
        default=0.02,
        help="Per-step mA cost penalty"
    )
    parser.add_argument(
        "--dose_weight",
        type=float,
        default=0.0,
        help="Weight for dose penalty in final reward (0 = dose penalized per-step only)"
    )
    parser.add_argument(
        "--quality_weight",
        type=float,
        default=1.0,
        help="Weight for image quality in reward"
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for PPO (higher = more exploration)"
    )
    parser.add_argument(
        "--ssim_percentile",
        type=float,
        default=5.0,
        help="Percentile of local SSIM map for quality metric (lower = more worst-region focus)"
    )
    
    # Training
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory for saving models and logs"
    )
    
    args = parser.parse_args()
    
    output_dir = train(args)
    print(f"\nTraining complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
