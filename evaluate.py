"""
Evaluation script for CT Dose Optimization RL Agent

Compares trained agent against baselines:
- Fixed mA (constant tube current)
- Oracle (perfect thickness-based modulation)
- Random policy

Usage:
    python evaluate.py --model_path outputs/PPO_xxx/best_model/best_model.zip
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from stable_baselines3 import PPO, DQN, SAC

from envs.ct_dose_env import CTDoseEnv, CTDoseEnvContinuous, ScanConfig


def fixed_mA_policy(mA_value: int):
    """Returns a policy that always chooses the same mA."""
    def policy(obs):
        # Map mA value to action index
        mA_to_action = {50: 0, 100: 1, 150: 2, 200: 3, 250: 4}
        return mA_to_action.get(mA_value, 2)  # Default to medium
    return policy


def oracle_policy(env: CTDoseEnv):
    """
    Oracle policy that modulates mA based on body thickness.
    Uses more mA for thicker regions (lateral views).
    """
    def policy(obs):
        # obs[1] is normalized thickness (0 to 1)
        thickness = obs[1]
        
        # Map thickness to mA: thicker = more mA
        if thickness < 0.3:
            return 0  # 50 mA
        elif thickness < 0.5:
            return 1  # 100 mA
        elif thickness < 0.7:
            return 2  # 150 mA
        elif thickness < 0.85:
            return 3  # 200 mA
        else:
            return 4  # 250 mA
    
    return policy


def random_policy(env: CTDoseEnv):
    """Random policy for baseline comparison."""
    def policy(obs):
        return env.action_space.sample()
    return policy


def evaluate_policy(
    env: CTDoseEnv,
    policy,
    n_episodes: int = 50,
    deterministic: bool = True,
    policy_name: str = "Policy",
):
    """
    Evaluate a policy on the environment.
    
    Returns dict of metrics.
    """
    results = {
        "rewards": [],
        "ssim": [],
        "total_dose": [],
        "mean_mA": [],
        "mA_histories": [],
    }
    
    for ep in tqdm(range(n_episodes), desc=f"Evaluating {policy_name}"):
        obs, info = env.reset(seed=ep)
        done = False
        episode_reward = 0
        
        while not done:
            if hasattr(policy, 'predict'):
                # Stable-baselines3 model
                action, _ = policy.predict(obs, deterministic=deterministic)
            else:
                # Custom policy function
                action = policy(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        results["rewards"].append(episode_reward)
        results["ssim"].append(info.get("ssim", 0))
        results["total_dose"].append(info.get("total_dose", 0))
        results["mean_mA"].append(info.get("mean_mA", 0))
        results["mA_histories"].append(env.mA_history.copy())
    
    # Compute statistics
    metrics = {
        "reward_mean": np.mean(results["rewards"]),
        "reward_std": np.std(results["rewards"]),
        "ssim_mean": np.mean(results["ssim"]),
        "ssim_std": np.std(results["ssim"]),
        "dose_mean": np.mean(results["total_dose"]),
        "dose_std": np.std(results["total_dose"]),
        "mA_mean": np.mean(results["mean_mA"]),
        "mA_std": np.std(results["mA_histories"]),
        "mA_histories": results["mA_histories"],
    }
    
    return metrics


def print_comparison_table(all_metrics: dict):
    """Print a comparison table of all policies."""
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Header
    print(f"\n{'Policy':<20} {'Reward':>12} {'SSIM':>12} {'Dose':>12} {'Mean mA':>12}")
    print("-" * 70)
    
    # Sort by reward
    sorted_policies = sorted(all_metrics.keys(), 
                             key=lambda x: all_metrics[x]["reward_mean"], 
                             reverse=True)
    
    for policy_name in sorted_policies:
        m = all_metrics[policy_name]
        print(f"{policy_name:<20} "
              f"{m['reward_mean']:>8.2f} ± {m['reward_std']:>4.2f} "
              f"{m['ssim_mean']:>8.3f} ± {m['ssim_std']:>4.3f} "
              f"{m['dose_mean']:>8.0f} ± {m['dose_std']:>4.0f} "
              f"{m['mA_mean']:>8.1f}")
    
    print("=" * 80)
    
    # Best policy summary
    best_policy = sorted_policies[0]
    print(f"\n🏆 Best policy: {best_policy}")
    print(f"   Achieves {all_metrics[best_policy]['ssim_mean']:.3f} SSIM "
          f"with {all_metrics[best_policy]['dose_mean']:.0f} total dose")


def plot_comparison(all_metrics: dict, save_path: str = None):
    """Create visualization comparing policies."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    policies = list(all_metrics.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))
    
    # 1. Reward comparison
    ax = axes[0, 0]
    rewards = [all_metrics[p]["reward_mean"] for p in policies]
    reward_stds = [all_metrics[p]["reward_std"] for p in policies]
    bars = ax.bar(policies, rewards, yerr=reward_stds, color=colors, capsize=5)
    ax.set_ylabel("Reward")
    ax.set_title("Total Reward (Higher is Better)")
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Quality vs Dose tradeoff
    ax = axes[0, 1]
    for i, policy in enumerate(policies):
        m = all_metrics[policy]
        ax.scatter(m["dose_mean"], m["ssim_mean"], 
                   color=colors[i], s=200, label=policy, edgecolors='black', linewidths=1.5)
        ax.errorbar(m["dose_mean"], m["ssim_mean"],
                    xerr=m["dose_std"], yerr=m["ssim_std"],
                    color=colors[i], fmt='none', capsize=3)
    ax.set_xlabel("Total Dose (mA·projections)")
    ax.set_ylabel("Image Quality (SSIM)")
    ax.set_title("Quality vs Dose Tradeoff")
    ax.legend(loc='lower right')
    
    # Ideal is upper-left (high quality, low dose)
    ax.annotate('Better →', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, color='green')
    
    # 3. mA profile comparison
    ax = axes[1, 0]
    for i, policy in enumerate(policies):
        histories = all_metrics[policy]["mA_histories"]
        if len(histories) > 0:
            mean_profile = np.mean(histories, axis=0)
            ax.plot(mean_profile, color=colors[i], label=policy, linewidth=2)
    ax.set_xlabel("Projection Index (Angle)")
    ax.set_ylabel("mA")
    ax.set_title("Mean mA Profile Across Scan")
    ax.legend()
    
    # 4. SSIM comparison
    ax = axes[1, 1]
    ssim_vals = [all_metrics[p]["ssim_mean"] for p in policies]
    ssim_stds = [all_metrics[p]["ssim_std"] for p in policies]
    bars = ax.bar(policies, ssim_vals, yerr=ssim_stds, color=colors, capsize=5)
    ax.set_ylabel("SSIM")
    ax.set_title("Image Quality (Higher is Better)")
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim([min(ssim_vals) * 0.95, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()


def visualize_single_episode(env: CTDoseEnv, policy, title: str = "Episode"):
    """Visualize a single episode with the given policy."""
    
    obs, info = env.reset(seed=0)
    done = False
    
    while not done:
        if hasattr(policy, 'predict'):
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = policy(obs)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    # Render final state
    env.render_mode = "human"
    print(f"\n{title}")
    print(f"SSIM: {info['ssim']:.4f}")
    print(f"Total Dose: {info['total_dose']}")
    print(f"Mean mA: {info['mean_mA']:.1f}")
    env.render()


def main():
    parser = argparse.ArgumentParser(description="Evaluate CT Dose RL Agent")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model (optional)"
    )
    parser.add_argument(
        "--phantom",
        type=str,
        default="shepp_logan",
        choices=["shepp_logan", "ellipses", "chest"],
        help="Phantom type"
    )
    parser.add_argument(
        "--n_angles",
        type=int,
        default=60,
        help="Number of projection angles"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.png",
        help="Output path for comparison plot"
    )
    
    args = parser.parse_args()
    
    # Create environment
    config = ScanConfig(n_angles=args.n_angles)
    env = CTDoseEnv(config=config, phantom_type=args.phantom)
    
    # Dictionary to store all metrics
    all_metrics = {}
    
    # Evaluate fixed mA policies
    for mA in [50, 150, 250]:
        policy = fixed_mA_policy(mA)
        metrics = evaluate_policy(env, policy, args.n_episodes, 
                                  policy_name=f"Fixed {mA} mA")
        all_metrics[f"Fixed {mA}"] = metrics
    
    # Evaluate oracle policy
    policy = oracle_policy(env)
    metrics = evaluate_policy(env, policy, args.n_episodes, 
                              policy_name="Oracle")
    all_metrics["Oracle"] = metrics
    
    # Evaluate random policy
    policy = random_policy(env)
    metrics = evaluate_policy(env, policy, args.n_episodes,
                              policy_name="Random")
    all_metrics["Random"] = metrics
    
    # Evaluate trained model if provided
    if args.model_path:
        print(f"\nLoading trained model from {args.model_path}")
        try:
            model = PPO.load(args.model_path, env=env)
            algo = "PPO"
        except:
            try:
                model = DQN.load(args.model_path, env=env)
                algo = "DQN"
            except:
                model = SAC.load(args.model_path, env=env)
                algo = "SAC"
        
        metrics = evaluate_policy(env, model, args.n_episodes,
                                  policy_name=f"Trained ({algo})")
        all_metrics[f"Trained ({algo})"] = metrics
    
    # Print results
    print_comparison_table(all_metrics)
    
    # Plot comparison
    plot_comparison(all_metrics, args.output)
    
    env.close()


if __name__ == "__main__":
    main()
