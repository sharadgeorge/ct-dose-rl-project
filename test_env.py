"""
Test script to verify CT Dose environment works correctly.

Usage:
    python test_env.py
"""

import numpy as np
import matplotlib.pyplot as plt

from envs.ct_dose_env import CTDoseEnv, ScanConfig


def test_basic_functionality():
    """Test that the environment loads and runs."""
    print("=" * 60)
    print("TEST 1: Basic Functionality")
    print("=" * 60)
    
    config = ScanConfig(n_angles=30)  # Small for quick test
    env = CTDoseEnv(config=config, phantom_type="shepp_logan")
    
    print(f"✓ Environment created")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"✓ Environment reset")
    print(f"  Initial observation: {obs}")
    print(f"  Info: {info}")
    
    # Run episode with random actions
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    
    print(f"✓ Episode completed")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final info: {info}")
    
    env.close()
    print("✓ Environment closed\n")
    return True


def test_all_phantoms():
    """Test that all phantom types work."""
    print("=" * 60)
    print("TEST 2: All Phantom Types")
    print("=" * 60)
    
    phantoms = ["shepp_logan", "ellipses", "chest"]
    
    for phantom_type in phantoms:
        config = ScanConfig(n_angles=20)
        env = CTDoseEnv(config=config, phantom_type=phantom_type)
        obs, _ = env.reset()
        
        # Run a few steps
        for _ in range(20):
            obs, reward, done, _, _ = env.step(env.action_space.sample())
            if done:
                break
        
        print(f"✓ {phantom_type}: OK")
        env.close()
    
    print()
    return True


def test_observation_bounds():
    """Test that observations stay within bounds."""
    print("=" * 60)
    print("TEST 3: Observation Bounds")
    print("=" * 60)
    
    config = ScanConfig(n_angles=60)
    env = CTDoseEnv(config=config)
    
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # Check bounds
            assert np.all(obs >= 0), f"Observation below 0: {obs}"
            assert np.all(obs <= 1), f"Observation above 1: {obs}"
            
            action = env.action_space.sample()
            obs, _, done, _, _ = env.step(action)
    
    print("✓ All observations within [0, 1] bounds")
    env.close()
    print()
    return True


def test_determinism():
    """Test that same seed gives same results."""
    print("=" * 60)
    print("TEST 4: Determinism")
    print("=" * 60)
    
    config = ScanConfig(n_angles=30)
    
    results = []
    for trial in range(2):
        env = CTDoseEnv(config=config)
        obs, _ = env.reset(seed=123)
        
        total_reward = 0
        done = False
        np.random.seed(456)  # Fix random actions
        
        while not done:
            action = np.random.randint(0, 5)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
        
        results.append((total_reward, info.get('ssim', 0)))
        env.close()
    
    if abs(results[0][0] - results[1][0]) < 0.001:
        print(f"✓ Deterministic: reward = {results[0][0]:.4f}")
    else:
        print(f"✗ Non-deterministic: {results[0][0]:.4f} vs {results[1][0]:.4f}")
        return False
    
    print()
    return True


def test_reward_structure():
    """Test that reward makes sense."""
    print("=" * 60)
    print("TEST 5: Reward Structure")
    print("=" * 60)
    
    config = ScanConfig(n_angles=30)
    
    # Test 1: All low mA should give lower reward (poor quality)
    env = CTDoseEnv(config=config)
    env.reset(seed=42)
    for _ in range(30):
        _, reward_low, done, _, info_low = env.step(0)  # 50 mA
    env.close()
    
    # Test 2: All high mA should give higher reward (good quality)
    env = CTDoseEnv(config=config)
    env.reset(seed=42)
    for _ in range(30):
        _, reward_high, done, _, info_high = env.step(4)  # 250 mA
    env.close()
    
    print(f"  Low mA (50):  SSIM={info_low['ssim']:.4f}, Dose={info_low['total_dose']}")
    print(f"  High mA (250): SSIM={info_high['ssim']:.4f}, Dose={info_high['total_dose']}")
    
    # High mA should have better quality
    if info_high['ssim'] > info_low['ssim']:
        print("✓ Higher mA gives better image quality")
    else:
        print("✗ Quality relationship unexpected")
        return False
    
    # Low mA should have lower dose
    if info_low['total_dose'] < info_high['total_dose']:
        print("✓ Lower mA gives lower dose")
    else:
        print("✗ Dose relationship unexpected")
        return False
    
    print()
    return True


def visualize_episode():
    """Run and visualize a sample episode."""
    print("=" * 60)
    print("TEST 6: Visualization")
    print("=" * 60)
    
    config = ScanConfig(n_angles=60)
    env = CTDoseEnv(config=config, phantom_type="chest", render_mode="human")
    
    obs, _ = env.reset(seed=42)
    
    # Run with a simple policy: more mA when thicker
    for i in range(60):
        thickness = obs[1]  # Body thickness
        if thickness > 0.7:
            action = 4  # 250 mA
        elif thickness > 0.5:
            action = 3  # 200 mA
        elif thickness > 0.3:
            action = 2  # 150 mA
        else:
            action = 1  # 100 mA
        
        obs, reward, done, _, info = env.step(action)
    
    print(f"✓ Episode complete")
    print(f"  SSIM: {info['ssim']:.4f}")
    print(f"  Total Dose: {info['total_dose']}")
    print(f"  Mean mA: {info['mean_mA']:.1f}")
    
    # Render final state
    env.render()
    
    env.close()
    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CT DOSE ENVIRONMENT TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("All Phantoms", test_all_phantoms),
        ("Observation Bounds", test_observation_bounds),
        ("Determinism", test_determinism),
        ("Reward Structure", test_reward_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} FAILED with error: {e}")
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Environment is ready.")
        
        # Ask about visualization
        try:
            response = input("\nRun visualization test? (y/n): ")
            if response.lower() == 'y':
                visualize_episode()
        except:
            pass
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    main()
