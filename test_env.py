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


def test_budget_mode_basic():
    """Test that budget mode tracks dose correctly without clamping."""
    print("=" * 60)
    print("TEST 6: Budget Mode Basic")
    print("=" * 60)

    # Budget = 30 angles * 150 mA = 4500 — exactly enough for all 150 mA
    config = ScanConfig(n_angles=30, dose_budget=4500.0, step_dose_penalty=0.0)
    env = CTDoseEnv(config=config, phantom_type="shepp_logan")
    obs, _ = env.reset(seed=42)

    for _ in range(30):
        obs, reward, done, _, info = env.step(2)  # action 2 = 150 mA

    assert done, "Episode should be done after 30 angles"
    assert info["n_clamped"] == 0, f"Expected 0 clamped, got {info['n_clamped']}"
    assert abs(info["total_dose"] - 4500) < 1e-6, f"Expected dose 4500, got {info['total_dose']}"
    assert abs(info["budget_remaining"]) < 1e-6, f"Expected 0 budget remaining, got {info['budget_remaining']}"

    print(f"  Total dose: {info['total_dose']}, Budget remaining: {info['budget_remaining']}")
    print(f"  Clamped: {info['n_clamped']}")
    print("  PASS: budget exactly exhausted, 0 clamps")
    env.close()
    print()
    return True


def test_budget_clamping():
    """Test that actions are clamped when budget is exceeded."""
    print("=" * 60)
    print("TEST 7: Budget Clamping")
    print("=" * 60)

    # Budget = 600, 10 angles. Spend 250+250 = 500, leaving 100.
    config = ScanConfig(n_angles=10, dose_budget=600.0, step_dose_penalty=0.0)
    env = CTDoseEnv(config=config, phantom_type="shepp_logan")
    obs, _ = env.reset(seed=42)

    # Step 1: 250 mA (action 4)
    _, _, _, _, info1 = env.step(4)
    assert not info1["was_clamped"], "Step 1 should not be clamped"
    assert abs(info1["budget_remaining"] - 350) < 1e-6

    # Step 2: 250 mA (action 4)
    _, _, _, _, info2 = env.step(4)
    assert not info2["was_clamped"], "Step 2 should not be clamped"
    assert abs(info2["budget_remaining"] - 100) < 1e-6

    # Step 3: request 250 mA but only 100 remaining — should clamp to 100
    _, _, _, _, info3 = env.step(4)
    assert info3["was_clamped"], "Step 3 should be clamped (250 > 100 remaining)"
    assert info3["n_clamped"] == 1
    assert abs(info3["budget_remaining"]) < 1e-6, f"Expected 0 remaining, got {info3['budget_remaining']}"

    # Step 4: request 200 mA but 0 remaining — forced to min mA (50)
    _, _, _, _, info4 = env.step(3)
    assert info4["was_clamped"], "Step 4 should be clamped (0 budget remaining)"
    assert info4["n_clamped"] == 2

    print(f"  After 250+250: budget_remaining={info2['budget_remaining']}")
    print(f"  After clamped 250->100: budget_remaining={info3['budget_remaining']}")
    print(f"  After forced to min mA: n_clamped={info4['n_clamped']}")
    print("  PASS: clamping works correctly")
    env.close()
    print()
    return True


def test_budget_observation_decreases():
    """Test that obs[5] (budget fraction) starts at 1.0 and strictly decreases."""
    print("=" * 60)
    print("TEST 8: Budget Observation Decreases")
    print("=" * 60)

    config = ScanConfig(n_angles=10, dose_budget=2000.0, step_dose_penalty=0.0)
    env = CTDoseEnv(config=config, phantom_type="shepp_logan")
    obs, _ = env.reset(seed=42)

    assert abs(obs[5] - 1.0) < 1e-6, f"Initial obs[5] should be 1.0, got {obs[5]}"

    prev_budget_obs = obs[5]
    for i in range(10):
        obs, _, done, _, _ = env.step(2)  # 150 mA each step
        if not done:
            assert obs[5] < prev_budget_obs, (
                f"Step {i+1}: obs[5]={obs[5]} should be < prev {prev_budget_obs}"
            )
            prev_budget_obs = obs[5]

    print(f"  obs[5] decreased from 1.0 to {prev_budget_obs:.4f}")
    print("  PASS: budget observation strictly decreasing")
    env.close()
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
        ("Budget Mode Basic", test_budget_mode_basic),
        ("Budget Clamping", test_budget_clamping),
        ("Budget Observation Decreases", test_budget_observation_decreases),
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
