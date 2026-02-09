# CT Dose Optimization with Reinforcement Learning

**Teaching AI to reduce radiation in CT scans while maintaining image quality.**

## 🎯 Project Overview

This project uses **Reinforcement Learning** to optimize tube current (mA) modulation during CT scans. The AI learns to use less radiation in thin body regions while maintaining image quality.

### Clinical Motivation

- CT scans expose patients to ionizing radiation
- Higher mA = better image but more dose
- Different projection angles pass through different body thicknesses
- **Insight**: Use high mA only where needed (thick regions)

### Why RL?

| Property | This Project |
|----------|-------------|
| Sequential decisions | ✅ Choose mA at each of 60 projection angles |
| Delayed reward | ✅ Image quality only known after reconstruction |
| State-dependent policy | ✅ Lateral views need more mA than AP views |
| No trivial solution | ✅ Optimal mA varies with anatomy |

## 📁 Project Structure

```
ct_dose_rl_project/
├── envs/
│   ├── __init__.py
│   └── ct_dose_env.py         # Gymnasium environment
├── train.py                    # Training script (PPO/DQN)
├── evaluate.py                 # Compare against baselines
├── test_env.py                 # Verify setup
├── CT_Dose_RL_Explained.ipynb  # Visual guide for lay audience
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Setup

```bash
cd ct_dose_rl_project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Test Environment

```bash
python test_env.py
```

### 3. Train Agent

```bash
# Basic training
python train.py --timesteps 50000

# With options
python train.py \
    --algorithm PPO \
    --phantom chest \
    --n_angles 60 \
    --timesteps 100000 \
    --dose_weight 0.5
```

### 4. Evaluate

```bash
# Compare trained agent against baselines
python evaluate.py --model_path outputs/PPO_xxx/best_model/best_model.zip

# Or just compare baselines (no trained model needed)
python evaluate.py --n_episodes 50
```

### 5. Educational Notebook

```bash
jupyter notebook CT_Dose_RL_Explained.ipynb
```

## 🎮 MDP Formulation

### State (Observation)
```
[scan_progress, body_thickness, dose_used, last_mA]
   0 to 1         0 to 1         0 to 1    0 to 1
```

### Actions
| Action | mA Value |
|--------|----------|
| 0 | 50 mA (lowest dose) |
| 1 | 100 mA |
| 2 | 150 mA |
| 3 | 200 mA |
| 4 | 250 mA (highest dose) |

### Reward
```python
# Computed at end of scan (delayed reward)
reward = quality_weight × SSIM - dose_weight × normalized_dose
```

- **SSIM**: Structural Similarity Index (0 to 1, higher = better quality)
- **Dose**: Total mA × projections (lower = safer)

### Episode
- 60 projections (one full CT rotation)
- Agent chooses mA for each projection
- Reward computed after reconstruction

## 📊 Expected Results

| Policy | SSIM | Total Dose | Notes |
|--------|------|------------|-------|
| Fixed 250 mA | ~0.95 | 15,000 | Best quality, highest dose |
| Fixed 50 mA | ~0.75 | 3,000 | Grainy image, lowest dose |
| **Trained RL** | ~0.92 | ~9,000 | **Best balance** |
| Oracle | ~0.93 | ~8,500 | Theoretical optimal |

**Result: ~40% dose reduction with minimal quality loss!**

## 🔧 Configuration Options

### ScanConfig Parameters

```python
from envs.ct_dose_env import ScanConfig

config = ScanConfig(
    n_angles=60,           # Projections per scan
    image_size=256,        # Phantom resolution
    mA_levels=(50, 100, 150, 200, 250),
    noise_scale=0.15,      # Base noise level
    dose_weight=0.5,       # Penalty for dose
    quality_weight=1.0,    # Reward for quality
)
```

### Phantom Types

| Type | Description |
|------|-------------|
| `shepp_logan` | Standard brain phantom (default) |
| `ellipses` | Simple ellipse (fast) |
| `chest` | Chest-like with lungs, spine, nodule |

## 📈 Training Tips

1. **Start small**: Use `--n_angles 30` for faster iteration
2. **Balance weights**: Adjust `--dose_weight` to control quality/dose tradeoff
3. **Monitor TensorBoard**: 
   ```bash
   tensorboard --logdir outputs/
   ```
4. **Use chest phantom**: Most clinically relevant anatomy

## 🧪 Experiments to Try

1. **Vary dose_weight**: How does reward weighting affect learned policy?
2. **Compare algorithms**: PPO vs DQN vs SAC
3. **Different phantoms**: Does policy transfer across anatomies?
4. **More angles**: Does finer angular sampling change the policy?

## 📚 Background Reading

- **CT Physics**: Bushberg et al., "The Essential Physics of Medical Imaging"
- **Tube Current Modulation**: Söderberg & Gunnarsson, "Automatic exposure control in CT"
- **RL Basics**: Sutton & Barto, "Reinforcement Learning: An Introduction"
- **Stable-Baselines3**: [Documentation](https://stable-baselines3.readthedocs.io/)

## 🏗️ Extending the Project

### Ideas for Enhancement

1. **3D Extension**: Stack multiple slices for volumetric scanning
2. **Real CT Data**: Replace phantom with real CT geometries
3. **Multi-objective RL**: Pareto optimization of quality vs dose
4. **Continuous Actions**: Use SAC with continuous mA output
5. **Patient-specific**: Adapt policy based on scout scan

### Adding New Phantoms

```python
def _create_my_phantom(self, size):
    phantom = np.zeros((size, size))
    # Add your structures here
    return phantom
```

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{ct_dose_rl,
  title={CT Dose Optimization with Reinforcement Learning},
  author={Your Name},
  year={2025},
  note={CAS Extended Intelligence Module Project}
}
```

## 📄 License

Educational use. CT simulation based on scikit-image.

---

## Quick Reference

```bash
# Setup
pip install -r requirements.txt

# Test
python test_env.py

# Train
python train.py --timesteps 100000

# Evaluate
python evaluate.py --model_path outputs/xxx/best_model/best_model.zip

# Learn
jupyter notebook CT_Dose_RL_Explained.ipynb
```
