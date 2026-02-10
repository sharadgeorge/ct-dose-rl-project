"""
CT Dose Optimization Environment

An RL environment where an agent controls tube current (mA) during a CT scan
to minimize radiation dose while maintaining image quality.

Clinical Motivation:
    CT scans deliver ionizing radiation. Higher mA = better image quality but more dose.
    Different projection angles need different mA (lateral views through more tissue).
    An RL agent can learn to adaptively modulate mA based on patient anatomy.

Physics:
    - Photon count N ∝ mA × exp(-∫μdx)
    - Noise σ ∝ 1/√N (Poisson statistics)
    - Higher mA → more photons → less noise → better image
    - But higher mA → more radiation dose to patient

Author: [Your Name]
Date: 2025
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import IntEnum


@dataclass
class ScanConfig:
    """Configuration for CT scan simulation."""
    n_angles: int = 60           # Number of projection angles
    image_size: int = 256        # Phantom/reconstruction size
    mA_levels: Tuple = (50, 100, 150, 200, 250)  # Available tube currents
    noise_scale: float = 0.5     # Global noise amplitude
    noise_exponent: float = 0.08 # Exponential noise strength (thick paths get more noise)
    het_noise_scale: float = 0.5 # Heterogeneity noise at density transitions
    reference_mA: float = 250    # Reference mA for noise scaling
    step_dose_penalty: float = 0.02  # Per-step mA cost
    dose_weight: float = 0.0     # Weight for dose penalty in reward (now per-step)
    quality_weight: float = 1.0  # Weight for image quality in reward
    ssim_percentile: float = 5.0 # Percentile of local SSIM map (lower = more worst-region focus)
    dose_budget: Optional[float] = None  # Fixed dose budget (None = legacy unlimited mode)


class CTDoseEnv(gym.Env):
    """
    Gymnasium environment for CT dose optimization.
    
    The agent controls tube current (mA) at each projection angle during a CT scan.
    Goal: Minimize total radiation dose while maintaining diagnostic image quality.
    
    Observation Space:
        - scan_progress: How far through the scan (0 to 1)
        - body_thickness: Peak attenuation at current angle (normalized)
        - mean_tissue: Mean attenuation of body region (tissue density)
        - std_tissue: Std of body region attenuation (tissue heterogeneity)
        - width_frac: Fraction of detector elements hitting body (geometric width)
        - dose_used: Cumulative dose so far (normalized)
        - last_mA: Previous mA setting (normalized)
        
    Action Space:
        Discrete(5): Choose mA from [50, 100, 150, 200, 250]
        
    Reward:
        - Delayed until end of scan
        - R = quality_weight × SSIM - dose_weight × normalized_dose
        - SSIM measures structural similarity to noise-free reference
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(
        self,
        config: Optional[ScanConfig] = None,
        render_mode: Optional[str] = None,
        phantom_type: str = "shepp_logan",  # or "ellipses" or "chest"
    ):
        """
        Initialize the CT dose optimization environment.
        
        Args:
            config: Scan configuration parameters
            render_mode: Rendering mode ("human" or "rgb_array")
            phantom_type: Type of digital phantom to use
        """
        super().__init__()
        
        self.config = config or ScanConfig()
        self.render_mode = render_mode
        self.phantom_type = phantom_type
        
        # Create phantom
        self.phantom = self._create_phantom()
        
        # Precompute angles
        self.angles = np.linspace(0, 180, self.config.n_angles, endpoint=False)
        
        # Precompute clean sinogram (for thickness estimates and reference)
        self._precompute_clean_sinogram()
        
        # Action space: discrete mA levels
        self.action_space = spaces.Discrete(len(self.config.mA_levels))
        
        # Observation space: [progress, thickness, mean_tissue, std_tissue, width_frac, dose_used, last_mA]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )
        
        # Episode state
        self.current_angle_idx = 0
        self.sinogram = []
        self.mA_history = []
        self.total_dose = 0
        self.last_mA = self.config.mA_levels[2]  # Start with medium mA
        
        # For visualization
        self.reconstruction = None
        
    def _create_phantom(self) -> np.ndarray:
        """Create a digital phantom for simulation."""
        size = self.config.image_size
        
        if self.phantom_type == "shepp_logan":
            from skimage.data import shepp_logan_phantom
            from skimage.transform import resize
            phantom = shepp_logan_phantom()
            phantom = resize(phantom, (size, size), anti_aliasing=True)
            
        elif self.phantom_type == "ellipses":
            # Simple ellipse phantom (faster, good for testing)
            phantom = self._create_ellipse_phantom(size)
            
        elif self.phantom_type == "chest":
            # Simplified chest-like phantom
            phantom = self._create_chest_phantom(size)
        else:
            raise ValueError(f"Unknown phantom type: {self.phantom_type}")
            
        return phantom.astype(np.float32)
    
    def _create_ellipse_phantom(self, size: int) -> np.ndarray:
        """Create a simple ellipse phantom."""
        phantom = np.zeros((size, size), dtype=np.float32)
        y, x = np.ogrid[:size, :size]
        center = size // 2
        
        # Outer body ellipse
        a, b = size * 0.4, size * 0.3  # Semi-axes
        mask = ((x - center) / a) ** 2 + ((y - center) / b) ** 2 <= 1
        phantom[mask] = 0.5
        
        # Inner structure (like spine)
        a2, b2 = size * 0.08, size * 0.15
        mask2 = ((x - center) / a2) ** 2 + ((y - center) / b2) ** 2 <= 1
        phantom[mask2] = 1.0
        
        # Small dense region (like nodule/lesion)
        cx, cy = center + size // 6, center - size // 8
        r = size * 0.03
        mask3 = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        phantom[mask3] = 0.8
        
        return phantom
    
    def _create_chest_phantom(self, size: int) -> np.ndarray:
        """Create a simplified chest-like phantom with asymmetric thickness."""
        phantom = np.zeros((size, size), dtype=np.float32)
        y, x = np.ogrid[:size, :size]
        center = size // 2
        
        # Body outline (elliptical, wider than tall)
        a_body, b_body = size * 0.42, size * 0.32
        body_mask = ((x - center) / a_body) ** 2 + ((y - center) / b_body) ** 2 <= 1
        phantom[body_mask] = 0.3  # Soft tissue
        
        # Left lung
        a_lung, b_lung = size * 0.15, size * 0.22
        lung_cx = center - size // 5
        lung_mask = ((x - lung_cx) / a_lung) ** 2 + ((y - center) / b_lung) ** 2 <= 1
        phantom[lung_mask] = 0.05  # Air
        
        # Right lung
        lung_cx_r = center + size // 5
        lung_mask_r = ((x - lung_cx_r) / a_lung) ** 2 + ((y - center) / b_lung) ** 2 <= 1
        phantom[lung_mask_r] = 0.05  # Air
        
        # Spine (posterior — high y in image coordinates)
        spine_a, spine_b = size * 0.05, size * 0.12
        spine_cy = center + int(size * 0.18)
        spine_mask = ((x - center) / spine_a) ** 2 + ((y - spine_cy) / spine_b) ** 2 <= 1
        phantom[spine_mask] = 1.0  # Bone

        # Heart (left of center, anterior — low y)
        heart_cx = center - int(size * 0.12)
        heart_cy = center - int(size * 0.05)
        heart_r = size * 0.1
        heart_mask = (x - heart_cx) ** 2 + (y - heart_cy) ** 2 <= heart_r ** 2
        phantom[heart_mask] = 0.5  # Dense tissue
        
        # Small nodule in right lung
        nodule_cx = center + size // 4
        nodule_cy = center - size // 10
        nodule_r = size * 0.02
        nodule_mask = (x - nodule_cx) ** 2 + (y - nodule_cy) ** 2 <= nodule_r ** 2
        phantom[nodule_mask] = 0.4  # Soft tissue density
        
        return phantom
    
    def _precompute_clean_sinogram(self):
        """Precompute noise-free sinogram for reference and thickness/density estimation."""
        from skimage.transform import radon
        self.clean_sinogram = radon(self.phantom, theta=self.angles, circle=True)

        # Profile statistics at each angle, computed over body region only
        # (detector bins where the ray actually hits tissue, not zero-padding)
        n_angles = self.clean_sinogram.shape[1]
        threshold = 0.01 * np.max(self.clean_sinogram)  # 1% of max to filter padding

        # Max: peak attenuation ("thickness" — existing feature)
        self.path_lengths = np.max(self.clean_sinogram, axis=0)
        self.path_lengths_normalized = self.path_lengths / np.max(self.path_lengths)

        # Per-angle statistics over body region only
        self.profile_mean_tissue = np.zeros(n_angles)
        self.profile_std_tissue = np.zeros(n_angles)
        self.profile_width_frac = np.zeros(n_angles)

        for i in range(n_angles):
            profile = self.clean_sinogram[:, i]
            body_mask = profile > threshold
            n_body = np.sum(body_mask)
            if n_body > 0:
                body_vals = profile[body_mask]
                self.profile_mean_tissue[i] = np.mean(body_vals)
                self.profile_std_tissue[i] = np.std(body_vals)
                self.profile_width_frac[i] = n_body / len(profile)

        # Normalize all to [0, 1]
        max_val = np.max(self.profile_mean_tissue)
        self.profile_mean_normalized = self.profile_mean_tissue / max_val if max_val > 0 else self.profile_mean_tissue

        max_val = np.max(self.profile_std_tissue)
        self.profile_std_normalized = self.profile_std_tissue / max_val if max_val > 0 else self.profile_std_tissue

        max_val = np.max(self.profile_width_frac)
        self.profile_width_normalized = self.profile_width_frac / max_val if max_val > 0 else self.profile_width_frac
    
    def _add_noise(self, projection: np.ndarray, mA: float) -> np.ndarray:
        """
        Add realistic CT noise to a projection.

        Two noise components:
        1. Exponential noise: thick paths get disproportionately more noise
           σ ∝ √(exp(exponent * |projection|) / mA)
        2. Heterogeneity noise: sharp density transitions (bone/air, tissue/air)
           get extra noise, modeling beam hardening and photon starvation artifacts.
           σ ∝ |gradient(projection)| / √mA
        """
        # Component 1: exponential (thickness-dependent) noise
        exponent = np.clip(self.config.noise_exponent * np.abs(projection), 0, 20)
        noise = (self.config.noise_scale
                 * np.sqrt(np.exp(exponent) / mA)
                 * np.random.randn(*projection.shape))

        # Component 2: heterogeneity noise at density transitions
        if self.config.het_noise_scale > 0:
            gradient = np.abs(np.diff(projection, prepend=projection[0]))
            het_noise = (self.config.het_noise_scale
                         * gradient / np.sqrt(mA)
                         * np.random.randn(*projection.shape))
            noise = noise + het_noise

        return projection + noise
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (7-dim: progress, profile stats, dose, last_mA)."""
        # Progress through scan (0 to 1)
        progress = self.current_angle_idx / self.config.n_angles

        # Profile statistics at current angle
        if self.current_angle_idx < self.config.n_angles:
            idx = self.current_angle_idx
            thickness = self.path_lengths_normalized[idx]        # peak attenuation
            mean_tissue = self.profile_mean_normalized[idx]      # avg tissue density
            std_tissue = self.profile_std_normalized[idx]        # tissue heterogeneity
            width_frac = self.profile_width_normalized[idx]      # geometric body width
        else:
            thickness = 0.0
            mean_tissue = 0.0
            std_tissue = 0.0
            width_frac = 0.0

        # Normalized dose / budget observation
        if self.config.dose_budget is not None:
            dose_normalized = max(0.0, self.budget_remaining / self.config.dose_budget)
        else:
            max_possible_dose = self.config.n_angles * max(self.config.mA_levels)
            dose_normalized = self.total_dose / max_possible_dose

        # Last mA used (normalized)
        mA_normalized = (self.last_mA - min(self.config.mA_levels)) / (
            max(self.config.mA_levels) - min(self.config.mA_levels)
        )

        return np.array([progress, thickness, mean_tissue, std_tissue, width_frac,
                         dose_normalized, mA_normalized], dtype=np.float32)
    
    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """Compute final reward based on image quality and dose."""
        from skimage.transform import iradon
        from skimage.metrics import structural_similarity as ssim
        
        # Reconstruct image from noisy sinogram
        sino_array = np.array(self.sinogram).T
        self.reconstruction = iradon(
            sino_array, theta=self.angles, filter_name='ramp',
            circle=True, output_size=self.config.image_size,
        )

        # Reconstruct reference from clean sinogram
        reference = iradon(
            self.clean_sinogram, theta=self.angles, filter_name='ramp',
            circle=True, output_size=self.config.image_size,
        )
        
        # Compute SSIM (structural similarity)
        # Resize if needed to match
        min_size = min(self.reconstruction.shape[0], reference.shape[0])
        recon_crop = self.reconstruction[:min_size, :min_size]
        ref_crop = reference[:min_size, :min_size]
        
        _, ssim_map = ssim(recon_crop, ref_crop, data_range=ref_crop.max() - ref_crop.min(), full=True)
        quality = float(np.percentile(ssim_map, self.config.ssim_percentile))
        global_ssim = float(np.mean(ssim_map))

        # Normalize dose (0 = minimum possible, 1 = maximum possible)
        min_dose = self.config.n_angles * min(self.config.mA_levels)
        max_dose = self.config.n_angles * max(self.config.mA_levels)
        dose_normalized = (self.total_dose - min_dose) / (max_dose - min_dose)

        # Compute reward
        reward = (
            self.config.quality_weight * quality -
            self.config.dose_weight * dose_normalized
        ) * 10  # Scale for easier learning

        metrics = {
            "ssim": quality,
            "global_ssim": global_ssim,
            "total_dose": self.total_dose,
            "dose_normalized": dose_normalized,
            "mean_mA": np.mean(self.mA_history),
            "mA_std": np.std(self.mA_history),
        }
        
        return reward, metrics
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Reset state
        self.current_angle_idx = 0
        self.sinogram = []
        self.mA_history = []
        self.total_dose = 0
        self.last_mA = self.config.mA_levels[2]  # Medium
        self.reconstruction = None

        # Budget mode state
        if self.config.dose_budget is not None:
            self.budget_remaining = self.config.dose_budget
        else:
            self.budget_remaining = float('inf')
        self.n_clamped = 0
        
        obs = self._get_observation()
        info = {
            "n_angles": self.config.n_angles,
            "phantom_type": self.phantom_type,
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step: acquire projection at current angle with chosen mA.
        
        Args:
            action: Index into mA_levels (0-4 for [50, 100, 150, 200, 250])
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get mA for this projection
        mA = self.config.mA_levels[action]

        # Budget clamping: if budget mode and mA exceeds remaining budget
        was_clamped = False
        if self.config.dose_budget is not None and mA > self.budget_remaining:
            # Find largest affordable mA level
            affordable = [m for m in self.config.mA_levels if m <= self.budget_remaining]
            mA = max(affordable) if affordable else min(self.config.mA_levels)
            was_clamped = True
            self.n_clamped += 1

        self.last_mA = mA
        self.mA_history.append(mA)

        # Get clean projection at current angle
        clean_projection = self.clean_sinogram[:, self.current_angle_idx]

        # Add noise based on mA
        noisy_projection = self._add_noise(clean_projection, mA)
        self.sinogram.append(noisy_projection)

        # Accumulate dose and update budget
        self.total_dose += mA
        if self.config.dose_budget is not None:
            self.budget_remaining = max(0.0, self.budget_remaining - mA)

        # Move to next angle
        self.current_angle_idx += 1

        # Check if scan complete
        terminated = self.current_angle_idx >= self.config.n_angles
        truncated = False

        # Per-step dose penalty: penalize high mA choices immediately
        min_mA = min(self.config.mA_levels)
        max_mA = max(self.config.mA_levels)
        step_reward = -self.config.step_dose_penalty * (mA - min_mA) / (max_mA - min_mA)

        if terminated:
            final_reward, metrics = self._compute_reward()
            reward = step_reward + final_reward
            info = metrics
        else:
            reward = step_reward
            info = {
                "current_angle": self.current_angle_idx,
                "mA_used": mA,
                "cumulative_dose": self.total_dose,
            }

        # Add budget info
        if self.config.dose_budget is not None:
            info["budget_remaining"] = self.budget_remaining
            info["n_clamped"] = self.n_clamped
            info["was_clamped"] = was_clamped

        obs = self._get_observation()

        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render current state."""
        if self.render_mode is None:
            return None
            
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Phantom
        axes[0, 0].imshow(self.phantom, cmap='gray')
        axes[0, 0].set_title('Phantom (Ground Truth)')
        axes[0, 0].axis('off')
        
        # Clean sinogram
        axes[0, 1].imshow(self.clean_sinogram, cmap='gray', aspect='auto')
        axes[0, 1].set_title('Clean Sinogram')
        axes[0, 1].set_xlabel('Angle Index')
        axes[0, 1].set_ylabel('Detector Position')
        
        # mA profile
        if len(self.mA_history) > 0:
            axes[0, 2].bar(range(len(self.mA_history)), self.mA_history, color='steelblue')
            axes[0, 2].axhline(y=np.mean(self.mA_history), color='red', linestyle='--', label=f'Mean: {np.mean(self.mA_history):.0f}')
            axes[0, 2].set_xlabel('Angle Index')
            axes[0, 2].set_ylabel('mA')
            axes[0, 2].set_title(f'mA per Projection (Total Dose: {self.total_dose})')
            axes[0, 2].legend()
        else:
            axes[0, 2].text(0.5, 0.5, 'No data yet', ha='center', va='center')
            axes[0, 2].set_title('mA per Projection')
        
        # Body thickness profile
        axes[1, 0].plot(self.angles, self.path_lengths_normalized, 'b-', linewidth=2)
        if self.current_angle_idx < self.config.n_angles:
            axes[1, 0].axvline(x=self.angles[self.current_angle_idx], color='r', linestyle='--')
        axes[1, 0].set_xlabel('Angle (degrees)')
        axes[1, 0].set_ylabel('Relative Thickness')
        axes[1, 0].set_title('Body Thickness vs Angle')
        
        # Noisy sinogram (if acquired)
        if len(self.sinogram) > 0:
            partial_sino = np.array(self.sinogram).T
            axes[1, 1].imshow(partial_sino, cmap='gray', aspect='auto')
            axes[1, 1].set_title(f'Acquired Sinogram ({len(self.sinogram)}/{self.config.n_angles})')
        else:
            axes[1, 1].text(0.5, 0.5, 'No projections yet', ha='center', va='center')
            axes[1, 1].set_title('Acquired Sinogram')
        
        # Reconstruction (if done)
        if self.reconstruction is not None:
            axes[1, 2].imshow(self.reconstruction, cmap='gray')
            axes[1, 2].set_title('Reconstruction')
            axes[1, 2].axis('off')
        else:
            axes[1, 2].text(0.5, 0.5, 'Scan not complete', ha='center', va='center')
            axes[1, 2].set_title('Reconstruction')
        
        plt.tight_layout()
        
        if self.render_mode == "human":
            plt.show()
            return None
        else:  # rgb_array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img
    
    def close(self):
        """Clean up resources."""
        pass


class CTDoseEnvContinuous(CTDoseEnv):
    """
    Continuous action version of CT Dose Environment.
    
    Action space: Box(1,) representing mA in range [50, 250]
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Override action space to continuous
        self.action_space = spaces.Box(
            low=min(self.config.mA_levels),
            high=max(self.config.mA_levels),
            shape=(1,),
            dtype=np.float32
        )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step with continuous mA action."""
        # Clip action to valid range
        min_mA = min(self.config.mA_levels)
        max_mA = max(self.config.mA_levels)
        mA = float(np.clip(action[0], min_mA, max_mA))

        # Budget clamping: if budget mode and mA exceeds remaining budget
        was_clamped = False
        if self.config.dose_budget is not None and mA > self.budget_remaining:
            mA = max(min_mA, self.budget_remaining)
            was_clamped = True
            self.n_clamped += 1

        self.last_mA = mA
        self.mA_history.append(mA)

        # Get clean projection at current angle
        clean_projection = self.clean_sinogram[:, self.current_angle_idx]

        # Add noise based on mA
        noisy_projection = self._add_noise(clean_projection, mA)
        self.sinogram.append(noisy_projection)

        # Accumulate dose and update budget
        self.total_dose += mA
        if self.config.dose_budget is not None:
            self.budget_remaining = max(0.0, self.budget_remaining - mA)

        # Move to next angle
        self.current_angle_idx += 1

        # Check if scan complete
        terminated = self.current_angle_idx >= self.config.n_angles

        # Per-step dose penalty
        step_reward = -self.config.step_dose_penalty * (mA - min_mA) / (max_mA - min_mA)

        if terminated:
            final_reward, metrics = self._compute_reward()
            reward = step_reward + final_reward
            info = metrics
        else:
            reward = step_reward
            info = {
                "current_angle": self.current_angle_idx,
                "mA_used": mA,
                "cumulative_dose": self.total_dose,
            }

        # Add budget info
        if self.config.dose_budget is not None:
            info["budget_remaining"] = self.budget_remaining
            info["n_clamped"] = self.n_clamped
            info["was_clamped"] = was_clamped

        obs = self._get_observation()

        return obs, reward, terminated, False, info


# Register environments
gym.register(
    id='CTDose-v0',
    entry_point='envs.ct_dose_env:CTDoseEnv',
)

gym.register(
    id='CTDose-Continuous-v0',
    entry_point='envs.ct_dose_env:CTDoseEnvContinuous',
)
