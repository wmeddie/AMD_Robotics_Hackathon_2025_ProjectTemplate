#!/usr/bin/env python3
"""
Training Script for Goal-Conditioned SmolVLA

Single GPU training for MI300X. Compatible with LeRobot 0.4.1.
Uses Flow Matching loss for action prediction.

Usage:
    # Train a single skill
    python train_goal_conditioned.py --skill flatten --data_dir ./data/flatten_goal

    # Train with custom parameters
    python train_goal_conditioned.py --skill zigzag --batch_size 16 --num_epochs 50
    
    # Train skills sequentially (one GPU available)
    for skill in flatten zigzag circle stamp place_rock; do
        python train_goal_conditioned.py --skill $skill
    done

Requirements:
    pip install lerobot==0.4.1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
"""

import argparse
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Try importing accelerate for multi-GPU
try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("Warning: accelerate not installed. Using single GPU training.")

# Try importing wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging disabled.")

# Try importing PIL for image loading
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Import our model
from goal_conditioned_smolvla import GoalConditionedSmolVLA, create_goal_conditioned_model


# Default training hyperparameters
DEFAULT_CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "num_epochs": 100,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "save_every_n_steps": 500,
    "eval_every_n_steps": 100,
    "log_every_n_steps": 10,
    "num_inference_steps": 10,
    "action_chunk_size": 10,
    "state_dim": 14,
    "action_dim": 14,
    "image_size": 224,
    "seed": 42,
}


class GoalConditionedDataset(Dataset):
    """
    Dataset for goal-conditioned policy training.
    
    Loads LeRobot-style datasets augmented with goal images.
    """
    
    def __init__(
        self,
        data_dir: str,
        goal_mapping_file: str = "goal_mapping.json",
        image_size: int = 224,
        action_chunk_size: int = 10,
        camera_key: str = "observation.images.top",
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.action_chunk_size = action_chunk_size
        self.camera_key = camera_key
        self.augment = augment
        
        # Load goal mapping
        mapping_path = self.data_dir / goal_mapping_file
        if mapping_path.exists():
            with open(mapping_path) as f:
                self.goal_mapping = json.load(f)
        else:
            raise FileNotFoundError(f"Goal mapping not found: {mapping_path}")
        
        # Build sample index
        self.samples = self._build_sample_index()
        print(f"Loaded dataset with {len(self.samples)} samples from {data_dir}")
    
    def _build_sample_index(self) -> list:
        """Build index of all training samples."""
        samples = []
        
        episodes = self.goal_mapping.get("episodes", {})
        for ep_idx, ep_info in episodes.items():
            start_idx = ep_info["start_idx"]
            end_idx = ep_info["end_idx"]
            goal_path = self.data_dir / ep_info["goal_path"]
            
            # Each timestep in the episode (except last action_chunk_size) is a sample
            for t in range(start_idx, end_idx - self.action_chunk_size):
                samples.append({
                    "episode": int(ep_idx),
                    "timestep": t,
                    "goal_path": goal_path,
                    "start_idx": start_idx,
                    "end_idx": end_idx
                })
        
        return samples
    
    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess an image."""
        if PIL_AVAILABLE:
            img = Image.open(path).convert("RGB")
            img = img.resize((self.image_size, self.image_size))
            img = np.array(img) / 255.0
        else:
            import cv2
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img / 255.0
        
        # Convert to tensor [C, H, W]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        
        return img
    
    def _load_episode_data(self, episode: int) -> Dict:
        """Load data for an episode from parquet or other format."""
        # Try loading from parquet
        parquet_dir = self.data_dir / "data"
        if parquet_dir.exists():
            try:
                import pandas as pd
                for pq_file in parquet_dir.glob("*.parquet"):
                    df = pd.read_parquet(pq_file)
                    # Filter to this episode
                    if "episode_index" in df.columns:
                        ep_data = df[df["episode_index"] == episode]
                        return ep_data.to_dict("list")
            except ImportError:
                pass
        
        # Fallback: return empty dict (data will be loaded per-sample)
        return {}
    
    def _get_observation_image(self, episode: int, timestep: int) -> torch.Tensor:
        """Get observation image for a specific timestep."""
        # Try loading from images directory
        images_dir = self.data_dir / "images" / self.camera_key.replace(".", "_")
        
        patterns = [
            f"episode_{episode:06d}_frame_{timestep:06d}.png",
            f"episode_{episode:06d}_{timestep:06d}.png",
            f"{episode:06d}_{timestep:06d}.png",
            f"frame_{timestep:06d}.png",
        ]
        
        for pattern in patterns:
            img_path = images_dir / pattern
            if img_path.exists():
                return self._load_image(img_path)
        
        # Return random image if not found (for testing)
        return torch.randn(3, self.image_size, self.image_size)
    
    def _get_state(self, episode: int, timestep: int) -> torch.Tensor:
        """Get proprioceptive state for a timestep."""
        # Try loading from data
        state_file = self.data_dir / "states" / f"episode_{episode:06d}.npy"
        if state_file.exists():
            states = np.load(state_file)
            if timestep < len(states):
                return torch.from_numpy(states[timestep]).float()
        
        # Return zeros if not found
        return torch.zeros(14)
    
    def _get_actions(self, episode: int, timestep: int) -> torch.Tensor:
        """Get action chunk starting at timestep."""
        action_file = self.data_dir / "actions" / f"episode_{episode:06d}.npy"
        if action_file.exists():
            actions = np.load(action_file)
            end_t = min(timestep + self.action_chunk_size, len(actions))
            action_chunk = actions[timestep:end_t]
            
            # Pad if necessary
            if len(action_chunk) < self.action_chunk_size:
                padding = np.zeros((self.action_chunk_size - len(action_chunk), actions.shape[1]))
                action_chunk = np.concatenate([action_chunk, padding], axis=0)
            
            return torch.from_numpy(action_chunk).float()
        
        # Return zeros if not found
        return torch.zeros(self.action_chunk_size, 14)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        episode = sample["episode"]
        timestep = sample["timestep"]
        
        # Load data
        observation = self._get_observation_image(episode, timestep)
        goal = self._load_image(sample["goal_path"])
        state = self._get_state(episode, timestep)
        actions = self._get_actions(episode, timestep)
        
        # Data augmentation
        if self.augment and random.random() > 0.5:
            # Random horizontal flip (flip both observation and goal consistently)
            observation = torch.flip(observation, dims=[2])
            goal = torch.flip(goal, dims=[2])
            # Flip x-coordinates in actions if applicable
            # actions[:, 0] = -actions[:, 0]  # Uncomment if actions have x-coordinate
        
        return {
            "observation": observation,
            "goal": goal,
            "state": state,
            "actions": actions,
            "episode": episode,
            "timestep": timestep
        }


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching loss for action prediction.
    
    Implements conditional flow matching where the model learns to predict
    the velocity field that transforms noise to the target action distribution.
    """
    
    def __init__(self, sigma_min: float = 0.001):
        super().__init__()
        self.sigma_min = sigma_min
    
    def forward(
        self,
        model: GoalConditionedSmolVLA,
        observation: torch.Tensor,
        goal: torch.Tensor,
        state: torch.Tensor,
        target_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            model: Goal-conditioned model
            observation: Observation images [B, C, H, W]
            goal: Goal images [B, C, H, W]
            state: Proprioceptive state [B, state_dim]
            target_actions: Target action chunks [B, T, action_dim]
            
        Returns:
            Flow matching loss
        """
        batch_size = observation.shape[0]
        device = observation.device
        
        # Flatten actions for flow matching
        target_flat = target_actions.view(batch_size, -1)
        
        # Sample random timestep t ~ U(0, 1)
        t = torch.rand(batch_size, device=device)
        
        # Sample noise
        noise = torch.randn_like(target_flat)
        
        # Interpolate between noise and target (linear interpolation for OT path)
        # x_t = (1 - t) * noise + t * target
        t_expanded = t.unsqueeze(-1)
        x_t = (1 - t_expanded) * noise + t_expanded * target_flat
        
        # Target velocity is the direction from noise to target
        # v_target = target - noise
        target_velocity = target_flat - noise
        
        # Predict velocity using model
        predicted_actions = model(
            observation, goal, state,
            timestep=t,
            noise=x_t
        )
        predicted_velocity = predicted_actions.view(batch_size, -1)
        
        # MSE loss between predicted and target velocity
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        return loss


class Trainer:
    """Trainer for goal-conditioned SmolVLA."""
    
    def __init__(
        self,
        model: GoalConditionedSmolVLA,
        train_dataset: GoalConditionedDataset,
        val_dataset: Optional[GoalConditionedDataset] = None,
        config: Dict = None,
        output_dir: str = "./checkpoints",
        skill: str = "unknown",
        use_wandb: bool = True
    ):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.output_dir = Path(output_dir) / skill
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skill = skill
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Set seed
        if ACCELERATE_AVAILABLE:
            set_seed(self.config["seed"])
        else:
            torch.manual_seed(self.config["seed"])
            np.random.seed(self.config["seed"])
            random.seed(self.config["seed"])
        
        # Initialize accelerator
        if ACCELERATE_AVAILABLE:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=1,
                mixed_precision="fp16"  # Use fp16 for MI300X
            )
            self.device = self.accelerator.device
        else:
            self.accelerator = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        self.model = model.to(self.device)
        
        # Datasets and dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Optimizer (only trainable parameters)
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        # Learning rate scheduler
        num_training_steps = len(self.train_loader) * self.config["num_epochs"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
            eta_min=self.config["learning_rate"] * 0.1
        )
        
        # Loss function
        self.loss_fn = FlowMatchingLoss()
        
        # Prepare with accelerator
        if self.accelerator is not None:
            self.model, self.optimizer, self.train_loader, self.scheduler = \
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_loader, self.scheduler
                )
            if self.val_loader is not None:
                self.val_loader = self.accelerator.prepare(self.val_loader)
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")
    
    def train(self):
        """Run the full training loop."""
        # Initialize wandb
        if self.use_wandb:
            if self.accelerator is None or self.accelerator.is_main_process:
                wandb.init(
                    project="zen-garden-smolvla",
                    name=f"{self.skill}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=self.config
                )
        
        print(f"\nStarting training for skill: {self.skill}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.config['num_epochs']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Output: {self.output_dir}")
        
        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                epoch_loss += loss
                num_batches += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config["log_every_n_steps"] == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    
                    if self.accelerator is None or self.accelerator.is_main_process:
                        print(f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                              f"Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                        
                        if self.use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                                "train/epoch": epoch,
                                "train/step": self.global_step
                            })
                
                # Evaluation
                if self.val_loader is not None and \
                   self.global_step % self.config["eval_every_n_steps"] == 0:
                    val_loss = self.evaluate()
                    
                    if self.accelerator is None or self.accelerator.is_main_process:
                        print(f"  Validation Loss: {val_loss:.4f}")
                        
                        if self.use_wandb:
                            wandb.log({
                                "val/loss": val_loss,
                                "val/step": self.global_step
                            })
                        
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint("best")
                
                # Checkpointing
                if self.global_step % self.config["save_every_n_steps"] == 0:
                    if self.accelerator is None or self.accelerator.is_main_process:
                        self.save_checkpoint(f"step_{self.global_step}")
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            if self.accelerator is None or self.accelerator.is_main_process:
                print(f"\nEpoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}\n")
                self.save_checkpoint(f"epoch_{epoch+1}")
        
        # Final save
        if self.accelerator is None or self.accelerator.is_main_process:
            self.save_checkpoint("final")
            print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.4f}")
            
            if self.use_wandb:
                wandb.finish()
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step."""
        observation = batch["observation"].to(self.device)
        goal = batch["goal"].to(self.device)
        state = batch["state"].to(self.device)
        actions = batch["actions"].to(self.device)
        
        # Forward pass and loss
        self.optimizer.zero_grad()
        
        loss = self.loss_fn(self.model, observation, goal, state, actions)
        
        # Backward pass
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config["max_grad_norm"] > 0:
            if self.accelerator is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["max_grad_norm"]
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["max_grad_norm"]
                )
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            observation = batch["observation"].to(self.device)
            goal = batch["goal"].to(self.device)
            state = batch["state"].to(self.device)
            actions = batch["actions"].to(self.device)
            
            loss = self.loss_fn(self.model, observation, goal, state, actions)
            total_loss += loss.item()
            num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, name: str):
        """Save a model checkpoint."""
        checkpoint_path = self.output_dir / name
        
        if self.accelerator is not None:
            # Unwrap model before saving
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(str(checkpoint_path))
        else:
            self.model.save_pretrained(str(checkpoint_path))
        
        # Save training state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }, checkpoint_path / "training_state.pt")
        
        print(f"Saved checkpoint: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train goal-conditioned SmolVLA for a specific skill"
    )
    parser.add_argument(
        "--skill",
        type=str,
        required=True,
        choices=["flatten", "zigzag", "circle", "stamp", "place_rock"],
        help="Skill to train"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to goal-augmented dataset (default: ./data/{skill}_goal)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="lerobot/smolvla_base",
        help="Path to pretrained SmolVLA"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set data directory
    data_dir = args.data_dir or f"./data/{args.skill}_goal"
    
    # Check data exists
    if not Path(data_dir).exists():
        print(f"Error: Data directory not found: {data_dir}")
        print(f"Please run prepare_goal_dataset.py first to create the goal-augmented dataset.")
        return
    
    # Create config
    config = {
        **DEFAULT_CONFIG,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "seed": args.seed
    }
    
    # Create model
    print(f"\nCreating goal-conditioned model...")
    model = create_goal_conditioned_model(
        skill=args.skill,
        pretrained_path=args.pretrained_path
    )
    
    # Create dataset
    print(f"\nLoading dataset from {data_dir}...")
    train_dataset = GoalConditionedDataset(
        data_dir=data_dir,
        image_size=config["image_size"],
        action_chunk_size=config["action_chunk_size"],
        augment=True
    )
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        config=config,
        output_dir=args.output_dir,
        skill=args.skill,
        use_wandb=not args.no_wandb
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
