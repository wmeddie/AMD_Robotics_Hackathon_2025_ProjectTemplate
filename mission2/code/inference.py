#!/usr/bin/env python3
"""
Inference Module for Goal-Conditioned SmolVLA

Provides a SkillPolicy class for loading trained models and running inference.

Usage:
    from inference import SkillPolicy
    
    policy = SkillPolicy("./checkpoints/flatten/best")
    action = policy.predict(observation_image, goal_image, state)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from goal_conditioned_smolvla import GoalConditionedSmolVLA


class SkillPolicy:
    """
    Wrapper for running inference with a trained goal-conditioned policy.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        device: Device to run inference on ("cuda" or "cpu")
        num_inference_steps: Number of flow matching denoising steps
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        num_inference_steps: int = 10,
        image_size: int = 224
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_inference_steps = num_inference_steps
        self.image_size = image_size
        
        # Load model
        self.model = GoalConditionedSmolVLA.load_pretrained(
            checkpoint_path, 
            device=str(self.device)
        )
        self.model.eval()
        self.model.to(self.device)
        
        # Image normalization stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        print(f"Loaded policy from {checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Inference steps: {num_inference_steps}")
    
    def preprocess_image(
        self,
        image: Union[np.ndarray, "Image.Image", torch.Tensor, str]
    ) -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image: Input image (numpy array, PIL Image, tensor, or file path)
            
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Load from path if string
        if isinstance(image, str):
            if PIL_AVAILABLE:
                image = Image.open(image).convert("RGB")
            elif CV2_AVAILABLE:
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise RuntimeError("No image loading library available")
        
        # Convert PIL to numpy
        if PIL_AVAILABLE and isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert numpy to tensor
        if isinstance(image, np.ndarray):
            # Resize if needed
            if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
                if PIL_AVAILABLE:
                    image = Image.fromarray(image).resize((self.image_size, self.image_size))
                    image = np.array(image)
                elif CV2_AVAILABLE:
                    image = cv2.resize(image, (self.image_size, self.image_size))
            
            # Normalize to [0, 1]
            if image.max() > 1.0:
                image = image / 255.0
            
            # Convert to tensor [C, H, W]
            image = torch.from_numpy(image).float()
            if image.dim() == 3 and image.shape[2] == 3:
                image = image.permute(2, 0, 1)
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Move to device and normalize
        image = image.to(self.device)
        image = (image - self.mean) / self.std
        
        return image
    
    def preprocess_state(
        self,
        state: Union[np.ndarray, torch.Tensor, list]
    ) -> torch.Tensor:
        """
        Preprocess proprioceptive state for model input.
        
        Args:
            state: Robot state vector
            
        Returns:
            State tensor [1, state_dim]
        """
        if isinstance(state, list):
            state = np.array(state)
        
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        return state.to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        observation_image: Union[np.ndarray, torch.Tensor, str],
        goal_image: Union[np.ndarray, torch.Tensor, str],
        state: Union[np.ndarray, torch.Tensor, list],
        return_full_chunk: bool = False
    ) -> np.ndarray:
        """
        Predict action given observation, goal, and state.
        
        Args:
            observation_image: Current camera observation
            goal_image: Target/goal image to achieve
            state: Current robot proprioceptive state
            return_full_chunk: If True, return all predicted future actions
            
        Returns:
            Predicted action(s) as numpy array
            If return_full_chunk=False: [action_dim]
            If return_full_chunk=True: [action_chunk_size, action_dim]
        """
        # Preprocess inputs
        obs = self.preprocess_image(observation_image)
        goal = self.preprocess_image(goal_image)
        state_tensor = self.preprocess_state(state)
        
        # Run inference
        actions = self.model.predict_action(
            obs, goal, state_tensor,
            num_inference_steps=self.num_inference_steps
        )
        
        # Convert to numpy
        actions = actions.cpu().numpy()
        
        if return_full_chunk:
            return actions[0]  # [action_chunk_size, action_dim]
        else:
            return actions[0, 0]  # First action only [action_dim]
    
    @torch.no_grad()
    def predict_batch(
        self,
        observation_images: list,
        goal_images: list,
        states: list
    ) -> np.ndarray:
        """
        Batch prediction for multiple inputs.
        
        Args:
            observation_images: List of observation images
            goal_images: List of goal images
            states: List of robot states
            
        Returns:
            Batch of predicted actions [B, action_dim]
        """
        # Preprocess all inputs
        obs_batch = torch.cat([self.preprocess_image(img) for img in observation_images])
        goal_batch = torch.cat([self.preprocess_image(img) for img in goal_images])
        state_batch = torch.cat([self.preprocess_state(s) for s in states])
        
        # Run inference
        actions = self.model.predict_action(
            obs_batch, goal_batch, state_batch,
            num_inference_steps=self.num_inference_steps
        )
        
        return actions[:, 0].cpu().numpy()  # First action from each chunk


class MultiSkillPolicy:
    """
    Manages multiple skill policies for the Zen Garden robot.
    
    Loads all trained skill policies and provides a unified interface.
    """
    
    SKILLS = ["flatten", "zigzag", "circle", "stamp", "place_rock"]
    
    def __init__(
        self,
        checkpoints_dir: str = "./checkpoints",
        device: str = "cuda",
        num_inference_steps: int = 10
    ):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.device = device
        self.num_inference_steps = num_inference_steps
        
        self.policies = {}
        self._load_policies()
    
    def _load_policies(self):
        """Load all available skill policies."""
        for skill in self.SKILLS:
            checkpoint_path = self.checkpoints_dir / skill / "best"
            if checkpoint_path.exists():
                try:
                    self.policies[skill] = SkillPolicy(
                        str(checkpoint_path),
                        device=self.device,
                        num_inference_steps=self.num_inference_steps
                    )
                    print(f"Loaded policy for skill: {skill}")
                except Exception as e:
                    print(f"Failed to load policy for {skill}: {e}")
            else:
                print(f"No checkpoint found for skill: {skill}")
    
    def predict(
        self,
        skill: str,
        observation_image: Union[np.ndarray, torch.Tensor, str],
        goal_image: Union[np.ndarray, torch.Tensor, str],
        state: Union[np.ndarray, torch.Tensor, list]
    ) -> np.ndarray:
        """
        Predict action for a specific skill.
        
        Args:
            skill: Which skill policy to use
            observation_image: Current observation
            goal_image: Goal image
            state: Robot state
            
        Returns:
            Predicted action
        """
        if skill not in self.policies:
            raise ValueError(f"Policy for skill '{skill}' not loaded. "
                           f"Available: {list(self.policies.keys())}")
        
        return self.policies[skill].predict(observation_image, goal_image, state)
    
    def get_available_skills(self) -> list:
        """Get list of loaded skills."""
        return list(self.policies.keys())


if __name__ == "__main__":
    # Test inference
    import argparse
    
    parser = argparse.ArgumentParser(description="Test skill policy inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--observation", type=str, help="Path to observation image")
    parser.add_argument("--goal", type=str, help="Path to goal image")
    args = parser.parse_args()
    
    # Load policy
    policy = SkillPolicy(args.checkpoint)
    
    # Create dummy inputs if not provided
    if args.observation and args.goal:
        obs = args.observation
        goal = args.goal
    else:
        print("Using random dummy inputs for testing...")
        obs = np.random.rand(224, 224, 3).astype(np.float32)
        goal = np.random.rand(224, 224, 3).astype(np.float32)
    
    state = np.zeros(14, dtype=np.float32)
    
    # Run inference
    action = policy.predict(obs, goal, state)
    print(f"Predicted action shape: {action.shape}")
    print(f"Predicted action: {action}")
