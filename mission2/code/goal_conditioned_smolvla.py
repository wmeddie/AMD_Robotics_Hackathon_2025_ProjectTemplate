#!/usr/bin/env python3
"""
Goal-Conditioned SmolVLA Model Architecture

This module implements a goal-conditioned variant of SmolVLA for the Zen Garden Robot.
The key modification is injecting a goal image embedding into the action expert as
additional conditioning.

Architecture:
    goal_image ---------> SigLIP (frozen) -> goal_proj (trainable) ----┐
                                                                        ├-> concat -> Action Expert -> actions
    observation_images -> SmolVLM-2 (frozen) -> vlm_features ----------┘

Usage:
    from goal_conditioned_smolvla import GoalConditionedSmolVLA
    
    model = GoalConditionedSmolVLA(pretrained_path="HuggingFaceTB/SmolVLA-base")
    actions = model(observation_images, goal_image, state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

# Try importing from transformers/lerobot
try:
    from transformers import AutoModel, AutoProcessor, AutoConfig
    from transformers import SiglipVisionModel, SiglipImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed")

# Try importing LeRobot policy components - LeRobot 0.4.x API
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    LEROBOT_SMOLVLA_AVAILABLE = True
except ImportError:
    LEROBOT_SMOLVLA_AVAILABLE = False
    print("Warning: LeRobot SmolVLA not available, using standalone implementation")


class GoalProjection(nn.Module):
    """Projects goal image features to match action expert input dimension."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        hidden_dim = hidden_dim or (input_dim + output_dim) // 2
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class GoalConditionedSmolVLA(nn.Module):
    """
    Goal-Conditioned SmolVLA for skill-based robot control.
    
    This model extends SmolVLA to accept a goal image that conditions the
    action predictions. The goal image represents the desired end state
    of the robot's action.
    
    Args:
        pretrained_path: Path or HuggingFace model ID for pretrained SmolVLA
        siglip_model: SigLIP model ID for goal encoding (default: uses SmolVLA's encoder)
        goal_embed_dim: Dimension of goal embedding (default: inferred from SigLIP)
        freeze_vlm: Whether to freeze the VLM backbone (default: True)
        freeze_goal_encoder: Whether to freeze the goal encoder (default: True)
        action_chunk_size: Number of future timesteps to predict (default: 10)
        state_dim: Dimension of proprioceptive state (default: 14 for SO-101)
        action_dim: Dimension of action space (default: 14 for SO-101)
    """
    
    # Default dimensions based on SmolVLA architecture
    DEFAULT_SIGLIP_DIM = 1152  # SigLIP-SO400M hidden size
    DEFAULT_VLM_DIM = 1536     # SmolLM2 hidden size
    DEFAULT_ACTION_EXPERT_DIM = 512
    DEFAULT_STATE_DIM = 14     # SO-101: 7 joints * 2 (pos + vel) or 6 joints + gripper
    DEFAULT_ACTION_DIM = 14    # SO-101 action space
    DEFAULT_ACTION_CHUNK_SIZE = 10
    
    def __init__(
        self,
        pretrained_path: str = "lerobot/smolvla_base",
        siglip_model: Optional[str] = None,
        goal_embed_dim: Optional[int] = None,
        freeze_vlm: bool = True,
        freeze_goal_encoder: bool = True,
        action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE,
        state_dim: int = DEFAULT_STATE_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        dropout: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.pretrained_path = pretrained_path
        self.freeze_vlm = freeze_vlm
        self.freeze_goal_encoder = freeze_goal_encoder
        self.action_chunk_size = action_chunk_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Initialize components
        self._init_base_model(pretrained_path)
        self._init_goal_encoder(siglip_model, goal_embed_dim)
        self._init_goal_projection(dropout)
        self._init_action_expert_modification()
        
        # Apply freezing
        self._apply_freezing()
    
    def _init_base_model(self, pretrained_path: str):
        """Initialize the base SmolVLA model."""
        if LEROBOT_SMOLVLA_AVAILABLE:
            try:
                # Load from LeRobot
                self.config = SmolVLAConfig.from_pretrained(pretrained_path)
                self.base_model = SmolVLAPolicy.from_pretrained(pretrained_path)
                self.vlm = self.base_model.model.vlm
                self.action_expert = self.base_model.model.action_expert
                self._using_lerobot = True
                print(f"Loaded SmolVLA from LeRobot: {pretrained_path}")
                return
            except Exception as e:
                print(f"Could not load from LeRobot: {e}")
        
        # Fallback: create components manually
        self._using_lerobot = False
        self._init_standalone_components()
    
    def _init_standalone_components(self):
        """Initialize components when LeRobot SmolVLA is not available."""
        print("Initializing standalone SmolVLA components...")
        
        # Load SmolVLM-2 for observation encoding
        if TRANSFORMERS_AVAILABLE:
            try:
                self.vlm = AutoModel.from_pretrained(
                    "HuggingFaceTB/SmolVLM2-1.7B-Instruct",
                    trust_remote_code=True
                )
                self.vlm_processor = AutoProcessor.from_pretrained(
                    "HuggingFaceTB/SmolVLM2-1.7B-Instruct",
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Could not load SmolVLM: {e}")
                self.vlm = None
                self.vlm_processor = None
        
        # Create action expert (simplified flow matching decoder)
        self.action_expert = ActionExpert(
            input_dim=self.DEFAULT_VLM_DIM + self.DEFAULT_ACTION_EXPERT_DIM,  # VLM + goal
            hidden_dim=self.DEFAULT_ACTION_EXPERT_DIM,
            output_dim=self.action_dim * self.action_chunk_size,
            state_dim=self.state_dim,
            num_layers=4
        )
    
    def _init_goal_encoder(self, siglip_model: Optional[str], goal_embed_dim: Optional[int]):
        """Initialize the goal image encoder (SigLIP)."""
        if siglip_model is None:
            # Try to extract SigLIP from the base model
            if hasattr(self, 'vlm') and self.vlm is not None:
                if hasattr(self.vlm, 'vision_model'):
                    self.goal_encoder = self.vlm.vision_model
                    self.siglip_dim = self.vlm.config.vision_config.hidden_size
                    print(f"Using VLM's vision encoder, dim={self.siglip_dim}")
                    return
                elif hasattr(self.vlm, 'vision_tower'):
                    self.goal_encoder = self.vlm.vision_tower
                    self.siglip_dim = getattr(self.vlm.config, 'vision_hidden_size', self.DEFAULT_SIGLIP_DIM)
                    print(f"Using VLM's vision tower, dim={self.siglip_dim}")
                    return
        
        # Load standalone SigLIP
        siglip_model = siglip_model or "google/siglip-so400m-patch14-384"
        if TRANSFORMERS_AVAILABLE:
            try:
                self.goal_encoder = SiglipVisionModel.from_pretrained(siglip_model)
                self.goal_processor = SiglipImageProcessor.from_pretrained(siglip_model)
                self.siglip_dim = self.goal_encoder.config.hidden_size
                print(f"Loaded SigLIP: {siglip_model}, dim={self.siglip_dim}")
            except Exception as e:
                print(f"Could not load SigLIP: {e}")
                self.goal_encoder = None
                self.siglip_dim = goal_embed_dim or self.DEFAULT_SIGLIP_DIM
        else:
            self.goal_encoder = None
            self.siglip_dim = goal_embed_dim or self.DEFAULT_SIGLIP_DIM
    
    def _init_goal_projection(self, dropout: float):
        """Initialize the goal projection layer."""
        # Project SigLIP features to action expert dimension
        self.goal_projection = GoalProjection(
            input_dim=self.siglip_dim,
            output_dim=self.DEFAULT_ACTION_EXPERT_DIM,
            dropout=dropout
        )
        print(f"Goal projection: {self.siglip_dim} -> {self.DEFAULT_ACTION_EXPERT_DIM}")
    
    def _init_action_expert_modification(self):
        """Modify action expert to accept concatenated features."""
        # The action expert now receives VLM features + goal features
        # We may need to modify its input layer
        if hasattr(self.action_expert, 'input_proj'):
            original_dim = self.action_expert.input_proj.in_features
            new_dim = original_dim + self.DEFAULT_ACTION_EXPERT_DIM
            
            # Create new input projection
            new_input_proj = nn.Linear(new_dim, self.action_expert.input_proj.out_features)
            
            # Initialize with original weights for VLM features
            with torch.no_grad():
                new_input_proj.weight[:, :original_dim] = self.action_expert.input_proj.weight
                new_input_proj.weight[:, original_dim:] = torch.zeros_like(
                    new_input_proj.weight[:, original_dim:]
                )
                new_input_proj.bias = self.action_expert.input_proj.bias
            
            self.action_expert.input_proj = new_input_proj
            print(f"Modified action expert input: {original_dim} -> {new_dim}")
    
    def _apply_freezing(self):
        """Freeze specified components."""
        if self.freeze_vlm and self.vlm is not None:
            for param in self.vlm.parameters():
                param.requires_grad = False
            print("Froze VLM backbone")
        
        if self.freeze_goal_encoder and self.goal_encoder is not None:
            for param in self.goal_encoder.parameters():
                param.requires_grad = False
            print("Froze goal encoder")
        
        # Ensure trainable components
        for param in self.goal_projection.parameters():
            param.requires_grad = True
        
        for param in self.action_expert.parameters():
            param.requires_grad = True
        
        print("Goal projection and action expert are trainable")
    
    def encode_goal(self, goal_image: torch.Tensor) -> torch.Tensor:
        """
        Encode the goal image using SigLIP.
        
        Args:
            goal_image: Goal image tensor [B, C, H, W]
            
        Returns:
            Goal embedding [B, goal_embed_dim]
        """
        if self.goal_encoder is None:
            # Return dummy embedding if no encoder
            batch_size = goal_image.shape[0]
            return torch.zeros(batch_size, self.siglip_dim, device=goal_image.device)
        
        with torch.no_grad() if self.freeze_goal_encoder else torch.enable_grad():
            # Get vision encoder output
            if hasattr(self.goal_encoder, 'forward'):
                outputs = self.goal_encoder(goal_image)
                if hasattr(outputs, 'last_hidden_state'):
                    # Pool over spatial dimensions (take CLS token or mean pool)
                    hidden_states = outputs.last_hidden_state
                    if hidden_states.dim() == 3:
                        # [B, num_patches, hidden_dim] -> [B, hidden_dim]
                        goal_features = hidden_states.mean(dim=1)
                    else:
                        goal_features = hidden_states
                elif hasattr(outputs, 'pooler_output'):
                    goal_features = outputs.pooler_output
                else:
                    goal_features = outputs
            else:
                goal_features = self.goal_encoder(goal_image)
        
        return goal_features
    
    def encode_observations(
        self,
        observation_images: torch.Tensor,
        state: torch.Tensor,
        instruction: Optional[str] = None
    ) -> torch.Tensor:
        """
        Encode observations using the VLM.
        
        Args:
            observation_images: Observation images [B, C, H, W] or [B, T, C, H, W]
            state: Proprioceptive state [B, state_dim]
            instruction: Optional text instruction
            
        Returns:
            VLM features [B, vlm_dim]
        """
        if self.vlm is None:
            batch_size = observation_images.shape[0]
            return torch.zeros(batch_size, self.DEFAULT_VLM_DIM, device=observation_images.device)
        
        with torch.no_grad() if self.freeze_vlm else torch.enable_grad():
            # Handle different input formats
            if observation_images.dim() == 5:
                # [B, T, C, H, W] -> use last frame
                observation_images = observation_images[:, -1]
            
            # Get VLM features
            if self._using_lerobot:
                # Use LeRobot's forward pass
                vlm_features = self.base_model.get_vlm_features(
                    observation_images, state, instruction
                )
            else:
                # Standalone forward pass
                outputs = self.vlm(observation_images)
                if hasattr(outputs, 'last_hidden_state'):
                    vlm_features = outputs.last_hidden_state.mean(dim=1)
                else:
                    vlm_features = outputs
        
        return vlm_features
    
    def forward(
        self,
        observation_images: torch.Tensor,
        goal_image: torch.Tensor,
        state: torch.Tensor,
        instruction: Optional[str] = None,
        timestep: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for goal-conditioned action prediction.
        
        Args:
            observation_images: Current observation [B, C, H, W]
            goal_image: Goal/target image [B, C, H, W]
            state: Proprioceptive state [B, state_dim]
            instruction: Optional text instruction
            timestep: Diffusion/flow timestep for training
            noise: Noise for flow matching training
            
        Returns:
            Predicted actions [B, action_chunk_size, action_dim]
        """
        batch_size = observation_images.shape[0]
        
        # Encode goal image
        goal_features = self.encode_goal(goal_image)
        goal_projected = self.goal_projection(goal_features)
        
        # Encode observations
        vlm_features = self.encode_observations(observation_images, state, instruction)
        
        # Concatenate goal conditioning with VLM features
        combined_features = torch.cat([vlm_features, goal_projected], dim=-1)
        
        # Predict actions through action expert
        if hasattr(self.action_expert, 'forward_with_flow'):
            # Flow matching training mode
            actions = self.action_expert.forward_with_flow(
                combined_features, state, timestep, noise
            )
        else:
            # Standard forward
            actions = self.action_expert(combined_features, state)
        
        # Reshape to action chunks
        if actions.dim() == 2:
            actions = actions.view(batch_size, self.action_chunk_size, self.action_dim)
        
        return actions
    
    def predict_action(
        self,
        observation_images: torch.Tensor,
        goal_image: torch.Tensor,
        state: torch.Tensor,
        instruction: Optional[str] = None,
        num_inference_steps: int = 10
    ) -> torch.Tensor:
        """
        Predict actions at inference time (with flow matching sampling).
        
        Args:
            observation_images: Current observation [B, C, H, W]
            goal_image: Goal/target image [B, C, H, W]
            state: Proprioceptive state [B, state_dim]
            instruction: Optional text instruction
            num_inference_steps: Number of denoising steps
            
        Returns:
            Predicted actions [B, action_chunk_size, action_dim]
        """
        self.eval()
        batch_size = observation_images.shape[0]
        device = observation_images.device
        
        with torch.no_grad():
            # Encode goal and observations
            goal_features = self.encode_goal(goal_image)
            goal_projected = self.goal_projection(goal_features)
            vlm_features = self.encode_observations(observation_images, state, instruction)
            combined_features = torch.cat([vlm_features, goal_projected], dim=-1)
            
            # Flow matching inference (Euler sampling)
            # Start from noise
            x = torch.randn(
                batch_size, self.action_chunk_size * self.action_dim,
                device=device
            )
            
            # Euler steps from t=0 to t=1
            dt = 1.0 / num_inference_steps
            for i in range(num_inference_steps):
                t = torch.full((batch_size,), i * dt, device=device)
                
                # Predict velocity
                if hasattr(self.action_expert, 'forward_with_timestep'):
                    velocity = self.action_expert.forward_with_timestep(
                        combined_features, state, x, t
                    )
                else:
                    velocity = self.action_expert(combined_features, state)
                
                # Euler step
                x = x + velocity * dt
            
            actions = x.view(batch_size, self.action_chunk_size, self.action_dim)
        
        return actions
    
    def get_trainable_parameters(self) -> list:
        """Get list of trainable parameters."""
        params = []
        params.extend(self.goal_projection.parameters())
        params.extend(self.action_expert.parameters())
        return params
    
    def save_pretrained(self, save_path: str):
        """Save the model checkpoint."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'goal_projection': self.goal_projection.state_dict(),
            'action_expert': self.action_expert.state_dict(),
            'config': {
                'pretrained_path': self.pretrained_path,
                'siglip_dim': self.siglip_dim,
                'action_chunk_size': self.action_chunk_size,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
            }
        }, save_path / 'goal_conditioned_smolvla.pt')
        
        print(f"Saved model to {save_path}")
    
    @classmethod
    def load_pretrained(cls, load_path: str, device: str = "cuda") -> "GoalConditionedSmolVLA":
        """Load a saved model checkpoint."""
        load_path = Path(load_path)
        checkpoint = torch.load(load_path / 'goal_conditioned_smolvla.pt', map_location=device)
        
        config = checkpoint['config']
        model = cls(
            pretrained_path=config['pretrained_path'],
            action_chunk_size=config['action_chunk_size'],
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            device=device
        )
        
        model.goal_projection.load_state_dict(checkpoint['goal_projection'])
        model.action_expert.load_state_dict(checkpoint['action_expert'])
        model.to(device)
        
        print(f"Loaded model from {load_path}")
        return model


class ActionExpert(nn.Module):
    """
    Simplified Action Expert for flow matching action prediction.
    
    This is a fallback implementation when the full SmolVLA action expert
    is not available. Uses a transformer-based architecture with flow matching.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        state_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # State embedding
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        
        # Time embedding for flow matching
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Noisy action embedding
        self.action_embed = nn.Linear(output_dim, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Standard forward pass (for inference without flow matching)."""
        batch_size = features.shape[0]
        
        # Project inputs
        feat_embed = self.input_proj(features)
        state_embed = self.state_proj(state)
        
        # Combine features
        combined = feat_embed + state_embed
        combined = combined.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Transformer
        output = self.transformer(combined)
        output = output.squeeze(1)  # [B, hidden_dim]
        
        # Project to actions
        actions = self.output_proj(output)
        
        return actions
    
    def forward_with_timestep(
        self,
        features: torch.Tensor,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with flow matching (predicts velocity field)."""
        batch_size = features.shape[0]
        
        # Embed inputs
        feat_embed = self.input_proj(features)
        state_embed = self.state_proj(state)
        time_embed = self.time_embed(timestep.unsqueeze(-1))
        action_embed = self.action_embed(noisy_actions)
        
        # Combine all embeddings
        combined = feat_embed + state_embed + time_embed + action_embed
        combined = combined.unsqueeze(1)
        
        # Transformer
        output = self.transformer(combined)
        output = output.squeeze(1)
        
        # Predict velocity
        velocity = self.output_proj(output)
        
        return velocity


def create_goal_conditioned_model(
    skill: str,
    pretrained_path: str = "lerobot/smolvla_base",
    device: str = "cuda"
) -> GoalConditionedSmolVLA:
    """
    Factory function to create a goal-conditioned SmolVLA for a specific skill.
    
    Args:
        skill: Skill name (flatten, zigzag, circle, stamp, place_rock)
        pretrained_path: Path to pretrained SmolVLA
        device: Device to load model on
        
    Returns:
        Initialized GoalConditionedSmolVLA model
    """
    model = GoalConditionedSmolVLA(
        pretrained_path=pretrained_path,
        device=device
    )
    model.to(device)
    
    print(f"Created goal-conditioned model for skill: {skill}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.get_trainable_parameters()):,}")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing GoalConditionedSmolVLA...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = GoalConditionedSmolVLA(device=device)
    model.to(device)
    
    # Test forward pass
    batch_size = 2
    observation = torch.randn(batch_size, 3, 224, 224, device=device)
    goal = torch.randn(batch_size, 3, 224, 224, device=device)
    state = torch.randn(batch_size, 14, device=device)
    
    print("\nRunning forward pass...")
    actions = model(observation, goal, state)
    print(f"Output shape: {actions.shape}")
    print(f"Expected: [{batch_size}, {model.action_chunk_size}, {model.action_dim}]")
    
    # Test inference
    print("\nRunning inference...")
    actions = model.predict_action(observation, goal, state)
    print(f"Inference output shape: {actions.shape}")
    
    print("\nTest complete!")
