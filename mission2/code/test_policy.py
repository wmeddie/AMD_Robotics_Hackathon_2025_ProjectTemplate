#!/usr/bin/env python3
"""
Test trained SmolVLA policy on the robot.

Usage:
    # Dry run (no robot needed):
    python test_policy.py --dry-run
    
    # Run on robot:
    python test_policy.py --duration 30
"""

import argparse
import time
import numpy as np
import torch
from pathlib import Path

# LeRobot imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from safetensors.torch import load_file

# Camera indices (adjust if needed)
CAMERA_ARM = 6      # Front/arm camera
CAMERA_OVERHEAD = 4 # Overhead camera  

# Default task description (used during inference)
DEFAULT_TASK = "pick up the rock and place it in the zen garden"


def load_policy_with_processors(checkpoint_path: str, device: str = "cuda"):
    """Load trained SmolVLA policy with pre/post processors from local checkpoint."""
    print(f"Loading policy from {checkpoint_path}...")
    checkpoint_path = Path(checkpoint_path)
    
    # Load config
    config = PreTrainedConfig.from_pretrained(
        str(checkpoint_path),
        local_files_only=True
    )
    # Override device
    config.device = device
    
    # Create policy
    policy = SmolVLAPolicy(config)
    
    # Load weights
    weights_path = checkpoint_path / "model.safetensors"
    state_dict = load_file(str(weights_path))
    policy.load_state_dict(state_dict)
    
    policy.to(device)
    policy.eval()
    
    # Load pre/post processors
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint_path)
    )
    
    print("Policy and processors loaded!")
    return policy, preprocessor, postprocessor


def setup_cameras():
    """Initialize cameras."""
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    
    print("Setting up cameras...")
    
    # Front camera (observation.images.front)
    front_config = OpenCVCameraConfig(
        index_or_path=CAMERA_ARM,
        width=640,
        height=480,
        fps=30,
    )
    front_camera = OpenCVCamera(front_config)
    
    # Top camera (observation.images.top)
    top_config = OpenCVCameraConfig(
        index_or_path=CAMERA_OVERHEAD,
        width=640,
        height=480,
        fps=30,
    )
    top_camera = OpenCVCamera(top_config)
    
    front_camera.connect()
    top_camera.connect()
    
    print("Cameras connected!")
    return front_camera, top_camera


def setup_robot():
    """Initialize robot."""
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    
    print("Setting up robot...")
    
    config = SO101FollowerConfig(
        port="/dev/ttyACM1",  # Follower port
    )
    robot = SO101Follower(config)
    robot.connect()
    
    print("Robot connected!")
    return robot


def get_observation_raw(front_camera, top_camera, robot):
    """Get current observation dict as tensors for preprocessing."""
    # Get camera images - returns numpy array [H, W, C] uint8
    front_img = front_camera.async_read()
    top_img = top_camera.async_read()
    
    # Get robot state
    state = robot.get_observation()["observation.state"]  # [6] for SO101
    
    # Convert images to tensors: [H, W, C] uint8 -> [C, H, W] float32 [0, 1]
    front_tensor = torch.from_numpy(front_img).permute(2, 0, 1).float() / 255.0
    top_tensor = torch.from_numpy(top_img).permute(2, 0, 1).float() / 255.0
    
    # Convert state to tensor
    state_tensor = torch.from_numpy(state).float()
    
    observation = {
        "observation.images.front": front_tensor,
        "observation.images.top": top_tensor,
        "observation.state": state_tensor,
    }
    
    return observation


def run_inference_loop(policy, preprocessor, postprocessor, front_camera, top_camera, robot, 
                       task=DEFAULT_TASK, device="cuda", control_hz=10, duration=30):
    """Run policy inference loop."""
    print(f"\nStarting inference loop at {control_hz}Hz for {duration}s...")
    print(f"Task: {task}")
    print("Press Ctrl+C to stop\n")
    
    dt = 1.0 / control_hz
    start_time = time.time()
    step = 0
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            
            # Get raw observation
            obs_raw = get_observation_raw(front_camera, top_camera, robot)
            
            # Add task to observation (preprocessor expects it in the dict)
            obs_raw["task"] = task
            
            # Preprocess observation (adds batch dim, normalizes, tokenizes task)
            obs_processed = preprocessor(obs_raw)
            
            # Run policy
            with torch.no_grad():
                action = policy.select_action(obs_processed)
            
            # Postprocess action (unnormalize)
            action = postprocessor(action)
            
            # action is tensor [action_dim]
            action_np = action["action"].cpu().numpy()
            
            # Send to robot
            robot.send_action({"action": action_np})
            
            step += 1
            if step % 10 == 0:
                print(f"Step {step}: action = {action_np}")
            
            # Maintain control frequency
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    print(f"\nCompleted {step} steps")


def main():
    parser = argparse.ArgumentParser(description="Test SmolVLA policy on robot")
    parser.add_argument("--checkpoint", type=str, 
                        default="./outputs/smolvla_place_rock/checkpoints/last/pretrained_model",
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hz", type=int, default=10, help="Control frequency")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task description")
    parser.add_argument("--dry-run", action="store_true", help="Test without robot")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load policy with processors
    policy, preprocessor, postprocessor = load_policy_with_processors(args.checkpoint, device)
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print("Testing policy with dummy inputs...")
        
        # Create dummy observation as tensors (matching what dataset provides)
        # Images: [C, H, W] float32 normalized to [0, 1]
        # State: [state_dim] float32
        dummy_obs_raw = {
            "observation.images.front": torch.rand(3, 480, 640),
            "observation.images.top": torch.rand(3, 480, 640),
            "observation.state": torch.zeros(6),
        }
        
        # Add task to observation
        dummy_obs_raw["task"] = args.task
        
        # Preprocess
        obs_processed = preprocessor(dummy_obs_raw)
        print(f"Preprocessed observation keys: {obs_processed.keys()}")
        
        # Run policy
        with torch.no_grad():
            action = policy.select_action(obs_processed)
        
        # Postprocess
        action = postprocessor(action)
        
        print(f"Output action: {action}")
        print("\nDry run successful!")
        return
    
    # Setup hardware
    front_camera, top_camera = setup_cameras()
    robot = setup_robot()
    
    try:
        # Run inference
        run_inference_loop(
            policy, preprocessor, postprocessor, front_camera, top_camera, robot,
            task=args.task, device=device, control_hz=args.hz, duration=args.duration
        )
    finally:
        # Cleanup
        print("\nCleaning up...")
        front_camera.disconnect()
        top_camera.disconnect()
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
