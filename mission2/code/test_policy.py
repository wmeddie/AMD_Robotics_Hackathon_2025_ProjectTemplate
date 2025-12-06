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
CAMERA_GOAL = 8     # Goal image camera (shows target pattern)

# Robot configuration (must match calibration used during training)
ROBOT_PORT = "/dev/ttyACM1"
ROBOT_ID = "my_awesome_follower_arm"  # Calibration ID from training

# Default task description (used during inference)
DEFAULT_TASK = "pick up the rock and place it in the zen garden"


def load_policy_with_processors(checkpoint_path: str, device: str = "cuda"):
    """Load trained SmolVLA policy with pre/post processors.
    
    Args:
        checkpoint_path: Either a local path or HuggingFace Hub repo ID (e.g., "wmeddie/smolvla_place_rock")
        device: Device to load model on
    """
    print(f"Loading policy from {checkpoint_path}...")
    
    # Check if it's a local path or HuggingFace Hub repo
    local_path = Path(checkpoint_path)
    is_local = local_path.exists() and local_path.is_dir()
    
    if is_local:
        print("Loading from local path...")
        # Load config from local
        config = PreTrainedConfig.from_pretrained(
            str(checkpoint_path),
            local_files_only=True
        )
        config.device = device
        
        # Create policy and load weights
        policy = SmolVLAPolicy(config)
        weights_path = local_path / "model.safetensors"
        state_dict = load_file(str(weights_path))
        policy.load_state_dict(state_dict)
        
        # Load processors from local
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint_path)
        )
    else:
        print("Loading from HuggingFace Hub...")
        # Load config from Hub
        config = PreTrainedConfig.from_pretrained(
            checkpoint_path,
            local_files_only=False
        )
        config.device = device
        
        # Load policy from Hub (handles weights automatically)
        policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
        
        # Load processors from Hub
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=checkpoint_path
        )
    
    policy.to(device)
    policy.eval()
    
    print("Policy and processors loaded!")
    return policy, preprocessor, postprocessor


def setup_cameras(use_goal_camera=False):
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
    
    # Goal camera (observation.images.goal) - optional, for goal-conditioned policies
    goal_camera = None
    if use_goal_camera:
        goal_config = OpenCVCameraConfig(
            index_or_path=CAMERA_GOAL,
            width=640,
            height=480,
            fps=30,
        )
        goal_camera = OpenCVCamera(goal_config)
        goal_camera.connect()
    
    print("Cameras connected!")
    return front_camera, top_camera, goal_camera


def setup_robot(max_speed=None):
    """Initialize robot.
    
    Args:
        max_speed: Maximum relative movement per step (in motor units). 
                   Lower = slower/smoother. Try 5-10 for slow, None for full speed.
    """
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    
    print("Setting up robot...")
    
    config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,  # Use same calibration as training
        max_relative_target=max_speed,  # Limit movement speed per step
    )
    robot = SO101Follower(config)
    robot.connect()
    
    if max_speed:
        print(f"Robot connected! (max_speed={max_speed})")
    else:
        print("Robot connected! (full speed)")
    return robot


def get_observation_raw(front_camera, top_camera, robot, goal_camera=None):
    """Get current observation dict as tensors for preprocessing."""
    # Get camera images - returns numpy array [H, W, C] uint8
    front_img = front_camera.async_read()
    top_img = top_camera.async_read()
    
    # Get robot state - SO101 returns {motor_name}.pos for each motor
    robot_obs = robot.get_observation()
    
    # Extract motor positions in order (6 DOF for SO101)
    # Motor names: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    state = np.array([robot_obs.get(f"{m}.pos", 0.0) for m in motor_names], dtype=np.float32)
    
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
    
    # Add goal camera if available
    if goal_camera is not None:
        goal_img = goal_camera.async_read()
        goal_tensor = torch.from_numpy(goal_img).permute(2, 0, 1).float() / 255.0
        observation["observation.images.goal"] = goal_tensor
    
    return observation


def run_inference_loop(policy, preprocessor, postprocessor, front_camera, top_camera, robot, 
                       goal_camera=None, task=DEFAULT_TASK, device="cuda", control_hz=10, duration=30):
    """Run policy inference loop."""
    print(f"\nStarting inference loop at {control_hz}Hz for {duration}s...")
    print(f"Task: {task}")
    if goal_camera:
        print("Goal camera enabled - policy will use goal image")
    print("Press Ctrl+C to stop\n")
    
    # Reset policy action queue before starting
    policy.reset()
    print("Policy reset.")
    
    dt = 1.0 / control_hz
    start_time = time.time()
    step = 0
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            
            # Get raw observation (including goal camera if available)
            obs_raw = get_observation_raw(front_camera, top_camera, robot, goal_camera)
            
            # Add task to observation (preprocessor expects it in the dict)
            obs_raw["task"] = task
            
            # Preprocess observation (adds batch dim, normalizes, tokenizes task)
            obs_processed = preprocessor(obs_raw)
            
            # Run policy
            with torch.no_grad():
                action = policy.select_action(obs_processed)
            
            # Postprocess action (unnormalize)
            action_out = postprocessor(action)
            
            # Debug: print type and structure
            if step == 0:
                print(f"DEBUG: action type = {type(action_out)}")
                if isinstance(action_out, dict):
                    print(f"DEBUG: action keys = {action_out.keys()}")
                    for k, v in action_out.items():
                        print(f"DEBUG:   {k}: type={type(v)}, shape={v.shape if hasattr(v, 'shape') else 'N/A'}")
                elif hasattr(action_out, 'shape'):
                    print(f"DEBUG: action shape = {action_out.shape}")
            
            # Get action tensor - handle both dict and tensor outputs
            if isinstance(action_out, dict):
                action_tensor = action_out.get("action", action_out.get("actions", None))
                if action_tensor is None:
                    # Try first value in dict
                    action_tensor = next(iter(action_out.values()))
            else:
                action_tensor = action_out
            
            # Handle different tensor shapes: [action_dim], [1, action_dim], [batch, chunk, action_dim]
            if action_tensor.dim() == 1:
                action_np = action_tensor.cpu().numpy()
            elif action_tensor.dim() == 2:
                action_np = action_tensor[0].cpu().numpy()  # Take first from batch
            else:
                action_np = action_tensor[0, 0].cpu().numpy()  # Take first from batch and chunk
            
            # Send to robot - convert to dict with motor names ending in .pos
            motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            action_dict = {f"{name}.pos": float(action_np[i]) for i, name in enumerate(motor_names)}
            robot.send_action(action_dict)
            
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
    parser.add_argument("--goal-camera", action="store_true", help="Use goal camera for goal-conditioned policy")
    parser.add_argument("--max-speed", type=float, default=None, 
                        help="Limit robot speed (lower=slower). Try 5-10 for slow, omit for full speed")
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
    front_camera, top_camera, goal_camera = setup_cameras(use_goal_camera=args.goal_camera)
    robot = setup_robot(max_speed=args.max_speed)
    
    try:
        # Run inference
        run_inference_loop(
            policy, preprocessor, postprocessor, front_camera, top_camera, robot,
            goal_camera=goal_camera, task=args.task, device=device, 
            control_hz=args.hz, duration=args.duration
        )
    finally:
        # Cleanup
        print("\nCleaning up...")
        front_camera.disconnect()
        top_camera.disconnect()
        if goal_camera:
            goal_camera.disconnect()
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
