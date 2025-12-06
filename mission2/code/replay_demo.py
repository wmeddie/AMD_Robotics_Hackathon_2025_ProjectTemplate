#!/usr/bin/env python3
"""
Replay recorded demonstrations with LLM-based skill selection.

Flow:
1. Capture overhead camera image
2. Send to Claude to decide: "horizontal" or "vertical" rake pattern
3. Replay the corresponding demonstration from the dataset

Usage:
    python replay_demo.py --dataset wmeddie/zenbot_rake1 --episode 0
    python replay_demo.py --auto  # LLM selects based on camera
"""

import argparse
import time
import numpy as np
import torch
import base64
import io

# Camera/Robot config
CAMERA_ARM = 6
CAMERA_OVERHEAD = 4
CAMERA_GOAL = 8
ROBOT_PORT = "/dev/ttyACM1"
ROBOT_ID = "my_awesome_follower_arm"


def load_episode_actions(dataset_repo: str, episode_idx: int):
    """Load actions from a specific episode in the dataset."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    
    # Download parquet
    parquet_path = hf_hub_download(
        repo_id=dataset_repo,
        filename="data/chunk-000/file-000.parquet",
        repo_type="dataset"
    )
    
    df = pd.read_parquet(parquet_path)
    
    # Filter to episode
    episode_df = df[df["episode_index"] == episode_idx]
    
    if len(episode_df) == 0:
        raise ValueError(f"Episode {episode_idx} not found in dataset")
    
    # Extract actions
    actions = np.stack(episode_df["action"].values)
    fps = 30  # Default, could read from meta/info.json
    
    print(f"Loaded episode {episode_idx}: {len(actions)} frames at {fps} FPS")
    return actions, fps


def setup_robot(max_speed=None):
    """Initialize robot."""
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    
    config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        max_relative_target=max_speed,
    )
    robot = SO101Follower(config)
    robot.connect()
    return robot


def setup_camera(camera_idx):
    """Initialize a single camera."""
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    
    config = OpenCVCameraConfig(
        index_or_path=camera_idx,
        width=640,
        height=480,
        fps=30,
    )
    camera = OpenCVCamera(config)
    camera.connect()
    return camera


def capture_image_base64(camera):
    """Capture image and return as base64 JPEG."""
    import cv2
    
    img = camera.async_read()  # numpy array [H, W, C]
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


def ask_claude_for_pattern(image_base64: str) -> str:
    """Ask Claude to analyze the image and decide on pattern."""
    try:
        import anthropic
    except ImportError:
        print("anthropic not installed, defaulting to 'horizontal'")
        return "horizontal"
    
    client = anthropic.Anthropic()
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": """Look at this zen garden image. I need to rake it.
Should I rake HORIZONTAL lines (left-right) or VERTICAL lines (up-down)?

Consider:
- Current state of the sand
- What would look better
- Variety if there are existing patterns

Reply with exactly one word: horizontal or vertical"""
                    }
                ]
            }
        ]
    )
    
    response = message.content[0].text.strip().lower()
    if "horizontal" in response:
        return "horizontal"
    elif "vertical" in response:
        return "vertical"
    else:
        print(f"Unexpected response: {response}, defaulting to horizontal")
        return "horizontal"


def replay_actions(robot, actions, fps, max_speed=None):
    """Replay recorded actions on the robot."""
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    dt = 1.0 / fps
    print(f"\nReplaying {len(actions)} actions at {fps} FPS...")
    print("Press Ctrl+C to stop\n")
    
    try:
        for i, action in enumerate(actions):
            start_time = time.time()
            
            # Convert to action dict
            action_dict = {f"{name}.pos": float(action[j]) for j, name in enumerate(motor_names)}
            robot.send_action(action_dict)
            
            if i % 30 == 0:
                print(f"Step {i}/{len(actions)}")
            
            # Maintain timing
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    print(f"Replay complete!")


def main():
    parser = argparse.ArgumentParser(description="Replay demonstrations with LLM selection")
    parser.add_argument("--dataset", type=str, default="wmeddie/zenbot_rake1",
                        help="HuggingFace dataset repo")
    parser.add_argument("--episode", type=int, default=None,
                        help="Specific episode to replay (0-indexed)")
    parser.add_argument("--horizontal-episode", type=int, default=0,
                        help="Episode index for horizontal pattern")
    parser.add_argument("--vertical-episode", type=int, default=1,
                        help="Episode index for vertical pattern")
    parser.add_argument("--auto", action="store_true",
                        help="Use Claude to select pattern based on camera")
    parser.add_argument("--max-speed", type=float, default=None,
                        help="Limit robot speed")
    parser.add_argument("--fps", type=int, default=None,
                        help="Override replay FPS (default: use dataset FPS)")
    args = parser.parse_args()
    
    # Setup robot
    print("Setting up robot...")
    robot = setup_robot(max_speed=args.max_speed)
    
    try:
        if args.episode is not None:
            # Replay specific episode
            episode_idx = args.episode
            print(f"Replaying episode {episode_idx}")
        elif args.auto:
            # Use Claude to decide
            print("Setting up camera for LLM analysis...")
            camera = setup_camera(CAMERA_OVERHEAD)
            
            print("Capturing image...")
            img_b64 = capture_image_base64(camera)
            camera.disconnect()
            
            print("Asking Claude for pattern recommendation...")
            pattern = ask_claude_for_pattern(img_b64)
            print(f"Claude recommends: {pattern}")
            
            episode_idx = args.horizontal_episode if pattern == "horizontal" else args.vertical_episode
        else:
            # Default to horizontal
            episode_idx = args.horizontal_episode
        
        # Load and replay
        actions, dataset_fps = load_episode_actions(args.dataset, episode_idx)
        replay_fps = args.fps if args.fps else dataset_fps
        
        replay_actions(robot, actions, replay_fps, args.max_speed)
        
    finally:
        print("Disconnecting robot...")
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
