#!/usr/bin/env python3
"""
Replay recorded demonstrations with LLM-based skill selection.

Flow:
1. Capture overhead camera image
2. Send to Claude to analyze garden and select appropriate skill
3. Replay the corresponding demonstration from the skill's dataset

Usage:
    # Manual replay from specific dataset
    python replay_demo.py --dataset wmeddie/zenbot_rake_horizontal --episode 0
    
    # Auto mode: Claude selects skill based on camera
    python replay_demo.py --auto
    
    # Continuous auto mode
    python replay_demo.py --auto --loop
"""

import argparse
import time
import numpy as np
import torch
import base64
import io
import random

# Camera/Robot config
CAMERA_ARM = 6
CAMERA_OVERHEAD = 4
CAMERA_GOAL = 8
ROBOT_PORT = "/dev/ttyACM1"
ROBOT_ID = "my_awesome_follower_arm"

# Skill-to-dataset mapping
# Each skill maps to a HuggingFace dataset repo
SKILL_DATASETS = {
    "rake_horizontal": "wmeddie/zenbot_rake_horizontal",
    "rake_vertical": "wmeddie/zenbot_rake_vertical",
    "flatten": "wmeddie/zenbot_flatten",
    "place_rock": "wmeddie/zenbot_place_rock",
    "circle": "wmeddie/zenbot_circle",
}

# Available skills for Claude to choose from
AVAILABLE_SKILLS = ["rake_horizontal", "rake_vertical"]


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


def ask_claude_for_skill(image_base64: str, available_skills: list = None) -> str:
    """Ask Claude to analyze the zen garden and select the best skill to execute."""
    if available_skills is None:
        available_skills = AVAILABLE_SKILLS
    
    try:
        import anthropic
    except ImportError:
        print("anthropic not installed, defaulting to random skill")
        return random.choice(available_skills)
    
    client = anthropic.Anthropic()
    
    skills_list = "\n".join(f"- {skill}" for skill in available_skills)
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
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
                        "text": f"""Analyze this zen garden overhead image. I have a robot arm that can perform different skills to rake the sand.

Available skills:
{skills_list}

Based on the current state of the garden, which skill should the robot perform next?

Consider:
- Current state of the sand (smooth, already has patterns, messy?)
- If there are existing lines, would perpendicular lines create a nice grid pattern?
- Variety and aesthetics

Reply with ONLY the skill name, nothing else. For example: rake_horizontal"""
                    }
                ]
            }
        ]
    )
    
    response = message.content[0].text.strip().lower()
    
    # Match response to available skills
    for skill in available_skills:
        if skill in response:
            return skill
    
    # Fallback: check for partial matches
    if "horizontal" in response:
        for skill in available_skills:
            if "horizontal" in skill:
                return skill
    if "vertical" in response:
        for skill in available_skills:
            if "vertical" in skill:
                return skill
    
    print(f"Unexpected response: {response}, defaulting to {available_skills[0]}")
    return available_skills[0]


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


def get_episode_count(dataset_repo: str) -> int:
    """Get the number of episodes in a dataset."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    
    try:
        parquet_path = hf_hub_download(
            repo_id=dataset_repo,
            filename="data/chunk-000/file-000.parquet",
            repo_type="dataset"
        )
        df = pd.read_parquet(parquet_path)
        return df["episode_index"].max() + 1
    except Exception as e:
        print(f"Warning: Could not get episode count: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Replay demonstrations with LLM selection")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset repo (overrides skill-based selection)")
    parser.add_argument("--skill", type=str, default=None,
                        choices=list(SKILL_DATASETS.keys()),
                        help="Skill to replay (uses skill's dataset)")
    parser.add_argument("--episode", type=int, default=None,
                        help="Specific episode to replay (0-indexed, default: random)")
    parser.add_argument("--auto", action="store_true",
                        help="Use Claude to select skill based on camera")
    parser.add_argument("--loop", action="store_true",
                        help="Continuously loop: analyze -> execute -> repeat")
    parser.add_argument("--loop-delay", type=float, default=3.0,
                        help="Delay between loop iterations (default: 3s)")
    parser.add_argument("--max-speed", type=float, default=None,
                        help="Limit robot speed")
    parser.add_argument("--fps", type=int, default=None,
                        help="Override replay FPS (default: use dataset FPS)")
    parser.add_argument("--skills", type=str, nargs="+",
                        default=AVAILABLE_SKILLS,
                        help=f"Skills to consider in auto mode (default: {AVAILABLE_SKILLS})")
    args = parser.parse_args()
    
    # Setup robot
    print("Setting up robot...")
    robot = setup_robot(max_speed=args.max_speed)
    
    camera = None
    if args.auto:
        print("Setting up overhead camera for LLM analysis...")
        camera = setup_camera(CAMERA_OVERHEAD)
    
    try:
        iteration = 0
        while True:
            iteration += 1
            if args.loop:
                print(f"\n{'='*50}")
                print(f"ITERATION {iteration}")
                print(f"{'='*50}")
            
            # Determine which dataset/episode to use
            if args.dataset:
                # Explicit dataset provided
                dataset_repo = args.dataset
                episode_idx = args.episode if args.episode is not None else 0
                print(f"Using explicit dataset: {dataset_repo}, episode {episode_idx}")
                
            elif args.skill:
                # Explicit skill provided
                if args.skill not in SKILL_DATASETS:
                    print(f"Error: No dataset configured for skill '{args.skill}'")
                    print(f"Available skills: {list(SKILL_DATASETS.keys())}")
                    break
                dataset_repo = SKILL_DATASETS[args.skill]
                episode_idx = args.episode if args.episode is not None else 0
                print(f"Using skill '{args.skill}': {dataset_repo}, episode {episode_idx}")
                
            elif args.auto:
                # Ask Claude to decide
                print("Capturing image for analysis...")
                img_b64 = capture_image_base64(camera)
                
                print("Asking Claude to select skill...")
                skill = ask_claude_for_skill(img_b64, args.skills)
                print(f"Claude selected: {skill}")
                
                if skill not in SKILL_DATASETS:
                    print(f"Warning: No dataset for skill '{skill}', skipping")
                    if not args.loop:
                        break
                    time.sleep(args.loop_delay)
                    continue
                
                dataset_repo = SKILL_DATASETS[skill]
                
                # Random episode from the dataset
                num_episodes = get_episode_count(dataset_repo)
                episode_idx = random.randint(0, num_episodes - 1)
                print(f"Selected random episode {episode_idx} from {num_episodes} available")
                
            else:
                # Default: first available skill
                default_skill = AVAILABLE_SKILLS[0]
                dataset_repo = SKILL_DATASETS[default_skill]
                episode_idx = args.episode if args.episode is not None else 0
                print(f"Using default skill '{default_skill}': {dataset_repo}")
            
            # Load and replay
            try:
                actions, dataset_fps = load_episode_actions(dataset_repo, episode_idx)
                replay_fps = args.fps if args.fps else dataset_fps
                replay_actions(robot, actions, replay_fps, args.max_speed)
            except Exception as e:
                print(f"Error during replay: {e}")
                if not args.loop:
                    raise
            
            # Exit or loop
            if not args.loop:
                break
            
            print(f"\nWaiting {args.loop_delay}s before next iteration...")
            print("Press Ctrl+C to stop")
            time.sleep(args.loop_delay)
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        if camera:
            camera.disconnect()
        print("Disconnecting robot...")
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
