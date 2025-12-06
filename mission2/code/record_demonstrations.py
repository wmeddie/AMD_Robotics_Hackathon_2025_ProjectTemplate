#!/usr/bin/env python3
"""
Demonstration Recording Wrapper for Zen Garden Robot

This script wraps the LeRobot recording tools to provide a streamlined workflow
for collecting goal-conditioned training data:

1. Capture goal image (target pattern)
2. Reset robot to start position
3. Record teleoperation demonstration
4. Save with goal image reference

Usage:
    # Record demonstrations for a single skill
    python record_demonstrations.py --skill flatten --num_episodes 15

    # Record with specific robot config
    python record_demonstrations.py --skill zigzag --robot so101 --num_episodes 10
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import our goal capture utility
from capture_goal import (
    GoalImageCapture, SKILL_TYPES,
    CAMERA_GOAL, CAMERA_OVERHEAD, CAMERA_ARM,
    CAMERA_WIDTH, CAMERA_HEIGHT
)

# Configuration
DEFAULT_ROBOT = "so101_follower"
DEFAULT_FPS = 30
DEFAULT_DATA_DIR = "./data"
DEFAULT_GOALS_DIR = "./goals"

# Robot hardware configuration
ROBOT_PORT = "/dev/ttyACM1"
ROBOT_ID = "my_awesome_follower_arm"
TELEOP_TYPE = "so101_leader"
TELEOP_PORT = "/dev/ttyACM0"
TELEOP_ID = "my_awesome_leader_arm"

# Camera setup:
# - CAMERA_GOAL (8): For capturing goal images
# - CAMERA_OVERHEAD (4): Overhead view for planner (top camera)
# - CAMERA_ARM (6): Robot arm camera for observations (front camera)

# Skill name mapping for dataset tasks
SKILL_TASK_NAMES = {
    "flatten": "Flatten Sand",
    "zigzag": "Draw Zigzag",
    "circle": "Draw Circles",
    "stamp": "Stamp Triangle",
    "place_rock": "Place Rock",
}


class DemonstrationRecorder:
    """Manages the full demonstration recording workflow."""
    
    def __init__(
        self,
        skill: str,
        robot: str = DEFAULT_ROBOT,
        fps: int = DEFAULT_FPS,
        data_dir: str = DEFAULT_DATA_DIR,
        goals_dir: str = DEFAULT_GOALS_DIR,
        camera_index: int = CAMERA_GOAL  # Use goal camera by default
    ):
        """
        Initialize the demonstration recorder.
        
        Args:
            skill: The skill being recorded (flatten, zigzag, etc.)
            robot: Robot type (e.g., so101)
            fps: Recording frames per second
            data_dir: Base directory for recorded data
            goals_dir: Base directory for goal images
            camera_index: Camera index for goal capture
        """
        self.skill = skill
        self.robot = robot
        self.fps = fps
        self.data_dir = Path(data_dir) / skill
        self.goals_dir = Path(goals_dir) / skill
        self.camera_index = camera_index
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.goals_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize goal capture with correct resolution
        self.goal_capturer = GoalImageCapture(
            camera_index=camera_index,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT
        )
        
        # Track recorded episodes
        self.current_episode = self._get_next_episode_number()
    
    def _get_next_episode_number(self) -> int:
        """Determine the next episode number based on existing data."""
        existing_episodes = list(self.data_dir.glob("episode_*"))
        if not existing_episodes:
            return 0
        
        # Extract episode numbers
        episode_nums = []
        for ep_dir in existing_episodes:
            try:
                num = int(ep_dir.name.split("_")[1])
                episode_nums.append(num)
            except (ValueError, IndexError):
                continue
        
        return max(episode_nums) + 1 if episode_nums else 0
    
    def capture_goal(self) -> Optional[str]:
        """Capture the goal image for the current episode."""
        goal_filename = f"{self.skill}_goal_{self.current_episode:04d}.jpg"
        goal_path = self.goals_dir / goal_filename
        
        print(f"\n{'='*60}")
        print(f"STEP 1: Capture Goal Image")
        print(f"{'='*60}")
        print(f"\nPosition the zen garden to show the TARGET PATTERN.")
        print(f"This is what the robot should achieve by the end of the demo.")
        input("\nPress ENTER when the target pattern is ready...")
        
        self.goal_capturer.connect()
        try:
            if self.goal_capturer.capture_with_preview(str(goal_path)):
                return str(goal_path)
            else:
                return None
        finally:
            self.goal_capturer.disconnect()
    
    def record_episode(self) -> bool:
        """Record a single teleoperation episode using LeRobot."""
        print(f"\n{'='*60}")
        print(f"STEP 2: Record Demonstration")
        print(f"{'='*60}")
        print(f"\nPrepare to teleoperate the robot to achieve the goal pattern.")
        print(f"Skill: {self.skill}")
        print(f"Episode: {self.current_episode}")
        
        input("\nPress ENTER when ready to start recording...")
        
        # Get task name for this skill
        task_name = SKILL_TASK_NAMES.get(self.skill, self.skill.replace("_", " ").title())
        
        # Build camera config string
        cameras_config = (
            f"{{ front: {{type: opencv, index_or_path: {CAMERA_ARM}, "
            f"width: {CAMERA_WIDTH}, height: {CAMERA_HEIGHT}, fps: {self.fps}}}, "
            f"top: {{type: opencv, index_or_path: {CAMERA_OVERHEAD}, "
            f"width: {CAMERA_WIDTH}, height: {CAMERA_HEIGHT}, fps: {self.fps}}} }}"
        )
        
        # Build repo_id for this skill's dataset
        repo_id = f"local/{self.skill}"
        
        # Construct LeRobot record command (LeRobot 0.4.x format)
        cmd = [
            "lerobot-record",
            f"--robot.type={self.robot}",
            f"--robot.port={ROBOT_PORT}",
            f"--robot.id={ROBOT_ID}",
            f"--robot.cameras={cameras_config}",
            f"--teleop.type={TELEOP_TYPE}",
            f"--teleop.port={TELEOP_PORT}",
            f"--teleop.id={TELEOP_ID}",
            "--display_data=true",
            f"--dataset.num_episodes=1",
            f"--dataset.single_task={task_name}",
            f"--dataset.repo_id={repo_id}",
        ]
        
        try:
            print(f"\nRunning: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            return True
        except FileNotFoundError:
            print("\nError: lerobot-record command not found.")
            print("Please ensure LeRobot is installed: pip install lerobot==0.4.1")
            return False
        except subprocess.CalledProcessError as e:
            print(f"\nRecording failed with error: {e}")
            return False
    
    def record_session(self, num_episodes: int = 1) -> list:
        """
        Record a full session of demonstrations.
        
        Args:
            num_episodes: Number of episodes to record
            
        Returns:
            List of tuples (episode_num, goal_path, success)
        """
        results = []
        
        print(f"\n{'#'*60}")
        print(f"# ZEN GARDEN DATA COLLECTION")
        print(f"# Skill: {self.skill}")
        print(f"# Episodes: {num_episodes}")
        print(f"# Starting episode: {self.current_episode}")
        print(f"{'#'*60}")
        
        for i in range(num_episodes):
            print(f"\n\n{'*'*60}")
            print(f"* EPISODE {self.current_episode} ({i+1}/{num_episodes})")
            print(f"{'*'*60}")
            
            # Step 1: Capture goal
            goal_path = self.capture_goal()
            if goal_path is None:
                print("\nGoal capture cancelled. Skipping this episode.")
                continue
            
            # Reset for recording
            print(f"\n{'='*60}")
            print(f"STEP 2: Reset Environment")
            print(f"{'='*60}")
            print(f"\nReset the zen garden to a NEUTRAL/STARTING position.")
            print(f"The robot will teleoperate to achieve the goal pattern.")
            input("\nPress ENTER when the garden is reset and robot is ready...")
            
            # Step 3: Record demonstration
            success = self.record_episode()
            
            results.append((self.current_episode, goal_path, success))
            self.current_episode += 1
            
            if i < num_episodes - 1:
                print(f"\nEpisode complete. Preparing for next episode...")
                time.sleep(2)
        
        # Summary
        print(f"\n\n{'#'*60}")
        print(f"# RECORDING SESSION COMPLETE")
        print(f"{'#'*60}")
        successful = sum(1 for _, _, s in results if s)
        print(f"\nSuccessful recordings: {successful}/{len(results)}")
        print(f"Data saved to: {self.data_dir}")
        print(f"Goals saved to: {self.goals_dir}")
        
        return results


def record_skill_dataset(
    skill: str,
    num_episodes: int = 15,
    robot: str = DEFAULT_ROBOT,
    repo_id: Optional[str] = None
):
    """
    Record a full dataset for a skill using lerobot-record directly.
    
    This runs the lerobot-record command once for all episodes,
    which is more efficient than recording one at a time.
    
    Args:
        skill: Skill name (flatten, zigzag, circle, stamp, place_rock)
        num_episodes: Number of episodes to record
        robot: Robot type
        repo_id: HuggingFace repo ID (default: local/{skill})
    """
    task_name = SKILL_TASK_NAMES.get(skill, skill.replace("_", " ").title())
    repo_id = repo_id or f"local/{skill}"
    
    # Build camera config string
    cameras_config = (
        f"{{ front: {{type: opencv, index_or_path: {CAMERA_ARM}, "
        f"width: {CAMERA_WIDTH}, height: {CAMERA_HEIGHT}, fps: 30}}, "
        f"top: {{type: opencv, index_or_path: {CAMERA_OVERHEAD}, "
        f"width: {CAMERA_WIDTH}, height: {CAMERA_HEIGHT}, fps: 30}} }}"
    )
    
    cmd = [
        "lerobot-record",
        f"--robot.type={robot}",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cameras={cameras_config}",
        f"--teleop.type={TELEOP_TYPE}",
        f"--teleop.port={TELEOP_PORT}",
        f"--teleop.id={TELEOP_ID}",
        "--display_data=true",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task={task_name}",
        f"--dataset.repo_id={repo_id}",
    ]
    
    print(f"\n{'#'*60}")
    print(f"# RECORDING: {skill.upper()}")
    print(f"# Task: {task_name}")
    print(f"# Episodes: {num_episodes}")
    print(f"# Repo: {repo_id}")
    print(f"{'#'*60}")
    print(f"\nCommand:\n{' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n Recording complete for {skill}!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nRecording failed: {e}")
        return False


def batch_record_all_skills(
    num_episodes_per_skill: int = 15,
    robot: str = DEFAULT_ROBOT,
    data_dir: str = DEFAULT_DATA_DIR,
    goals_dir: str = DEFAULT_GOALS_DIR
):
    """Record demonstrations for all skills."""
    
    print(f"\n{'#'*60}")
    print(f"# BATCH RECORDING - ALL SKILLS")
    print(f"# Episodes per skill: {num_episodes_per_skill}")
    print(f"{'#'*60}")
    
    all_results = {}
    
    for skill in SKILL_TYPES:
        print(f"\n\n{'@'*60}")
        print(f"@ SKILL: {skill.upper()}")
        print(f"{'@'*60}")
        
        input(f"\nPrepare for {skill} demonstrations. Press ENTER to continue...")
        
        recorder = DemonstrationRecorder(
            skill=skill,
            robot=robot,
            data_dir=data_dir,
            goals_dir=goals_dir
        )
        
        results = recorder.record_session(num_episodes_per_skill)
        all_results[skill] = results
    
    # Final summary
    print(f"\n\n{'#'*60}")
    print(f"# ALL RECORDINGS COMPLETE")
    print(f"{'#'*60}")
    
    for skill, results in all_results.items():
        successful = sum(1 for _, _, s in results if s)
        print(f"  {skill}: {successful}/{len(results)} successful")


def main():
    parser = argparse.ArgumentParser(
        description="Record demonstrations for goal-conditioned SmolVLA training"
    )
    parser.add_argument(
        "--skill",
        type=str,
        choices=SKILL_TYPES,
        help="Skill to record demonstrations for"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=15,
        help="Number of episodes to record (default: 15)"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=DEFAULT_ROBOT,
        help=f"Robot type (default: {DEFAULT_ROBOT})"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Recording FPS (default: {DEFAULT_FPS})"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Data output directory (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--goals_dir",
        type=str,
        default=DEFAULT_GOALS_DIR,
        help=f"Goals output directory (default: {DEFAULT_GOALS_DIR})"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_GOAL,
        help="Camera index for goal capture (default: 0)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Record demonstrations for all skills"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Simple mode: just run lerobot-record directly (recommended)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace repo ID for dataset (default: local/{skill})"
    )
    
    args = parser.parse_args()
    
    if args.skill is None and not args.batch:
        parser.error("--skill is required unless using --batch mode")
    
    if args.simple or True:  # Default to simple mode
        # Simple mode: just run lerobot-record directly
        if args.batch:
            for skill in SKILL_TYPES:
                print(f"\n\nReady to record: {skill}")
                input("Press ENTER when ready...")
                record_skill_dataset(
                    skill=skill,
                    num_episodes=args.num_episodes,
                    robot=args.robot,
                    repo_id=args.repo_id
                )
        else:
            record_skill_dataset(
                skill=args.skill,
                num_episodes=args.num_episodes,
                robot=args.robot,
                repo_id=args.repo_id
            )
    else:
        # Original mode with goal capture
        if args.batch:
            batch_record_all_skills(
                num_episodes_per_skill=args.num_episodes,
                robot=args.robot,
                data_dir=args.data_dir,
                goals_dir=args.goals_dir
            )
        else:
            recorder = DemonstrationRecorder(
                skill=args.skill,
                robot=args.robot,
                fps=args.fps,
                data_dir=args.data_dir,
                goals_dir=args.goals_dir,
                camera_index=args.camera
            )
            
            recorder.record_session(args.num_episodes)


if __name__ == "__main__":
    main()
