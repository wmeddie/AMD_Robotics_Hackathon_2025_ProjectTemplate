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
from capture_goal import GoalImageCapture, SKILL_TYPES

# Configuration
DEFAULT_ROBOT = "so101"
DEFAULT_FPS = 30
DEFAULT_DATA_DIR = "./data"
DEFAULT_GOALS_DIR = "./goals"


class DemonstrationRecorder:
    """Manages the full demonstration recording workflow."""
    
    def __init__(
        self,
        skill: str,
        robot: str = DEFAULT_ROBOT,
        fps: int = DEFAULT_FPS,
        data_dir: str = DEFAULT_DATA_DIR,
        goals_dir: str = DEFAULT_GOALS_DIR,
        camera_index: int = 0
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
        
        # Initialize goal capture
        self.goal_capturer = GoalImageCapture(camera_index=camera_index)
        
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
        
        # Construct LeRobot record command
        output_dir = self.data_dir / f"episode_{self.current_episode:04d}"
        
        # Try different LeRobot command formats
        cmd_options = [
            # New LeRobot CLI format
            [
                "lerobot-record",
                "--robot", self.robot,
                "--fps", str(self.fps),
                "--output", str(output_dir),
                "--task", self.skill,
            ],
            # Alternative format
            [
                "python", "-m", "lerobot.scripts.control_robot",
                "record",
                f"--robot-path=lerobot/configs/robot/{self.robot}.yaml",
                f"--fps={self.fps}",
                f"--repo-id=local/{self.skill}",
                f"--root={str(self.data_dir)}",
                "--single-task", self.skill,
            ],
        ]
        
        success = False
        for cmd in cmd_options:
            try:
                print(f"\nRunning: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True)
                success = True
                break
            except FileNotFoundError:
                continue
            except subprocess.CalledProcessError as e:
                print(f"Recording failed with error: {e}")
                continue
        
        if not success:
            print("\nWarning: Could not run LeRobot recording command.")
            print("Please ensure LeRobot is installed and configured.")
            print("\nAlternative: Run the recording manually:")
            print(f"  lerobot-record --robot {self.robot} --task {self.skill}")
            return False
        
        return True
    
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
        default=0,
        help="Camera index for goal capture (default: 0)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Record demonstrations for all skills"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        batch_record_all_skills(
            num_episodes_per_skill=args.num_episodes,
            robot=args.robot,
            data_dir=args.data_dir,
            goals_dir=args.goals_dir
        )
    else:
        if args.skill is None:
            parser.error("--skill is required unless using --batch mode")
        
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
