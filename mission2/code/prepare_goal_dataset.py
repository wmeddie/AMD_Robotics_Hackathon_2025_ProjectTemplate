#!/usr/bin/env python3
"""
Dataset Preparation Script for Goal-Conditioned SmolVLA Training

This script takes a LeRobot dataset and augments it with goal images for
goal-conditioned policy training. 

Two modes of operation:
1. Use final frame as goal: Extract the last observation frame of each episode
2. Use separate goal images: Match recorded episodes with pre-captured goal images

Usage:
    # Use final frame as goal
    python prepare_goal_dataset.py --data_dir ./data/flatten --output_dir ./data/flatten_goal
    
    # Use separate goal images
    python prepare_goal_dataset.py --data_dir ./data/flatten --goals_dir ./goals/flatten --output_dir ./data/flatten_goal
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try importing LeRobot dataset utilities
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.push_dataset_to_hub.utils import save_images_concurrently
    import torch
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: LeRobot not installed. Using basic dataset handling.")

# Try importing image libraries
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


class GoalDatasetPreparer:
    """Prepares goal-conditioned datasets from LeRobot recordings."""
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        goals_dir: Optional[str] = None,
        use_final_frame: bool = True,
        camera_key: str = "observation.images.top"
    ):
        """
        Initialize the dataset preparer.
        
        Args:
            data_dir: Path to the source LeRobot dataset
            output_dir: Path to save the augmented dataset
            goals_dir: Path to pre-captured goal images (optional)
            use_final_frame: If True and no goals_dir, use final frame as goal
            camera_key: Key for the camera observations in the dataset
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.goals_dir = Path(goals_dir) if goals_dir else None
        self.use_final_frame = use_final_frame
        self.camera_key = camera_key
        
        # Validate inputs
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        if self.goals_dir and not self.goals_dir.exists():
            raise ValueError(f"Goals directory not found: {self.goals_dir}")
    
    def load_lerobot_dataset(self) -> Dict:
        """Load a LeRobot dataset from disk."""
        if LEROBOT_AVAILABLE:
            # Use LeRobot's native loading
            dataset = LeRobotDataset(str(self.data_dir))
            return dataset
        else:
            # Fallback: load from raw files
            return self._load_dataset_manually()
    
    def _load_dataset_manually(self) -> Dict:
        """Manually load dataset when LeRobot is not available."""
        dataset = {
            "episodes": [],
            "meta": {}
        }
        
        # Load metadata if it exists
        meta_path = self.data_dir / "meta_data.json"
        if meta_path.exists():
            with open(meta_path) as f:
                dataset["meta"] = json.load(f)
        
        # Load episode info
        episodes_path = self.data_dir / "episodes.json"
        if episodes_path.exists():
            with open(episodes_path) as f:
                dataset["episodes"] = json.load(f)
        
        # Try loading parquet data
        parquet_path = self.data_dir / "data"
        if parquet_path.exists():
            try:
                import pandas as pd
                for parquet_file in parquet_path.glob("*.parquet"):
                    df = pd.read_parquet(parquet_file)
                    dataset["data"] = df
                    break
            except ImportError:
                print("Warning: pandas not available for parquet loading")
        
        return dataset
    
    def get_episode_indices(self, dataset) -> List[Tuple[int, int]]:
        """Get start and end indices for each episode."""
        if LEROBOT_AVAILABLE and hasattr(dataset, 'episode_data_index'):
            # Use LeRobot's episode indexing
            starts = dataset.episode_data_index['from'].tolist()
            ends = dataset.episode_data_index['to'].tolist()
            return list(zip(starts, ends))
        
        # Fallback: infer from data
        if "episodes" in dataset and dataset["episodes"]:
            indices = []
            for ep in dataset["episodes"]:
                indices.append((ep.get("start", 0), ep.get("end", 0)))
            return indices
        
        # Last resort: treat all data as one episode
        if "data" in dataset:
            return [(0, len(dataset["data"]))]
        
        return []
    
    def extract_goal_from_episode(
        self,
        dataset,
        episode_idx: int,
        start_idx: int,
        end_idx: int
    ) -> np.ndarray:
        """Extract goal image from an episode (uses final frame)."""
        if LEROBOT_AVAILABLE:
            # Get the last frame of the episode
            last_frame_idx = end_idx - 1
            sample = dataset[last_frame_idx]
            
            # Extract the camera image
            if self.camera_key in sample:
                goal_image = sample[self.camera_key]
                if isinstance(goal_image, torch.Tensor):
                    goal_image = goal_image.numpy()
                # Convert from CHW to HWC if needed
                if goal_image.shape[0] in [1, 3]:
                    goal_image = np.transpose(goal_image, (1, 2, 0))
                return goal_image
        
        # Fallback: try to load from image files
        images_dir = self.data_dir / "images" / self.camera_key.replace(".", "_")
        if images_dir.exists():
            # Find the last image for this episode
            image_files = sorted(images_dir.glob(f"episode_{episode_idx:06d}_*.png"))
            if image_files:
                last_image = image_files[-1]
                if PIL_AVAILABLE:
                    return np.array(Image.open(last_image))
                elif CV2_AVAILABLE:
                    return cv2.imread(str(last_image))
        
        raise RuntimeError(f"Could not extract goal image for episode {episode_idx}")
    
    def load_goal_image(self, skill: str, episode_idx: int) -> np.ndarray:
        """Load a pre-captured goal image."""
        if self.goals_dir is None:
            raise ValueError("Goals directory not set")
        
        # Try different naming conventions
        patterns = [
            f"{skill}_goal_{episode_idx:04d}.jpg",
            f"{skill}_goal_{episode_idx:04d}.png",
            f"goal_{episode_idx:04d}.jpg",
            f"goal_{episode_idx:04d}.png",
            f"{episode_idx:04d}.jpg",
            f"{episode_idx:04d}.png",
        ]
        
        for pattern in patterns:
            goal_path = self.goals_dir / pattern
            if goal_path.exists():
                if PIL_AVAILABLE:
                    return np.array(Image.open(goal_path))
                elif CV2_AVAILABLE:
                    img = cv2.imread(str(goal_path))
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Try using the first available goal image for all episodes
        goal_files = list(self.goals_dir.glob("*.jpg")) + list(self.goals_dir.glob("*.png"))
        if goal_files:
            print(f"Warning: Using {goal_files[0].name} as goal for episode {episode_idx}")
            if PIL_AVAILABLE:
                return np.array(Image.open(goal_files[0]))
            elif CV2_AVAILABLE:
                img = cv2.imread(str(goal_files[0]))
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        raise FileNotFoundError(
            f"No goal image found for episode {episode_idx} in {self.goals_dir}"
        )
    
    def prepare_dataset(self, skill: str = "unknown") -> Path:
        """
        Prepare the goal-conditioned dataset.
        
        Args:
            skill: Name of the skill (used for goal image lookup)
            
        Returns:
            Path to the output dataset directory
        """
        print(f"\nPreparing goal-conditioned dataset")
        print(f"  Source: {self.data_dir}")
        print(f"  Output: {self.output_dir}")
        print(f"  Skill: {skill}")
        print(f"  Goals: {self.goals_dir or 'Using final frame'}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load source dataset
        print("\nLoading source dataset...")
        dataset = self.load_lerobot_dataset()
        
        # Get episode boundaries
        episode_indices = self.get_episode_indices(dataset)
        num_episodes = len(episode_indices)
        print(f"Found {num_episodes} episodes")
        
        # Create output structure
        goals_output_dir = self.output_dir / "goals"
        goals_output_dir.mkdir(exist_ok=True)
        
        # Process each episode
        goal_mapping = {}
        
        for ep_idx, (start_idx, end_idx) in enumerate(episode_indices):
            print(f"Processing episode {ep_idx + 1}/{num_episodes}...", end=" ")
            
            # Get goal image
            try:
                if self.goals_dir:
                    goal_image = self.load_goal_image(skill, ep_idx)
                elif self.use_final_frame:
                    goal_image = self.extract_goal_from_episode(
                        dataset, ep_idx, start_idx, end_idx
                    )
                else:
                    raise ValueError("No goal source specified")
                
                # Save goal image
                goal_filename = f"goal_episode_{ep_idx:06d}.png"
                goal_path = goals_output_dir / goal_filename
                
                if PIL_AVAILABLE:
                    Image.fromarray(goal_image).save(goal_path)
                elif CV2_AVAILABLE:
                    cv2.imwrite(str(goal_path), cv2.cvtColor(goal_image, cv2.COLOR_RGB2BGR))
                
                goal_mapping[ep_idx] = {
                    "goal_path": str(goal_path.relative_to(self.output_dir)),
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "num_frames": end_idx - start_idx
                }
                
                print("OK")
                
            except Exception as e:
                print(f"FAILED: {e}")
                continue
        
        # Copy original dataset files
        print("\nCopying original dataset files...")
        for item in self.data_dir.iterdir():
            if item.name != "goals":  # Don't overwrite our goals
                dest = self.output_dir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        
        # Save goal mapping
        mapping_path = self.output_dir / "goal_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump({
                "skill": skill,
                "num_episodes": num_episodes,
                "use_final_frame": self.use_final_frame and self.goals_dir is None,
                "goals_dir": str(self.goals_dir) if self.goals_dir else None,
                "camera_key": self.camera_key,
                "episodes": goal_mapping
            }, f, indent=2)
        
        print(f"\nDataset prepared successfully!")
        print(f"  Output: {self.output_dir}")
        print(f"  Goal mapping: {mapping_path}")
        print(f"  Episodes processed: {len(goal_mapping)}/{num_episodes}")
        
        return self.output_dir


def create_combined_dataset(
    skill_datasets: Dict[str, str],
    output_dir: str,
    task_column: str = "task"
) -> Path:
    """
    Combine multiple skill datasets into one multi-task dataset.
    
    Args:
        skill_datasets: Dict mapping skill names to dataset directories
        output_dir: Path to save combined dataset
        task_column: Name of the column to store task/skill labels
        
    Returns:
        Path to the combined dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    combined_mapping = {
        "skills": list(skill_datasets.keys()),
        "datasets": {},
        "total_episodes": 0
    }
    
    episode_offset = 0
    
    for skill, dataset_dir in skill_datasets.items():
        mapping_path = Path(dataset_dir) / "goal_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                skill_mapping = json.load(f)
            
            # Offset episode indices for combined dataset
            for ep_idx, ep_data in skill_mapping["episodes"].items():
                new_idx = int(ep_idx) + episode_offset
                combined_mapping["datasets"][str(new_idx)] = {
                    "skill": skill,
                    "original_episode": int(ep_idx),
                    "source_dir": str(dataset_dir),
                    **ep_data
                }
            
            episode_offset += len(skill_mapping["episodes"])
    
    combined_mapping["total_episodes"] = episode_offset
    
    # Save combined mapping
    combined_path = output_path / "combined_goal_mapping.json"
    with open(combined_path, "w") as f:
        json.dump(combined_mapping, f, indent=2)
    
    print(f"Combined dataset mapping created at: {combined_path}")
    print(f"Total episodes: {episode_offset}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare goal-conditioned dataset for SmolVLA training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to source LeRobot dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save augmented dataset"
    )
    parser.add_argument(
        "--goals_dir",
        type=str,
        default=None,
        help="Path to pre-captured goal images (optional)"
    )
    parser.add_argument(
        "--skill",
        type=str,
        default="unknown",
        help="Skill name for this dataset"
    )
    parser.add_argument(
        "--camera_key",
        type=str,
        default="observation.images.top",
        help="Key for camera observations in dataset"
    )
    parser.add_argument(
        "--use_final_frame",
        action="store_true",
        default=True,
        help="Use final frame of each episode as goal (default: True)"
    )
    
    args = parser.parse_args()
    
    preparer = GoalDatasetPreparer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        goals_dir=args.goals_dir,
        use_final_frame=args.use_final_frame,
        camera_key=args.camera_key
    )
    
    preparer.prepare_dataset(skill=args.skill)


if __name__ == "__main__":
    main()
