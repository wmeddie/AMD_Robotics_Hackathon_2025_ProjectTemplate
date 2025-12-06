#!/usr/bin/env python3
"""
Goal Image Capture Utility for Zen Garden Robot

This script captures target pattern images before teleoperation recording.
The goal images are used to condition the SmolVLA model during training.

Usage:
    python capture_goal.py --skill flatten --output ./goals/
    python capture_goal.py --skill zigzag --episode 1 --output ./goals/
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# LeRobot camera utilities (if available) - LeRobot 0.4.x API
try:
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: LeRobot not installed. Using fallback OpenCV capture.")


SKILL_TYPES = ["flatten", "zigzag", "circle", "stamp", "place_rock"]


class GoalImageCapture:
    """Captures goal images for goal-conditioned policy training."""
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Initialize the goal image capture system.
        
        Args:
            camera_index: Index of the camera to use (default: 0 for top camera)
            width: Image width in pixels
            height: Image height in pixels
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.camera = None
        
    def connect(self):
        """Connect to the camera."""
        if LEROBOT_AVAILABLE:
            config = OpenCVCameraConfig(
                camera_index=self.camera_index,
                fps=30,
                width=self.width,
                height=self.height,
            )
            self.camera = OpenCVCamera(config)
            self.camera.connect()
        else:
            self.camera = cv2.VideoCapture(self.camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")
        print(f"Camera {self.camera_index} connected successfully.")
    
    def disconnect(self):
        """Disconnect from the camera."""
        if self.camera is not None:
            if LEROBOT_AVAILABLE:
                self.camera.disconnect()
            else:
                self.camera.release()
            self.camera = None
    
    def capture_frame(self) -> np.ndarray:
        """Capture a single frame from the camera."""
        if LEROBOT_AVAILABLE:
            frame = self.camera.read()
            # LeRobot returns CHW format, convert to HWC for display/saving
            if frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.camera.read()
            if not ret:
                raise RuntimeError("Failed to capture frame")
        return frame
    
    def capture_with_preview(self, save_path: str, preview_window: str = "Goal Capture"):
        """
        Capture goal image with live preview.
        
        Shows a live preview window. Press:
        - SPACE or ENTER to capture and save
        - 'q' or ESC to quit without saving
        
        Args:
            save_path: Path to save the captured image
            preview_window: Name of the preview window
            
        Returns:
            True if image was captured and saved, False otherwise
        """
        print(f"\nGoal Image Capture")
        print("=" * 40)
        print(f"Output: {save_path}")
        print("\nControls:")
        print("  SPACE/ENTER - Capture and save")
        print("  q/ESC       - Quit without saving")
        print("=" * 40)
        
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_window, self.width, self.height)
        
        captured = False
        
        while True:
            frame = self.capture_frame()
            
            # Add overlay text
            display_frame = frame.copy()
            cv2.putText(
                display_frame, 
                "Press SPACE to capture goal image",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.imshow(preview_window, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # SPACE or ENTER to capture
            if key in [ord(' '), 13]:
                # Ensure output directory exists
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
                
                # Save the frame (without overlay)
                cv2.imwrite(save_path, frame)
                print(f"\nGoal image saved to: {save_path}")
                captured = True
                
                # Show captured frame briefly
                cv2.putText(
                    frame,
                    "CAPTURED!",
                    (self.width // 2 - 80, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3
                )
                cv2.imshow(preview_window, frame)
                cv2.waitKey(1000)
                break
            
            # Q or ESC to quit
            elif key in [ord('q'), 27]:
                print("\nCapture cancelled.")
                break
        
        cv2.destroyAllWindows()
        return captured
    
    def capture_multiple(self, output_dir: str, skill: str, num_images: int = 1, 
                         start_episode: int = 0):
        """
        Capture multiple goal images for a skill.
        
        Args:
            output_dir: Directory to save images
            skill: Skill type (flatten, zigzag, etc.)
            num_images: Number of images to capture
            start_episode: Starting episode number
            
        Returns:
            List of saved file paths
        """
        saved_paths = []
        
        for i in range(num_images):
            episode = start_episode + i
            filename = f"{skill}_goal_{episode:04d}.jpg"
            save_path = os.path.join(output_dir, skill, filename)
            
            print(f"\nCapturing goal image {i+1}/{num_images} for skill '{skill}'")
            
            if self.capture_with_preview(save_path):
                saved_paths.append(save_path)
            else:
                print(f"Skipped image {i+1}")
            
            if i < num_images - 1:
                print("\nPrepare the next target pattern...")
                time.sleep(1)
        
        return saved_paths


def batch_capture_goals(output_dir: str, skills: list = None, episodes_per_skill: int = 1):
    """
    Batch capture goal images for multiple skills.
    
    Args:
        output_dir: Base output directory
        skills: List of skills to capture (default: all skills)
        episodes_per_skill: Number of goal images per skill
    """
    if skills is None:
        skills = SKILL_TYPES
    
    capturer = GoalImageCapture()
    capturer.connect()
    
    try:
        for skill in skills:
            print(f"\n{'='*50}")
            print(f"SKILL: {skill.upper()}")
            print(f"{'='*50}")
            print(f"Please prepare the target pattern for '{skill}'")
            input("Press ENTER when ready...")
            
            capturer.capture_multiple(output_dir, skill, episodes_per_skill)
    
    finally:
        capturer.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Capture goal images for goal-conditioned SmolVLA training"
    )
    parser.add_argument(
        "--skill",
        type=str,
        choices=SKILL_TYPES,
        help="Skill type to capture goal for"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode number for naming (default: 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./goals",
        help="Output directory for goal images (default: ./goals)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to use (default: 0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Image width (default: 640)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height (default: 480)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch capture mode: capture goals for all skills"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of goal images to capture per skill in batch mode (default: 1)"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch capture mode
        batch_capture_goals(
            output_dir=args.output,
            skills=[args.skill] if args.skill else None,
            episodes_per_skill=args.num_episodes
        )
    else:
        # Single capture mode
        if args.skill is None:
            parser.error("--skill is required unless using --batch mode")
        
        capturer = GoalImageCapture(
            camera_index=args.camera,
            width=args.width,
            height=args.height
        )
        capturer.connect()
        
        try:
            filename = f"{args.skill}_goal_{args.episode:04d}.jpg"
            save_path = os.path.join(args.output, args.skill, filename)
            capturer.capture_with_preview(save_path)
        finally:
            capturer.disconnect()


if __name__ == "__main__":
    main()
