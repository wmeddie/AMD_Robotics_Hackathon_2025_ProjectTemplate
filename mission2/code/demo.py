#!/usr/bin/env python3
"""
Main Demo Loop for Zen Garden Robot

Combines the LLM planner with skill execution to create zen garden patterns.

Usage:
    # Run with a target pattern
    python demo.py --target ./goals/pattern1.jpg
    
    # Run in simulation mode (no robot)
    python demo.py --target ./goals/pattern1.jpg --simulate
    
    # Run with mock planner (no API calls)
    python demo.py --target ./goals/pattern1.jpg --mock
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Try importing OpenCV for camera
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Try importing PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Import our modules
from planner import ZenGardenPlanner, MockPlanner, create_planner, SkillCall
from skill_executor import SkillExecutor, create_executor
from capture_goal import (
    CAMERA_GOAL, CAMERA_OVERHEAD, CAMERA_ARM,
    CAMERA_WIDTH, CAMERA_HEIGHT
)

# Camera usage in demo:
# - CAMERA_OVERHEAD (4): Used by planner to compare current state vs goal
# - CAMERA_ARM (6): Used by skill policies for observations during execution


class Camera:
    """Simple camera interface for capturing observations."""
    
    def __init__(self, camera_index: int = CAMERA_OVERHEAD, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
    
    def connect(self):
        """Connect to the camera."""
        if CV2_AVAILABLE:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")
            print(f"Camera {self.camera_index} connected")
        else:
            print("Warning: OpenCV not available, using dummy camera")
    
    def disconnect(self):
        """Disconnect from the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def capture(self) -> np.ndarray:
        """Capture a frame from the camera."""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Return dummy image
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
    
    def save_capture(self, path: str) -> str:
        """Capture and save an image."""
        frame = self.capture()
        
        if PIL_AVAILABLE:
            Image.fromarray(frame).save(path)
        elif CV2_AVAILABLE:
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            # Save as numpy
            np.save(path.replace('.jpg', '.npy'), frame)
        
        return path


class ZenGardenDemo:
    """
    Main demo class for the Zen Garden Robot.
    
    Orchestrates the planner and executor to create patterns.
    """
    
    def __init__(
        self,
        checkpoints_dir: str = "./checkpoints",
        use_mock_planner: bool = False,
        simulate: bool = False,
        output_dir: str = "./demo_output",
        device: str = "cuda"
    ):
        """
        Initialize the demo.
        
        Args:
            checkpoints_dir: Path to trained policy checkpoints
            use_mock_planner: Use mock planner instead of Claude API
            simulate: Run in simulation mode (no real robot)
            output_dir: Directory for saving outputs
            device: Device for inference
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.simulate = simulate
        
        # Initialize camera
        self.camera = Camera()
        if not simulate:
            try:
                self.camera.connect()
            except Exception as e:
                print(f"Camera connection failed: {e}")
                print("Running in simulation mode")
                self.simulate = True
        
        # Initialize planner
        self.planner = create_planner(use_mock=use_mock_planner)
        
        # Initialize executor
        self.executor = create_executor(
            checkpoints_dir=checkpoints_dir,
            robot=None if simulate else self._get_robot(),
            camera=self.camera if not simulate else None,
            device=device
        )
        
        # Demo state
        self.step_count = 0
        self.history = []
    
    def _get_robot(self):
        """Get robot interface (placeholder for actual robot connection)."""
        # In real deployment, this would return the actual robot interface
        # For now, return None to use simulation
        return None
    
    def capture_current_state(self) -> str:
        """Capture the current garden state and save it."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"current_state_{self.step_count:03d}_{timestamp}.jpg"
        path = self.output_dir / filename
        
        if self.simulate:
            # Create dummy image
            dummy = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
            if PIL_AVAILABLE:
                Image.fromarray(dummy).save(path)
            else:
                np.save(str(path).replace('.jpg', '.npy'), dummy)
        else:
            self.camera.save_capture(str(path))
        
        return str(path)
    
    def execute_skill(self, skill: SkillCall, goal_image_path: str) -> bool:
        """
        Execute a single skill.
        
        Args:
            skill: The skill to execute
            goal_image_path: Path to the goal image
            
        Returns:
            True if successful
        """
        print(f"\n{'='*50}")
        print(f"Executing: {skill}")
        print(f"{'='*50}")
        
        try:
            if skill.name == "flatten":
                return self.executor.flatten(goal_image_path)
            elif skill.name == "draw_zigzag":
                return self.executor.draw_zigzag(goal_image_path)
            elif skill.name == "draw_circles":
                return self.executor.draw_circles(goal_image_path)
            elif skill.name == "stamp_triangle":
                x = skill.args.get("x", 0.5)
                y = skill.args.get("y", 0.5)
                return self.executor.stamp_triangle(goal_image_path, x, y)
            elif skill.name == "place_rock":
                x = skill.args.get("x", 0.5)
                y = skill.args.get("y", 0.5)
                return self.executor.place_rock(x, y)
            elif skill.name == "done":
                return True
            else:
                print(f"Unknown skill: {skill.name}")
                return False
        except Exception as e:
            print(f"Skill execution failed: {e}")
            return False
    
    def run(
        self,
        target_image_path: str,
        max_steps: int = 10,
        step_delay: float = 1.0
    ) -> bool:
        """
        Run the full demo loop.
        
        Args:
            target_image_path: Path to the target pattern image
            max_steps: Maximum number of skill executions
            step_delay: Delay between steps in seconds
            
        Returns:
            True if pattern was completed successfully
        """
        print(f"\n{'#'*60}")
        print(f"# ZEN GARDEN DEMO")
        print(f"# Target: {target_image_path}")
        print(f"# Max steps: {max_steps}")
        print(f"# Simulation: {self.simulate}")
        print(f"{'#'*60}\n")
        
        # Verify target image exists
        if not Path(target_image_path).exists():
            print(f"Error: Target image not found: {target_image_path}")
            return False
        
        # Reset state
        self.step_count = 0
        self.history = []
        self.planner.reset()
        
        # Main loop
        completed = False
        
        for step in range(max_steps):
            self.step_count = step + 1
            print(f"\n--- Step {self.step_count}/{max_steps} ---")
            
            # Capture current state
            current_image_path = self.capture_current_state()
            print(f"Captured: {current_image_path}")
            
            # Get next skill from planner
            try:
                skill = self.planner.get_next_skill(target_image_path, current_image_path)
                print(f"Planner says: {skill}")
            except Exception as e:
                print(f"Planner error: {e}")
                break
            
            # Check if done
            if skill.name == "done":
                print("\nPlanner indicates pattern is complete!")
                completed = True
                break
            
            # Execute skill
            success = self.execute_skill(skill, target_image_path)
            
            # Record history
            self.history.append({
                "step": self.step_count,
                "current_image": current_image_path,
                "skill": str(skill),
                "success": success
            })
            
            if not success:
                print(f"Skill execution failed at step {self.step_count}")
                # Continue anyway - the planner will adapt
            
            # Delay before next step
            time.sleep(step_delay)
        
        # Cleanup
        self.executor.done()
        
        # Final capture
        final_path = self.output_dir / "final_result.jpg"
        if not self.simulate:
            self.camera.save_capture(str(final_path))
            print(f"\nFinal result saved: {final_path}")
        
        # Summary
        print(f"\n{'#'*60}")
        print(f"# DEMO COMPLETE")
        print(f"# Steps executed: {self.step_count}")
        print(f"# Completed: {completed}")
        print(f"# Output: {self.output_dir}")
        print(f"{'#'*60}")
        
        return completed
    
    def cleanup(self):
        """Clean up resources."""
        if self.camera:
            self.camera.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Run the Zen Garden Robot demo"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to target pattern image"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints",
        help="Path to trained policy checkpoints"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./demo_output",
        help="Output directory for demo artifacts"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Maximum number of skill executions"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulation mode (no real robot)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock planner (no API calls)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = ZenGardenDemo(
        checkpoints_dir=args.checkpoints,
        use_mock_planner=args.mock,
        simulate=args.simulate,
        output_dir=args.output,
        device=args.device
    )
    
    try:
        success = demo.run(
            target_image_path=args.target,
            max_steps=args.max_steps
        )
        return 0 if success else 1
    finally:
        demo.cleanup()


if __name__ == "__main__":
    exit(main())
