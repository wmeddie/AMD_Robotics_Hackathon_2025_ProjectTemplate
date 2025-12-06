#!/usr/bin/env python3
"""
Skill Executor for Zen Garden Robot

Executes learned skill policies along with hardcoded tool pickup/putdown trajectories.

Usage:
    from skill_executor import SkillExecutor
    
    executor = SkillExecutor()
    executor.flatten(goal_image)
    executor.draw_zigzag(goal_image)
"""

import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, Callable

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Import inference module
from inference import SkillPolicy, MultiSkillPolicy


# Tool positions in robot workspace (normalized coordinates)
TOOL_POSITIONS = {
    "flat_rake": {"pickup": (0.1, 0.8, 0.05), "putdown": (0.1, 0.8, 0.0)},
    "zigzag_rake": {"pickup": (0.2, 0.8, 0.05), "putdown": (0.2, 0.8, 0.0)},
    "circle_rake": {"pickup": (0.3, 0.8, 0.05), "putdown": (0.3, 0.8, 0.0)},
    "stamp": {"pickup": (0.4, 0.8, 0.05), "putdown": (0.4, 0.8, 0.0)},
    "rock": {"pickup": (0.5, 0.8, 0.05), "putdown": None},  # Rock doesn't get put down
}

# Skill to tool mapping
SKILL_TOOLS = {
    "flatten": "flat_rake",
    "zigzag": "zigzag_rake",
    "circle": "circle_rake",
    "stamp": "stamp",
    "place_rock": "rock",
}

# Home position for the robot arm
HOME_POSITION = np.array([0.0, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0])  # 7 DOF


class HardcodedTrajectory:
    """
    Generates hardcoded trajectories for tool manipulation.
    
    These trajectories are not learned but pre-programmed for reliability.
    """
    
    def __init__(self, control_frequency: int = 30, max_velocity: float = 0.5):
        self.control_frequency = control_frequency
        self.max_velocity = max_velocity
    
    def interpolate(
        self,
        start: np.ndarray,
        end: np.ndarray,
        duration: float
    ) -> np.ndarray:
        """
        Linear interpolation between two poses.
        
        Args:
            start: Starting pose
            end: Ending pose
            duration: Duration in seconds
            
        Returns:
            Trajectory as array [T, pose_dim]
        """
        num_steps = int(duration * self.control_frequency)
        trajectory = np.zeros((num_steps, len(start)))
        
        for i in range(num_steps):
            t = i / (num_steps - 1) if num_steps > 1 else 1.0
            # Smooth interpolation using cosine
            t_smooth = (1 - np.cos(t * np.pi)) / 2
            trajectory[i] = start + t_smooth * (end - start)
        
        return trajectory
    
    def goto_position(
        self,
        current_pose: np.ndarray,
        target_xyz: Tuple[float, float, float],
        duration: float = 1.0
    ) -> np.ndarray:
        """
        Generate trajectory to move end-effector to target XYZ position.
        
        Args:
            current_pose: Current robot pose
            target_xyz: Target (x, y, z) position
            duration: Movement duration
            
        Returns:
            Trajectory
        """
        # Create target pose (simplified: just modify XYZ, keep orientation)
        target_pose = current_pose.copy()
        target_pose[0:3] = target_xyz
        
        return self.interpolate(current_pose, target_pose, duration)
    
    def pickup_tool(
        self,
        tool_name: str,
        current_pose: np.ndarray
    ) -> np.ndarray:
        """
        Generate trajectory to pick up a tool.
        
        Args:
            tool_name: Name of tool to pick up
            current_pose: Current robot pose
            
        Returns:
            Full pickup trajectory
        """
        if tool_name not in TOOL_POSITIONS:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool_pos = TOOL_POSITIONS[tool_name]["pickup"]
        
        trajectories = []
        
        # 1. Move above tool
        above_pos = (tool_pos[0], tool_pos[1], tool_pos[2] + 0.1)
        traj1 = self.goto_position(current_pose, above_pos, duration=1.0)
        trajectories.append(traj1)
        
        # 2. Lower to tool
        traj2 = self.goto_position(traj1[-1], tool_pos, duration=0.5)
        trajectories.append(traj2)
        
        # 3. Close gripper (add gripper close action)
        grip_traj = traj2[-1:].copy()
        grip_traj[:, -1] = 1.0  # Close gripper (assuming last DOF is gripper)
        trajectories.append(np.repeat(grip_traj, 15, axis=0))  # Hold for 0.5s
        
        # 4. Lift tool
        lift_pos = (tool_pos[0], tool_pos[1], tool_pos[2] + 0.1)
        current = trajectories[-1][-1]
        traj4 = self.goto_position(current, lift_pos, duration=0.5)
        trajectories.append(traj4)
        
        return np.concatenate(trajectories, axis=0)
    
    def putdown_tool(
        self,
        tool_name: str,
        current_pose: np.ndarray
    ) -> np.ndarray:
        """
        Generate trajectory to put down a tool.
        
        Args:
            tool_name: Name of tool to put down
            current_pose: Current robot pose
            
        Returns:
            Full putdown trajectory
        """
        if tool_name not in TOOL_POSITIONS:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        putdown_pos = TOOL_POSITIONS[tool_name]["putdown"]
        if putdown_pos is None:
            # Tool doesn't get put down (e.g., rock)
            return np.array([]).reshape(0, len(current_pose))
        
        trajectories = []
        
        # 1. Move above putdown position
        above_pos = (putdown_pos[0], putdown_pos[1], putdown_pos[2] + 0.1)
        traj1 = self.goto_position(current_pose, above_pos, duration=1.0)
        trajectories.append(traj1)
        
        # 2. Lower to position
        traj2 = self.goto_position(traj1[-1], putdown_pos, duration=0.5)
        trajectories.append(traj2)
        
        # 3. Open gripper
        release_traj = traj2[-1:].copy()
        release_traj[:, -1] = 0.0  # Open gripper
        trajectories.append(np.repeat(release_traj, 15, axis=0))  # Hold for 0.5s
        
        # 4. Lift away
        lift_pos = (putdown_pos[0], putdown_pos[1], putdown_pos[2] + 0.1)
        current = trajectories[-1][-1]
        traj4 = self.goto_position(current, lift_pos, duration=0.5)
        trajectories.append(traj4)
        
        return np.concatenate(trajectories, axis=0)
    
    def goto_home(self, current_pose: np.ndarray) -> np.ndarray:
        """Generate trajectory to return to home position."""
        return self.interpolate(current_pose, HOME_POSITION, duration=1.5)


class SkillExecutor:
    """
    Main executor for Zen Garden robot skills.
    
    Combines learned policies with hardcoded tool manipulation.
    """
    
    def __init__(
        self,
        checkpoints_dir: str = "./checkpoints",
        robot: Optional[object] = None,
        camera: Optional[object] = None,
        device: str = "cuda",
        control_frequency: int = 30,
        max_steps_per_skill: int = 300
    ):
        """
        Initialize the skill executor.
        
        Args:
            checkpoints_dir: Directory containing trained policy checkpoints
            robot: Robot interface (if None, will print actions)
            camera: Camera interface (if None, will use dummy images)
            device: Device for policy inference
            control_frequency: Robot control frequency in Hz
            max_steps_per_skill: Maximum steps before timing out
        """
        self.robot = robot
        self.camera = camera
        self.control_frequency = control_frequency
        self.max_steps_per_skill = max_steps_per_skill
        
        # Load policies
        self.policies = MultiSkillPolicy(
            checkpoints_dir=checkpoints_dir,
            device=device
        )
        
        # Hardcoded trajectory generator
        self.trajectory_gen = HardcodedTrajectory(
            control_frequency=control_frequency
        )
        
        # Current state tracking
        self.current_tool = None
        self.current_pose = HOME_POSITION.copy()
    
    def get_observation(self) -> np.ndarray:
        """Capture current observation from camera."""
        if self.camera is not None:
            return self.camera.capture()
        else:
            # Return dummy observation
            return np.random.rand(224, 224, 3).astype(np.float32)
    
    def get_robot_state(self) -> np.ndarray:
        """Get current robot proprioceptive state."""
        if self.robot is not None:
            return self.robot.get_state()
        else:
            # Return current pose padded to state_dim
            state = np.zeros(14)
            state[:len(self.current_pose)] = self.current_pose
            return state
    
    def execute_trajectory(self, trajectory: np.ndarray):
        """
        Execute a trajectory on the robot.
        
        Args:
            trajectory: Array of poses [T, pose_dim]
        """
        if len(trajectory) == 0:
            return
        
        for i, pose in enumerate(trajectory):
            if self.robot is not None:
                self.robot.send_action(pose)
            else:
                if i % 30 == 0:  # Print every second
                    print(f"  Step {i}: pose = {pose[:3]}...")
            
            self.current_pose = pose
            time.sleep(1.0 / self.control_frequency)
    
    def pickup_tool(self, tool_name: str):
        """Pick up a tool."""
        print(f"Picking up tool: {tool_name}")
        
        if self.current_tool is not None:
            print(f"  First putting down current tool: {self.current_tool}")
            self.putdown_tool()
        
        trajectory = self.trajectory_gen.pickup_tool(tool_name, self.current_pose)
        self.execute_trajectory(trajectory)
        self.current_tool = tool_name
        print(f"  Picked up {tool_name}")
    
    def putdown_tool(self):
        """Put down the current tool."""
        if self.current_tool is None:
            return
        
        print(f"Putting down tool: {self.current_tool}")
        trajectory = self.trajectory_gen.putdown_tool(self.current_tool, self.current_pose)
        self.execute_trajectory(trajectory)
        self.current_tool = None
        print("  Tool put down")
    
    def goto_home(self):
        """Return robot to home position."""
        print("Returning to home position")
        trajectory = self.trajectory_gen.goto_home(self.current_pose)
        self.execute_trajectory(trajectory)
        print("  At home position")
    
    def execute_skill(
        self,
        skill: str,
        goal_image: Union[np.ndarray, str],
        **kwargs
    ) -> bool:
        """
        Execute a learned skill with the given goal.
        
        Args:
            skill: Skill name
            goal_image: Target/goal image
            **kwargs: Additional skill-specific arguments
            
        Returns:
            True if skill completed successfully
        """
        if skill not in SKILL_TOOLS:
            raise ValueError(f"Unknown skill: {skill}")
        
        tool = SKILL_TOOLS[skill]
        
        # Pick up required tool
        if self.current_tool != tool:
            self.pickup_tool(tool)
        
        # Execute learned policy
        print(f"Executing skill: {skill}")
        
        for step in range(self.max_steps_per_skill):
            # Get current observation and state
            observation = self.get_observation()
            state = self.get_robot_state()
            
            # Get action from policy
            try:
                action = self.policies.predict(skill, observation, goal_image, state)
            except Exception as e:
                print(f"  Policy error: {e}")
                return False
            
            # Execute action
            if self.robot is not None:
                self.robot.send_action(action)
            else:
                if step % 30 == 0:
                    print(f"  Step {step}: action = {action[:3]}...")
            
            self.current_pose[:len(action)] = action
            time.sleep(1.0 / self.control_frequency)
            
            # Check if goal is reached (simplified: just run for fixed steps)
            # In practice, you'd have a goal detector here
        
        print(f"  Skill {skill} completed")
        return True
    
    # Convenience methods for each skill
    
    def flatten(self, goal_image: Union[np.ndarray, str]) -> bool:
        """Flatten the sand surface."""
        return self.execute_skill("flatten", goal_image)
    
    def draw_zigzag(self, goal_image: Union[np.ndarray, str]) -> bool:
        """Draw zigzag pattern in the sand."""
        return self.execute_skill("zigzag", goal_image)
    
    def draw_circles(self, goal_image: Union[np.ndarray, str]) -> bool:
        """Draw concentric circles in the sand."""
        return self.execute_skill("circle", goal_image)
    
    def stamp_triangle(
        self,
        goal_image: Union[np.ndarray, str],
        x: float = 0.5,
        y: float = 0.5
    ) -> bool:
        """
        Stamp a triangle at the given position.
        
        Args:
            goal_image: Target pattern image
            x: Normalized x position (0-1)
            y: Normalized y position (0-1)
        """
        # Move to position first
        stamp_pos = (x, y, 0.05)
        trajectory = self.trajectory_gen.goto_position(
            self.current_pose, stamp_pos, duration=1.0
        )
        self.execute_trajectory(trajectory)
        
        return self.execute_skill("stamp", goal_image)
    
    def place_rock(self, x: float = 0.5, y: float = 0.5) -> bool:
        """
        Place the rock at the given position.
        
        Args:
            x: Normalized x position (0-1)
            y: Normalized y position (0-1)
        """
        # Pick up rock
        self.pickup_tool("rock")
        
        # Move to target position
        target_pos = (x, y, 0.05)
        trajectory = self.trajectory_gen.goto_position(
            self.current_pose, target_pos, duration=1.0
        )
        self.execute_trajectory(trajectory)
        
        # Lower and release
        lower_pos = (x, y, 0.02)
        traj = self.trajectory_gen.goto_position(self.current_pose, lower_pos, 0.5)
        self.execute_trajectory(traj)
        
        # Open gripper
        release_pose = self.current_pose.copy()
        release_pose[-1] = 0.0  # Open gripper
        self.execute_trajectory(np.array([release_pose]))
        
        self.current_tool = None
        print(f"  Rock placed at ({x}, {y})")
        
        return True
    
    def done(self):
        """Finish execution and cleanup."""
        print("Task complete - cleaning up")
        self.putdown_tool()
        self.goto_home()
        print("Done!")


def create_executor(
    checkpoints_dir: str = "./checkpoints",
    robot=None,
    camera=None,
    device: str = "cuda"
) -> SkillExecutor:
    """
    Factory function to create a skill executor.
    
    Args:
        checkpoints_dir: Path to trained checkpoints
        robot: Robot interface
        camera: Camera interface
        device: Device for inference
        
    Returns:
        Configured SkillExecutor
    """
    return SkillExecutor(
        checkpoints_dir=checkpoints_dir,
        robot=robot,
        camera=camera,
        device=device
    )


if __name__ == "__main__":
    # Test skill executor
    print("Testing SkillExecutor...")
    
    executor = SkillExecutor(
        checkpoints_dir="./checkpoints",
        robot=None,  # No robot - will print actions
        camera=None  # No camera - will use dummy images
    )
    
    # Test with dummy goal
    dummy_goal = np.random.rand(224, 224, 3).astype(np.float32)
    
    print("\nTesting flatten skill...")
    executor.flatten(dummy_goal)
    
    print("\nTesting cleanup...")
    executor.done()
    
    print("\nTest complete!")
