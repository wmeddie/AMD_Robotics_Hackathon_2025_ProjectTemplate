#!/usr/bin/env python3
"""
LLM Planner for Zen Garden Robot

Uses Claude API to sequence skill calls based on target and current images.

Usage:
    from planner import ZenGardenPlanner
    
    planner = ZenGardenPlanner()
    skill_call = planner.get_next_skill(target_image_path, current_image_path)
"""

import base64
import re
from pathlib import Path
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass

# Try importing anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not installed. Run: pip install anthropic")


@dataclass
class SkillCall:
    """Represents a parsed skill call from the planner."""
    name: str
    args: dict
    raw: str
    
    def __str__(self):
        if self.args:
            args_str = ", ".join(f"{k}={v}" for k, v in self.args.items())
            return f"{self.name}({args_str})"
        return f"{self.name}()"


SYSTEM_PROMPT = """You are controlling a zen garden robot. You can see the target pattern to create and the current state of the garden.

Available skills:
- flatten() - smooths the entire sand surface with flat rake
- draw_zigzag() - creates parallel zigzag lines with zigzag rake
- draw_circles() - creates concentric circles with two-point rake  
- stamp_triangle(x, y) - stamps a triangle at normalized coordinates (0-1, 0-1)
- place_rock(x, y) - places the rock at normalized coordinates (0-1, 0-1)
- done() - call when the pattern is complete

Rules:
1. Always flatten() first if the sand isn't smooth
2. Place rock last if the pattern includes one
3. Output exactly ONE skill call per turn
4. Output just the function call, nothing else

Example output: flatten()
Example output: stamp_triangle(0.3, 0.5)
Example output: done()
"""


def encode_image(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_media_type(image_path: str) -> str:
    """Get the media type for an image file."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return media_types.get(suffix, "image/jpeg")


def parse_skill_call(response_text: str) -> SkillCall:
    """
    Parse a skill call from the LLM response.
    
    Args:
        response_text: Raw text response from the LLM
        
    Returns:
        Parsed SkillCall object
    """
    text = response_text.strip()
    
    # Try to match function call pattern: name(args)
    match = re.match(r'(\w+)\s*\((.*?)\)', text)
    
    if not match:
        # Try without parentheses
        if text in ["flatten", "draw_zigzag", "draw_circles", "done"]:
            return SkillCall(name=text, args={}, raw=text)
        raise ValueError(f"Could not parse skill call: {text}")
    
    name = match.group(1)
    args_str = match.group(2).strip()
    
    # Parse arguments
    args = {}
    if args_str:
        # Handle positional args (x, y)
        parts = [p.strip() for p in args_str.split(",")]
        if name in ["stamp_triangle", "place_rock"]:
            if len(parts) >= 2:
                try:
                    args["x"] = float(parts[0])
                    args["y"] = float(parts[1])
                except ValueError:
                    pass
        # Handle keyword args (key=value)
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                try:
                    args[key.strip()] = float(value.strip())
                except ValueError:
                    args[key.strip()] = value.strip()
    
    return SkillCall(name=name, args=args, raw=text)


class ZenGardenPlanner:
    """
    LLM-based planner for sequencing zen garden robot skills.
    
    Uses Claude to analyze target and current images and determine
    the next skill to execute.
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 100,
        system_prompt: str = SYSTEM_PROMPT
    ):
        """
        Initialize the planner.
        
        Args:
            model: Claude model to use
            max_tokens: Maximum tokens in response
            system_prompt: System prompt for the model
        """
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("anthropic package not installed")
        
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        
        # Track conversation history for context
        self.history = []
    
    def reset(self):
        """Reset conversation history."""
        self.history = []
    
    def get_next_skill(
        self,
        target_image_path: str,
        current_image_path: str,
        additional_context: Optional[str] = None
    ) -> SkillCall:
        """
        Get the next skill to execute.
        
        Args:
            target_image_path: Path to the target pattern image
            current_image_path: Path to the current garden state image
            additional_context: Optional additional context for the model
            
        Returns:
            Parsed SkillCall to execute
        """
        # Encode images
        target_b64 = encode_image(target_image_path)
        current_b64 = encode_image(current_image_path)
        target_media = get_media_type(target_image_path)
        current_media = get_media_type(current_image_path)
        
        # Build message content
        content = [
            {"type": "text", "text": "Target pattern:"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": target_media,
                    "data": target_b64
                }
            },
            {"type": "text", "text": "Current state:"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": current_media,
                    "data": current_b64
                }
            },
            {"type": "text", "text": "What is the next skill to execute?"}
        ]
        
        if additional_context:
            content.insert(0, {"type": "text", "text": additional_context})
        
        # Call Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": content}]
        )
        
        # Parse response
        response_text = response.content[0].text
        skill_call = parse_skill_call(response_text)
        
        # Track history
        self.history.append({
            "target": target_image_path,
            "current": current_image_path,
            "response": response_text,
            "skill": skill_call
        })
        
        return skill_call
    
    def get_full_plan(
        self,
        target_image_path: str,
        current_image_path: str,
        max_steps: int = 10
    ) -> list:
        """
        Get a full plan of skills to execute.
        
        Note: This is a simplified version that doesn't actually
        execute skills. For real use, you'd need to execute each
        skill and capture new images.
        
        Args:
            target_image_path: Path to target image
            current_image_path: Path to current state image
            max_steps: Maximum number of planning steps
            
        Returns:
            List of SkillCall objects
        """
        plan = []
        self.reset()
        
        for _ in range(max_steps):
            skill = self.get_next_skill(target_image_path, current_image_path)
            plan.append(skill)
            
            if skill.name == "done":
                break
            
            # In real use, you'd execute the skill here and capture
            # a new current_image_path
        
        return plan


class MockPlanner:
    """
    Mock planner for testing without API access.
    
    Returns a predefined sequence of skills.
    """
    
    def __init__(self):
        self.step = 0
        self.default_sequence = [
            SkillCall("flatten", {}, "flatten()"),
            SkillCall("draw_zigzag", {}, "draw_zigzag()"),
            SkillCall("stamp_triangle", {"x": 0.5, "y": 0.5}, "stamp_triangle(0.5, 0.5)"),
            SkillCall("done", {}, "done()")
        ]
    
    def reset(self):
        self.step = 0
    
    def get_next_skill(
        self,
        target_image_path: str,
        current_image_path: str,
        additional_context: Optional[str] = None
    ) -> SkillCall:
        if self.step >= len(self.default_sequence):
            return SkillCall("done", {}, "done()")
        
        skill = self.default_sequence[self.step]
        self.step += 1
        return skill


def create_planner(use_mock: bool = False) -> "ZenGardenPlanner":
    """
    Factory function to create a planner.
    
    Args:
        use_mock: If True, return a mock planner for testing
        
    Returns:
        Planner instance
    """
    if use_mock or not ANTHROPIC_AVAILABLE:
        print("Using mock planner (no API calls)")
        return MockPlanner()
    return ZenGardenPlanner()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test zen garden planner")
    parser.add_argument("--target", type=str, help="Path to target image")
    parser.add_argument("--current", type=str, help="Path to current state image")
    parser.add_argument("--mock", action="store_true", help="Use mock planner")
    args = parser.parse_args()
    
    planner = create_planner(use_mock=args.mock)
    
    if args.target and args.current:
        skill = planner.get_next_skill(args.target, args.current)
        print(f"Next skill: {skill}")
    else:
        print("Testing mock planner...")
        planner = MockPlanner()
        for i in range(5):
            skill = planner.get_next_skill("target.jpg", "current.jpg")
            print(f"Step {i+1}: {skill}")
            if skill.name == "done":
                break
