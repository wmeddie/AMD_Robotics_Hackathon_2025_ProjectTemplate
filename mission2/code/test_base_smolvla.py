#!/usr/bin/env python3
"""
Test base SmolVLA model (no fine-tuning) to see how it handles tasks.
This helps understand what the pre-trained model can do out of the box.
"""

import argparse
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import PolicyFeature, FeatureType

def main():
    parser = argparse.ArgumentParser(description="Test base SmolVLA model")
    parser.add_argument("--task", type=str, default="pick up the rock", help="Task description")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load base SmolVLA config and model
    print("Loading base SmolVLA model...")
    config = SmolVLAConfig()
    
    # Set up minimal config for our robot using proper PolicyFeature types
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=[6]),
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=[3, 480, 640]),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=[3, 480, 640]),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=[6]),
    }
    
    policy = SmolVLAPolicy(config)
    policy.to(device)
    policy.eval()
    print("Model loaded!")
    
    # Create dummy observation
    print(f"\nTesting with task: '{args.task}'")
    
    # The base model needs tokenized input
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    
    # Tokenize task
    tokens = processor.tokenizer(
        args.task,
        return_tensors="pt",
        padding="max_length",
        max_length=48,
        truncation=True
    )
    
    dummy_obs = {
        "observation.images.front": torch.rand(1, 3, 480, 640).to(device),
        "observation.images.top": torch.rand(1, 3, 480, 640).to(device),
        "observation.state": torch.zeros(1, 6).to(device),
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].to(device),
    }
    
    print("\nRunning inference...")
    policy.reset()
    
    with torch.no_grad():
        action = policy.select_action(dummy_obs)
    
    print(f"\nOutput action shape: {action.shape}")
    print(f"Output action: {action.cpu().numpy()}")
    
    # Run a few more steps to see if output changes
    print("\nRunning 5 more steps...")
    for i in range(5):
        with torch.no_grad():
            action = policy.select_action(dummy_obs)
        print(f"  Step {i+1}: {action.cpu().numpy()}")

if __name__ == "__main__":
    main()
