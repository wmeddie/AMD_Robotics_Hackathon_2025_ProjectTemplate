# Goal-Conditioned SmolVLA for Zen Garden Robot - Hackathon Project

## Context

We're at a 48-hour AMD-sponsored hackathon with access to an MI300X cluster. We have 3 SO-101 robot arms with teleoperation setups ready for data collection. The goal is to build a zen garden robot that can recreate patterns shown in a target image.

## High-Level Architecture

We're building a hierarchical system:

1. **5 specialized skill policies** (each a goal-conditioned SmolVLA variant):
   - `flatten_policy` - uses flat rake to smooth sand
   - `zigzag_policy` - uses zigzag rake for parallel lines
   - `circle_policy` - uses two-point rake for concentric circles
   - `stamp_policy` - uses triangle stamp at specified positions
   - `place_rock_policy` - picks and places the rock

2. **LLM planner** (Claude API) - looks at target image and current state, sequences skill calls

3. **Hardcoded trajectories** for tool pickup/putdown (not learned)

## Your Tasks

### Task 1: Environment Setup

```bash
# Clone required repos
git clone https://github.com/huggingface/lerobot
git clone https://github.com/huggingface/smolvla

# Set up ROCm-compatible PyTorch for MI300X
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Install dependencies
pip install -e ./lerobot
pip install -e ./smolvla
```

Verify ROCm works:
```python
import torch
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### Task 2: Goal-Conditioned SmolVLA Architecture Modification

The key modification: inject a **goal image embedding** into the action expert as additional conditioning.

**Current SmolVLA flow:**
```
observation_images -> SmolVLM-2 (SigLIP + SmolLM2) -> vlm_features -> Action Expert -> actions
```

**Modified flow:**
```
goal_image ---------> SigLIP (frozen) -> goal_proj (trainable) ----┐
                                                                    ├-> concat -> Action Expert -> actions
observation_images -> SmolVLM-2 (frozen) -> vlm_features ----------┘
```

Create a new file `goal_conditioned_smolvla.py` that:

1. Loads the pretrained SmolVLA model
2. Extracts the SigLIP encoder for goal image encoding
3. Adds a `goal_projection` layer: `nn.Linear(siglip_embed_dim, action_expert_input_dim)`
4. Modifies the action expert's forward pass to concatenate goal features with VLM features
5. Freezes: SigLIP encoder, entire VLM backbone
6. Trainable: goal_projection layer, action expert

```python
# Pseudocode structure to implement:

class GoalConditionedSmolVLA(nn.Module):
    def __init__(self, pretrained_smolvla_path):
        super().__init__()
        # Load pretrained SmolVLA
        self.smolvla = load_pretrained(pretrained_smolvla_path)
        
        # Extract SigLIP for goal encoding (shared weights, frozen)
        self.goal_encoder = self.smolvla.vlm.vision_encoder
        self.goal_encoder.requires_grad_(False)
        
        # Freeze VLM backbone
        self.smolvla.vlm.requires_grad_(False)
        
        # New trainable projection for goal embedding
        siglip_dim = 768  # or whatever SigLIP outputs, verify this
        action_expert_dim = self.smolvla.action_expert.input_dim  # verify this
        self.goal_projection = nn.Linear(siglip_dim, action_expert_dim)
        
        # Action expert stays trainable
        self.smolvla.action_expert.requires_grad_(True)
    
    def forward(self, observation_images, goal_image, state, instruction=None):
        # Encode goal image
        with torch.no_grad():
            goal_features = self.goal_encoder(goal_image)
        goal_projected = self.goal_projection(goal_features.mean(dim=1))  # pool spatial dims
        
        # Get VLM features from observations
        with torch.no_grad():
            vlm_features = self.smolvla.vlm(observation_images, instruction, state)
        
        # Concatenate goal conditioning
        combined_features = torch.cat([vlm_features, goal_projected], dim=-1)
        
        # Action expert predicts actions
        actions = self.smolvla.action_expert(combined_features)
        return actions
```

**Important**: Look at the actual SmolVLA code to get the correct:
- Layer names and access patterns
- Embedding dimensions
- How VLM features are extracted before the action expert
- Action expert input format

The action expert uses interleaved cross-attention and self-attention. You may need to modify its first layer to accept the larger concatenated input, or add a down-projection.

### Task 3: Dataset Structure for Goal-Conditioned Training

LeRobot datasets are stored in a specific format. We need to augment them with goal images.

For each trajectory collected:
- The **goal image** is the final frame of that trajectory (what the completed pattern looks like)
- Or: we photograph the target pattern separately before each demo

Create a script `prepare_goal_dataset.py` that:

1. Takes a LeRobot dataset directory as input
2. For each episode:
   - Extracts the final observation image as `goal_image`
   - OR loads a separately saved target image if we collected those
3. Saves augmented dataset with goal images accessible during training

```python
# Dataset structure we need:
# {
#     "observation.images.top": [T, C, H, W],  # camera observations over time
#     "observation.state": [T, state_dim],      # proprioceptive state
#     "action": [T, action_dim],                # actions
#     "goal_image": [C, H, W],                  # single goal image for this trajectory
#     "task": str,                              # e.g., "flatten", "zigzag", etc.
# }
```

### Task 4: Training Script

Create `train_goal_conditioned.py`:

```python
# Key requirements:
# - Multi-GPU training on MI300X cluster (use accelerate or native DDP)
# - Load goal-conditioned model
# - Load augmented dataset with goal images
# - MSE loss on action predictions (or whatever SmolVLA uses - check if flow matching)
# - Checkpointing every N steps
# - Wandb logging (optional but nice for the demo)

# Training hyperparameters to start with:
# - lr: 1e-4 for action expert, 1e-4 for goal_projection
# - batch_size: 32 per GPU (adjust based on memory)
# - epochs: 100 (small dataset, will converge fast)
# - warmup_steps: 100
```

Use HuggingFace Accelerate for multi-GPU:
```bash
accelerate config  # set up for multi-GPU
accelerate launch train_goal_conditioned.py --skill flatten --data_dir ./data/flatten
```

Create a training script that can be run separately for each skill:
```bash
# Train all 5 policies (can run in parallel on cluster)
accelerate launch train_goal_conditioned.py --skill flatten
accelerate launch train_goal_conditioned.py --skill zigzag
accelerate launch train_goal_conditioned.py --skill circle
accelerate launch train_goal_conditioned.py --skill stamp
accelerate launch train_goal_conditioned.py --skill place_rock
```

### Task 5: Inference and Skill Execution

Create `inference.py` with a `SkillPolicy` class:

```python
class SkillPolicy:
    def __init__(self, checkpoint_path, device="cuda"):
        self.model = GoalConditionedSmolVLA.load(checkpoint_path)
        self.model.eval()
        self.model.to(device)
    
    def predict(self, observation_image, goal_image, state):
        with torch.no_grad():
            action = self.model(observation_image, goal_image, state)
        return action.cpu().numpy()
```

Create `skill_executor.py` that:
1. Loads all 5 trained policies
2. Has hardcoded tool pickup/putdown trajectories
3. Exposes methods like:
   - `executor.flatten(goal_image)`
   - `executor.draw_zigzag(goal_image)`
   - `executor.draw_circles(goal_image)`
   - `executor.stamp_triangle(goal_image, x, y)`
   - `executor.place_rock(x, y)`

### Task 6: LLM Planner Integration

Create `planner.py`:

```python
import anthropic
import base64

client = anthropic.Anthropic()

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

def get_next_skill(target_image_path, current_image_path):
    target_b64 = encode_image(target_image_path)
    current_b64 = encode_image(current_image_path)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Target pattern:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": target_b64}},
                {"type": "text", "text": "Current state:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": current_b64}},
                {"type": "text", "text": "What is the next skill to execute?"}
            ]
        }]
    )
    return parse_skill_call(response.content[0].text)
```

### Task 7: Main Demo Loop

Create `demo.py`:

```python
def run_demo(target_image_path):
    executor = SkillExecutor()
    camera = Camera()
    
    while True:
        current_image = camera.capture()
        current_image.save("/tmp/current.jpg")
        
        skill_call = get_next_skill(target_image_path, "/tmp/current.jpg")
        
        if skill_call.name == "done":
            print("Pattern complete!")
            break
        
        print(f"Executing: {skill_call}")
        executor.execute(skill_call)
    
    # Final glamour shot
    final_image = camera.capture()
    final_image.save("final_result.jpg")
```

### Task 8: Recording Script Modifications (if needed)

Check if `lerobot-record` needs modifications to save goal images. If we're using final frame as goal, no changes needed—we extract it in post-processing.

If we want to photograph target patterns separately:
```bash
# Before each recording session, capture the target:
python capture_goal.py --output ./goals/zigzag_01.jpg

# Then record the trajectory:
lerobot-record --robot so101 --task zigzag --output ./data/zigzag/
```

Create `capture_goal.py` for this workflow if needed.

## Directory Structure

```
zen-garden-smolvla/
├── goal_conditioned_smolvla.py   # Model architecture
├── prepare_goal_dataset.py        # Dataset preprocessing
├── train_goal_conditioned.py      # Training script
├── inference.py                   # Single policy inference
├── skill_executor.py              # Multi-skill execution + hardcoded trajectories
├── planner.py                     # Claude API integration
├── demo.py                        # Main demo loop
├── capture_goal.py                # Goal image capture utility
├── configs/
│   ├── flatten.yaml
│   ├── zigzag.yaml
│   ├── circle.yaml
│   ├── stamp.yaml
│   └── place_rock.yaml
├── data/                          # LeRobot datasets go here
│   ├── flatten/
│   ├── zigzag/
│   ├── circle/
│   ├── stamp/
│   └── place_rock/
├── checkpoints/                   # Trained models
└── goals/                         # Target pattern images
```

## Priority Order

1. **First**: Get environment working, verify MI300X + ROCm + PyTorch
2. **Second**: Implement `GoalConditionedSmolVLA` class (this is the core novelty)
3. **Third**: Training script with multi-GPU support
4. **Fourth**: Dataset preparation script
5. **Fifth**: Inference and skill executor
6. **Sixth**: Planner integration
7. **Last**: Demo polish

## Notes

- SmolVLA uses Flow Matching for action prediction, not simple MSE. Check their loss function.
- The action expert outputs "action chunks" (multiple future timesteps). Handle this in inference.
- SigLIP outputs 64 tokens per image in SmolVLA's config. The goal projection needs to handle this.
- We have ~15 demos per skill, so regularization matters. Consider dropout, early stopping.
- MI300X has 192GB HBM3—memory is not a constraint, batch size up if training is slow.

## Questions to Resolve by Reading SmolVLA Code

1. Exact layer names for SigLIP encoder access
2. Action expert input dimension and format
3. How flow matching loss is computed
4. Whether instruction text is required or optional
5. State dimension for SO-101

Start by exploring the SmolVLA repo structure and reading their model implementation before writing code.
