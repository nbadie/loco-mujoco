# Parallel Multi-Seed Evaluation - Implementation Summary

## Overview
The evaluation script has been updated to support **parallel evaluation of multiple checkpoints across multiple seeds** with organized output structure.

## Key Changes

### 1. **Command-Line Arguments**
- Added `--num_seeds` parameter (default: 10)
```bash
python updated_eval_test_class_stacked_seeds_t.py \
  --folder_path /path/to/checkpoints \
  --config_file config.pkl \
  --num_seeds 10
```

### 2. **Train State Stacking**
- **Before**: Stacked train_states for N checkpoints → shape `(N,)`
- **After**: Stacked train_states for N checkpoints × M seeds → shape `(N*M,)`
- Each checkpoint is replicated M times (once per seed)
- Random seeds are generated independently for each checkpoint-seed combination

```python
train_states = []
for ckpt_file in checkpoint_files:
    for seed_idx in range(num_seeds):
        train_states.append(loaded_checkpoints[ckpt_file]['agent_state'].train_state)
        checkpoint_files_expanded.append((ckpt_file, seed_idx))
```

### 3. **Parallel Execution**
- **Rollout dimension**: Now `(num_checkpoints * num_seeds,)` instead of `(num_checkpoints,)`
- **RNG handling**: Independent RNG keys for each checkpoint-seed combination
  ```python
  rngs_batches = jax.random.split(_rng, num_batches)
  action, train_state = sample_actions_jit(train_state, obs, rngs_batches)
  ```
- **JAX vmap**: Still operates on single batch dimension (all checkpoint-seed pairs)
- **True parallelism**: All N×M combinations run simultaneously (same ~30s total time)

### 4. **Output Organization**
#### Folder Structure
```
results_parent_folder/
├── {timestamp}_{ckpt_name_1}/
│   ├── seed_0_evaluation_results_3000steps.pkl
│   ├── seed_1_evaluation_results_3000steps.pkl
│   ├── ...
│   └── seed_9_evaluation_results_3000steps.pkl
├── {timestamp}_{ckpt_name_2}/
│   ├── seed_0_evaluation_results_3000steps.pkl
│   ├── seed_1_evaluation_results_3000steps.pkl
│   ├── ...
│   └── seed_9_evaluation_results_3000steps.pkl
└── ...
```

#### Saved Data Enhancement
Each `.pkl` file now includes:
- `"checkpoint"`: Name of the checkpoint
- `"seed"`: Seed index (0-9)
- All existing metrics (rewards, forces, activations, etc.)

### 5. **Results Summary**
The final output now shows:
```
Results by Checkpoint:
  ckpt_1                                   | Mean: 1234.5678 ± 12.3456
    Seed 0: 1245.1234
    Seed 1: 1232.5678
    ...
    Seed 9: 1240.1234
  ckpt_2                                   | Mean: 1245.6789 ± 11.2345
    Seed 0: 1256.1234
    ...

Overall average reward: 1240.1234
```

## Performance Characteristics

| Aspect | Before | After |
|--------|--------|-------|
| **Checkpoints** | N | N |
| **Seeds per checkpoint** | 1 | M |
| **Total evaluation units** | N | N × M |
| **Execution time** | ~30s × N | ~30s (all parallel) |
| **Parallelization** | Checkpoints only | Checkpoints × Seeds |
| **Output structure** | Flat files | Organized folders |

### Example: 5 Checkpoints × 10 Seeds
- **Before**: 5 × 30s = 150 seconds
- **After**: 1 × 30s = 30 seconds (50 evaluation units in parallel)

## Important Implementation Details

### 1. **Extract Function**
Updated to handle batch indexing across checkpoint-seed combinations:
```python
def extract_batch_data(data, batch_idx):
    """Extract slice batch_idx from all arrays in nested structures."""
```

### 2. **Muscle Activation Computation**
All muscle activations computed once for all batches, then sliced per batch:
- `sum_step_activations`: Sum across time steps
- `step_norm_activations`: Normalized by number of steps
- `step_muscles_norm_activations`: Normalized by both steps and muscle count

### 3. **Environment State Replication**
Single environment reset replicated for all checkpoint-seed combinations:
```python
def replicate_for_batches(x):
    """Replicate a leaf value for all batches."""
```

## Usage Example

```bash
# Run with 10 seeds per checkpoint
python updated_eval_test_class_stacked_seeds_t.py \
  --folder_path /path/to/checkpoints \
  --config_file /path/to/config.pkl \
  --num_seeds 10

# Results will be organized as:
# /path/to/results/20250102_120530_ckpt_1/seed_0.pkl
# /path/to/results/20250102_120530_ckpt_1/seed_1.pkl
# /path/to/results/20250102_120530_ckpt_2/seed_0.pkl
# ...
```

## Benefits

1. **True Parallelism**: All checkpoint-seed combinations evaluated simultaneously
2. **Organized Results**: Each checkpoint has its own folder with seed-specific files
3. **Statistical Robustness**: Multiple seeds per checkpoint for uncertainty quantification
4. **Backward Compatible**: Same metrics and data structures, just extended dimensions
5. **No Training Required**: Evaluation-only feature on existing checkpoints

## Data Access Pattern

When analyzing results:
```python
import pickle

# Load a specific checkpoint-seed result
with open('20250102_120530_ckpt_1/seed_5_evaluation_results_3000steps.pkl', 'rb') as f:
    data = pickle.load(f)
    
# Access metrics
checkpoint = data['checkpoint']
seed = data['seed']
total_reward = data['total_reward']
avg_activation = data['step_num_norm_activations']['back_muscles_left']
```

## Troubleshooting

### Memory Issues
If you run out of memory with all checkpoint-seed combinations:
- Reduce `num_seeds`
- Process checkpoints in smaller batches

### RNG Behavior
Each checkpoint-seed combination gets independent randomness:
- Actions sampled with different seeds
- Environment behavior varies per seed
- Results naturally show stochasticity

### Output Not Generated
Check that:
1. Output folders are created successfully
2. Disk space available
3. File permissions in parent folder
