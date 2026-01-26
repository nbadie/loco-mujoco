# Quick Reference: Multi-Seed Parallel Evaluation

## What Changed

Your evaluation script now supports **parallel evaluation of multiple seeds per checkpoint**. Instead of evaluating one checkpoint at a time, it evaluates all checkpoint-seed combinations simultaneously.

## Quick Start

### Run with 10 seeds per checkpoint (default):
```bash
python updated_eval_test_class_stacked_seeds_t.py \
  --folder_path /path/to/checkpoints \
  --config_file /path/to/config.pkl
```

### Run with custom number of seeds:
```bash
python updated_eval_test_class_stacked_seeds_t.py \
  --folder_path /path/to/checkpoints \
  --config_file /path/to/config.pkl \
  --num_seeds 5
```

## Output Structure

Each checkpoint gets its own folder with seed-specific results:

```
parent_folder/
├── 20250102_120530_ckpt_1/
│   ├── seed_0_evaluation_results_3000steps.pkl
│   ├── seed_1_evaluation_results_3000steps.pkl
│   ├── ...
│   └── seed_9_evaluation_results_3000steps.pkl
├── 20250102_120530_ckpt_2/
│   ├── seed_0_evaluation_results_3000steps.pkl
│   ├── seed_1_evaluation_results_3000steps.pkl
│   ├── ...
│   └── seed_9_evaluation_results_3000steps.pkl
└── ...
```

## Key Features

| Feature | Details |
|---------|---------|
| **Parallelization** | All checkpoint×seed combinations run simultaneously |
| **Execution Time** | ~30s for N checkpoints × M seeds (same as single checkpoint) |
| **Output** | Each checkpoint has dedicated folder with per-seed files |
| **Data** | All original metrics preserved, now with seed tracking |
| **Independence** | Each seed gets independent RNG for action sampling |

## Sample Results Output

```
PARALLEL CHECKPOINT-SEED EVALUATION COMPLETE
=======================================================================
Total time: 32.45s
Total Total Time: 35.67s
Processed 3 checkpoints × 10 seeds = 30 total batches
All combinations run in parallel

Results by Checkpoint:
  ckpt_1                                   | Mean: 1234.5678 ± 12.3456
    Seed 0: 1245.1234
    Seed 1: 1232.5678
    ...
    Seed 9: 1240.1234
  ckpt_2                                   | Mean: 1245.6789 ± 11.2345
    ...
  ckpt_3                                   | Mean: 1250.1234 ± 15.6789
    ...

  Overall average reward: 1240.1234

All results organized in checkpoint-specific folders under: /path/to/results
```

## Accessing Results

```python
import pickle

# Load a specific result
with open('20250102_120530_ckpt_1/seed_5_evaluation_results_3000steps.pkl', 'rb') as f:
    data = pickle.load(f)

# Access data
print(f"Checkpoint: {data['checkpoint']}")
print(f"Seed: {data['seed']}")
print(f"Total Reward: {data['total_reward']}")
print(f"Steps: {data['total_steps']}")

# Get muscle activation data
left_back_activation = data['step_num_norm_activations']['back_muscles_left']
```

## Understanding the Parallelization

### Before
- Sequence: ckpt_1 → ckpt_2 → ckpt_3
- Time: ~30s per checkpoint = 90s total
- Seeds: 1 per checkpoint

### After  
- Parallel: All checkpoint-seed pairs evaluated simultaneously
- Time: ~30s for all 30 combinations (3 ckpts × 10 seeds)
- Seeds: 10 per checkpoint (configurable)
- **Speed-up: 3x for this example**

## Data Structure Changes

Each saved `.pkl` file now includes:
```python
{
    'checkpoint': 'ckpt_1.pkl',           # Which checkpoint
    'seed': 5,                             # Which seed (0-9)
    'total_reward': 1234.567,             # Reward for this combo
    'total_steps': 3000,
    'avg_reward_per_step': 0.411,
    
    # All original metrics:
    'all_grf_l': [...],
    'all_grf_r': [...],
    'all_sensor_force': {...},
    'all_actions': [...],
    'all_body_poses': {...},
    'all_body_vels': {...},
    'step_num_norm_activations': {...},
    'sum_muscles_activations': {...},
    # ... all other metrics preserved ...
}
```

## Performance Tips

1. **Memory**: If you run out of memory, reduce `--num_seeds`
2. **Disk Space**: With N checkpoints and M seeds, expect ~N×M times the single-seed output size
3. **Analysis**: Use seed index to group results and compute statistics across seeds
4. **Reproducibility**: Seed index determines RNG, same seed index = same randomness pattern

## Verification

The script will print:
- ✓ Number of checkpoints loaded
- ✓ Total batches (checkpoints × seeds)
- ✓ Progress updates every 500 steps
- ✓ Per-checkpoint mean and std deviation
- ✓ All save locations with ✓ confirmation

## Troubleshooting

**Issue**: "No checkpoints loaded successfully!"
- **Solution**: Check `--folder_path` contains `.pkl` files starting with "ckpt"

**Issue**: Output folders not created
- **Solution**: Check write permissions in parent directory of `--folder_path`

**Issue**: Very long execution time
- **Solution**: You may have a slow disk or insufficient parallelization hardware. This is expected if not using GPU.

**Issue**: Different rewards for same seed each run
- **Solution**: Each checkpoint-seed pair gets independent RNG, which is expected. To get identical results, ensure same initial RNG state.
