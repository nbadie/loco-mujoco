"""
COMPARISON: Sequential vs Improved Code Structure

================================================================================
BEFORE (Original nested while loops):
================================================================================

for param_name in randomization_params_names:
    for name in randomization_params_names:
        env._domain_randomizer.rand_conf[f"randomize_{name}"] = False
    
    if randomization_params_eval[f"randomize_{param_name}"]:
        env._domain_randomizer.rand_conf[f"randomize_{param_name}"] = True
        
        if "stiffness" in param_name or "damping" in param_name:
            current_value = min_val
            while current_value <= max_val:
                # Configure and run
                env._domain_randomizer.rand_conf[...]
                run_evaluation_loop(...)
                current_value += increment
                
        elif "position" in param_name or "orientation" in param_name:
            for axis in directions:
                current_value = min_val
                while current_value <= max_val + 1e-6:
                    # Configure and run (but commented out!)
                    # run_evaluation_loop(...)
                    current_value += inc

Problems:
- Deeply nested structure (4 levels)
- Evaluation calls were commented out in position/orientation branch
- Hard to see total number of configurations
- Difficult to add progress tracking
- Code duplication between branches

================================================================================
AFTER (Improved with pre-computed configurations):
================================================================================

# Step 1: Build all configurations upfront
eval_configs = []
for param_name in randomization_params_names:
    if randomization_params_eval[f"randomize_{param_name}"]:
        if "stiffness" in param_name or "damping" in param_name:
            current_value = min_val
            while current_value <= max_val:
                eval_configs.append({...})
                current_value += increment
        elif "position" in param_name or "orientation" in param_name:
            for axis in directions:
                current_value = min_val
                while current_value <= max_val + 1e-6:
                    eval_configs.append({...})
                    current_value += inc

print(f"Total configurations: {len(eval_configs)}")

# Step 2: Single loop to run all configurations
for config_idx, config in enumerate(eval_configs):
    # Extract config details
    param_name = config['param_name']
    param_value = config['param_value']
    direction = config['direction']
    
    # Configure domain randomization
    # ... (setup code)
    
    # Run evaluation
    print(f"[{config_idx+1}/{len(eval_configs)}] Running: {param_name}={param_value}")
    run_evaluation_loop(...)

print("EVALUATION COMPLETE")
print(f"Total time: {time_total:.2f}s")
print(f"Avg per config: {time_avg:.2f}s")

Benefits:
- Flat structure (1 level)
- All evaluations actually run (no commented code)
- Clear total count and progress tracking
- Easy to parallelize later (just modify the loop)
- No code duplication
- Better error handling potential

================================================================================
KEY INSIGHT:
================================================================================

The original code had a structural issue where the position/orientation 
evaluations were not actually being executed (commented out). The refactored
code fixes this and makes the structure much cleaner and easier to maintain.

The "parallel" aspect comes from:
1. Pre-computing all configs (easier to split into batches)
2. Each evaluation internally uses JAX vmap for step-level parallelism
3. Optional: Process multiple configs simultaneously (see parallel version)

================================================================================
"""
