# Socket Joint Randomization Issues - Detailed Analysis

## Problem Summary
Your socket joint randomization is causing the foot on the socket side to penetrate through the ground, and the model cannot walk properly. This is a **critical state synchronization problem** between how you sample the socket joint values and how they are applied during simulation.

## Root Causes

### Issue 1: **Sampling Occurs at Reset Only, But Data.qpos is Not Updated at Reset**

**Location in Code:**
- `prosthesis.py` lines 613: `prosthesis_socket_joint_value, carry = self._sample_socket_joint_value(model, data, carry, backend)`
- `prosthesis.py` lines 795-827: In the `update()` function, `data.qpos` is modified
- `prosthesis.py` line 461: Socket joint values are sampled from `model.qpos0` (the default position)

**The Problem:**
```python
# In _sample_socket_joint_value (line 461)
prosthesis_socket_joint_value[joint_name] = backend.array([model.qpos0[self._socket_joint_indices[joint_name]]])
```

You sample from `model.qpos0`, which is the **default/initial configuration**. However:
1. At reset, the socket joint position is sampled
2. The sampled value is stored in the carry state
3. In the `update()` function (line 816-823), you set `data.qpos` to this sampled value
4. **But `mjx.forward()` is NOT called after modifying `data.qpos`**

This means:
- The joint position changes, but
- The body positions (talus, socket) are NOT recomputed
- The talus body position remains at its old location while the joint position changes
- This creates an inconsistent state where the talus is disconnected from the socket

### Issue 2: **Missing Forward Pass After Modifying Joint Positions**

**The Critical Missing Step:**

After you modify `data.qpos` for the socket joint, you need to call `mjx.forward()` to:
1. Recompute all body positions based on the new joint positions
2. Update body velocities and accelerations
3. Ensure geometric consistency

Currently in `prosthesis.py` (lines 816-827):
```python
if self.rand_conf["randomize_prosthesis_socket_joint"]:
    # ... set data.qpos ...
    data = data.replace(qpos=all_joint_values)
    
    # ❌ MISSING: mjx.forward() to recompute body positions!
```

### Issue 3: **Talus Position Offset Applied Inconsistently**

**Location:** `prosthesis.py` lines 838-865

Your code tries to adjust the talus position based on socket_ty offset:
```python
new_talus_pos = self.init_talus_pos - backend.array([0,talus_offset_y,0])
```

**The Problems:**
1. `self.init_talus_pos` is only set once (line 846), then reused for all subsequent steps
2. The talus offset is being applied to `model.body_pos` (the kinematic tree), not to the actual simulated position
3. **The offset is applied BEFORE the forward pass**, so it gets overwritten

Even worse, when both `randomize_prosthesis_socket_joint` AND `randomize_prosthesis_body_position` are disabled but socket joint is randomized, the logic is:
- Socket joint changes the `data.qpos` 
- Talus body position is adjusted in the model
- But without a forward pass, these don't become consistent
- The foot remains at its old position while the socket has moved

### Issue 4: **Double Application of Talus Offset**

Looking at `mujoco_mjx.py` lines 1340-1365 and `prosthesis.py` lines 898-911:

If `randomize_prosthesis_body_position` is True, the talus offset is applied twice:
1. Once in the socket joint randomization block (line 842-851)
2. Again in the body position randomization block (line 898-911)

This creates inconsistent geometry.

### Issue 5: **Spring Reference Position Not Synchronized with Data**

**Location:** `prosthesis.py` lines 827-830

```python
updated_springref, carry = self._set_joint_springref(...)
model = self._set_attribute_in_model(model, 'qpos_spring', updated_springref, backend)
```

You update `model.qpos_spring` (the spring reference position), but:
1. This is only used in the MuJoCo model
2. It's applied in `mjx_render_domain_randomization()` for visualization
3. **During `mjx_step()`, this update may not be properly applied to the simulation model**

The rendering code (`mujoco_mjx.py` lines 1285-1305) properly handles this, but the step function may not.

## Why the Foot Goes Through the Ground

**Sequence of Events:**
1. Reset occurs, socket_ty is randomized to some value (e.g., 0.01 m)
2. `data.qpos` is updated with this new socket_ty value
3. **But `mjx.forward()` is NOT called**, so body positions aren't recomputed
4. The talus body position in the kinematic tree is adjusted in `model.body_pos`
5. However, the actual simulated `data.xpos` (Cartesian position) for the talus is stale
6. When physics simulation runs, it computes contact forces based on stale positions
7. The foot thinks it's at the old position but the joint says it should be elsewhere
8. Ground collision is computed incorrectly, or the foot ends up penetrating

## Why the Model Can't Walk

1. **Inconsistent state** causes the leg kinematics to be incorrect
2. The foot contact points are misaligned with the actual geometry
3. Ground reaction forces are applied at wrong locations
4. The controller receives incorrect observations (foot position is wrong)
5. This leads to loss of balance and inability to maintain walking

## Solutions

### Solution 1: **Call mjx.forward() After Modifying qpos**

**In `prosthesis.py`, modify the `update()` function around line 816:**

```python
if self.rand_conf["randomize_prosthesis_socket_joint"]:
    # existing code...
    data = data.replace(qpos=all_joint_values)
    
    # ✅ ADD THIS: Recompute body positions and derivatives
    if backend == jnp:
        data = mjx.forward(model, data)  # Recompute kinematics
    elif backend == np:
        mujoco.mj_forward(model, data)
    
    # Now the talus position adjustment will be based on consistent state
```

### Solution 2: **Ensure Talus Position Offset is Applied Correctly**

The talus offset should be applied AFTER the forward pass, and it should update both the model and ensure consistency:

```python
if 'socket_ty'+self.prosthesis_side_str in self._socket_joint_indices:
    talus_offset_y = domrand_state.prosthesis_socket_joint_value[f"socket_ty"+self.prosthesis_side_str]
    
    # After forward pass, the talus position is now consistent
    # Adjust it based on socket offset
    new_talus_pos = model.body_pos[self._talus_idx].copy()
    new_talus_pos[1] -= talus_offset_y  # Adjust Y position
    
    body_pos = model.body_pos.at[jnp.array(self._talus_idx)].set(new_talus_pos)
    model = self._set_attribute_in_model(model, "body_pos", body_pos, backend)
```

### Solution 3: **Don't Double-Apply the Talus Offset**

In the `update()` function in `mujoco_mjx.py`, lines 1278-1317:
- Remove the redundant talus offset application if it's already being done in the socket joint block
- OR refactor so the offset is applied only once

### Solution 4: **Properly Synchronize qpos_spring in the Step Function**

Ensure that when domain randomization updates the spring reference, it's actually used in the step simulation:

```python
# In mjx_step, after calling _mjx_simulation_pre_step
# Verify that the randomized model includes updated qpos_spring values
if self._domain_randomizer.rand_conf.get("randomize_prosthesis_socket_joint"):
    # Ensure model's qpos_spring reflects the randomization
    sys = mjx.put_model(model)  # Update the JAX model from the NumPy model
```

### Solution 5: **Initialize init_talus_pos Properly**

Change line 846 in `prosthesis.py`:

```python
# ❌ Current (problematic):
# self.init_talus_pos = model.body_pos[self._talus_idx].copy()

# ✅ Better: Initialize once during init_state
# In init_state (line 469):
if 'socket_ty'+self.prosthesis_side_str in self._socket_joint_indices:
    self._init_talus_pos = model.body_pos[self._talus_idx].copy()
    self.init_talus_pos = self._init_talus_pos  # Use consistent reference

# Then in update, reuse:
if 'socket_ty'+self.prosthesis_side_str in self._socket_joint_indices:
    talus_offset_y = domrand_state.prosthesis_socket_joint_value[...]
    new_talus_pos = self._init_talus_pos.copy() - backend.array([0, talus_offset_y, 0])
```

## Recommended Implementation Priority

1. **CRITICAL:** Add `mjx.forward()` call after modifying `data.qpos` (Solution 1)
2. **HIGH:** Ensure proper initialization of reference positions (Solution 5)
3. **HIGH:** Avoid double-applying talus offset (Solution 3)
4. **MEDIUM:** Properly synchronize qpos_spring (Solution 4)
5. **MEDIUM:** Refactor talus offset application for clarity (Solution 2)

## Testing Recommendations

After implementing fixes:

1. **Verify state consistency:**
   - Check that `data.xpos[talus_id]` equals the expected position from the socket + joint offset
   
2. **Test with small offsets first:**
   - Set `socket_joint_range = {'socket_ty': [0.001, 0.001]}` (tiny offset)
   - Verify the model still walks
   
3. **Visualize using mjx_render_domain_randomization:**
   - This should show the randomized geometry correctly
   - Compare it with what's actually simulated
   
4. **Check observation validity:**
   - Print the foot position observations
   - Verify they match the visual/geometric positions

## Code References

- **Main Update:** `prosthesis.py` lines 648-924
- **Socket Joint Sampling:** `prosthesis.py` lines 1569-1610
- **Rendering (correct implementation):** `mujoco_mjx.py` lines 1027-1075
- **Model Update:** `mujoco_mjx.py` lines 1184-1414

## Additional Notes

The `mjx_render_domain_randomization` function correctly applies all randomizations and then calls `parallel_render()`. The issue is that during the actual `mjx_step()` simulation, the full sequence of updates isn't being applied with proper forward passes to maintain state consistency.
