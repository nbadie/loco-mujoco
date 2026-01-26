# Socket Joint Randomization - Visual Summary

## The Problem in 3 Steps

### ❌ WHAT'S HAPPENING NOW (BROKEN):

```
Step 1: Reset
────────────
model.qpos0 → Sample socket_ty value → 0.01 m
             ↓
        Store in carry state
             ↓
        Add to domain_randomizer_state

Step 2: Update (in mjx_step)
────────────
data.qpos[socket_ty_idx] ← 0.01 m (CHANGED)
             ↓
        ❌ NO FORWARD PASS!
             ↓
data.xpos (body positions) ← OLD VALUES (NOT UPDATED!)
             ↓
Model body_pos is modified, but body kinematics are stale

Step 3: Physics Simulation
────────────
mj_step() / mjx_step() runs with INCONSISTENT state:
- Joint position says: socket moved 0.01 m
- Body positions say: socket at old location
- Talus foot is at stale position
             ↓
        Ground collision computed incorrectly
             ↓
Foot penetrates ground OR loses contact unexpectedly
```

---

### ✅ WHAT SHOULD HAPPEN (FIXED):

```
Step 1: Reset
────────────
Same as before - sample socket_ty = 0.01 m

Step 2: Update (in mjx_step)
────────────
data.qpos[socket_ty_idx] ← 0.01 m (CHANGED)
             ↓
        ✅ CALL FORWARD PASS!
             ↓
mjx.forward(model, data) / mj_forward(model, data)
             ↓
All body positions (xpos) recomputed from new qpos
All body velocities (cvel) recomputed
All joint derivatives computed
             ↓
Body positions NOW CONSISTENT with joint positions!

Step 3: Talus Offset Adjustment
────────────
talus_y_new = talus_y_init - socket_ty_offset
             ↓
Apply this to model.body_pos[talus_idx]
             ↓
Call forward pass AGAIN to ensure consistency
             ↓
state is now FULLY CONSISTENT

Step 4: Physics Simulation
────────────
mj_step() / mjx_step() runs with CONSISTENT state:
- Joint position: socket moved 0.01 m
- Body positions: socket at correct new location
- Talus foot: at adjusted position (ground contact)
             ↓
        Ground collision computed correctly
             ↓
Foot remains in contact with ground ✅
Model can walk properly ✅
```

---

## The Key Problem: Missing Forward Pass

### Analogy

Imagine building a robot:
1. You move joint 3 to a new angle (like rotating the socket_ty)
2. **You MUST recalculate all the positions of all body parts** that depend on that joint
3. If you don't recalculate, the foot thinks it's in the old place while the joint is in a new place
4. The foot ends up penetrating the ground

In MuJoCo:
- Modifying `qpos` = moving a joint
- Calling `mj_forward()` / `mjx.forward()` = recalculating all body positions
- **Skipping the forward pass = forgetting to recalculate = geometrically inconsistent state**

---

## The Root Cause Chain

```
╔════════════════════════════════════════════════════════════════╗
║  Root Cause: Missing mjx.forward() after qpos modification    ║
╚════════════════════════════════════════════════════════════════╝
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
        ┌───────────────────┐  ┌─────────────────┐
        │ Joint positions   │  │ Body Cartesian  │
        │ are UPDATED       │  │ positions are   │
        │ (qpos changed)    │  │ STALE (xpos old)│
        └─────────┬─────────┘  └────────┬────────┘
                  │                     │
                  ↓                     ↓
          ┌──────────────────────────────────────┐
          │   GEOMETRICAL INCONSISTENCY          │
          │                                      │
          │  Joint says: Socket moved 0.01m      │
          │  Body says: Socket at old position   │
          │  Foot: Penetrates ground!            │
          └──────────────────────────────────────┘
                  ↓
    ┌─────────────┴─────────────┐
    ↓                           ↓
Can't walk              Foot goes through
properly                 ground
```

---

## Code Locations and Fixes

### Fix Location #1: Core Update Function

**File:** `prosthesis.py` Line 816-827

**Before (BROKEN):**
```python
data = data.replace(qpos=all_joint_values)
# ❌ NOTHING HAPPENS - body positions are stale!
```

**After (FIXED):**
```python
data = data.replace(qpos=all_joint_values)
✅ data = mjx.forward(model, data)  # Recompute everything!
```

### Fix Location #2: Talus Position Reference

**File:** `prosthesis.py` Line 846

**Before (BROKEN):**
```python
self.init_talus_pos = model.body_pos[self._talus_idx].copy()  # Reset every time!
# Problem: Changes baseline every iteration, causes drift
```

**After (FIXED):**
```python
# In init_state (line 469):
self._init_talus_pos = model.body_pos[self._talus_idx].copy()  # Set ONCE

# In update (line 846):
new_talus_pos = self._init_talus_pos - backend.array([0, talus_offset_y, 0])
# Now it's always relative to the TRUE initial position
```

---

## Why Your Visualization Works But Simulation Doesn't

Your `mjx_render_domain_randomization()` function works correctly because:

```python
def mjx_render_domain_randomization(self, state, record=False):
    model = self.update_mjM_post_domain_randomizer(state.additional_carry)
    # update_mjM_post_domain_randomizer() applies ALL randomizations
    # Then immediately renders without stepping
    
    # ✅ It modifies the model but doesn't simulate, so inconsistency doesn't matter for visualization
```

But in `mjx_step()`, the inconsistency causes problems because:

```python
def mjx_step(self, state, action):
    # Modifies qpos without forward pass
    # ❌ Then immediately simulates with inconsistent state
    # ❌ Physics breaks down with inconsistent geometry
```

---

## Checklist of Issues and Fixes

### Issue 1: Missing Forward Pass
- **Location:** `prosthesis.py` line 816-827
- **Severity:** 🔴 CRITICAL
- **Fix:** Add `mjx.forward()` or `mj_forward()`
- **Impact:** Fixes foot penetration and walking

### Issue 2: Talus Position Reset Every Update
- **Location:** `prosthesis.py` line 846
- **Severity:** 🔴 CRITICAL  
- **Fix:** Use stored `_init_talus_pos`, don't reset
- **Impact:** Prevents position drift

### Issue 3: Talus Position Never Initialized
- **Location:** `prosthesis.py` line 469
- **Severity:** 🟠 HIGH
- **Fix:** Store `_init_talus_pos` during `init_state()`
- **Impact:** Provides stable reference point

### Issue 4: Double Talus Offset
- **Location:** `prosthesis.py` line 898-911
- **Severity:** 🟠 HIGH
- **Fix:** Conditional logic to avoid double application
- **Impact:** Prevents geometry corruption

### Issue 5: JAX Model Not Updated
- **Location:** `mujoco_mjx.py` line ~830
- **Severity:** 🟡 MEDIUM
- **Fix:** Call `mjx.put_model()` after randomization
- **Impact:** Ensures JAX system reflects changes

---

## Expected Behavior After Fixes

### Before Fix
```
Rollout step 0: Model standing (randomized position)
Rollout step 1: Foot penetrates ground 💔
Rollout step 2-3: Model collapses / can't maintain balance
Rollout step 4+: Simulation unstable
```

### After Fix
```
Rollout step 0: Model standing (randomized position)
Rollout step 1: Model takes first step (balanced)
Rollout step 2-3: Model walks smoothly 💪
Rollout step 4+: Consistent walking motion
```

---

## Testing the Fix

Quick test to verify state consistency:

```python
# After env_state = jit_reset(env_keys)

socket_ty_idx = env._domain_randomizer._socket_joint_indices['socket_ty_r']
socket_ty_value = env_state.data.qpos[socket_ty_idx]
talus_y = env_state.data.xpos[env._domain_randomizer._talus_idx][1]
expected_y = env._domain_randomizer._init_talus_pos[1] - socket_ty_value

print(f"Socket_ty: {socket_ty_value:.6f}")
print(f"Talus Y: {talus_y:.6f}")
print(f"Expected Y: {expected_y:.6f}")
print(f"Difference: {abs(talus_y - expected_y):.6f}")

# After fix, difference should be < 0.001
# Before fix, difference could be > 0.01
```

---

## Files Modified Summary

```
prosthesis.py
├── Line 469: Add _init_talus_pos storage
├── Line 816-827: Add forward pass (CRITICAL)
├── Line 846: Use _init_talus_pos instead of resetting
└── Line 898-911: Add condition to avoid double offset

mujoco_mjx.py
├── Line ~830: Add mjx.put_model() call (if needed)
└── Lines 1285-1305: Already correct (reference only)
```

---

## Why This Matters

The socket joint randomization is trying to:
- Change the position of the prosthetic socket relative to the leg
- This is a REALISTIC variation (socket fit varies, prosthesis wear, etc.)
- It's ESSENTIAL for robust learning

But without proper state synchronization:
- The simulation becomes PHYSICALLY INVALID
- Ground contacts are computed INCORRECTLY
- The agent learns from CORRUPTED data

With the fix:
- Simulation remains physically valid
- Ground contacts are correct
- Agent learns from realistic, valid interactions
