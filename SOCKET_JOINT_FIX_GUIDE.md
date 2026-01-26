# Socket Joint Randomization - Practical Fix Guide

## Fix #1: Add Forward Pass After Modifying qpos (CRITICAL)

**File:** `/home/nadinebadie/loco-mujoco/loco_mujoco/core/domain_randomizer/prosthesis.py`

**Location:** Lines 795-827 (in the `update()` method)

**Current Code (BROKEN):**
```python
if self.rand_conf["randomize_prosthesis_socket_joint"]:
    # jax.debug.print('data.qpos before: {qpos}', qpos = data.qpos)
    # print('qpos before:', data.qpos)
    if backend == jnp:
        # Use JAX for randomization
        # Unpack joint stiffness dictionary into indices and values
        joint_names = list(domrand_state.prosthesis_socket_joint_value.keys())
        joint_values = list(domrand_state.prosthesis_socket_joint_value.values())
        joint_indices = [self._socket_joint_indices[name] for name in joint_names]

        # Convert to jnp arrays
        joint_indices = jnp.array(joint_indices)
        joint_values = jnp.array(joint_values)
        
        jax.debug.print('joint_values: {joint_values}', joint_values=joint_values)

        # Apply to data
        all_joint_values = data.qpos.at[joint_indices].set(joint_values)
        data = data.replace(qpos=all_joint_values) 
    elif backend == np: 
        all_joint_values = self._init_prosthesis_socket_joint_value.copy()
        for joint_name, value in domrand_state.prosthesis_socket_joint_value.items():
            idx = self._socket_joint_indices[joint_name]
            all_joint_values[idx] = value
        data.qpos = all_joint_values
    # ❌ MISSING: Forward pass here!
```

**FIXED Code:**
```python
if self.rand_conf["randomize_prosthesis_socket_joint"]:
    # jax.debug.print('data.qpos before: {qpos}', qpos = data.qpos)
    # print('qpos before:', data.qpos)
    if backend == jnp:
        # Use JAX for randomization
        joint_names = list(domrand_state.prosthesis_socket_joint_value.keys())
        joint_values = list(domrand_state.prosthesis_socket_joint_value.values())
        joint_indices = [self._socket_joint_indices[name] for name in joint_names]

        joint_indices = jnp.array(joint_indices)
        joint_values = jnp.array(joint_values)
        
        jax.debug.print('joint_values: {joint_values}', joint_values=joint_values)

        all_joint_values = data.qpos.at[joint_indices].set(joint_values)
        data = data.replace(qpos=all_joint_values)
        
        # ✅ ADD: Forward pass to recompute body positions and derivatives
        data = mjx.forward(model, data)
        
    elif backend == np: 
        all_joint_values = self._init_prosthesis_socket_joint_value.copy()
        for joint_name, value in domrand_state.prosthesis_socket_joint_value.items():
            idx = self._socket_joint_indices[joint_name]
            all_joint_values[idx] = value
        data.qpos = all_joint_values
        
        # ✅ ADD: Forward pass to recompute body positions and derivatives
        mujoco.mj_forward(model, data)
```

---

## Fix #2: Initialize Talus Position Correctly and Reuse It

**File:** `/home/nadinebadie/loco-mujoco/loco_mujoco/core/domain_randomizer/prosthesis.py`

**Location 1:** In `init_state()` method, around line 469

**Current Code:**
```python
###### Indices for tibia, socket, talus for initialization and parameter adaption for different socket_ty 
if 'socket_ty'+self.prosthesis_side_str in self._socket_joint_indices and "randomize_prosthesis_socket_joint" in self.rand_conf and self.rand_conf["randomize_prosthesis_socket_joint"]:
    tibia_name = 'tibia' + self.prosthesis_side_str
    socket_name = 'pylon_socket' + self.prosthesis_side_str
    talus_name = 'talus' + self.prosthesis_side_str
    # ... code ...
    # Get indices
    self._tibia_idx= mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tibia_name)
    self._socket_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, socket_name)
    self._talus_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, talus_name)
    # ❌ NOT STORING INITIAL TALUS POSITION HERE
```

**FIXED Code:**
```python
###### Indices for tibia, socket, talus for initialization and parameter adaption for different socket_ty 
if 'socket_ty'+self.prosthesis_side_str in self._socket_joint_indices and "randomize_prosthesis_socket_joint" in self.rand_conf and self.rand_conf["randomize_prosthesis_socket_joint"]:
    tibia_name = 'tibia' + self.prosthesis_side_str
    socket_name = 'pylon_socket' + self.prosthesis_side_str
    talus_name = 'talus' + self.prosthesis_side_str
    # ... code ...
    # Get indices
    self._tibia_idx= mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tibia_name)
    self._socket_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, socket_name)
    self._talus_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, talus_name)
    
    # ✅ ADD: Store the initial talus position at initialization
    self._init_talus_pos = model.body_pos[self._talus_idx].copy()
```

---

**Location 2:** In `__init__()` method, around line 70

**Current Code:**
```python
def __init__(self, env, **kwargs): 
    self._init_prosthesis_joint_stiffness = None 
    self._init_prosthesis_dof_damping = None
    self._init_prosthesis_body_position = None
    self._init_prosthesis_body_orientation = None
    self._init_prosthesis_socket_joint_value = None
    self._init_prosthesis_socket_joint_springref = None
    self.init_talus_pos = None  # ❌ This is initialized as None but should be set once
    super().__init__(env, **kwargs)
```

**FIXED Code:**
```python
def __init__(self, env, **kwargs): 
    self._init_prosthesis_joint_stiffness = None 
    self._init_prosthesis_dof_damping = None
    self._init_prosthesis_body_position = None
    self._init_prosthesis_body_orientation = None
    self._init_prosthesis_socket_joint_value = None
    self._init_prosthesis_socket_joint_springref = None
    self._init_talus_pos = None  # ✅ Use _init_talus_pos (consistent naming)
    super().__init__(env, **kwargs)
```

---

**Location 3:** In `update()` method, around line 838-865

**Current Code:**
```python
if 'socket_ty'+self.prosthesis_side_str in self._socket_joint_indices:
    # if not np.any(self.init_talus_pos):
        # jax.debug.print("IN LOOOOOOPPPPP")    
    self.init_talus_pos = model.body_pos[self._talus_idx].copy()  # ❌ RESET EVERY TIME!
    jax.debug.print("init_talus_pos: {init_talus_pos}", init_talus_pos=self.init_talus_pos)
    
    talus_offset_y = domrand_state.prosthesis_socket_joint_value[f"socket_ty"+self.prosthesis_side_str]
    jax.debug.print("talus_offset_y: {talus_offset_y}", talus_offset_y = talus_offset_y)
    
    new_talus_pos = self.init_talus_pos - backend.array([0,talus_offset_y,0])
    # ... rest of code ...
```

**FIXED Code:**
```python
if 'socket_ty'+self.prosthesis_side_str in self._socket_joint_indices:
    talus_offset_y = domrand_state.prosthesis_socket_joint_value[f"socket_ty"+self.prosthesis_side_str]
    jax.debug.print("talus_offset_y: {talus_offset_y}", talus_offset_y = talus_offset_y)
    
    # ✅ USE the initialized reference position, don't reset it
    new_talus_pos = self._init_talus_pos - backend.array([0, talus_offset_y, 0])
    jax.debug.print("new_talus_pos: {new_talus_pos}", new_talus_pos=new_talus_pos)
    
    if not self.rand_conf["randomize_prosthesis_body_position"]:
        jax.debug.print("new talus pos: {new_talus_pos}", new_talus_pos=new_talus_pos) 
        if backend == np: 
            body_pos = self._init_prosthesis_body_position.copy()
            body_pos[self._talus_idx] = new_talus_pos
        elif backend == jnp: 
            body_pos = model.body_pos.at[jnp.array(self._talus_idx)].set(new_talus_pos)
        jax.debug.print("body_pos: {body_pos}", body_pos=body_pos)
        model = self._set_attribute_in_model(model, "body_pos", body_pos, backend)
```

---

## Fix #3: Avoid Double-Applying Talus Offset

**File:** `/home/nadinebadie/loco-mujoco/loco_mujoco/core/domain_randomizer/prosthesis.py`

**Location:** Lines 898-911

**Current Code (with potential double application):**
```python
if self.rand_conf["randomize_prosthesis_body_position"]:
    # ... sample body positions ...
    
    # If talus height needs to be adapted depending on socket_ty (so if socket_ty in randomize_prosthesis_socket_joint)
    if hasattr(self, '_talus_idx'):
        talus_offset_y = domrand_state.prosthesis_socket_joint_value[f"socket_ty"+self.prosthesis_side_str]
        # This is SECOND application if socket_joint randomization also adjusted talus!
        talus_offset_array = backend.array([0,talus_offset_y,0])
        if backend == np: 
            body_pos[self._talus_idx] -= talus_offset_array
        elif backend == jnp: 
            body_pos = body_pos.at[jnp.array(self._talus_idx)].set(body_pos.at[jnp.array(self._talus_idx)].get() - talus_offset_array)
```

**FIXED Code:**
```python
if self.rand_conf["randomize_prosthesis_body_position"]:
    # ... sample body positions ...
    
    # ✅ ONLY apply talus offset if socket joint randomization is NOT enabled
    # (avoid double application)
    if hasattr(self, '_talus_idx') and not self.rand_conf.get("randomize_prosthesis_socket_joint", False):
        talus_offset_y = domrand_state.prosthesis_socket_joint_value[f"socket_ty"+self.prosthesis_side_str]
        talus_offset_array = backend.array([0, talus_offset_y, 0])
        if backend == np: 
            body_pos[self._talus_idx] -= talus_offset_array
        elif backend == jnp: 
            body_pos = body_pos.at[jnp.array(self._talus_idx)].set(
                body_pos.at[jnp.array(self._talus_idx)].get() - talus_offset_array)
```

---

## Fix #4: Verify qpos_spring is Applied in Simulation

**File:** `/home/nadinebadie/loco-mujoco/loco_mujoco/core/mujoco_mjx.py`

**Location:** In `mjx_step()` method, after line 827

**Current Code:**
```python
# modify data and model *before* step if needed
sys, data, carry = self._mjx_simulation_pre_step(self.sys, data, carry)
# ... then step ...
```

**FIXED Code:**
```python
# modify data and model *before* step if needed
sys, data, carry = self._mjx_simulation_pre_step(self.sys, data, carry)

# ✅ ADD: If domain randomization modified the model, update the JAX system
if self._domain_randomizer.rand_conf.get("randomize_prosthesis_socket_joint", False):
    # Ensure the JAX model includes any updates to qpos_spring or other model params
    sys = mjx.put_model(self._model)

# ... then step ...
```

---

## Fix #5: Update the Visualization Function (For Consistency)

**File:** `/home/nadinebadie/loco-mujoco/loco_mujoco/core/mujoco_mjx.py`

**Location:** `mjx_render_domain_randomization()` around line 1285-1305

The rendering function is already doing this correctly, but ensure it's consistent. No change needed here, but this is a reference for what the simulation should be doing.

---

## Testing Script

Add this to your evaluation script to verify the fix works:

```python
# After creating the environment and before the rollout loop:

# Test 1: Check state consistency
def check_state_consistency(env, state):
    """Verify that qpos and body positions are consistent."""
    model = env.get_model()
    
    # Get socket_ty joint index
    socket_ty_idx = env._domain_randomizer._socket_joint_indices.get('socket_ty_r', None)
    if socket_ty_idx is None:
        print("Socket joint not found, skipping consistency check")
        return
    
    socket_ty_value = state.data.qpos[socket_ty_idx]
    talus_y_pos = state.data.xpos[env._domain_randomizer._talus_idx][1]
    init_talus_y = env._domain_randomizer._init_talus_pos[1]
    expected_talus_y = init_talus_y - socket_ty_value
    
    print(f"\n=== State Consistency Check ===")
    print(f"Socket_ty value: {socket_ty_value:.6f}")
    print(f"Talus Y position: {talus_y_pos:.6f}")
    print(f"Expected talus Y: {expected_talus_y:.6f}")
    print(f"Difference: {abs(talus_y_pos - expected_talus_y):.6f}")
    
    if abs(talus_y_pos - expected_talus_y) > 0.001:
        print("⚠️  WARNING: State inconsistency detected!")
    else:
        print("✅ State is consistent")

# In your rollout loop:
for step in range(n_steps):
    if step == 0:
        check_state_consistency(env, env_state)
    
    # ... normal step ...
    env_state = jit_step(env_state, action)
```

---

## Summary of Changes

| Issue | File | Line | Fix |
|-------|------|------|-----|
| Missing forward pass | prosthesis.py | 816-827 | Add `mjx.forward()` or `mj_forward()` |
| Talus position reset every update | prosthesis.py | 846 | Use stored `_init_talus_pos` instead |
| Talus position never initialized | prosthesis.py | 469 | Store `_init_talus_pos` in `init_state()` |
| Double talus offset | prosthesis.py | 898-911 | Add condition to prevent double application |
| JAX model not updated | mujoco_mjx.py | ~830 | Call `mjx.put_model()` after randomization |

These fixes should resolve the foot penetration and walking issues!
