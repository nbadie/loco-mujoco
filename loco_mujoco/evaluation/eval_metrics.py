import mujoco
import jax.numpy as jnp
import numpy as np


class ProsthesisEvalMetrics(): 

    def __init__(self, env):
        self.env = env
        self.model = env.get_model()



    def extract_checkpoint_data(self,data, checkpoint_idx):
        """
        Extract the checkpoint_idx slice from all arrays in nested structures.
        For body_xposes: {body: [step0_array(num_ckpt, 3), step1_array(num_ckpt, 3), ...]}
        Result: {body: [step0_array(3,), step1_array(3,), ...]}  - all steps for one checkpoint
        """
        if isinstance(data, dict):
            # For dicts, recursively extract from all values
            return {k: self.extract_checkpoint_data(v, checkpoint_idx) for k, v in data.items()}
        elif isinstance(data, list) and len(data) > 0:
            # For lists of arrays (like steps), extract checkpoint_idx from each array in the list
            result = []
            for step_array in data:
                if isinstance(step_array, (jnp.ndarray, list)):
                    try:
                        # Extract checkpoint_idx from this step's array
                        result.append(step_array[checkpoint_idx])
                    except (IndexError, TypeError):
                        result.append(step_array)
                else:
                    # Scalar or other type - keep as is
                    result.append(step_array)
            return result
        elif isinstance(data, (jnp.ndarray, list)) and hasattr(data, '__getitem__'):
            # Single array: try to extract
            try:
                return data[checkpoint_idx]
            except (IndexError, TypeError):
                return data
        else:
            return data



    def get_xpos(self, mjx_data):
        """
        Get the position of a body from the Mujoco data.
        """
        body_xpos = {}
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            # Get all seeds for body i
            body_pos = mjx_data.xpos[:, i] 
            body_xpos[body_name] = body_pos
            
        return body_xpos

    

    def get_cvel(self, mjx_data):
        """
        Get the velocity of a body from the Mujoco data.
        """
        body_cvel = {}
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            # Get all seeds for body i
            body_vel = mjx_data.cvel[:, i]  # shape: [1,n_bodies,n_seeds]
            body_cvel[body_name] = body_vel
            # jax.debug.print("Body: {b}, Vel shape all: {p}, Body Vel all: {body_pos}", b=body_name, p=mjx_data.cvel.shape, body_pos=mjx_data.cvel)
        return body_cvel



    def get_sensor_data(self, mjx_data):
        """
        Get sensor data from the Mujoco data.
        Returns (sensor_data, sensor_names) where
        - sensor_data is a dict name -> [batch, 3] (PyTree, JAX-friendly)
        - sensor_names is a list of names (static Python)
        """

        # Cache sensor names (static, not traced)
        if not hasattr(self, "_batched_sensor_names"):
            self._batched_sensor_names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                for i in range(self.model.nsensor)
            ]
        sensor_names = self._batched_sensor_names
        nsensor = self.model.nsensor

        # sensordata shape: [batch, nsensor*3]
        sensordata = mjx_data.sensordata

        # Split into [batch, nsensor, 3]
        sensordata_split = sensordata.reshape(-1, nsensor, 3)

        # Make a dict {name: [batch, 3]}
        # sensor_data = {name: sensordata_split[:, i, :] for i, name in enumerate(sensor_names)}
        sensor_data = {name: sensordata_split[:, i] for i, name in enumerate(sensor_names)}

        return sensor_data, sensor_names
    


    def get_joint_angles(self,mjx_data): 
        """ 
        Get all joint angles and save in dictionary with joint name as key and angle as value.
        """
        joint_angles = {}

        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_address = self.model.jnt_qposadr[i]
            joint_angle = mjx_data.qpos[...,qpos_address]
            joint_angles[joint_name] = joint_angle
            
        return joint_angles
    
    

    def get_grf(self, mjx_data, foot_name):
        """
        Get the ground reaction forces (GRF) from the Mujoco data.
        """
        # Get the body IDs for left and right foot
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_name)


        # Extract the GRF for left and right foot
        grf = mjx_data.cfrc_ext[:, body_id,:]

        return grf
    


class ProsthesisPlot(): 

    def get_start_steps_from_grfZ(self, grfZ, min_walk_step_length=60, threshold=0.5):
        """
        Detect step start indices from vertical GRF data.

        Args:
            grfZ (list or np.array): Vertical ground reaction force data.
            min_walk_step_length (int): Minimum number of frames between steps.
            threshold (float): GRF threshold to detect foot contact.

        Returns:
            list: Indices where steps start.
        """
        # Alternative only next 20 time steps still above 0 
        grfZ = np.array(grfZ)
        above_threshold = grfZ > threshold
        step_starts = []

        for i in range(1, len(above_threshold) - 20):  # Ensure we have 20 values ahead
            # Detect rising edge and check next 20 values
            if above_threshold[i] and not above_threshold[i - 1]:
                if np.all(grfZ[i:i+20] > 0):
                    if not step_starts or (i - step_starts[-1]) > min_walk_step_length:
                        step_starts.append(i)

        return step_starts
    

    def get_contact_lengths_from_grfZ(self, grfZ, min_walk_step_length=20, threshold=50):
        """
        Detect contact lengths from vertical GRF data.

        Args:
            grfZ (list or np.array): Vertical ground reaction force data.
            min_walk_step_length (int): Minimum number of frames to consider a valid contact.
            threshold (float): GRF threshold to define contact (default is 50 N).

        Returns:
            list: Lengths of valid contact segments.
        """
        grfZ = np.array(grfZ)
        above_threshold = grfZ > threshold

        contact_lengths = []
        in_contact = False
        start_idx = None

        for i, val in enumerate(above_threshold):
            if val and not in_contact:
                # Start of contact
                in_contact = True
                start_idx = i
            elif not val and in_contact:
                # End of contact
                end_idx = i
                length = end_idx - start_idx
                if length >= min_walk_step_length:
                    contact_lengths.append(length)
                in_contact = False

        # Handle case where contact continues till the end
        if in_contact:
            length = len(grfZ) - start_idx
            if length >= min_walk_step_length:
                contact_lengths.append(length)

        return contact_lengths
    
    def get_contact_from_grfZ(self, grfZ, threshold=50):
        """
        Get binary contact information from vertical GRF data.
        """
        grfZ = np.array(grfZ)
        above_threshold = grfZ > threshold
        in_contact = 100*above_threshold
        return in_contact
    


    def plot_grf_Fz_switch_all_dirs(
        self,
        all_loaded_data,
        run_step_data,
        interp_len=100,
        body_weight_to_normalize=1,
        threshold=1e-2,
    ):
        """
        Plots mean GRF Fz for each seed separately.
        
        """
        # Structure to return: {category: {seed: [switches]}}
        right_switches = {}
        left_switches = {}
        
        # 1. Iterate through Categories (e.g., 'z_-6deg')
        for cat_key, seeds_dict in all_loaded_data.items():
            right_switches[cat_key] = {}
            left_switches[cat_key] = {}
            
            # 2. Iterate through Seeds (e.g., 'seed1')
            for seed_key, run_dict in seeds_dict.items():
                
                # --- SAFE ACCESS TO STEP DATA ---
                # We check if cat_key and seed_key exist in run_step_data
                if cat_key not in run_step_data:
                    print(f"Skipping: Category '{cat_key}' not found in run_step_data")
                    continue
                
                cat_step_entry = run_step_data[cat_key]
                
                # Check if run_step_data is nested [cat][seed] or flat [cat]
                if isinstance(cat_step_entry, dict) and seed_key in cat_step_entry:
                    step_data = cat_step_entry[seed_key]
                else:
                    # Fallback if step_data is just indexed by category
                    step_data = cat_step_entry

                # Extract data arrays
                all_grf_l = run_dict.get("all_grf_l", [])
                all_grf_r = run_dict.get("all_grf_r", [])
                all_step_start_left = step_data.get("step_start_left", [])
                all_step_start_right = step_data.get("step_start_right", [])

                # --- Process Right Side ---
                right_steps = []
                if all_grf_r is not None and len(all_step_start_right) > 1:
                    for j in range(len(all_step_start_right) - 1):
                        start, end = int(all_step_start_right[j]-1), int(all_step_start_right[j + 1]-1)
                        # Get 5th index (Fz) for this segment
                        segment = all_grf_r[start:end]
                        if len(segment) < 2: continue
                        
                        y = [float(np.array(a)[5]) for a in segment]
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        right_steps.append(np.interp(x_new, x_old, y))
                
                if right_steps:
                    mean_right = np.mean(right_steps, axis=0) / body_weight_to_normalize
                    above = mean_right > threshold
                    switches = [i for i in range(1, len(mean_right)) if above[i-1] and not above[i]]
                    
                    right_switches[cat_key][seed_key] = switches

                # --- Process Left Side ---
                left_steps = []
                if all_grf_l is not None and len(all_step_start_left) > 1:
                    for j in range(len(all_step_start_left) - 1):
                        start, end = int(all_step_start_left[j]-1), int(all_step_start_left[j + 1]-1)
                        segment = all_grf_l[start:end]
                        if len(segment) < 2: continue
                            
                        y = [float(np.array(a)[5]) for a in segment]
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        left_steps.append(np.interp(x_new, x_old, y))
                
                if left_steps:
                    mean_left = np.mean(left_steps, axis=0) / body_weight_to_normalize
                    above = mean_left > threshold
                    switches = [i for i in range(1, len(mean_left)) if above[i-1] and not above[i]]
                    
                    left_switches[cat_key][seed_key] = switches

        return left_switches, right_switches