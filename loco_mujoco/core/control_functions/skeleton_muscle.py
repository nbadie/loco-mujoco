from loco_mujoco.core.control_functions import DefaultControl
import jax.numpy as jnp
import mujoco

class SkeletonMuscleControl(DefaultControl): 
    """
        Control function for skeleton muscle control. 
    """

    def generate_action(self, env, action, model, data, carry, backend): 
        """
        Calculates the action. This function scales the muscle action from [-1,1] to [0,1].
        """

        for i in range(model.nu):
            if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE:
                # scale the action from [0,1] to the original action space
                action = action.at[i].set(self.adapt_sigmoid(action[i]))

        return action, carry
    

    def adapt_sigmoid(self, action):
        """
        Applies a sigmoid activation function to the action, adapted to the actuator limit. 
        The sigmoid has a constant for a steeper slope

        Args: 
            action: the action to be scaled, expected to be in the range [-1,1]

        Returns:
            action: the scaled action, in the range [0,1]
        """
        q = 5
        return 1 / (1 + jnp.exp(-q * action))


