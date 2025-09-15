import jax 
import jax.numpy as jnp

import matplotlib.pyplot as plt


# Gholizadeh, Hossein, et al. "A new approach for the pistoning measurement in transtibial prosthesis." Prosthetics and orthotics international 35.4 (2011): 360-364.

disp_30N = -0.004
disp_60N = -0.006
disp_90N = -0.008
disp_0N = 0.0
disp_BW = 0.01 # 10 mm relative to 0N point

force_90N = 90.0
force_60N = 60.0
force_30N = 30.0
force_0N = 0.0
force_BW = 750.0

xp = jnp.array([
    disp_90N,
    disp_60N,
    disp_30N,
    disp_0N,
    disp_BW
    ])
fp = jnp.array([
    -force_90N,
    -force_60N,
    -force_30N,
    force_0N,
    force_BW
    ])

x = jnp.linspace(-0.02, 0.03, 500)

F1 = jnp.interp(x, xp, fp, left="extrapolate", right="extrapolate")
# plot the dots in the data
plt.scatter(xp*1e3, fp, color='red')

plt.plot(x*1e3, F1, label=f'kp socket data')
plt.xlabel('displacement (mm)')
plt.ylabel('force (N)')
plt.legend()
plt.grid(True)
plt.savefig("force_vs_displacement_test_comparison.png")