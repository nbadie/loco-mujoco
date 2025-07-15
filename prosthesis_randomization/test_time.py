
import jax
import jax.numpy as jnp
import time

# Allocate a large matrix
x = jnp.ones((3000, 3000))

# Warmup
_ = jnp.dot(x, x).block_until_ready()

# Measure execution time
start = time.time()
_ = jnp.dot(x, x).block_until_ready()
print("Time on GPU:", time.time() - start)

with jax.default_device(jax.devices("cpu")[0]):
    x_cpu = jnp.ones((3000, 3000))
    _ = jnp.dot(x_cpu, x_cpu).block_until_ready()
    start = time.time()
    _ = jnp.dot(x_cpu, x_cpu).block_until_ready()
    print("Time on CPU:", time.time() - start)