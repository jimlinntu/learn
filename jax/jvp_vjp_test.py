import unittest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def simple_model(x: jax.Array, w1: jax.Array, w2: jax.Array):
  h = jnp.matmul(x, w1)
  h = jnp.matmul(h, w2)
  return jnp.sum(jnp.square(h))


class TestJVPAndVJP(unittest.TestCase):
  def test_jvp_vjp_equivalence(self):
    x = jax.random.uniform(jax.random.key(0), (16, 48), dtype=jnp.float32, minval=0, maxval=10)
    w1 = jax.random.normal(jax.random.key(1), (48, 32))
    w2 = jax.random.normal(jax.random.key(1), (32, 64))

    # Evaluates dL/dw1 via jax.grad
    expected_dw1 = jax.grad(simple_model, argnums=1)(x, w1, w2)


    # Evaluates dL/dw1 via jax.jvp (forward mode)
    got_dw1_from_jvp = jnp.zeros_like(expected_dw1)

    # Loop over each standard basis vectors of w1's space
    for i in range(w1.size):
      unflattened_index = jnp.unravel_index(i, shape=w1.shape)

      w1_tangent = jnp.zeros_like(w1).at[unflattened_index].set(1)

      f = lambda w: simple_model(x, w, w2)

      primal, dL_dw1 = jax.jvp(f, (w1,), (w1_tangent,))
      del primal  # Not important in this test
      got_dw1_from_jvp =  got_dw1_from_jvp.at[unflattened_index].set(dL_dw1)

    np.testing.assert_allclose(got_dw1_from_jvp, expected_dw1, rtol=1e-4)

if __name__ == "__main__":
  unittest.main()
