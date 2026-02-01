import unittest

import jax
import numpy as np
from jax import numpy as jnp
import functools

P = jax.P

# Force set 12 devices
jax.config.update("jax_num_cpu_devices", 12)


class TestMoe(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.assertEqual(jax.device_count(), 12)
        self.mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(3, 4),
                                      axis_names=("x", "y"))

        self.B = 36  # Batch size
        self.T = 128  # Number of tokens per example
        self.H = 256  # Hidden size
        self.E = 48  # The numbder of experts
        self.K = 3  # Top-k experts to route to

        self.input_sharding = jax.NamedSharding(self.mesh, spec=P(("x", "y")))
        self.replicated_sharding = jax.NamedSharding(self.mesh, spec=P())
        self.expert_sharding = jax.NamedSharding(self.mesh, spec=P(("x", "y")))

        # (B_xy, T, H)
        self.inputs: jax.Array = jax.device_put(
            jax.random.normal(jax.random.key(0),
                              shape=(self.B, self.T, self.H)),
            self.input_sharding)

        # (H, E)
        self.router_param = jax.device_put(
            jax.random.normal(jax.random.key(1), shape=(self.H, self.E)),
            self.replicated_sharding)

        # (E_xy, H, H): Each expert is a MLP that maps vector from H to H
        self.experts = jax.device_put(jax.random.normal(jax.random.key(2),
                                                        shape=(self.E, self.H,
                                                               self.H)),
                                      device=self.expert_sharding)

        # Only for debugging
        self._identity_experts = jax.device_put(
            self._create_identity_experts(), device=self.expert_sharding)

    def _create_identity_experts(self) -> jax.Array:
        """Creates identity expert parameters using jax.vmap"""
        experts = jax.vmap(lambda _: jnp.identity(n=self.H))(jnp.arange(self.E))
        self.assertEqual(experts.shape, (self.E, self.H, self.H))
        return experts

    def test_naive_moe_with_identity_experts(self):
        """Ensures if experts are all initialized as I, x == moe(x) holds"""

        result = jax.jit(
            functools.partial(_naive_moe, topk=3),
            in_shardings=(self.input_sharding, self.replicated_sharding,
                          self.expert_sharding))(self.inputs,
                                                 self.router_param,
                                                 self._identity_experts)
        self.assertEqual(result.shape, self.inputs.shape)
        np.testing.assert_allclose(result, self.inputs, rtol=1e-6)


def _naive_moe(inputs: jax.Array, router_param: jax.Array, experts: jax.Array,
               topk: int) -> jax.Array:
    B, T, H = inputs.shape
    H, E = router_param.shape
    E, H, H = experts.shape

    # (B, T, E)
    routing_weights = jnp.einsum("BTH,HE->BTE", inputs, router_param)
    assert routing_weights.shape == (B, T, E)

    # (B, T, E) -> Expert indices (B, T, K) and top k weights (B, T, K)
    expert_weights, expert_indices = jax.lax.top_k(routing_weights,
                                                   k=topk,
                                                   axis=-1)
    expert_weights = jax.nn.softmax(expert_weights, axis=-1)
    assert expert_indices.shape == (B, T, topk)

    # Use expert indices to slice/copy expert parameters
    # NOTE: In practice, we don't want to make lots of copies of moe weights
    #       because the amount of weights can be huge.
    # (B, T, K) takes (E, H, H) -> (B, T, K, H, H)
    expert_param_each_token = jnp.take(experts, indices=expert_indices, axis=0)

    # (B, T, K, H, H) @ (B, T, H) = (B, T, K, H)
    #           ^              ^ <--- contraction
    #  ^  ^              ^  ^    <--- batch dimension
    #        ^                   <--- duplicate dimension
    tokens_after_experts = jnp.einsum("BTKHH,BTH->BTKH",
                                      expert_param_each_token, inputs)

    # Weighted by expert weights
    # (B, T, K) and (B, T, K, H) -> (B, T, K, H)
    result = jnp.expand_dims(expert_weights, axis=-1) * tokens_after_experts
    assert result.shape == (B, T, topk, H)
    # (B, T, K, H) -> (B, T, H)
    return jnp.sum(result, axis=2)


if __name__ == "__main__":
    unittest.main()
