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
        self.mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(3, 4),
            axis_names=("x", "y"),
            axis_types=(jax.sharding.AxisType.Explicit,
                        jax.sharding.AxisType.Explicit))

        self.device_count = jax.device_count()
        self.x, self.y = 3, 4
        self.num_data_parallelism = self.x * self.y
        self.num_expert_parallelism = self.y  # Expert params is sharded `y` ways

        self.B = 36  # Batch size
        self.T = 128  # Number of tokens per example
        self.H = 256  # Hidden size
        self.E = 48  # The number of experts
        self.K = 3  # Top-k experts to route to
        # Expert's capacity:
        # If tokens are distributed perfectly equal, each device will get:
        # (BTK // num_expert_parallelim) tokens. Adding a 2 multiplier to leave
        # some headroom on each device.
        self.expert_capacity_per_device = int(
            (self.B * self.T * self.K) // self.num_expert_parallelism * 2)

        self.input_sharding = jax.NamedSharding(self.mesh, spec=P(("x", "y")))
        self.replicated_sharding = jax.NamedSharding(self.mesh, spec=P())
        self.expert_sharding = jax.NamedSharding(self.mesh, spec=P("y"))

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
        experts = jax.vmap(lambda _: jnp.identity(n=self.H))(jnp.arange(
            self.E))
        self.assertEqual(experts.shape, (self.E, self.H, self.H))
        return experts

    def test_naive_moe_with_identity_experts(self):
        """Ensures if experts are all initialized as I, x == moe(x) holds"""

        result = jax.jit(
            functools.partial(_naive_moe, topk=3, mesh=self.mesh),
            in_shardings=(self.input_sharding, self.replicated_sharding,
                          self.expert_sharding))(self.inputs,
                                                 self.router_param,
                                                 self._identity_experts)
        self.assertEqual(result.shape, self.inputs.shape)
        np.testing.assert_allclose(result, self.inputs, rtol=1e-6)

    def test_ragged_all_to_all_moe(self):
        self.skipTest("Skips because ragged_all_to_all is not supported on CPU.")
        result = jax.jit(
            functools.partial(
                _ragged_all_to_all_moe,
                topk=3,
                num_data_parallelism=self.num_data_parallelism,
                num_expert_parallelism=self.num_expert_parallelism,
                mesh=self.mesh),
            in_shardings=(self.input_sharding, self.replicated_sharding,
                          self.expert_sharding))(self.inputs,
                                                 self.router_param,
                                                 self.experts)

    def test_compute_ragged_all_to_all_info(self):
        num_tokens_per_device = 13

        # Creates fake tokens to test distributing these
        # tokens to devices and bringing them back
        fake_tokens = jnp.arange(0, num_tokens_per_device *
                                 self.device_count).reshape(
                                     -1,
                                     # 1 represents the hidden dimension
                                     1)

        # Ideally, each device should get in average
        # `num_tokens_per_device` number of tokens when distributed evenly
        # but in practice, we want to leave some headroom.
        # So I multiply the number by `4` here.
        expert_capacity_per_device = num_tokens_per_device * 4
        expert_capacity_per_expert = (expert_capacity_per_device *
                                      self.y) // self.E

        # This simulates the expert routing results (to which expert indices)
        fake_expert_indices = jax.random.randint(jax.random.key(1234),
                                                 shape=(num_tokens_per_device *
                                                        self.device_count, ),
                                                 minval=0,
                                                 maxval=self.E)
        # Each expert does: f_expert(x) = x + expert_index (for testing)
        fake_experts = jnp.arange(0, self.E).reshape(self.E, 1)

        fake_expert_indices = jax.device_put(fake_expert_indices,
                                             self.input_sharding)
        fake_tokens = jax.device_put(fake_tokens, self.input_sharding)
        fake_experts = jax.device_put(fake_experts, self.expert_sharding)

        pad_value = -1000

        expected_result = fake_tokens + fake_expert_indices.reshape(-1, 1)

        @jax.shard_map(in_specs=(self.input_sharding.spec,
                                 self.input_sharding.spec,
                                 self.expert_sharding.spec),
                       out_specs=(self.input_sharding.spec),
                       mesh=self.mesh,
                       check_vma=False)
        def _func(local_tokens, local_expert_indices, local_expert_param):
            sorter = jnp.argsort(local_expert_indices)
            unsorter = jnp.argsort(sorter)

            # Sorted so that tokens belonging to the each experts are next to each other
            sorted_local_tokens = local_tokens[sorter]
            sorted_expert_indices = local_expert_indices[sorter]
            num_tokens, hidden_dim = local_tokens.shape

            num_devices = jax.lax.axis_size("y")

            all_expert_indices = _get_expert_indices(self.E)
            assert all_expert_indices.shape == (self.E, )
            # O(E log num_tokens) to find all the offsets via binary searches (not sure how is this performance?)
            expert_start_offsets = jnp.searchsorted(sorted_expert_indices,
                                                    all_expert_indices)
            assert expert_start_offsets.shape == (self.E, )
            num_tokens_send_to_each_expert = jnp.concat(
                [expert_start_offsets[1:],
                 jnp.array([num_tokens])]) - expert_start_offsets
            assert num_tokens_send_to_each_expert.shape == (self.E, )

            # Useful for debugging
            if False:
                jax.debug.print(
                    "sorted tokens {}, sorted indices: {}, value: {}",
                    sorted_local_tokens.reshape(-1), sorted_expert_indices,
                    num_tokens_send_to_each_expert)

            # Pads the first dimension for `dynamic_slice_in_dim` later.
            # This is needed because the last device might not get any tokens so
            # slicing via [offset:offset+expert_capacity_per_expert] will not go out of bound
            padded_sorted_local_tokens = jnp.pad(
                sorted_local_tokens, ((0, expert_capacity_per_expert), (0, 0)))

            def _slice_tokens(expert_idx):
                offset = expert_start_offsets[expert_idx]
                tokens = jax.lax.dynamic_slice_in_dim(
                    # Use the padded version due to static `slice_size`
                    operand=padded_sorted_local_tokens,
                    start_index=offset,
                    # Due to XLA's static shape, constraint,
                    # we need to grab `expert_capacity_per_device` here
                    slice_size=expert_capacity_per_expert,
                    axis=0)
                # Generates a mask for putting garbage values
                # i.e. tokens[offset+size:] = -1000 (make it large for easier debugging)
                size = num_tokens_send_to_each_expert[expert_idx]
                mask: jax.Array = jnp.where(
                    jnp.arange(0, expert_capacity_per_expert) < size, 1, 0)
                tokens = (mask[..., None] *
                          tokens) + (1 - mask[..., None]) * jnp.broadcast_to(
                              pad_value, tokens.shape)
                return tokens

            tokens_to_send: jax.Array = jax.vmap(_slice_tokens)(
                all_expert_indices)
            assert tokens_to_send.shape == (self.E, expert_capacity_per_expert,
                                            hidden_dim)

            tokens_to_me = jax.lax.all_to_all(
                tokens_to_send,
                axis_name="y",  # Expert parallelism axis
                split_axis=0,
                concat_axis=1,
                tiled=True,
            )
            num_local_experts = local_expert_param.shape[0]
            assert tokens_to_me.shape == (
                num_local_experts,
                # This dimension is concatenated
                self.y * expert_capacity_per_expert,
                local_tokens.shape[-1])

            # Applies expert's parameter to each token
            mask = tokens_to_me != pad_value
            applied_tokens = tokens_to_me + local_expert_param.reshape(
                num_local_experts, 1, 1)
            applied_tokens = jnp.where(mask, applied_tokens, pad_value)

            # Bring those tokens back their original devices
            applied_tokens = jax.lax.all_to_all(applied_tokens,
                                                axis_name="y",
                                                split_axis=1,
                                                concat_axis=0,
                                                tiled=True)
            assert applied_tokens.shape == (self.E, expert_capacity_per_expert,
                                            hidden_dim)

            # Brings back the tokens into (num_tokens, hidden_dim)
            # using the sorting trick to push all the padding values to the back
            applied_tokens = applied_tokens.reshape(-1, hidden_dim)
            discard_padding_sorter = jnp.argsort(
                jnp.where(applied_tokens[..., 0] != pad_value, 0, 1))

            # We know that [:num_tokens] is all the tokens we need (after the sorting)
            # and all the padding values are on the back
            result = applied_tokens[discard_padding_sorter][:num_tokens]
            assert result.shape == (num_tokens, hidden_dim)
            return result[unsorter]

        got_result = _func(fake_tokens, fake_expert_indices, fake_experts)
        np.testing.assert_array_equal(got_result, expected_result)


def _naive_moe(inputs: jax.Array, router_param: jax.Array, experts: jax.Array,
               topk: int, mesh: jax.sharding.Mesh) -> jax.Array:
    B, T, H1 = inputs.shape
    H2, E1 = router_param.shape
    E2, H3, H4 = experts.shape

    assert H1 == H2 == H3 == H4
    assert E1 == E2

    H = H1
    E = E1

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
    all_gathered_experts = jax.reshard(experts,
                                       jax.sharding.NamedSharding(mesh, P()))
    expert_param_each_token = all_gathered_experts.at[expert_indices].get(
        out_sharding=jax.sharding.NamedSharding(mesh, P(("x", "y"))))

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


def _ragged_all_to_all_moe(
        inputs: jax.Array,
        router_param: jax.Array,
        experts: jax.Array,
        *,
        # Other Jax-untrace-able variables
        topk: int,
        num_data_parallelism: int,
        num_expert_parallelism: int,
        mesh: jax.sharding.Mesh):
    """Implements tokens distribution and collection by AllToAll collective."""
    B, T, H1 = inputs.shape
    H2, E1 = router_param.shape
    E2, H3, H4 = experts.shape

    assert H1 == H2 == H3 == H4
    assert E1 == E2

    H = H1
    E = E1

    # (B, T, E)
    routing_weights = jnp.einsum("BTH,HE->BTE", inputs, router_param)
    assert routing_weights.shape == (B, T, E)

    # (B, T, E) -> Expert indices (B, T, K) and top k weights (B, T, K)
    expert_weights, expert_indices = jax.lax.top_k(routing_weights,
                                                   k=topk,
                                                   axis=-1)
    expert_weights = jax.nn.softmax(expert_weights, axis=-1)
    assert expert_indices.shape == (B, T, topk)

    # Broadcasts (B, T, H) to (B, T, K, H) before distributing token
    # vectors to each device
    duplicate_inputs = jnp.broadcast_to(jnp.expand_dims(inputs, axis=2),
                                        shape=(B, T, topk, H))
    assert duplicate_inputs.shape == (B, T, topk, H)

    @jax.shard_map(
        in_specs=(
            P(("x", "y")),
            P(("x", "y")),
            P("y"),  # Experts are sharde only along y axis
        ),
        out_specs=P(("x", "y")),
        mesh=mesh,
        check_vma=False,
    )
    def _shard_map_func(local_inputs: jax.Array,
                        local_expert_indices: jax.Array,
                        local_experts: jax.Array):
        assert local_inputs.shape == (B // num_data_parallelism, T, topk, H)
        assert local_expert_indices.shape == (B // num_data_parallelism, T,
                                              topk)
        assert local_experts.shape == (E // num_expert_parallelism, H, H)

        # Flattens inputs and expert indices
        flattened_inputs = jnp.reshape(local_inputs, shape=(-1, H))
        flattened_expert_indices = jnp.reshape(local_expert_indices,
                                               shape=(-1, ))

        # Argsorts expert indices
        sorter = jnp.argsort(flattened_expert_indices)

        sorted_expert_indices = flattened_expert_indices[sorter]
        sorted_flattened_inputs = flattened_inputs[flattened_expert_indices]
        assert sorted_expert_indices.shape == (B * T * topk //
                                               num_data_parallelism, )
        assert sorted_flattened_inputs.shape == (B * T * topk //
                                                 num_data_parallelism, H)

        # Computes expert boundaries for `ragged_all_to_all`'s `input_offsets`
        device_expert_start_indices = _get_device_expert_start_indices(
            E, num_expert_parallelism)
        assert device_expert_start_indices.shape == (jax.lax.axis_size("y"), )

        input_offsets = jnp.searchsorted(sorted_expert_indices,
                                         device_expert_start_indices)

        return

    # (B, T, K, H) -> (B, T, K, H) (transformed by experts)
    tokens_after_experts = _shard_map_func(duplicate_inputs, expert_indices,
                                           experts)


def _get_device_expert_start_indices(num_experts: int,
                                     num_expert_parallelism: int) -> jax.Array:
    """Returns an 1D array with each device's starting expert index."""
    assert num_experts % num_expert_parallelism == 0
    num_experts_per_device = num_experts // num_expert_parallelism

    # Let `num_experts_per_device=x`
    # [0, E_x, E_2x, ..., E_num_experts-x]
    return jnp.arange(start=0, stop=num_experts, step=num_experts_per_device)


def _get_expert_indices(num_experts: int):
    return jnp.arange(0, num_experts)


# TODO: Revisit when `jax.lax.ragged_all_to_all` is available on CPUs.
# https://github.com/jax-ml/jax/issues/34755
def _compute_ragged_all_to_all_info(num_tokens: int, input_offsets: jax.Array):
    """Computes info needed for `ragged_all_to_all` in `jax.shard_map`."""

    send_sizes = jnp.concat([input_offsets[1:],
                             jnp.array([num_tokens])]) - input_offsets
    recv_sizes = jax.lax.all_to_all(
        send_sizes,
        # Ideally we can paramaterize this
        axis_name="y",
        split_axis=0,
        concat_axis=0,
        tiled=False,  # not need to tile because len(send_sizes) == |y|
    )
    send_to_me_offsets = jnp.concat(
        [jnp.array([0]), jnp.cumsum(recv_sizes, axis=0)[:-1]])
    me_to_dst_offsets = jax.lax.all_to_all(
        send_to_me_offsets,
        axis_name="y",
        split_axis=0,
        concat_axis=0,
        tiled=False,  # not need to tile because len(send_sizes) == |y|
    )
    output_offsets = me_to_dst_offsets
    return send_sizes, recv_sizes, output_offsets


if __name__ == "__main__":
    unittest.main()
