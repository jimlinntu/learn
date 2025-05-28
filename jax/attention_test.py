import unittest
import attention
import jax
import numpy as np


class TestAttention(unittest.TestCase):
  def test_correctness(self):
    key = jax.random.PRNGKey(42)
    q_key, k_key, v_key = jax.random.split(key, num=3)
    q = jax.random.normal(q_key, (2048, 128))
    k = jax.random.normal(k_key, (2048, 128))
    v = jax.random.normal(v_key, (2048, 128))
    expected = attention.one_head_attention(q, k, v)
    got = attention.one_head_flash_attention_in_for_loop(q, k, v)

    np.testing.assert_allclose(got, expected, atol=1e-4)

  def test_correctness_against_pallas_cpu(self):
    key = jax.random.PRNGKey(42)
    q_key, k_key, v_key = jax.random.split(key, num=3)
    q = jax.random.normal(q_key, (2048, 128))
    k = jax.random.normal(k_key, (2048, 128))
    v = jax.random.normal(v_key, (2048, 128))
    expected = attention.one_head_attention(q, k, v)
    got = attention.one_head_flash_attention_in_pallas(q, k, v)

    np.testing.assert_allclose(got, expected, atol=1e-4)


if __name__ == "__main__":
  unittest.main()