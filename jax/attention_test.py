import unittest
import attention
import jax


class TestAttention(unittest.TestCase):
  def test_correctness(self):
    key = jax.random.PRNGKey(42)
    q_key, k_key, v_key = jax.random.split(key, num=3)
    q = jax.random.normal(q_key, (2048, 128))
    k = jax.random.normal(k_key, (2048, 128))
    v = jax.random.normal(v_key, (2048, 128))
    result = attention.one_head_attention(q, k, v)
    # TODO(jimlinntu): Compares against flash attention implementation

if __name__ == "__main__":
  unittest.main()