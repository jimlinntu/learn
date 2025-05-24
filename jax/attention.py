"""Implements one-head attention

Following the notation from Flash attention paper.
https://arxiv.org/pdf/2205.14135
"""
import jax
from jax import numpy as jnp

def _validate_q_k_v(q: jax.Array, k: jax.Array, v: jax.Array):
  if not (q.shape == k.shape == v.shape):
    raise ValueError("q, k, v should be of the same dimension")
  if not (q.ndim == k.ndim == v.ndim == 2):
    raise ValueError("q, k, v should be 2 dimensional arrays")


def one_head_attention(q: jax.Array, k: jax.Array, v: jax.Array):
  """Single-head attention, i.e. softmax(Q (K^T)) V ."""
  _validate_q_k_v(q, k, v)

  # This step makes the memory quadratic (O(seq_len^2)) and
  # we want to avoid this.
  s = jnp.matmul(q, k.T)
  p = jax.nn.softmax(s, axis=1)
  return jnp.matmul(p, v)


def one_head_flash_attention_in_for_loop(q: jax.Array, k: jax.Array, v: jax.Array):
  """Flash attention implemented in for loops.

  We should be able to optimize this further using jax vmap.
  """

  _validate_q_k_v(q, k, v)

  n, d = q.shape

  TILE_SIZE = 256

  # m represents the accumulated max
  m = jnp.full((n, ), float("-inf"), dtype=q.dtype)
  # s represents the accumulated exp sum
  s = jnp.zeros((n, ), dtype=q.dtype)

  out = jnp.zeros((n, d), dtype=q.dtype)

  # NOTE: The outer loop is for k because only j dimension is parallelizable
  # without data dependencies
  for j in range(0, n, TILE_SIZE):
    # Loads a k block to SRAM
    k_block = k[j:j+TILE_SIZE, :]

    # The loop below can be run in parallel because
    # each loop iteration write to different output memory block in HBM
    for i in range(0, n, TILE_SIZE):
      # Loads a q block to SRAM
      q_block = q[i:i+TILE_SIZE, :]

      # q_j x k_i^T = (q k^T)_{ji}
      q_k_t_ij_block = jnp.matmul(q_block, k_block.T)

      # Loads v block to SRAM
      v_j_block = v[j:j+TILE_SIZE, :]

      # Loads m block to SRAM
      m_i_block = m[i:i+TILE_SIZE]
      current_m_block = jnp.max(q_k_t_ij_block, axis=1)
      new_m_i_block = jnp.maximum(m_i_block, current_m_block)

      # Loads s block to SRAM
      s_i_block = s[i:i+TILE_SIZE]

      # Minus a max value for each row
      # (TILE_SIZE, TILE_SIZE) - (TILE_SIZE, 1)
      softmax_nominator = jnp.exp(q_k_t_ij_block - current_m_block[:, None])
      # Applies attention value on each token embedding vector
      # (TILE_SIZE, TILE_SIZE) * (TILE_SIZE, d)
      current_out_block = jnp.matmul(softmax_nominator, v_j_block)

      # (TILE_SIZE, TILE_SIZE) -> (TILE_SIZE, )
      current_s_block = jnp.sum(softmax_nominator, axis=1)
  
      scaling_factor_for_prev = jnp.exp(m_i_block - new_m_i_block)
      scaling_factor_for_curr = jnp.exp(current_m_block - new_m_i_block)

      new_s_i_block = scaling_factor_for_prev * s_i_block + scaling_factor_for_curr * current_s_block


      # Loads out block to SRAM
      out_i_block = out[i:i+TILE_SIZE, :]
      new_out_i_block = scaling_factor_for_prev[:, None] * out_i_block + scaling_factor_for_curr[:, None] * current_out_block

      # Writes m block back to HBM
      m = m.at[i:i+TILE_SIZE].set(new_m_i_block)
      # Writes s block back to HBM
      s = s.at[i:i+TILE_SIZE].set(new_s_i_block)
      # Writes o block back to HBM
      out = out.at[i:i+TILE_SIZE, :].set(new_out_i_block)

  # Lastly, divides out by s to obtain the true softmax result
  return out / s[:, None]
