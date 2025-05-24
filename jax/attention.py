"""Implements one-head attention

Following the notation from Flash attention paper.
https://arxiv.org/pdf/2205.14135
"""
import jax
from jax import numpy as jnp

def one_head_attention(q: jax.Array, k: jax.Array, v: jax.Array):
  """Single-head attention, i.e. softmax(Q (K^T)) V ."""
  if not (q.shape == k.shape == v.shape):
    raise ValueError("q, k, v should be of the same dimension")
  if not (q.ndim == k.ndim == v.ndim == 2):
    raise ValueError("q, k, v should be 2 dimensional arrays")

  s = jnp.matmul(q, k.T)
  p = jax.nn.softmax(s, axis=1)
  return jnp.matmul(p, v)

def one_head_flash_attention(q: jax.Array, k: jax.Array, v: jax.Array):
  # TODO(jimlinntu): Practice implementing flash attention in jax
  pass